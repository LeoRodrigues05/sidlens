#!/usr/bin/env python
"""Logit lens and user-disjoint linear probes on one captured AR activation store.

Protocol: `protocol.md` beside this file (declared before any decoding was
computed). Observational: decodable is not the same as used.

Traps, each with the guard that closes it
-----------------------------------------
* **A store that is not what it claims.** The store's checkpoint sha must
  equal the frozen manifest, every shard sha256 is re-verified, and the
  template/split must be eval/test.
* **Probe leakage across users.** Folds are by user (sha256 of the user id),
  and the L2 strength is chosen on an inner split of TRAINING users only, so no
  held-out user influences a probe that scores them.
* **Phantom classes.** A class absent from a training fold has no probe
  output at all (the output layer spans only the training classes), so it can
  never be predicted, as declared.
* **Logit-lens mismatch.** The `model.norm` site must reproduce the capture's
  own digit ranks (>= 99% agreement), proving the unembedding rows and the
  final-norm handling are right before any intermediate layer is read.

    python run.py --capture $SIDLENS_WORK/derived/ar_capture/281053 --out <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.hooks.store import StoreReader                              # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402

SEED = 20260927
LAMBDAS = (1e-4, 1e-3, 1e-2)
SHUFFLE_LAYERS = (0, 7, 14, 21, 27)
PROBES = {  # name: (site key, label key)
    "P1_target_d0_at_readout0": ("readout0", "target_d0"),
    "P2_recent_d0_at_readout0": ("readout0", "recent_d0"),
    "P3_recent_d0_at_recent_last": ("recent_last", "recent_d0"),
    "P4_target_d1_at_readout1": ("readout1", "target_d1"),
}


def h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def fit_predict(Xtr, ytr, Xte, lam: float, iters: int):
    """Multinomial logistic regression (standardized, L2) by full-batch L-BFGS."""
    classes, yi = torch.unique(ytr, return_inverse=True)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(A.shape[1], len(classes), device=A.device, requires_grad=True)
    b = torch.zeros(len(classes), device=A.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, yi) + lam * (W ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return classes[(B @ W + b).argmax(1)]


def probe(X, y, users, fold, inner, iters, shuffle_rng=None):
    """Out-of-fold predictions and the chosen lambda per fold."""
    pred = torch.empty_like(y)
    lams = {}
    for f in range(5):
        tr, te = fold != f, fold == f
        ytr = y[tr]
        if shuffle_rng is not None:
            ytr = ytr[torch.as_tensor(shuffle_rng.permutation(int(tr.sum())), device=y.device)]
        itr, iva = tr & (inner != 0), tr & (inner == 0)
        yi_tr = ytr[(inner[tr] != 0)]
        best = None
        for lam in LAMBDAS:
            p = fit_predict(X[itr], yi_tr, X[iva], lam, iters)
            acc = float((p == (y[iva] if shuffle_rng is None else ytr[(inner[tr] == 0)])).float().mean())
            if best is None or acc >= best[0]:            # ties -> the stronger penalty
                best = (acc, lam)
        lams[f] = best[1]
        pred[te] = fit_predict(X[tr], ytr, X[te], best[1], iters)
    return pred, lams


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", type=Path, required=True, help="derived/ar_capture/<job-id>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--layers", default="all", help="'all' or comma list (pilot)")
    ap.add_argument("--cell", type=int, default=None, help="accepted from the array wrapper")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    # ---- the store, verified -------------------------------------------------
    store = StoreReader(args.capture / "capture" / "store", verify=True)
    meta = store.meta
    ckpt = meta["model"].split("/", 1)[1]
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    if meta["checkpoint_sha256"] != want or meta["template"] != "eval" or meta["split"] != "test":
        raise RuntimeError(f"store meta does not match the protocol: {meta}")
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    examples = P.load_examples(vocab.variant, "next-item", "test")
    ex_by_id = {e.example_id: e for e in examples}
    rows = store.rows
    log(f"store ok: {ckpt} rows={len(rows)} sites={len(store.sites)}")

    # ---- positions and labels per example ---------------------------------------
    def pick(mask):
        g = rows[mask]
        if g.example_id.duplicated().any():
            raise ValueError("a position is not unique per example")
        return g.set_index("example_id")

    order = [e.example_id for e in examples]
    hist_len = pd.Series({e.example_id: len(e.history_sids) for e in examples})
    rows = rows.assign(hist_len=rows.example_id.map(hist_len))
    hdr = rows[rows.role == "response_header"].sort_values("pos").groupby("example_id").tail(1).set_index("example_id")
    tgt = {d: pick((rows.role == "target_sid") & (rows.item == 0) & (rows.digit == d)) for d in range(n)}
    rec_last = pick((rows.role == "hist_sid") & (rows.item == rows.hist_len - 1) & (rows.digit == n - 1))
    rec_d0 = pick((rows.role == "hist_sid") & (rows.item == rows.hist_len - 1) & (rows.digit == 0))
    site_rows = {"readout0": hdr.loc[order, "row"].to_numpy(),
                 **{f"readout{d}": tgt[d - 1].loc[order, "row"].to_numpy() for d in range(1, n)},
                 "recent_last": rec_last.loc[order, "row"].to_numpy()}
    labels = {"target_d0": tgt[0].loc[order, "code"].to_numpy(),
              "target_d1": tgt[1].loc[order, "code"].to_numpy(),
              "recent_d0": rec_d0.loc[order, "code"].to_numpy(),
              **{f"target_d{d}": tgt[d].loc[order, "code"].to_numpy() for d in range(n)}}
    for e in examples[:50]:                           # the store's codes are the CSV's
        if P.parse_sid(e.target_sids[0], n)[0] != labels["target_d0"][order.index(e.example_id)]:
            raise ValueError(f"{e.example_id}: stored target code differs from the CSV")
    users = np.array([ex_by_id[x].user_id for x in order])
    fold = torch.as_tensor([h(f"{SEED}|{u}") % 5 for u in users], device=args.device)
    inner = torch.as_tensor([h(f"{SEED}|inner|{u}") % 5 for u in users], device=args.device)

    # ---- unembedding (tied) and final norm --------------------------------------
    from safetensors import safe_open
    cfg = json.loads((paths.FROZEN_CKPT / "ar" / ckpt / "config.json").read_text())
    with safe_open(str(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors"), "pt") as fh:
        emb = fh.get_tensor("model.embed_tokens.weight")
        norm_w = fh.get_tensor("model.norm.weight").float().to(args.device)
    E = {d: emb[torch.as_tensor(vocab.ids(d))].float().to(args.device) for d in range(n)}
    codes = {d: np.asarray(vocab.codes(d)) for d in range(n)}
    col = {d: {c: j for j, c in enumerate(codes[d])} for d in range(n)}
    del emb
    eps = cfg["rms_norm_eps"]

    sites = [s for s in store.sites if s.startswith("model.layers.")]
    sites = sorted(sites, key=lambda s: int(s.split(".")[-1])) + ["model.norm"]
    if args.layers != "all":
        keep = {int(x) for x in args.layers.split(",")}
        sites = [s for s in sites if s == "model.norm" or int(s.split(".")[-1]) in keep]
    inputs = {"capture": str(args.capture.relative_to(paths.WORK)), "store_meta": meta, "checkpoint": ckpt,
              "variant": vocab.variant.name, "n_digits": n, "n_examples": len(order),
              "n_users": int(len(set(users))), "sites": sites, "lambdas": LAMBDAS, "iters": args.iters,
              "seed": SEED, "probes": PROBES, "shuffle_layers": SHUFFLE_LAYERS,
              "store_manifest_sha256": hashing.sha256_file(args.capture / "capture" / "store" / "manifest.json")}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))

    lens_recs, probe_recs, lam_recs = [], [], []
    need = np.unique(np.concatenate(list(site_rows.values())))
    pos_in = {k: np.searchsorted(need, v) for k, v in site_rows.items()}
    for s in sites:
        t0 = time.time()
        X = store.load(s, rows=need).to(args.device).float()
        layer = 28 if s == "model.norm" else int(s.split(".")[-1])
        # logit lens
        for d in range(n):
            hh = X[torch.as_tensor(pos_in[f"readout{d}"], device=args.device)]
            if s != "model.norm":
                hh = hh * torch.rsqrt(hh.pow(2).mean(-1, keepdim=True) + eps) * norm_w
            lg = hh @ E[d].T
            logp = torch.log_softmax(lg, -1)
            j = torch.as_tensor([col[d][c] for c in labels[f"target_d{d}"]], device=args.device)
            ar_ = torch.arange(len(j), device=args.device)
            rank = (lg > lg[ar_, j][:, None]).sum(-1).cpu().numpy()
            # The model's own logits are bf16, where near-tied codes share a value;
            # the acceptance check ranks at that precision (protocol amendment).
            lb = lg.bfloat16().float()
            rank_bf16 = (lb > lb[ar_, j][:, None]).sum(-1).cpu().numpy()
            rec = {"layer": layer, "site": s, "digit": d, "example_id": order, "user_id": users,
                   "golden_rank": rank, "golden_rank_bf16": rank_bf16,
                   "golden_logp": logp[ar_, j].cpu().numpy(),
                   "top1_code": codes[d][lg.argmax(-1).cpu().numpy()]}
            if d == 0:
                jr = torch.as_tensor([col[0][c] for c in labels["recent_d0"]], device=args.device)
                rec["recent_rank"] = (lg > lg[ar_, jr][:, None]).sum(-1).cpu().numpy()
            lens_recs.append(pd.DataFrame(rec))
        # probes
        for name, (site_key, lab) in PROBES.items():
            Xs = X[torch.as_tensor(pos_in[site_key], device=args.device)]
            y = torch.as_tensor(labels[lab], device=args.device)
            runs = [("probe", None)]
            if layer in SHUFFLE_LAYERS:
                runs.append(("shuffled", np.random.default_rng(h(f"{SEED}|{name}|{layer}") % 2**32)))
            for kind, rng in runs:
                pred, lams = probe(Xs, y, users, fold, inner, args.iters, rng)
                probe_recs.append(pd.DataFrame({"probe": name, "kind": kind, "layer": layer,
                                                "example_id": order, "user_id": users,
                                                "fold": fold.cpu().numpy(), "label": y.cpu().numpy(),
                                                "pred": pred.cpu().numpy()}))
                lam_recs += [{"probe": name, "kind": kind, "layer": layer, "fold": f, "lambda": l}
                             for f, l in lams.items()]
        log(f"{s}: {time.time() - t0:.0f}s")
        del X

    lens = pd.concat(lens_recs, ignore_index=True)
    lens.to_parquet(args.out / "logit_lens.parquet", index=False)
    pr = pd.concat(probe_recs, ignore_index=True)
    # baselines needing no training: majority of the training fold, and the copy rule for P1
    base = []
    for name, (_, lab) in PROBES.items():
        y = labels[lab]
        f = fold.cpu().numpy()
        maj = np.empty_like(y)
        for k in range(5):
            vals, cnt = np.unique(y[f != k], return_counts=True)
            maj[f == k] = vals[cnt.argmax()]
        base.append(pd.DataFrame({"probe": name, "kind": "majority", "layer": -1, "example_id": order,
                                  "user_id": users, "fold": f, "label": y, "pred": maj}))
    base.append(pd.DataFrame({"probe": "P1_target_d0_at_readout0", "kind": "copy_recent_d0", "layer": -1,
                              "example_id": order, "user_id": users, "fold": fold.cpu().numpy(),
                              "label": labels["target_d0"], "pred": labels["recent_d0"]}))
    pr = pd.concat([pr, *base], ignore_index=True)
    pr["correct"] = (pr.label == pr.pred).astype(float)
    pr.to_parquet(args.out / "probes.parquet", index=False)
    pd.DataFrame(lam_recs).to_csv(args.out / "probe_lambdas.csv", index=False)

    # ---- validation: the model.norm lens reproduces the capture's own ranks -----
    ds = pd.read_csv(args.capture / "capture" / "digit_scores.csv")
    fin = lens[lens.site == "model.norm"][["example_id", "digit", "golden_rank", "golden_rank_bf16",
                                           "golden_logp"]]
    m = ds.merge(fin, on=["example_id", "digit"])
    agree = float((m["rank"] == m.golden_rank_bf16).mean())
    validation = {"store_verified": True, "norm_site_rank_agreement_with_capture_bf16_logits": agree,
                  "norm_site_rank_agreement_with_capture_fp32_logits": float((m["rank"] == m.golden_rank).mean()),
                  "norm_site_top1_agreement": float(((m["rank"] == 0) == (m.golden_rank_bf16 == 0)).mean()),
                  "norm_site_max_abs_diff_logp": float((m.logp_codes - m.golden_logp).abs().max()),
                  "n_compared": len(m), "seconds": round(time.time() - t_start, 1)}
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1))
    if agree < 0.99:
        raise RuntimeError(f"logit lens at model.norm agrees with the capture on only {agree:.4f} of ranks")
    log(f"done: {json.dumps(validation)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
