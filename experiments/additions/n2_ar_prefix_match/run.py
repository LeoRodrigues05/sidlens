#!/usr/bin/env python
"""N2 (a)-(e): where an AR grid model matches a history item's prefix (attention inputs, teacher-forced).

Protocol: `protocol.md` beside this file, declared before any measurement.
One invocation = one AR next-item grid run at its best state. Designs and the
suffix / none replacements are N1's (`../n1_prefix_conditioning/run.py`,
`design`). Every design's inputs have one length, so each chunk of designs is
one length bucket and the runner's row index is the design's (trap 3 of
`interventions.qkv`).

Forwards per chunk (rows = designs):
    full / suffix / none, observed      (a) attention to h's digit-d token
    suffix, k and v captured            source of (b)
    full with late k (or v) from suffix (b)
    full / suffix / none with the early knockout   (c)
    donor, late q captured; full with that q, observed   (d)
    full, o_proj inputs + final residual captured   (e)

    python run.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 256]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import grid                                                  # noqa: E402
from sidlens.data import ar_prompts as P                                  # noqa: E402
from sidlens.grid import registry                                         # noqa: E402
from sidlens.interventions import history as Hi, qkv                      # noqa: E402
from sidlens.models import ar                                             # noqa: E402

N1 = Path(__file__).resolve().parents[1] / "n1_prefix_conditioning" / "run.py"


def n1_design():
    spec = importlib.util.spec_from_file_location("_n1_run", N1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.design


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="an AR next-item grid weight state (grid:<key>@best)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=64, help="designs per forward")
    ap.add_argument("--device", default="cuda", help="cpu runs float32 (smoke tests; bit-exact controls hold on GPU only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t_start = time.time()
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)       # noqa: E731
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    from transformers import AutoTokenizer
    from sidlens.grid import inputs as GI
    from sidlens.interventions.runner import FixedShapeRunner, Job, length_buckets
    from sidlens.interventions.scoring import DigitScorer

    entry = registry.load().entry(args.ckpt)
    if entry["paradigm"] != "ar" or entry["predtype"] != "next-item":
        ap.error("N2 is declared for AR next-item runs")
    ckpt = args.ckpt
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    tok = AutoTokenizer.from_pretrained(str(ar.checkpoint_dir(ckpt)), local_files_only=True)
    gin = GI.ArSetup(ckpt, "test", tok, vocab, task="next-item")
    if not all(v["ok"] for v in gin.validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(gin.validation, default=str)[:1500]}")
    examples = sorted(gin.examples, key=lambda e: e.row)[:args.limit] if args.limit else gin.examples
    pool = Hi.ControlPool(gin.table, gin.item2id)
    digit_codes = [sorted(set(pool.codes[:, d].tolist())) for d in range(n)]
    design = n1_design()

    # ---- designs: every variant of one design has the clean row's length ----------
    D = []
    for ex in examples:
        hist = [P.parse_sid(s, n) for s in ex.history_sids]
        target = P.parse_sid(ex.target_sids[0], n)
        for r in design(ex.example_id, hist, target, pool.codes, pool.item_ids,
                        {*ex.history_item_ids, *ex.target_item_ids}, digit_codes):
            d, k, h = r["d"], r["k"], r["h"]
            enc = {"full": P.encode(ex, tok, vocab, template=gin.template, with_target=True)}
            for cond in ("suffix", "none"):
                codes, item_id, synth, nc = r["repl"][cond]
                enc[cond] = P.encode(Hi.replace_history_item(ex, k, Hi.Control(item_id, codes, nc)), tok, vocab,
                                     template=gin.template, with_target=True)
                Hi.check_replacement(enc["full"], enc[cond], k, next(j for j in range(n) if codes[j] != h[j]))
            j2 = next((j for j in range(len(hist) - 1, -1, -1)
                       if j != k and hist[j][:d] != target[:d] and hist[j][d] != h[d]), None)
            if j2 is not None:
                donor = Hi.replace_target_slot(ex, 0, Hi.Control(ex.history_item_ids[j2], hist[j2], 0))
                enc["donor"] = P.encode(donor, tok, vocab, template=gin.template, with_target=True)
            e = enc["full"]
            D.append({"example_id": ex.example_id, "user_id": ex.user_id, "d": d, "k": k, "j2": j2,
                      "h_code": h[d], "h2_code": hist[j2][d] if j2 is not None else -1, "gold_code": target[d],
                      "synthetic": bool(r["repl"]["suffix"][2] or r["repl"]["none"][2]),
                      "readout": e.predict_pos(0, d), "hpos": e.positions("hist_sid", item=k, digit=d)[0],
                      "hprev": [e.positions("hist_sid", item=k, digit=j)[0] for j in range(d)],
                      "h2pos": e.positions("hist_sid", item=j2, digit=d)[0] if j2 is not None else -1, "enc": enc})
    log(f"{len(examples)} rows, {len(D)} designs, {sum(x['j2'] is not None for x in D)} with a donor")

    model = ar.load_model(ckpt, device=args.device, dtype="bfloat16" if args.device == "cuda" else "float32")
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    late, early = tuple(range(nL // 2, nL)), tuple(range(nL // 2))
    lens = [len(x["enc"]["full"]) for x in D]
    buckets = length_buckets(lens)
    runner = FixedShapeRunner(model, DigitScorer(vocab, gin.table), tok.pad_token_id, buckets[-1], args.rows, 1,
                              buckets=buckets)
    col = [{c: j for j, c in enumerate(vocab.codes(d))} for d in range(n)]
    W_O = {L: model.model.layers[L].self_attn.o_proj.weight.detach().float() for L in late}
    gamma = model.model.norm.weight.detach().float()
    eps = model.config.rms_norm_eps
    W_U = model.lm_head.weight.detach()
    hd = model.config.hidden_size // nH

    recs = []
    A = {c: np.full((len(D), nL, nH), np.nan, np.float32) for c in ("full", "suffix", "none", "qswap_h", "qswap_h2",
                                                                    "full_h2")}
    DLA = np.full((len(D), len(late), nH), np.nan, np.float32)
    exact = {"k_self": 0.0, "v_self": 0.0}
    t0 = time.time()
    for chunk in runner.batches(range(len(D)), lambda i: lens[i]):
        X = [D[i] for i in chunk]
        rows = list(range(len(X)))
        R, Hp, H2 = [x["readout"] for x in X], [x["hpos"] for x in X], [x["h2pos"] for x in X]

        def lp_of(cl, codes_key):
            out = []
            for r, x in enumerate(X):
                lp = torch.log_softmax(cl[x["d"]][r].double(), -1)
                c = x[codes_key]
                out.append(float(lp[col[x["d"]][c]]) if c >= 0 else np.nan)
            return np.array(out)

        def jobs(cond, ko=False):
            return [Job(x["enc"][cond], knockouts=[([x["hpos"]], x["hprev"], early, None, False)] if ko else [])
                    for x in X]

        obs_pairs = [(r, R[r]) for r in rows]
        res = {}
        for cond in ("full", "suffix", "none"):                                   # (a)
            if cond == "full":
                with qkv.o_input(model, late, rows, R) as oi:
                    _, cl, cap, obs = runner.forward(jobs(cond), capture=[f"model.layers.{nL - 1}"],
                                                     observe={"layers": list(range(nL)), "pairs": obs_pairs})
                resid = cap[f"model.layers.{nL - 1}"][torch.arange(len(X), device=cap[f"model.layers.{nL - 1}"].device),
                                                      torch.as_tensor(R, device=cap[f"model.layers.{nL - 1}"].device)].float()
                rms = resid.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
                for r, x in enumerate(X):                                         # (e)
                    ids = torch.as_tensor(vocab.ids(x["d"]), device=W_U.device)
                    u = W_U[vocab.id_of(x["d"], x["h_code"])].float() - W_U[ids].float().mean(0)
                    u = gamma * u / rms[r]
                    for li, L in enumerate(late):
                        z = oi[L][r].float().view(nH, hd)
                        contrib = torch.einsum("dhk,hk->hd", W_O[L].view(-1, nH, hd), z)
                        DLA[chunk[r], li] = (contrib @ u).cpu().numpy()
            else:
                _, cl, _, obs = runner.forward(jobs(cond), observe={"layers": list(range(nL)), "pairs": obs_pairs})
            res[cond] = lp_of(cl, "h_code")
            if cond == "full":
                res["full_h2"] = lp_of(cl, "h2_code")
            for L in range(nL):
                pr = obs[L].numpy()
                for r in rows:
                    A[cond][chunk[r], L] = pr[r, :, Hp[r]]
                    if cond == "full" and H2[r] >= 0:
                        A["full_h2"][chunk[r], L] = pr[r, :, H2[r]]
        with qkv.capture(model, "k", late, rows, Hp) as kc, qkv.capture(model, "v", late, rows, Hp) as vc:  # (b) source
            runner.forward(jobs("suffix"))
        with qkv.capture(model, "k", late, rows, Hp) as kf:
            runner.forward(jobs("full"))
        with qkv.patch(model, "k", kf, rows, Hp):                                  # no-op control
            _, cl, _, _ = runner.forward(jobs("full"))
        exact["k_self"] = max(exact["k_self"], float(np.nanmax(np.abs(lp_of(cl, "h_code") - res["full"]))))
        with qkv.patch(model, "k", kc, rows, Hp):
            _, cl, _, _ = runner.forward(jobs("full"))
        res["k_from_suffix"] = lp_of(cl, "h_code")
        with qkv.patch(model, "v", vc, rows, Hp):
            _, cl, _, _ = runner.forward(jobs("full"))
        res["v_from_suffix"] = lp_of(cl, "h_code")
        for cond in ("full", "suffix", "none"):                                   # (c)
            _, cl, _, _ = runner.forward(jobs(cond, ko=True))
            res[f"{cond}_early_ko"] = lp_of(cl, "h_code")
        dn = [r for r, x in enumerate(X) if x["j2"] is not None]                  # (d)
        res["qswap_h"] = np.full(len(X), np.nan)
        res["qswap_h2"] = np.full(len(X), np.nan)
        if dn:
            donor_jobs = [Job(X[r]["enc"]["donor"]) for r in dn]
            with qkv.capture(model, "q", late, list(range(len(dn))), [R[r] for r in dn]) as qc:
                runner.forward(donor_jobs)
            sub = [Job(X[r]["enc"]["full"]) for r in dn]
            with qkv.patch(model, "q", qc, list(range(len(dn))), [R[r] for r in dn]):
                _, cl, _, obs = runner.forward(sub, observe={"layers": list(late),
                                                             "pairs": [(i, R[r]) for i, r in enumerate(dn)]})
            for i, r in enumerate(dn):
                lp = torch.log_softmax(cl[X[r]["d"]][i].double(), -1)
                res["qswap_h"][r] = float(lp[col[X[r]["d"]][X[r]["h_code"]]])
                res["qswap_h2"][r] = float(lp[col[X[r]["d"]][X[r]["h2_code"]]])
                for L in late:
                    pr = obs[L].numpy()
                    A["qswap_h"][chunk[r], L] = pr[i, :, Hp[r]]
                    A["qswap_h2"][chunk[r], L] = pr[i, :, H2[r]]
        for r, x in enumerate(X):
            recs.append({kk: x[kk] for kk in ("example_id", "user_id", "d", "k", "h_code", "h2_code", "gold_code",
                                              "synthetic")} | {"design": chunk[r], "has_donor": x["j2"] is not None,
                                                               **{f"logp_{c}": float(v[r]) for c, v in res.items()}})
        if len(recs) % 2048 < len(X):
            log(f"{len(recs)}/{len(D)} designs, {runner.n_forwards} forwards, {time.time() - t0:.0f}s")
    df = pd.DataFrame(recs).sort_values("design")
    df.to_parquet(args.out / "records.parquet", index=False)
    np.savez_compressed(args.out / "attention.npz", late=np.asarray(late), **{f"attn_{k}": v for k, v in A.items()},
                        dla=DLA)
    validation = {**gin.validation, "noop_k_self_patch_max_abs_diff": exact["k_self"],
                  "noop_k_self_patch_exact": exact["k_self"] == 0.0,
                  "seconds": round(time.time() - t_start, 1)}
    if not validation["noop_k_self_patch_exact"]:
        raise RuntimeError(f"patching keys with their own values changed the outputs ({exact['k_self']})")
    (args.out / "inputs.json").write_text(json.dumps({
        "checkpoint": ckpt, "grid_key": entry["key"], "n_rows": len(examples), "n_designs": len(D),
        "n_donor": int(df.has_donor.sum()), "late_layers": list(late), "early_layers": list(early),
        "buckets": buckets, "batch_rows": args.rows, "limited": args.limit, **gin.info,
        **grid.cell_info(ckpt)}, indent=1, default=str))
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: {len(D)} designs, {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
