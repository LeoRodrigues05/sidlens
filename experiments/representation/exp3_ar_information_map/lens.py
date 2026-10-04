#!/usr/bin/env python
"""Stage L: what the model thinks its answer is at every layer (protocol.md).

L1  tuned lens: per layer, an affine translator h + A h + b (zero-initialised)
    trained so that lm_head(norm(.)) over each digit's code tokens matches the
    final code distribution at the readout positions (KL). The logit lens is
    the same with A = b = 0. Scored on test: trie-constrained top-1 at R_d
    (given the model's own answer prefix) against the answer, golden and r1.
L2  SID decoders: per layer and task, a small GRU that reads ONE residual
    vector and writes a whole SID, trie-constrained (beam 10). Tasks: the item
    at a history position (from its last or first token), the previous item,
    and, from the first readout, the model's answer, the golden target, r1.

Traps guarded here:

* Readout tokens come from the self-forced capture; the token at R_d sits
  after the answer's first d digits, so the "prefix" for the trie is the
  answer prefix, never the golden one.
* Final distributions are computed from the captured final-norm site through
  the lm_head, not re-run; the constrained argmax of that distribution must
  equal the stored greedy digit on >= 99% of readouts (bf16 ties aside), or
  the run stops: a lower rate means the readouts and answers are misaligned.
* Decoders are fitted on train tokens only; early stopping uses a 5% train
  user holdout; test tokens are only scored.

    python lens.py --cell 0 --train-capture <dir> --test-capture <dir> --out <new dir>
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
import torch.nn as nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "src"))

from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.data.sids import SidTable                                   # noqa: E402
from sidlens.hooks.store import StoreReader                              # noqa: E402
from sidlens.models import ar                                            # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
DEC_LAYERS = tuple(range(0, 27, 2)) + (27,)
SEED = 20261003
TASKS = ("item_at_last", "item_at_first", "prev_item_at_last", "answer_at_R0", "golden_at_R0", "r1_at_R0")


def user_bucket(user: str, mod: int) -> int:
    return int(hashlib.sha256(f"{SEED}|{user}".encode()).hexdigest(), 16) % mod


def trie(table: SidTable, n: int):
    nxt: dict[tuple, list] = {}
    for row in map(tuple, table.codes.tolist()):
        for d in range(n):
            nxt.setdefault(row[:d], set()).add(row[d])
    return {k: sorted(v) for k, v in nxt.items()}


def token_frame(store: StoreReader, capture: Path, split: str, variant: str, n: int) -> pd.DataFrame:
    """Readout and history tokens with everything the lenses and decoders need."""
    rows = store.rows.reset_index(drop=True)
    ex = {e.example_id: e for e in P.load_examples(variant, "next-item", split)}
    g = pd.read_csv(capture / "self_greedy.csv").set_index("example_id")
    hdr = rows.role == "response_header"
    last_hdr = rows[hdr].groupby("example_id").pos.transform("max")
    is_r0 = np.zeros(len(rows), bool)
    is_r0[rows.index[hdr][rows[hdr].pos.to_numpy() == last_hdr.to_numpy()]] = True
    out = []
    for i, r in enumerate(rows.itertuples(index=False)):
        e = ex[r.example_id]
        hc = [P.parse_sid(s, n) for s in e.history_sids]
        ans = P.parse_sid(g.at[r.example_id, "greedy_sid"], n)
        base = {"i": i, "example_id": r.example_id, "user_id": e.user_id,
                "answer": ans, "golden": P.parse_sid(e.target_sids[0], n), "r1": hc[-1]}
        if is_r0[i]:
            out.append({**base, "kind": "R", "d": 0})
        elif r.role == "target_sid" and 0 <= int(r.digit) < n - 1:
            out.append({**base, "kind": "R", "d": int(r.digit) + 1})
        elif r.role == "hist_sid" and int(r.digit) in (0, n - 1):
            k = int(r.item)
            out.append({**base, "kind": "H", "d": int(r.digit), "k": k, "item": hc[k],
                        "prev": hc[k - 1] if k >= 1 else None})
    return pd.DataFrame(out)


class Unembed:
    """lm_head over each digit's code tokens, with the model's final RMSNorm."""

    def __init__(self, model, vocab, n, dev):
        self.n = n
        self.W = [model.lm_head.weight[torch.as_tensor(vocab.ids(d), device=model.lm_head.weight.device)]
                  .detach().float().to(dev) for d in range(n)]
        self.codes = [vocab.codes(d) for d in range(n)]
        self.norm_w = model.model.norm.weight.detach().float().to(dev)
        self.eps = float(model.model.norm.variance_epsilon)

    def norm(self, h):
        return self.norm_w * h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)

    def logits(self, h_normed, d):
        return h_normed @ self.W[d].T


def constrained_top1(logits: torch.Tensor, codes: list[int], prefixes, nxt) -> list[int]:
    col = {c: j for j, c in enumerate(codes)}
    out = []
    lg = logits.cpu()
    for r, p in enumerate(prefixes):
        legal = nxt[p]
        j = [col[c] for c in legal]
        out.append(legal[int(lg[r, j].argmax())])
    return out


def run_lens(tr, te, trf, tef, unemb, nxt, n, layers, dev, log):
    """L1: tuned + logit lens at every layer; per-token predictions on test."""
    rec = []
    tr_r, te_r = trf[trf.kind == "R"], tef[tef.kind == "R"]
    fin_tr = tr.load("model.norm", tr_r.i.to_numpy()).float().to(dev)   # captured site is post-norm
    fin_te = te.load("model.norm", te_r.i.to_numpy()).float().to(dev)
    dtr, dte = tr_r.d.to_numpy(), te_r.d.to_numpy()
    p_fin_tr = {d: torch.log_softmax(unemb.logits(fin_tr[dtr == d], d), -1) for d in range(n)}
    rows_of = {d: np.flatnonzero(dtr == d) for d in range(n)}            # p_fin_tr[d] row order
    pref_te = [tuple(a[:d]) for a, d in zip(te_r.answer, dte)]
    # alignment check (amendment A1): the greedy digit must be a constrained argmax of the
    # recomputed final logits up to bf16 resolution. Greedy decoding ran on bf16 logits,
    # whose spacing near |logit| ~ 16 is 0.125, so exact fp32 argmax agreement is only a
    # ceiling (reported), not a test of alignment.
    strict, tied = [], []
    for d in range(n):
        m = dte == d
        lg = unemb.logits(fin_te[m], d).cpu()
        col = {c: j for j, c in enumerate(unemb.codes[d])}
        for r, (p, a) in enumerate(zip([p for p, mm in zip(pref_te, m) if mm], te_r.answer[m])):
            j = [col[c] for c in nxt[p]]
            mx = float(lg[r, j].max())
            g = float(lg[r, col[a[d]]])
            strict.append(nxt[p][int(lg[r, j].argmax())] == a[d])
            tied.append(mx - g <= max(abs(mx), 1.0) * 2.0 ** -7)
    align, ceiling = float(np.mean(tied)), float(np.mean(strict))
    log(f"[lens] greedy digit is a constrained argmax up to bf16 resolution on {align:.4f} of test "
        f"readouts; strict fp32 argmax agreement (ceiling) {ceiling:.4f}")
    if align < 0.99:
        raise RuntimeError(f"readouts and answers misaligned ({align:.4f} < 0.99)")
    hold = np.array([user_bucket(u, 20) == 0 for u in tr_r.user_id])
    for L in layers:
        site = f"model.layers.{L}"
        h_tr = tr.load(site, tr_r.i.to_numpy()).float().to(dev)
        h_te = te.load(site, te_r.i.to_numpy()).float().to(dev)
        A = torch.zeros(h_tr.shape[1], h_tr.shape[1], device=dev, requires_grad=True)
        b = torch.zeros(h_tr.shape[1], device=dev, requires_grad=True)
        opt = torch.optim.Adam([A, b], lr=1e-3)
        g = torch.Generator(device="cpu").manual_seed(SEED + L)
        idx_train = np.flatnonzero(~hold)
        for ep in range(3):
            perm = idx_train[torch.randperm(len(idx_train), generator=g).numpy()]
            for s in range(0, len(perm), 2048):
                bi = perm[s:s + 2048]
                loss = 0.0
                for d in range(n):
                    m = bi[dtr[bi] == d]
                    if not len(m):
                        continue
                    hp = unemb.norm(h_tr[m] + h_tr[m] @ A.T + b)
                    lq = torch.log_softmax(unemb.logits(hp, d), -1)
                    pos = np.searchsorted(rows_of[d], m)
                    lp = p_fin_tr[d][pos]
                    loss = loss + (lp.exp() * (lp - lq)).sum() / len(bi)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            def kl_on(ix):
                tot = 0.0
                for d in range(n):
                    m = ix[dtr[ix] == d]
                    if not len(m):
                        continue
                    pos = np.searchsorted(rows_of[d], m)
                    lp = p_fin_tr[d][pos]
                    lq = torch.log_softmax(unemb.logits(unemb.norm(h_tr[m] + h_tr[m] @ A.T + b), d), -1)
                    tot += float((lp.exp() * (lp - lq)).sum())
                return tot / len(ix)
            kl_hold = kl_on(np.flatnonzero(hold))
            for lens, hh in (("tuned", h_te + h_te @ A.T + b), ("logit", h_te)):
                hn = unemb.norm(hh)
                for d in range(n):
                    m = dte == d
                    lg = unemb.logits(hn[m], d)
                    lq = torch.log_softmax(lg, -1)
                    lp = torch.log_softmax(unemb.logits(fin_te[m], d), -1)
                    kl = (lp.exp() * (lp - lq)).sum(-1).cpu().numpy()
                    top = constrained_top1(lg, unemb.codes[d], [p for p, mm in zip(pref_te, m) if mm], nxt)
                    sub = te_r[m]
                    for t, k_, r in zip(top, kl, sub.itertuples(index=False)):
                        rec.append({"layer": L, "lens": lens, "d": d, "example_id": r.example_id,
                                    "user_id": r.user_id, "top1": t, "answer_d": r.answer[d],
                                    "golden_d": r.golden[d], "r1_d": r.r1[d], "kl_final": float(k_),
                                    "copy_answer": r.answer == r.r1})
        log(f"[lens] L{L}: train-holdout KL {kl_hold:.4f}; tuned D1 "
            f"{np.mean([x['top1'] == x['answer_d'] for x in rec if x['layer'] == L and x['lens'] == 'tuned']):.3f}")
        del h_tr, h_te
        torch.cuda.empty_cache()
    return pd.DataFrame(rec), (align, ceiling)


class SidDecoder(nn.Module):
    """One residual vector -> a whole SID, digit by digit."""

    def __init__(self, d_in, K, n, width=512):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.code_emb = nn.Embedding(K + 1, width)          # K = BOS
        self.digit_emb = nn.Embedding(n, width)
        self.cell = nn.GRUCell(width, width)
        self.out = nn.Linear(width, K)
        self.K, self.n = K, n

    def forward(self, x, codes):                            # teacher forced; codes (B, n)
        h = torch.tanh(self.inp(x))
        prev = torch.full((len(x),), self.K, device=x.device)
        logits = []
        for d in range(self.n):
            dd = torch.full_like(prev, d)
            h = self.cell(self.code_emb(prev) + self.digit_emb(dd), h)
            logits.append(self.out(h))
            prev = codes[:, d]
        return torch.stack(logits, 1)                       # (B, n, K)

    @torch.no_grad()
    def beam(self, x, masks, width=10):
        """Trie-constrained batched beam search; returns the top-`width` SIDs per row.

        `masks` maps a prefix tuple to a bool (K,) tensor of legal next codes.
        Beams that run out of legal continuations keep score -inf and are dropped.
        """
        B, K, dev = len(x), self.K, x.device
        h = torch.tanh(self.inp(x)).unsqueeze(1)                     # (B, 1, H)
        scores = torch.zeros(B, 1, device=dev)
        prev = torch.full((B, 1), K, device=dev)
        pref = [[()] for _ in range(B)]
        dead = torch.zeros(K, dtype=torch.bool, device=dev)
        for d in range(self.n):
            W = scores.shape[1]
            hn = self.cell(self.code_emb(prev.reshape(-1)) + self.digit_emb(torch.full((B * W,), d, device=dev)),
                           h.reshape(B * W, -1))
            lp = torch.log_softmax(self.out(hn), -1).reshape(B, W, K)
            m = torch.stack([masks.get(p, dead) if p is not None else dead for row in pref for p in row])
            lp = lp.masked_fill(~m.reshape(B, W, K), float("-inf"))
            tot = (scores.unsqueeze(-1) + lp).reshape(B, -1)
            v, j = tot.topk(min(width, W * K), dim=-1)
            bi, ci = (j // K).cpu().numpy(), (j % K).cpu().numpy()
            fin = torch.isfinite(v).cpu().numpy()
            pref = [[(pref[r][bi[r, t]] + (int(ci[r, t]),)) if fin[r, t] and pref[r][bi[r, t]] is not None else None
                     for t in range(v.shape[1])] for r in range(B)]
            h = hn.reshape(B, W, -1)[torch.arange(B, device=dev)[:, None], torch.as_tensor(bi, device=dev)]
            scores, prev = v, torch.as_tensor(ci, device=dev)
        return [[p for p in row if p is not None] for row in pref]


def task_rows(f: pd.DataFrame, task: str, n: int):
    if task == "item_at_last":
        g = f[(f.kind == "H") & (f.d == n - 1)]
        return g, list(g["item"])
    if task == "item_at_first":
        g = f[(f.kind == "H") & (f.d == 0)]
        return g, list(g["item"])
    if task == "prev_item_at_last":
        g = f[(f.kind == "H") & (f.d == n - 1) & f["prev"].notna()]
        return g, list(g["prev"])
    g = f[(f.kind == "R") & (f.d == 0)]
    col = {"answer_at_R0": "answer", "golden_at_R0": "golden", "r1_at_R0": "r1"}[task]
    return g, list(g[col])


def fit_decoder(Xtr, Ytr, Xho, Yho, K, n, dev, zero_input=False):
    torch.manual_seed(SEED)
    model = SidDecoder(Xtr.shape[1], K, n).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    best, best_state = np.inf, None
    for ep in range(10):
        model.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(perm), 1024):
            bi = perm[s:s + 1024]
            x = torch.zeros_like(Xtr[bi]) if zero_input else Xtr[bi]
            lg = model(x, Ytr[bi])
            loss = ce(lg.reshape(-1, K), Ytr[bi].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            x = torch.zeros_like(Xho) if zero_input else Xho
            v = float(ce(model(x, Yho).reshape(-1, K), Yho.reshape(-1)))
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def run_decoders(tr, te, trf, tef, table, nxt, n, layers, dev, log):
    K = table.variant.codebook_size
    masks = {}
    for p, legal in nxt.items():
        m = torch.zeros(K, dtype=torch.bool, device=dev)
        m[torch.as_tensor(legal, device=dev)] = True
        masks[p] = m
    rec, summ = [], []
    for task in TASKS:
        gtr, ytr = task_rows(trf, task, n)
        gte, yte = task_rows(tef, task, n)
        hold = np.array([user_bucket(u, 20) == 0 for u in gtr.user_id])
        Ytr_all = torch.as_tensor(np.array(ytr), device=dev)
        Yte = np.array(yte)
        for L in ("prior", *layers):
            site = f"model.layers.{layers[0] if L == 'prior' else L}"
            Xtr = tr.load(site, gtr.i.to_numpy()).float().to(dev)
            Xte = te.load(site, gte.i.to_numpy()).float().to(dev)
            mu, sd = Xtr[~torch.as_tensor(hold)].mean(0), Xtr[~torch.as_tensor(hold)].std(0) + 1e-4
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            hm = torch.as_tensor(hold, device=dev)
            model, ho = fit_decoder(Xtr[~hm], Ytr_all[~hm], Xtr[hm], Ytr_all[hm], K, n, dev,
                                    zero_input=(L == "prior"))
            xin = torch.zeros_like(Xte) if L == "prior" else Xte
            beams = []
            for s0 in range(0, len(xin), 512):
                beams += model.beam(xin[s0:s0 + 512], masks, width=10)
            top1 = np.array([b[0] for b in beams])
            exact = (top1 == Yte).all(1)
            top10 = np.array([tuple(y) in set(b) for y, b in zip(map(tuple, Yte), beams)])
            for r, e, t10, t, y in zip(gte.itertuples(index=False), exact, top10, top1, Yte):
                rec.append({"task": task, "layer": str(L), "example_id": r.example_id, "user_id": r.user_id,
                            "exact": bool(e), "top10": bool(t10), "top1": ",".join(map(str, t)),
                            "target": ",".join(map(str, y))})
            summ.append({"task": task, "layer": str(L), "holdout_ce": ho, "n_train": int((~hold).sum()),
                         "n_test": len(gte), "exact": float(exact.mean()), "top10": float(top10.mean()),
                         **{f"digit{d}": float((top1[:, d] == Yte[:, d]).mean()) for d in range(n)}})
            log(f"[dec] {task} L{L}: exact {exact.mean():.3f} top10 {top10.mean():.3f}")
            del Xtr, Xte, model
            torch.cuda.empty_cache()
    return pd.DataFrame(rec), pd.DataFrame(summ)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--train-capture", type=Path, required=True, help="ar_capture --out dir (holds store/)")
    ap.add_argument("--test-capture", type=Path, required=True)
    ap.add_argument("--parts", default="L1,L2")
    ap.add_argument("--lens-layers", default=",".join(map(str, range(28))))
    ap.add_argument("--dec-layers", default=",".join(map(str, DEC_LAYERS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)       # noqa: E731
    ckpt = CELLS[args.cell]
    tr, te = StoreReader(args.train_capture / "store"), StoreReader(args.test_capture / "store")
    for st, split in ((tr, "train"), (te, "test")):
        if st.meta.get("target") != "self-greedy" or st.meta["split"] != split or st.meta["model"] != f"ar/{ckpt}":
            raise RuntimeError(f"{st.root}: not the self-greedy {split} capture of {ckpt}")
    variant = tr.meta["variant"]
    table = SidTable.load(variant)
    n = table.variant.n_codebook
    nxt = trie(table, n)
    trf = token_frame(tr, args.train_capture, "train", variant, n)
    tef = token_frame(te, args.test_capture, "test", variant, n)
    info = {"cell": ckpt, "variant": variant, "n_train_tokens": len(trf), "n_test_tokens": len(tef),
            "train_capture": str(args.train_capture), "test_capture": str(args.test_capture),
            "parts": args.parts, "seed": SEED}
    parts = set(args.parts.split(","))
    if "L1" in parts:
        model = ar.load_model(ckpt, device=args.device, dtype="float32")
        vocab = ar.load_vocab(ckpt)
        unemb = Unembed(model, vocab, n, args.device)
        del model
        torch.cuda.empty_cache()
        lens, align = run_lens(tr, te, trf, tef, unemb, nxt, n,
                               [int(x) for x in args.lens_layers.split(",")], args.device, log)
        lens.to_parquet(args.out / "lens_tokens.parquet", index=False)
        info["final_alignment_bf16_tolerant"], info["final_strict_agreement_ceiling"] = align
    if "L2" in parts:
        recs, summ = run_decoders(tr, te, trf, tef, table, nxt, n,
                                  [int(x) for x in args.dec_layers.split(",")], args.device, log)
        recs.to_parquet(args.out / "decoder_tokens.parquet", index=False)
        summ.to_csv(args.out / "decoder_summary.csv", index=False)
    (args.out / "inputs.json").write_text(json.dumps(info, indent=1, default=str))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
