#!/usr/bin/env python
"""N1: does the copy of a history item's digit need its whole prefix, or only the digit before it?

Protocol: `protocol.md` beside this file, declared before any condition was
scored. One invocation = one next-item grid run (AR or DiffGRM) at its best
state. Teacher-forced: the golden prefix is given and digit d is scored, with
the most recent prefix-matching history item h replaced by h' (`design`).

Traps, each with its guard
--------------------------
1. **Another item supporting the same code.** Rows where any other history
   item supports h's digit-d code under either reading are excluded in
   `design`, so `none` really removes the support.
2. **A replacement that changes more than h.** AR: `history.check_replacement`
   proves only h's tokens from its first changed digit on differ. DiffGRM: the
   variant history differs from the clean one in slot k only (checked).
3. **Codes scored by vocabulary position.** AR code logits are indexed through
   `vocab.codes(d)`, never by the code value.
4. **Synthetic SIDs mixed with catalog items.** Every record carries
   `synthetic`; the summary reports the two apart.

    python run.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 256] [--device cpu]
"""

from __future__ import annotations

import argparse
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
from sidlens.interventions import history as Hi                           # noqa: E402
from sidlens.models import ar                                             # noqa: E402

SEED = 20261021
CONDS = ("suffix", "none", "prefix")


def design(eid: str, hist: list[tuple], target: tuple, pool_codes: np.ndarray, pool_ids: np.ndarray,
           excl_ids: set, digit_codes: list[list[int]]) -> list[dict]:
    """Eligible (d, h) of one row and the replacement h' of each condition (protocol table)."""
    n = len(target)
    hset = set(hist) | {target}
    usable = ~np.isin(pool_ids, np.fromiter(excl_ids, dtype=np.int64)) & \
        ~np.array([tuple(c) in hset for c in pool_codes.tolist()])
    out = []
    for d in range(2, n):
        ks = [k for k, c in enumerate(hist) if c[:d] == target[:d]]
        if not ks:
            continue
        k = ks[-1]
        h = hist[k]
        if any(j != k and c[d] == h[d] and (c[:d] == target[:d] or c[d - 1] == target[d - 1])
               for j, c in enumerate(hist)):
            continue
        C, T = pool_codes, np.asarray(target)
        masks = {"suffix": (C[:, d - 1] == h[d - 1]) & (C[:, d] == h[d]) & ~np.all(C[:, :d - 1] == T[:d - 1], axis=1),
                 "none": (C[:, d - 1] != T[d - 1]) & (C[:, d] != h[d]),
                 "prefix": np.all(C[:, :d] == T[:d], axis=1) & (C[:, d] != h[d])}
        rec = {"d": d, "k": k, "h": h, "repl": {}}
        for cond, m in masks.items():
            rng = Hi.condition_rng(SEED, eid, f"{cond}|{k}", d)
            idx = np.flatnonzero(m & usable)
            if len(idx):
                j = idx[rng.integers(len(idx))]
                rec["repl"][cond] = (tuple(int(x) for x in C[j]), int(pool_ids[j]), False, len(idx))
                continue
            c = list(h)                                   # outside the catalog (trap 4)
            pick = lambda dd, avoid: int(rng.choice([x for x in digit_codes[dd] if x != avoid]))  # noqa: E731
            if cond == "suffix":
                c[0] = pick(0, target[0])
            elif cond == "none":
                c[d - 1], c[d] = pick(d - 1, target[d - 1]), pick(d, h[d])
            else:
                c[d] = pick(d, h[d])
            if tuple(c) in hset:
                continue
            rec["repl"][cond] = (tuple(c), -1, True, 0)
        if len(rec["repl"]) == len(CONDS):
            out.append(rec)
    return out


def run_ar(args, entry, log):
    from transformers import AutoTokenizer
    from sidlens.grid import inputs as GI
    from sidlens.interventions.runner import FixedShapeRunner, Job, length_buckets
    from sidlens.interventions.scoring import DigitScorer

    ckpt = args.ckpt
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    tok = AutoTokenizer.from_pretrained(str(ar.checkpoint_dir(ckpt)), local_files_only=True)
    gin = GI.ArSetup(ckpt, "test", tok, vocab, task="next-item")
    if not all(v["ok"] for v in gin.validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(gin.validation, default=str)[:1500]}")
    clean = [P.encode(ex, tok, vocab, template=gin.template, with_target=True) for ex in gin.examples]
    examples, clean = GI.sort_by_length(gin.examples, clean)
    if args.limit:
        examples, clean = examples[:args.limit], clean[:args.limit]
    pool = Hi.ControlPool(gin.table, gin.item2id)
    digit_codes = [sorted(set(pool.codes[:, d].tolist())) for d in range(n)]
    col = [{c: j for j, c in enumerate(vocab.codes(d))} for d in range(n)]
    buckets = length_buckets([len(e) for e in clean])
    model = ar.load_model(ckpt, device=args.device, dtype="bfloat16" if args.device == "cuda" else "float32")
    runner = FixedShapeRunner(model, DigitScorer(vocab, gin.table), tok.pad_token_id, buckets[-1], args.rows, 1,
                              buckets=buckets)
    jobs, recs = [], []

    def flush():
        if not jobs:
            return
        _, cl, _, _ = runner.forward(jobs)
        for r, job in enumerate(jobs):
            m = job.meta
            for d in m["digits"]:
                lp = torch.log_softmax(cl[d][r].double(), -1)
                rec = {kk: v for kk, v in m.items() if kk != "digits"}
                if m["condition"] == "full":
                    rec = {**rec, **m["by_d"][d]}
                rec.update({"d": d, "logp_h": float(lp[col[d][rec["h_code"]]]),
                            "logp_alt": float(lp[col[d][rec["alt_code"]]]),
                            "logp_gold": float(lp[col[d][rec["gold_code"]]])})
                rec.pop("by_d", None)
                recs.append(rec)
        jobs.clear()

    def push(job):
        jobs.append(job)
        if len(jobs) >= runner.B:
            flush()

    n_design = 0
    for i, ex in enumerate(examples):
        hist = [P.parse_sid(s, n) for s in ex.history_sids]
        target = P.parse_sid(ex.target_sids[0], n)
        rows = design(ex.example_id, hist, target, pool.codes, pool.item_ids,
                      {*ex.history_item_ids, *ex.target_item_ids}, digit_codes)
        if not rows:
            continue
        n_design += len(rows)
        base = {"example_id": ex.example_id, "user_id": ex.user_id}
        by_d = {r["d"]: {"k": r["k"], "recency": len(hist) - r["k"], "h_code": r["h"][r["d"]],
                         "alt_code": r["h"][r["d"]], "gold_code": target[r["d"]], "synthetic": False,
                         "n_candidates": 0} for r in rows}
        push(Job(clean[i], meta={**base, "condition": "full", "digits": sorted(by_d), "by_d": by_d}))
        for r in rows:
            d, k, h = r["d"], r["k"], r["h"]
            for cond, (codes, item_id, synth, ncand) in r["repl"].items():
                e2 = P.encode(Hi.replace_history_item(ex, k, Hi.Control(item_id, codes, ncand)), tok, vocab,
                              template=gin.template, with_target=True)
                Hi.check_replacement(clean[i], e2, k, next(j for j in range(n) if codes[j] != h[j]))
                push(Job(e2, meta={**base, "condition": cond, "digits": [d], "k": k, "recency": len(hist) - k,
                                   "h_code": h[d], "alt_code": codes[d], "gold_code": target[d],
                                   "synthetic": synth, "n_candidates": ncand}))
        if (i + 1) % 1000 == 0:
            log(f"{i + 1}/{len(examples)} rows, {n_design} designs, {runner.n_forwards} forwards")
    flush()
    return recs, {"n_rows": len(examples), "n_designs": n_design, "buckets": buckets,
                  "n_forwards": runner.n_forwards}


def run_diffgrm(args, entry, log):
    from sidlens.grid import diff_setup as DS
    from sidlens.interventions import diffusion as DI
    from sidlens.models import diffusion

    dev = args.device
    S = DS.load(args.ckpt, device=dev)
    chk = DS.check(S, device=dev)
    if not chk["ok"]:
        raise RuntimeError(f"grid checks failed: {json.dumps(chk, default=str)[:1500]}")
    c, model = S.cohort, S.model
    n = int(model.n_digit)
    N = min(args.limit, len(c.users)) if args.limit else len(c.users)
    pool = Hi.ControlPool(S.table, S.item2id)
    digit_codes = [sorted(set(pool.codes[:, d].tolist())) for d in range(n)]
    hist_ids = DS.history_item_ids(S)
    recs, n_design, t0 = [], 0, time.time()
    for u0 in range(0, N, args.users_per_chunk):
        variants, meta = [], []                        # one encoder row per (user, condition, d)
        for i in range(u0, min(N, u0 + args.users_per_chunk)):
            L = int(c.history_lengths[i])
            hist = [tuple(int(x) for x in c.histories[i, j]) for j in range(L)]
            target = tuple(int(x) for x in c.target_sids[i])
            rows = design(c.example_ids[i], hist, target, pool.codes, pool.item_ids,
                          {*hist_ids[i], int(c.target_item_ids[i])}, digit_codes)
            if not rows:
                continue
            n_design += len(rows)
            base = {"example_id": c.example_ids[i], "user_id": c.users[i]}
            variants.append(c.histories[i])
            ci = len(variants) - 1
            for r in rows:
                d, k, h = r["d"], r["k"], r["h"]
                meta.append((ci, {**base, "condition": "full", "d": d, "k": k, "recency": L - k, "h_code": h[d],
                                  "alt_code": h[d], "gold_code": target[d], "synthetic": False, "n_candidates": 0}))
                for cond, (codes, item_id, synth, ncand) in r["repl"].items():
                    v = c.histories[i].copy()
                    v[k] = codes
                    if not (np.array_equal(np.delete(v, k, 0), np.delete(c.histories[i], k, 0))):
                        raise RuntimeError("a variant changed more than slot k")
                    variants.append(v)
                    meta.append((len(variants) - 1, {**base, "condition": cond, "d": d, "k": k, "recency": L - k,
                                                     "h_code": h[d], "alt_code": codes[d], "gold_code": target[d],
                                                     "synthetic": synth, "n_candidates": ncand}))
        if not variants:
            continue
        H = torch.as_tensor(np.stack(variants)).to(dev)
        M = torch.as_tensor(np.stack(variants)[..., 0] >= 0).to(dev)
        with torch.inference_mode():
            enc = torch.cat([diffusion.encode(model, H[s:s + 512], M[s:s + 512]) for s in range(0, len(H), 512)])
            cache = DI.project_cross_cache(model, enc)
        recs += score_chunk(model, enc, cache, meta, c, n, dev, args.dec_batch)
        log(f"{min(N, u0 + args.users_per_chunk)}/{N} users, {n_design} designs, {time.time() - t0:.0f}s")
    return recs, {"n_rows": N, "n_designs": n_design, "grid_checks": chk}


def score_chunk(model, enc, cache, meta, cohort, n, dev, dec_batch) -> list[dict]:
    """Digit d's log-probabilities at the golden prefix for every (encoder row, record)."""
    from sidlens.interventions import diffusion as DI

    index = {e: i for i, e in enumerate(cohort.example_ids)}
    out = []
    for s in range(0, len(meta), dec_batch):
        part = meta[s:s + dec_batch]
        rows = torch.as_tensor([p[0] for p in part], device=dev)
        codes = torch.as_tensor(np.stack([cohort.target_sids[index[m["example_id"]]] for _, m in part]),
                                device=dev)
        d = torch.as_tensor([m["d"] for _, m in part], device=dev)
        masked = torch.arange(n, device=dev)[None, :] >= d[:, None]          # golden prefix 0..d-1 revealed
        with torch.inference_mode():
            lp = DI.score_states(model, enc, cache, rows, codes, masked)
        for r, (_, m) in enumerate(part):
            dd = m["d"]
            out.append({**m, "logp_h": float(lp[r, dd, m["h_code"]]), "logp_alt": float(lp[r, dd, m["alt_code"]]),
                        "logp_gold": float(lp[r, dd, m["gold_code"]])})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a next-item grid weight state (grid:<key>@best)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128, help="AR: rows per fixed-shape forward")
    ap.add_argument("--users-per-chunk", type=int, default=256, help="DiffGRM: users encoded together")
    ap.add_argument("--dec-batch", type=int, default=2048, help="DiffGRM: decoder rows per forward")
    ap.add_argument("--device", default="cuda", help="cpu runs float32 (smoke tests; bit-exact controls hold on GPU only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t_start = time.time()

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    entry = registry.load().entry(args.ckpt)
    if entry["predtype"] != "next-item":
        ap.error("N1 is declared for next-item runs")
    recs, info = (run_ar if entry["paradigm"] == "ar" else run_diffgrm)(args, entry, log)
    df = pd.DataFrame(recs)
    df.to_parquet(args.out / "records.parquet", index=False)
    full = df[df.condition == "full"]
    (args.out / "inputs.json").write_text(json.dumps({
        "checkpoint": args.ckpt, "grid_key": entry["key"], "paradigm": entry["paradigm"], "seed": SEED,
        "limited": args.limit, **{k: v for k, v in info.items() if k != "grid_checks"},
        "n_records": len(df), "by_condition": df.condition.value_counts().to_dict(),
        "synthetic_by_condition": df[df.synthetic].condition.value_counts().to_dict(),
        **grid.cell_info(args.ckpt)}, indent=1, default=str))
    (args.out / "validation.json").write_text(json.dumps({
        "grid_checks": info.get("grid_checks"), "every_design_has_all_conditions":
            bool(len(full) and all(df.groupby(["example_id", "d"]).condition.nunique() == len(CONDS) + 1)),
        "seconds": round(time.time() - t_start, 1)}, indent=1, default=str))
    log(f"done: {len(full)} designs, {len(df)} records, {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
