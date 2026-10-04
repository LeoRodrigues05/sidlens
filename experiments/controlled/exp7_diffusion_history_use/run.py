#!/usr/bin/env python
"""DiffGRM history use: item replacement (A), prefix-matched copying (C), cross-attention knockout (K).

Protocol: `protocol.md` beside this file (declared before any DiffGRM forward
on this cluster). One invocation = one matched cell. Every digit score goes
through `sidlens.interventions.diffusion.score_states` (projected cross cache,
the path exp1 validated); every decoder forward carries a cross-attention
mask, all ones unless something is knocked out, so all conditions share one
code path. Encoder and decoder batches have fixed shapes (short batches are
filled with copies of a real row whose outputs are dropped).

    python run.py --cell 0 --out <new dir> [--limit 64]
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

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.matched_decode import decode as matched_decode     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.data.sids import SidTable, load_item2id                     # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.interventions.history import ControlPool, condition_rng     # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

CELLS = ("diff-next1-rqkmeans-3cb-128", "diff-next1-rqvae-4cb-128")
SEED = 20260928
MAX_RECENCY = 10
ENC_B, DEC_B, USERS_PER_CHUNK = 512, 2048, 64
DEV = "cuda"


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def encode_fixed(model, hist: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """Encoder states for every row, in fixed ENC_B-row batches."""
    out = []
    for s in range(0, len(hist), ENC_B):
        h, m = hist[s:s + ENC_B], mask[s:s + ENC_B]
        real = len(h)
        if real < ENC_B:
            h = np.concatenate([h, np.repeat(h[:1], ENC_B - real, 0)])
            m = np.concatenate([m, np.repeat(m[:1], ENC_B - real, 0)])
        e = diffusion.encode(model, torch.from_numpy(h).to(DEV), torch.from_numpy(m).to(DEV))
        out.append(e[:real])
    return torch.cat(out)


def score_fixed(model, enc, cache, jobs, n, K, n_layers, S, enc_mask):
    """jobs: dicts with enc_row, codes (n,), masked (n,), knock (list of (layers, queries, slots)).

    Returns log-probs [len(jobs), n, K] (CPU float32), in fixed DEC_B-row forwards.
    """
    outs = []
    for s in range(0, len(jobs), DEC_B):
        chunk = jobs[s:s + DEC_B]
        real = len(chunk)
        chunk = chunk + [chunk[0]] * (DEC_B - real)
        rows = torch.as_tensor([j["enc_row"] for j in chunk], device=DEV)
        codes = torch.as_tensor(np.stack([j["codes"] for j in chunk]), device=DEV)
        masked = torch.as_tensor(np.stack([j["masked"] for j in chunk]), device=DEV)
        edges = [(r, ly, q, sl, j["enc_row"]) for r, j in enumerate(chunk[:real])
                 for ly, q, sl in j["knock"]]
        xm = torch.from_numpy(DI.knockout_mask(n_layers, DEC_B, n, S, edges, enc_mask)).to(DEV)
        with torch.inference_mode():
            lp = DI.score_states(model, enc, cache, rows, codes, masked, xmask=xm)
        outs.append(lp[:real].cpu())
    return torch.cat(outs) if outs else torch.empty(0, n, K)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N users (pilot)")
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    global DEV
    DEV = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t_start = time.time()

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    ckpt_id = CELLS[args.cell]
    entry = load_runtime()[ckpt_id]
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    cohort = load_eval_cohort(entry)
    model, report = diffusion.load(entry, device=DEV)
    n, K = int(model.n_digit), int(model.codebook_size)
    n_layers, S = len(model.decoder_blocks), cohort.histories.shape[1]
    table = SidTable.load(entry["sem_ids_name"])
    pool = ControlPool(table, load_item2id("Industrial_and_Scientific"))
    if pool.n_digits != n:
        raise ValueError("SID table depth differs from the model")
    N = min(args.limit, len(cohort)) if args.limit else len(cohort)

    # ---- validation before any condition ----------------------------------------
    validation = {"model_report_ok": report["ok"],
                  "V1_exp1_pilot_checks": exp1_validate()(
                      model, torch.from_numpy(cohort.histories[:8]).to(DEV),
                      torch.from_numpy(cohort.history_mask[:8]).to(DEV), cohort.sid_buckets, matched_decode)}
    enc8 = encode_fixed(model, cohort.histories[:8], cohort.history_mask[:8])
    cache8 = DI.project_cross_cache(model, enc8)
    rows8 = torch.arange(8, device=DEV)
    allm = torch.ones((8, n), dtype=torch.bool, device=DEV)
    zero = torch.zeros((8, n), dtype=torch.long, device=DEV)
    with torch.inference_mode():
        a = DI.score_states(model, enc8, cache8, rows8, zero, allm)
        b = DI.score_states(model, enc8, cache8, rows8, zero, allm,
                            xmask=torch.ones((n_layers, 8, n, S), device=DEV))
    validation["V2_all_ones_mask_is_noop"] = bool(torch.equal(a, b))
    if not validation["V2_all_ones_mask_is_noop"]:
        raise RuntimeError("an all-ones cross-attention mask changed the logits")
    inputs = {"checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
              "cohort_sha256": cohort.cohort_sha256, "input_sha256": cohort.input_sha256, "n_users": N,
              "limited": args.limit, "n_digit": n, "codebook_size": K, "history_slots": S,
              "max_recency": MAX_RECENCY, "seed": SEED, "enc_batch": ENC_B, "dec_batch": DEC_B,
              "device": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "torch": torch.__version__}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    log(f"inputs ok: {ckpt_id} users={N} n={n} K={K}; V1 passed; V2 {validation['V2_all_ones_mask_is_noop']}")

    rec_a, rec_k, rec_clean, missing, inv = [], [], [], [], 0.0
    t0 = time.time()
    for u0 in range(0, N, USERS_PER_CHUNK):
        users = list(range(u0, min(N, u0 + USERS_PER_CHUNK)))
        hist, hmask, meta = [], [], []
        for i in users:                                          # clean row, then a duplicate (V3)
            for kind in ("clean", "clean_dup"):
                hist.append(cohort.histories[i]); hmask.append(cohort.history_mask[i])
                meta.append({"i": i, "kind": kind})
        for i in users:                                          # Part A variants
            L = int(cohort.history_lengths[i])
            codes_h = [tuple(int(c) for c in cohort.histories[i, j]) for j in range(L)]
            tgt = tuple(int(c) for c in cohort.target_sids[i])
            excl_codes = set(codes_h) | {tgt}
            excl_ids = {int(cohort.target_item_ids[i])}
            for r in range(1, min(MAX_RECENCY, L) + 1):
                k = L - r
                for m in range(n):
                    idx = pool.candidates(codes_h[k], m, excl_ids, excl_codes)
                    if len(idx) == 0:
                        missing.append({"user": cohort.users[i], "r": r, "m": m})
                        continue
                    j = idx[condition_rng(SEED, cohort.users[i], k, m).integers(len(idx))]
                    h = cohort.histories[i].copy()
                    h[k] = pool.codes[j]
                    hist.append(h); hmask.append(cohort.history_mask[i])
                    meta.append({"i": i, "kind": "A", "r": r, "k": k, "m": m,
                                 "control_item": int(pool.item_ids[j]), "n_candidates": int(len(idx))})
        hist, hmask = np.stack(hist), np.stack(hmask)
        enc = encode_fixed(model, hist, hmask)
        cache = DI.project_cross_cache(model, enc)

        jobs, jmeta = [], []
        for e, mt in enumerate(meta):
            i = mt["i"]
            tgt = cohort.target_sids[i].astype(np.int64)
            L = int(cohort.history_lengths[i])
            kstar = L - 1
            codes_h = cohort.histories[i]
            for d in range(n):
                masked = np.arange(n) >= d                       # S_pref(d); d = 0 is S_full
                codes = np.where(masked, 0, tgt)
                base = {"enc_row": e, "codes": codes, "masked": masked}
                if mt["kind"] in ("clean_dup", "A"):
                    jobs.append({**base, "knock": []}); jmeta.append({**mt, "d": d, "cond": "K0"})
                    continue
                match = [j for j in range(L) if tuple(codes_h[j, :d]) == tuple(tgt[:d])]
                ctrl = next((j for j in range(L - 2, -1, -1) if d == 0 or j not in match), None)
                conds = {"K0": [], "K1": [(range(n_layers), [d], [kstar])],
                         "K2": [(range(n_layers), list(range(n)), [kstar])]}
                if ctrl is not None:
                    conds["K3"] = [(range(n_layers), [d], [ctrl])]
                for Ly in range(n_layers):
                    conds[f"K4_{Ly}"] = [([Ly], [d], [kstar])]
                if d >= 1 and match:
                    conds["K5"] = [(range(n_layers), [d], match)]
                for c, knock in conds.items():
                    jobs.append({**base, "knock": [(list(ly), q, sl) for ly, q, sl in knock]})
                    jmeta.append({**mt, "d": d, "cond": c, "match": bool(tuple(codes_h[kstar, :d]) == tuple(tgt[:d])),
                                  "ctrl_slot": -1 if ctrl is None else ctrl, "n_match_items": len(match)})
        lp = score_fixed(model, enc, cache, jobs, n, K, n_layers, S, hmask)
        tg = torch.as_tensor(cohort.target_sids[[m["i"] for m in jmeta]], dtype=torch.long)
        gold = lp.gather(2, tg[:, :, None])[:, :, 0]                                   # (J, n)
        rank = (lp > gold[:, :, None]).sum(-1)
        top1 = lp.argmax(-1)
        for jx, mt in enumerate(jmeta):
            i, d = mt["i"], mt["d"]
            base = {"user": cohort.users[i], "user_row": i, "hist_len": int(cohort.history_lengths[i]), "d": d}
            digits = range(n) if d == 0 and mt["cond"] == "K0" else [d]     # S_full scores every digit
            for dd in digits:
                rec = {**base, "state": "full" if d == 0 else "pref", "digit": dd,
                       "logp": float(gold[jx, dd]), "rank": int(rank[jx, dd]), "top1": int(top1[jx, dd])}
                if mt["kind"] == "clean":
                    if mt["cond"] == "K0":
                        rec_clean.append(rec)
                    if dd == d:
                        rec_k.append({**rec, "cond": mt["cond"], "match": mt["match"],
                                      "ctrl_slot": mt["ctrl_slot"], "n_match_items": mt["n_match_items"]})
                elif mt["kind"] == "clean_dup":
                    rec_clean.append({**rec, "dup": True})
                else:
                    rec_a.append({**rec, "r": mt["r"], "k": mt["k"], "m": mt["m"],
                                  "control_item": mt["control_item"], "n_candidates": mt["n_candidates"]})
        log(f"{users[-1] + 1}/{N} users, {len(jobs)} decoder rows in chunk, {time.time() - t0:.0f}s")

    clean = pd.DataFrame(rec_clean)
    c1 = clean[clean.get("dup", pd.Series(False, index=clean.index)).fillna(False) == False]   # noqa: E712
    c2 = clean[clean.get("dup", pd.Series(False, index=clean.index)).fillna(False) == True]    # noqa: E712
    j = c1.merge(c2, on=["user", "state", "d", "digit"], suffixes=("", "_dup"))
    validation["V3_batch_invariance"] = {"compared": len(j), "max_abs_diff_logp": float((j.logp - j.logp_dup).abs().max()),
                                         "exact": bool((j.logp == j.logp_dup).all())}
    c1.drop(columns=[c for c in ("dup",) if c in c1]).to_parquet(args.out / "clean.parquet", index=False)
    pd.DataFrame(rec_a).to_parquet(args.out / "part_a.parquet", index=False)
    pd.DataFrame(rec_k).to_parquet(args.out / "part_k.parquet", index=False)
    pd.DataFrame(missing, columns=["user", "r", "m"]).to_csv(args.out / "part_a_no_control.csv", index=False)
    validation["part_a_no_control"] = len(missing)
    validation["clean_top1_full_by_digit"] = c1[c1.state == "full"].groupby("digit")["rank"].apply(
        lambda x: float((x == 0).mean())).to_dict()
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: V3 {validation['V3_batch_invariance']}; {validation['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
