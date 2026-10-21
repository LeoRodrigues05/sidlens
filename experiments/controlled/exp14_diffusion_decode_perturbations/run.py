#!/usr/bin/env python
"""controlled/exp14: perturbations held through DiffGRM decoding (copy block, order swap, position shift).

Protocol: `protocol.md` beside this file, declared before any forward pass of
this experiment. One invocation = one checkpoint. Every condition is decoded
with `matched_decode.decode` (confidence policy, batch 32, chunk 1,024, the
settings that reproduce controlled exp1's archive). Decoder-side blocks go
through a `score_fn` that adds a cross-attention mask; V2 shows that this path
with nothing blocked is the plain decode bit for bit.

Traps, each with the guard that closes it
-----------------------------------------
1. **A block on padding.** Blocking a padded slot changes the softmax without
   removing an item; `masked_scorer` refuses it (and `encoder_block_mask` does
   the same in the encoder).
2. **A shifted history that is not the same history.** `shifted` checks that
   the real slots hold the same items in the same order, and that the padding
   count is unchanged.
3. **Ineligible users counted as unaffected.** A user who cannot take a
   condition (L < 2, no control, L + s > 50) is decoded under the clean input
   for that condition and flagged; the summary uses eligible users only.

    python run.py --cell 6 --out <new dir> [--limit 256] [--device cpu]
    python run.py --ckpt 'grid:<key>@best' --out <new dir>

Grid runs (`sidlens.grid`): every condition is decoded with the run's own
decoder (`grid.decode_diff`, OneDiffRec's `joint_catalog_v1`: left to right,
catalog-legal codes at every step) at the run's beam and batch of 8 users;
the decoder-side block is a cross-attention mask entered around each step's
decoder forward. V2 is the same no-op check; V3 compares the clean decode with
the run's saved lists (a gate where the cohort is the evaluator's, a report
where users are sampled, since batches then differ; `diff_setup.check` has
already proven the decoder on the evaluator's first 64 users).
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
from sidlens.analysis.decode_checks import archive_check, item_hits, sid_rank  # noqa: E402
from sidlens.analysis.matched_decode import decode as matched_decode     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.data.sids import load_item2id                               # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

CELLS = tuple(f"diff-next1-{q}-{d}cb-{w}" for q in ("MQ", "rqkmeans", "rqvae")
              for d in (3, 4, 5) for w in (128, 512)) + ("diff-next1-MQ-5cb-256", "diff-next1-rqvae-3cb-256")
N_EXP1 = 18
BEAMS = (64, 256)
BATCH, CHUNK = 32, 1024
CONDS = ("clean", "block_r1", "block_ctrl", "swap", "shift1", "shift5")
N_LAYERS = 4
DEV = "cuda"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def masked_scorer(model, enc, hmask, slots):
    """`score_fn` blocking cross-attention from every digit to slot slots[b] (-1: none), in every block."""
    if bool(((slots >= 0) & ~hmask[torch.arange(len(slots), device=slots.device), slots.clamp_min(0)]).any()):
        raise ValueError("decoder block of a padded slot")
    cache = DI.project_cross_cache(model, enc)
    n, S = int(model.n_digit), enc.shape[1]

    def score_fn(step, partial, history_rows):
        R = partial.shape[0]
        xm = torch.ones((N_LAYERS, R, n, S), device=partial.device)
        sl = slots[history_rows]
        has = (sl >= 0).nonzero()[:, 0]
        if len(has):
            xm[:, has, :, sl[has]] = 0
        return DI.score_states(model, enc, cache, history_rows, partial.clamp_min(0), partial < 0, xmask=xm)
    return score_fn


def encode(model, hist, mask, blocked=None):
    """Encoder states of one batch; `blocked` [B] slot (-1 none) that every other real slot stops reading."""
    h, m = torch.from_numpy(hist).to(DEV), torch.from_numpy(mask).to(DEV)
    if blocked is None:
        return diffusion.encode(model, h, m)
    em = torch.from_numpy(DI.encoder_block_mask(mask, [(r, int(k)) for r, k in enumerate(blocked) if k >= 0])).to(DEV)
    with DI.encoder_attention_masks(model, em):
        return diffusion.encode(model, h, m)


def shifted(hist, mask, s):
    """Every real item moved s slots right, the first s slots padded (trap 2). None where L + s > S."""
    S = hist.shape[0]
    L = int(mask.sum())
    if L + s > S:
        return None
    h = np.full_like(hist, -1)
    m = np.zeros_like(mask)
    h[s:s + L], m[s:s + L] = hist[:L], True
    if not (np.array_equal(h[m], hist[mask]) and (~m).sum() == (~mask).sum()):
        raise RuntimeError("shift changed the items or the padding count")
    return h, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)))
    ap.add_argument("--ckpt", help="a next-item DiffGRM grid weight state (grid:<key>@best) instead of --cell")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--archive", type=Path, default=paths.DERIVED / "controlled" / "exp1_matched_beam" / "181944")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    global DEV
    DEV = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    t_start = time.time()

    grid_setup, beams = None, BEAMS
    if args.ckpt:                    # a grid run: its own cohort, table, model and decoder
        from sidlens.grid import decode_diff as DD, diff_setup as DS, predictions as GP
        grid_setup = DS.load(args.ckpt, device=DEV)
        ckpt_id, entry, cohort = args.ckpt, grid_setup.compat, grid_setup.cohort
        model, report = grid_setup.model, grid_setup.report
        proto = GP.diffgrm_protocol(grid_setup.entry, require_full=not args.limit)
        beams = (proto["beam_size"],)
    else:
        ckpt_id = CELLS[args.cell]
        entry = load_runtime()[ckpt_id]
        cohort = load_eval_cohort(entry)
        model, report = diffusion.load(entry, device=DEV)
    validation = {"V4_checkpoint_sha256": sha256_file(Path(entry["ckpt_path"])) == entry["ckpt_sha256"]}
    if not validation["V4_checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    n, K = int(model.n_digit), int(model.codebook_size)
    N = min(args.limit, len(cohort)) if args.limit else len(cohort)
    allowed = set(cohort.sid_buckets)
    if grid_setup is not None:
        validation["V1_grid_checks"] = DS.check(grid_setup, device=DEV)
        if not validation["V1_grid_checks"]["ok"]:
            raise RuntimeError(f"grid checks failed: {json.dumps(validation, default=str)[:1500]}")
    else:
        validation["V1_exp1_pilot_checks"] = exp1_validate()(
            model, torch.from_numpy(cohort.histories[:8]).to(DEV), torch.from_numpy(cohort.history_mask[:8]).to(DEV),
            cohort.sid_buckets, matched_decode)

    # ---- users: r1, control, item ids, eligibility --------------------------------------
    H, M = cohort.histories[:N], cohort.history_mask[:N]
    L = cohort.history_lengths[:N]
    if grid_setup is not None:
        r1_item = np.array([h[-1] for h in DS.history_item_ids(grid_setup)[:N]])
    else:
        item2id = load_item2id("Industrial_and_Scientific")
        seqs = json.loads((paths.FROZEN / "data" / "sequences" / "all_item_seqs.json").read_text())
        r1_item = np.array([item2id[seqs[u][:-1][-1]] for u in cohort.users[:N]])
    if not all(tuple(cohort.item_to_sid[int(r1_item[i])]) == tuple(H[i, L[i] - 1]) for i in range(N)):
        raise RuntimeError("rebuilt r1 items do not match the cohort's SIDs")
    r1_sid = H[np.arange(N), L - 1]
    ctrl = np.full(N, -1)
    for i in range(N):
        for j in range(L[i] - 2, -1, -1):
            if H[i, j, 0] != r1_sid[i, 0]:
                ctrl[i] = j
                break
    tgt_items, tgt_sids = cohort.target_item_ids[:N], cohort.target_sids[:N]
    relation = np.where(tgt_items == r1_item, "repeat",
                        np.where((tgt_sids == r1_sid).all(1), "same_sid",
                                 np.where(tgt_sids[:, 0] == r1_sid[:, 0], "same_d0", "new")))
    swap_h = H.copy()
    two = L >= 2
    swap_h[two, L[two] - 1], swap_h[two, L[two] - 2] = H[two, L[two] - 2], H[two, L[two] - 1]
    shift = {}
    for s in (1, 5):
        hs, ms, ok = H.copy(), M.copy(), np.zeros(N, dtype=bool)
        for i in range(N):
            out = shifted(H[i], M[i], s)
            if out is not None:
                hs[i], ms[i], ok[i] = out[0], out[1], True
        shift[s] = (hs, ms, ok)
    eligible = {"clean": np.ones(N, bool), "block_r1": two, "block_ctrl": two & (ctrl >= 0), "swap": two,
                "shift1": shift[1][2], "shift5": shift[5][2]}
    inputs = {"checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
              "cohort_sha256": cohort.cohort_sha256, "input_sha256": cohort.input_sha256, "n_users": N,
              "limited": args.limit, "n_digit": n, "codebook_size": K, "beams": list(beams),
              "batch": BATCH if grid_setup is None else proto["batch_size"],
              "decoder": "matched_decode (confidence)" if grid_setup is None else "grid joint_catalog_v1",
              "decoder_chunk": CHUNK, "conditions": list(CONDS), "eligible": {k: int(v.sum()) for k, v in eligible.items()},
              "device": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "torch": torch.__version__,
              **({k: entry[k] for k in ("grid_key", "grid_state", "dataset")} if grid_setup else {})}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    log(f"{ckpt_id}: users={N}; eligible {inputs['eligible']}")

    out = {(c, b): {"codes": np.full((N, 10, n), -1, np.int16), "scores": np.full((N, 10), -np.inf, np.float32),
                    "counts": np.zeros(N, np.int16)} for c in CONDS for b in beams}
    kl0 = {c: np.zeros(N, np.float32) for c in CONDS}
    opts = {"policy": "confidence", "top_k": 10, "allowed_sids": allowed, "decoder_chunk_size": CHUNK}

    def run_decode(enc, hmask, slots, beam):
        """(codes, scores, counts) of one batch under one condition."""
        if grid_setup is None:
            kw = {} if slots is None else {"score_fn": masked_scorer(model, enc, hmask, slots)}
            r = matched_decode(model, enc, beam_width=beam, **opts, **kw)
            return r.codes.cpu().numpy(), r.scores.cpu().numpy(), r.counts.cpu().numpy()
        if slots is not None and bool(((slots >= 0) & ~hmask[torch.arange(len(slots), device=slots.device),
                                                              slots.clamp_min(0)]).any()):
            raise ValueError("decoder block of a padded slot")
        ctx = None if slots is None else DS.slot_block_context(model, slots, enc.shape[1])
        hyps = DD.decode(model, None, grid_setup.catalog, beam_size=beam, top_k=min(10, proto["top_k"]),
                         seed=proto["seed"],
                         encoded=enc, decoder_context=ctx)
        return DS.as_arrays(hyps, n)

    step = BATCH if grid_setup is None else proto["batch_size"]
    t0 = time.time()
    for start in range(0, N, step):
        stop = min(start + step, N)
        sl = slice(start, stop)
        B = stop - start
        hm = torch.from_numpy(M[sl]).to(DEV)
        enc_clean = encode(model, H[sl], M[sl])
        no_slot = torch.full((B,), -1, device=DEV, dtype=torch.long)
        r1_slot = torch.as_tensor(np.where(two[sl], L[sl] - 1, -1), device=DEV)
        c_slot = torch.as_tensor(np.where(eligible["block_ctrl"][sl], ctrl[sl], -1), device=DEV)
        runs = {"clean": (enc_clean, None, hm),
                "block_r1": (encode(model, H[sl], M[sl], blocked=r1_slot.cpu().numpy()), r1_slot, hm),
                "block_ctrl": (encode(model, H[sl], M[sl], blocked=c_slot.cpu().numpy()), c_slot, hm),
                "swap": (encode(model, swap_h[sl], M[sl]), None, hm),
                "shift1": (encode(model, shift[1][0][sl], shift[1][1][sl]), None, torch.from_numpy(shift[1][1][sl]).to(DEV)),
                "shift5": (encode(model, shift[5][0][sl], shift[5][1][sl]), None, torch.from_numpy(shift[5][1][sl]).to(DEV))}
        if start == 0:                                                 # V2: the masked path with nothing blocked
            enc_noop = encode(model, H[sl], M[sl], blocked=no_slot.cpu().numpy())
            same = bool(torch.equal(enc_noop, enc_clean))
            for beam in beams:
                a = run_decode(enc_clean, hm, None, beam)
                b = run_decode(enc_noop, hm, no_slot, beam)
                same &= all(np.array_equal(x, y) for x, y in zip(a, b))
            validation["V2_masked_path_noop_is_plain_decode"] = same
            if not same:
                raise RuntimeError("the masked decode path with nothing blocked differs from the plain decode")
        for cond, (enc, slots, hmask) in runs.items():
            for beam in beams:
                o = out[(cond, beam)]
                o["codes"][sl], o["scores"][sl], o["counts"][sl] = run_decode(enc, hmask, slots, beam)
            rows = torch.arange(B, device=DEV)
            zeros = torch.zeros((B, n), dtype=torch.long, device=DEV)
            xm = None
            if slots is not None:
                xm = torch.ones((N_LAYERS, B, n, enc.shape[1]), device=DEV)
                has = (slots >= 0).nonzero()[:, 0]
                xm[:, has, :, slots[has]] = 0
            with torch.inference_mode():
                lp = DI.score_states(model, enc, DI.project_cross_cache(model, enc), rows, zeros,
                                     torch.ones_like(zeros, dtype=torch.bool), xmask=xm)
            if cond == "clean":
                lp_clean = lp
            kl0[cond][sl] = (lp_clean[:, 0].exp() * (lp_clean[:, 0] - lp[:, 0])).sum(-1).cpu().numpy()
        if start == 0 or stop == N or stop % 1024 == 0:
            log(f"{stop}/{N} users, {time.time() - t0:.0f}s")


    users = pd.DataFrame({"user_index": np.arange(N), "L": L, "relation": relation, "r1_d0": r1_sid[:, 0],
                          "r2_d0": np.where(two, H[np.arange(N), np.maximum(L - 2, 0), 0], -1),
                          "tgt_d0": tgt_sids[:, 0], "target_multiplicity": [len(cohort.sid_buckets[tuple(int(c) for c in s)]) for s in tgt_sids],
                          **{f"eligible_{c}": eligible[c] for c in CONDS}, **{f"kl0_{c}": kl0[c] for c in CONDS}})
    for cond in CONDS:
        for beam in beams:
            o = out[(cond, beam)]
            codes = o["codes"].astype(np.int64)
            users[f"{cond}_b{beam}_sid_rank"] = sid_rank(codes, tgt_sids)
            lo, hi = item_hits(codes, o["counts"], tgt_items, tgt_sids, cohort.sid_buckets)
            users[f"{cond}_b{beam}_item_hit_lo"], users[f"{cond}_b{beam}_item_hit_hi"] = lo, hi
            users[f"{cond}_b{beam}_top1_d0"] = codes[:, 0, 0]
            users[f"{cond}_b{beam}_top1_same_as_clean"] = (codes[:, 0] == out[("clean", beam)]["codes"][:, 0]).all(1)
    users.to_csv(args.out / "users.csv.gz", index=False)
    np.savez_compressed(args.out / "decodes.npz", **{f"{c}_b{b}_{k}": v for (c, b), o in out.items() for k, v in o.items()})
    validation["sid_hr10"] = {f"{c}_b{b}": float((users[f"{c}_b{b}_sid_rank"] > 0)[eligible[c]].mean()) for c in CONDS for b in beams}
    # V3 after the outputs are on disk: a failed gate leaves them for diagnosis, but no validation.json,
    # so the summary never reads the cell.
    if grid_setup is not None:
        b = beams[0]
        mine = {cohort.example_ids[i]: [{"sids": [DS._sid_text(out[("clean", b)]["codes"][i, j])],
                                         "score": float(out[("clean", b)]["scores"][i, j])}
                                        for j in range(int(out[("clean", b)]["counts"][i]))] for i in range(N)}
        v3 = GP.compare(mine, GP.load(grid_setup.entry), "diffgrm", score_atol=1e-4, tie_atol=1e-5,
                        only_saved=bool(args.limit))
        sampled = grid_setup.sampled
        validation["V3_grid_saved_lists"] = {**v3, "status": "ok" if v3["ok"] else ("report" if sampled else "failed"),
                                             "gate": not sampled}
        if not v3["ok"] and not sampled:
            (args.out / "validation.failed.json").write_text(json.dumps(validation, indent=1, default=str))
            raise RuntimeError("the clean decode does not reproduce the run's saved lists")
        validation["V3_exp1_archive"] = {"status": "grid run: see V3_grid_saved_lists"}
    elif args.cell < N_EXP1:
        decoded = {b: {k: out[("clean", b)][k].astype(np.int64 if k != "scores" else np.float32) for k in ("codes", "counts", "scores")}
                   for b in BEAMS}
        try:
            validation["V3_exp1_archive"] = archive_check(args.archive / f"cell-{args.cell:02d}", entry, cohort, N, decoded)
        except Exception as exc:
            validation["V3_exp1_archive"] = {"status": "failed", "error": str(exc)}
            (args.out / "validation.failed.json").write_text(json.dumps(validation, indent=1, default=str))
            raise
    else:
        validation["V3_exp1_archive"] = {"status": "not checked", "reason": "cell outside exp1's grid"}
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {validation['seconds']}s; archive {validation['V3_exp1_archive']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
