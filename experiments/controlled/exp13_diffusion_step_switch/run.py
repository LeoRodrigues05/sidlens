#!/usr/bin/env python
"""controlled/exp13: the most recent item replaced at single reveal steps of the DiffGRM decode.

Protocol: `protocol.md` beside this file, declared before any forward pass of
this experiment. One invocation = one checkpoint. Users are walked in the
cohort's order in batches of 32 (controlled exp1's batches), so the plain
decode can be checked against exp1's archive. A user without a donor keeps
their own history as "donor" and is flagged ineligible.

    python run.py --cell 6 --out <new dir> [--limit 256] [--device cpu]
    python run.py --ckpt 'grid:<key>@best' --out <new dir>

Grid runs (`sidlens.grid`): decoded with the run's own decoder
(`grid.decode_diff`: left to right, catalog-legal codes at every step) at
beam 1 and the run's beam, in the evaluator's batches of 8. Reveal step t is
digit t there, so a schedule reads the donor's encoder states at the digits it
names (`step_encoded`); V2 (none = plain) and V3 (all = donor) hold bit for bit
by construction and are checked anyway. V4 compares the plain decode with the
run's saved lists (a gate unless users are sampled).
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
from sidlens.analysis import diffusion_lens as DL                        # noqa: E402
from sidlens.analysis.decode_checks import archive_check, sid_rank       # noqa: E402
from sidlens.analysis.matched_decode import decode as matched_decode     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.data.sids import SidTable, load_item2id                     # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.interventions.history import ControlPool, condition_rng     # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

CELLS = tuple(f"diff-next1-{q}-{d}cb-{w}" for q in ("MQ", "rqkmeans", "rqvae")
              for d in (3, 4, 5) for w in (128, 512)) + ("diff-next1-MQ-5cb-256", "diff-next1-rqvae-3cb-256")
N_EXP1 = 18
BEAMS = (1, 64)
BATCH, CHUNK = 32, 1024
SEED_EXP7 = 20260928
DEV = "cuda"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def schedules(n: int) -> dict[str, np.ndarray]:
    out = {"none": np.zeros(n, bool), "all": np.ones(n, bool)}
    for t in range(n):
        out[f"only_{t}"] = np.arange(n) == t
        out[f"but_{t}"] = np.arange(n) != t
    return out


def same(a, b) -> bool:
    return bool(torch.equal(a.codes, b.codes) and torch.equal(a.scores, b.scores) and torch.equal(a.counts, b.counts))


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

    grid_setup, beams, batch = None, BEAMS, BATCH
    if args.ckpt:                    # a grid run: its own cohort, table, model and decoder
        from sidlens.grid import decode_diff as DD, diff_setup as DS, predictions as GP
        grid_setup = DS.load(args.ckpt, device=DEV)
        ckpt_id, entry, cohort = args.ckpt, grid_setup.compat, grid_setup.cohort
        model, report, table, item2id = grid_setup.model, grid_setup.report, grid_setup.table, grid_setup.item2id
        proto = GP.diffgrm_protocol(grid_setup.entry, require_full=not args.limit)
        beams, batch = (1, proto["beam_size"]), proto["batch_size"]
    else:
        ckpt_id = CELLS[args.cell]
        entry = load_runtime()[ckpt_id]
        cohort = load_eval_cohort(entry)
        model, report = diffusion.load(entry, device=DEV)
        table, item2id = SidTable.load(entry["sem_ids_name"]), load_item2id("Industrial_and_Scientific")
    validation = {"V5_checkpoint_sha256": sha256_file(Path(entry["ckpt_path"])) == entry["ckpt_sha256"]}
    if not validation["V5_checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    n, K = int(model.n_digit), int(model.codebook_size)
    N = min(args.limit, len(cohort)) if args.limit else len(cohort)
    allowed = set(cohort.sid_buckets)
    pool = ControlPool(table, item2id)
    if grid_setup is not None:
        validation["V1_grid_checks"] = DS.check(grid_setup, device=DEV)
        if not validation["V1_grid_checks"]["ok"]:
            raise RuntimeError(f"grid checks failed: {json.dumps(validation, default=str)[:1500]}")
    else:
        validation["V1_exp1_pilot_checks"] = exp1_validate()(
            model, torch.from_numpy(cohort.histories[:8]).to(DEV), torch.from_numpy(cohort.history_mask[:8]).to(DEV),
            cohort.sid_buckets, matched_decode)

    H, M, L = cohort.histories[:N], cohort.history_mask[:N], cohort.history_lengths[:N]
    donor_h, eligible, ctrl_item = H.copy(), np.zeros(N, bool), np.full(N, -1)
    for i in range(N):
        k = int(L[i]) - 1
        codes_h = [tuple(int(c) for c in H[i, j]) for j in range(int(L[i]))]
        tgt = tuple(int(c) for c in cohort.target_sids[i])
        idx = pool.candidates(codes_h[k], 0, {int(cohort.target_item_ids[i])}, set(codes_h) | {tgt})
        if len(idx):
            j = idx[condition_rng(SEED_EXP7, cohort.users[i], k, 0).integers(len(idx))]
            donor_h[i, k] = pool.codes[j]
            eligible[i], ctrl_item[i] = True, int(pool.item_ids[j])
    sched = schedules(n)
    inputs = {"checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
              "cohort_sha256": cohort.cohort_sha256, "input_sha256": cohort.input_sha256, "n_users": N,
              "limited": args.limit, "n_digit": n, "codebook_size": K, "beams": list(beams), "batch": batch,
              "decoder": "matched_decode (confidence)" if grid_setup is None else "grid joint_catalog_v1",
              "decoder_chunk": CHUNK, "schedules": list(sched), "eligible": int(eligible.sum()),
              "device": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "torch": torch.__version__,
              **({k: entry[k] for k in ("grid_key", "grid_state", "dataset")} if grid_setup else {})}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    log(f"{ckpt_id}: users={N}, eligible {int(eligible.sum())}, {len(sched)} schedules")

    out = {(s, b): {"codes": np.full((N, 10, n), -1, np.int16), "scores": np.full((N, 10), -np.inf, np.float32),
                    "counts": np.zeros(N, np.int16)} for s in sched for b in beams}
    greedy = {s: {"order": np.full((N, n), -1, np.int8), "codes": np.full((N, n), -1, np.int16)} for s in sched}
    plain = {b: {"codes": np.zeros((N, 10, n), np.int64), "scores": np.zeros((N, 10), np.float32),
                 "counts": np.zeros(N, np.int64)} for b in beams}
    opts = {"policy": "confidence", "top_k": 10, "allowed_sids": allowed, "decoder_chunk_size": CHUNK}
    v2 = v3 = True
    t0 = time.time()
    def grid_run(enc, beam, mask=None, enc_d=None):
        """(codes, scores, counts) of the grid decoder; `mask` [n] reads enc_d at those digits."""
        step = None if mask is None else (lambda col: enc_d if mask[col] else None)
        hyps = DD.decode(model, None, grid_setup.catalog, beam_size=beam, top_k=min(10, proto["top_k"], beam),
                         seed=proto["seed"], encoded=enc, step_encoded=step)
        return DS.as_arrays(hyps, n)

    for start in range(0, N, batch):
        stop = min(start + batch, N)
        sl = slice(start, stop)
        enc = diffusion.encode(model, torch.from_numpy(H[sl]).to(DEV), torch.from_numpy(M[sl]).to(DEV))
        enc_d = diffusion.encode(model, torch.from_numpy(donor_h[sl]).to(DEV), torch.from_numpy(M[sl]).to(DEV))
        if grid_setup is not None:
            for beam in beams:
                base, donor = grid_run(enc, beam), grid_run(enc_d, beam)
                for k, v in zip(("codes", "scores", "counts"), base):
                    plain[beam][k][sl] = v
                for name, mask in sched.items():
                    res = grid_run(enc, beam, mask, enc_d)
                    if name == "none":
                        v2 &= all(np.array_equal(a, b) for a, b in zip(res, base))
                    if name == "all":
                        v3 &= all(np.array_equal(a, b) for a, b in zip(res, donor))
                    o = out[(name, beam)]
                    o["codes"][sl], o["scores"][sl], o["counts"][sl] = res
                    if beam == 1:                      # left to right: digit t is revealed at step t
                        greedy[name]["order"][sl] = np.arange(n)
                        greedy[name]["codes"][sl] = res[0][:, 0]
            if not (v2 and v3):
                raise RuntimeError(f"switch controls failed at users {start}-{stop}: none==plain {v2}, all==donor {v3}")
            continue
        for beam in BEAMS:
            base = matched_decode(model, enc, beam_width=beam, **opts)
            donor = matched_decode(model, enc_d, beam_width=beam, **opts)
            for k in ("codes", "scores", "counts"):
                plain[beam][k][sl] = getattr(base, k).cpu().numpy()
            for name, mask in sched.items():
                sm = torch.as_tensor(np.broadcast_to(mask, (stop - start, n)).copy(), device=DEV)
                trace = DL.BeamTrace(K) if beam == 1 else None
                res = matched_decode(model, enc, beam_width=beam, observer=trace,
                                     score_fn=DI.history_switch_scorer(model, enc, enc_d, sm), **opts)
                if name == "none":
                    v2 &= same(res, base)
                if name == "all":
                    v3 &= same(res, donor)
                o = out[(name, beam)]
                o["codes"][sl], o["scores"][sl], o["counts"][sl] = (res.codes.cpu().numpy(), res.scores.cpu().numpy(),
                                                                    res.counts.cpu().numpy())
                if beam == 1:
                    final = trace.final_states[:, 0]                   # greedy SID, legal or not
                    path = trace.path_to(final)
                    if not bool(path["found"].all()):
                        raise RuntimeError("greedy path not found in its own trace")
                    greedy[name]["order"][sl], greedy[name]["codes"][sl] = path["order"].numpy(), path["codes"].numpy()
        if not (v2 and v3):
            raise RuntimeError(f"switch controls failed at users {start}-{stop}: none==plain {v2}, all==donor {v3}")
        if start == 0 or stop == N or stop % 1024 == 0:
            log(f"{stop}/{N} users, {time.time() - t0:.0f}s")
    validation["V2_none_is_plain_decode"], validation["V3_all_is_donor_decode"] = v2, v3

    tgt = cohort.target_sids[:N]
    users = pd.DataFrame({"user_index": np.arange(N), "L": L, "eligible": eligible, "control_item": ctrl_item,
                          "r1_d0": H[np.arange(N), L - 1, 0], "tgt_d0": tgt[:, 0]})
    wide = max(beams)
    for name in sched:
        users[f"b{wide}_{name}_sid_rank"] = sid_rank(out[(name, wide)]["codes"].astype(np.int64), tgt)
        users[f"b1_{name}_greedy_hit"] = (greedy[name]["codes"].astype(np.int64)[np.arange(N)[:, None], np.argsort(greedy[name]["order"], 1)] == tgt).all(1)
    users.to_csv(args.out / "users.csv.gz", index=False)
    np.savez_compressed(args.out / "decodes.npz",
                        **{f"b{b}_{s}_{k}": v for (s, b), o in out.items() for k, v in o.items()},
                        **{f"greedy_{s}_{k}": v for s, g in greedy.items() for k, v in g.items()})
    # V4 after the outputs are on disk: a failed gate leaves them for diagnosis, but no validation.json.
    if grid_setup is not None:
        mine = {cohort.example_ids[i]: [{"sids": [DS._sid_text(plain[wide]["codes"][i, j])],
                                         "score": float(plain[wide]["scores"][i, j])}
                                        for j in range(int(plain[wide]["counts"][i]))] for i in range(N)}
        v4 = GP.compare(mine, GP.load(grid_setup.entry), "diffgrm", score_atol=1e-4, tie_atol=1e-5,
                        only_saved=bool(args.limit))
        validation["V4_grid_saved_lists"] = {**v4, "gate": not grid_setup.sampled}
        validation["V4_exp1_archive"] = {"status": "grid run: see V4_grid_saved_lists"}
        if not v4["ok"] and not grid_setup.sampled:
            (args.out / "validation.failed.json").write_text(json.dumps(validation, indent=1, default=str))
            raise RuntimeError("the plain decode does not reproduce the run's saved lists")
    elif args.cell < N_EXP1:
        try:
            validation["V4_exp1_archive"] = archive_check(args.archive / f"cell-{args.cell:02d}", entry, cohort, N,
                                                          {64: plain[64]})
        except Exception as exc:
            validation["V4_exp1_archive"] = {"status": "failed", "error": str(exc)}
            (args.out / "validation.failed.json").write_text(json.dumps(validation, indent=1, default=str))
            raise
    else:
        validation["V4_exp1_archive"] = {"status": "not checked", "reason": "cell outside exp1's grid"}
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {validation['seconds']}s; archive {validation['V4_exp1_archive']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
