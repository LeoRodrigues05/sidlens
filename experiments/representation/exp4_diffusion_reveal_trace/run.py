#!/usr/bin/env python
"""DiffGRM reveal trace: what the decode holds at every step (observational).

Protocol: `protocol.md` beside this file, declared before any forward pass of
this experiment. One invocation = one checkpoint. The recording itself is
`common.trace_batch`; this file loads, gates and writes. Batch 32 and decoder
chunk 1024 are the settings the exp1 archive was made with and are not
arguments: other settings have failed exact reconstruction on full cohorts.

    python run.py --cell 7 --out <new dir> [--limit 64] [--device cpu]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common                                                            # noqa: E402
from sidlens import paths                                                # noqa: E402
from sidlens.analysis import diffusion_lens as DL                        # noqa: E402
from sidlens.analysis.matched_decode import decode as matched_decode     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

LIMITS = {"v3_path_score_max_abs_diff": 1e-4, "v4_last_block_max_abs_diff": 1e-5,
          "v5_target_score_max_excess_over_lattice": 1e-4}


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def check_archive(archive: Path, entry: dict, cohort, data: dict, n: int) -> dict:
    """V6: beam-64 and beam-256 confidence lists against controlled exp1's archive."""
    previous = json.loads((archive / "result.json").read_text())
    if previous["checkpoint_sha256"] != entry["ckpt_sha256"]:
        raise ValueError("archive checkpoint differs")
    if previous["cohort_sha256"] != cohort.cohort_sha256 or \
            sorted(previous["input_sha256"].values()) != sorted(cohort.input_sha256.values()):
        raise ValueError("archive cohort/input hashes differ")
    checks = []
    with np.load(archive / "predictions.npz", allow_pickle=False) as old:
        for name, mine in (("users", np.asarray(cohort.users[:n])), ("user_ids", cohort.user_ids[:n]),
                           ("target_item_ids", cohort.target_item_ids[:n]), ("targets", cohort.target_sids[:n])):
            if not np.array_equal(mine, old[name][:n]):
                raise ValueError(f"archive {name} identity differs")
        axes = list(zip(old["condition_beams"].tolist(), old["condition_policies"].tolist()))
        for beam in (64, 256):
            j = axes.index((beam, "confidence"))
            for name in ("codes", "counts"):
                if not np.array_equal(data[f"beam{beam}_{name}"], old[name][j, :n]):
                    raise ValueError(f"archive beam {beam} {name} differ")
            actual, expected = data[f"beam{beam}_scores"], old["scores"][j, :n]
            finite = np.isfinite(expected)
            if not np.array_equal(np.isfinite(actual), finite) or not np.allclose(
                    actual[finite], expected[finite], atol=1e-4, rtol=1e-4):
                raise ValueError(f"archive beam {beam} finite scores differ")
            checks.append({"beam": beam, "identical_codes_counts": True, "score_max_absolute_difference":
                           float(np.max(np.abs(actual[finite] - expected[finite]), initial=0))})
    return {"status": "passed", "users": n, "conditions": checks, "archive": str(archive), "archive_sha256": {
        name: sha256_file(archive / name) for name in ("result.json", "predictions.npz")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(common.CELLS)))
    ap.add_argument("--ckpt", help="a next-item DiffGRM grid weight state (grid:<key>@<state>) instead of --cell")
    ap.add_argument("--limit", type=int, default=None, help="first N users (pilot)")
    ap.add_argument("--device", default="cuda", help="cpu only for smoke tests")
    ap.add_argument("--archive", type=Path, default=paths.DERIVED / "controlled" / "exp1_matched_beam" / "181944")
    ap.add_argument("--allow-missing-archive", action="store_true",
                    help="run without V6 when the exp1 archive is not on this cluster; results are provisional")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    dev = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json", "trace.npz")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t_start = time.time()

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    grid_setup = None
    if args.ckpt:                    # a grid run: its own cohort, table and model (sidlens.grid.diff_setup)
        from sidlens.grid import diff_setup as DS
        grid_setup = DS.load(args.ckpt, device=dev)
        ckpt_id, entry, cohort = args.ckpt, grid_setup.compat, grid_setup.cohort
        model, report = grid_setup.model, grid_setup.report
        archive = None
    else:
        ckpt_id = common.CELLS[args.cell]
        entry = load_runtime()[ckpt_id]
        archive = args.archive / f"cell-{args.cell:02d}" if args.cell < len(common.EXP1_CELLS) else None
        if archive is not None and not (archive / "predictions.npz").exists():
            if not args.allow_missing_archive:
                raise FileNotFoundError(f"V6 needs {archive}; pass --allow-missing-archive to run provisionally")
            archive = None
        cohort = load_eval_cohort(entry)
        model, report = diffusion.load(entry, device=dev)
    if entry["status"] != "trained" or entry["task"] != "next1":
        raise ValueError(f"not a trained next-item checkpoint: {ckpt_id}")
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    n, K = int(model.n_digit), int(model.codebook_size)
    N = min(args.limit, len(cohort)) if args.limit else len(cohort)
    catalogue = DL.Catalogue.from_buckets(cohort.sid_buckets, K, device=dev)
    allowed = set(cohort.sid_buckets)

    def batch(start):
        stop = min(start + common.BATCH, N)
        return diffusion.encode(model, torch.from_numpy(cohort.histories[start:stop]).to(dev),
                                torch.from_numpy(cohort.history_mask[start:stop]).to(dev))

    # ---- validation before any recording ----------------------------------------
    if grid_setup is not None:       # no exp1 archive: the run's own lists (V2a) and the scorer check
        validation = {"model_report_ok": report["ok"], "V1_grid_checks": DS.check(grid_setup, device=dev)}
        if not validation["V1_grid_checks"]["ok"]:
            raise RuntimeError(f"grid checks failed: {json.dumps(validation, default=str)[:1500]}")
    else:
        validation = {"model_report_ok": report["ok"], "V1_exp1_pilot_checks": exp1_validate()(
            model, torch.from_numpy(cohort.histories[:8]).to(dev), torch.from_numpy(cohort.history_mask[:8]).to(dev),
            cohort.sid_buckets, matched_decode)}
    enc0 = batch(0)
    options = {"beam_width": 64, "policy": "confidence", "top_k": 10, "allowed_sids": allowed,
               "decoder_chunk_size": common.CHUNK}
    plain = matched_decode(model, enc0, **options)
    watched = matched_decode(model, enc0, observer=DL.BeamTrace(K, catalogue), **options)
    validation["V2_observer_is_noop"] = bool(torch.equal(plain.codes, watched.codes)
                                             and torch.equal(plain.scores, watched.scores))
    if not validation["V2_observer_is_noop"]:
        raise RuntimeError("the observer changed the decode")
    inputs = {"checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
              "cohort_sha256": cohort.cohort_sha256, "input_sha256": cohort.input_sha256, "n_users": N,
              "limited": args.limit, "n_digit": n, "codebook_size": K, "catalogue_sids": len(allowed),
              "catalogue_items": catalogue.n_items, "beams": list(common.BEAMS), "primary_beam": common.PRIMARY_BEAM,
              "batch": common.BATCH, "decoder_chunk": common.CHUNK, "top_codes": common.TOP_CODES,
              "training": entry.get("training"), "archive": None if archive is None else str(archive),
              "device": torch.cuda.get_device_name() if dev == "cuda" else dev, "torch": torch.__version__,
              **({k: entry[k] for k in ("grid_key", "grid_state", "dataset")} if grid_setup else {})}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    log(f"inputs ok: {ckpt_id} users={N} n={n} K={K}; V1 passed; V2 {validation['V2_observer_is_noop']}")

    parts, checks = [], {}
    t0 = time.time()
    for start in range(0, N, common.BATCH):
        stop = min(start + common.BATCH, N)
        out, check = common.trace_batch(model, enc0 if start == 0 else batch(start),
                                        torch.from_numpy(cohort.target_sids[start:stop]).to(dev), catalogue, allowed)
        parts.append(out)
        for name, value in check.items():
            checks[name] = max(checks.get(name, float("-inf")), value)
        if start == 0 or stop == N or stop % 512 == 0:
            log(f"{stop}/{N} users, {time.time() - t0:.0f}s")
    data = {name: torch.cat([part[name] for part in parts]).numpy() for name in parts[0]}

    validation["V3_V4_V5"] = {name: {"value": checks[name], "limit": LIMITS[name],
                                     "ok": bool(checks[name] < LIMITS[name])} for name in LIMITS}
    if not all(v["ok"] for v in validation["V3_V4_V5"].values()):
        (args.out / "validation.failed.json").write_text(json.dumps(validation, indent=1, default=str))
        raise RuntimeError(f"trace does not reproduce the decode: {validation['V3_V4_V5']}")
    validation["V6_exp1_archive"] = (check_archive(archive, entry, cohort, data, N) if archive is not None else
                                     {"status": "not checked", "reason": "grid run (see V1_grid_checks)"
                                      if grid_setup is not None else "cell outside exp1's grid"
                                      if args.cell >= len(common.EXP1_CELLS) else "archive not on this cluster"})
    np.savez_compressed(
        args.out / "trace.npz", users=np.asarray(cohort.users[:N]), user_ids=cohort.user_ids[:N],
        targets=cohort.target_sids[:N], target_item_ids=cohort.target_item_ids[:N],
        history_lengths=cohort.history_lengths[:N],
        target_multiplicity=np.asarray([len(cohort.sid_buckets[tuple(int(v) for v in row)])
                                        for row in cohort.target_sids[:N]]), **data)
    validation["sid_hit@10"] = {f"beam{beam}": float((data[f"beam{beam}_target_rank"] > 0).mean())
                                for beam in common.BEAMS}
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: V3-V5 {checks}; V6 {validation['V6_exp1_archive']['status']}; {validation['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
