#!/usr/bin/env python
"""Seven declared reveal conditions on one frozen depth-three checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.controlled.exp1_matched_beam.run import (
    METRICS, cell_ids, configure_precision, evaluate_prediction,
)
from experiments.controlled.exp1_matched_beam.validate import verify_model_and_decoder
from experiments.controlled.exp2_fixed_orders.common import CELL_INDICES, conditions, label
from sidlens import paths
from sidlens.analysis.matched_decode import decode
from sidlens.data.diffusion_eval import load_eval_cohort
from sidlens.models import diffusion
from sidlens.provenance.hashing import sha256_file
from sidlens.registry.diffusion import load_runtime


def check_archive(out, entry, cohort, axes, data, archive):
    previous = json.loads((archive / "result.json").read_text())
    if previous["checkpoint_sha256"] != entry["ckpt_sha256"]:
        raise ValueError("baseline checkpoint differs")
    if previous["cohort_sha256"] != cohort.cohort_sha256 or previous["input_sha256"] != cohort.input_sha256:
        raise ValueError("baseline cohort/input hashes differ")
    n = data["codes"].shape[1]
    seed_order = tuple(previous["fixed_order"])
    checks = []
    with np.load(archive / "predictions.npz", allow_pickle=False) as old:
        for name in ("users", "user_ids", "target_item_ids", "targets"):
            if not np.array_equal(data[name], old[name][:n]):
                raise ValueError(f"baseline {name} identity differs")
        old_axes = list(zip(old["condition_beams"].tolist(), old["condition_policies"].tolist()))
        for i, (beam, policy, order) in enumerate(axes):
            if policy == "fixed" and order != seed_order:
                continue
            j = old_axes.index((beam, policy))
            for name in ("codes", "counts", "ranks", "metrics"):
                if not np.array_equal(data[name][i], old[name][j, :n]):
                    raise ValueError(f"baseline {beam}/{label(policy, order)} {name} differs")
            actual, expected = data["scores"][i], old["scores"][j, :n]
            finite = np.isfinite(expected)
            if not np.array_equal(np.isfinite(actual), finite) or not np.allclose(
                    actual[finite], expected[finite], atol=1e-4, rtol=1e-4):
                raise ValueError("baseline finite scores differ")
            checks.append({"beam": beam, "policy": policy, "order": order,
                           "identical_predictions_ranks_metrics": True,
                           "score_max_absolute_difference": float(np.max(np.abs(actual[finite] - expected[finite]), initial=0))})
    return {"status": "passed", "users": n, "conditions": checks,
            "baseline": str(archive), "baseline_sha256": {
                name: sha256_file(archive / name) for name in ("result.json", "predictions.npz")}}


@torch.inference_mode()
def validate_orders(model, hidden, axes, buckets):
    checks = []
    hidden = hidden[:2].to("cuda")
    for beam, policy, order in axes:
        kwargs = dict(beam_width=beam, policy=policy, order=order, allowed_sids=buckets)
        batch = decode(model, hidden, decoder_chunk_size=256, **kwargs)
        singles = [decode(model, h[None], decoder_chunk_size=17, **kwargs) for h in hidden]
        if not torch.equal(batch.codes, torch.cat([x.codes for x in singles])):
            raise ValueError(f"pilot batch/chunk rankings differ: {beam}/{label(policy, order)}")
        scores = torch.cat([x.scores for x in singles])
        finite = torch.isfinite(batch.scores)
        if not torch.equal(finite, torch.isfinite(scores)) or not torch.allclose(
                batch.scores[finite], scores[finite], atol=1e-4, rtol=1e-4):
            raise ValueError("pilot batch/chunk scores differ")
        checks.append({"beam": beam, "policy": policy, "order": order, "identical_rankings": True})
    return {"status": "passed", "users": len(hidden), "checks": checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell-index", type=int, choices=CELL_INDICES, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beam-widths", nargs="+", type=int, default=[64])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--decoder-chunk-size", type=int, default=1024)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--archive", type=Path, default=paths.DERIVED / "controlled/exp1_matched_beam/181944")
    args = ap.parse_args()
    if args.limit < 0 or args.batch_size < 1 or args.decoder_chunk_size < 1:
        raise ValueError("invalid limits")
    if (args.out / "result.json").exists() or (args.out / "predictions.npz").exists():
        raise FileExistsError(f"existing results: {args.out}")
    if not torch.cuda.is_available():
        raise RuntimeError("run inside a GPU sbatch allocation")
    axes = conditions(args.beam_widths)
    configure_precision()
    started = time.perf_counter()
    entry = load_runtime()[cell_ids()[args.cell_index]]
    if entry["status"] != "trained" or entry["task"] != "next1" or entry["config"]["n_digit"] != 3:
        raise ValueError("unexpected checkpoint status/depth")
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise ValueError("checkpoint hash drift")
    cohort = load_eval_cohort(entry)
    n = min(args.limit, len(cohort)) if args.limit else len(cohort)
    if len(cohort) != 6297:
        raise ValueError("declared evaluation cohort changed")
    model, model_report = diffusion.load(entry, device="cuda")
    print(f"[load] {cell_ids()[args.cell_index]} users={n} conditions={len(axes)}", flush=True)
    validation = {}
    if args.validate:
        count = min(8, n)
        validation["model"] = verify_model_and_decoder(
            model, torch.from_numpy(cohort.histories[:count]).to("cuda"),
            torch.from_numpy(cohort.history_mask[:count]).to("cuda"), cohort.sid_buckets, decode)
    with torch.inference_mode():
        hidden = torch.cat([
            diffusion.encode(model, torch.from_numpy(cohort.histories[i:i + args.batch_size]).to("cuda"),
                             torch.from_numpy(cohort.history_mask[i:i + args.batch_size]).to("cuda")).cpu()
            for i in range(0, n, args.batch_size)
        ])[:n]
    if args.validate:
        validation["order_invariance"] = validate_orders(model, hidden, axes, cohort.sid_buckets)
        print("[validation] model and all order batch/chunk checks passed", flush=True)
    shape = (len(axes), n)
    data = {
        "users": np.asarray(cohort.users[:n]), "user_ids": cohort.user_ids[:n],
        "targets": cohort.target_sids[:n], "target_item_ids": cohort.target_item_ids[:n],
        "target_multiplicity": np.asarray([len(cohort.sid_buckets[tuple(map(int, row))]) for row in cohort.target_sids[:n]]),
        "condition_beams": np.asarray([a[0] for a in axes]),
        "condition_policies": np.asarray([a[1] for a in axes]),
        "condition_orders": np.asarray([a[2] if a[2] is not None else (-1, -1, -1) for a in axes]),
        "codes": np.full((*shape, 10, 3), -1, dtype=np.int64),
        "scores": np.full((*shape, 10), -np.inf, dtype=np.float32),
        "counts": np.zeros(shape, dtype=np.int64), "ranks": np.zeros(shape, dtype=np.int64),
        "metric_names": np.asarray(METRICS), "metrics": np.zeros((*shape, len(METRICS))),
    }
    path_fields = ("generated_counts", "unique_counts", "invalid_counts", "legal_unique_counts")
    for field in path_fields:
        data[field] = np.zeros(shape, dtype=np.int64)
    diagnostics = []
    for ci, (beam, policy, order) in enumerate(axes):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        decoder_rows = 0
        for offset in range(0, n, args.batch_size):
            end = min(offset + args.batch_size, n)
            decoded = decode(model, hidden[offset:end].to("cuda"), beam_width=beam,
                             policy=policy, order=order, top_k=10, allowed_sids=cohort.sid_buckets,
                             decoder_chunk_size=args.decoder_chunk_size)
            data["codes"][ci, offset:end] = decoded.codes.cpu().numpy()
            data["scores"][ci, offset:end] = decoded.scores.cpu().numpy()
            data["counts"][ci, offset:end] = decoded.counts.cpu().numpy()
            for field in path_fields:
                data[field][ci, offset:end] = getattr(decoded, field).cpu().numpy()
            decoder_rows += (end - offset) * (1 + sum(decoded.beam_sizes[:-1]))
            for row in range(offset, end):
                rank, metrics = evaluate_prediction(data["codes"][ci, row, :data["counts"][ci, row]],
                                                    cohort.target_sids[row], cohort.target_item_ids[row], cohort.sid_buckets)
                data["ranks"][ci, row] = rank or 0
                data["metrics"][ci, row] = [metrics[name] for name in METRICS]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        diagnostics.append({"beam_width": beam, "policy": policy, "order": order,
            "label": label(policy, order), "seconds": elapsed, "users_per_second": n / elapsed,
            "decoder_rows": decoder_rows, "active_beam_sizes": list(decoded.beam_sizes),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "rows_with_ten_predictions": int((data["counts"][ci] == 10).sum()),
            "mean_list_length": float(data["counts"][ci].mean()),
            **{field: int(data[field][ci].sum()) for field in path_fields},
            "metrics": {name: float(data["metrics"][ci, :, mi].mean()) for mi, name in enumerate(METRICS)}})
        print(f"[condition] beam={beam} {label(policy, order)} users={n} seconds={elapsed:.2f}", flush=True)
    validation["archive"] = check_archive(args.out, entry, cohort, axes, data, args.archive / f"cell-{args.cell_index:02d}")
    validation["status"] = "passed"
    result = {
        "experiment": "controlled-exp2-fixed-orders", "checkpoint": cell_ids()[args.cell_index],
        "cell_index": args.cell_index, "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "quantizer": entry["quantizer"], "depth": 3, "codebook_size": entry["codebook_size"],
        "n_users": n, "full_cohort": n == len(cohort), "cohort_sha256": cohort.cohort_sha256,
        "checkpoint_sha256": entry["ckpt_sha256"], "config": entry["config"],
        "input_sha256": cohort.input_sha256, "sid_lineage": cohort.sid_lineage,
        "model_shape_validation": model_report, "conditions": diagnostics,
        "precision": "float32; TF32 disabled; no autocast; eval mode", "gpu": torch.cuda.get_device_name(0),
        "total_seconds": time.perf_counter() - started, "validation": validation,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "predictions.npz", **data)
    with (args.out / "per_user.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "user", "target_item", "beam", "policy", "order", "rank", "count", *path_fields, *METRICS])
        for ci, (beam, policy, order) in enumerate(axes):
            for row in range(n):
                writer.writerow([result["checkpoint"], cohort.users[row], int(cohort.target_item_ids[row]),
                    beam, policy, "" if order is None else "".join(map(str, order)),
                    data["ranks"][ci, row], data["counts"][ci, row],
                    *[data[field][ci, row] for field in path_fields], *data["metrics"][ci, row]])
    inputs = {key: result[key] for key in ("snapshot", "checkpoint", "checkpoint_sha256", "cohort_sha256", "input_sha256", "sid_lineage")}
    for name, value in (("result.json", result), ("inputs.json", inputs), ("validation.json", validation),
                        ("arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})):
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(f"[done] {result['checkpoint']} output={args.out}", flush=True)


if __name__ == "__main__":
    main()
