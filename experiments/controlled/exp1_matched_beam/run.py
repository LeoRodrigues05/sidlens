#!/usr/bin/env python
"""New matched-beam inference on one frozen next-item DiffGRM checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sidlens import paths
from sidlens.analysis.collisions import exact_rank, item_rank_bounds
from sidlens.analysis.matched_decode import decode
from sidlens.data.diffusion_eval import load_eval_cohort
from sidlens.models import diffusion
from sidlens.provenance.hashing import sha256_file
from sidlens.registry.diffusion import load_runtime


POLICIES = ("confidence", "fixed", "legacy_confidence")
CUTOFFS = (1, 3, 5, 10)
METRICS = tuple(
    f"{metric}@{cutoff}"
    for metric in ("sid_hit", "ndcg", "item_lower", "item_uniform", "item_upper")
    for cutoff in CUTOFFS
)


def cell_ids() -> list[str]:
    return [
        f"diff-next1-{quantizer}-{depth}cb-{width}"
        for quantizer in ("MQ", "rqkmeans", "rqvae")
        for depth in (3, 4, 5)
        for width in (128, 512)
    ]


def evaluate_prediction(codes, target_sid, target_item, buckets):
    """Exact-SID retrieval plus ranked item-expansion sensitivities."""
    sids = [tuple(int(code) for code in row) for row in codes]
    target = tuple(int(code) for code in target_sid)
    if len(sids) != len(set(sids)) or any(sid not in buckets for sid in sids):
        raise ValueError("decoder returned duplicate or illegal visible SID")
    rank = exact_rank(sids, target)
    bounds = item_rank_bounds(sids, int(target_item), buckets, target_sid=target)
    metrics = {}
    for cutoff in CUTOFFS:
        found = rank is not None and rank <= cutoff
        metrics[f"sid_hit@{cutoff}"] = float(found)
        metrics[f"ndcg@{cutoff}"] = 1 / math.log2(rank + 1) if found else 0.0
        for key, value in bounds.hit_bounds(cutoff).items():
            if key != "catalogue":
                metrics[f"item_{key}@{cutoff}"] = value
    return rank, metrics


def configure_precision():
    torch.set_grad_enabled(False)
    torch.manual_seed(20260909)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beam-widths", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--limit", type=int, default=0,
                        help="first N rows for implementation pilot; 0=all test users")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder-chunk-size", type=int, default=256)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.cell_index < 18:
        raise ValueError("cell-index must be 0..17")
    if args.limit < 0 or args.batch_size <= 0 or args.decoder_chunk_size <= 0:
        raise ValueError("invalid row or batch limits")
    if len(set(args.beam_widths)) != len(args.beam_widths):
        raise ValueError("repeated beam widths")
    if any(width < 10 for width in args.beam_widths):
        raise ValueError("top10 experiment requires beam widths >=10")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference must run in an sbatch GPU allocation")
    configure_precision()
    started = time.perf_counter()
    registry = load_runtime()
    ckpt_id = cell_ids()[args.cell_index]
    entry = registry[ckpt_id]
    if entry["status"] != "trained" or entry["task"] != "next1":
        raise ValueError(f"not a trained next-item checkpoint: {ckpt_id}")
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise ValueError("checkpoint hash drift")
    cohort = load_eval_cohort(entry)
    n = min(args.limit, len(cohort)) if args.limit else len(cohort)
    depth = entry["config"]["n_digit"]
    order = torch.randperm(depth, generator=torch.Generator().manual_seed(42)).tolist()
    model, model_report = diffusion.load(entry, device="cuda")
    print(f"[load] {ckpt_id} users={n}/{len(cohort)} fixed_order={order} "
          f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    validation = None
    if args.validate:
        from validate import verify_model_and_decoder
        count = min(8, n)
        validation = verify_model_and_decoder(
            model,
            torch.from_numpy(cohort.histories[:count]).to("cuda"),
            torch.from_numpy(cohort.history_mask[:count]).to("cuda"),
            cohort.sid_buckets,
            decode,
        )
        print(f"[validation] {json.dumps(validation)}", flush=True)

    # Store the same encoder states once for all conditions. FP32 and eval mode
    # are retained across encoding and decoding; no model or data selection.
    encoder = []
    with torch.inference_mode():
        for offset in range(0, n, args.batch_size):
            end = min(offset + args.batch_size, n)
            enc = diffusion.encode(
                model,
                torch.from_numpy(cohort.histories[offset:end]).to("cuda"),
                torch.from_numpy(cohort.history_mask[offset:end]).to("cuda"),
            )
            encoder.append(enc.cpu())
    encoder = torch.cat(encoder, dim=0)
    conditions = [(beam, policy) for beam in args.beam_widths for policy in POLICIES]
    shape = (len(conditions), n)
    predictions = np.full((*shape, 10, depth), -1, dtype=np.int64)
    scores = np.full((*shape, 10), -np.inf, dtype=np.float32)
    counts = np.zeros(shape, dtype=np.int64)
    ranks = np.zeros(shape, dtype=np.int64)
    values = np.zeros((*shape, len(METRICS)), dtype=np.float64)
    diagnostics = []
    for condition_index, (beam, policy) in enumerate(conditions):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        generated = unique = invalid = legal_unique = 0
        beam_sizes = None
        scores_available = policy != "legacy_confidence"
        if not scores_available:
            scores[condition_index] = np.nan
        for offset in range(0, n, args.batch_size):
            end = min(offset + args.batch_size, n)
            with torch.inference_mode():
                if policy == "legacy_confidence":
                    from legacy import decode_legacy_confidence
                    decoded = decode_legacy_confidence(
                        model, encoder[offset:end].to("cuda"), beam_width=beam,
                        top_k=10, allowed_sids=cohort.sid_buckets,
                    )
                else:
                    decoded = decode(
                        model, encoder[offset:end].to("cuda"), beam_width=beam,
                        policy=policy, order=order if policy == "fixed" else None,
                        top_k=10, allowed_sids=cohort.sid_buckets,
                        decoder_chunk_size=args.decoder_chunk_size,
                    )
            pred = decoded.codes.cpu().numpy()
            size = decoded.counts.cpu().numpy()
            predictions[condition_index, offset:end] = pred
            if scores_available:
                scores[condition_index, offset:end] = decoded.scores.cpu().numpy()
            counts[condition_index, offset:end] = size
            for index in range(end - offset):
                row = offset + index
                rank, metrics = evaluate_prediction(
                    pred[index, :size[index]], cohort.target_sids[row],
                    cohort.target_item_ids[row], cohort.sid_buckets,
                )
                ranks[condition_index, row] = rank or 0
                values[condition_index, row] = [metrics[name] for name in METRICS]
            if scores_available:
                generated += int(decoded.generated_counts.sum().item())
                unique += int(decoded.unique_counts.sum().item())
                invalid += int(decoded.invalid_counts.sum().item())
                legal_unique += int(decoded.legal_unique_counts.sum().item())
                beam_sizes = list(decoded.beam_sizes)
            if offset == 0 or end == n or end % 512 == 0:
                print(f"[decode] beam={beam} policy={policy} {end}/{n} "
                      f"seconds={time.perf_counter()-begin:.1f}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        diagnostics.append({
            "beam_width": beam, "policy": policy, "seconds": elapsed,
            "users_per_second": n / elapsed,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scores_available": scores_available,
            "active_beam_sizes": beam_sizes,
            "generated_paths": generated if scores_available else None,
            "unique_final_sids": unique if scores_available else None,
            "invalid_final_paths": invalid if scores_available else None,
            "legal_unique_final_sids": legal_unique if scores_available else None,
            "rows_with_ten_predictions": int((counts[condition_index] == 10).sum()),
            "rows_with_zero_predictions": int((counts[condition_index] == 0).sum()),
            "metrics": {name: float(values[condition_index, :, index].mean())
                        for index, name in enumerate(METRICS)},
        })
        print(f"[condition] {json.dumps(diagnostics[-1])}", flush=True)

    # Anchor both original reported conditions. Shared confidence intentionally
    # changes final-step expansion and is not itself a reproduction target.
    historical = {}
    for key, condition, suffix in (
        ("historical_fixed64_reconstruction", (64, "fixed"), "_random"),
        ("historical_confidence256_reconstruction", (256, "legacy_confidence"), ""),
    ):
        reconstruction = {}
        if condition not in conditions:
            historical[key] = reconstruction
            continue
        idx = conditions.index(condition)
        for outcome, legacy in (("sid_hit@10", "recall@10" + suffix),
                                ("ndcg@10", "ndcg@10" + suffix)):
            actual = float(values[idx, :, METRICS.index(outcome)].mean())
            old = float(entry["recorded_metrics"][legacy])
            reconstruction[outcome] = {
                "new": actual, "recorded": old, "delta": actual - old,
                "comparable_full_cohort": n == len(cohort),
                "within_one_hit_equivalent": abs(actual - old) <= 1 / len(cohort) + 1e-7,
            }
        historical[key] = reconstruction
    result = {
        "experiment": "controlled-exp1-matched-beam", "checkpoint": ckpt_id,
        "cell_index": args.cell_index, "category": "Industrial_and_Scientific",
        "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "quantizer": entry["quantizer"], "depth": depth,
        "codebook_size": entry["codebook_size"], "n_users": n,
        "full_cohort": n == len(cohort), "cohort_sha256": cohort.cohort_sha256,
        "fixed_order": order, "fixed_order_seed": 42,
        "checkpoint_sha256": entry["ckpt_sha256"], "config": entry["config"],
        "input_sha256": cohort.input_sha256, "sid_lineage": cohort.sid_lineage,
        "model_shape_validation": model_report, "pilot_validation": validation,
        "precision": "float32; TF32 disabled; no autocast; eval mode",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size, "decoder_chunk_size": args.decoder_chunk_size,
        "conditions": diagnostics, **historical,
        "total_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Matched search policies conditional on fixed confidence-selected "
            "checkpoints and one fixed seed42 order. Both policies fully expand "
            "the last digit; historical confidence used greedy final fill. "
            "legacy_confidence retains that original greedy final fill as a "
            "separate literal beam-only diagnostic and reproduction anchor. "
            "Equal beam caps do not imply equal candidate counts or runtime."),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "predictions.npz", users=np.asarray(cohort.users[:n]),
        user_ids=cohort.user_ids[:n], targets=cohort.target_sids[:n],
        target_item_ids=cohort.target_item_ids[:n],
        target_multiplicity=np.asarray([
            len(cohort.sid_buckets[tuple(int(v) for v in row)])
            for row in cohort.target_sids[:n]]),
        condition_beams=np.asarray([beam for beam, _ in conditions]),
        condition_policies=np.asarray([policy for _, policy in conditions]),
        codes=predictions, scores=scores, counts=counts, ranks=ranks,
        metric_names=np.asarray(METRICS), metrics=values,
    )
    with (args.out / "per_user.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["checkpoint", "user", "target_item", "target_multiplicity",
                         "beam", "policy", "rank", "n_predictions", *METRICS])
        for ci, (beam, policy) in enumerate(conditions):
            for row in range(n):
                target = tuple(int(v) for v in cohort.target_sids[row])
                writer.writerow([
                    ckpt_id, cohort.users[row], int(cohort.target_item_ids[row]),
                    len(cohort.sid_buckets[target]), beam, policy,
                    int(ranks[ci, row]), int(counts[ci, row]), *values[ci, row],
                ])
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"[done] {ckpt_id} wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
