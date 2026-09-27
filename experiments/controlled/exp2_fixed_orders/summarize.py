#!/usr/bin/env python
"""Validate six complete order-sweep cells and bootstrap paired users."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.controlled.exp1_matched_beam.summarize import (
    _prediction_checks, _require, _verified_files, sha256,
)
from experiments.controlled.exp2_fixed_orders.common import (
    BOOTSTRAP_SEED, CELL_INDICES, conditions, label, paired_effects,
)

OUTCOMES = ("sid_hit@10", "ndcg@10", "item_uniform@10")


def check_all_hashes(folder):
    hashes = _verified_files(folder)
    for line in (folder / "output.sha256").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        p = Path(name.lstrip("*"))
        p = p if p.is_absolute() else folder / p
        _require(p.resolve().parent == folder.resolve(), "hash manifest escapes cell")
        _require(sha256(p) == digest, f"artifact hash drift: {p}")
    return hashes


def load_grid(root, expected_users=6297):
    root = Path(root)
    expected = {f"cell-{i:02d}" for i in CELL_INDICES}
    _require({p.name for p in root.glob("cell-*") if p.is_dir()} == expected,
             "summary requires exactly the six declared cells")
    cells, values, hashes, quality = [], [], {}, []
    common_users = common_ids = common_targets = common_axes = common_source = common_args = None
    for index in CELL_INDICES:
        folder = root / f"cell-{index:02d}"
        status = (folder / "status.txt").read_text()
        _require(status.startswith("complete ") and "exit=0" in status, f"incomplete cell {index}")
        hashes.update(check_all_hashes(folder))
        result = json.loads((folder / "result.json").read_text())
        quantizer = ("MQ", "rqkmeans", "rqvae")[index // 6]
        width = 128 if index % 2 == 0 else 512
        _require((result["cell_index"], result["checkpoint"], result["depth"], result["codebook_size"]) ==
                 (index, f"diff-next1-{quantizer}-3cb-{width}", 3, width), "cell/checkpoint mismatch")
        _require(result["n_users"] == expected_users and result["full_cohort"], "partial user cohort")
        _require(result["validation"]["status"] == "passed" and result["validation"]["archive"]["status"] == "passed",
                 "missing baseline validation")
        arguments = json.loads((folder / "arguments.json").read_text())
        argument_signature = {k: v for k, v in arguments.items() if k not in ("out", "cell_index")}
        source = (folder / "source.sha256").read_text()
        if common_source is not None:
            _require(source == common_source, "source hashes differ across cells")
            _require(argument_signature == common_args, "settings differ across cells")
        common_source, common_args = source, argument_signature
        with np.load(folder / "predictions.npz", allow_pickle=False) as artifact:
            data = {k: artifact[k] for k in artifact.files}
        axes = [(int(b), str(p), None if str(p) == "confidence" else tuple(map(int, order)))
                for b, p, order in zip(data["condition_beams"], data["condition_policies"], data["condition_orders"], strict=True)]
        _require(axes == conditions(arguments["beam_widths"]), "missing, duplicate, or permuted declared conditions")
        reported_axes = [(int(x["beam_width"]), x["policy"], None if x["order"] is None else tuple(x["order"]))
                         for x in result["conditions"]]
        _require(axes == reported_axes, "NPZ/result condition axes differ")
        pseudo_axes = [(b, label(p, order)) for b, p, order in axes]
        reported = {a: r for a, r in zip(pseudo_axes, result["conditions"], strict=True)}
        names = _prediction_checks(data, n=expected_users, d=3, width=width,
                                   conditions=pseudo_axes, reported=reported, label=str(folder))
        if common_users is not None:
            for key, previous in (("users", common_users), ("user_ids", common_ids), ("target_item_ids", common_targets)):
                _require(np.array_equal(data[key], previous), f"cross-cell {key} mismatch")
            _require(axes == common_axes, "cross-cell condition mismatch")
        common_users, common_ids, common_targets = data["users"], data["user_ids"], data["target_item_ids"]
        common_axes = axes
        selected = data["metrics"][:, :, [names.index(name) for name in OUTCOMES]]
        values.append(selected)
        for ci, (beam, policy, order) in enumerate(axes):
            diag = result["conditions"][ci]
            for mi, name in enumerate(names):
                _require(abs(float(data["metrics"][ci, :, mi].mean()) - diag["metrics"][name]) < 1e-12,
                         "stored mean differs from per-user recomputation")
            row = {"cell_index": index, "checkpoint": result["checkpoint"], "beam": beam,
                   "condition": label(policy, order), "order_zero_based": "" if order is None else "".join(map(str, order)),
                   "n_users": expected_users, "seconds": diag["seconds"], "decoder_rows": diag["decoder_rows"],
                   "peak_cuda_allocated_bytes": diag["peak_cuda_allocated_bytes"],
                   "ten_predictions_pct": float((data["counts"][ci] == 10).mean() * 100),
                   "mean_list_length": float(data["counts"][ci].mean())}
            for field in ("generated_counts", "unique_counts", "invalid_counts", "legal_unique_counts"):
                _require(data[field].shape == (len(axes), expected_users), "path diagnostic shape")
                _require(int(data[field][ci].sum()) == diag[field], "path diagnostic total mismatch")
                row[field] = diag[field]
            row["duplicate_final_paths"] = row["generated_counts"] - row["unique_counts"]
            row.update({name + "_pct": float(selected[ci, :, mi].mean() * 100) for mi, name in enumerate(OUTCOMES)})
            quality.append(row)
        cells.append(result)
    return cells, np.stack(values), common_axes, hashes, quality


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    args = ap.parse_args()
    _require(not args.out.exists(), "summary output already exists")
    pilot_hashes = check_all_hashes(args.pilot)
    pilot = json.loads((args.pilot / "validation.json").read_text())
    _require(pilot["status"] == pilot["model"]["status"] == pilot["order_invariance"]["status"] == "passed",
             "pilot numerical checks failed")
    cells, values, axes, hashes, quality = load_grid(args.input)
    def source_entries(folder):
        return {name: digest for digest, name in
                (line.split(maxsplit=1) for line in (folder / "source.sha256").read_text().splitlines())}
    pilot_source = source_entries(args.pilot)
    full_source = source_entries(args.input / "cell-00")
    for name in ("experiments/controlled/exp2_fixed_orders/run.py",
                 "experiments/controlled/exp2_fixed_orders/common.py",
                 "experiments/controlled/exp1_matched_beam/run.py",
                 "experiments/controlled/exp1_matched_beam/validate.py",
                 "src/sidlens/analysis/matched_decode.py"):
        _require(name in pilot_source and pilot_source[name] == full_source.get(name),
                 f"pilot numerical source differs from full run: {name}")
    stats = paired_effects(values, axes, n_boot=args.bootstrap, seed=args.seed)
    overall, per_cell, per_order = [], [], []
    for bi, beam in enumerate(stats["beams"]):
        confidence = axes.index((beam, "confidence", None))
        fixed = [i for i, (b, p, _) in enumerate(axes) if b == beam and p == "fixed"]
        for mi, metric in enumerate(OUTCOMES):
            row = {"beam": beam, "outcome": metric,
                   "confidence_pct": float(values[:, confidence, :, mi].mean() * 100),
                   "mean_fixed_pct": float(values[:, fixed, :, mi].mean() * 100),
                   "difference_pp": float(stats["overall"][bi, mi] * 100),
                   "ci95_pp": (stats["overall_ci"][:, bi, mi] * 100).tolist()}
            overall.append(row)
            for ci, cell in enumerate(cells):
                per_cell.append({"checkpoint": cell["checkpoint"], "beam": beam, "outcome": metric,
                    "confidence_pct": float(values[ci, confidence, :, mi].mean() * 100),
                    "mean_fixed_pct": float(values[ci, fixed, :, mi].mean() * 100),
                    "difference_pp": float(stats["per_cell"][bi, ci, mi] * 100),
                    "ci95_low_pp": float(stats["per_cell_ci"][0, bi, ci, mi] * 100),
                    "ci95_high_pp": float(stats["per_cell_ci"][1, bi, ci, mi] * 100)})
        for ai in fixed:
            per_order.append({"beam": beam, "order_zero_based": label(axes[ai][1], axes[ai][2]),
                              **{m + "_pct": float(values[:, ai, :, mi].mean() * 100) for mi, m in enumerate(OUTCOMES)}})
    result = {"experiment": "controlled-exp2-fixed-orders", "status": "complete",
        "n_cells": 6, "n_users": int(values.shape[2]), "beams": stats["beams"],
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed, "unit": "paired user, all six cells together"},
        "overall": overall, "per_cell": per_cell, "per_order": per_order,
        "decode_scoring_seconds": sum(row["seconds"] for row in quality),
        "input_hashes": {**hashes, **pilot_hashes}, "pilot": str(args.pilot),
        "validation": {"all_six_cells": True, "all_6297_users": True, "same_source_and_arguments": True,
                       "independent_prediction_rank_and_metric_checks": True, "archive_baselines_reproduced": True},
        "claim_boundary": "Frozen confidence-selected checkpoints; user uncertainty excludes training seeds; exact-SID accuracy does not resolve item collisions; equal beams are not equal compute."}
    args.out.mkdir(parents=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out / "validation.json").write_text(json.dumps(result["validation"], indent=2) + "\n")
    write_csv(args.out / "per_cell.csv", per_cell)
    write_csv(args.out / "per_order.csv", per_order)
    write_csv(args.out / "quality_cost.csv", quality)
    lines = ["# All six fixed reveal orders versus confidence", "",
             f"Completed six depth-three cells × {values.shape[2]:,} paired users × seven conditions at beams {stats['beams']}.", "",
             "Orders use zero-based positions: 012 means digit 1, then 2, then 3. Fixed means average outcome across six separate lists.", "",
             "| Beam | Outcome | Confidence | Mean fixed | Difference (pp) | Paired 95% interval (pp) |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in overall:
        lo, hi = row["ci95_pp"]
        lines.append(f"| {row['beam']} | {row['outcome']} | {row['confidence_pct']:.3f}% | {row['mean_fixed_pct']:.3f}% | {row['difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Per-order averages (exploratory)", "", "| Beam | Fixed order | SID HR@10 | NDCG@10 |",
              "| --- | --- | ---: | ---: |"]
    for row in per_order:
        lines.append(f"| {row['beam']} | {row['order_zero_based']} | {row['sid_hit@10_pct']:.3f}% | {row['ndcg@10_pct']:.3f}% |")
    lines += ["", "## Validation and scope", "",
        "- All six full cohorts, condition axes, input/output hashes, settings and source identities passed.",
        "- Seed-42 fixed-order and shared confidence predictions/ranks/metrics reproduced archived results exactly for every full-cohort cell.",
        "- A separate pilot passed vendor fixed-order reconstruction and all-order/confidence batch/chunk checks at beams 64 and 256.",
        f"- Intervals resample users jointly across all cells ({args.bootstrap:,} replicates, seed {args.seed}).",
        f"- Total decoding/scoring time was {result['decode_scoring_seconds']/60:.2f} GPU minutes, excluding loading, queue and reporting.",
        "- Quality/cost, per-cell effects and all per-user predictions are retained alongside this report.",
        "- Best test order is descriptive; selecting an order for deployment needs a separate validation set.",
        "- Exact-SID outcomes can conceal shared-item buckets. Item-uniform outcomes assume equal weights within each bucket.",
        "- Results concern these frozen confidence-selected checkpoints; uncertainty across training seeds is not measured."]
    if 256 not in stats["beams"]:
        lines.append("- Beam 256 was completed only for the 64-user implementation pilot; the full primary result is beam 64.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.8))
        rows = [r for r in per_cell if r["outcome"] == "sid_hit@10"]
        for bi, beam in enumerate(stats["beams"]):
            sub = [r for r in rows if r["beam"] == beam]
            y = np.arange(len(sub)) + bi * .15
            x = np.asarray([r["difference_pp"] for r in sub])
            low = np.asarray([r["ci95_low_pp"] for r in sub])
            high = np.asarray([r["ci95_high_pp"] for r in sub])
            ax.errorbar(x, y, xerr=np.array([x - low, high - x]), fmt="o", capsize=3, label=f"Beam {beam}")
        ax.set_yticks(np.arange(6), [c["checkpoint"].replace("diff-next1-", "") for c in cells])
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_xlabel("Confidence minus mean of six fixed orders, HR@10 (pp)")
        ax.legend()
        fig.tight_layout()
        (args.out / "figures").mkdir()
        fig.savefig(args.out / "figures/order_effect.png", dpi=180)
        fig.savefig(args.out / "figures/order_effect.pdf")
        plt.close(fig)
    except ImportError:
        lines.append("- Matplotlib unavailable; CSV tables are the exportable visualization source.")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    (args.out / "status.txt").write_text("complete exit=0\n")
    (args.out / "output.sha256").write_text("".join(
        f"{sha256(p)}  {p.relative_to(args.out)}\n" for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "output.sha256"))
    print(json.dumps({"report": str(args.out / "report.md"), "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
