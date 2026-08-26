#!/usr/bin/env python3
"""Render every collected variant metric into the project notes' table layout.

Emits a markdown report (one HR/NDCG table per category x SID method, grouped by
codebook size, rows = digit count) plus a flat CSV of everything for further
analysis. Variants that have not finished yet are shown as blank cells so the
report doubles as a progress view of the sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants as V  # noqa: E402

REPORT_COLUMNS = ["HR@3", "HR@5", "HR@10", "NDCG@3", "NDCG@5", "NDCG@10"]
METHOD_LABELS = {"rqvae": "RQ-VAE", "rqkmeans": "RQ-KMeans", "MQ": "Parallel (MQ)"}
TREE_LABELS = {"next-item": "Next Item Prediction", "two-item": "Next 2 Items Prediction"}

FLAT_FIELDS = [
    "variant_id", "tree", "category", "method", "codebooks", "codebook_size",
    "items", "unique_full_sids", "collisions", "collision_rate",
    "component_tokens", "codebook_utilization",
    "best_eval_loss", "stopped_at_epoch", "stopped_at_step",
    "train_seconds", "eval_seconds",
] + REPORT_COLUMNS


def load_records(metrics_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name.endswith(".assets.json"):
            continue
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "variant_id" in record:
            records[record["variant_id"]] = record
    return records


def cell(record: dict | None, column: str) -> str:
    if not record:
        return ""
    value = record.get("metrics", {}).get(column)
    return f"{value:.2f}" if isinstance(value, (int, float)) else ""


def render_markdown(records: dict[str, dict], trees: list[str]) -> str:
    lines: list[str] = ["# OneDiffRec SFT sweep results", ""]

    done = len(records)
    lines.append(f"{done} variants collected. Metrics are percentages, HR/NDCG from `calc.py`.")
    lines.append("")

    for tree in trees:
        lines.append(f"## {TREE_LABELS.get(tree, tree)}")
        lines.append("")
        for category in V.CATEGORIES:
            for method in V.METHODS:
                by_size = {
                    size: [
                        records.get(V.Variant(tree, category, method, cb, size).variant_id)
                        for cb in V.CODEBOOKS
                    ]
                    for size in V.SIZES
                }
                if not any(any(group) for group in by_size.values()):
                    continue  # nothing collected for this method yet
                lines.append(f"### {category} — {METHOD_LABELS.get(method, method)}")
                lines.append("")
                for size in V.SIZES:
                    present = by_size[size]
                    if not any(present):
                        continue
                    lines.append(f"**Codebook size = {size}**")
                    lines.append("")
                    lines.append("| Digits | " + " | ".join(REPORT_COLUMNS)
                                 + " | Collision % | Codebook util % | Best eval loss |")
                    lines.append("|---" * (len(REPORT_COLUMNS) + 4) + "|")
                    for cb, record in zip(V.CODEBOOKS, present):
                        cells = [cell(record, col) for col in REPORT_COLUMNS]
                        coll = record.get("collision_rate") if record else None
                        util = record.get("codebook_utilization") if record else None
                        loss = record.get("best_eval_loss") if record else None
                        cells.append(f"{coll * 100:.2f}" if isinstance(coll, (int, float)) else "")
                        cells.append(f"{util * 100:.1f}" if isinstance(util, (int, float)) else "")
                        cells.append(f"{loss:.4f}" if isinstance(loss, (int, float)) else "")
                        lines.append(f"| {cb} | " + " | ".join(cells) + " |")
                    lines.append("")
        lines.append("")
    return "\n".join(lines)


def write_csv(records: dict[str, dict], path: Path, trees: list[str]) -> int:
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for tree in trees:
            for variant in V.enumerate_variants(tree):
                record = records.get(variant.variant_id)
                if not record:
                    continue
                row = {field: record.get(field) for field in FLAT_FIELDS}
                row.update({col: record.get("metrics", {}).get(col) for col in REPORT_COLUMNS})
                writer.writerow(row)
                rows += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="/l/users/leo.rodrigues/onediffrec/sweep")
    parser.add_argument("--trees", nargs="+", default=["next-item", "two-item"])
    parser.add_argument("--out-markdown", default=None)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    root = Path(args.sweep_root)
    records: dict[str, dict] = {}
    for tree in args.trees:
        metrics_dir = root / tree / "metrics"
        if metrics_dir.is_dir():
            records.update(load_records(metrics_dir))

    markdown = render_markdown(records, args.trees)
    if args.out_markdown:
        Path(args.out_markdown).write_text(markdown)
        print(f"wrote {args.out_markdown}")
    else:
        print(markdown)

    if args.out_csv:
        n = write_csv(records, Path(args.out_csv), args.trees)
        print(f"wrote {n} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
