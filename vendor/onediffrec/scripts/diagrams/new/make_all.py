#!/usr/bin/env python3
"""Regenerate the complete collision-analysis figure suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from parse_workbook import DEFAULT_WORKBOOK, build_tables, write_tables  # noqa: E402
from parse_research_notes import (  # noqa: E402
    DEFAULT_PDF,
    build_office_table,
)
from parse_resources import (  # noqa: E402
    DEFAULT_SWEEP_ROOT,
    DEFAULT_WANDB_ROOT,
    DependencyError,
    build_resource_rows,
    write_csv as write_resource_csv,
)
from plot_analysis import DEFAULT_OUTPUT_DIR, generate_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--research-notes", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sweep-metrics-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--wandb-root", type=Path, default=DEFAULT_WANDB_ROOT)
    parser.add_argument(
        "--skip-resource-refresh",
        action="store_true",
        help="use the checked-in resource snapshot instead of reading experiment logs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"

    print("Parsing and validating workbook")
    tables = build_tables(args.workbook.resolve())
    write_tables(tables, data_dir)

    print("Extracting the Office codebook-256 comparison")
    office_runs = build_office_table(args.research_notes.resolve())
    office_runs.to_csv(data_dir / "office_core_codebook256.csv", index=False)
    print(f"  office core  {len(office_runs):3d} rows -> {data_dir / 'office_core_codebook256.csv'}")

    print("Collecting training-compute and accelerator measurements")
    resource_path = data_dir / "resource_runs.csv"
    resources: pd.DataFrame | None = None
    if not args.skip_resource_refresh:
        try:
            resource_rows, resource_report = build_resource_rows(
                args.sweep_metrics_root.resolve(), args.wandb_root.resolve()
            )
            write_resource_csv(resource_rows, resource_path)
            resources = pd.DataFrame.from_records(resource_rows)
            print(
                f"  resources    {len(resources):3d} rows -> {resource_path} "
                f"({resource_report.unpaired_sweep_rows} unpaired)"
            )
        except (DependencyError, FileNotFoundError, ValueError) as exc:
            print(f"  resource refresh unavailable: {exc}")

    if resources is None:
        snapshot = DEFAULT_OUTPUT_DIR / "data" / "resource_runs.csv"
        if snapshot.exists():
            resources = pd.read_csv(snapshot)
            if snapshot.resolve() != resource_path.resolve():
                resources.to_csv(resource_path, index=False)
            print(f"  resources    {len(resources):3d} rows <- snapshot {snapshot}")
        else:
            print("  resources      0 rows (no logs or snapshot; resource figure omitted)")

    print("Generating analysis tables, figures, and report")
    generate_analysis(
        tables,
        output_dir,
        args.workbook.resolve(),
        office_runs=office_runs,
        research_notes=args.research_notes.resolve(),
        resources=resources,
    )
    print(f"Done. Open {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
