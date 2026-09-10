#!/usr/bin/env python
"""Run retrospective Experiment 2: per-digit and prefix error decomposition."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .analysis import (
        CAVEATS,
        METRIC_DEFINITIONS,
        TOP_K,
        ValidationError,
        example_csv_rows,
        load_all_runs,
        summarize_all,
        validation_csv_rows,
    )
except ImportError:  # Allow direct ``python path/to/run.py`` execution.
    from analysis import (  # type: ignore[no-redef]
        CAVEATS,
        METRIC_DEFINITIONS,
        TOP_K,
        ValidationError,
        example_csv_rows,
        load_all_runs,
        summarize_all,
        validation_csv_rows,
    )


DEFAULT_WORK = Path(os.environ.get("SIDLENS_WORK", "/l/users/leo.rodrigues/sidlens"))
DEFAULT_BOOTSTRAP_SEED = 20260906
EXPECTED_CATEGORY = "Industrial_and_Scientific"
EXPECTED_GRID = frozenset(
    (quantizer, depth, width)
    for quantizer in ("MQ", "rqkmeans", "rqvae")
    for depth in (3, 4, 5)
    for width in (128, 512)
)


def validate_expected_grid(runs: Iterable[Any]) -> None:
    """Fail if the frozen balanced 3 x 3 x 2 analysis grid changes."""

    variants = [run.variant for run in runs]
    categories = {variant.category for variant in variants}
    observed = {
        (variant.quantizer, variant.depth, variant.width)
        for variant in variants
    }
    if categories != {EXPECTED_CATEGORY} or observed != EXPECTED_GRID:
        raise ValidationError(
            "retained next-item grid changed; "
            f"categories={sorted(categories)}, "
            f"missing={sorted(EXPECTED_GRID - observed)}, "
            f"extra={sorted(observed - EXPECTED_GRID)}"
        )
    if len(variants) != len(EXPECTED_GRID):
        raise ValidationError(
            f"retained next-item grid contains duplicate cells: {len(variants)} "
            f"runs for {len(EXPECTED_GRID)} expected cells"
        )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first))
        writer.writeheader()
        writer.writerow({key: "" if value is None else value for key, value in first.items()})
        count = 1
        for row in iterator:
            if list(row) != list(first):
                raise ValueError(f"inconsistent CSV schema while writing {path}")
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
            count += 1
    return count


def _metric_row(
    rows: list[dict[str, Any]],
    *,
    stratum_type: str,
    stratum_value: str,
    metric: str,
    digit: int | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["stratum_type"] == stratum_type
        and row["stratum_value"] == stratum_value
        and row["metric"] == metric
        and row["digit"] == digit
        and row["k"] == k
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one summary row for {(stratum_type, stratum_value, metric, digit, k)}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _optional_metric_row(
    rows: list[dict[str, Any]],
    **query: Any,
) -> dict[str, Any] | None:
    try:
        return _metric_row(rows, **query)
    except ValueError:
        return None


def _format_interval(row: dict[str, Any] | None, *, percent: bool = True) -> str:
    if row is None:
        return "n/a"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    estimate = float(row["estimate"]) * scale
    if row["ci_low"] is None or row["ci_high"] is None:
        return f"{estimate:.2f}{suffix} (CI not computed)"
    low = float(row["ci_low"]) * scale
    high = float(row["ci_high"]) * scale
    return f"{estimate:.2f}{suffix} (95% CI {low:.2f}–{high:.2f}{suffix})"


def render_report(result: dict[str, Any]) -> str:
    """Render a concise human view; ``result.json`` remains authoritative."""

    rows = result["strata_metrics"]
    overall = lambda metric, digit=None, k=None: _metric_row(  # noqa: E731
        rows,
        stratum_type="overall",
        stratum_value="all",
        metric=metric,
        digit=digit,
        k=k,
    )
    lcp = overall("top1_lcp_length")
    lcp_fraction = overall("top1_lcp_fraction")
    top1_exact = overall("top1_full_sid_correct")
    exact_hr10 = overall("full_sid_hr_exact", k=10)
    legacy_hr10 = overall("full_sid_hr_legacy", k=10)
    digits = sorted(
        {
            int(row["digit"])
            for row in rows
            if row["stratum_type"] == "overall"
            and row["metric"] == "top1_first_error_rate"
            and row["digit"] is not None
        }
    )
    error_rows = [overall("top1_first_error_rate", digit=d) for d in digits]
    most_common_error = max(error_rows, key=lambda row: row["estimate"])

    lines = [
        "# Retrospective Experiment 2 — SID prefix errors",
        "",
        "> This is a generated reading aid. `result.json` is the source of truth.",
        "",
        "## Scope and validation",
        "",
        (
            f"All {result['scope']['n_configs']} retained next-item AR configurations "
            f"validated: {result['scope']['n_example_config_rows']:,} example–configuration "
            f"rows, {result['scope']['n_users']:,} unique users, and "
            f"{len(result['validation']) * len(result['scope']['top_k']) * 2} recorded HR/NDCG "
            "values reproduced within four-decimal rounding tolerance."
        ),
        "",
        "## Headline findings",
        "",
        (
            f"- Rank 1 matches a mean **{_format_interval(lcp, percent=False)} digits**, or "
            f"**{_format_interval(lcp_fraction)}** of SID depth. Exact rank-1 full-SID "
            f"accuracy is **{_format_interval(top1_exact)}**."
        ),
        (
            f"- The modal first mismatch is digit {most_common_error['digit']}: "
            f"**{_format_interval(most_common_error)}** of applicable rows. Error hazard "
            "falls after a prefix survives, but this is descriptive survivorship, not an "
            "independent digit effect."
        ),
        (
            f"- At K=10, exact full-SID HR is **{_format_interval(exact_hr10)}**. The "
            f"historical title-compatible value is **{_format_interval(legacy_hr10)}**; "
            "the gap comes from legacy matching rather than better SID-prefix recovery."
        ),
        "",
        "### First-error process at rank 1",
        "",
        "| Digit | Eligible configs | First-error rate | Conditional accuracy | Conditional hazard | Surviving rows |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for digit in digits:
        error = overall("top1_first_error_rate", digit=digit)
        accuracy = _optional_metric_row(
            rows,
            stratum_type="overall",
            stratum_value="all",
            metric="top1_conditional_digit_accuracy",
            digit=digit,
        )
        hazard = _optional_metric_row(
            rows,
            stratum_type="overall",
            stratum_value="all",
            metric="top1_first_error_hazard",
            digit=digit,
        )
        eligible_configs = (
            hazard["n_configs_contributing"]
            if hazard is not None
            else error["n_configs_contributing"]
        )
        surviving_rows = int(hazard["denominator"]) if hazard is not None else 0
        lines.append(
            f"| {digit} | {eligible_configs} | "
            f"{_format_interval(error)} | {_format_interval(accuracy)} | "
            f"{_format_interval(hazard)} | {surviving_rows:,} |"
        )

    lines.extend(
        (
            "",
            "### Prefix hit rate",
            "",
            "| Prefix depth | Eligible configs | hit@1 | hit@10 | hit@50 |",
            "|---:|---:|---:|---:|---:|",
        )
    )
    for digit in digits:
        values = [overall("prefix_hit", digit=digit, k=k) for k in (1, 10, 50)]
        lines.append(
            f"| {digit} | {values[0]['n_configs_contributing']} | "
            + " | ".join(_format_interval(row) for row in values)
            + " |"
        )

    lines.extend(
        (
            "",
            "### Quantizer strata",
            "",
            "| Quantizer | Rank-1 LCP / depth | Exact full-SID HR@10 |",
            "|---|---:|---:|",
        )
    )
    quantizers = sorted(
        {
            row["stratum_value"]
            for row in rows
            if row["stratum_type"] == "quantizer"
        }
    )
    for quantizer in quantizers:
        q_lcp = _metric_row(
            rows,
            stratum_type="quantizer",
            stratum_value=quantizer,
            metric="top1_lcp_fraction",
        )
        q_hr = _metric_row(
            rows,
            stratum_type="quantizer",
            stratum_value=quantizer,
            metric="full_sid_hr_exact",
            k=10,
        )
        lines.append(
            f"| {quantizer} | {_format_interval(q_lcp)} | {_format_interval(q_hr)} |"
        )

    empty_slots = sum(v["empty_beam_slots"] for v in result["validation"])
    title_fallbacks = sum(v["earliest_title_fallback_hits"] for v in result["validation"])
    lines.extend(
        (
            "",
            "## Validity limits",
            "",
            f"- The 50-slot beams contain {empty_slots:,} strictly trailing empty padding "
            "slots; padding is retained at rank and cannot count as a hit.",
            f"- {title_fallbacks:,} example–configuration rows have a title match as their "
            "earliest legacy hit. Prefix metrics never use that equivalence.",
        )
    )
    lines.extend(f"- {caveat}" for caveat in result["caveats"])
    lines.extend(("", "See `result.json` and the long-form CSVs for exact denominators, hashes, and intervals.", ""))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work",
        type=Path,
        default=DEFAULT_WORK,
        help="SidLens bulk-data root (default: %(default)s)",
    )
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--info-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--metric-tolerance-points",
        type=float,
        default=5.1e-5,
        help="maximum absolute discrepancy from four-decimal recorded percentages",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen = args.work / "frozen"
    metrics_dir = args.metrics_dir or frozen / "results/sweep_metrics/next-item/metrics"
    test_dir = args.test_dir or frozen / "data/splits/next-item/test"
    info_dir = args.info_dir or frozen / "sids/info/next-item"
    out_dir = args.out or args.work / "derived/retrospective/exp2_prefix"

    # All validation and statistics complete before any result file is opened.
    runs = load_all_runs(
        metrics_dir,
        test_dir=test_dir,
        info_dir=info_dir,
        tolerance_points=args.metric_tolerance_points,
    )
    validate_expected_grid(runs)
    config_rows, strata_rows = summarize_all(
        runs,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "examples.csv": write_csv(out_dir / "examples.csv", example_csv_rows(runs)),
        "config_metrics.csv": write_csv(out_dir / "config_metrics.csv", config_rows),
        "strata_metrics.csv": write_csv(out_dir / "strata_metrics.csv", strata_rows),
        "metric_reproduction.csv": write_csv(
            out_dir / "metric_reproduction.csv", validation_csv_rows(runs)
        ),
    }
    snapshot_path = Path(__file__).resolve().parents[3] / "manifests/CURRENT"
    snapshot = snapshot_path.read_text(encoding="utf-8").strip() if snapshot_path.is_file() else None
    result = {
        "schema_version": 1,
        "experiment": "retrospective_exp2_prefix_error_decomposition",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshot": snapshot,
        "scope": {
            "task": "next-item",
            "paradigm": "autoregressive",
            "category": sorted({run.variant.category for run in runs}),
            "n_configs": len(runs),
            "n_examples_per_config": sorted({len(run.examples) for run in runs}),
            "n_example_config_rows": sum(len(run.examples) for run in runs),
            "n_users": len({e.user_id for run in runs for e in run.examples}),
            "top_k": list(TOP_K),
        },
        "definitions": {
            "lcp": "Number of consecutive equal SID digits from digit 1.",
            "first_error_digit": (
                "One-based first unequal rank-1 digit; null when the full SID matches."
            ),
            "bootstrap": (
                "Percentile 95% interval from resampling users with replacement. Every "
                "row and retained configuration for a sampled user stays in the cluster."
            ),
            "metrics": METRIC_DEFINITIONS,
        },
        "bootstrap": {
            "unit": "user_id",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
            "interval": "percentile_95",
            "paired_across_configs": True,
        },
        "metric_reproduction_tolerance_points": args.metric_tolerance_points,
        "caveats": list(CAVEATS),
        "validation": [dict(run.validation) for run in runs],
        "config_metrics": config_rows,
        "strata_metrics": strata_rows,
        "output_row_counts": counts,
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    print(
        f"validated {len(runs)} configs and {sum(len(r.examples) for r in runs)} "
        f"example-config rows; wrote {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
