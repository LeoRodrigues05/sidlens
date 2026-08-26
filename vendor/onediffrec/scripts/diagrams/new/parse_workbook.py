#!/usr/bin/env python3
"""Parse the collision/performance workbook into validated tidy tables.

The source workbook is hand-maintained: a collision-rate matrix occupies the
first ten rows, followed by a sequence of metric blocks.  This module converts
that layout into tables suitable for plotting and performs structural checks so
that silently shifted rows or headers fail early.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKBOOK = REPO_ROOT / "data" / "CollisionRates_GenRec_Paradigms.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "diagrams" / "new" / "data"

METRICS = ["HR@3", "HR@5", "HR@10", "NDCG@3", "NDCG@5", "NDCG@10"]
TOKENIZER_ORDER = ["RQ-VAE", "RQ-KMeans", "MQ"]

COLLISION_RE = re.compile(r"([\d.]+)%\s*\(([\d,]+)\)")
CODEBOOK_RE = re.compile(r"codebook\s+size\s*=\s*(\d+)", re.IGNORECASE)
DEPTH_RE = re.compile(r"\(\s*(\d+)\s*-?\s*digits?\s*\)", re.IGNORECASE)


def _text(value: object) -> str:
    """Return a normalized, printable cell value."""

    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def _tokenizer_from_text(value: object) -> str | None:
    low = _text(value).lower().replace("_", "-")
    if "rq-vae" in low:
        return "RQ-VAE"
    if "rq-kmeans" in low or "rq-kmeans" in low.replace(" ", ""):
        return "RQ-KMeans"
    if "parallel sid" in low or low == "mq" or low.startswith("mq "):
        return "MQ"
    return None


def parse_collisions(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract the two side-by-side collision matrices."""

    blocks = [
        # dataset, method column, depth column, codebook columns
        ("Industrial", 1, 2, {128: 3, 256: 4, 512: 5}),
        ("Office", 8, 9, {128: 10, 256: 11, 512: 12}),
    ]
    records: list[dict[str, object]] = []

    for dataset, method_col, depth_col, codebook_cols in blocks:
        for row in range(1, min(10, len(raw))):
            tokenizer = _tokenizer_from_text(raw.iat[row, method_col])
            if tokenizer is None:
                continue
            depth = int(raw.iat[row, depth_col])
            for codebook, col in codebook_cols.items():
                match = COLLISION_RE.fullmatch(_text(raw.iat[row, col]))
                if match is None:
                    raise ValueError(
                        f"Could not parse collision cell at row {row + 1}, "
                        f"column {col + 1}: {raw.iat[row, col]!r}"
                    )
                collision_pct = float(match.group(1))
                collision_excess = int(match.group(2).replace(",", ""))
                records.append(
                    {
                        "dataset": dataset,
                        "tokenizer": tokenizer,
                        "depth": depth,
                        "codebook": codebook,
                        "collision_pct": collision_pct,
                        # The project computes this as N - number of unique
                        # full SIDs: excess assignments, not the number of all
                        # items that belong to a non-singleton SID group.
                        "collision_excess": collision_excess,
                        # These provenance flags come directly from row-one/two
                        # notes in the workbook.
                        "collision_source_status": (
                            "previous_run"
                            if dataset == "Office" or codebook == 256
                            else "current_run"
                        ),
                    }
                )

    collisions = pd.DataFrame.from_records(records)
    collisions["tokenizer"] = pd.Categorical(
        collisions["tokenizer"], TOKENIZER_ORDER, ordered=True
    )
    return collisions.sort_values(
        ["dataset", "tokenizer", "depth", "codebook"]
    ).reset_index(drop=True)


def _metric_family(model: str) -> str:
    low = model.lower()
    if "diffusion" in low or "diffgrm" in low:
        # The workbook changes the row label from "Mask Diffusion" at
        # codebooks 128/512 to "DiffGRM" at 256.  Keep the shared analytical
        # class visibly tentative instead of claiming identical implementations.
        return "Diffusion-family"
    return "Autoregressive"


def _clean_model(raw_model: str) -> str:
    model = DEPTH_RE.sub("", raw_model)
    model = re.sub(r"\s+", " ", model).strip()
    return model.replace("- ", "-")


def parse_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Walk the metric blocks while tracking tokenizer and codebook headers."""

    records: list[dict[str, object]] = []
    tokenizer: str | None = None
    codebook: int | None = None

    for row in range(10, len(raw)):
        label = _text(raw.iat[row, 0])
        if not label:
            continue

        header_tokenizer = _tokenizer_from_text(label)
        if "sid" in label.lower() and header_tokenizer is not None:
            tokenizer = header_tokenizer

        codebook_match = CODEBOOK_RE.search(label)
        if codebook_match is not None:
            codebook = int(codebook_match.group(1))

        depth_match = DEPTH_RE.search(label)
        if depth_match is None:
            continue
        if tokenizer is None or codebook is None:
            raise ValueError(
                f"Metric row {row + 1} appeared before tokenizer/codebook headers"
            )

        values = {
            metric: pd.to_numeric(raw.iat[row, col + 1], errors="coerce")
            for col, metric in enumerate(METRICS)
        }
        if all(pd.isna(value) for value in values.values()):
            # The workbook includes placeholders for planned runs.
            continue
        if any(pd.isna(value) for value in values.values()):
            missing = [metric for metric, value in values.items() if pd.isna(value)]
            raise ValueError(f"Partially filled metric row {row + 1}: {missing}")

        model = _clean_model(label)
        family = _metric_family(model)
        is_core = model == "Qwen2.5-1.5B" or model in {
            "Mask Diffusion",
            "DiffGRM",
        }
        records.append(
            {
                "dataset": "Industrial",
                "dataset_assignment_inferred": True,
                "tokenizer": tokenizer,
                "codebook": codebook,
                "depth": int(depth_match.group(1)),
                "model": model,
                "family": family,
                "is_core": is_core,
                "workbook_row": row + 1,
                **values,
            }
        )

    metrics = pd.DataFrame.from_records(records)
    metrics["tokenizer"] = pd.Categorical(
        metrics["tokenizer"], TOKENIZER_ORDER, ordered=True
    )
    return metrics.sort_values(
        ["codebook", "tokenizer", "depth", "family", "model"]
    ).reset_index(drop=True)


def validate_tables(collisions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    collision_key = ["dataset", "tokenizer", "depth", "codebook"]
    if collisions.duplicated(collision_key).any():
        raise ValueError("Duplicate collision configurations found")

    expected_collision_configs = {
        (dataset, tokenizer, depth, codebook)
        for dataset in ("Industrial", "Office")
        for tokenizer in TOKENIZER_ORDER
        for depth in (3, 4, 5)
        for codebook in (128, 256, 512)
    }
    actual_collision_configs = {
        tuple(row)
        for row in collisions[collision_key].itertuples(index=False, name=None)
    }
    if actual_collision_configs != expected_collision_configs:
        missing = expected_collision_configs - actual_collision_configs
        extra = actual_collision_configs - expected_collision_configs
        raise ValueError(f"Unexpected collision grid; missing={missing}, extra={extra}")

    core = metrics[metrics["is_core"]]
    core_key = ["tokenizer", "depth", "codebook", "family"]
    if core.duplicated(core_key).any():
        raise ValueError("Duplicate core metric rows found")

    family_counts = core.groupby(
        ["tokenizer", "depth", "codebook"], observed=True
    )["family"].nunique()
    if len(family_counts) != 27 or not family_counts.eq(2).all():
        raise ValueError(
            "Core grid must contain both model families for all 27 SID configurations"
        )


def build_tables(workbook: Path) -> dict[str, pd.DataFrame]:
    if not workbook.exists():
        raise FileNotFoundError(
            f"Workbook not found: {workbook}\n"
            "The repository ignores data/*.xlsx, so copy the local workbook into "
            "data/ or pass --workbook."
        )

    raw = pd.read_excel(workbook, sheet_name="Sheet1", header=None, engine="openpyxl")
    collisions = parse_collisions(raw)
    metrics = parse_metrics(raw)
    validate_tables(collisions, metrics)

    industrial_collisions = collisions[collisions["dataset"] == "Industrial"].drop(
        columns="dataset"
    )
    runs = metrics.merge(
        industrial_collisions,
        on=["tokenizer", "depth", "codebook"],
        how="left",
        validate="many_to_one",
    )
    if runs["collision_pct"].isna().any():
        raise ValueError("At least one metric row has no matching collision rate")

    core_runs = runs[runs["is_core"]].copy().reset_index(drop=True)
    return {
        "collisions": collisions,
        "metrics": metrics,
        "runs": runs,
        "core_runs": core_runs,
    }


def write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        print(f"  {name:12s} {len(table):3d} rows -> {path}")


def main(workbook: Path = DEFAULT_WORKBOOK, output_dir: Path = DEFAULT_OUTPUT_DIR):
    tables = build_tables(Path(workbook))
    write_tables(tables, Path(output_dir))
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    main(args.workbook, args.output_dir)
