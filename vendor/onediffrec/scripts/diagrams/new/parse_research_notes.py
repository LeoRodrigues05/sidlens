#!/usr/bin/env python3
"""Extract the complete Office codebook-256 next-item table from the notes PDF.

The source table spans PDF pages 1--3.  Its text layout is irregular (two row
labels wrap onto the following page), so extraction deliberately uses the
stable table ordering after validating the section markers and the exact number
of six-metric rows.  This is safer than trying to infer row identity from visual
line breaks produced by a particular PDF text extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from parse_workbook import METRICS, TOKENIZER_ORDER


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PDF = REPO_ROOT / "data" / "Research Notes - OneDiffRec (3).pdf"
DEFAULT_OUTPUT = REPO_ROOT / "diagrams" / "new" / "data" / "office_core_codebook256.csv"

FLOAT_RE = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")
START_MARKER = "Next Item Prediction: Office"
END_MARKER = "Jul 14, 2026"


def build_office_table(pdf_path: Path) -> pd.DataFrame:
    """Return 18 tidy rows: 3 tokenizers x 3 depths x 2 paradigms."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Research-notes PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    if len(reader.pages) < 3:
        raise ValueError("Research-notes PDF must contain at least three pages")

    page_texts = [
        reader.pages[index].extract_text(
            extraction_mode="layout", layout_mode_space_vertically=False
        )
        or ""
        for index in range(3)
    ]
    text = "\n".join(page_texts)
    normalized_text = re.sub(r"\s+", " ", text)
    if START_MARKER not in normalized_text or END_MARKER not in normalized_text:
        raise ValueError("Could not locate the Office next-item table boundaries")
    for marker in ("RQ-VAE SID", "RQ-Kmeans SID", "Parallel SID"):
        if marker not in normalized_text:
            raise ValueError(f"Missing expected Office table marker: {marker!r}")

    # Every metric row contains exactly six decimal values.  Labels and dates
    # contain fewer, so this remains stable despite the two page-wrap anomalies.
    value_rows: list[tuple[int, str, list[float]]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        for line in page_text.splitlines():
            values = [float(value) for value in FLOAT_RE.findall(line)]
            if len(values) >= len(METRICS):
                value_rows.append((page_number, line, values[-len(METRICS) :]))
    if len(value_rows) != 18:
        raise ValueError(
            f"Expected 18 six-metric Office rows on PDF pages 1--3; found {len(value_rows)}"
        )

    records: list[dict[str, object]] = []
    row_index = 0
    for tokenizer in TOKENIZER_ORDER:
        for depth in (3, 4, 5):
            for family, model in (
                ("Autoregressive", "Qwen2.5-1.5B"),
                ("Diffusion-family", "DiffGRM"),
            ):
                source_page, source_line, metric_values = value_rows[row_index]
                expected_prefix = "Qwen2.5-1" if family == "Autoregressive" else "DiffGRM"
                if expected_prefix not in source_line:
                    raise ValueError(
                        "Office table row order or model label changed at row "
                        f"{row_index + 1}: expected {expected_prefix!r}"
                    )
                for hit_index, ndcg_index in ((0, 3), (1, 4), (2, 5)):
                    if metric_values[hit_index] < metric_values[ndcg_index]:
                        raise ValueError(
                            "Office table violates HR@k >= NDCG@k at row "
                            f"{row_index + 1}"
                        )
                if not (
                    metric_values[0] <= metric_values[1] <= metric_values[2]
                    and metric_values[3] <= metric_values[4] <= metric_values[5]
                ):
                    raise ValueError(
                        "Office table cutoff ordering changed at row "
                        f"{row_index + 1}"
                    )
                records.append(
                    {
                        "dataset": "Office",
                        "tokenizer": tokenizer,
                        "codebook": 256,
                        "depth": depth,
                        "model": model,
                        "family": family,
                        "is_core": True,
                        "source_file": pdf_path.name,
                        "source_page": source_page,
                        "source_pages": "1-3",
                        "source_table_date": "2026-07-21",
                        "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                        "codebook_source_status": (
                            "explicit_heading"
                            if tokenizer == "RQ-VAE"
                            else "inferred_from_continued_section"
                        ),
                        **dict(zip(METRICS, metric_values)),
                    }
                )
                row_index += 1

    table = pd.DataFrame.from_records(records)
    table["tokenizer"] = pd.Categorical(
        table["tokenizer"], TOKENIZER_ORDER, ordered=True
    )

    # Sentinels make a silently shifted extraction fail loudly.
    first = table.iloc[0]
    last = table.iloc[-1]
    if not (first["HR@10"] == 9.19 and last["NDCG@10"] == 8.37):
        raise ValueError("Office table sentinel values do not match the source PDF")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    table = build_office_table(args.pdf.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"office core   {len(table):3d} rows -> {args.output}")


if __name__ == "__main__":
    main()
