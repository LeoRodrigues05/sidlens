"""Parse data/CollisionRates_GenRec_Paradigms.xlsx into tidy CSVs.

The workbook is a hand-maintained sheet: one collision-rate block at the top
(two datasets side by side) followed by a stack of per-configuration metric
blocks.  This script flattens both into long-format tables that every plotting
script in this folder consumes.

Outputs (under diagrams/data/):
    collisions.csv  dataset, tokenizer, depth, codebook, collision_pct, n_collided
    metrics.csv     tokenizer, codebook, depth, model, family, HR@k / NDCG@k
    joined.csv      metrics.csv left-joined onto the matching collision cell
"""

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
XLSX = REPO / "data" / "CollisionRates_GenRec_Paradigms.xlsx"
OUT = REPO / "diagrams" / "data"

# The sheet writes the parallel/multi-quantiser tokenizer as "MQ" in the
# collision block and "Parallel SID" in the metric blocks.  Same tokenizer.
TOKENIZERS = {
    "rq-vae": "RQ-VAE",
    "rq-kmeans": "RQ-KMeans",
    "mq": "MQ",
    "parallel": "MQ",
}

METRIC_COLS = ["HR@3", "HR@5", "HR@10", "NDCG@3", "NDCG@5", "NDCG@10"]

CELL_RE = re.compile(r"([\d.]+)%\s*\(([\d,]+)\)")
DIGITS_RE = re.compile(r"\((\d)\s*-?\s*digits?\)", re.I)
CODEBOOK_RE = re.compile(r"codebook size\s*=\s*(\d+)", re.I)


def _norm(cell):
    return "" if pd.isna(cell) else str(cell).strip()


def parse_collisions(df):
    """Rows 1-9 hold two side-by-side dataset blocks sharing the same layout."""
    blocks = [("Industrial", 1, 2, {128: 3, 256: 4, 512: 5}),
              ("Office", 8, 9, {128: 10, 256: 11, 512: 12})]
    rows = []
    for dataset, method_col, depth_col, cbk_cols in blocks:
        for r in range(1, 10):
            method = _norm(df.iat[r, method_col]).lower()
            if method not in TOKENIZERS:
                continue
            depth = int(df.iat[r, depth_col])
            for codebook, col in cbk_cols.items():
                m = CELL_RE.search(_norm(df.iat[r, col]))
                if not m:
                    continue
                rows.append({
                    "dataset": dataset,
                    "tokenizer": TOKENIZERS[method],
                    "depth": depth,
                    "codebook": codebook,
                    "collision_pct": float(m.group(1)),
                    "n_collided": int(m.group(2).replace(",", "")),
                })
    return pd.DataFrame(rows)


def _family(name):
    low = name.lower()
    if "diffusion" in low or "diffgrm" in low:
        return "Masked diffusion"
    return "Autoregressive"


def _clean_model(name):
    name = DIGITS_RE.sub("", name).strip()
    return re.sub(r"\s+", " ", name).replace("- ", "-")


def parse_metrics(df):
    """Walk the sheet top-down, tracking the tokenizer/codebook block headers."""
    rows = []
    tokenizer = codebook = None
    for r in range(11, len(df)):
        label = _norm(df.iat[r, 0])
        if not label:
            continue

        low = label.lower()
        # A block header may name the tokenizer, the codebook size, or both.
        header = False
        for key, canon in TOKENIZERS.items():
            if low.startswith(key) and "sid" in low:
                tokenizer, header = canon, True
                break
        m = CODEBOOK_RE.search(label)
        if m:
            codebook, header = int(m.group(1)), True
        if header:
            continue

        m = DIGITS_RE.search(label)
        if not m or tokenizer is None or codebook is None:
            continue

        values = {c: pd.to_numeric(df.iat[r, i + 1], errors="coerce")
                  for i, c in enumerate(METRIC_COLS)}
        if all(pd.isna(v) for v in values.values()):
            continue  # a planned run that has not been filled in yet

        model = _clean_model(label)
        rows.append({
            "dataset": "Industrial",
            "tokenizer": tokenizer,
            "codebook": codebook,
            "depth": int(m.group(1)),
            "model": model,
            "family": _family(model),
            **values,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_excel(XLSX, sheet_name="Sheet1", header=None)
    collisions = parse_collisions(df)
    metrics = parse_metrics(df)

    # Catalogue size is implied by count / rate; useful for coverage figures.
    collisions["n_items"] = (
        collisions["n_collided"] / (collisions["collision_pct"] / 100)
    ).round().astype(int)

    joined = metrics.merge(
        collisions[collisions.dataset == "Industrial"]
        .drop(columns=["dataset"]),
        on=["tokenizer", "depth", "codebook"],
        how="left",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    collisions.to_csv(OUT / "collisions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    joined.to_csv(OUT / "joined.csv", index=False)

    print(f"collisions: {len(collisions)} rows -> {OUT / 'collisions.csv'}")
    print(f"metrics:    {len(metrics)} rows -> {OUT / 'metrics.csv'}")
    print(f"joined:     {len(joined)} rows "
          f"({joined.collision_pct.notna().sum()} with a collision rate)")
    return collisions, metrics, joined


if __name__ == "__main__":
    main()
