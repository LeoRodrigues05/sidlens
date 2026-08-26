#!/usr/bin/env python3
"""Read next-2-item results into the doc's table layout.

Two sources, one metric convention. Both sides score a predicted pair with
partial credit (both items right = 1.0, one right = 0.5, neither = 0.0) and
report HR@k = max relevance in the top k, NDCG@k = best_rel / log2(pos + 2):

    AR (Qwen)   sweep/two-item/metrics/<variant>.json      written by calc_two_item.py
    Diffusion   diffgrm_2item/<category>/<variant>.txt     "Test Results: OrderedDict([...])"

Diffusion logs carry two beam modes; the doc's numbers are the unsuffixed
"confidence" set, so keys ending in _random are dropped.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP

WORK = "/l/users/leo.rodrigues/onediffrec"
COLUMNS = ("HR@3", "HR@5", "HR@10", "NDCG@3", "NDCG@5", "NDCG@10")
METHOD_LABEL = {"rqvae": "RQ-VAE", "rqkmeans": "RQ-Kmeans", "MQ": "Parallel"}


def pct(x) -> str:
    """Percent to 2dp, half-up - Python's round() would give banker's rounding."""
    return str(Decimal(str(float(x) * 100)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def read_diffusion(path: str) -> dict | None:
    """Pull the final "Test Results" mapping out of a run log."""
    with open(path, "r", errors="replace") as fh:
        blob = fh.read()
    hits = re.findall(r"Test Results:\s*OrderedDict\((\[.*?\])\)", blob, re.S)
    if not hits:
        return None
    # np.float64(...) is not literal syntax; unwrap before parsing.
    raw = re.sub(r"np\.float64\(([^)]*)\)", r"\1", hits[-1])
    pairs = dict(ast.literal_eval(raw))
    out = {}
    for key in COLUMNS:
        if key in pairs:
            out[key] = pairs[key]
        # next-item logs name them recall@k/ndcg@k; two-item logs use HR@k/NDCG@k
        elif key.lower().replace("hr@", "recall@") in pairs:
            out[key] = pairs[key.lower().replace("hr@", "recall@")]
    return out or None


def read_ar(path: str) -> dict | None:
    with open(path) as fh:
        rec = json.load(fh)
    metrics = rec.get("metrics") or {}
    # calc_two_item reports percentages already; diffusion reports fractions.
    return {k: metrics[k] / 100 for k in COLUMNS if k in metrics} or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--sizes", default="128,256,512")
    args = ap.parse_args()

    diff_dir = os.path.join(WORK, "diffgrm_2item", args.category)
    ar_dir = os.path.join(WORK, "sweep", "two-item", "metrics")

    for size in [int(s) for s in args.sizes.split(",")]:
        rows = []
        for method in ("rqvae", "rqkmeans", "MQ"):
            for digits in (3, 4, 5):
                ar_path = os.path.join(
                    ar_dir, f"twoitem__{args.category}__{method}__{digits}cb__{size}.json"
                )
                ar = read_ar(ar_path) if os.path.exists(ar_path) else None

                dpath = os.path.join(diff_dir, f"{method}_{digits}codebook_{size}.txt")
                diff = read_diffusion(dpath) if os.path.exists(dpath) else None

                for label, vals in (
                    (f"Qwen2.5-1.5B ({digits}-digits)", ar),
                    (f"Inside Diff, Outside AR ({digits}-digits)", diff),
                ):
                    cells = [pct(vals[c]) if vals and c in vals else "-" for c in COLUMNS]
                    rows.append((METHOD_LABEL[method], label, cells))

        done = sum(1 for _, _, c in rows if c[0] != "-")
        print(f"\n### Codebook size = {size}   ({done}/{len(rows)} cells)\n")
        print("| SID | Model | " + " | ".join(COLUMNS) + " |")
        print("|---|---|" + "---|" * len(COLUMNS))
        prev = None
        for sid, label, cells in rows:
            shown = f"**{sid}**" if sid != prev else ""
            prev = sid
            print(f"| {shown} | {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
