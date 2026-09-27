#!/usr/bin/env python
"""Experiment 2b -- refinement beyond mechanical subdivision.

Runs the chance-corrected geometry tools in `sidlens.analysis.refinement` over
all 27 SID variants. CPU only, no model, reads nothing but frozen bytes.

Outputs (to $SIDLENS_WORK/derived/exp2_refinement/):
    refinement.csv     per (variant, depth): R2, null R2, refinement index,
                       neighbour preservation, brand AMI/purity
    digits.csv         per (variant, digit): conditional entropy, effective
                       branching, contribution when deleted
    digit_pairs.csv    per (variant, digit pair): normalized mutual information
    result.json        the above plus the provenance stamp
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.analysis import geometry as G
from sidlens.analysis import refinement as R
from sidlens.data import embeddings as E
from sidlens.data.sids import SidTable, available
from sidlens.provenance import manifest as manifest_mod

CATEGORY = "Industrial_and_Scientific"
KNN = 10


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def brand_labels(category: str, keys: list[str]) -> np.ndarray:
    """Integer brand code per item in `keys` order; -1 where absent."""
    meta = json.loads(
        (paths.FROZEN_DATA / "item_meta" / f"{category}.item.json").read_text())
    from sidlens.data.sids import load_item2id
    item2id = load_item2id(category)
    vocab: dict[str, int] = {}
    out = np.full(len(keys), -1, dtype=np.int64)
    for i, k in enumerate(keys):
        rec = meta.get(str(item2id[k])) or {}
        b = (rec.get("brand") or "").strip()
        if b:
            out[i] = vocab.setdefault(b, len(vocab))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default=CATEGORY)
    ap.add_argument("--knn", type=int, default=KNN)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else paths.DERIVED / "exp2_refinement"
    snapshot = manifest_mod.current_id()
    variants = available("diffgrm")

    # The embedding matrix and its neighbour graph are identical for every
    # variant, so both are built once and shared across all 27.
    ref = SidTable.load(variants[0])
    z = E.load_aligned(args.category, ref.keys)
    zc = z - z.mean(axis=0)
    total_ss = float(np.einsum("ij,ij->i", zc, zc).sum())
    knn = R.knn_indices(zc, args.knn)
    brands = brand_labels(args.category, ref.keys)
    n = len(zc)
    print(f"{n} items, {args.knn}-NN graph, "
          f"{int((brands >= 0).sum())} branded, snapshot {snapshot}\n")

    rows, digit_rows, pair_rows = [], [], []

    for v in variants:
        table = SidTable.load(v)
        if table.keys != ref.keys:
            raise ValueError(f"{v.name}: key order differs from {variants[0].name}")
        codes = table.codes
        common = {
            "snapshot": snapshot, "category": args.category,
            "quantizer": v.quantizer, "n_codebook": v.n_codebook,
            "codebook_size": v.codebook_size, "variant": v.name,
        }

        prof = G.profile(codes, zc)
        for s in prof:
            k = s.n_clusters
            npres = R.neighbour_preservation(codes, knn, s.depth)
            attr = R.attribute_alignment(codes, brands, s.depth)
            rows.append({
                **common, "depth": s.depth, "n_clusters": k,
                "median_radius": s.median_radius,
                "r2_within": s.r2_within,
                "r2_null": R.null_r2(n, k),
                "refinement_index": R.refinement_index(s.r2_within, n, k),
                "knn_observed": npres["observed"],
                "knn_chance": npres["chance"],
                "knn_lift": npres["lift"],
                "brand_ami": attr["ami"],
                "brand_purity": attr["purity"],
            })

        ent = R.digit_conditional_entropy(codes)
        con = R.digit_contribution(codes, zc, total_ss)
        budget = float(np.log2(n))          # bits needed to name one item
        cum = 0.0
        for e, c in zip(ent, con):
            cum += e["cond_entropy_bits"]
            digit_rows.append({**common, **e,
                               **{kk: vv for kk, vv in c.items() if kk != "digit"},
                               "max_entropy_bits": float(np.log2(v.codebook_size)),
                               "cum_entropy_bits": cum,
                               "identity_budget_bits": budget,
                               "budget_share": cum / budget})
        for p in R.digit_pair_mi(codes):
            pair_rows.append({**common, **p})

        last = rows[-1]
        print(f"  {v.name:28s} RI_final={last['refinement_index']:.4f}  "
              f"knn_lift={last['knn_lift']:.4f}  "
              f"brandAMI={last['brand_ami']:.4f}")

    write_csv(out_dir / "refinement.csv", rows)
    write_csv(out_dir / "digits.csv", digit_rows)
    write_csv(out_dir / "digit_pairs.csv", pair_rows)
    (out_dir / "result.json").write_text(json.dumps({
        "snapshot": snapshot, "category": args.category,
        "n_items": n, "knn": args.knn,
        "null_model": (
            "R2_null(n,k) = (n-k)/(n-1), the expected within-cluster variance "
            "share of a random partition into k parts. Validated against "
            "simulation on these embeddings to <= 5e-4 for k in [51, 2910] "
            "over equal and singleton-heavy size profiles."),
        "caveats": (
            "RQ-VAE is measured in the shared input space, not its own encoder "
            "latent, which is unrecoverable. MQ is not residual, so its depth "
            "axis is an intersection of parallel partitions."),
        "refinement": rows, "digits": digit_rows, "digit_pairs": pair_rows,
    }, indent=2))

    print(f"\nwrote {len(rows)} refinement, {len(digit_rows)} digit, "
          f"{len(pair_rows)} pair rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
