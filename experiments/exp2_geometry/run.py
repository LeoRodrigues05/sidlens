#!/usr/bin/env python
"""Experiment 2 -- RQ cluster geometry at every digit.

Runs entirely offline from the frozen substrate: the (3105, 2560)
pre-quantization embeddings and the 27 SID tables. No model, no GPU.

Outputs (to $SIDLENS_WORK/derived/exp2_geometry/):
    depth_stats.csv        one row per (variant, depth)
    size_breakdown.csv     radius distribution conditioned on cluster size
    collisions.csv         collision + codebook-utilization per variant
    result.json            the above plus the provenance stamp

Every row carries the snapshot id, so a number can always be traced to the bytes
it came from.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sidlens import paths
from sidlens.analysis import geometry as G
from sidlens.data import embeddings as E
from sidlens.data.sids import SidTable, available
from sidlens.provenance import manifest as manifest_mod

CATEGORY = "Industrial_and_Scientific"
# MQ quantizes the full embedding independently per digit, so its depth axis is
# an intersection of parallel partitions rather than a refinement hierarchy.
NESTED = {"rqvae": False, "rqkmeans": True, "MQ": False}
RESIDUAL = {"rqvae": True, "rqkmeans": True, "MQ": False}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default=CATEGORY)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else paths.DERIVED / "exp2_geometry"
    snapshot = manifest_mod.current_id()
    variants = available("diffgrm")

    depth_rows, size_rows, coll_rows, util_rows = [], [], [], []

    for v in variants:
        table = SidTable.load(v)
        z = E.load_aligned(args.category, table.keys)
        prof = G.profile(table.codes, z)

        common = {
            "snapshot": snapshot,
            "category": args.category,
            "quantizer": v.quantizer,
            "n_codebook": v.n_codebook,
            "codebook_size": v.codebook_size,
            "variant": v.name,
            "residual": RESIDUAL[v.quantizer],
            "nested": NESTED[v.quantizer],
        }

        for s in prof:
            depth_rows.append({**common, **s.as_dict()})

        for d in range(1, v.n_codebook + 1):
            for b in G.radius_by_cluster_size(table.codes, z, d):
                size_rows.append({**common, "depth": d, **b})

        cs = table.collision_stats
        util = table.codebook_utilization
        for u in util:
            util_rows.append({**common, **u})
        coll_rows.append({
            **common,
            **cs,
            "mean_utilization": sum(u["utilization"] for u in util) / len(util),
            "digit0_utilization": util[0]["utilization"],
            "digit0_max_count": util[0]["max_count"],
            "mean_entropy_bits": sum(u["entropy_bits"] for u in util) / len(util),
            "max_entropy_bits": util[0]["max_entropy_bits"],
            "sem_ids_source": str(table.source),
        })

        r2 = prof[-1].r2_within
        print(f"  {v.name:28s} R2_final={r2:.4f}  "
              f"collision={cs['collision_rate']:6.2%}  "
              f"unique={cs['unique_sids']:5d}")

    write_csv(out_dir / "depth_stats.csv", depth_rows)
    write_csv(out_dir / "size_breakdown.csv", size_rows)
    write_csv(out_dir / "collisions.csv", coll_rows)
    write_csv(out_dir / "utilization.csv", util_rows)
    (out_dir / "result.json").write_text(json.dumps({
        "snapshot": snapshot,
        "category": args.category,
        "n_variants": len(variants),
        "embedding_space": {
            "source": "Qwen3-Embedding-4B, mean-pooled over [title, description]",
            "dim": E.EMB_DIM,
            "pca": False,
            "l2_normalized": False,
            "caveat": (
                "All quantizers measured in this shared pre-quantization space. "
                "RQ-VAE actually quantizes its own encoder latent, which is "
                "unrecoverable (no RQ-VAE checkpoints exist on disk), so its "
                "numbers are not the geometry of the space it was fitted in."),
        },
        "mq_caveat": (
            "MQ is not a residual quantizer: its digits independently quantize "
            "the full embedding, so depth-d clusters are intersections of "
            "parallel partitions, not refinements. R2_d is well defined but does "
            "not measure progressive refinement and must not share an axis with "
            "the residual quantizers without that note."),
        "depth_stats": depth_rows,
        "collisions": coll_rows,
        "utilization": util_rows,
    }, indent=2))

    print(f"\nwrote {len(depth_rows)} depth, {len(size_rows)} size, "
          f"{len(coll_rows)} collision, {len(util_rows)} utilization rows -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
