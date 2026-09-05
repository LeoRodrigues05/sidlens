"""Experiment 1 -- what each SID digit adds, measured on the item side.

The brief asks, at each position d, what digit d adds once digits 1..d-1 are
known. On the item side that is a question about the prefix hierarchy: take the
depth-(d-1) cluster, split it by digit d, and describe how the children differ
from the parent and from each other.

Nothing here loads a model. This is the substrate the probe/patching half of
Experiment 1 gets compared against: if a probe reads an attribute off digit d
that the items under digit d do not actually share, the probe is reading the
history, not the digit.

One caveat is structural and is carried on every row. For RQ-VAE and RQ-KMeans
the depth axis is a genuine refinement -- digit d quantizes the residual left by
digits 1..d-1, so a depth-d cluster nests inside its depth-(d-1) parent. MQ
quantizes the full embedding independently at every digit, so its depth-d
"cluster" is an intersection of parallel partitions. Both are computed; only the
residual ones may be read as "digit d refines what digit d-1 said".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RESIDUAL = {"rqvae": True, "rqkmeans": True, "MQ": False}


@dataclass
class CodeCell:
    """One (prefix, digit-value) cell of the atlas."""
    variant: str
    quantizer: str
    residual: bool
    depth: int                      # length of the prefix, 1-based
    prefix: tuple[int, ...]         # the full prefix INCLUDING this digit
    parent: tuple[int, ...]         # prefix[:-1]
    digit: int                      # which position this cell fixes (depth-1)
    code: int                       # the value at that position
    n_items: int
    n_children: int                 # distinct next-digit values under this cell
    radius: float                   # RMS distance to centroid, embedding space
    parent_radius: float
    shrink: float                   # radius / parent_radius; 1.0 = no tightening
    top_brand: str
    top_brand_count: int
    brand_purity: float
    exemplars: list[str] = field(default_factory=list)   # ASINs nearest centroid
    members: list[str] = field(default_factory=list)     # all ASINs, cluster order


def _rms_radius(z: np.ndarray) -> float:
    """Root-mean-square distance from the cluster centre.

    Matches the brief's r(C); a singleton is exactly 0 and is kept as 0 rather
    than dropped, because "this code addresses one item" is a finding.
    """
    if len(z) == 0:
        return float("nan")
    return float(np.sqrt(((z - z.mean(0)) ** 2).sum(1).mean()))


def _nearest_to_centroid(z: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k items closest to the centroid.

    These are the exemplars a human reads to name a code. Picking by proximity
    to the centre rather than by catalogue order matters: the first-listed item
    of a 70-item cell is an arbitrary item, the central one is the cell's claim
    about itself.
    """
    d = ((z - z.mean(0)) ** 2).sum(1)
    return np.argsort(d, kind="stable")[:k]


def build(table, z: np.ndarray, keys: list[str], meta: dict,
          max_depth: int | None = None, n_exemplars: int = 6,
          keep_members: bool = True) -> list[CodeCell]:
    """One CodeCell per (prefix) for every depth in 1..n_codebook.

    `z` must be row-aligned with `keys`, which must be `table.keys`.
    """
    q = table.variant.quantizer
    residual = RESIDUAL[q]
    max_depth = max_depth or table.variant.n_codebook
    pos = {k: i for i, k in enumerate(keys)}

    # Parent radii, indexed by prefix; depth 0 is the whole catalogue.
    radius_of: dict[tuple[int, ...], float] = {(): _rms_radius(z)}
    cells: list[CodeCell] = []

    for depth in range(1, max_depth + 1):
        clusters = table.clusters(depth)
        children: dict[tuple[int, ...], set[int]] = {}
        for pfx in clusters:
            if depth < table.variant.n_codebook:
                children.setdefault(pfx, set())
        if depth < table.variant.n_codebook:
            for pfx_next in table.clusters(depth + 1):
                children.setdefault(pfx_next[:depth], set()).add(pfx_next[depth])

        for pfx, asins in sorted(clusters.items()):
            idx = np.array([pos[a] for a in asins])
            zc = z[idx]
            r = _rms_radius(zc)
            radius_of[pfx] = r
            pr = radius_of.get(pfx[:-1], float("nan"))

            brands = [meta[a].brand for a in asins if a in meta and meta[a].brand]
            if brands:
                vals, counts = np.unique(brands, return_counts=True)
                j = int(counts.argmax())
                top_brand, top_n = str(vals[j]), int(counts[j])
            else:
                top_brand, top_n = "", 0

            ex = [asins[i] for i in _nearest_to_centroid(zc, n_exemplars)]
            cells.append(CodeCell(
                variant=table.variant.name,
                quantizer=q,
                residual=residual,
                depth=depth,
                prefix=pfx,
                parent=pfx[:-1],
                digit=depth - 1,
                code=pfx[-1],
                n_items=len(asins),
                n_children=len(children.get(pfx, ())),
                radius=r,
                parent_radius=pr,
                shrink=(r / pr) if pr and np.isfinite(pr) and pr > 0 else float("nan"),
                top_brand=top_brand,
                top_brand_count=top_n,
                brand_purity=(top_n / len(asins)) if asins else 0.0,
                exemplars=ex,
                members=list(asins) if keep_members else [],
            ))
    return cells


def digit_summary(cells: list[CodeCell]) -> list[dict]:
    """Per-digit rollup: how much of the work each position is doing.

    `median_shrink` is the headline. A digit whose cells are no tighter than
    their parents (shrink near 1) is not refining anything in this space; a
    digit with shrink well under 1 is.
    """
    out = []
    depths = sorted({c.depth for c in cells})
    n_items_total = sum(c.n_items for c in cells if c.depth == depths[0])
    for d in depths:
        sub = [c for c in cells if c.depth == d]
        # Singletons have radius exactly 0 and shrink exactly 0. Including them
        # drives every median to 0 as soon as half the cells are singletons,
        # which says "the catalogue ran out of items", not "this digit refines".
        # Every dispersion statistic is therefore over multi-item cells, and the
        # singleton share is reported alongside so the two are never confused.
        multi = [c for c in sub if c.n_items >= 2]
        shrinks = np.array([c.shrink for c in multi if np.isfinite(c.shrink)])
        n_singleton_items = sum(c.n_items for c in sub if c.n_items == 1)
        out.append({
            "variant": sub[0].variant,
            "quantizer": sub[0].quantizer,
            "residual": sub[0].residual,
            "digit": d - 1,
            "depth": d,
            "n_cells": len(sub),
            "codes_used": len({c.code for c in sub}),
            "n_multi_cells": len(multi),
            "n_singleton_cells": sum(1 for c in sub if c.n_items == 1),
            "singleton_item_share": round(n_singleton_items / n_items_total, 4),
            "largest_cell": max(c.n_items for c in sub),
            "median_cell_size": float(np.median([c.n_items for c in sub])),
            "median_multi_cell_size": float(np.median([c.n_items for c in multi])) if multi else 0.0,
            "median_radius": float(np.median([c.radius for c in multi])) if multi else 0.0,
            "median_shrink": float(np.median(shrinks)) if len(shrinks) else float("nan"),
            "mean_brand_purity": float(np.mean([c.brand_purity for c in multi])) if multi else 0.0,
        })
    return out
