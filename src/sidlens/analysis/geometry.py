"""Cluster geometry of residual-quantization codebooks (Experiment 2).

For a quantizer of depth D, items sharing the first d digits form a depth-d
cluster. Two complementary summaries, per the project brief:

  r(C)   = sqrt( mean_{i in C} ||z_i - mu_C||^2 )      RMS radius of one cluster
  r~_d   = median{ r(C) : C in C_d, |C| >= 2 }         typical radius at depth d
  R^2_d  = sum_C sum_{i in C} ||z_i - mu_C||^2
           / sum_i ||z_i - mu||^2                      share of catalogue
                                                       variance still inside
                                                       depth-d clusters

r~_d weights every cluster equally; R^2_d weights every item equally. Together
they separate "most branches tightened" from "a few large clusters dominate".

Singletons are reported separately and excluded from r~_d: their radius is
identically zero, so including them would make the median a statement about how
many singletons a quantizer produces rather than about how tight its clusters
are.

Two caveats that limit what these numbers can claim:

  * All three quantizers are measured in the SHARED 2560-d pre-quantization
    space. RQ-VAE actually quantizes its own encoder latent, which is
    unrecoverable here -- the RQ-VAE checkpoints do not exist on disk. Measuring
    everything in input space is the right call for cross-quantizer
    comparability, but RQ-VAE's numbers are not the geometry of the space it
    was fitted in.
  * MQ is NOT a residual quantizer. Its digits are independent parallel
    quantizations of the full embedding (3cb and 4cb share digit 0 for 17/3105
    items), so a "depth-d cluster" is an intersection of parallel partitions,
    not a refinement of a coarser one. R^2_d is still well defined but does not
    measure progressive refinement, and must not be plotted on the same axis as
    the residual quantizers without saying so.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DepthStats:
    depth: int
    n_clusters: int
    n_singletons: int
    n_multi: int
    largest_cluster: int
    median_radius: float | None      # r~_d, over |C| >= 2
    q1_radius: float | None
    q3_radius: float | None
    mean_radius: float | None
    r2_within: float                 # R^2_d, all items
    r2_multi: float                  # restricted to items still sharing a cluster
    frac_items_in_singletons: float

    def as_dict(self) -> dict:
        return asdict(self)


def cluster_radius(z: np.ndarray) -> float:
    """RMS distance of a cluster's members from its own centroid."""
    if len(z) < 2:
        return 0.0
    mu = z.mean(axis=0)
    return float(np.sqrt(((z - mu) ** 2).sum(axis=1).mean()))


def _within_ss(z: np.ndarray) -> float:
    if len(z) < 2:
        return 0.0
    mu = z.mean(axis=0)
    return float(((z - mu) ** 2).sum())


def _group_by_prefix(codes: np.ndarray, depth: int):
    """Return (sorted row order, group start offsets, group sizes).

    Sorting once and slicing beats building a dict of lists: the geometry has to
    touch every item at every depth for 27 variants, and the Python-level loop
    was the whole cost of the sweep.
    """
    prefix = codes[:, :depth]
    order = np.lexsort(tuple(prefix[:, c] for c in range(depth - 1, -1, -1)))
    sp = prefix[order]
    starts = np.concatenate((
        [0], np.flatnonzero(np.any(sp[1:] != sp[:-1], axis=1)) + 1))
    sizes = np.diff(np.concatenate((starts, [len(order)])))
    return order, starts, sizes


def _within_ss_per_group(z: np.ndarray, order, starts, sizes) -> np.ndarray:
    """Per-group sum of squared deviations from the group centroid.

    Uses the identity  sum_i ||z_i - mu||^2 = sum_i ||z_i||^2 - n*||mu||^2  on
    GLOBALLY CENTERED data. Two details matter:

    * Centering first is what makes the identity safe here. On raw embeddings
      the two terms are dominated by the shared mean and nearly cancel; after
      centering they are the same order as the quantity itself, so the
      subtraction costs a digit or two, not ten. Verified against the explicit
      two-pass computation to ~1e-12 relative.
    * It avoids materializing per-item deviations. The explicit two-pass form
      allocates several (n_items, 2560) float64 arrays per depth -- 64 MB each --
      and this analysis is memory-bandwidth bound, not loop bound. Only the
      (n_groups, 2560) group sums are built.
    """
    zs = z[order]
    sq = np.einsum("ij,ij->i", zs, zs)          # ||z_i||^2, no (n, dim) temporary
    group_sq = np.add.reduceat(sq, starts)
    group_sum = np.add.reduceat(zs, starts, axis=0)
    mu_sq = np.einsum("ij,ij->i", group_sum, group_sum) / sizes
    return np.maximum(group_sq - mu_sq, 0.0)    # clamp float noise on singletons


def depth_stats(codes: np.ndarray, z: np.ndarray, depth: int,
                total_ss: float | None = None) -> DepthStats:
    """Geometry of every depth-`depth` cluster.

    `codes` is (n_items, D) and `z` is (n_items, dim) in the SAME row order.
    """
    if len(codes) != len(z):
        raise ValueError(f"codes {len(codes)} and embeddings {len(z)} misaligned")
    if total_ss is None:
        total_ss = float(((z - z.mean(axis=0)) ** 2).sum())

    if depth == 0:
        # One cluster holding everything: the reference point where R^2 == 1.
        r = cluster_radius(z)
        return DepthStats(
            depth=0, n_clusters=1, n_singletons=0, n_multi=1,
            largest_cluster=len(z),
            median_radius=r, q1_radius=r, q3_radius=r, mean_radius=r,
            r2_within=1.0, r2_multi=1.0,
            frac_items_in_singletons=0.0)

    order, starts, sizes = _group_by_prefix(codes, depth)
    ss = _within_ss_per_group(z, order, starts, sizes)

    multi = sizes >= 2
    radii = np.sqrt(ss[multi] / sizes[multi]) if multi.any() else None
    n_singleton_items = int(sizes[~multi].sum())

    # R^2 restricted to items that still SHARE a cluster.
    #
    # R^2_d collapses toward zero largely because clusters become singletons,
    # and a singleton's within-cluster spread is identically zero. At depth 5 on
    # codebook 256 that is 77-99% of the catalogue, so the headline R^2_d mostly
    # measures how fast a quantizer isolates items -- the "mechanical effect of
    # subdividing a cluster" the brief asks to look past. Restricting BOTH the
    # numerator and the denominator to non-singleton items asks the different
    # question: among items that are still grouped, how much of their spread is
    # within-group rather than between-group?
    if multi.any():
        multi_rows = np.concatenate([
            order[s0:s0 + n] for s0, n in zip(starts[multi], sizes[multi])])
        zm = z[multi_rows]
        zm_dev = zm - zm.mean(axis=0)
        multi_total = float(np.einsum("ij,ij->i", zm_dev, zm_dev).sum())
        r2_multi = float(ss[multi].sum() / multi_total) if multi_total > 0 else 0.0
    else:
        r2_multi = 0.0

    return DepthStats(
        depth=depth,
        n_clusters=len(sizes),
        n_singletons=int((~multi).sum()),
        n_multi=int(multi.sum()),
        largest_cluster=int(sizes.max()),
        median_radius=float(np.median(radii)) if radii is not None else None,
        q1_radius=float(np.percentile(radii, 25)) if radii is not None else None,
        q3_radius=float(np.percentile(radii, 75)) if radii is not None else None,
        mean_radius=float(radii.mean()) if radii is not None else None,
        r2_within=float(ss.sum() / total_ss),
        r2_multi=r2_multi,
        frac_items_in_singletons=n_singleton_items / len(z),
    )


def radius_by_cluster_size(codes: np.ndarray, z: np.ndarray, depth: int,
                           bins=((2, 2), (3, 5), (6, 10), (11, 10**9))) -> list[dict]:
    """Radius distribution split by cluster size.

    A falling median radius can mean every branch tightened, or merely that the
    depth produced more small clusters. Conditioning on size separates the two.
    """
    order, starts, sizes = _group_by_prefix(codes, depth)
    ss = _within_ss_per_group(z, order, starts, sizes)
    radii = np.sqrt(np.divide(ss, sizes, out=np.zeros_like(ss), where=sizes > 0))

    out = []
    for lo, hi in bins:
        sel = (sizes >= lo) & (sizes <= hi)
        r = radii[sel] if sel.any() else None
        out.append({
            "size_min": lo,
            "size_max": None if hi >= 10**9 else hi,
            "n_clusters": int(sel.sum()),
            "n_items": int(sizes[sel].sum()),
            "median_radius": float(np.median(r)) if r is not None else None,
            "q1_radius": float(np.percentile(r, 25)) if r is not None else None,
            "q3_radius": float(np.percentile(r, 75)) if r is not None else None,
        })
    return out


def profile(codes: np.ndarray, z: np.ndarray) -> list[DepthStats]:
    """Geometry at every depth 0..D.

    Centers once here so every depth shares the same centered matrix -- both for
    the numerical reason in `_within_ss_per_group` and to avoid recentering D+1
    times.
    """
    zc = z - z.mean(axis=0)
    total_ss = float(np.einsum("ij,ij->i", zc, zc).sum())
    return [depth_stats(codes, zc, d, total_ss) for d in range(codes.shape[1] + 1)]
