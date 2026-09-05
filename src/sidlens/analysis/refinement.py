"""Refinement beyond mechanical subdivision (Experiment 2, extended).

`geometry.py` answers "how tight are the depth-d clusters". That number falls
even for a quantizer that assigns digits at random, because splitting a set into
more parts always lowers within-part variance. The brief asks the harder
question -- how the radius changes "beyond the mechanical effect of subdividing
a cluster" -- and that needs a chance baseline.

The baseline, and why it is analytic
------------------------------------
For a partition of n items into k parts drawn at random, sampling without
replacement gives E[SS_within] = (n - k)/(n - 1) * SS_total, independent of the
part-size profile. So

    R2_null(n, k) = (n - k) / (n - 1)

Verified against simulation on the real embeddings at k = 51 ... 2910, over both
equal and singleton-heavy size profiles: agreement to <= 5e-4. That means no
Monte Carlo is needed, and every variant is scored against exactly the same
curve -- the comparison is between quantizers, not between random seeds.

From it, one number that is comparable across depths, codebook sizes and
quantizers:

    refinement index  =  1 - R2_actual / R2_null

    0   the partition explains no more variance than a random one with the
        same number of clusters -- pure mechanical subdivision
    1   clusters have collapsed to zero internal spread

This is the only quantity here that can be read across the whole 27-variant
grid without renormalizing, because it already divides out both catalogue scale
and cluster count.

The other four tools
-------------------
neighbour preservation   does the SID keep an item's embedding neighbours with
                         it? Chance rate is sum_C n_C(n_C-1) / n(n-1), so the
                         lift is again chance-corrected.
conditional entropy      H(digit_d | prefix) in bits, and 2^H as an effective
                         branching factor -- how much choice actually remains at
                         each digit versus the nominal codebook size.
digit contribution       R2 recomputed with digit d deleted from the SID. A
                         digit whose deletion changes nothing was redundant,
                         which is the scaling question in RQ3 stated as an
                         experiment rather than an intuition.
attribute alignment      adjusted mutual information against brand. AMI is
                         chance-corrected, which matters enormously here: raw
                         MI rises automatically with cluster count, so an
                         uncorrected score would "discover" that deeper digits
                         carry more brand information no matter what they do.
"""

from __future__ import annotations

import numpy as np

from sidlens.analysis.geometry import _group_by_prefix, _within_ss_per_group


# ---------------------------------------------------------------- baseline --

def null_r2(n: int, k: int) -> float:
    """Expected R^2 of a random partition of n items into k parts."""
    if n <= 1:
        return 0.0
    return (n - k) / (n - 1)


def refinement_index(r2_actual: float, n: int, k: int) -> float | None:
    """How far below chance the observed within-cluster spread sits.

    None when the baseline is degenerate (k == n): every part is a singleton, so
    a random partition also has zero spread and the ratio says nothing.
    """
    base = null_r2(n, k)
    if base <= 0:
        return None
    return 1.0 - r2_actual / base


# ------------------------------------------------------ neighbour retention --

def knn_indices(z: np.ndarray, k: int = 10) -> np.ndarray:
    """(n, k) row indices of each item's k nearest neighbours, self excluded.

    Computed once per category and reused across all 27 variants: the embedding
    matrix is identical everywhere, and this is the only O(n^2) step in the
    analysis.
    """
    zf = np.ascontiguousarray(z, dtype=np.float32)
    sq = np.einsum("ij,ij->i", zf, zf)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b; the constant ||a||^2 per row does
    # not affect the ordering, so it is left out.
    d = sq[None, :] - 2.0 * (zf @ zf.T)
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, k, axis=1)[:, :k]


def neighbour_preservation(codes: np.ndarray, knn: np.ndarray,
                           depth: int) -> dict:
    """Share of embedding-space neighbours that land in the same depth-d cluster."""
    n = len(codes)
    if depth == 0:
        return {"depth": 0, "observed": 1.0, "chance": 1.0, "lift": None}

    pref = codes[:, :depth]
    # Map each distinct prefix to an integer label, then compare labels.
    _, labels = np.unique(pref, axis=0, return_inverse=True)
    same = labels[:, None] == labels[knn]
    observed = float(same.mean())

    sizes = np.bincount(labels).astype(np.float64)
    chance = float((sizes * (sizes - 1)).sum() / (n * (n - 1)))
    lift = None if chance >= 1.0 else (observed - chance) / (1.0 - chance)
    return {"depth": depth, "observed": observed, "chance": chance, "lift": lift}


# ----------------------------------------------------------- digit entropy --

def _entropy_bits(counts: np.ndarray) -> float:
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def digit_conditional_entropy(codes: np.ndarray) -> list[dict]:
    """H(digit_d | prefix_<d) and the effective branching factor 2^H.

    Digit 1 is unconditional. A digit whose effective branching is far below the
    codebook size is not using the codebook it was given -- and, unlike raw
    utilization, this is weighted by how often each branch is actually taken.

    Interpret against the catalogue's information budget, not against the
    codebook size. Distinguishing 3105 items needs only log2(3105) = 11.6 bits
    in total, so once the earlier digits have spent that, the later ones have
    nothing left to encode and their conditional entropy must fall to zero --
    for any quantizer, however good. The sum of these entropies is the SID's
    total information content; comparing that sum to 11.6 bits says how much of
    the budget each digit actually bought. A low entropy at digit 5 is only
    evidence against the quantizer if the budget was not already spent.
    """
    n, D = codes.shape
    out = []
    for d in range(D):
        if d == 0:
            h = _entropy_bits(np.bincount(codes[:, 0]))
        else:
            order, starts, sizes = _group_by_prefix(codes, d)
            col = codes[order, d]
            h = 0.0
            for s0, m in zip(starts, sizes):
                h += (m / n) * _entropy_bits(np.bincount(col[s0:s0 + m]))
        out.append({
            "digit": d + 1,
            "cond_entropy_bits": h,
            "effective_branching": float(2.0 ** h),
        })
    return out


def digit_pair_mi(codes: np.ndarray) -> list[dict]:
    """Redundancy between every pair of digits, chance-corrected.

    Near zero is what a well-behaved quantizer should show: digits carrying
    independent information. A high value means two digits partly encode the
    same distinction -- redundancy that costs vocabulary and decoding steps
    without buying resolution.

    Read `ami`, not `nmi`. The plug-in MI estimator is badly biased at this
    sample size: 3105 items over a 256 x 256 contingency table leaves most cells
    empty, and the bias is of the same order as the quantity. Uncorrected NMI
    rates all three quantizers at 0.40-0.62 and hides the actual result, which
    is that MQ's digits are uniformly redundant (AMI ~ 0.32 on every pair, as
    parallel quantization of one vector must be) while the residual quantizers
    sit at 0.01-0.16. `nmi` and `mi_bits` are kept only so that gap stays
    visible. Verified against a label-shuffled null: AMI ~ 0.000.
    """
    from sklearn.metrics import adjusted_mutual_info_score
    n, D = codes.shape
    out = []
    for a in range(D):
        for b in range(a + 1, D):
            ca, cb = codes[:, a], codes[:, b]
            # Exact integer contingency table. histogram2d agrees here but bins
            # by value range, which is only incidentally right for dense
            # integer codes and silently wrong for sparse ones.
            kb = int(cb.max()) + 1
            joint = np.bincount(ca.astype(np.int64) * kb + cb,
                                minlength=(int(ca.max()) + 1) * kb
                                ).reshape(int(ca.max()) + 1, kb)
            pj = joint / n
            pa = pj.sum(axis=1, keepdims=True)
            pb = pj.sum(axis=0, keepdims=True)
            nz = pj > 0
            mi = float((pj[nz] * np.log2(pj[nz] / (pa @ pb)[nz])).sum())
            ha, hb = _entropy_bits(np.bincount(ca)), _entropy_bits(np.bincount(cb))
            denom = min(ha, hb)
            out.append({
                "digit_a": a + 1, "digit_b": b + 1,
                "ami": float(adjusted_mutual_info_score(ca, cb)),
                "mi_bits": mi,
                "nmi": float(mi / denom) if denom > 0 else 0.0,
            })
    return out


# ------------------------------------------------------- digit contribution --

def digit_contribution(codes: np.ndarray, z: np.ndarray,
                       total_ss: float) -> list[dict]:
    """R^2 recomputed with digit d deleted, against the full-depth R^2.

    Reported as the refinement index of the reduced SID, so the comparison is
    not confounded by the reduced SID having fewer clusters -- deleting a digit
    always merges clusters, which always raises raw R^2.
    """
    n, D = codes.shape
    order, starts, sizes = _group_by_prefix(codes, D)
    full_r2 = float(_within_ss_per_group(z, order, starts, sizes).sum() / total_ss)
    full_k = len(sizes)
    full_ri = refinement_index(full_r2, n, full_k)

    out = []
    for d in range(D):
        keep = [c for c in range(D) if c != d]
        sub = np.ascontiguousarray(codes[:, keep])
        o2, s2, z2 = _group_by_prefix(sub, D - 1)
        r2 = float(_within_ss_per_group(z, o2, s2, z2).sum() / total_ss)
        ri = refinement_index(r2, n, len(z2))
        out.append({
            "digit": d + 1,
            "r2_without": r2,
            "n_clusters_without": int(len(z2)),
            "refinement_without": ri,
            "r2_delta": r2 - full_r2,
            "refinement_delta": None if (ri is None or full_ri is None)
                                else full_ri - ri,
        })
    return out


# -------------------------------------------------------- attribute alignment --

def attribute_alignment(codes: np.ndarray, labels: np.ndarray,
                        depth: int) -> dict:
    """Chance-corrected agreement between depth-d clusters and an attribute.

    `labels` is an integer code per item, negative where the attribute is
    missing; those rows are dropped. Purity is reported alongside AMI because
    the two disagree in an informative way: purity rises mechanically with
    cluster count, AMI does not.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    valid = labels >= 0
    lab = labels[valid]
    if depth == 0:
        clus = np.zeros(valid.sum(), dtype=np.int64)
    else:
        _, clus = np.unique(codes[valid, :depth], axis=0, return_inverse=True)

    # purity: per cluster, the count of its most common attribute value
    order = np.argsort(clus, kind="stable")
    cs, ls = clus[order], lab[order]
    bounds = np.concatenate(([0], np.flatnonzero(cs[1:] != cs[:-1]) + 1,
                             [len(cs)]))
    hits = sum(np.bincount(ls[a:b]).max() for a, b in zip(bounds[:-1], bounds[1:]))
    return {
        "depth": depth,
        "n_labelled": int(valid.sum()),
        "ami": float(adjusted_mutual_info_score(clus, lab)),
        "purity": float(hits / len(cs)),
    }
