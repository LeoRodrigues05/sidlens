#!/usr/bin/env python
"""Structure exp3 -- why do items share a Semantic ID?

Tests the hypotheses declared in ``protocol.md`` (H1 processing step, H2
near-duplicates, H3 shared residual codebooks, H4 anisotropy, H5 redundant
parallel digits) on the 27 frozen Industrial tables, and with RQ-KMeans and
parallel-digit refits. CPU only; no recommender is run.

Traps this script guards against:

* faiss k-means results depend on the OpenMP thread count. The archived
  RQ-KMeans tables reproduce only at 96 threads, so every fit runs at
  ``--threads`` (default 96) and Part C refuses to continue unless the
  native refit reproduces every nested archived table exactly.
* SID tables are ASIN-keyed and the embedding is item-id-indexed. Everything
  here works in item-id order; ``load_codes`` is the one bridge.
* ``tokens2item`` is lossy; collisions are computed from the tables directly.
* The frozen RQ-KMeans 5x512 table is NOT the nested extension of 3x512 /
  4x512 (it is a separate, balanced fit), so it is excluded from the native
  reproduction check and from the nested depth trend.
* MQ's build recipe is unknown. MQ refits (Part E) are counterfactuals and
  are labelled exploratory.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from sidlens import paths
from sidlens.data import meta as meta_mod
from sidlens.data.sids import SidTable, available, load_item2id
from sidlens.provenance.hashing import sha256_file

CATEGORY = "Industrial_and_Scientific"
SEEDS = (1234, 1, 2, 3, 4)
NATIVE_SEED = 1234
PERM_SEED = 20261003
WIDTHS = (128, 256, 512)
FIT_LEVELS = 5
KNN_ND = 5            # near-duplicate neighbourhood
KNN_COST = 10         # neighbour agreement used as the semantic cost of a refit
JACCARD_ND = 0.5
PQ_DIM = 2400         # divisible by 3, 4 and 5
RQVAE_SK_EPS = 0.003  # generate_indices.py fallback for the last level
RQVAE_SK_ITERS = 50   # rqvae.py --sk_iters default
RQVAE_MAX_ROUNDS = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- inputs --
def load_inputs():
    emb_path = paths.FROZEN_DATA / "embeddings" / f"{CATEGORY}.emb-qwen-td.npy"
    raw = np.load(emb_path)
    if raw.dtype != np.float16 or raw.shape != (3105, 2560):
        raise ValueError(f"unexpected embedding {raw.dtype} {raw.shape}")
    item2id = load_item2id(CATEGORY)
    n = raw.shape[0]
    if sorted(item2id.values()) != list(range(n)):
        raise ValueError("item2id does not cover embedding rows 0..N-1 exactly")
    metas = meta_mod.load(CATEGORY)
    titles = [""] * n
    brands = [""] * n
    for asin, m in metas.items():
        titles[m.item_id] = m.title
        brands[m.item_id] = m.brand
    return raw, item2id, titles, brands, emb_path


def load_codes(name: str, item2id: dict[str, int], n: int) -> np.ndarray:
    """(N, D) codes in item-id order; refuses a table that misses any item."""
    t = SidTable.load(name)
    codes = np.full((n, t.variant.n_codebook), -1, dtype=np.int64)
    for asin, c in t.asin2codes.items():
        codes[item2id[asin]] = c
    if (codes < 0).any():
        raise ValueError(f"{name}: {(codes < 0).any(1).sum()} items without SID")
    return codes


# ------------------------------------------------------------ primitives --
def keys_of(codes: np.ndarray, K: int) -> np.ndarray:
    """Mixed-radix int64 key per row (512^5 < 2^63, so this cannot overflow)."""
    k = np.zeros(len(codes), dtype=np.int64)
    for d in range(codes.shape[1]):
        k = k * K + codes[:, d]
    return k


def collided(keys: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    return cnt[inv] > 1


def n_distinct(keys: np.ndarray) -> int:
    return len(np.unique(keys))


def rates(codes: np.ndarray, K: int) -> dict:
    k = keys_of(codes, K)
    m = collided(k)
    _, cnt = np.unique(k, return_counts=True)
    return {"collision_rate": float(m.mean()),
            "collision_rate_old": 1.0 - len(cnt) / len(k),
            "max_bucket": int(cnt.max()),
            "n_buckets_multi": int((cnt > 1).sum())}


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def r2_by_groups(X: np.ndarray, keys: np.ndarray, total_ss: float) -> float:
    _, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv)
    sums = np.zeros((len(cnt), X.shape[1]))
    np.add.at(sums, inv, X)
    between = ((sums ** 2).sum(1) / cnt).sum() - (X.sum(0) ** 2).sum() / len(X)
    return float(between / total_ss)


def knn_agreement(knn: np.ndarray, labels: np.ndarray) -> float:
    return float((labels[knn] == labels[:, None]).mean())


# --------------------------------------------------------------- Part A --
def part_a(raw, titles, brands, out: Path):
    log("Part A: catalogue")
    X = raw.astype(np.float64)
    n = len(X)
    D2 = sqdist(X, X)
    np.fill_diagonal(D2, np.inf)
    dist = np.sqrt(D2)
    knn_sorted = np.argsort(D2, axis=1, kind="stable")
    knn10 = knn_sorted[:, :KNN_COST]
    knn5 = knn_sorted[:, :KNN_ND]
    iu = np.triu_indices(n, 1)
    all_pair = np.sort(dist[iu])
    nn1 = dist[np.arange(n), knn_sorted[:, 0]]

    # identical stored inputs
    _, ident_inv, ident_cnt = np.unique(raw, axis=0, return_inverse=True,
                                        return_counts=True)
    ident_inv = ident_inv.ravel()
    ident_item = ident_cnt[ident_inv] > 1

    # spectrum
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2 / (n - 1)
    share = ev / ev.sum()
    norms = np.linalg.norm(X, axis=1)

    # near-duplicate pairs from the catalogue alone
    def fold(b):
        return re.sub(r"\s+", " ", b.lower()).strip()
    bfold = [fold(b) for b in brands]
    toks = [set(re.findall(r"[a-z0-9]+", t.lower())) for t in titles]
    cand = set()
    for i in range(n):
        for j in knn5[i]:
            cand.add((min(i, int(j)), max(i, int(j))))
    nd_pairs = []
    for i, j in sorted(cand):
        ident = ident_inv[i] == ident_inv[j]
        jac = (len(toks[i] & toks[j]) / len(toks[i] | toks[j])
               if (toks[i] | toks[j]) else 0.0)
        same_brand = bool(bfold[i]) and bfold[i] == bfold[j]
        if ident or (same_brand and jac >= JACCARD_ND):
            nd_pairs.append((i, j, float(dist[i, j]), jac, bool(ident)))
    # identical pairs that are not within each other's 5-NN (large groups)
    groups = defaultdict(list)
    for i, g in enumerate(ident_inv):
        if ident_cnt[g] > 1:
            groups[int(g)].append(i)
    have = {(a, b) for a, b, *_ in nd_pairs}
    for members in groups.values():
        for a in members:
            for b in members:
                if a < b and (a, b) not in have:
                    nd_pairs.append((a, b, 0.0, 1.0, True))
    nd_adj = defaultdict(set)
    for a, b, *_ in nd_pairs:
        nd_adj[a].add(b)
        nd_adj[b].add(a)

    pd.DataFrame(nd_pairs, columns=["i", "j", "distance", "title_jaccard",
                                    "identical_input"]).assign(
        title_i=lambda d: [titles[i][:90] for i in d.i],
        title_j=lambda d: [titles[j][:90] for j in d.j],
    ).to_csv(out / "partA_near_duplicate_pairs.csv", index=False)

    summary = {
        "n_items": n,
        "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
        "var_share_top10": float(share[:10].sum()),
        "var_share_top64": float(share[:64].sum()),
        "var_share_top256": float(share[:256].sum()),
        "var_share_top2400": float(share[:PQ_DIM].sum()),
        "n_pcs_95pct": int(np.searchsorted(np.cumsum(share), 0.95) + 1),
        "norm_min": float(norms.min()), "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "norm_cv": float(norms.std() / norms.mean()),
        "mean_vector_norm": float(np.linalg.norm(X.mean(0))),
        "rms_radius": float(np.sqrt((Xc ** 2).sum(1).mean())),
        "n_identical_input_items": int(ident_item.sum()),
        "n_identical_input_groups": len(groups),
        "identical_group_sizes": sorted(len(v) for v in groups.values()),
        "n_near_duplicate_pairs": len(nd_pairs),
        "n_items_in_near_duplicate_pairs": len(nd_adj),
        "share_items_in_near_duplicate_pairs": len(nd_adj) / n,
        "pair_distance_p01_p50_p99": [float(np.quantile(all_pair, q))
                                      for q in (0.01, 0.5, 0.99)],
        "nn1_distance_p10_p50_p90": [float(np.quantile(nn1, q))
                                     for q in (0.1, 0.5, 0.9)],
        "n_empty_brand": int(sum(1 for b in bfold if not b)),
    }
    (out / "partA_catalogue.json").write_text(json.dumps(summary, indent=2))
    cat = {
        "X": X, "dist": dist, "all_pair": all_pair, "knn10": knn10,
        "ident_inv": ident_inv, "ident_item": ident_item, "nd_adj": nd_adj,
        "nd_pairs": nd_pairs, "Xc": Xc, "total_ss": float((Xc ** 2).sum()),
        "titles": titles, "bfold": bfold,
    }
    return summary, cat


# --------------------------------------------------------------- Part B --
def null_n1(codes, K, rng, n_perm):
    out = np.empty(n_perm)
    for p in range(n_perm):
        sh = np.stack([codes[rng.permutation(len(codes)), d]
                       for d in range(codes.shape[1])], 1)
        out[p] = collided(keys_of(sh, K)).mean()
    return out


def merge_ratio(codes, K, l, rng, n_perm):
    """Excess items after adding digit l, observed / mean under N2."""
    n = len(codes)
    pref = keys_of(codes[:, :l], K)
    obs = n - n_distinct(pref * K + codes[:, l])
    null = np.empty(n_perm)
    for p in range(n_perm):
        null[p] = n - n_distinct(pref * K + codes[rng.permutation(n), l])
    return obs, float(null.mean()), (obs / null.mean() if null.mean() > 0 else np.nan)


def balanced(counts: np.ndarray, n: int, K: int) -> bool:
    return bool((counts > 0).all() and counts.min() >= n // K
                and counts.max() <= -(-n // K))


def nd_shares(codes, K, cat):
    """Near-duplicate share of collided items, against multi-item parents."""
    nd_adj = cat["nd_adj"]
    D = codes.shape[1]
    full = keys_of(codes, K)
    par = keys_of(codes[:, :D - 1], K)
    coll = collided(full)
    multi_par = collided(par)
    def has_partner(mask, key):
        out = np.zeros(len(codes), bool)
        for i in np.flatnonzero(mask):
            out[i] = any(key[j] == key[i] for j in nd_adj.get(i, ()))
        return out
    nd_coll = has_partner(coll, full)
    nd_par = has_partner(multi_par, par)
    n_pairs = len(cat["nd_pairs"])
    pairs_coll = sum(1 for a, b, *_ in cat["nd_pairs"] if full[a] == full[b])
    return {
        "nd_share_collided": float(nd_coll[coll].mean()) if coll.any() else np.nan,
        "nd_share_multi_parent": float(nd_par[multi_par].mean()) if multi_par.any() else np.nan,
        "nd_pair_recall": pairs_coll / n_pairs if n_pairs else np.nan,
        "n_collided": int(coll.sum()),
    }


def pair_distance_percentiles(codes, K, cat):
    full = keys_of(codes, K)
    _, inv, cnt = np.unique(full, return_inverse=True, return_counts=True)
    dist, all_pair = cat["dist"], cat["all_pair"]
    pct_item = []
    for g in np.flatnonzero(cnt > 1):
        mem = np.flatnonzero(inv == g)
        sub = dist[np.ix_(mem, mem)]
        np.fill_diagonal(sub, np.inf)
        pct_item.extend(np.searchsorted(all_pair, sub.min(1), side="right")
                        / len(all_pair))
    if not pct_item:
        return {"closest_mate_pct_median": np.nan, "share_mate_in_closest_0.1pct": np.nan}
    pct_item = np.array(pct_item)
    return {"closest_mate_pct_median": float(np.median(pct_item)),
            "share_mate_in_closest_0.1pct": float((pct_item <= 0.001).mean())}


def radius_spearman(codes, K, cat):
    X = cat["X"]
    D = codes.shape[1]
    par = keys_of(codes[:, :D - 1], K)
    coll = collided(keys_of(codes, K))
    _, inv, cnt = np.unique(par, return_inverse=True, return_counts=True)
    radii, shares = [], []
    for g in np.flatnonzero(cnt > 1):
        mem = np.flatnonzero(inv == g)
        Z = X[mem] - X[mem].mean(0)
        radii.append(np.sqrt((Z ** 2).sum(1).mean()))
        shares.append(coll[mem].mean())
    if len(radii) < 10 or np.ptp(shares) == 0:
        return np.nan, len(radii)
    return float(spearmanr(radii, shares).statistic), len(radii)


def mean_pairwise_nmi(codes):
    D = codes.shape[1]
    vals = [normalized_mutual_info_score(codes[:, a], codes[:, b])
            for a in range(D) for b in range(a + 1, D)]
    return float(np.mean(vals))


def part_b(tables, cat, out, n_perm):
    log("Part B: archived tables")
    rng = np.random.default_rng(PERM_SEED)
    rows, lvl_rows, bucket_rows = [], [], []
    n = len(cat["X"])
    for name, codes in tables.items():
        q, D, K = name.split("_")[0], codes.shape[1], int(name.split("_")[-1])
        r = rates(codes, K)
        n1 = null_n1(codes, K, rng, n_perm)
        full = keys_of(codes, K)
        coll = collided(full)
        ident = cat["ident_item"]
        # buckets: identical-input composition
        _, inv, cnt = np.unique(full, return_inverse=True, return_counts=True)
        pure_ident_items = 0
        small_coll = small_ident = 0
        for g in np.flatnonzero(cnt > 1):
            mem = np.flatnonzero(inv == g)
            n_distinct_inputs = len(np.unique(cat["ident_inv"][mem]))
            if n_distinct_inputs == 1:
                pure_ident_items += len(mem)
            if len(mem) < 3:
                small_coll += len(mem)
                small_ident += int(n_distinct_inputs == 1) * len(mem)
            if q == "rqvae" and len(mem) >= 3:
                sub = cat["dist"][np.ix_(mem, mem)][np.triu_indices(len(mem), 1)]
                bucket_rows.append({
                    "table": name, "sid": "".join(f"({c})" for c in codes[mem[0]]),
                    "size": len(mem), "n_distinct_inputs": n_distinct_inputs,
                    "median_pair_distance_pct": float(np.median(
                        np.searchsorted(cat["all_pair"], sub) / len(cat["all_pair"]))),
                    "n_brands": len({cat["bfold"][i] for i in mem}),
                    "titles": " | ".join(cat["titles"][i][:50] for i in mem[:8]),
                })
        row = {
            "table": name, "quantizer": q, "depth": D, "width": K, **r,
            "n0_rate": 1 - (1 - 1 / K ** D) ** (n - 1),
            "n1_rate_p05": float(np.quantile(n1, 0.05)),
            "n1_rate_p50": float(np.median(n1)),
            "n1_rate_p95": float(np.quantile(n1, 0.95)),
            "obs_over_n1": float(r["collision_rate"] / np.median(n1)) if np.median(n1) > 0 else np.inf,
            "mean_pairwise_nmi": mean_pairwise_nmi(codes),
            "share_collided_identical_input": float(ident[coll].mean()) if coll.any() else np.nan,
            "share_collided_in_pure_identical_buckets": pure_ident_items / max(1, coll.sum()),
            "share_small_bucket_items_identical": small_ident / small_coll if small_coll else np.nan,
            "identical_pairs_split": int(sum(
                1 for a, b, *_ in cat["nd_pairs"]
                if cat["ident_inv"][a] == cat["ident_inv"][b] and full[a] != full[b])),
            "repeat_rate_mean": float(np.mean([(codes[:, l] == codes[:, l - 1]).mean()
                                               for l in range(1, D)])),
            "share_collided_last2_equal": float((codes[coll, -1] == codes[coll, -2]).mean()) if coll.any() else np.nan,
            "share_all_last2_equal": float((codes[:, -1] == codes[:, -2]).mean()),
            "all_levels_balanced": all(balanced(np.bincount(codes[:, l], minlength=K), n, K)
                                       for l in range(D)),
            **nd_shares(codes, K, cat),
            **pair_distance_percentiles(codes, K, cat),
        }
        row["radius_spearman"], row["radius_n_parents"] = radius_spearman(codes, K, cat)
        # A1 (post hoc): the same last-level merge ratio and near-duplicate
        # share with the identical-input items removed -- no assignment can
        # split those, so they mask a dedup step's signature.
        keep = ~ident
        sub = codes[keep]
        a1_obs, a1_null, a1_ratio = merge_ratio(sub, K, D - 1, rng, n_perm)
        row.update(a1_last_merged_obs=a1_obs, a1_last_merged_n2=a1_null,
                   a1_last_merge_ratio=a1_ratio,
                   a1_collision_rate=float(collided(keys_of(sub, K)).mean()))
        rows.append(row)
        for l in range(D):
            cnts = np.bincount(codes[:, l], minlength=K)
            used = cnts[cnts > 0]
            p = used / used.sum()
            lr = {"table": name, "quantizer": q, "depth": D, "width": K, "level": l,
                  "n_used": int((cnts > 0).sum()), "count_min": int(used.min()),
                  "count_max": int(cnts.max()), "max_code_share": float(cnts.max() / n),
                  "norm_entropy": float(-(p * np.log(p)).sum() / np.log(K)),
                  "balanced": balanced(cnts, n, K),
                  "repeat_rate": float((codes[:, l] == codes[:, l - 1]).mean()) if l else np.nan}
            if l:
                obs, null, ratio = merge_ratio(codes, K, l, rng, n_perm)
                lr.update(merged_obs=obs, merged_n2=null, merge_ratio=ratio)
            lvl_rows.append(lr)
        log(f"  {name}: rate={r['collision_rate']:.4f} n1={np.median(n1):.4f} "
            f"nd={row['nd_share_collided']:.3f}")
    tab = pd.DataFrame(rows)
    lvl = pd.DataFrame(lvl_rows)
    tab.to_csv(out / "partB_tables.csv", index=False)
    lvl.to_csv(out / "partB_levels.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv(out / "partB_rqvae_buckets.csv", index=False)
    return tab, lvl


# --------------------------------------------------------------- Part C --
def fit_rq(X32, L, K, seed_fn, scale_norm=False):
    """Per-level k-means on residuals, nearest-centroid assignment.

    With ``seed_fn = lambda l: 1234`` and ``scale_norm=False`` this is exactly
    faiss.ResidualQuantizer(Train_default, beam 1) at the same thread count
    (checked against the archive in ``validate_native``). The non-scaled path
    must keep ``R - centroids[a]`` in float32: the archive was made that way.
    """
    N, d = X32.shape
    R = np.ascontiguousarray(X32.copy())
    codes = np.zeros((N, L), np.int64)
    cbs, scales = [], []
    for l in range(L):
        if scale_norm and l >= 1:
            _, inv = np.unique(keys_of(codes[:, :l], K), return_inverse=True)
            ss = (np.bincount(inv, weights=(R.astype(np.float64) ** 2).sum(1))
                  / np.bincount(inv))
            s = np.sqrt(ss)[inv]
            s[s < 1e-12] = 1.0  # singleton parents: residual is ~0 anyway
            U = np.ascontiguousarray((R / s[:, None]).astype(np.float32))
        else:
            s = None
            U = R
        km = faiss.Kmeans(d, K, niter=10, seed=int(seed_fn(l)), verbose=False)
        km.train(U)
        _, a = km.index.search(U, 1)
        a = a[:, 0].astype(np.int64)
        codes[:, l] = a
        cbs.append(km.centroids.copy())
        scales.append(s)
        if s is None:
            R = R - km.centroids[a]
        else:
            R = np.ascontiguousarray(
                (R - s[:, None].astype(np.float32) * km.centroids[a]).astype(np.float32))
    return codes, cbs, scales


def make_transforms(X64):
    Xc = X64 - X64.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n = len(X64)
    def pca(k, white):
        Z = Xc @ Vt[:k].T
        if white:
            Z = Z / (S[:k] / np.sqrt(n - 1))
        return np.ascontiguousarray(Z.astype(np.float32))
    return {
        "L2": np.ascontiguousarray((X64 / np.linalg.norm(X64, axis=1, keepdims=True)).astype(np.float32)),
        "PCA-256": pca(256, False),
        "PCA-256-white": pca(256, True),
        "PCA-64-white": pca(64, True),
    }, (Xc, Vt)


CONDITIONS = ("native", "per-level seed", "scale-norm", "scale-norm + per-level seed",
              "L2", "PCA-256", "PCA-256-white", "PCA-64-white")


def validate_native(X32, tables, item_out, widths):
    """Native seed-1234 refits must equal every nested archived RQ-KMeans table."""
    checks, fits = [], {}
    for K in widths:
        codes, cbs, _ = fit_rq(X32, FIT_LEVELS, K, lambda l: NATIVE_SEED)
        fits[K] = (codes, cbs)
        for D in (3, 4, 5):
            name = f"rqkmeans_{D}codebook_{K}"
            arch = tables[name]
            eq = float((codes[:, :D] == arch).all(1).mean())
            nested = not (K == 512 and D == 5)
            checks.append({"table": name, "nested_expected": nested,
                           "row_match": eq,
                           "ari_first_digit": float(_ari(codes[:, 0], arch[:, 0]))})
            if nested and eq != 1.0:
                item_out.write_text(json.dumps(checks, indent=2))
                raise RuntimeError(
                    f"native refit does not reproduce {name} ({eq:.4f} of rows); "
                    f"thread count is {faiss.omp_get_max_threads()} -- refusing to "
                    f"interpret any refit")
    item_out.write_text(json.dumps(checks, indent=2))
    return checks, fits


def _ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def refit_metrics(codes, K, cat):
    X, total = cat["X"], cat["total_ss"]
    out = []
    r2_1 = r2_by_groups(X, keys_of(codes[:, :1], K), total)
    r2_2 = r2_by_groups(X, keys_of(codes[:, :2], K), total)
    knn1 = knn_agreement(cat["knn10"], codes[:, 0])
    for D in (3, 4, 5):
        c = codes[:, :D]
        r = rates(c, K)
        coll = collided(keys_of(c, K))
        out.append({
            "depth": D, **r,
            "repeat_rate_mean": float(np.mean([(c[:, l] == c[:, l - 1]).mean()
                                               for l in range(1, D)])),
            "share_collided_last2_equal": float((c[coll, -1] == c[coll, -2]).mean()) if coll.any() else np.nan,
            "nd_share_collided": nd_shares(c, K, cat)["nd_share_collided"],
            "r2_depth1": r2_1, "r2_depth2": r2_2, "knn10_agree_depth1": knn1,
        })
    return out


def part_c(X32, tables, cat, out, seeds, widths):
    log("Part C: RQ-KMeans refits")
    checks, native_fits = validate_native(X32, tables, out / "validation.json", widths)
    log(f"  native reproduction ok: {[c['row_match'] for c in checks]}")
    transforms, _ = make_transforms(cat["X"])
    rows = []
    for K in widths:
        for seed in seeds:
            for cond in CONDITIONS:
                t0 = time.time()
                per_level = "per-level" in cond
                seed_fn = (lambda l, s=seed: s + 7919 * l) if per_level else (lambda l, s=seed: s)
                if cond == "native" and seed == NATIVE_SEED:
                    codes = native_fits[K][0]
                elif cond in transforms:
                    codes, _, _ = fit_rq(transforms[cond], FIT_LEVELS, K, seed_fn)
                else:
                    codes, _, _ = fit_rq(X32, FIT_LEVELS, K, seed_fn,
                                         scale_norm=cond.startswith("scale-norm"))
                for m in refit_metrics(codes, K, cat):
                    rows.append({"condition": cond, "width": K, "seed": seed, **m})
                log(f"  K={K} seed={seed} {cond}: "
                    f"rate@3={rows[-3]['collision_rate']:.4f} "
                    f"@5={rows[-1]['collision_rate']:.4f} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / "partC_refits.csv", index=False)
    summ = summarise_refits(df)
    summ.to_csv(out / "partC_summary.csv", index=False)
    return df, summ, native_fits


BASELINE = {"per-level seed": "native", "scale-norm": "native",
            "scale-norm + per-level seed": "per-level seed", "L2": "native",
            "PCA-256": "native", "PCA-256-white": "PCA-256", "PCA-64-white": "PCA-256"}


def summarise_refits(df):
    rows = []
    for (cond, K, D), g in df.groupby(["condition", "width", "depth"]):
        r = {"condition": cond, "width": K, "depth": D, "n_seeds": len(g)}
        for col in ("collision_rate", "repeat_rate_mean", "nd_share_collided",
                    "r2_depth1", "r2_depth2", "knn10_agree_depth1"):
            r[f"{col}_mean"] = float(g[col].mean())
            r[f"{col}_min"] = float(g[col].min())
            r[f"{col}_max"] = float(g[col].max())
        base = BASELINE.get(cond)
        if base is not None:
            b = df[(df.condition == base) & (df.width == K) & (df.depth == D)]
            m = g.set_index("seed")["collision_rate"].sub(
                b.set_index("seed")["collision_rate"]).dropna()
            r["baseline"] = base
            r["paired_diff_mean"] = float(m.mean())
            r["paired_diff_min"] = float(m.min())
            r["paired_diff_max"] = float(m.max())
            r["ranges_disjoint"] = bool(g.collision_rate.max() < b.collision_rate.min()
                                        or g.collision_rate.min() > b.collision_rate.max())
            r["knn10_diff_mean"] = float(g.knn10_agree_depth1.mean() - b.knn10_agree_depth1.mean())
        rows.append(r)
    return pd.DataFrame(rows)


def hub_codes(native_fits, X32, out):
    """A2 (post hoc): is each level's most-used code a near-origin centroid?

    Uses the native seed-1234 refits, whose codes equal the archive (checked
    in validate_native), so these are the archived tables' own centroids.
    """
    rows = []
    for K, (codes5, cbs) in native_fits.items():
        R = X32.astype(np.float64)
        for l in range(FIT_LEVELS):
            c = codes5[:, l]
            cnt = np.bincount(c, minlength=K)
            hub = int(cnt.argmax())
            cn = np.linalg.norm(cbs[l].astype(np.float64), axis=1)
            rn = np.linalg.norm(R, axis=1)
            row = {"width": K, "level": l, "hub_code": hub,
                   "hub_share": float(cnt[hub] / len(c)),
                   "hub_centroid_norm": float(cn[hub]),
                   "median_centroid_norm": float(np.median(cn[cnt > 0])),
                   "hub_norm_rank_from_smallest": int((cn[cnt > 0] < cn[hub]).sum()),
                   "residual_norm_hub_items": float(rn[c == hub].mean()),
                   "residual_norm_other_items": float(rn[c != hub].mean())}
            for D in (3, 4, 5):
                if l == D - 1 and not (K == 512 and D == 5):
                    coll = collided(keys_of(codes5[:, :D], K))
                    row[f"share_collided_with_hub_last_D{D}"] = float((c[coll] == hub).mean())
                    row[f"share_all_with_hub_last_D{D}"] = float((c == hub).mean())
            rows.append(row)
            R = R - cbs[l][c].astype(np.float64)
    df = pd.DataFrame(rows)
    df.to_csv(out / "partC_a2_hub_codes.csv", index=False)
    return df


# --------------------------------------------------------------- Part D --
def rqvae_sinkhorn_argmax(d):
    """VectorQuantizer.center_distance_for_constraint + sinkhorn_algorithm."""
    mx, mn = d.max(), d.min()
    mid = (mx + mn) / 2
    amp = mx - mid + 1e-5
    Q = np.exp(-((d - mid) / amp) / RQVAE_SK_EPS)
    B, K = Q.shape
    Q = Q / Q.sum()
    for _ in range(RQVAE_SK_ITERS):
        Q = Q / Q.sum(1, keepdims=True) / B
        Q = Q / Q.sum(0, keepdims=True) / K
    Q = Q * B
    if not np.isfinite(Q).all():
        return None
    return Q.argmax(1)


def part_d1(native_fits, cat, out):
    log("Part D1: RQ-VAE last-digit loop on RQ-KMeans geometry")
    X64 = cat["X"]
    rows, brows = [], []
    for K, (codes5, cbs) in native_fits.items():
        for D in (3, 4, 5):
            if K == 512 and D == 5:
                continue  # not an archived table
            codes = codes5[:, :D].copy()
            R = X64.copy()
            for l in range(D - 1):
                R -= cbs[l][codes[:, l]].astype(np.float64)
            C = cbs[D - 1].astype(np.float64)
            dist_last = sqdist(R, C)
            native_last = codes[:, -1].copy()
            start = rates(codes, K)["collision_rate"]
            rounds = 0
            n_nan = 0
            for rnd in range(RQVAE_MAX_ROUNDS):
                k = keys_of(codes, K)
                _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
                groups = [np.flatnonzero(inv == g) for g in np.flatnonzero(cnt > 1)]
                if not groups:
                    break
                rounds += 1
                for mem in groups:
                    a = rqvae_sinkhorn_argmax(dist_last[mem])
                    if a is None:
                        n_nan += 1
                        continue
                    codes[mem, -1] = a
            fin = rates(codes, K)
            k = keys_of(codes, K)
            _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
            ident_b = other_b = ident_items = other_items = 0
            for g in np.flatnonzero(cnt > 1):
                mem = np.flatnonzero(inv == g)
                if len(np.unique(cat["ident_inv"][mem])) == 1:
                    ident_b += 1
                    ident_items += len(mem)
                else:
                    other_b += 1
                    other_items += len(mem)
                    brows.append({"table": f"rqkmeans_{D}codebook_{K}", "size": len(mem),
                                  "n_distinct_inputs": len(np.unique(cat["ident_inv"][mem])),
                                  "titles": " | ".join(cat["titles"][i][:50] for i in mem[:6])})
            changed = codes[:, -1] != native_last
            idx = np.arange(len(codes))
            inc = dist_last[idx, codes[:, -1]] - dist_last[idx, native_last]
            rows.append({
                "table": f"rqkmeans_{D}codebook_{K}", "rate_before": start,
                "rate_after": fin["collision_rate"], "rate_after_old": fin["collision_rate_old"],
                "identical_floor": float(cat["ident_item"].mean()),
                "rounds": rounds, "sinkhorn_nonfinite_groups": n_nan,
                "remaining_identical_buckets": ident_b, "remaining_identical_items": ident_items,
                "remaining_other_buckets": other_b, "remaining_other_items": other_items,
                "share_last_digit_changed": float(changed.mean()),
                "last_level_sq_residual_native": float(dist_last[idx, native_last].mean()),
                "last_level_sq_residual_increase_mean": float(inc.mean()),
                "last_level_sq_residual_increase_changed": float(inc[changed].mean()) if changed.any() else 0.0,
            })
            log(f"  {rows[-1]['table']}: {start:.4f} -> {fin['collision_rate']:.4f} "
                f"in {rounds} rounds")
    pd.DataFrame(rows).to_csv(out / "partD1_rqvae_loop.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "partD1_remaining_buckets.csv", index=False)
    return pd.DataFrame(rows)


def pot_sinkhorn(a, b, M, reg, num_iter=1000, stop=1e-9):
    """ot.sinkhorn(method='sinkhorn') as in POT's sinkhorn_knopp."""
    u = np.ones(len(a)) / len(a)
    v = np.ones(len(b)) / len(b)
    Kmat = np.exp(M / (-reg))
    Kp = (1 / a)[:, None] * Kmat
    for it in range(num_iter):
        up, vp = u, v
        KtU = Kmat.T @ u
        v = b / KtU
        u = 1.0 / (Kp @ v)
        if (KtU == 0).any() or not (np.isfinite(u).all() and np.isfinite(v).all()):
            u, v = up, vp
            break
        if it % 10 == 0:
            err = np.linalg.norm(np.einsum("i,ij,j->j", u, Kmat, v) - b)
            if err < stop:
                break
    return u[:, None] * Kmat * v[None, :]


def uniform_mapping(X32, codes, cbs, K, topk=32, seed=42):
    """rqkmeans_faiss.sinkhorn_uniform_mapping, reimplemented without POT."""
    N, M = codes.shape
    bal = codes.copy()
    for l in range(M):
        R = X32.astype(np.float32).copy()
        for p in range(l):
            R -= cbs[p][bal[:, p]]
        C = cbs[l]
        cap = np.full(K, N // K, dtype=np.int64)
        cap[: N % K] += 1
        Dm = sqdist(R, C)
        spread = np.percentile(Dm - Dm.min(1, keepdims=True), 90, axis=1)
        tau = max(float(np.median(spread) * 0.1), 1e-6)
        P = pot_sinkhorn(np.ones(N) / N, cap / N, Dm, tau)
        rng = np.random.RandomState(seed + l)
        order = np.arange(N)
        rng.shuffle(order)
        remaining = cap.copy()
        for i in order:
            probs = P[i]
            cand = np.argpartition(-probs, topk - 1)[:topk]
            cand = cand[np.argsort(-probs[cand])]
            chosen = next((int(c) for c in cand if remaining[c] > 0), -1)
            if chosen < 0:
                c = int(np.argmax(probs))
                chosen = c if remaining[c] > 0 else int(np.argmin(remaining))
            remaining[chosen] -= 1
            bal[i, l] = chosen
    return bal


def part_d2(X32, native_fits, cat, out):
    log("Part D2: Sinkhorn uniform mapping")
    rows = []
    for K, D in ((128, 3), (512, 5)):
        codes5, cbs = native_fits[K]
        codes = codes5[:, :D]
        bal = uniform_mapping(X32, codes, cbs[:D], K)
        for label, c in (("native", codes), ("balanced", bal)):
            full = keys_of(c, K)
            rows.append({
                "fit": f"{D}x{K}", "mapping": label, **rates(c, K),
                "all_levels_balanced": all(balanced(np.bincount(c[:, l], minlength=K), len(c), K)
                                           for l in range(D)),
                "identical_pairs_split": int(sum(
                    1 for a, b, *_ in cat["nd_pairs"]
                    if cat["ident_inv"][a] == cat["ident_inv"][b] and full[a] != full[b])),
                "n_identical_pairs": int(sum(1 for a, b, *_ in cat["nd_pairs"]
                                             if cat["ident_inv"][a] == cat["ident_inv"][b])),
                "nd_share_collided": nd_shares(c, K, cat)["nd_share_collided"],
                "knn10_agree_depth1": knn_agreement(cat["knn10"], c[:, 0]),
            })
        log(f"  {D}x{K}: native {rows[-2]['collision_rate']:.4f} -> balanced "
            f"{rows[-1]['collision_rate']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "partD2_uniform_mapping.csv", index=False)
    return df


# --------------------------------------------------------------- Part E --
def part_e(cat, out, seeds, n_perm):
    log("Part E: parallel-digit refits (exploratory)")
    Xc = cat["Xc"]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:PQ_DIM].T
    rng_perm = np.random.default_rng(PERM_SEED + 1)
    rows = []
    for seed in seeds:
        rot = np.linalg.qr(np.random.RandomState(seed).randn(PQ_DIM, PQ_DIM))[0]
        Zr = Z @ rot
        for D in (3, 4, 5):
            w = PQ_DIM // D
            alloc = {
                "random-rotation": [Zr[:, d * w:(d + 1) * w] for d in range(D)],
                "pca-round-robin": [Z[:, d::D] for d in range(D)],
                "pca-contiguous": [Z[:, d * w:(d + 1) * w] for d in range(D)],
            }
            for K in WIDTHS:
                for name, blocks in alloc.items():
                    codes = np.zeros((len(Z), D), np.int64)
                    for d, B in enumerate(blocks):
                        B = np.ascontiguousarray(B.astype(np.float32))
                        km = faiss.Kmeans(B.shape[1], K, niter=25, seed=seed, verbose=False)
                        km.train(B)
                        codes[:, d] = km.index.search(B, 1)[1][:, 0]
                    r = rates(codes, K)
                    n1 = null_n1(codes, K, rng_perm, n_perm)
                    rows.append({"allocation": name, "depth": D, "width": K, "seed": seed,
                                 **r, "n1_rate_p50": float(np.median(n1)),
                                 "obs_over_n1": float(r["collision_rate"] / np.median(n1)) if np.median(n1) > 0 else np.inf,
                                 "mean_pairwise_nmi": mean_pairwise_nmi(codes),
                                 "knn10_agree_depth1": knn_agreement(cat["knn10"], codes[:, 0])})
        log(f"  seed {seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(out / "partE_parallel_refits.csv", index=False)
    return df


# ---------------------------------------------------------------- verdict --
def verdicts(tab, lvl, summ, d1, d2):
    v = {}
    rqvae = tab[tab.quantizer == "rqvae"]
    rqk = tab[tab.quantizer == "rqkmeans"]
    mq = tab[tab.quantizer == "MQ"]
    last = lvl[lvl.level == lvl.depth - 1].set_index("table")
    v["H1a"] = {
        "rqvae_last_merge_ratio": last.loc[rqvae.table, "merge_ratio"].to_dict(),
        "rqkmeans_mq_last_merge_ratio_min": float(last.loc[pd.concat([rqk.table, mq.table]), "merge_ratio"].min()),
        "rqvae_small_bucket_identical_share": rqvae.set_index("table")["share_small_bucket_items_identical"].to_dict(),
        "d1_after_minus_floor_pp": (100 * (d1.rate_after - d1.identical_floor)).round(3).tolist() if d1 is not None else None,
    }
    a1 = tab.set_index("table")["a1_last_merge_ratio"]
    v["H1a_A1_post_hoc"] = {
        "rqvae_a1_ratio": a1.loc[rqvae.table].round(3).to_dict(),
        "rqkmeans_mq_a1_ratio_min": float(a1.loc[pd.concat([rqk.table, mq.table])].min()),
        "rqvae_a1_collision_rate": tab.set_index("table").loc[rqvae.table, "a1_collision_rate"].round(4).to_dict(),
    }
    v["H1a_A1_post_hoc"]["supported"] = bool(
        (a1.loc[rqvae.table].fillna(0) < 0.2).all()
        and (a1.loc[pd.concat([rqk.table, mq.table])] >= 0.5).all())
    v["H1a"]["supported"] = bool(
        (last.loc[rqvae.table, "merge_ratio"].fillna(0) < 0.2).all()
        and (rqvae.share_small_bucket_items_identical.fillna(1) >= 0.8).all()
        and (d1 is not None and ((d1.rate_after - d1.identical_floor) <= 0.005).all()))
    others = rqk[rqk.table != "rqkmeans_5codebook_512"]
    v["H1b"] = {"5x512_balanced": bool(rqk.set_index("table").loc["rqkmeans_5codebook_512", "all_levels_balanced"]),
                "others_balanced": others.set_index("table")["all_levels_balanced"].to_dict()}
    v["H1b"]["supported"] = bool(v["H1b"]["5x512_balanced"] and not others.all_levels_balanced.any())
    nested = others.copy()
    trend = {}
    for K, g in nested.groupby("width"):
        g = g.sort_values("depth")
        trend[int(K)] = g.nd_share_collided.round(4).tolist()
    enrich = (tab.nd_share_collided > tab.nd_share_multi_parent)
    deepest = nested[(nested.depth == 5) & (nested.width == 256)].nd_share_collided
    v["H2"] = {"enriched_tables": int(enrich.sum()), "n_tables": len(tab),
               "rqkmeans_nd_share_by_depth": trend,
               "nd_share_5x256": float(deepest.iloc[0]) if len(deepest) else None}
    v["H2"]["enriched_supported"] = bool(enrich.sum() == len(tab))
    v["H2"]["rises_with_depth"] = bool(all(np.all(np.diff(x) >= 0) for x in trend.values()))
    v["H2"]["floor_supported"] = bool(v["H2"]["nd_share_5x256"] is not None and v["H2"]["nd_share_5x256"] > 0.5)
    rl = lvl[(lvl.quantizer == "rqkmeans") & (lvl.level >= 1)]
    v["H3a"] = {"share_levels_ratio_gt1": float((rl.merge_ratio > 1).mean()),
                "median_ratio": float(rl.merge_ratio.median())}
    v["H3a"]["supported"] = bool(v["H3a"]["share_levels_ratio_gt1"] > 0.5)
    rep = tab.set_index("table")
    v["H3b"] = {"repeat_rate": rep.repeat_rate_mean.round(4).to_dict()}
    if summ is not None:
        s = summ[summ.condition == "per-level seed"]
        v["H3b"]["per_level_seed_lowers"] = s[["width", "depth", "paired_diff_mean", "ranges_disjoint"]].to_dict("records")
        ok_rep = all(rep.loc[t, "repeat_rate_mean"] >= 3 / rep.loc[t, "width"] for t in rqk.table)
        ok_ctrl = all(rep.loc[t, "repeat_rate_mean"] < 3 / rep.loc[t, "width"] for t in pd.concat([rqvae.table, mq.table]))
        lowers = bool(((s.paired_diff_mean < 0) & s.ranges_disjoint).all())
        v["H3b"]["supported"] = bool(ok_rep and ok_ctrl and lowers)
        v["H3b"]["repeat_criterion"] = bool(ok_rep and ok_ctrl)
        v["H3b"]["lowers_all_cells"] = lowers
        s = summ[summ.condition == "scale-norm"]
        v["H3c"] = {"cells": s[["width", "depth", "paired_diff_mean", "ranges_disjoint", "knn10_diff_mean"]].to_dict("records")}
        v["H3c"]["supported"] = bool(((s.paired_diff_mean < 0) & s.ranges_disjoint & (s.knn10_diff_mean > -0.05)).all())
        s = summ[summ.condition == "PCA-256-white"]
        v["H4"] = {"cells": s[["width", "depth", "paired_diff_mean", "ranges_disjoint", "knn10_diff_mean"]].to_dict("records")}
        v["H4"]["supported"] = bool(((s.paired_diff_mean < 0) & s.ranges_disjoint).all())
    v["H5"] = {"mq_obs_over_n1": mq.set_index("table").obs_over_n1.round(3).to_dict(),
               "mq_nmi": mq.set_index("table").mean_pairwise_nmi.round(3).to_dict()}
    v["H5"]["supported"] = bool((mq.obs_over_n1 >= 2).all())
    return v


def md_table(df: pd.DataFrame) -> str:
    """Markdown table without tabulate (not installed in this venv)."""
    def fmt(x):
        if isinstance(x, float):
            return "" if np.isnan(x) else f"{x:.4g}"
        return str(x)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def write_report(out, catalogue, tab, lvl, summ, d1, d2, pe, v):
    L = ["# Structure exp3: why items share a Semantic ID -- generated report", ""]
    L.append("Auto-generated by run.py; interpretation lives in RESULTS.md.")
    L.append("")
    L.append("## Catalogue")
    L.append("```")
    L.append(json.dumps(catalogue, indent=1))
    L.append("```")
    cols = ["table", "collision_rate", "n1_rate_p50", "obs_over_n1", "mean_pairwise_nmi",
            "nd_share_collided", "nd_share_multi_parent", "repeat_rate_mean",
            "share_collided_last2_equal", "all_levels_balanced", "radius_spearman",
            "a1_last_merge_ratio", "a1_collision_rate"]
    L += ["## Archived tables", "", tab[cols].round(4).pipe(md_table), ""]
    last = lvl[lvl.level >= 1][["table", "level", "merge_ratio", "max_code_share", "norm_entropy", "repeat_rate"]]
    L += ["## Merge ratio by level", "", last.round(3).pipe(md_table), ""]
    if summ is not None:
        L += ["## Refits (Part C)", "", summ.round(4).pipe(md_table), ""]
    hub = out / "partC_a2_hub_codes.csv"
    if hub.exists():
        L += ["## Hub codes of the native refits (A2, post hoc)", "",
              pd.read_csv(hub).round(4).pipe(md_table), ""]
    if d1 is not None:
        L += ["## RQ-VAE loop on RQ-KMeans (Part D1)", "", d1.round(5).pipe(md_table), ""]
    if d2 is not None:
        L += ["## Uniform mapping (Part D2)", "", d2.round(4).pipe(md_table), ""]
    if pe is not None:
        g = pe.groupby(["allocation", "depth", "width"]).agg(
            rate=("collision_rate", "mean"), rate_min=("collision_rate", "min"),
            rate_max=("collision_rate", "max"), nmi=("mean_pairwise_nmi", "mean"),
            obs_over_n1=("obs_over_n1", "mean"), knn=("knn10_agree_depth1", "mean")).reset_index()
        L += ["## Parallel-digit refits (Part E, exploratory)", "", g.round(4).pipe(md_table), ""]
    L += ["## Verdicts by the declared rules", "```", json.dumps(v, indent=1, default=str), "```"]
    (out / "report.md").write_text("\n".join(L))


# ------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: 2 seeds, 20 permutations, width 128 refits only")
    ap.add_argument("--skip", default="", help="comma list of parts to skip (C,D,E)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    faiss.omp_set_num_threads(args.threads)
    skip = set(filter(None, args.skip.split(",")))
    seeds = SEEDS[:2] if args.quick else SEEDS
    n_perm = 20 if args.quick else args.n_perm
    widths = (128,) if args.quick else WIDTHS

    raw, item2id, titles, brands, emb_path = load_inputs()
    names = [v.name for v in available("diffgrm")]
    if len(names) != 27:
        raise ValueError(f"expected 27 tables, found {len(names)}")
    tables = {nm: load_codes(nm, item2id, len(raw)) for nm in names}
    inputs = {"embedding": {"path": str(emb_path), "sha256": sha256_file(emb_path)},
              "item_meta": {"path": str(paths.FROZEN_DATA / "item_meta" / f"{CATEGORY}.item.json"),
                            "sha256": sha256_file(paths.FROZEN_DATA / "item_meta" / f"{CATEGORY}.item.json")},
              "tables": {nm: sha256_file(paths.FROZEN_SIDS / "sem_ids" / "diffgrm" / f"{nm}.sem_ids")
                         for nm in names}}
    (out / "inputs.json").write_text(json.dumps(inputs, indent=2))
    (out / "args.json").write_text(json.dumps({
        **vars(args), "seeds": seeds, "n_perm_used": n_perm, "widths": widths,
        "faiss": faiss.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "omp_threads": faiss.omp_get_max_threads()}, indent=2))

    catalogue, cat = part_a(raw, titles, brands, out)
    tab, lvl = part_b(tables, cat, out, n_perm)
    X32 = np.ascontiguousarray(raw.astype(np.float32))
    summ = d1 = d2 = pe = None
    if "C" not in skip:
        _, summ, native_fits = part_c(X32, tables, cat, out, seeds, widths)
        hub_codes(native_fits, X32, out)
        if "D" not in skip:
            d1 = part_d1(native_fits, cat, out)
            d2 = part_d2(X32, native_fits, cat, out) if 512 in native_fits else None
    if "E" not in skip:
        pe = part_e(cat, out, seeds, 50 if not args.quick else 10)
    v = verdicts(tab, lvl, summ, d1, d2)
    (out / "result.json").write_text(json.dumps({"catalogue": catalogue, "verdicts": v},
                                                indent=2, default=str))
    write_report(out, catalogue, tab, lvl, summ, d1, d2, pe, v)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
