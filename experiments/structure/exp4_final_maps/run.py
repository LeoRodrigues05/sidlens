#!/usr/bin/env python
"""Structure exp4 -- the final (v3) SID maps of one catalog.

Runs the parts declared in ``protocol.md`` on the 27 native v3 tables of one
catalog and on their ZCR twins. CPU only; no recommender is run. Every
measure is computed by the code that produced the old numbers:
``sidlens.analysis`` (geometry, refinement, attribute, naming), the helpers of
``experiments/structure/semantic_mapping/figures.py`` (purity, variance per
digit) and of ``experiments/structure/exp3_collision_causes/run.py`` (Parts A,
B, C, the dedup-loop reimplementation). This script only feeds them the v3
tables and adds the parts that need the new weights.

Traps this script guards against:

* faiss k-means depends on the OpenMP thread count. The v3 RQ-KMeans tables
  were built at 8 threads (tables.json, ZCR stats). Part 0 refits at
  ``--threads`` and Part C refuses to run unless the native refit reproduces
  every RQ-KMeans table of the catalog item for item.
* GNU OpenMP is not fork-safe once the parent has run a parallel region, so
  every pool uses the ``spawn`` start method and its workers load their inputs
  themselves (large arrays through memory-mapped .npy files).
* exp3's Part B reads its permutation seed from a module constant. The seed of
  this experiment is set on that module before the call, and recorded.
* The v3 RQ-KMeans 5xW tables are nested like the others (Part 0 checks it),
  so unlike exp3 nothing is excluded from the depth trends -- but only if the
  check passes; the flag is read from validation, never assumed.
* Label fields differ by catalog: Office's level-1 breadcrumb has 3 classes,
  so its primary field is cat_l2 (protocol, "Category fields").

    python experiments/structure/exp4_final_maps/run.py --catalog industrial --out DIR
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sidlens import paths
from sidlens.analysis import attribute as AT
from sidlens.analysis import geometry as G
from sidlens.analysis import naming as NM
from sidlens.analysis import refinement as R
from sidlens.data import codebooks_v3 as V
from sidlens.data import embeddings as E
from sidlens.data import labels as L
from sidlens.data import meta as M
from sidlens.data.sids import SidTable, SidVariant, check_nesting
from sidlens.provenance.hashing import sha256_file

HERE = Path(__file__).resolve().parent
EXP3_RUN = paths.REPO / "experiments/structure/exp3_collision_causes/run.py"
FIGURES = paths.REPO / "experiments/structure/semantic_mapping/figures.py"

PERM_SEED = 20261020
N_PERM = 200
NATIVE_SEED = 1234
SEEDS = (1234, 1, 2, 3, 4)
WIDTHS = (128, 256, 512)
FIT_LEVELS = 5
KNN = 10
PURITY_FIELDS = ("cat_l1", "cat_l2", "cat_l3")
AMI_FIELDS = {"industrial": ("cat_l1", "cat_l2"), "office": ("cat_l2", "cat_l3")}
PRIMARY_FIELD = {"industrial": "cat_l1", "office": "cat_l2"}
HISTORY = 50
ALL_PARTS = ("0", "A", "B", "D", "HUB", "C", "MQ", "X", "M1", "Z", "M5", "R")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E3 = _load_module("exp3_collision_causes_run", EXP3_RUN)
F = _load_module("semantic_mapping_figures", FIGURES)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def meta_cols(catalog: str, kind: str, name: str) -> dict:
    v = SidVariant.parse(name)
    return {"catalog": catalog, "kind": kind, "variant": name, "quantizer": v.quantizer,
            "n_codebook": v.n_codebook, "codebook_size": v.codebook_size}


# ================================================================ state ==
# Each worker builds this once (spawn start method); the parent builds it too.
_S: dict = {}


def init_state(catalog: str, shared: str, n_perm: int, blas_threads: int) -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(blas_threads)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    cat = V.category(catalog)
    tables = {"native": {n: V.load_table(catalog, n) for n in V.names(catalog)},
              "zcr": {n: V.load_table(catalog, n, "zcr") for n in V.zcr_names(catalog)}}
    keys = tables["native"][V.names(catalog)[0]].keys
    sh = Path(shared)
    _S.clear()
    _S.update(catalog=catalog, category=cat, tables=tables, keys=keys, n_perm=n_perm,
              shared=sh)
    if (sh / "zc.npy").exists():
        _S["zc"] = np.load(sh / "zc.npy", mmap_mode="r")
        _S["knn"] = np.load(sh / "knn.npy", mmap_mode="r")
        _S["total_ss"] = float(json.loads((sh / "scalars.json").read_text())["total_ss"])


def prepare_shared(catalog: str, out: Path) -> Path:
    """Centred embedding and 10-NN graph, computed once, read by every worker."""
    sh = out / "_shared"
    sh.mkdir(exist_ok=True)
    keys = _S["keys"]
    z = E.load_aligned(_S["category"], keys)                # float64, item order
    zc = z - z.mean(axis=0)
    np.save(sh / "zc.npy", zc)
    np.save(sh / "knn.npy", R.knn_indices(zc, KNN))
    (sh / "scalars.json").write_text(json.dumps(
        {"total_ss": float(np.einsum("ij,ij->i", zc, zc).sum())}))
    return sh


# ============================================================ Part 0 ==
def part0(catalog: str, out: Path, threads: int) -> dict:
    """Validation. Writes validation.json; returns the facts later parts need."""
    import faiss
    log("Part 0: validation")
    faiss.omp_set_num_threads(threads)
    val: dict = {"catalog": catalog, "dataset": V.REPO_ID, "revision": V.REVISION,
                 "faiss": faiss.__version__, "faiss_threads": faiss.omp_get_max_threads()}
    T = _S["tables"]
    n = len(_S["keys"])
    val["hashes"] = "every native table matched done.json/tables.json; every ZCR index matched " \
                    "its manifest and names its native table by hash (checked on load)"
    # -- ZCR invariants
    zrows = []
    for name, zt in T["zcr"].items():
        nt = T["native"][name]
        b = V.zcr_bundle(catalog, name)
        nc, zc_ = nt.codes, zt.codes
        K = nt.variant.codebook_size
        inv = b["manifest"]["invariants"]
        zrows.append({
            "table": name,
            "native_codes_npy_equal": bool(np.array_equal(b["native_codes"], nc)),
            "zcr_codes_npy_equal": bool(np.array_equal(b["zcr_codes"], zc_)),
            "prefix_preserved": bool((nc[:, :-1] == zc_[:, :-1]).all()),
            "collision_free": bool(E3.rates(zc_, K)["max_bucket"] == 1),
            "changed_tails": int((nc[:, -1] != zc_[:, -1]).sum()),
            "recorded_changed_tails": inv["changed_tails"],
            "recorded_minimum": inv["minimum_tail_changes"],
        })
    zdf = pd.DataFrame(zrows)
    zdf["ok"] = (zdf.native_codes_npy_equal & zdf.prefix_preserved & zdf.collision_free
                 & (zdf.changed_tails == zdf.recorded_changed_tails)
                 & (zdf.changed_tails == zdf.recorded_minimum))
    zdf.to_csv(out / "part0_zcr_checks.csv", index=False)
    val["zcr_all_ok"] = bool(zdf.ok.all())
    # -- nesting, every family
    nest = []
    for q in ("rqkmeans", "rqvae", "MQ"):
        for K in WIDTHS:
            for a, b in ((3, 4), (3, 5), (4, 5)):
                r = check_nesting(T["native"][f"{q}_{a}codebook_{K}"], T["native"][f"{q}_{b}codebook_{K}"])
                nest.append({"quantizer": q, "width": K, **r})
    ndf = pd.DataFrame(nest)
    ndf.to_csv(out / "part0_nesting.csv", index=False)
    val["rqkmeans_nested_all"] = bool(ndf[ndf.quantizer == "rqkmeans"].nested.all())
    val["rqvae_nested_any"] = bool(ndf[ndf.quantizer == "rqvae"].nested.any())
    val["mq_nested_any"] = bool(ndf[ndf.quantizer == "MQ"].nested.any())
    # -- RQ-KMeans: native refit at the build's thread count, and the bundle codebooks
    X32 = np.ascontiguousarray(E.load(_S["category"], np.float32))
    refit_rows, refit_ok = [], True
    native_fits = {}
    for K in WIDTHS:
        codes5, cbs, _ = E3.fit_rq(X32, FIT_LEVELS, K, lambda l: NATIVE_SEED)
        native_fits[K] = (codes5, cbs)
        g5 = V.geometry(catalog, f"rqkmeans_5codebook_{K}")["codebooks"]
        for D in (3, 4, 5):
            name = f"rqkmeans_{D}codebook_{K}"
            arch = T["native"][name].codes
            match = float((codes5[:, :D] == arch).all(1).mean())
            gD = V.geometry(catalog, name)["codebooks"]
            refit_rows.append({
                "table": name, "row_match_refit": match,
                "refit_vs_bundle_codebooks_maxabs": float(max(
                    np.abs(cbs[l] - gD[l]).max() for l in range(D))),
                "bundle_codebooks_equal_5level_prefix": bool(np.array_equal(gD, g5[:D])),
            })
            refit_ok &= match == 1.0
        # the bundle's own codebooks must give the table's codes by nearest centroid
        Rz = X32.astype(np.float64)
        rec = np.zeros_like(codes5)
        for l in range(FIT_LEVELS):
            d = E3.sqdist(Rz, g5[l].astype(np.float64))
            rec[:, l] = d.argmin(1)
            Rz = Rz - g5[l][rec[:, l]].astype(np.float64)
        refit_rows.append({"table": f"rqkmeans_5codebook_{K} (bundle codebooks, float64 nearest)",
                           "row_match_refit": float((rec == T["native"][f"rqkmeans_5codebook_{K}"].codes)
                                                    .all(1).mean())})
    pd.DataFrame(refit_rows).to_csv(out / "part0_rqkmeans_refit.csv", index=False)
    val["rqkmeans_refit_reproduces_all"] = bool(refit_ok)
    _S["native_fits"] = native_fits
    # -- RQ-VAE checkpoints: encoder + nearest codes + exported geometry
    rv_rows, pre = [], {}
    for name in [x for x in V.names(catalog) if x.startswith("rqvae")]:
        _, ck = V.rqvae_checkpoint(catalog, name)
        sd = ck["state_dict"]
        z = V.rqvae_encode(sd, X32)
        cbs = V.rqvae_codebooks(sd)
        pc, res = V.rqvae_nearest(z, cbs)
        g = V.geometry(catalog, name)
        t = T["native"][name].codes
        argmin_geo = E3.sqdist(g["vectors"], g["centroids"]).argmin(1)
        pre[name] = pc
        np.save(out / f"pre_reassignment_{name}.npy", pc)
        rv_rows.append({
            "table": name, "epoch": ck.get("epoch"),
            "ckpt_best_collision_rate": ck.get("best_collision_rate"),
            "prefix_match": float((pc[:, :-1] == t[:, :-1]).all(1).mean()),
            "last_digit_nearest_share": float((pc[:, -1] == t[:, -1]).mean()),
            "geometry_argmin_equals_ckpt_nearest": float((argmin_geo == pc[:, -1]).mean()),
            "lsv_maxabs_diff": float(np.abs(res[-1] - g["vectors"]).max()),
            "centroids_maxabs_diff": float(np.abs(cbs[-1] - g["centroids"]).max()),
            "pre_collision_rate_old_def": E3.rates(pc, SidVariant.parse(name).codebook_size)["collision_rate_old"],
        })
    rdf = pd.DataFrame(rv_rows)
    rdf.to_csv(out / "part0_rqvae_checkpoints.csv", index=False)
    val["rqvae_prefix_reproduced_all"] = bool((rdf.prefix_match == 1.0).all())
    _S["pre"] = pre
    # -- MQ models: eval-mode encode reproduces the table
    mq_rows = []
    for name in [x for x in V.names(catalog) if x.startswith("MQ")]:
        _, ck = V.mq_checkpoint(catalog, name)
        mc, last = V.mq_encode(ck["state_dict"], X32)
        t = T["native"][name].codes
        g = V.geometry(catalog, name)
        mq_rows.append({"table": name,
                        "all_digits_match": float((mc == t).all(1).mean()),
                        "min_digit_match": float(min((mc[:, d] == t[:, d]).mean() for d in range(t.shape[1]))),
                        "lsv_maxabs_diff": float(np.abs(last - g["vectors"]).max())})
    mdf = pd.DataFrame(mq_rows)
    mdf.to_csv(out / "part0_mq_models.csv", index=False)
    val["mq_reproduced_all"] = bool((mdf.all_digits_match == 1.0).all())
    (out / "validation.json").write_text(json.dumps(val, indent=2))
    log(f"  {json.dumps({k: v for k, v in val.items() if isinstance(v, bool)})}")
    return val


# ======================================================= Part A and B ==
def item_order_inputs():
    """exp3's load_inputs, for a v3 catalog: raw float16 rows + titles + brands."""
    cat = _S["category"]
    raw = np.load(paths.FROZEN_DATA / "embeddings" / f"{cat}.emb-qwen-td.npy")
    n = len(_S["keys"])
    if raw.dtype != np.float16 or raw.shape != (n, E.EMB_DIM):
        raise ValueError(f"unexpected embedding {raw.dtype} {raw.shape}")
    metas = M.load(cat)
    titles, brands = [""] * n, [""] * n
    for _, m in metas.items():
        titles[m.item_id] = m.title
        brands[m.item_id] = m.brand
    return raw, titles, brands


def native_codes() -> dict[str, np.ndarray]:
    return {n: t.codes for n, t in _S["tables"]["native"].items()}


# ============================================================ Part D ==
def dedup_loop(codes: np.ndarray, dist_last: np.ndarray, K: int):
    """exp3's part_d1 loop body, unchanged, from any starting table."""
    codes = codes.copy()
    rounds = n_nan = 0
    for _ in range(E3.RQVAE_MAX_ROUNDS):
        k = E3.keys_of(codes, K)
        _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        groups = [np.flatnonzero(inv == g) for g in np.flatnonzero(cnt > 1)]
        if not groups:
            break
        rounds += 1
        for mem in groups:
            a = E3.rqvae_sinkhorn_argmax(dist_last[mem])
            if a is None:
                n_nan += 1
                continue
            codes[mem, -1] = a
    return codes, rounds, n_nan


def last_level_distances(name: str) -> tuple[np.ndarray, str]:
    """(N, K) squared distances to the last codebook, in the table's own space."""
    v = SidVariant.parse(name)
    g = V.geometry(_S["catalog"], name)
    if v.quantizer == "rqkmeans":
        # exp3 built the residual from the codebooks in float64; do the same and
        # check it against the bundle's float32 residual.
        X64 = E.load(_S["category"], np.float64)
        codes = _S["tables"]["native"][name].codes
        Rz = X64.copy()
        for l in range(v.n_codebook - 1):
            Rz -= g["codebooks"][l][codes[:, l]].astype(np.float64)
        diff = float(np.abs(Rz - g["vectors"]).max())
        return E3.sqdist(Rz, g["codebooks"][-1].astype(np.float64)), f"residual (bundle maxabs diff {diff:.2e})"
    return E3.sqdist(g["vectors"], g["centroids"]), g["space"]


def part_d(cat: dict, out: Path) -> pd.DataFrame:
    log("Part D: post-processing (D0 RQ-VAE before/after, D1 loop on every table)")
    rows, brows = [], []
    ident_inv = cat["ident_inv"]
    for name, t in _S["tables"]["native"].items():
        v = t.variant
        K = v.codebook_size
        dist_last, space = last_level_distances(name)
        idx = np.arange(len(dist_last))
        published = t.codes
        if v.quantizer == "rqvae":
            start = _S["pre"][name]
            if not (start[:, :-1] == published[:, :-1]).all():
                raise RuntimeError(f"{name}: pre-reassignment prefix differs from the table")
        else:
            start = published
        native_last = start[:, -1].copy()
        nearest = dist_last.argmin(1)
        fin, rounds, n_nan = dedup_loop(start, dist_last, K)
        r0, r1, rp = E3.rates(start, K), E3.rates(fin, K), E3.rates(published, K)
        k = E3.keys_of(fin, K)
        _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        ib = ob = ii = oi = 0
        for gi in np.flatnonzero(cnt > 1):
            mem = np.flatnonzero(inv == gi)
            if len(np.unique(ident_inv[mem])) == 1:
                ib += 1
                ii += len(mem)
            else:
                ob += 1
                oi += len(mem)
                brows.append({"table": name, "size": len(mem),
                              "n_distinct_inputs": len(np.unique(ident_inv[mem])),
                              "titles": " | ".join(cat["titles"][i][:50] for i in mem[:6])})
        changed = fin[:, -1] != native_last
        inc = dist_last[idx, fin[:, -1]] - dist_last[idx, native_last]
        row = {**meta_cols(_S["catalog"], "native", name), "space": space,
               "start": "pre-reassignment" if v.quantizer == "rqvae" else "published",
               "start_last_is_nearest": float((native_last == nearest).mean()),
               "rate_start": r0["collision_rate"], "rate_start_old": r0["collision_rate_old"],
               "rate_after_loop": r1["collision_rate"], "rate_after_loop_old": r1["collision_rate_old"],
               "rate_published": rp["collision_rate"],
               "identical_floor": float(cat["ident_item"].mean()),
               "rounds": rounds, "sinkhorn_nonfinite_groups": n_nan,
               "remaining_identical_buckets": ib, "remaining_identical_items": ii,
               "remaining_other_buckets": ob, "remaining_other_items": oi,
               "share_last_digit_changed": float(changed.mean()),
               "last_level_sq_dist_start_mean": float(dist_last[idx, native_last].mean()),
               "last_level_sq_dist_increase_mean": float(inc.mean()),
               "last_level_sq_dist_increase_changed": float(inc[changed].mean()) if changed.any() else 0.0}
        if v.quantizer == "rqvae":
            row.update(loop_equals_published_items=float((fin == published).all(1).mean()),
                       published_last_changed_share=float((published[:, -1] != native_last).mean()))
        rows.append(row)
        log(f"  {name}: {r0['collision_rate']:.4f} -> {r1['collision_rate']:.4f} "
            f"(published {rp['collision_rate']:.4f}) in {rounds} rounds")
    df = pd.DataFrame(rows)
    df.to_csv(out / "partD_loop.csv", index=False)
    pd.DataFrame(brows).to_csv(out / "partD_remaining_buckets.csv", index=False)
    return df


# =========================================================== hub codes ==
def part_hub(out: Path) -> pd.DataFrame:
    """exp3 A2 on the v3 5-level fits, including 5xW (nested here)."""
    log("Part HUB: near-origin hub codes")
    X64 = E.load(_S["category"], np.float64)
    rows = []
    for K in WIDTHS:
        name5 = f"rqkmeans_5codebook_{K}"
        codes5 = _S["tables"]["native"][name5].codes
        cbs = V.geometry(_S["catalog"], name5)["codebooks"]
        Rz = X64.copy()
        for l in range(FIT_LEVELS):
            c = codes5[:, l]
            cnt = np.bincount(c, minlength=K)
            hub = int(cnt.argmax())
            cn = np.linalg.norm(cbs[l].astype(np.float64), axis=1)
            rn = np.linalg.norm(Rz, axis=1)
            row = {"catalog": _S["catalog"], "width": K, "level": l, "hub_code": hub,
                   "hub_share": float(cnt[hub] / len(c)),
                   "hub_centroid_norm": float(cn[hub]),
                   "median_centroid_norm": float(np.median(cn[cnt > 0])),
                   "hub_norm_rank_from_smallest": int((cn[cnt > 0] < cn[hub]).sum()),
                   "residual_norm_hub_items": float(rn[c == hub].mean()),
                   "residual_norm_other_items": float(rn[c != hub].mean())}
            for D in (3, 4, 5):
                if l == D - 1:
                    coll = E3.collided(E3.keys_of(codes5[:, :D], K))
                    row[f"share_collided_with_hub_last_D{D}"] = float((c[coll] == hub).mean())
                    row[f"share_all_with_hub_last_D{D}"] = float((c == hub).mean())
            rows.append(row)
            Rz = Rz - cbs[l][c].astype(np.float64)
    df = pd.DataFrame(rows)
    df.to_csv(out / "partHUB_hub_codes.csv", index=False)
    return df


# ============================================================ Part C ==
def _c_init(catalog, shared, n_perm, threads, catdir):
    init_state(catalog, shared, n_perm, threads)
    import faiss
    faiss.omp_set_num_threads(threads)
    X64 = E.load(_S["category"], np.float64)
    nd = pd.read_csv(Path(catdir) / "partA_near_duplicate_pairs.csv")
    nd_adj = defaultdict(set)
    for a, b in zip(nd.i, nd.j):
        nd_adj[int(a)].add(int(b))
        nd_adj[int(b)].add(int(a))
    Xc = X64 - X64.mean(0)
    _S["cat_c"] = {"X": X64, "total_ss": float((Xc ** 2).sum()),
                   "knn10": np.load(Path(catdir) / "_partA_knn10.npy"),
                   "nd_adj": nd_adj, "nd_pairs": list(zip(nd.i, nd.j))}
    _S["X32"] = np.ascontiguousarray(X64.astype(np.float32))
    _S["transforms"], _ = E3.make_transforms(X64)


def _c_task(args):
    K, seed = args
    cat, X32, transforms = _S["cat_c"], _S["X32"], _S["transforms"]
    rows = []
    for cond in E3.CONDITIONS:
        t0 = time.time()
        per_level = "per-level" in cond
        seed_fn = (lambda l, s=seed: s + 7919 * l) if per_level else (lambda l, s=seed: s)
        if cond in transforms:
            codes, _, _ = E3.fit_rq(transforms[cond], FIT_LEVELS, K, seed_fn)
        else:
            codes, _, _ = E3.fit_rq(X32, FIT_LEVELS, K, seed_fn,
                                    scale_norm=cond.startswith("scale-norm"))
        for m in E3.refit_metrics(codes, K, cat):
            rows.append({"catalog": _S["catalog"], "condition": cond, "width": K,
                         "seed": seed, "seconds": round(time.time() - t0, 1), **m})
    return rows


def part_c(catalog, out, shared, workers, threads, seeds, widths):
    log("Part C: RQ-KMeans refits")
    tasks = [(K, s) for K in widths for s in seeds]
    ctx = mp.get_context("spawn")
    n = max(1, min(len(tasks), workers // threads))
    with ctx.Pool(n, initializer=_c_init,
                  initargs=(catalog, str(shared), _S["n_perm"], threads, str(out))) as pool:
        rows = [r for chunk in pool.imap_unordered(_c_task, tasks) for r in chunk]
    df = pd.DataFrame(rows).sort_values(["width", "seed", "condition", "depth"])
    df.to_csv(out / "partC_refits.csv", index=False)
    summ = E3.summarise_refits(df.drop(columns=["catalog", "seconds"]))
    summ.insert(0, "catalog", catalog)
    summ.to_csv(out / "partC_summary.csv", index=False)
    # native seed-1234 at 8 threads must equal the tables (Part 0 checked the
    # parent's fit; this checks the workers' fits through their metrics)
    return df, summ


# =========================================================== MQ audit ==
def part_mq(out: Path) -> pd.DataFrame:
    log("Part MQ: model audit")
    rows, digit_rows = [], []
    for name in [x for x in V.names(_S["catalog"]) if x.startswith("MQ")]:
        v = SidVariant.parse(name)
        _, ck = V.mq_checkpoint(_S["catalog"], name)
        aud = pd.DataFrame(V.mq_init_audit(ck["state_dict"], v.codebook_size))
        st = json.loads((V.ROOT / _S["catalog"] / "tokenizers" / name / "geometry" / "stats.json").read_text())
        rows.append({**meta_cols(_S["catalog"], "native", name),
                     "n_params_checked": int(len(aud)),
                     "all_at_init": bool(aud.within_bound.all()),
                     "uniform_std_ratio_min": float(aud.std_over_uniform_std.dropna().min()),
                     "uniform_std_ratio_max": float(aud.std_over_uniform_std.dropna().max()),
                     "best_epoch": st.get("best_epoch"), "best_valid_loss": st.get("best_valid_loss"),
                     "kmeans_empty_cluster_restarts": st.get("kmeans_empty_cluster_restarts"),
                     "codes_used": json.dumps(st.get("codes_used")),
                     "largest_code_share": json.dumps(st.get("largest_code_share"))})
        aud.insert(0, "variant", name)
        digit_rows.append(aud)
    df = pd.DataFrame(rows)
    df.to_csv(out / "partMQ_models.csv", index=False)
    pd.concat(digit_rows).to_csv(out / "partMQ_params.csv", index=False)
    return df


# ============================================================ Part X ==
def part_x(cat: dict, out: Path) -> None:
    log("Part X: exploratory (RQ-VAE latent, collision exposure)")
    catalog = _S["catalog"]
    X32 = np.ascontiguousarray(E.load(_S["category"], np.float32))
    # -- RQ-VAE in its own latent
    lat_rows = []
    for name in [x for x in V.names(catalog) if x.startswith("rqvae")]:
        v = SidVariant.parse(name)
        _, ck = V.rqvae_checkpoint(catalog, name)
        sd = ck["state_dict"]
        z = V.rqvae_encode(sd, X32).astype(np.float64)
        cbs = V.rqvae_codebooks(sd)
        pc, res = V.rqvae_nearest(z, cbs)
        zc = z - z.mean(0)
        tot = float((zc ** 2).sum())
        pub = _S["tables"]["native"][name].codes
        for l, cb in enumerate(cbs):
            used = np.unique(pc[:, l])
            dead = np.setdiff1d(np.arange(len(cb)), used)
            r = res[l].astype(np.float64)
            dmin = np.sqrt(E3.sqdist(cb.astype(np.float64), r).min(1))   # codeword -> nearest residual
            _, inv = np.unique(pc[:, l], return_inverse=True)
            lat_rows.append({
                **meta_cols(catalog, "native", name), "level": l,
                "codes_used_nearest": int(len(used)), "codes_used_published": int(len(np.unique(pub[:, l]))),
                "dead_codewords": int(len(dead)),
                "residual_rms": float(np.sqrt((r ** 2).sum(1).mean())),
                "codeword_norm_median_live": float(np.median(np.linalg.norm(cb[used], axis=1))),
                "codeword_norm_median_dead": float(np.median(np.linalg.norm(cb[dead], axis=1))) if len(dead) else np.nan,
                "dist_to_data_median_live": float(np.median(dmin[used])),
                "dist_to_data_median_dead": float(np.median(dmin[dead])) if len(dead) else np.nan,
                "latent_var_explained_alone": float(F.between_share(z, inv.reshape(-1))),
                # exp3's r2_by_groups returns the BETWEEN-group share (explained)
                "latent_var_explained_by_prefix": float(E3.r2_by_groups(
                    zc, E3.keys_of(pc[:, :l + 1], v.codebook_size), tot)),
                "latent_refinement_index_prefix": R.refinement_index(
                    1.0 - E3.r2_by_groups(zc, E3.keys_of(pc[:, :l + 1], v.codebook_size), tot),
                    len(z), len(np.unique(E3.keys_of(pc[:, :l + 1], v.codebook_size)))),
            })
    pd.DataFrame(lat_rows).to_csv(out / "partX_rqvae_latent.csv", index=False)
    # -- collision exposure of the data and of next-item test targets
    seqs = json.loads((paths.FROZEN_DATA / "interactions" / f"{_S['category']}.inter.json").read_text())
    all_items = np.array([i for s in seqs.values() for i in s], dtype=np.int64)
    tests = [(s[-1], s[-1 - HISTORY:-1]) for s in seqs.values()]
    nd_adj = cat["nd_adj"]
    repeat = np.array([t in set(h) for t, h in tests])
    variant = np.array([(not rp) and any(j in nd_adj.get(t, ()) for j in h)
                        for (t, h), rp in zip(tests, repeat)])
    exp_rows = []
    for kind in ("native", "zcr"):
        for name, tab in _S["tables"][kind].items():
            codes = tab.codes
            K = tab.variant.codebook_size
            keys = E3.keys_of(codes, K)
            coll = E3.collided(keys)
            partner = np.array([(not rp) and any(keys[j] == keys[t] and j != t for j in h)
                                for (t, h), rp in zip(tests, repeat)])
            var_only = variant & ~partner
            tgt = np.array([t for t, _ in tests])
            exp_rows.append({
                **meta_cols(catalog, kind, name), "n_interactions": int(len(all_items)),
                "n_test_targets": int(len(tests)),
                "interactions_on_collided_items": float(coll[all_items].mean()),
                "test_targets_collided": float(coll[tgt].mean()),
                "test_repeat": float(repeat.mean()),
                "test_sid_partner": float(partner.mean()),
                "test_variant": float(var_only.mean()),
                "test_new": float((~repeat & ~partner & ~var_only).mean()),
            })
    pd.DataFrame(exp_rows).to_csv(out / "partX_exposure.csv", index=False)


# =========================================================== Part M1 ==
def _m1_init(catalog, shared, n_perm):
    init_state(catalog, shared, n_perm, 1)
    cat = _S["category"]
    _S["labels"] = {f: L.codes(cat, _S["keys"], f, force=True)[0] for f in PURITY_FIELDS}


def _m1_task(args):
    kind, name, ti = args
    S = _S
    t = S["tables"][kind][name]
    codes = t.codes
    v = t.variant
    D, K = v.n_codebook, v.codebook_size
    zc, knn, total_ss = S["zc"], S["knn"], S["total_ss"]
    n = len(codes)
    mc = meta_cols(S["catalog"], kind, name)
    digit_rows, depth_rows, pair_rows, ami_rows = [], [], [], []
    prof = G.profile(codes, zc)
    ent = R.digit_conditional_entropy(codes)
    con = R.digit_contribution(codes, np.asarray(zc), total_ss)
    util = t.codebook_utilization
    cum = 0.0
    for j in range(D):
        _, inv = np.unique(codes[:, j], return_inverse=True)
        cum += ent[j]["cond_entropy_bits"]
        digit_rows.append({
            **mc, "digit": j + 1, "codes_used": int(inv.max() + 1),
            "utilization": float((inv.max() + 1) / K),
            "norm_entropy": util[j]["entropy_bits"] / util[j]["max_entropy_bits"],
            "largest_code_share": util[j]["max_count"] / n,
            "var_alone": F.between_share(np.asarray(zc), inv.reshape(-1)),
            "var_given_prefix": prof[j].r2_within - prof[j + 1].r2_within,
            "cond_entropy_bits": ent[j]["cond_entropy_bits"],
            "effective_branching": ent[j]["effective_branching"],
            "cum_bits": cum, "identity_budget_bits": float(np.log2(n)),
            "refinement_without": con[j]["refinement_without"],
            "refinement_lost_if_deleted": con[j]["refinement_delta"],
        })
    rng = np.random.default_rng([PERM_SEED, ti])
    prev_groups = 1
    for s in prof:
        d = s.depth
        row = {**mc, **s.as_dict(), "r2_null": R.null_r2(n, s.n_clusters),
               "refinement_index": R.refinement_index(s.r2_within, n, s.n_clusters)}
        if d:
            npres = R.neighbour_preservation(codes, knn, d)
            row.update(knn_observed=npres["observed"], knn_chance=npres["chance"], knn_lift=npres["lift"],
                       capacity=float(K) ** d, occupied_share=s.n_clusters / float(K) ** d,
                       mean_children=s.n_clusters / prev_groups,
                       cum_bits=digit_rows[d - 1]["cum_bits"])
            group, counts = F.prefix_groups(codes, d)
            for f in PURITY_FIELDS:
                lab = S["labels"][f]
                res = F.purity_multi(group, counts, lab, rng, S["n_perm"])
                for k_, v_ in res.items():
                    row[f"{f}_{k_}"] = v_
                row[f"{f}_purity_all_items"] = F.item_weighted_purity(group, lab)
            prev_groups = s.n_clusters
        depth_rows.append(row)
    for p in R.digit_pair_mi(codes):
        pair_rows.append({**mc, **p})
    for f in AMI_FIELDS[S["catalog"]]:
        y = S["labels"][f]
        for r in AT.alignment(t, y, S["keys"], f, n_perm=S["n_perm"]):
            r.pop("variant", None)
            r.pop("quantizer", None)
            ami_rows.append({**mc, **r})
    return digit_rows, depth_rows, pair_rows, ami_rows


def part_m1(catalog, out, shared, workers, quick=False):
    log("Part M1: what each digit encodes")
    tasks = [(k, n) for k in ("native", "zcr") for n in _S["tables"][k]]
    if quick:
        tasks = tasks[:3] + [t for t in tasks if t[0] == "zcr"][:1]
    tasks = [(k, n, i) for i, (k, n) in enumerate(tasks)]
    ctx = mp.get_context("spawn")
    res = {"digits": [], "depths": [], "pairs": [], "ami": []}
    with ctx.Pool(min(workers, len(tasks)), initializer=_m1_init,
                  initargs=(catalog, str(shared), _S["n_perm"])) as pool:
        for i, (a, b, c, d) in enumerate(pool.imap_unordered(_m1_task, tasks)):
            res["digits"] += a
            res["depths"] += b
            res["pairs"] += c
            res["ami"] += d
            log(f"  M1 {i + 1}/{len(tasks)} done ({a[0]['kind']} {a[0]['variant']})")
    for k, rows in res.items():
        pd.DataFrame(rows).sort_values(["kind", "variant"]).to_csv(out / f"partM1_{k}.csv", index=False)


# ============================================================ Part Z ==
def part_z(out: Path) -> pd.DataFrame:
    log("Part Z: ZCR twins")
    rows = []
    for name, zt in _S["tables"]["zcr"].items():
        nt = _S["tables"]["native"][name]
        dist, space = last_level_distances(name)
        idx = np.arange(len(dist))
        nl, zl = nt.codes[:, -1], zt.codes[:, -1]
        st = V.zcr_bundle(_S["catalog"], name)["stats"]
        z = st.get("zcr", {})
        before, after = float(dist[idx, nl].sum()), float(dist[idx, zl].sum())
        rows.append({**meta_cols(_S["catalog"], "zcr", name), "space": space,
                     "native_collision_rate": E3.rates(nt.codes, nt.variant.codebook_size)["collision_rate"],
                     "changed_tail_share": float((nl != zl).mean()),
                     "sq_dist_before": before, "sq_dist_after": after,
                     "sq_dist_rel_increase": after / before - 1.0,
                     "recorded_distance_before": z.get("distance_before"),
                     "recorded_distance_after": z.get("distance_after"),
                     "native_last_is_nearest": float((nl == dist.argmin(1)).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "partZ_zcr.csv", index=False)
    return df


# =========================================================== Part M5 ==
def _m5_init(catalog, shared, n_perm):
    init_state(catalog, shared, n_perm, 1)
    m = M.load(_S["category"])
    _S["titles"] = {k: m[k].title for k in _S["keys"] if k in m}


def _m5_task(name):
    t = _S["tables"]["native"][name]
    nm = NM.describe_cells(t, _S["titles"])
    nmd = NM.depth_summary(nm)
    cop = AT.copurchase_preservation(t, _S["keys"], _S["category"])
    mc = meta_cols(_S["catalog"], "native", name)
    return ([{**mc, **{k: v for k, v in r.items() if k not in ("variant", "quantizer")}} for r in nmd],
            [{**mc, **{k: v for k, v in r.items() if k not in ("variant", "quantizer")}} for r in cop])


def part_m5(catalog, out, shared, workers, quick=False):
    log("Part M5: naming and co-purchase")
    names = list(_S["tables"]["native"])
    if quick:
        names = names[:2]
    ctx = mp.get_context("spawn")
    nrows, crows = [], []
    with ctx.Pool(min(workers, len(names)), initializer=_m5_init,
                  initargs=(catalog, str(shared), _S["n_perm"])) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(_m5_task, names)):
            nrows += a
            crows += b
            log(f"  M5 {i + 1}/{len(names)}")
    pd.DataFrame(nrows).sort_values(["variant", "digit"]).to_csv(out / "partM5_naming.csv", index=False)
    pd.DataFrame(crows).sort_values(["variant", "depth"]).to_csv(out / "partM5_copurchase.csv", index=False)


# ============================================================ Part R ==
def part_r(out: Path) -> None:
    """Industrial only: v3 tables against the frozen tables of the same name."""
    from sklearn.metrics import adjusted_rand_score
    log("Part R: v3 against the frozen tables")
    rows = []
    for name, t in _S["tables"]["native"].items():
        old = SidTable.load(name)
        oc = np.array([old.asin2codes[k] for k in t.keys], dtype=np.int64)
        nc = t.codes
        K = t.variant.codebook_size
        row = {"variant": name, "same_full_sid": float((oc == nc).all(1).mean()),
               "rate_old": E3.rates(oc, K)["collision_rate"], "rate_v3": E3.rates(nc, K)["collision_rate"]}
        for d in range(nc.shape[1]):
            row[f"digit{d + 1}_agree"] = float((oc[:, d] == nc[:, d]).mean())
            row[f"prefix{d + 1}_ari"] = float(adjusted_rand_score(E3.keys_of(oc[:, :d + 1], K),
                                                                   E3.keys_of(nc[:, :d + 1], K)))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "partR_old_vs_v3.csv", index=False)


# ============================================================== main ==
def write_inputs(out: Path) -> None:
    cat, catalog = _S["category"], _S["catalog"]
    files = sorted(p for p in (V.ROOT / catalog).rglob("*") if p.is_file())
    doc = {"dataset": V.REPO_ID, "revision": V.REVISION, "root": str(V.ROOT / catalog),
           "codebook_files": {str(p.relative_to(V.ROOT)): sha256_file(p) for p in files},
           "embedding": sha256_file(paths.FROZEN_DATA / "embeddings" / f"{cat}.emb-qwen-td.npy"),
           "item2id": sha256_file(paths.FROZEN_DATA / "id_maps" / f"{cat}.item2id"),
           "item_meta": sha256_file(paths.FROZEN_DATA / "item_meta" / f"{cat}.item.json"),
           "interactions": sha256_file(paths.FROZEN_DATA / "interactions" / f"{cat}.inter.json"),
           "labels": sha256_file(paths.EXTERNAL / "labels" / f"{cat}.labels.json"),
           "labels_manifest": json.loads((paths.EXTERNAL / "labels" / f"{cat}.labels.manifest.json").read_text())["source"],
           "exp3_run_py": sha256_file(EXP3_RUN), "figures_py": sha256_file(FIGURES)}
    (out / "inputs.json").write_text(json.dumps(doc, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalog", required=True, choices=sorted(V.CATALOGS))
    ap.add_argument("--out", type=Path, required=True, help="parent dir; <catalog>/ must not exist")
    ap.add_argument("--parts", default=",".join(ALL_PARTS))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--threads", type=int, default=8, help="faiss OpenMP threads (v3 build: 8)")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--quick", action="store_true", help="smoke run: few tables, seeds, permutations")
    args = ap.parse_args(argv)
    parts = [p for p in args.parts.split(",") if p]
    bad = set(parts) - set(ALL_PARTS)
    if bad:
        raise SystemExit(f"unknown parts {bad}")
    out = args.out / args.catalog
    out.mkdir(parents=True, exist_ok=False)
    n_perm = 5 if args.quick else args.n_perm
    t_start = time.time()
    init_state(args.catalog, str(out / "_shared"), n_perm, min(args.threads, args.workers))
    E3.PERM_SEED = PERM_SEED          # Part B reads it at call time
    write_inputs(out)
    log(f"{args.catalog}: {len(_S['tables']['native'])} native, {len(_S['tables']['zcr'])} ZCR tables; "
        f"parts {parts}; workers {args.workers}; n_perm {n_perm}")
    shared = prepare_shared(args.catalog, out)
    val = part0(args.catalog, out, args.threads) if "0" in parts else None

    cat = None
    if any(p in parts for p in ("A", "B", "D", "C", "X")):
        raw, titles, brands = item_order_inputs()
        _, cat = E3.part_a(raw, titles, brands, out)
        np.save(out / "_partA_knn10.npy", cat["knn10"])
    if "B" in parts:
        E3.part_b(native_codes(), cat, out, n_perm)
    if "D" in parts:
        if "pre" not in _S:
            raise SystemExit("Part D needs Part 0 (pre-reassignment RQ-VAE tables)")
        part_d(cat, out)
    if "HUB" in parts:
        part_hub(out)
    if "C" in parts:
        # Amendment A1: the v3 tables are not reproducible by refitting on this
        # hardware, so Part C is a within-run counterfactual whatever Part 0
        # says. The flag travels with the output instead of gating it.
        seeds = SEEDS[:2] if args.quick else SEEDS
        widths = WIDTHS[:1] if args.quick else WIDTHS
        log(f"Part C: native refit reproduces the published tables: "
            f"{None if val is None else val['rqkmeans_refit_reproduces_all']} (amendment A1)")
        part_c(args.catalog, out, shared, args.workers, args.threads, seeds, widths)
    if "MQ" in parts:
        part_mq(out)
    if "X" in parts:
        part_x(cat, out)
    if "M1" in parts:
        part_m1(args.catalog, out, shared, args.workers, args.quick)
    if "Z" in parts:
        part_z(out)
    if "M5" in parts:
        part_m5(args.catalog, out, shared, args.workers, args.quick)
    if "R" in parts and args.catalog == "industrial":
        part_r(out)
    (out / "result.json").write_text(json.dumps({
        "catalog": args.catalog, "parts": parts, "n_perm": n_perm, "perm_seed": PERM_SEED,
        "quick": args.quick, "threads": args.threads, "workers": args.workers,
        "seconds": round(time.time() - t_start, 1), "python": platform.python_version(),
        "primary_field": PRIMARY_FIELD[args.catalog], "ami_fields": AMI_FIELDS[args.catalog],
        "validation": val}, indent=2))
    for p in (shared / "zc.npy", shared / "knn.npy"):
        p.unlink()                      # 362 MB of recomputable floats; keep the rest
    log(f"done in {time.time() - t_start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
