#!/usr/bin/env python
"""Summarize structure exp4: merge both catalogs, score the declared
expectations, and recompute every RQ1 number the current draft quotes.

Reads only the CSV/JSON outputs of ``run.py`` under ``--run`` and writes
``<run>/summary/``:

    <part>.csv           both catalogs concatenated (one file per part)
    expectations.csv     E1-E11 per catalog: held / failed, the rule, the tables that fail
    claims.csv           each RQ1 claim of the 2026-10-05 draft: old value, v3 value per catalog
    collision_causes.csv the paper's collision-cause table, per catalog and family
    report.md            all of the above, readable

Traps this guards against:

* Two collision definitions. ``collision_rate`` is the share of items in a
  shared SID (the paper's); ``collision_rate_old`` is 1 - unique/N. Every
  number below says which.
* Digit indexing. ``attribute.alignment`` numbers digits from 0; every other
  table from 1. Converted here, once.
* A declared rule scored on a different population. Each rule below is the
  protocol's wording turned into code; where the protocol is ambiguous the
  choice is written next to the rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATALOGS = ("industrial", "office")
PRIMARY = {"industrial": "cat_l1", "office": "cat_l2"}
FAMILY = {"rqkmeans": "RQ-KMeans", "rqvae": "RQ-VAE", "MQ": "MQ"}


def load(run: Path, name: str) -> pd.DataFrame:
    frames = []
    for c in CATALOGS:
        p = run / c / name
        if p.exists():
            df = pd.read_csv(p)
            if "catalog" not in df.columns:
                df.insert(0, "catalog", c)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def vname(q: str, d: int, w: int) -> str:
    return f"{q}_{d}codebook_{w}"


def pct(x: float, nd: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f} %"


def rng_pct(s: pd.Series, nd: int = 1) -> str:
    s = s.dropna()
    return "n/a" if s.empty else f"{100 * s.min():.{nd}f}–{100 * s.max():.{nd}f} %"


def rng(s: pd.Series, nd: int = 2) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    sep = " to " if s.min() < 0 else "–"
    return f"{s.min():.{nd}f}{sep}{s.max():.{nd}f}"


# ------------------------------------------------------------- expectations --
def expectations(P: dict) -> pd.DataFrame:
    rows = []

    def add(eid, cat, rule, ok, failing, note=""):
        rows.append({"expectation": eid, "catalog": cat, "held": bool(ok),
                     "rule": rule, "failing": "; ".join(failing) if failing else "", "note": note})

    dig, dep = P["m1_digits"], P["m1_depths"]
    nat_d = dig[dig.kind == "native"] if not dig.empty else dig
    nat_p = dep[dep.kind == "native"] if not dep.empty else dep
    for cat in (CATALOGS if not dig.empty else ()):
        d = nat_d[nat_d.catalog == cat]
        p = nat_p[nat_p.catalog == cat]
        if d.empty:
            continue
        # E1
        fail = []
        for v, g in d.groupby("variant"):
            g = g.sort_values("digit")
            q = g.quantizer.iloc[0]
            if q in ("rqkmeans", "rqvae"):
                if not g.var_alone.iloc[0] > g.var_alone.iloc[1:].max():
                    fail.append(v)
            elif g.var_alone.max() / g.var_alone.min() >= 1.5:
                fail.append(f"{v} ({g.var_alone.max() / g.var_alone.min():.2f}x)")
        add("E1", cat, "residual: digit 1 alone explains more variance than any later digit alone; "
            "MQ: largest/smallest single-digit share < 1.5", not fail, fail)
        # E2 (Industrial cat_l1 declared; Office reported without expectation)
        if cat == "industrial":
            fail = []
            for _, r in p[(p.depth == 3) & p.quantizer.isin(["rqkmeans", "rqvae"]) & (p.n_codebook >= 4)].iterrows():
                if not r.cat_l1_purity >= 0.95:
                    fail.append(f"{r.variant} ({r.cat_l1_purity:.3f}, multi-item share "
                                f"{r.cat_l1_multi_item_share:.3f})")
            for D in (3, 4, 5):
                for W in (128, 256, 512):
                    mq = p[(p.variant == vname("MQ", D, W)) & (p.depth == 3)].cat_l1_purity
                    rk = p[(p.variant == vname("rqkmeans", D, W)) & (p.depth == 3)].cat_l1_purity
                    if len(mq) and len(rk) and not mq.iloc[0] < rk.iloc[0]:
                        fail.append(f"MQ {D}x{W} not below RQ-KMeans ({mq.iloc[0]:.3f} vs {rk.iloc[0]:.3f})")
            add("E2", cat, "cat_l1 purity at prefix length 3 >= 0.95 in every residual table with D >= 4, "
                "and MQ < RQ-KMeans at the same (D, W)", not fail, fail)
        # E3
        r1 = d[(d.quantizer == "rqvae") & (d.digit == 1)]
        k = int((r1.utilization <= 0.5).sum())
        add("E3", cat, "RQ-VAE uses <= 50 % of its first codebook in >= 6 of 9 tables", k >= 6,
            [f"{r.variant} ({r.utilization:.2f})" for _, r in r1.iterrows() if r.utilization > 0.5],
            f"{k}/9 tables at <= 50 %")
        # E4
        last = p[(p.quantizer == "rqvae") & (p.depth == p.n_codebook)]
        fail = [f"{r.variant} ({r.frac_items_in_singletons:.3f})" for _, r in last.iterrows()
                if r.frac_items_in_singletons < 0.95]
        add("E4", cat, ">= 95 % of items alone after the last digit in every RQ-VAE table", not fail, fail)

    D = P["d_loop"]
    for cat in (CATALOGS if not D.empty else ()):
        g = D[D.catalog == cat]
        if g.empty:
            continue
        rv = g[g.quantizer == "rqvae"]
        fail = [r.variant for _, r in rv.iterrows() if not r.rate_start > r.rate_published]
        near = (rv.rate_published - rv.identical_floor) <= 0.005
        ok = not fail and int(near.sum()) >= 7
        add("E5", cat, "every RQ-VAE pre-reassignment rate > published rate, and published within "
            "0.5 pp of the identical-input floor in >= 7 of 9", ok,
            fail + [f"{r.variant} published {100 * r.rate_published:.2f} % vs floor {100 * r.identical_floor:.2f} %"
                    for _, r in rv[~near].iterrows()], f"{int(near.sum())}/9 within 0.5 pp")
        o = g[g.quantizer.isin(["rqkmeans", "MQ"])]
        bad = o[(o.rate_after_loop - o.identical_floor) > 0.005]
        add("E6", cat, "the reassignment loop brings every RQ-KMeans and MQ table within 0.5 pp of the floor",
            bad.empty, [f"{r.variant} {100 * r.rate_after_loop:.2f} %" for _, r in bad.iterrows()])

    B = P["b_tables"]
    for cat in (CATALOGS if not B.empty else ()):
        g = B[(B.catalog == cat)]
        if g.empty:
            continue
        fail = []
        for W in (128, 256, 512):
            s = [g[g.table == vname("rqkmeans", D_, W)].nd_share_collided.iloc[0] for D_ in (3, 4, 5)]
            if not (s[0] < s[1] < s[2]):
                fail.append(f"W={W} not rising ({', '.join(f'{x:.2f}' for x in s)})")
            if not s[2] > 0.5:
                fail.append(f"5x{W} share {s[2]:.2f}")
        add("E7", cat, "nested RQ-KMeans: near-duplicate share of collided items rises with depth at each "
            "width and > 0.5 at 5xW", not fail, fail)
        mq = g[g.quantizer == "MQ"]
        fail = [f"{r.table} ({r.obs_over_n1:.1f}x)" for _, r in mq.iterrows() if not r.obs_over_n1 >= 2]
        add("E8", cat, "every MQ table's collision rate >= 2x its independent-digit null (N1 median)",
            not fail, fail)

    H = P["hub"]
    for cat in (CATALOGS if not H.empty else ()):
        g = H[(H.catalog == cat) & (H.level >= 1)]
        if g.empty:
            continue
        fail = [f"W={r.width} level {r.level + 1} rank {r.hub_norm_rank_from_smallest}"
                for _, r in g.iterrows() if r.hub_norm_rank_from_smallest > 1]
        add("E9", cat, "at every level >= 2 of every 5-level RQ-KMeans fit, the hub code's centroid has "
            "norm rank <= 1", not fail, fail)

    C = P["c_summary"]
    for cat in (CATALOGS if not C.empty else ()):
        g = C[(C.catalog == cat) & (C.condition == "per-level seed") & (C.depth == 3)]
        if g.empty:
            continue
        fail = [f"W={r.width}: {100 * r.paired_diff_mean:+.1f} pp, disjoint={r.ranges_disjoint}"
                for _, r in g.iterrows() if not (r.paired_diff_mean > 0 and r.ranges_disjoint)]
        add("E10", cat, "per-level seed raises the depth-3 collision rate at every width, seed ranges "
            "disjoint (counterfactual on this cluster's refit, amendment A1)", not fail, fail)

    MQ = P["mq_models"]
    for cat in (CATALOGS if not MQ.empty else ()):
        g = MQ[MQ.catalog == cat]
        if g.empty:
            continue
        fail = list(g[~g.all_at_init.astype(bool)].variant)
        add("E11", cat, "MQ encoders and pos are at their initialisation in every MQ model", not fail, fail)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ claims --
def claims(P: dict) -> pd.DataFrame:
    """Each RQ1 number of the 2026-10-05 draft (sections/05results.tex), recomputed."""
    dig, dep, B, D = P["m1_digits"], P["m1_depths"], P["b_tables"], P["d_loop"]
    A = P["a_catalogue"]
    nat_d, nat_p = dig[dig.kind == "native"], dep[dep.kind == "native"]
    rows = []

    def add(cid, text, old, per_cat):
        r = {"claim": cid, "draft_text": text, "old_value_industrial": old}
        for c in CATALOGS:
            try:
                r[f"v3_{c}"] = per_cat(c)
            except Exception as e:                      # a missing part says so
                r[f"v3_{c}"] = f"unavailable ({type(e).__name__})"
        rows.append(r)

    def d_of(c, q, D_, W, digit):
        s = nat_d[(nat_d.catalog == c) & (nat_d.variant == vname(q, D_, W)) & (nat_d.digit == digit)]
        return s.iloc[0]

    def p_of(c, q, D_, W, depth):
        s = nat_p[(nat_p.catalog == c) & (nat_p.variant == vname(q, D_, W)) & (nat_p.depth == depth)]
        return s.iloc[0]

    add("var_digit1_4x128", "digit 1 alone explains 53 % (RQ-KMeans) and 34 % (RQ-VAE) of the variance, 4x128",
        "53 % / 34 %",
        lambda c: f"{pct(d_of(c, 'rqkmeans', 4, 128, 1).var_alone)} / {pct(d_of(c, 'rqvae', 4, 128, 1).var_alone)}")
    add("var_later_residual_4x128", "each later residual digit 8-18 %, 4x128", "8–18 %",
        lambda c: rng_pct(nat_d[(nat_d.catalog == c) & nat_d.variant.isin([vname('rqkmeans', 4, 128),
                                                                            vname('rqvae', 4, 128)])
                                & (nat_d.digit > 1)].var_alone, 0))
    add("var_mq_4x128", "every MQ digit explains 36-38 % on its own, 4x128", "36–38 %",
        lambda c: rng_pct(nat_d[(nat_d.catalog == c) & (nat_d.variant == vname('MQ', 4, 128))].var_alone, 0))
    add("var_mq_all", "MQ single-digit shares, all 9 MQ tables (new)", "n/a",
        lambda c: rng_pct(nat_d[(nat_d.catalog == c) & (nat_d.quantizer == 'MQ')].var_alone, 0))
    add("refinement_5x128", "five MQ digits reach a refinement index of 0.80 and five RQ-KMeans digits 0.95 (5x128)",
        "0.80 / 0.95",
        lambda c: f"{p_of(c, 'MQ', 5, 128, 5).refinement_index:.2f} / {p_of(c, 'rqkmeans', 5, 128, 5).refinement_index:.2f}")
    add("purity_len3_4x128", "items sharing three digits: group's main category 97-99 % (residual) and 81 % (MQ); "
        "shuffled 39-50 % (primary field: Industrial cat_l1, Office cat_l2)", "97–99 % / 81 %; null 39–50 %",
        lambda c: "RQ-KMeans {} RQ-VAE {} MQ {}; null {}".format(
            *[pct(p_of(c, q, 4, 128, 3)[f"{PRIMARY[c]}_purity"]) for q in ("rqkmeans", "rqvae", "MQ")],
            rng_pct(pd.Series([p_of(c, q, 4, 128, 3)[f"{PRIMARY[c]}_null_mean"] for q in ("rqkmeans", "rqvae", "MQ")]), 0)))
    add("rqvae_first_codebook", "RQ-VAE uses only 2-31 % of its first codebook in eight of nine maps", "2–31 % in 8/9",
        lambda c: (lambda s: f"{rng_pct(s, 0)}; {int((s <= 0.5).sum())}/9 at <= 50 %")(
            nat_d[(nat_d.catalog == c) & (nat_d.quantizer == 'rqvae') & (nat_d.digit == 1)].utilization))
    add("rqvae_alone_after_4", "after four digits 99 % of RQ-VAE items have a SID of their own (4x128)", "99 %",
        lambda c: pct(p_of(c, 'rqvae', 4, 128, 4).frac_items_in_singletons))
    add("collision_3x128", "44.8 % (RQ-KMeans) and 54.5 % (MQ) of items share a SID at 3x128", "44.8 % / 54.5 %",
        lambda c: " / ".join(pct(B[(B.catalog == c) & (B.table == vname(q, 3, 128))].collision_rate.iloc[0])
                             for q in ("rqkmeans", "MQ")))
    add("unique_prefix_rqvae_4x128", "RQ-VAE 4x128 items alone at prefix length 3 (the plan's P5 wording; the "
        "draft's 92.5 % is a different statistic, recomputed in compare/claims.csv)",
        "79.6 % (old map, paper data package)", lambda c: pct(p_of(c, 'rqvae', 4, 128, 3).frac_items_in_singletons))
    add("identical_floor", "eleven pairs of items have identical embeddings (0.71 %)", "11 pairs, 0.71 %",
        lambda c: "{} items in {} groups ({})".format(A[c]["n_identical_input_items"], A[c]["n_identical_input_groups"],
                                                      pct(A[c]["n_identical_input_items"] / A[c]["n_items"], 2)))
    add("identical_split", "every map except one gives each identical pair a single SID", "26 of 27",
        lambda c: (lambda b: f"{int((b.identical_pairs_split == 0).sum())} of {len(b)} split none")(
            B[B.catalog == c].iloc[np.arange(27)]))
    add("nd_share_deepest", "73-85 % of collided items at the deepest nested RQ-KMeans fits have a near-duplicate "
        "bucket-mate", "73–85 % (5x128, 5x256, 4x512)",
        lambda c: ", ".join(f"5x{W} {pct(B[(B.catalog == c) & (B.table == vname('rqkmeans', 5, W))].nd_share_collided.iloc[0], 0)}"
                            for W in (128, 256, 512)))
    add("loop_rqkmeans", "applying RQ-VAE's reassignment to RQ-KMeans takes every table from 10-45 % to the 0.71 % floor",
        "10–45 % → 0.71 %",
        lambda c: (lambda g: f"{rng_pct(g.rate_start)} → {rng_pct(g.rate_after_loop, 2)} (floor {pct(g.identical_floor.iloc[0], 2)})")(
            D[(D.catalog == c) & (D.quantizer == 'rqkmeans')]))
    add("loop_changes", "changing the last digit of up to 44 % of items", "up to 44 %",
        lambda c: pct(D[(D.catalog == c) & (D.quantizer == 'rqkmeans')].share_last_digit_changed.max(), 0))
    add("rqvae_pre_vs_published", "RQ-VAE pre-reassignment rate → published rate (new, direct)", "not measurable",
        lambda c: (lambda g: f"{rng_pct(g.rate_start)} → {rng_pct(g.rate_published, 2)}; last digit rewritten for "
                             f"{rng_pct(g.share_last_digit_changed, 0)} of items")(D[(D.catalog == c) & (D.quantizer == 'rqvae')]))
    add("hub_share", "the code nearest the origin collects 9-17 % of all items at every level after the first",
        "9–17 %", lambda c: rng_pct(P["hub"][(P["hub"].catalog == c) & (P["hub"].level >= 1)].hub_share, 0))
    add("radius_spearman", "parents with a smaller radius collide more (Spearman -0.41 to -0.68)", "−0.41 to −0.68",
        lambda c: rng(B[(B.catalog == c) & (B.quantizer == 'rqkmeans')].radius_spearman))
    add("mq_nmi", "MQ digits: mean pairwise NMI 0.50-0.70", "0.50–0.70",
        lambda c: rng(B[(B.catalog == c) & (B.quantizer == 'MQ')].mean_pairwise_nmi))
    add("mq_n1_3x128", "MQ 3x128 would collide 7.4 % with independent digits instead of 54.5 %", "54.5 % vs 7.4 %",
        lambda c: (lambda r: f"{pct(r.collision_rate)} vs {pct(r.n1_rate_p50)}")(
            B[(B.catalog == c) & (B.table == vname('MQ', 3, 128))].iloc[0]))
    add("capacity_3x128", "uniform codes would give at most 0.15 % at 3x128", "0.15 %",
        lambda c: pct(B[(B.catalog == c) & (B.table == vname('rqkmeans', 3, 128))].n0_rate.iloc[0], 2))
    add("rqkmeans_5x512", "RQ-KMeans 5x512 is a Sinkhorn-balanced table, 0.64 %", "0.64 %, balanced",
        lambda c: (lambda r: f"{pct(r.collision_rate, 2)}, balanced={bool(r.all_levels_balanced)}, "
                             f"identical pairs split {int(r.identical_pairs_split)}")(
            B[(B.catalog == c) & (B.table == vname('rqkmeans', 5, 512))].iloc[0]))
    Z = P["zcr"]
    add("zcr_3x128", "ZCR changes the last digit of 910 items (29 %) at RQ-KMeans 3x128", "910 (29 %)",
        lambda c: (lambda r: f"{int(round(r.changed_tail_share * A[c]['n_items']))} ({pct(r.changed_tail_share, 0)}); "
                             f"last-level squared distance +{100 * r.sq_dist_rel_increase:.1f} %")(
            Z[(Z.catalog == c) & (Z.variant == vname('rqkmeans', 3, 128))].iloc[0]))
    C = P["c_summary"]
    add("per_level_seed_3x128", "the reused k-means seed lowers collisions (+13.0 points at 3x128 without it)",
        "+13.0 pp",
        lambda c: (lambda r: f"{100 * r.paired_diff_mean:+.1f} pp (disjoint={bool(r.ranges_disjoint)}; this cluster's refit)")(
            C[(C.catalog == c) & (C.condition == 'per-level seed') & (C.width == 128) & (C.depth == 3)].iloc[0]))
    return pd.DataFrame(rows)


def collision_causes(P: dict) -> pd.DataFrame:
    """One row per catalog and family: the paper's Table (collision causes)."""
    B, D, H, A = P["b_tables"], P["d_loop"], P["hub"], P["a_catalogue"]
    rows = []
    for c in CATALOGS:
        b, d = B[B.catalog == c], D[D.catalog == c]
        if b.empty:
            continue
        n = A[c]["n_items"]
        rows.append({"catalog": c, "maps": "All",
                     "evidence": f"floor: {A[c]['n_identical_input_items']} identical-input items "
                                 f"({100 * A[c]['n_identical_input_items'] / n:.2f} %); "
                                 f"{int((b.identical_pairs_split == 0).sum())}/27 maps split no identical pair; "
                                 f"near-duplicate share of collided items at the deepest RQ-KMeans fits "
                                 f"{rng_pct(b[b.table.isin([vname('rqkmeans', 5, W) for W in (128, 256, 512)])].nd_share_collided, 0)}"})
        rv = d[d.quantizer == "rqvae"]
        rk = d[d.quantizer == "rqkmeans"]
        rows.append({"catalog": c, "maps": "RQ-VAE",
                     "evidence": f"last-digit reassignment: {rng_pct(rv.rate_start)} before → "
                                 f"{rng_pct(rv.rate_published, 2)} published; rewrites "
                                 f"{rng_pct(rv.share_last_digit_changed, 0)} of last digits; applied to RQ-KMeans "
                                 f"{rng_pct(rk.rate_start)} → {rng_pct(rk.rate_after_loop, 2)}"})
        hub = H[(H.catalog == c) & (H.level >= 1)]
        rows.append({"catalog": c, "maps": "RQ-KMeans",
                     "evidence": f"hub code (smallest centroid norm at {int((hub.hub_norm_rank_from_smallest == 0).sum())}"
                                 f"/{len(hub)} levels >= 2) holds {rng_pct(hub.hub_share, 0)} of items; radius Spearman "
                                 f"{rng(b[b.quantizer == 'rqkmeans'].radius_spearman)}"})
        mq = b[b.quantizer == "MQ"]
        r3 = mq[mq.table == vname("MQ", 3, 128)]
        pa = P["m1_pairs"]
        pa = pa[(pa.catalog == c) & (pa.kind == "native")].groupby(["quantizer", "variant"]).ami.mean()
        rows.append({"catalog": c, "maps": "MQ",
                     "evidence": f"dependent digits (per-map mean pairwise AMI {rng(pa['MQ'])} against "
                                 f"{rng(pd.concat([pa['rqkmeans'], pa['rqvae']]))} for residual maps); 3x128: "
                                 f"{pct(r3.collision_rate.iloc[0]) if len(r3) else 'n/a'} observed, "
                                 f"{pct(r3.n1_rate_p50.iloc[0]) if len(r3) else 'n/a'} with independent digits; "
                                 f"encoders untrained (random projections)"})
        rows.append({"catalog": c, "maps": "Not causes",
                     "evidence": f"capacity (uniform codes {pct(b[b.table == vname('rqkmeans', 3, 128)].n0_rate.iloc[0], 2)} "
                                 f"at 3x128); see Part C for seed, norm and whitening counterfactuals"})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols=None) -> str:
    if df.empty:
        return "(none)\n"
    df = df[cols] if cols else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in df.itertuples(index=False))
    return head + body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="the job directory holding <catalog>/")
    ap.add_argument("--out", type=Path, default=None, help="default <run>/summary (must not exist)")
    args = ap.parse_args(argv)
    out = args.out or args.run / "summary"
    out.mkdir(parents=True, exist_ok=False)
    files = {"m1_digits": "partM1_digits.csv", "m1_depths": "partM1_depths.csv", "m1_pairs": "partM1_pairs.csv",
             "m1_ami": "partM1_ami.csv", "b_tables": "partB_tables.csv", "b_levels": "partB_levels.csv",
             "b_rqvae_buckets": "partB_rqvae_buckets.csv", "d_loop": "partD_loop.csv",
             "d_buckets": "partD_remaining_buckets.csv", "hub": "partHUB_hub_codes.csv",
             "c_refits": "partC_refits.csv", "c_summary": "partC_summary.csv", "mq_models": "partMQ_models.csv",
             "x_latent": "partX_rqvae_latent.csv", "x_exposure": "partX_exposure.csv", "zcr": "partZ_zcr.csv",
             "m5_naming": "partM5_naming.csv", "m5_copurchase": "partM5_copurchase.csv",
             "r_old_vs_v3": "partR_old_vs_v3.csv", "v_zcr": "part0_zcr_checks.csv", "v_nesting": "part0_nesting.csv",
             "v_refit": "part0_rqkmeans_refit.csv", "v_rqvae": "part0_rqvae_checkpoints.csv",
             "v_mq": "part0_mq_models.csv"}
    P = {k: load(args.run, f) for k, f in files.items()}
    if not P["m1_ami"].empty:
        P["m1_ami"]["digit"] = P["m1_ami"]["digit"] + 1         # alignment() counts from 0
    P["a_catalogue"] = {c: json.loads((args.run / c / "partA_catalogue.json").read_text())
                        for c in CATALOGS if (args.run / c / "partA_catalogue.json").exists()}
    validation = {c: json.loads((args.run / c / "validation.json").read_text())
                  for c in CATALOGS if (args.run / c / "validation.json").exists()}
    for k, df in P.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.to_csv(out / f"{k}.csv", index=False)
    ex = expectations(P)
    ex.to_csv(out / "expectations.csv", index=False)
    cl = claims(P)
    cl.to_csv(out / "claims.csv", index=False)
    cc = collision_causes(P)
    cc.to_csv(out / "collision_causes.csv", index=False)
    (out / "validation.json").write_text(json.dumps(validation, indent=2))

    rep = [f"# Structure exp4 summary: the final (v3) SID maps\n",
           f"Run: `{args.run}`. Generated by `experiments/structure/exp4_final_maps/summarize.py` "
           f"from the run's CSVs only. Collision rate = share of items whose full SID is shared.\n",
           "## Validation (Part 0)\n"]
    for c, v in validation.items():
        rep.append(f"- **{c}**: " + ", ".join(f"{k}={v[k]}" for k in v if isinstance(v[k], bool)) + "\n")
    rep += ["\n## Expectations (declared in protocol.md)\n", md_table(ex[["expectation", "catalog", "held", "failing", "note"]]),
            "\nRules:\n"]
    for e, g in ex.groupby("expectation", sort=False):
        rep.append(f"- {e}: {g.rule.iloc[0]}\n")
    rep += ["\n## RQ1 numbers of the current draft, recomputed\n", md_table(cl),
            "\n## Why items share a SID\n", md_table(cc)]
    (out / "report.md").write_text("".join(rep))
    print(f"wrote {out}")
    print(ex[["expectation", "catalog", "held", "failing"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
