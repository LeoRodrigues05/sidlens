#!/usr/bin/env python
"""HTML report and fact sheet of structure exp4, from ``<run>/summary`` only.

Writes
    <run>/summary/facts.md                      every number the write-up cites, with its source table
    docs/results/final_maps/sidlens_final_maps.html   self-contained report (figures embedded)
    docs/results/final_maps/figures/            PDF + PNG + CSV copies of the summary figures

Every number in the narrative is read from the summary CSVs here (``compute``),
never typed, so the page cannot drift from the run. The sentences around the
numbers are fixed text in ``narrative.py``, written after reading ``facts.md``
of run 327442; rerunning on another run needs those sentences reread.

    python experiments/structure/exp4_final_maps/report.py --run <job dir>
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sidlens import paths
from sidlens.viz.diffusion_report import page

CATALOGS = ("industrial", "office")
CL = {"industrial": "Industrial", "office": "Office"}
PRIMARY = {"industrial": "cat_l1", "office": "cat_l2"}
QL = {"rqkmeans": "RQ-KMeans", "rqvae": "RQ-VAE", "MQ": "MQ"}
DOCS = paths.REPO / "docs/results/final_maps"


def v(q, d, w):
    return f"{q}_{d}codebook_{w}"


def p(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f} %"


def pr(s, nd=1):
    s = pd.Series(s).dropna()
    if s.empty:
        return "n/a"
    lo, hi = f"{100 * s.min():.{nd}f}".replace("-", "−"), f"{100 * s.max():.{nd}f}".replace("-", "−")
    if lo == hi:
        return f"{lo} %"
    return f"{lo}{' to ' if s.min() < 0 else '–'}{hi} %"


def r2(s, nd=2):
    s = pd.Series(s).dropna()
    if s.empty:
        return "n/a"
    lo, hi = f"{s.min():.{nd}f}".replace("-", "−"), f"{s.max():.{nd}f}".replace("-", "−")
    return lo if lo == hi else f"{lo}{' to ' if s.min() < 0 else '–'}{hi}"


class Facts:
    """name -> (value text, source) for each catalog; written to facts.md."""

    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []
        self.d: dict = {}

    def add(self, cat, key, value, source):
        self.rows.append((cat, key, value, source))
        self.d[(cat, key)] = value

    def __call__(self, cat, key):
        return self.d[(cat, key)]


def compute(S: Path) -> tuple[Facts, dict]:
    T = {f.stem: pd.read_csv(f) for f in S.glob("*.csv")}
    A = {c: json.loads((S.parent / c / "partA_catalogue.json").read_text()) for c in CATALOGS}
    F = Facts()
    dg = T["m1_digits"][T["m1_digits"].kind == "native"]
    dp = T["m1_depths"][T["m1_depths"].kind == "native"]
    B, D, H, Z, X = T["b_tables"], T["d_loop"], T["hub"], T["zcr"], T["x_exposure"]
    C, L, MQ, R, AMI = T["c_summary"], T["x_latent"], T["mq_models"], T.get("r_old_vs_v3"), T["m1_ami"]
    NM, CP = T["m5_naming"], T["m5_copurchase"]
    for c in CATALOGS:
        a = A[c]
        n = a["n_items"]
        F.add(c, "n_items", f"{n:,}", "partA_catalogue.json")
        F.add(c, "identical", f"{a['n_identical_input_items']} items in {a['n_identical_input_groups']} groups "
                              f"({p(a['n_identical_input_items'] / n, 2)})", "partA_catalogue.json")
        F.add(c, "near_dup", f"{a['n_near_duplicate_pairs']:,} pairs covering "
                             f"{a['n_items_in_near_duplicate_pairs']:,} items ({p(a['share_items_in_near_duplicate_pairs'])})",
              "partA_catalogue.json")
        F.add(c, "spectrum", f"participation ratio {a['participation_ratio']:.0f}; top 10/64/256 PCs "
                             f"{p(a['var_share_top10'], 0)} / {p(a['var_share_top64'], 0)} / {p(a['var_share_top256'], 0)}",
              "partA_catalogue.json")
        b, d_, z, x = B[B.catalog == c], D[D.catalog == c], Z[Z.catalog == c], X[X.catalog == c]
        g, q_ = dg[dg.catalog == c], dp[dp.catalog == c]
        for q in QL:
            F.add(c, f"rate_{q}", pr(b[b.quantizer == q].collision_rate), "b_tables.collision_rate (9 maps)")
        F.add(c, "rate_3x128", " / ".join(f"{QL[q]} {p(b[b.table == v(q, 3, 128)].collision_rate.iloc[0])}"
                                          for q in QL), "b_tables")
        # what each digit encodes (4x128)
        for q in QL:
            s = g[g.variant == v(q, 4, 128)].sort_values("digit")
            F.add(c, f"var_alone_{q}_4x128", ", ".join(f"{100 * x_:.0f}" for x_ in s.var_alone) + " %",
                  "m1_digits.var_alone, digits 1-4")
            F.add(c, f"var_given_{q}_4x128", ", ".join(f"{100 * x_:.1f}" for x_ in s.var_given_prefix) + " %",
                  "m1_digits.var_given_prefix, digits 1-4")
        mq = g[g.quantizer == "MQ"]
        F.add(c, "mq_var_alone_all", pr(mq.var_alone, 0), "m1_digits.var_alone, MQ, all digits of 9 maps")
        ratios = mq.groupby("variant").var_alone.agg(lambda s: s.max() / s.min())
        F.add(c, "mq_var_ratio", r2(ratios), "max/min single-digit share per MQ map")
        rk = g[g.quantizer == "rqkmeans"]
        F.add(c, "rk_d1_alone", pr(rk[rk.digit == 1].var_alone, 0), "m1_digits, RQ-KMeans digit 1, 9 maps")
        F.add(c, "rk_later_alone", pr(rk[rk.digit > 1].var_alone, 0), "m1_digits, RQ-KMeans digits >= 2")
        rv = g[g.quantizer == "rqvae"]
        F.add(c, "rv_d1_alone", pr(rv[rv.digit == 1].var_alone, 0), "m1_digits, RQ-VAE digit 1")
        F.add(c, "rv_d1_given", pr(rv[rv.digit == 1].var_given_prefix, 0), "m1_digits, RQ-VAE digit 1")
        F.add(c, "rv_d2_given", pr(rv[rv.digit == 2].var_given_prefix, 0), "m1_digits, RQ-VAE digit 2 given digit 1")
        F.add(c, "rv_d2_beats_d1", f"{int((rv[rv.digit == 2].set_index('variant').var_given_prefix > rv[rv.digit == 1].set_index('variant').var_given_prefix).sum())}/9",
              "RQ-VAE maps where digit 2 given digit 1 explains more than digit 1")
        F.add(c, "rv_late_given_max", p(rv[rv.digit >= 4].var_given_prefix.max(), 1),
              "m1_digits, RQ-VAE digits 4-5 given prefix, max")
        F.add(c, "rv_util_d1", pr(rv[rv.digit == 1].utilization, 0) + f" ({int((rv[rv.digit == 1].utilization <= 0.5).sum())}/9 at <= 50 %)",
              "m1_digits.utilization, RQ-VAE digit 1")
        F.add(c, "rk_util", pr(rk.utilization, 0), "m1_digits.utilization, RQ-KMeans all digits")
        F.add(c, "mq_util", pr(mq.utilization, 0), "m1_digits.utilization, MQ all digits")
        f = PRIMARY[c]
        pur = q_[q_.depth == 3]
        for q in QL:
            r = pur[pur.variant == v(q, 4, 128)].iloc[0]
            F.add(c, f"purity3_{q}", f"{p(r[f'{f}_purity'])} (null {p(r[f'{f}_null_mean'])}; "
                                     f"{p(r[f'{f}_multi_item_share'], 0)} of items still grouped)",
                  f"m1_depths.{f}_purity, prefix length 3, {q} 4x128")
        res = q_[(q_.depth == 3) & q_.quantizer.isin(["rqkmeans", "rqvae"]) & (q_.n_codebook >= 4)]
        F.add(c, "purity3_residual_all", pr(res[f"{f}_purity"]), f"m1_depths.{f}_purity, residual maps D>=4, length 3")
        F.add(c, "purity3_mq_all", pr(pur[pur.quantizer == "MQ"][f"{f}_purity"]), f"m1_depths.{f}_purity, MQ, length 3")
        F.add(c, "ri_5x128", " / ".join(f"{QL[q]} {q_[(q_.variant == v(q, 5, 128)) & (q_.depth == 5)].refinement_index.iloc[0]:.2f}"
                                        for q in QL), "m1_depths.refinement_index, 5x128, length 5")
        F.add(c, "unique3_rqvae_4x128", p(q_[(q_.variant == v('rqvae', 4, 128)) & (q_.depth == 3)].frac_items_in_singletons.iloc[0]),
              "m1_depths.frac_items_in_singletons, RQ-VAE 4x128, length 3")
        last = q_[q_.depth == q_.n_codebook]
        for q in QL:
            F.add(c, f"alone_last_{q}", pr(last[last.quantizer == q].frac_items_in_singletons),
                  "m1_depths.frac_items_in_singletons at full length")
        bits = g[g.digit == g.n_codebook]
        F.add(c, "bits_budget", f"{np.log2(n):.1f}", "log2(items)")
        for q in QL:
            F.add(c, f"bits_full_{q}", r2(bits[bits.quantizer == q].cum_bits, 1), "m1_digits.cum_bits at the last digit")
        am = AMI[(AMI.catalog == c) & (AMI.kind == "native") & (AMI.field == f)]
        for q in QL:
            s = am[am.variant == v(q, 4, 128)].sort_values("digit")
            F.add(c, f"ami_{q}_4x128", ", ".join(("n/a" if np.isnan(x_) else f"{x_:.2f}") + ("" if sup or dgt == 1 else "†")
                                                for x_, sup, dgt in zip(s.conditional_ami, s.conditional_supported, s.digit)),
                  f"m1_ami.conditional_ami ({f}), digits 1-4; † = unsupported")
        # post-processing
        rvd, rkd, mqd = d_[d_.quantizer == "rqvae"], d_[d_.quantizer == "rqkmeans"], d_[d_.quantizer == "MQ"]
        F.add(c, "floor", p(d_.identical_floor.iloc[0], 2), "d_loop.identical_floor")
        F.add(c, "rv_pre", pr(rvd.rate_start), "d_loop.rate_start (RQ-VAE pre-reassignment)")
        F.add(c, "rv_pub", pr(rvd.rate_published, 2), "d_loop.rate_published (RQ-VAE)")
        F.add(c, "rv_changed", pr(rvd.share_last_digit_changed, 0), "d_loop.share_last_digit_changed (RQ-VAE)")
        F.add(c, "rv_loop_reproduces", f"{int((rvd.loop_equals_published_items == 1.0).sum())}/9 tables item for item "
                                       f"(min {p(rvd.loop_equals_published_items.min(), 2)})",
              "d_loop.loop_equals_published_items")
        F.add(c, "rv_at_floor", f"{int(((rvd.rate_published - rvd.identical_floor) <= 0.005).sum())}/9",
              "RQ-VAE published within 0.5 pp of floor")
        for nm, gg in (("rk", rkd), ("mq", mqd)):
            F.add(c, f"{nm}_loop", f"{pr(gg.rate_start)} → {pr(gg.rate_after_loop, 2)}", "d_loop rate_start → rate_after_loop")
            F.add(c, f"{nm}_loop_changed", pr(gg.share_last_digit_changed, 0), "d_loop.share_last_digit_changed")
            F.add(c, f"{nm}_loop_notfloor", ", ".join(f"{r.variant} {p(r.rate_after_loop, 2)}" for r in gg.itertuples()
                                                      if r.rate_after_loop - r.identical_floor > 0.005) or "none",
                  "d_loop: tables left > 0.5 pp above the floor")
            ratio = gg.last_level_sq_dist_increase_changed / gg.last_level_sq_dist_start_mean
            F.add(c, f"{nm}_loop_cost", r2(ratio, 1) + "x", "moved items' added last-level sq. distance / table mean")
        # near-duplicates
        F.add(c, "nd_deepest", ", ".join(f"5×{W} {p(b[b.table == v('rqkmeans', 5, W)].nd_share_collided.iloc[0], 0)}"
                                         for W in (128, 256, 512)), "b_tables.nd_share_collided")
        F.add(c, "nd_trend", "; ".join(f"W={W}: " + " → ".join(f"{100 * b[b.table == v('rqkmeans', D_, W)].nd_share_collided.iloc[0]:.0f}"
                                                               for D_ in (3, 4, 5)) for W in (128, 256, 512)),
              "b_tables.nd_share_collided, depths 3→4→5 (%)")
        F.add(c, "nd_mq_deep", pr(b[(b.quantizer == "MQ") & (b.depth == 5)].nd_share_collided, 0), "b_tables, MQ depth 5")
        F.add(c, "identical_split", f"{int((b.identical_pairs_split == 0).sum())}/27", "b_tables.identical_pairs_split == 0")
        F.add(c, "closest_mate", pr(b[b.quantizer == "rqkmeans"]["share_mate_in_closest_0.1pct"], 0),
              "b_tables.share_mate_in_closest_0.1pct (RQ-KMeans)")
        # hub
        h = H[(H.catalog == c) & (H.level >= 1)]
        F.add(c, "hub_share", pr(h.hub_share, 0), "hub.hub_share, levels >= 2")
        F.add(c, "hub_rank0", f"{int((h.hub_norm_rank_from_smallest == 0).sum())}/{len(h)}", "hub rank 0 count")
        F.add(c, "radius_spearman", r2(b[b.quantizer == "rqkmeans"].radius_spearman), "b_tables.radius_spearman")
        F.add(c, "repeat_rate", r2(b[b.quantizer == "rqkmeans"].repeat_rate_mean, 3) + " (1/K = 0.002–0.008)",
              "b_tables.repeat_rate_mean (RQ-KMeans)")
        # MQ
        mb = b[b.quantizer == "MQ"]
        F.add(c, "mq_nmi", r2(mb.mean_pairwise_nmi), "b_tables.mean_pairwise_nmi")
        F.add(c, "rk_nmi", r2(b[b.quantizer == "rqkmeans"].mean_pairwise_nmi), "b_tables.mean_pairwise_nmi")
        F.add(c, "rv_nmi", r2(b[b.quantizer == "rqvae"].mean_pairwise_nmi), "b_tables.mean_pairwise_nmi")
        fin = mb.obs_over_n1.replace(np.inf, np.nan)
        F.add(c, "mq_obs_n1", f"at least {fin.min():.0f}x; N1 is 0 in {int(np.isinf(mb.obs_over_n1).sum())} of 9 maps",
              "b_tables.obs_over_n1")
        pa = T["m1_pairs"][(T["m1_pairs"].catalog == c) & (T["m1_pairs"].kind == "native")]
        per_map = pa.groupby(["quantizer", "variant"]).ami.mean()
        for q in QL:
            F.add(c, f"pair_ami_{q}", r2(per_map[q], 2), "m1_pairs: per-map mean pairwise AMI (chance-corrected)")
            F.add(c, f"pair_nmi_{q}", r2(b[b.quantizer == q].mean_pairwise_nmi), "b_tables.mean_pairwise_nmi (plug-in, biased)")
        r3 = mb[mb.table == v("MQ", 3, 128)].iloc[0]
        F.add(c, "mq_3x128_n1", f"{p(r3.collision_rate)} observed, {p(r3.n1_rate_p50)} with independent digits",
              "b_tables MQ 3x128")
        m = MQ[MQ.catalog == c]
        F.add(c, "mq_init", f"{int(m.all_at_init.astype(bool).sum())}/{len(m)} models; parameter std / uniform std "
                            f"{m.uniform_std_ratio_min.min():.2f}–{m.uniform_std_ratio_max.max():.2f}", "mq_models")
        # refits (counterfactual)
        cc = C[C.catalog == c]

        def cdiff(cond, W, D_):
            r = cc[(cc.condition == cond) & (cc.width == W) & (cc.depth == D_)].iloc[0]
            return f"{100 * r.paired_diff_mean:+.1f}{'' if r.ranges_disjoint else ' (ranges overlap)'}"
        for cond in ("per-level seed", "scale-norm", "L2", "PCA-256-white", "PCA-64-white"):
            F.add(c, f"c_{cond}", ", ".join(f"W={W}: {cdiff(cond, W, 3)} / {cdiff(cond, W, 5)}" for W in (128, 256, 512)),
                  f"c_summary paired diff (pp), depth 3 / depth 5, vs {cc[cc.condition == cond].baseline.iloc[0]}")
        nat = cc[(cc.condition == "native")]
        F.add(c, "c_native_vs_pub", "; ".join(
            f"{D_}×{W}: refit {p(nat[(nat.width == W) & (nat.depth == D_)].collision_rate_mean.iloc[0])} vs published "
            f"{p(b[b.table == v('rqkmeans', D_, W)].collision_rate.iloc[0])}" for W in (128, 512) for D_ in (3, 5)),
              "c_summary native mean over 5 seeds vs b_tables")
        # ZCR
        F.add(c, "zcr_n", f"{len(z)}", "zcr rows")
        tj = json.loads((paths.EXTERNAL / "sid_codebooks/WFRaain-onediffrec-sid-codebooks-9733eda" / c
                         / "tokenizers/tables.json").read_text())["tables"]
        inf = {k: r_ for k, r_ in tj.items() if r_["zcr"] == "infeasible"}
        F.add(c, "zcr_infeasible", ", ".join(f"{k} (largest {int(k.split('_')[1][0]) - 1}-digit prefix group: "
                                             f"{r_['largest_prefix_group']} items)"
                                             for k, r_ in sorted(inf.items())) or "none", "tables.json zcr state")
        F.add(c, "zcr_below5", f"{sum(r_['zcr'] == 'below 5%' for r_ in tj.values())}", "tables.json zcr == 'below 5%'")
        for q in QL:
            zz = z[z.quantizer == q]
            if len(zz):
                F.add(c, f"zcr_changed_{q}", pr(zz.changed_tail_share, 1), "zcr.changed_tail_share")
                F.add(c, f"zcr_cost_{q}", pr(zz.sq_dist_rel_increase, 1), "zcr.sq_dist_rel_increase")
        zr = z[z.variant == v("rqkmeans", 3, 128)].iloc[0]
        F.add(c, "zcr_rk3x128", f"{p(zr.changed_tail_share, 0)} of last digits; last-level sq. distance +{100 * zr.sq_dist_rel_increase:.1f} %",
              "zcr rqkmeans 3x128")
        # exposure
        xn = x[x.kind == "native"]
        F.add(c, "x_targets", f"{int(xn.n_test_targets.iloc[0]):,}", "x_exposure")
        F.add(c, "x_repeat", p(xn.test_repeat.iloc[0]), "x_exposure.test_repeat")
        for q in QL:
            F.add(c, f"x_partner_{q}", pr(xn[xn.quantizer == q].test_sid_partner), "x_exposure.test_sid_partner")
            F.add(c, f"x_collided_{q}", pr(xn[xn.quantizer == q].test_targets_collided, 0), "x_exposure.test_targets_collided")
            F.add(c, f"x_inter_{q}", pr(xn[xn.quantizer == q].interactions_on_collided_items, 1),
                  "x_exposure.interactions_on_collided_items")
        xr = xn[xn.variant == v("rqkmeans", 3, 128)].iloc[0]
        xz = x[(x.kind == "zcr") & (x.variant == v("rqkmeans", 3, 128))].iloc[0]
        F.add(c, "x_rk3x128", f"SID partner {p(xr.test_sid_partner)}, variant {p(xr.test_variant)} (ZCR twin: "
                              f"{p(xz.test_sid_partner)} / {p(xz.test_variant)})", "x_exposure rqkmeans 3x128")
        F.add(c, "x_new", pr(xn.test_new), "x_exposure.test_new")
        # RQ-VAE latent
        l0 = L[(L.catalog == c) & (L.level == 0)]
        F.add(c, "lat_dead", f"{int(l0.dead_codewords.min())}–{int(l0.dead_codewords.max())} of 128–512",
              "x_latent level 0 dead_codewords")
        F.add(c, "lat_norm", f"unused {r2(l0.codeword_norm_median_dead, 1)} vs used {r2(l0.codeword_norm_median_live, 1)}",
              "x_latent level 0 codeword norms (medians)")
        F.add(c, "lat_var", pr(l0.latent_var_explained_alone, 0), "x_latent level 0 latent variance explained by digit 1")
        # M5
        nm = NM[(NM.catalog == c) & (NM.digit == 0)]
        F.add(c, "naming_d1", " / ".join(f"{QL[q]} {r2(nm[nm.quantizer == q].median_auc)}" for q in QL),
              "m5_naming median held-out AUC, digit 1, 9 maps each")
        cp = CP[CP.catalog == c]
        F.add(c, "copurchase", " / ".join(f"{QL[q]} {r2(cp[(cp.quantizer == q) & (cp.depth == 1)].lift, 0)}x" for q in QL),
              "m5_copurchase lift at prefix length 1")
        if R is not None and c == "industrial":
            rr = R[R.variant.str.startswith("rqkmeans") & (R.variant != v("rqkmeans", 5, 512))]
            F.add(c, "old_rk_d1", pr(rr.digit1_agree, 0) + " (8 tables; the old 5×512 was a separate balanced fit: "
                  f"{p(R[R.variant == v('rqkmeans', 5, 512)].digit1_agree.iloc[0], 0)})",
                  "r_old_vs_v3 digit1_agree RQ-KMeans")
            F.add(c, "old_rk_d2", " / ".join(f"W={W} {p(rr[rr.variant == v('rqkmeans', 3, W)].digit2_agree.iloc[0], 0)}"
                                             for W in (128, 256, 512)), "r_old_vs_v3 digit2_agree")
            F.add(c, "old_focal", f"RQ-KMeans 3×128 same full SID {p(R[R.variant == v('rqkmeans', 3, 128)].same_full_sid.iloc[0], 0)}; "
                                  f"RQ-VAE 4×128 first-digit ARI {R[R.variant == v('rqvae', 4, 128)].prefix1_ari.iloc[0]:.2f}; "
                                  f"MQ first-digit ARI {r2(R[R.variant.str.startswith('MQ')].prefix1_ari)}",
                  "r_old_vs_v3")
    return F, T


def table_html(df: pd.DataFrame, caption: str = "", num_cols=()) -> str:
    head = "".join(f"<th>{html.escape(str(cname))}</th>" for cname in df.columns)
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if cname in num_cols else ""}">{html.escape(str(val))}</td>' for cname, val in zip(df.columns, row))
        + "</tr>" for row in df.itertuples(index=False))
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def img(path: Path, caption: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><figcaption>{html.escape(caption)}</figcaption>'
            f'<img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}" style="width:100%;height:auto;background:#fff;border-radius:6px"></figure>')


def build(run: Path, narrative: dict) -> str:
    S = run / "summary"
    F, T = compute(S)
    (S / "facts.md").write_text("# Facts cited in the exp4 write-up\n\n| catalog | key | value | source |\n|---|---|---|---|\n"
                                + "".join(f"| {c} | {k} | {val} | {src} |\n" for c, k, val, src in F.rows))
    ex = T["expectations"]
    toc, body = [], []

    def sec(anchor, title, inner, level=2):
        toc.append((anchor, title, level))
        body.append(f'<h{level} id="{anchor}">{html.escape(title)}</h{level}>{inner}')

    def both(key):
        return " · ".join(f"{CL[c]}: {html.escape(F(c, key))}" for c in CATALOGS)

    sec("summary", "Summary", narrative["summary"](F, both))
    v_rows = []
    for c in CATALOGS:
        val = json.loads((run / c / "validation.json").read_text())
        v_rows.append({"catalog": CL[c], **{k: val[k] for k in val if isinstance(val[k], bool)}})
    sec("validation", "Validation (Part 0)", narrative["validation"](F, both) + table_html(pd.DataFrame(v_rows)))
    exd = ex.assign(held=ex.held.map({True: "held", False: "failed"}))[["expectation", "catalog", "held", "rule", "failing", "note"]]
    sec("expectations", "Declared expectations", narrative["expectations"](F, both) + table_html(exd))
    figs = S / "figures"
    sec("digits", "What each digit encodes (M1)", narrative["digits"](F, both)
        + '<div class="figs">' + img(figs / "fig_v3_map_digits.png", "Codes in use, variance explained by one digit, purity by prefix length (4×128)")
        + img(figs / "fig_v3_map_identity.png", "Identity resolved, refinement lost per deleted digit, collision share, refinement index") + "</div>")
    sec("collisions", "Why items share a SID (M2)", narrative["collisions"](F, both)
        + img(figs / "fig_v3_post_processing.png", "Collision share before and after RQ-VAE's last-digit loop")
        + table_html(T["collision_causes"].assign(catalog=lambda d: d.catalog.map(CL)), "Collision causes, per catalog"))
    sec("refits", "Refits of RQ-KMeans (counterfactual)", narrative["refits"](F, both))
    sec("zcr", "ZCR twins", narrative["zcr"](F, both))
    sec("exposure", "Collision exposure of the test targets (exploratory)", narrative["exposure"](F, both)
        + img(figs / "fig_v3_exposure.png", "Next-item test targets by class, every native map"))
    sec("latent", "RQ-VAE in its own latent (exploratory)", narrative["latent"](F, both))
    sec("m5", "Naming and co-purchase (M5)", narrative["m5"](F, both))
    sec("old", "How the final maps relate to the old ones", narrative["old"](F, both))
    cl = T["claims"].rename(columns={"old_value_industrial": "old maps (Industrial)", "v3_industrial": "v3 Industrial",
                                     "v3_office": "v3 Office"})
    sec("claims", "Draft numbers, old against v3", narrative["claims"](F, both) + table_html(cl))
    sec("caveats", "What this does not establish", narrative["caveats"](F, both))
    sec("files", "Files", narrative["files"](run))
    return page("SID maps of the final run", toc, "".join(body),
                f"Structure exp4 · run <code>{html.escape(str(run))}</code> · both catalogs, 54 native maps and their ZCR twins · "
                f"no recommender is run")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    from narrative import NARRATIVE   # fixed text, next to this file
    out_html = build(args.run, NARRATIVE)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "figures").mkdir(exist_ok=True)
    for f in sorted((args.run / "summary" / "figures").iterdir()):
        shutil.copyfile(f, DOCS / "figures" / f.name)
    (DOCS / "sidlens_final_maps.html").write_text(out_html)
    shutil.copyfile(args.run / "summary" / "facts.md", DOCS / "facts.md")
    print("wrote", DOCS / "sidlens_final_maps.html")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
