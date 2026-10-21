#!/usr/bin/env python
"""Figures of structure exp4, drawn from ``<run>/summary/*.csv`` only.

The two map figures of the paper (fig_map_digits, fig_map_identity; drawn by
docs/paper/archive/rewrite-20260928/tools/make_figures.py from the old maps)
are redrawn on the v3 maps with one row per catalog, plus two figures for the
parts that are new:

    fig_v3_map_digits      codes in use, variance explained by one digit alone,
                           and primary-field purity against its shuffled null,
                           for the three 4x128 maps
    fig_v3_map_identity    bits resolved, refinement lost when one digit is
                           deleted (4x128), share of items in a shared SID (all
                           27 maps), refinement index by prefix length (5x128,
                           5x512)
    fig_v3_post_processing RQ-VAE before and after its last-digit reassignment;
                           the same loop applied to RQ-KMeans and MQ; ZCR's
                           share of changed last digits
    fig_v3_exposure        next-item test targets by class (repeat, SID partner,
                           variant, new) for every native map (exploratory)

One style (`sidlens.viz.style`): RQ-KMeans blue, RQ-VAE orange, MQ aqua
everywhere; codebook width is W. Every figure writes `<stem>.csv` with the
numbers it plots.

    python experiments/structure/exp4_final_maps/figures.py --run <job dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from sidlens.viz import style as st  # noqa: E402

FULL = 7.0
QC, QL, QO = st.QUANTIZER_COLOR, st.QUANTIZER_LABEL, st.QUANTIZER_ORDER
CATALOGS = ("industrial", "office")
CAT_LABEL = {"industrial": "Industrial", "office": "Office"}
PRIMARY = {"industrial": ("cat_l1", "level-1 category"), "office": ("cat_l2", "level-2 category")}
WIDTH_MARKER = {128: "o", 256: "s", 512: "^"}


def v(q, d, w):
    return f"{q}_{d}codebook_{w}"


def label(ax, t):
    ax.text(-0.02, 1.04, f"({t})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, fontweight="bold", color=st.INK)


def save(fig, out: Path, stem: str, table: pd.DataFrame):
    st.save(fig, out / stem)
    table.to_csv(out / f"{stem}.csv", index=False)
    plt.close(fig)


def fig_digits(S: Path, out: Path):
    dg = pd.read_csv(S / "m1_digits.csv")
    dp = pd.read_csv(S / "m1_depths.csv")
    dg, dp = dg[dg.kind == "native"], dp[dp.kind == "native"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.7))
    plt.subplots_adjust(wspace=0.42, hspace=0.62)
    rows = []
    for i, c in enumerate(CATALOGS):
        f, flab = PRIMARY[c]
        for q in QO:
            s = dg[(dg.catalog == c) & (dg.variant == v(q, 4, 128))].sort_values("digit")
            p = dp[(dp.catalog == c) & (dp.variant == v(q, 4, 128)) & (dp.depth >= 1)].sort_values("depth")
            col = QC[q]
            axes[i, 0].plot(s.digit, 100 * s.utilization, "-o", color=col)
            axes[i, 1].plot(s.digit, 100 * s.var_alone, "-o", color=col)
            axes[i, 2].plot(p.depth, 100 * p[f"{f}_purity"], "-o", color=col)
            axes[i, 2].plot(p.depth, 100 * p[f"{f}_null_mean"], ":", color=col, lw=1.2)
            axes[i, 2].fill_between(p.depth, 100 * p[f"{f}_null_low"], 100 * p[f"{f}_null_high"],
                                    color=col, alpha=0.12, lw=0)
            for _, r in s.iterrows():
                rows.append({"catalog": c, "variant": r.variant, "digit": r.digit,
                             "codes_in_use": r.utilization, "variance_alone": r.var_alone})
            for _, r in p.iterrows():
                rows.append({"catalog": c, "variant": r.variant, "prefix_length": r.depth, "field": f,
                             "purity": r[f"{f}_purity"], "purity_null": r[f"{f}_null_mean"],
                             "multi_item_share": r[f"{f}_multi_item_share"]})
        axes[i, 0].set_ylabel(f"{CAT_LABEL[c]}\nCodes in use (%)")
        axes[i, 0].set_ylim(0, 105)
        axes[i, 1].set_ylabel("Variance explained (%)")
        axes[i, 1].set_ylim(0, 70)
        axes[i, 2].set_ylabel(f"Items in group's main\n{flab} (%)")
        axes[i, 2].set_ylim(0, 105)
        for a in axes[i, :2]:
            a.set_xlabel("Digit (alone)")
            a.set_xticks([1, 2, 3, 4])
        axes[i, 2].set_xlabel("Shared prefix length")
        axes[i, 2].set_xticks([1, 2, 3, 4])
    for a, t in zip(axes.ravel(), "abcdef"):
        label(a, t)
    h = [Line2D([], [], color=QC[q], marker="o", label=QL[q]) for q in QO]
    h += [Line2D([], [], color=st.INK_2, marker="o", label="Observed"),
          Line2D([], [], color=st.INK_2, ls=":", label="Categories shuffled")]
    fig.legend(handles=h, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    save(fig, out, "fig_v3_map_digits", pd.DataFrame(rows))


def fig_identity(S: Path, out: Path):
    dg = pd.read_csv(S / "m1_digits.csv")
    dp = pd.read_csv(S / "m1_depths.csv")
    bt = pd.read_csv(S / "b_tables.csv")
    floor = pd.read_csv(S / "d_loop.csv").groupby("catalog").identical_floor.first()
    dg, dp = dg[dg.kind == "native"], dp[dp.kind == "native"]
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.8))
    plt.subplots_adjust(wspace=0.6, hspace=0.62)
    rows = []
    for i, c in enumerate(CATALOGS):
        g = dg[dg.catalog == c]
        n_items = int(dp[(dp.catalog == c) & (dp.depth == 0)].largest_cluster.iloc[0])
        budget = math.log2(n_items)
        for qi, q in enumerate(QO):
            s = g[g.variant == v(q, 4, 128)].sort_values("digit")
            axes[i, 0].plot(np.r_[0, s.digit], np.r_[0, s.cum_bits], "-o", color=QC[q])
            axes[i, 1].bar(s.digit + (qi - 1) * 0.26, s.refinement_lost_if_deleted, width=0.25, color=QC[q])
            for _, r in s.iterrows():
                rows.append({"catalog": c, "variant": r.variant, "digit": r.digit, "cum_bits": r.cum_bits,
                             "refinement_lost_if_deleted": r.refinement_lost_if_deleted})
        axes[i, 0].axhline(budget, color=st.MUTED, lw=0.9)
        axes[i, 0].annotate(f"$\\log_2 N$ = {budget:.1f} bits", (4.1, budget), xytext=(0, -3),
                            textcoords="offset points", fontsize=6, color=st.MUTED, ha="right", va="top")
        axes[i, 0].set_ylim(0, budget + 2)
        axes[i, 0].set_xticks([0, 1, 2, 3, 4])
        axes[i, 0].set_xlabel("Digits read")
        axes[i, 0].set_ylabel(f"{CAT_LABEL[c]}\nIdentity resolved (bits)")
        axes[i, 1].set_xticks([1, 2, 3, 4])
        axes[i, 1].set_xlabel("Deleted digit")
        axes[i, 1].set_ylabel("Refinement lost")
        a = axes[i, 2]
        b = bt[bt.catalog == c]
        for qi, q in enumerate(QO):
            for w, mk in WIDTH_MARKER.items():
                s = b[(b.quantizer == q) & (b.width == w)]
                a.plot(s.depth + (qi - 1) * 0.2 + {128: -0.05, 256: 0, 512: 0.05}[w],
                       100 * s.collision_rate, mk, color=QC[q], ms=4)
                for _, r in s.iterrows():
                    rows.append({"catalog": c, "variant": r.table, "collision_rate": r.collision_rate})
        fl = 100 * floor[c]
        a.set_yscale("log")
        a.set_ylim(min(0.2, fl * 0.6), 90)
        a.axhline(fl, color=st.MUTED, lw=0.9)
        a.annotate("floor", (2.55, fl), xytext=(0, 1.5), textcoords="offset points", fontsize=6, color=st.MUTED)
        a.set_xticks([3, 4, 5])
        a.set_xlim(2.5, 5.5)
        a.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        a.set_xlabel("SID length $D$")
        a.set_ylabel("Items sharing a SID (%)")
        a = axes[i, 3]
        for q in QO:
            for w, lw, al in ((128, st.LINE_W, 1.0), (512, 1.0, 0.45)):
                s = dp[(dp.catalog == c) & (dp.variant == v(q, 5, w)) & (dp.depth >= 1)].sort_values("depth")
                a.plot(s.depth, s.refinement_index, "-" if w == 128 else "--", color=QC[q], lw=lw, alpha=al,
                       marker="o" if w == 128 else None)
                for _, r in s.iterrows():
                    rows.append({"catalog": c, "variant": r.variant, "prefix_length": r.depth,
                                 "refinement_index": r.refinement_index})
        a.set_ylim(0, 1.02)
        a.set_xticks([1, 2, 3, 4, 5])
        a.set_xlabel("Prefix length")
        a.set_ylabel("Refinement index")
    for a, t in zip(axes.ravel(), "abcdefgh"):
        label(a, t)
    h = [Patch(color=QC[q], label=QL[q]) for q in QO]
    h += [Line2D([], [], marker=m, ls="", color=st.MUTED, label=f"$W$={w}") for w, m in WIDTH_MARKER.items()]
    h += [Line2D([], [], color=st.INK_2, label="$5\\times128$"),
          Line2D([], [], color=st.INK_2, ls="--", lw=1.0, label="$5\\times512$")]
    fig.legend(handles=h, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.02), columnspacing=0.9)
    save(fig, out, "fig_v3_map_identity", pd.DataFrame(rows))


def fig_post(S: Path, out: Path):
    d = pd.read_csv(S / "d_loop.csv")
    z = pd.read_csv(S / "zcr.csv")
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 3.9))
    plt.subplots_adjust(wspace=0.5, hspace=0.75)
    rows = []
    order = [(D, W) for D in (3, 4, 5) for W in (128, 256, 512)]
    xt = [f"{D}×{W}" for D, W in order]
    for i, c in enumerate(CATALOGS):
        g = d[d.catalog == c]
        floor = 100 * g.identical_floor.iloc[0]
        for j, (q, title) in enumerate((("rqvae", "RQ-VAE: before → published"),
                                        ("rqkmeans", "RQ-KMeans: loop applied"),
                                        ("MQ", "MQ: loop applied"))):
            a = axes[i, j]
            s = g[g.quantizer == q].set_index("variant")
            x = np.arange(len(order))
            before = [100 * s.loc[v(q, D, W)].rate_start for D, W in order]
            after = [100 * s.loc[v(q, D, W)].rate_after_loop for D, W in order]
            a.vlines(x, after, before, color=QC[q], lw=1.4, alpha=0.6)
            a.plot(x, before, "o", color=QC[q], ms=4, label="before")
            a.plot(x, after, "o", mfc=st.SURFACE, mec=QC[q], ms=4, label="after")
            a.axhline(floor, color=st.MUTED, lw=0.9)
            a.set_yscale("log")
            a.set_ylim(0.3, 90)
            a.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
            a.set_xticks(x, xt, rotation=90, fontsize=6)
            a.set_title(title, fontsize=7, loc="left")
            if j == 0:
                a.set_ylabel(f"{CAT_LABEL[c]}\nItems sharing a SID (%)")
            for (D, W), b_, a_ in zip(order, before, after):
                rows.append({"catalog": c, "variant": v(q, D, W), "rate_before_pct": b_, "rate_after_pct": a_,
                             "zcr_changed_tail_share": (z[(z.catalog == c) & (z.variant == v(q, D, W))]
                                                        .changed_tail_share.squeeze() if len(
                                 z[(z.catalog == c) & (z.variant == v(q, D, W))]) else np.nan)})
    for a, t in zip(axes.ravel(), "abcdef"):
        label(a, t)
    h = [Line2D([], [], marker="o", ls="", color=st.INK_2, label="before the loop"),
         Line2D([], [], marker="o", ls="", mfc=st.SURFACE, mec=st.INK_2, label="after the loop"),
         Line2D([], [], color=st.MUTED, label="identical-input floor")]
    fig.legend(handles=h, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    save(fig, out, "fig_v3_post_processing", pd.DataFrame(rows))


def fig_exposure(S: Path, out: Path):
    e = pd.read_csv(S / "x_exposure.csv")
    e = e[e.kind == "native"]
    # "new" is the remainder to 100 % and is left undrawn so the three small
    # classes keep a readable scale; its share is in the CSV twin.
    cls = [("test_repeat", "repeat", st.INK_2), ("test_sid_partner", "SID partner", "#eb6834"),
           ("test_variant", "variant", "#eda100")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True)
    plt.subplots_adjust(wspace=0.08)
    rows = []
    for a, c in zip(axes, CATALOGS):
        g = e[e.catalog == c]
        names = [v(q, D, W) for q in QO for D in (3, 4, 5) for W in (128, 256, 512)]
        g = g.set_index("variant").loc[[n for n in names if n in set(g.variant)]]
        x = np.arange(len(g))
        bottom = np.zeros(len(g))
        for col, lab, colour in cls:
            a.bar(x, 100 * g[col], bottom=bottom, color=colour, width=0.8, label=lab)
            bottom += 100 * g[col].to_numpy()
        a.set_xticks(x, [n.replace("codebook_", "×").replace("rqkmeans_", "RK ").replace("rqvae_", "RV ")
                         .replace("MQ_", "MQ ") for n in g.index], rotation=90, fontsize=5.5)
        a.set_ylim(0, 25)
        a.set_title(f"{CAT_LABEL[c]} ({int(g.n_test_targets.iloc[0]):,} test targets)", fontsize=7, loc="left")
        for n_, r in g.iterrows():
            rows.append({"catalog": c, "variant": n_, **{col: r[col] for col, _, _ in cls},
                         "test_new": r.test_new, "test_targets_collided": r.test_targets_collided})
    axes[0].set_ylabel("Next-item test targets (%)\n(the rest are new items)")
    axes[1].legend(loc="upper right", fontsize=6.5, ncol=2)
    save(fig, out, "fig_v3_exposure", pd.DataFrame(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    S = args.run / "summary"
    out = S / "figures"
    out.mkdir(exist_ok=False)
    st.apply()
    for f in (fig_digits, fig_identity, fig_post, fig_exposure):
        f(S, out)
        print("wrote", f.__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
