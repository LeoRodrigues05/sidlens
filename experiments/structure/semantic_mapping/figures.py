#!/usr/bin/env python
"""Publication figures for the SID cluster-structure and semantic-mapping study.

CPU only, no model. Every panel is computed from frozen SID tables, the frozen
item embeddings, the external Amazon-2018 labels, and the completed derived
tables (exp1_atlas, exp2_geometry, exp2_refinement, semantic_mapping). Nothing
here re-fits a quantizer or touches a checkpoint.

Each figure writes the table behind it as CSV next to the PDF/PNG, so a number
in the paper can always be traced to a row rather than to a pixel.

    fig1_icicle          the prefix tree of one 4-digit variant per quantizer,
                         width = items, colour = majority level-1 category.
                         The "clusters shrink from digit 1 to 4" picture.
    fig2_cluster_sizes   item-weighted ECDF of cluster size at each depth.
    fig3_geometry        RMS radius and null-corrected refinement per depth,
                         all 27 variants.
    fig4_coherence       level-1/2 category purity above a shuffled-label null.
    fig5_digit_info      what each digit POSITION knows: codes used, standalone
                         variance explained, marginal and conditional AMI.
    fig6_zoom            embedding map, zooming into one RQ-KMeans branch.
    fig7_standalone      the same map coloured by ONE digit at a time, RQ vs MQ:
                         residual digits only mean something given the prefix,
                         parallel digits each partition the whole space.
    fig8_collisions      full-SID collision rate across depth x width.

Reads only; writes to $SIDLENS_WORK/derived/semantic_mapping/figures-<stamp>/.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from sidlens import paths
from sidlens.data import embeddings, labels, meta
from sidlens.data.sids import SidTable, available, is_nested_family
from sidlens.viz import style as S

CATEGORY = "Industrial_and_Scientific"
PRIMARY = {q: f"{q}_4codebook_128" for q in S.QUANTIZER_ORDER}   # the digit-1..4 grid
N_PERM = 200
SEED = 20260922


# ----------------------------------------------------------------- helpers --
def parse_variant(name: str) -> tuple[str, int, int]:
    q, rest = name.split("_", 1)
    d, w = rest.replace("codebook_", " ").split()
    return q, int(d), int(w)


def majority(codes_: np.ndarray) -> tuple[int, int, int]:
    """(majority class, its count, labelled count) ignoring -1."""
    lab = codes_[codes_ >= 0]
    if lab.size == 0:
        return -1, 0, 0
    counts = np.bincount(lab)
    return int(counts.argmax()), int(counts.max()), int(lab.size)


def prefix_groups(codes: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer group id per item for the depth-d prefix, plus group sizes."""
    if depth == 0:
        return np.zeros(len(codes), dtype=np.int64), np.array([len(codes)])
    _, inv, counts = np.unique(codes[:, :depth], axis=0, return_inverse=True,
                               return_counts=True)
    return inv.reshape(-1), counts


def item_weighted_purity(group: np.ndarray, lab: np.ndarray) -> float:
    """Share of LABELLED items whose label is the majority label of their group."""
    keep = lab >= 0
    g, y = group[keep], lab[keep]
    n_g, n_y = g.max() + 1, y.max() + 1
    joint = np.bincount(g * n_y + y, minlength=n_g * n_y).reshape(n_g, n_y)
    return float(joint.max(axis=1).sum() / keep.sum())


def purity_multi(group: np.ndarray, counts: np.ndarray, lab: np.ndarray,
                 rng: np.random.Generator, n_perm: int) -> dict:
    """Majority purity over items in MULTI-item groups, with a shuffled-label null.

    A singleton is trivially pure, so once most items are alone (deep prefixes)
    an all-item purity says more about capacity than about coherence. Restricting
    both the observed value and its null to items that still share a group keeps
    the question "are the items that ARE grouped, grouped by category?".
    """
    keep = (lab >= 0) & (counts[group] >= 2)
    share = float(keep.mean())
    if keep.sum() < 2:
        return {"purity": np.nan, "null_mean": np.nan, "null_low": np.nan,
                "null_high": np.nan, "excess": np.nan, "multi_item_share": share}
    g, y = group[keep], lab[keep]
    obs = item_weighted_purity(g, y)
    null = np.array([item_weighted_purity(g, rng.permutation(y)) for _ in range(n_perm)])
    return {"purity": obs, "null_mean": float(null.mean()),
            "null_low": float(np.quantile(null, 0.025)),
            "null_high": float(np.quantile(null, 0.975)),
            "excess": obs - float(null.mean()), "multi_item_share": share}


def between_share(z: np.ndarray, group: np.ndarray) -> float:
    """Fraction of total variance that lies BETWEEN groups (1 - R^2_within)."""
    n_g = group.max() + 1
    counts = np.bincount(group, minlength=n_g).astype(np.float64)
    sums = np.zeros((n_g, z.shape[1]), dtype=np.float64)
    np.add.at(sums, group, z)
    mu = sums / counts[:, None]
    total = float(((z - z.mean(0)) ** 2).sum())
    within = float(((z - mu[group]) ** 2).sum())
    return 1.0 - within / total


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


class Data:
    """Everything the figures share, loaded once."""

    def __init__(self, mapping_run: str):
        self.tables = {v.name: SidTable.load(v) for v in available()}
        first = next(iter(self.tables.values()))
        self.keys = first.keys
        for t in self.tables.values():
            if t.keys != self.keys:
                raise ValueError(f"{t.variant.name}: key order differs from {first.variant.name}")
        self.z = embeddings.load_aligned(CATEGORY, self.keys, np.float32).astype(np.float64)
        self.lab1, self.vocab1 = labels.codes(CATEGORY, self.keys, "cat_l1", force=True)
        self.lab2, self.vocab2 = labels.codes(CATEGORY, self.keys, "cat_l2", force=True)
        derived = paths.DERIVED
        self.geometry = pd.read_csv(derived / "exp2_geometry" / "depth_stats.csv")
        self.collisions = pd.read_csv(derived / "exp2_geometry" / "collisions.csv")
        self.utilization = pd.read_csv(derived / "exp2_geometry" / "utilization.csv")
        self.refinement = pd.read_csv(derived / "exp2_refinement" / "refinement.csv")
        self.attribute = pd.concat(
            pd.read_csv(p) for p in sorted((derived / "exp1_atlas").glob("*/attribute.csv")))
        self.coherence = pd.read_csv(derived / "semantic_mapping" / mapping_run / "coherence.csv")
        # ASIN-keyed titles; the info files are keyed by integer item id.
        self.titles = {asin: m.title for asin, m in meta.load(CATEGORY).items()}

    def codes(self, variant: str) -> np.ndarray:
        return self.tables[variant].codes


# --------------------------------------------------------------- figure 1 --
def fig1_icicle(D: Data, out: Path) -> None:
    """Prefix trees, one per quantizer, digit 1 at the top."""
    class_counts = np.bincount(D.lab1[D.lab1 >= 0])
    top = [int(c) for c in np.argsort(-class_counts)[:7]]
    slot_of = {c: S.CATEGORY_SLOTS[i] for i, c in enumerate(top)}

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    rows_out = []
    for ax, q in zip(axes, S.QUANTIZER_ORDER):
        variant = PRIMARY[q]
        codes = D.codes(variant)
        n, depth_max = codes.shape
        rects, colors = [], []
        stats = defaultdict(lambda: {"n_clusters": 0, "singletons": 0, "sizes": []})

        def rec(idx: np.ndarray, depth: int, x0: float):
            vals = codes[idx, depth - 1]
            order = np.argsort(vals, kind="stable")
            uniq, starts, counts = np.unique(vals[order], return_index=True, return_counts=True)
            children = []
            for u, s, c in zip(uniq, starts, counts):
                members = idx[order[s:s + c]]
                maj, _, _ = majority(D.lab1[members])
                key = (S.CATEGORY_SLOTS.index(slot_of[maj]) if maj in slot_of else 99, -c)
                children.append((key, members, maj))
            children.sort(key=lambda t: t[0])
            x = x0
            for _, members, maj in children:
                w = len(members) / n
                rects.append(Rectangle((x, depth - 0.34), w, 0.68))
                colors.append(slot_of.get(maj, S.OTHER))
                st = stats[depth]
                st["n_clusters"] += 1
                st["singletons"] += int(len(members) == 1)
                st["sizes"].append(len(members))
                if depth < depth_max:
                    rec(members, depth + 1, x)
                x += w

        rec(np.arange(n), 1, 0.0)
        widths = np.array([r.get_width() for r in rects])
        # A surface-coloured edge is the "2px gap"; skip it where the rectangle
        # itself is narrower than the edge would be.
        wide = widths * 2.1 * 200 > 3  # panel ~2.1 in wide at 200 dpi
        pc = PatchCollection([r for r, ok in zip(rects, wide) if ok],
                             facecolors=[c for c, ok in zip(colors, wide) if ok],
                             edgecolors=S.SURFACE, linewidths=0.35)
        ax.add_collection(pc)
        pc2 = PatchCollection([r for r, ok in zip(rects, wide) if not ok],
                              facecolors=[c for c, ok in zip(colors, wide) if not ok],
                              edgecolors="none", linewidths=0)
        ax.add_collection(pc2)
        for d in range(1, depth_max + 1):
            st = stats[d]
            sizes = np.array(st["sizes"])
            single = st["singletons"] / n
            # Inverted axis: va="bottom" grows toward smaller y, i.e. into the
            # 0.32-unit gap below the previous band, never into a band.
            ax.text(1.0, d - 0.36, f"{st['n_clusters']:,} groups · {single:.0%} of items alone",
                    ha="right", va="bottom", fontsize=6.2, color=S.INK_2)
            rows_out.append({"variant": variant, "depth": d, "n_groups": st["n_clusters"],
                             "singleton_item_share": round(single, 4),
                             "median_group_size": float(np.median(sizes)),
                             "largest_group": int(sizes.max())})
        ax.set_xlim(0, 1)
        ax.set_ylim(depth_max + 0.55, 0.3)
        ax.set_xticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{S.QUANTIZER_LABEL[q]}  ·  4 digits × 128 codes", loc="left")
    axes[0].set_yticks(range(1, 5))
    axes[0].set_yticklabels([f"digits 1–{d}" if d > 1 else "digit 1" for d in range(1, 5)])
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_ylabel("items sharing the prefix", color=S.INK_2)
    handles = [Patch(facecolor=slot_of[c], label=D.vocab1[c]) for c in top]
    handles.append(Patch(facecolor=S.OTHER, label="other / unlabelled"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               fontsize=6.8, handlelength=1.2, columnspacing=1.2)
    fig.suptitle("Each row splits the row above it: width = share of the 3,105 items, colour = majority level-1 category",
                 fontsize=7.5, color=S.INK_2, y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    S.save(fig, out / "fig1_icicle")
    write_rows(out / "fig1_icicle.csv", rows_out)


# --------------------------------------------------------------- figure 2 --
def fig2_cluster_sizes(D: Data, out: Path) -> None:
    """Item-weighted ECDF of prefix-group size at each depth."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    rows_out = []
    for ax, q in zip(axes, S.QUANTIZER_ORDER):
        codes = D.codes(PRIMARY[q])
        n = len(codes)
        for d in range(1, codes.shape[1] + 1):
            group, counts = prefix_groups(codes, d)
            size_of_item = counts[group]
            xs = np.sort(size_of_item)
            ys = np.arange(1, n + 1) / n
            ax.step(xs, ys, where="post", color=S.DEPTH_RAMP[d], label=f"digits 1–{d}" if d > 1 else "digit 1")
            single = float((size_of_item == 1).mean())
            ax.plot([1], [single], marker="o", color=S.DEPTH_RAMP[d], markersize=3.5, linestyle="none")
            for pct in (50, 90):
                rows_out.append({"variant": PRIMARY[q], "depth": d, "stat": f"item_p{pct}_group_size",
                                 "value": float(np.percentile(size_of_item, pct))})
            rows_out.append({"variant": PRIMARY[q], "depth": d, "stat": "singleton_item_share", "value": single})
        ax.set_xscale("log")
        ax.set_xlim(0.9, 500)
        ax.set_ylim(0, 1.02)
        ax.set_title(S.QUANTIZER_LABEL[q], loc="left")
        ax.set_xlabel("size of the item's prefix group (items)")
        ax.grid(True, axis="x", color=S.GRID, linewidth=0.5)
    axes[0].set_ylabel("share of items in a group ≤ size")
    axes[-1].legend(loc="lower right", fontsize=6.8, title="prefix", title_fontsize=6.8)
    fig.tight_layout()
    S.save(fig, out / "fig2_cluster_sizes")
    write_rows(out / "fig2_cluster_sizes.csv", rows_out)


# --------------------------------------------------------------- figure 3 --
def _variant_lines(ax, df, ycol, q, depth_col="depth"):
    """Nine variants of one quantizer, colour = width, shorter fits fainter."""
    sub = df[df.quantizer == q]
    for w in (128, 256, 512):
        for d_fit in (3, 4, 5):
            v = f"{q}_{d_fit}codebook_{w}"
            s = sub[(sub.variant == v) & (sub[depth_col] > 0)].sort_values(depth_col)
            if s.empty:
                continue
            alpha = {3: 0.45, 4: 0.7, 5: 1.0}[d_fit]
            ax.plot(s[depth_col], s[ycol], color=S.WIDTH_RAMP[w], alpha=alpha,
                    marker="o", markersize=2.8, linewidth=1.3 if d_fit == 5 else 1.0,
                    label=f"width {w}" if d_fit == 5 else None)


def fig3_geometry(D: Data, out: Path) -> None:
    g = D.geometry.copy()
    r = D.refinement.copy()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    for j, q in enumerate(S.QUANTIZER_ORDER):
        _variant_lines(axes[0, j], g, "median_radius", q)
        _variant_lines(axes[1, j], r, "refinement_index", q)
        axes[0, j].set_title(S.QUANTIZER_LABEL[q], loc="left")
        axes[1, j].set_xlabel("prefix depth (digits)")
        axes[1, j].set_xticks([1, 2, 3, 4, 5])
        axes[1, j].set_ylim(0, 1.02)
        axes[0, j].set_ylim(0, 45)
    axes[0, 0].set_ylabel("median RMS radius of\nmulti-item groups")
    # refinement_index = 1 - R2_within / E[R2 of a random partition into the
    # same number of groups] = 1 - R2 / ((n-k)/(n-1)); see analysis/refinement.py
    axes[1, 0].set_ylabel("tightness beyond a random\npartition (same no. of groups)")
    S.null_line(axes[1, 0], 0, "random partition", xmax=5.2)
    axes[0, 2].legend(loc="upper right", fontsize=6.8)
    fig.text(0.005, 0.5, "", rotation=90)
    fig.suptitle("Solid = 5-digit fit; fainter = 3- and 4-digit fits (RQ-KMeans depths are one nested fit; RQ-VAE and MQ refit per depth)",
                 fontsize=7, color=S.INK_2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    S.save(fig, out / "fig3_geometry")
    cols = ["variant", "quantizer", "n_codebook", "codebook_size", "depth", "n_clusters",
            "median_radius", "r2_within", "frac_items_in_singletons"]
    merged = g[cols].merge(r[["variant", "depth", "r2_null", "refinement_index", "knn_lift"]],
                           on=["variant", "depth"], how="left")
    merged.to_csv(out / "fig3_geometry.csv", index=False)


# --------------------------------------------------------------- figure 4 --
def fig4_coherence(D: Data, out: Path) -> None:
    """Majority-category purity minus its shuffled-label null, per depth."""
    rng = np.random.default_rng(SEED)
    rows = []
    for variant, table in D.tables.items():
        q, d_fit, w = parse_variant(variant)
        codes = table.codes
        for field, lab in (("cat_l1", D.lab1), ("cat_l2", D.lab2)):
            for depth in range(1, d_fit + 1):
                group, counts = prefix_groups(codes, depth)
                res = purity_multi(group, counts, lab, rng, N_PERM)
                res.update({"purity_all_items": item_weighted_purity(group, lab)})
                rows.append({"variant": variant, "quantizer": q, "n_codebook": d_fit,
                             "codebook_size": w, "field": field, "depth": depth, **res})
    df = pd.DataFrame(rows)
    df.to_csv(out / "fig4_coherence.csv", index=False)
    # Below this share of items the "grouped" population is a handful of
    # duplicate pairs; show the point hollow and do not draw a line to it.
    thin = df.multi_item_share < 0.10
    plot_df = df.copy()
    plot_df.loc[thin, "excess"] = np.nan

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True, sharey="row")
    for i, field in enumerate(("cat_l1", "cat_l2")):
        for j, q in enumerate(S.QUANTIZER_ORDER):
            ax = axes[i, j]
            _variant_lines(ax, plot_df[plot_df.field == field], "excess", q)
            hollow = df[(df.field == field) & (df.quantizer == q) & thin]
            ax.plot(hollow.depth, hollow.excess, linestyle="none", marker="o", markersize=3.2,
                    markerfacecolor=S.SURFACE, markeredgecolor=S.MUTED, markeredgewidth=0.8)
            ax.set_ylim(0, 0.8)
            if i == 0:
                ax.set_title(S.QUANTIZER_LABEL[q], loc="left")
            if i == 1:
                ax.set_xlabel("prefix depth (digits)")
                ax.set_xticks([1, 2, 3, 4, 5])
        S.null_line(axes[i, 0], 0, "shuffled labels", xmax=5.2)
    axes[0, 0].set_ylabel("level-1 category purity\nabove shuffled-label null")
    axes[1, 0].set_ylabel("level-2 category purity\nabove shuffled-label null")
    axes[0, 2].legend(loc="upper right", fontsize=6.8)
    hollow_h = Line2D([], [], linestyle="none", marker="o", markerfacecolor=S.SURFACE,
                      markeredgecolor=S.MUTED, label="<10% of items still share a group")
    axes[1, 2].legend(handles=[hollow_h], loc="upper right", fontsize=6.3)
    fig.suptitle("Purity = share of items (in multi-item prefix groups) whose category is the group majority; "
                 "the null keeps the groups and shuffles labels", fontsize=7, color=S.INK_2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    S.save(fig, out / "fig4_coherence")


# --------------------------------------------------------------- figure 5 --
def fig5_digit_info(D: Data, out: Path) -> None:
    """What a digit POSITION knows on its own vs given the earlier digits."""
    rows = []
    for variant, table in D.tables.items():
        q, d_fit, w = parse_variant(variant)
        codes = table.codes
        for j in range(d_fit):
            _, inv = np.unique(codes[:, j], return_inverse=True)
            inv = inv.reshape(-1)
            rows.append({"variant": variant, "quantizer": q, "n_codebook": d_fit,
                         "codebook_size": w, "digit": j + 1,
                         "codes_used": int(inv.max() + 1),
                         "utilization": float((inv.max() + 1) / w),
                         "standalone_between_share": between_share(D.z, inv)})
    df = pd.DataFrame(rows)
    att = D.attribute[D.attribute.field == "cat_l1"][
        ["variant", "digit", "marginal_ami", "conditional_ami", "conditional_supported"]].copy()
    att["digit"] = att["digit"] + 1
    # Digit 1 has nothing to condition on: its "conditional" AMI is the marginal
    # AMI, which needs no within-parent support, so it is drawn filled.
    att.loc[att.digit == 1, "conditional_supported"] = True
    df = df.merge(att, on=["variant", "digit"], how="left")
    df.to_csv(out / "fig5_digit_info.csv", index=False)
    # Lines only through supported estimates; unsupported ones are hollow dots.
    plot_df = df.copy()
    unsupported = plot_df.conditional_supported == False  # noqa: E712
    plot_df.loc[unsupported, "conditional_ami"] = np.nan

    panels = [("utilization", "share of the codebook used"),
              ("standalone_between_share", "variance explained by\nthis digit alone"),
              ("marginal_ami", "category AMI of\nthis digit alone"),
              ("conditional_ami", "category AMI added\ngiven earlier digits")]
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.0), sharex=True, sharey="row")
    for i, (col, ylab) in enumerate(panels):
        for j, q in enumerate(S.QUANTIZER_ORDER):
            ax = axes[i, j]
            _variant_lines(ax, plot_df, col, q, depth_col="digit")
            if col == "conditional_ami":
                # unsupported estimates (too few labelled items per parent) hollow
                sub = df[(df.quantizer == q) & unsupported]
                ax.plot(sub.digit, sub[col], linestyle="none", marker="o", markersize=3.2,
                        markerfacecolor=S.SURFACE, markeredgecolor=S.MUTED, markeredgewidth=0.8,
                        clip_on=True)
            if i == 0:
                ax.set_title(S.QUANTIZER_LABEL[q], loc="left")
            if i == 3:
                ax.set_xlabel("digit position")
                ax.set_xticks([1, 2, 3, 4, 5])
        axes[i, 0].set_ylabel(ylab)
    axes[0, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylim(0, 0.8)
    axes[2, 0].set_ylim(0, 0.6)
    axes[3, 0].set_ylim(-0.05, 0.75)
    axes[0, 2].legend(loc="lower left", fontsize=6.8)
    hollow = Line2D([], [], linestyle="none", marker="o", markerfacecolor=S.SURFACE,
                    markeredgecolor=S.MUTED, label="unsupported (few labelled items per parent)")
    axes[3, 2].legend(handles=[hollow], loc="upper right", fontsize=6.3)
    fig.suptitle("Solid = 5-digit fit; fainter = 3- and 4-digit fits. AMI is against the 25 level-1 categories.",
                 fontsize=7, color=S.INK_2, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    S.save(fig, out / "fig5_digit_info")


# ------------------------------------------------------- embedding maps --
def embedding_map(z: np.ndarray, method: str, cache: Path | None = None,
                  seed: int = SEED) -> np.ndarray:
    """2-D map of the item embedding; t-SNE is cached because it takes minutes
    on a loaded login node and every map figure needs the same one."""
    from sklearn.decomposition import PCA
    path = None if cache is None else cache / f"map-{method}-{seed}.npy"
    if path is not None and path.exists():
        return np.load(path)
    if method == "pca":
        xy = PCA(2, random_state=seed).fit_transform(z)
    else:
        from sklearn.manifold import TSNE
        z50 = PCA(50, random_state=seed).fit_transform(z)
        xy = TSNE(2, init="pca", perplexity=30, random_state=seed).fit_transform(z50)
    if path is not None:
        np.save(path, xy)
    return xy


def _scatter_by_code(ax, xy, code, top_k=7, order_by_size=True, legend=True) -> dict:
    """Colour the top-k codes by size in fixed slots, everything else grey.

    Returns {code: colour} for the coloured codes so a companion panel can
    reuse the same assignment.
    """
    vals, counts = np.unique(code, return_counts=True)
    order = np.argsort(-counts) if order_by_size else np.arange(len(vals))
    top = vals[order[:top_k]]
    rest = ~np.isin(code, top)
    ax.scatter(xy[rest, 0], xy[rest, 1], s=5, color=S.OTHER, linewidths=0, alpha=0.8, zorder=1)
    colour = {}
    for i, v in enumerate(top):
        m = code == v
        colour[int(v)] = S.CATEGORY_SLOTS[i]
        ax.scatter(xy[m, 0], xy[m, 1], s=7, color=S.CATEGORY_SLOTS[i], linewidths=0.4,
                   edgecolors=S.SURFACE, zorder=2, label=f"{v} ({m.sum()})")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(S.GRID)
    if legend:
        handles, texts = ax.get_legend_handles_labels()
        if rest.any():
            # Grey is every code outside the top k, not noise or empty space;
            # say so in the legend so the reader does not have to guess.
            handles.append(Line2D([], [], linestyle="none", marker="o", markersize=3,
                                  markerfacecolor=S.OTHER, markeredgecolor=S.OTHER))
            texts.append(f"other {len(vals) - len(top)} codes ({int(rest.sum())})")
        leg = ax.legend(handles, texts, loc="upper right", fontsize=5.2, markerscale=0.9,
                        handletextpad=0.3, borderpad=0.3, labelspacing=0.15, frameon=True,
                        framealpha=0.85, edgecolor="none", title="code (items)",
                        title_fontsize=5.2)
        leg.get_frame().set_facecolor(S.SURFACE)
    return colour


def fig6_zoom(D: Data, out: Path, method: str) -> None:
    """Zoom into one RQ-KMeans branch: global map, then local PCA per level."""
    from sklearn.decomposition import PCA
    variant = PRIMARY["rqkmeans"]
    codes = D.codes(variant)
    xy = embedding_map(D.z, method, cache=out)
    idx = np.arange(len(codes))
    path: list[int] = []
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_tab = fig.add_subplot(gs[0, 3])
    rows = []
    colour_c: dict = {}
    idx_c = idx
    for level, ax in enumerate(axes):
        if level == 0:
            _scatter_by_code(ax, xy, codes[:, 0])
            ax.set_title("all 3,105 items · by digit 1", loc="left", fontsize=6.8)
        else:
            local = PCA(2, random_state=SEED).fit_transform(D.z[idx])
            colour_c = _scatter_by_code(ax, local, codes[idx, level])
            idx_c = idx
            prefix = "(" + ")(".join(str(c) for c in path) + ")"
            maj, cnt, tot = majority(D.lab1[idx])
            ax.set_title(f"{prefix} · {len(idx)} items · by digit {level + 1}",
                         loc="left", fontsize=6.8)
            ax.text(0.02, 0.02, f"{D.vocab1[maj]}: {cnt}/{tot}", transform=ax.transAxes,
                    fontsize=5.8, color=S.INK_2, va="bottom")
        ax.set_aspect("equal", adjustable="datalim")
        if level == 2:
            break
        # Next branch: the largest child that is still worth zooming into.
        vals, counts = np.unique(codes[idx, level], return_counts=True)
        chosen = None
        for v in vals[np.argsort(-counts)]:
            members = idx[codes[idx, level] == v]
            if len(members) >= 8 and len(np.unique(codes[members, level + 1])) >= 3:
                chosen = v
                break
        chosen = vals[np.argmax(counts)] if chosen is None else chosen
        path.append(int(chosen))
        idx = idx[codes[idx, level] == chosen]
        rows.append({"level": level + 1, "prefix": path.copy(), "n_items": int(len(idx))})

    # Panel D: the items behind panel C, so the reader sees what digits 3-4 separate.
    ax_tab.axis("off")
    prefix_c = "(" + ")(".join(str(c) for c in path) + ")"
    ax_tab.set_title(f"{prefix_c}: digits 3·4 and titles", loc="left", fontsize=6.8)
    members = sorted(idx_c, key=lambda i: (codes[i, 2], codes[i, 3], D.titles.get(D.keys[i], "")))
    max_rows = 22
    dy = 1.0 / (max_rows + 1)
    for r, i in enumerate(members[:max_rows]):
        y = 1.0 - (r + 0.8) * dy
        c = colour_c.get(int(codes[i, 2]), S.OTHER)
        ax_tab.add_patch(Rectangle((0.0, y - 0.35 * dy), 0.035, 0.7 * dy, facecolor=c,
                                   transform=ax_tab.transAxes, clip_on=False))
        title = D.titles.get(D.keys[i], D.keys[i])
        title = title if len(title) <= 36 else title[:35].rstrip() + "…"
        ax_tab.text(0.05, y, f"{codes[i, 2]:>3}·{codes[i, 3]:<3} {title}", transform=ax_tab.transAxes,
                    fontsize=5.0, va="center", ha="left", color=S.INK, family="DejaVu Sans")
    if len(members) > max_rows:
        ax_tab.text(0.05, 1.0 - (max_rows + 0.8) * dy, f"… {len(members) - max_rows} more",
                    transform=ax_tab.transAxes, fontsize=5.0, va="center", color=S.MUTED)
    fig.suptitle(f"RQ-KMeans 4×128: each map zooms into one group of the previous one "
                 f"({'t-SNE' if method == 'tsne' else 'PCA'} of the shared item embedding; local PCA inside a group)",
                 fontsize=6.8, color=S.INK_2, y=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02)
    S.save(fig, out / "fig6_zoom")
    (out / "fig6_zoom.json").write_text(json.dumps(
        {"variant": variant, "method": method, "path": rows,
         "panel_c_members": [{"asin": D.keys[i], "sid": [int(c) for c in codes[i]],
                              "title": D.titles.get(D.keys[i], "")} for i in members]}, indent=1))


def fig7_standalone(D: Data, out: Path, method: str) -> None:
    """Same map, coloured by one digit position at a time, RQ-KMeans vs MQ."""
    xy = embedding_map(D.z, method, cache=out)
    quantizers = ("rqkmeans", "MQ")
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8))
    rows = []
    for i, q in enumerate(quantizers):
        codes = D.codes(PRIMARY[q])
        for j in range(4):
            ax = axes[i, j]
            _scatter_by_code(ax, xy, codes[:, j], legend=False)
            _, inv = np.unique(codes[:, j], return_inverse=True)
            share = between_share(D.z, inv.reshape(-1))
            ax.set_title(f"digit {j + 1} alone · {share:.0%} of variance", loc="left", fontsize=6.8)
            rows.append({"variant": PRIMARY[q], "digit": j + 1, "standalone_between_share": share})
            ax.set_aspect("equal", adjustable="datalim")
        axes[i, 0].set_ylabel(f"{S.QUANTIZER_LABEL[q]} 4×128", fontsize=7.5, color=S.INK)
    fig.suptitle("Colour = the seven most populated codes of that one digit, ignoring the others (grey = remaining codes)",
                 fontsize=6.8, color=S.INK_2, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    S.save(fig, out / "fig7_standalone")
    write_rows(out / "fig7_standalone.csv", rows)


# --------------------------------------------------------------- figure 8 --
def fig8_collisions(D: Data, out: Path) -> None:
    c = D.collisions.copy()
    c["collision_pct"] = 100 * c.collision_rate
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, q in zip(axes, S.QUANTIZER_ORDER):
        sub = c[c.quantizer == q]
        for w in (128, 256, 512):
            s = sub[sub.codebook_size == w].sort_values("n_codebook")
            ax.plot(s.n_codebook, s.collision_pct, color=S.WIDTH_RAMP[w], marker="o",
                    markersize=3.2, label=f"width {w}")
        ax.set_yscale("log")
        ax.set_ylim(0.2, 60)
        ax.set_xticks([3, 4, 5])
        ax.set_xlabel("SID depth (digits)")
        ax.set_title(S.QUANTIZER_LABEL[q], loc="left")
        ax.grid(True, axis="y", which="both", color=S.GRID, linewidth=0.5)
    floor = 100 * float(c.floor_rate.iloc[0])
    S.null_line(axes[0], floor, "identical-text floor (11 items)", xmax=5.15)
    axes[0].set_ylabel("items sharing a full SID (%)")
    axes[0].set_yticks([0.35, 1, 3, 10, 30])
    axes[0].set_yticklabels(["0.35", "1", "3", "10", "30"])
    axes[2].legend(loc="upper right", fontsize=6.8)
    fig.tight_layout()
    S.save(fig, out / "fig8_collisions")
    c[["variant", "quantizer", "n_codebook", "codebook_size", "unique_sids", "collisions",
       "collision_rate", "max_bucket", "digit0_utilization", "mean_utilization"]].to_csv(
        out / "fig8_collisions.csv", index=False)


# ------------------------------------------------------------------- main --
FIGURES = {
    "1": fig1_icicle, "2": fig2_cluster_sizes, "3": fig3_geometry, "4": fig4_coherence,
    "5": fig5_digit_info, "6": fig6_zoom, "7": fig7_standalone, "8": fig8_collisions,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--mapping-run", default="194268")
    ap.add_argument("--only", default="12345678")
    ap.add_argument("--map", choices=("tsne", "pca"), default="tsne")
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = args.out or paths.DERIVED / "semantic_mapping" / f"figures-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    S.apply()
    D = Data(args.mapping_run)
    for k in args.only:
        fn = FIGURES[k]
        if k in ("6", "7"):
            fn(D, out, args.map)
        else:
            fn(D, out)
        print("wrote", fn.__name__)
    (out / "manifest.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).isoformat(),
        "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "mapping_run": args.mapping_run, "map": args.map, "seed": SEED, "n_perm": N_PERM,
        "primary_variants": PRIMARY,
    }, indent=1))
    print("out", out)


if __name__ == "__main__":
    main()
