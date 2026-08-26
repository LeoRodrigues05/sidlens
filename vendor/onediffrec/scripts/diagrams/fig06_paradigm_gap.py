"""Fig 6 - where masked diffusion loses to autoregression.

Pairing the two paradigms on identical SIDs cancels the collision confound
entirely: both models see the same tokenization, so the difference between them
is a clean model-level comparison.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import style as S
from fig02_collision_vs_performance import load_core


def paired(d, metric="HR@10"):
    d = d.copy()
    d["fam"] = np.where(d.family == "Autoregressive", "AR", "DIFF")
    p = d.pivot_table(index=["tokenizer", "codebook", "depth", "collision_pct"],
                      columns="fam", values=metric).reset_index()
    p["gap"] = p.DIFF - p.AR
    return p


def main():
    S.apply_base_style()
    import matplotlib.pyplot as plt

    p = paired(load_core())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

    # --- left: gap vs SID depth ------------------------------------------
    ax = axes[0]
    ax.axhline(0, color=S.AXIS, lw=1.0, zorder=1)
    rng = np.random.default_rng(0)
    for tok in S.TOKENIZER_ORDER:
        g = p[p.tokenizer == tok]
        jitter = rng.uniform(-0.13, 0.13, len(g))
        ax.scatter(g.depth + jitter, g.gap, s=52,
                   c=S.TOKENIZER_COLOR[tok], edgecolors=S.SURFACE,
                   linewidths=1.6, zorder=3, label=S.tok_label(tok))
    means = p.groupby("depth").gap.mean()
    ax.plot(means.index, means.values, color=S.INK, lw=1.8, ls="--",
            alpha=0.6, zorder=2)
    for k, v in means.items():
        ax.annotate(f"{v:+.1f}", xy=(k, v), xytext=(0, -15),
                    textcoords="offset points", ha="center", fontsize=7.8,
                    color=S.INK, fontweight="bold")

    rho, pv = spearmanr(p.depth, p.gap)
    ax.set_xticks([3, 4, 5])
    ax.set_xlabel("SID depth (digits)")
    ax.set_ylabel("HR@10 gap  (diffusion - autoregressive)")
    ax.set_title("Diffusion degrades as semantic IDs get longer")
    S.despine(ax)
    ax.annotate(f"Spearman $\\rho$ = {rho:+.2f}, p = {pv:.1g}",
                xy=(0.035, 0.05), xycoords="axes fraction", fontsize=8.2,
                color=S.INK, bbox=dict(boxstyle="round,pad=0.4", fc=S.SURFACE,
                                       ec=S.GRID, lw=0.8))
    ax.legend(loc="upper right")

    # --- right: gap vs collision rate (the null result) -------------------
    ax = axes[1]
    ax.axhline(0, color=S.AXIS, lw=1.0, zorder=1)
    for tok in S.TOKENIZER_ORDER:
        g = p[p.tokenizer == tok]
        ax.scatter(g.collision_pct, g.gap,
                   s=[S.DEPTH_SIZE[k] for k in g.depth],
                   c=S.TOKENIZER_COLOR[tok], edgecolors=S.SURFACE,
                   linewidths=1.6, zorder=3)
    rho2, pv2 = spearmanr(p.collision_pct, p.gap)
    ax.set_xscale("log")
    ax.set_xlabel("Collision rate (%, log scale)")
    ax.set_title("But the gap is unrelated to collisions")
    S.despine(ax)
    ax.annotate(f"Spearman $\\rho$ = {rho2:+.2f}, p = {pv2:.2g}\n"
                "no relationship: collisions inflate\nboth paradigms equally",
                xy=(0.035, 0.05), xycoords="axes fraction", fontsize=8.2,
                color=S.INK, bbox=dict(boxstyle="round,pad=0.4", fc=S.SURFACE,
                                       ec=S.GRID, lw=0.8))

    S.subtitle(fig,
               "Pairing the paradigms on identical SIDs isolates a real model effect",
               "Each point is one tokenizer configuration, scored by both paradigms. "
               "Below zero means autoregression wins. Marker size encodes SID depth.")
    fig.tight_layout(rect=[0, 0.055, 1, 0.865])
    S.footnote(fig, "n=27 matched pairs. Because both models consume the same "
                    "semantic IDs, any collision-driven inflation cancels in the "
                    "difference -- so this comparison is not confounded by fig 2.")
    S.save(fig, "fig06_paradigm_gap")

    print(f"\n  gap vs depth  rho={rho:+.3f} p={pv:.3g}")
    print(f"  gap vs collision rho={rho2:+.3f} p={pv2:.3g}")
    print(p.groupby("depth").gap.agg(["mean", "min", "max"]).round(2))


if __name__ == "__main__":
    main()
