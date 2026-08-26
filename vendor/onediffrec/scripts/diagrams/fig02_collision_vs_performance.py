"""Fig 2 - the headline: reported accuracy tracks collision rate."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import style as S

CORE = ["Qwen2.5-1.5B", "Mask Diffusion", "DiffGRM"]


def load_core():
    """The balanced 3x3x3 grid run with both paradigms (extra backbones out)."""
    j = pd.read_csv(S.DATA / "joined.csv")
    return j[j.model.isin(CORE)].copy()


def main():
    S.apply_base_style()
    import matplotlib.pyplot as plt

    d = load_core()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9))

    for ax, metric in zip(axes, ["HR@10", "NDCG@10"]):
        x = np.log10(d.collision_pct)
        y = d[metric]
        # Pooled trend across every configuration in the grid.
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs, a + b * xs, color=S.INK, lw=1.4, ls="--",
                alpha=0.55, zorder=1)

        for tok in S.TOKENIZER_ORDER:
            for fam in S.FAMILY_ORDER:
                g = d[(d.tokenizer == tok) & (d.family == fam)]
                ax.scatter(g.collision_pct, g[metric],
                           s=[S.DEPTH_SIZE[k] for k in g.depth],
                           c=S.TOKENIZER_COLOR[tok], marker=S.FAMILY_MARKER[fam],
                           edgecolors=S.SURFACE, linewidths=1.6, zorder=3)

        rho, p = spearmanr(d.collision_pct, y)
        ax.set_xscale("log")
        ax.set_xlabel("Collision rate (%, log scale)")
        ax.set_ylabel(f"Reported {metric}")
        ax.set_title(metric)
        S.despine(ax)
        ax.annotate(f"Spearman $\\rho$ = {rho:+.2f}\n"
                    f"p = {p:.1e}\n"
                    f"+{b:.1f} pts per 10x collisions",
                    xy=(0.035, 0.965), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8.4, color=S.INK,
                    bbox=dict(boxstyle="round,pad=0.45", fc=S.SURFACE,
                              ec=S.GRID, lw=0.8))

    # Legend: hue = tokenizer, shape = paradigm, size = depth.
    handles = [plt.Line2D([], [], marker="s", ls="", markersize=7,
                          markerfacecolor=S.TOKENIZER_COLOR[t],
                          markeredgecolor=S.SURFACE, label=S.tok_label(t))
               for t in S.TOKENIZER_ORDER]
    handles += [plt.Line2D([], [], marker=S.FAMILY_MARKER[f], ls="",
                           markersize=7, markerfacecolor=S.MUTED,
                           markeredgecolor=S.SURFACE, label=f)
                for f in S.FAMILY_ORDER]
    handles += [plt.Line2D([], [], marker="o", ls="",
                           markersize=np.sqrt(S.DEPTH_SIZE[k]),
                           markerfacecolor=S.MUTED, markeredgecolor=S.SURFACE,
                           label=f"{k} digits") for k in (3, 4, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, 0.055), columnspacing=1.4)

    S.subtitle(fig,
               "Reported accuracy rises with collision rate, not with tokenizer quality",
               "Every point is one (tokenizer x depth x codebook x paradigm) run on the "
               "Industrial split. The configurations that address items least uniquely "
               "post the highest scores.")
    fig.tight_layout(rect=[0, 0.135, 1, 0.86])
    S.footnote(fig, "n=54 runs. Collisions inflate scores because "
                    "genrec/evaluator.py scores a hit on semantic-ID equality "
                    "(cur_pred == cur_label), not item identity.", y=0.012)
    S.save(fig, "fig02_collision_vs_performance")


if __name__ == "__main__":
    main()
