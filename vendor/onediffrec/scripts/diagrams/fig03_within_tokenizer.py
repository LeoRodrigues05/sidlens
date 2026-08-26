"""Fig 3 - the relationship is not a between-tokenizer artefact.

Faceting by tokenizer holds the quantiser fixed, so the only thing moving
inside each panel is SID depth and codebook size. If collisions were merely a
proxy for "which tokenizer", the within-panel slopes would vanish.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import style as S
from fig02_collision_vs_performance import load_core


def main():
    S.apply_base_style()
    import matplotlib.pyplot as plt

    d = load_core()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.5), sharey=True)

    for ax, tok in zip(axes, S.TOKENIZER_ORDER):
        g = d[d.tokenizer == tok]
        colour = S.TOKENIZER_COLOR[tok]

        for fam in S.FAMILY_ORDER:
            gg = g[g.family == fam]
            ax.scatter(gg.collision_pct, gg["HR@10"],
                       s=[S.DEPTH_SIZE[k] for k in gg.depth],
                       c=colour, marker=S.FAMILY_MARKER[fam],
                       edgecolors=S.SURFACE, linewidths=1.6, zorder=3)

        rho, p = spearmanr(g.collision_pct, g["HR@10"])
        spread = g.collision_pct.max() / g.collision_pct.min()
        if spread > 3:  # a fit is only meaningful with real spread in x
            x = np.log10(g.collision_pct)
            b, a = np.polyfit(x, g["HR@10"], 1)
            xs = np.linspace(x.min(), x.max(), 40)
            ax.plot(10 ** xs, a + b * xs, color=colour, lw=1.4, ls="--",
                    alpha=0.7, zorder=2)
            note = f"$\\rho$ = {rho:+.2f}   p = {p:.1g}"
        else:
            note = (f"$\\rho$ = {rho:+.2f}   p = {p:.2g}\n"
                    "collision rate is flat here\n(0.35-1.2%): no spread to test")

        ax.set_xscale("log")
        ax.set_xlim(0.2, 60)
        ax.set_title(S.tok_label(tok), color=colour)
        ax.set_xlabel("Collision rate (%, log scale)")
        S.despine(ax)
        ax.annotate(note, xy=(0.04, 0.96), xycoords="axes fraction",
                    ha="left", va="top", fontsize=8.2, color=S.INK,
                    bbox=dict(boxstyle="round,pad=0.4", fc=S.SURFACE,
                              ec=S.GRID, lw=0.8))

    axes[0].set_ylabel("Reported HR@10")

    # The natural experiment: one RQ-KMeans cell lands on the collision floor.
    g = d[(d.tokenizer == "RQ-KMeans")]
    lo = g[g.collision_pct < 1]["HR@10"].mean()
    hi = g[g.collision_pct > 25]["HR@10"].mean()
    axes[1].annotate(
        f"same tokenizer, 5 digits x 512:\ncollisions fall to the floor,\n"
        f"HR@10 falls {hi:.1f} -> {lo:.1f}",
        xy=(S.FLOOR_PCT, lo), xytext=(1.5, 22.6), fontsize=7.4, color=S.INK_2,
        ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=0.9))

    handles = [plt.Line2D([], [], marker=S.FAMILY_MARKER[f], ls="", markersize=7,
                          markerfacecolor=S.MUTED, markeredgecolor=S.SURFACE,
                          label=f) for f in S.FAMILY_ORDER]
    handles += [plt.Line2D([], [], marker="o", ls="",
                           markersize=np.sqrt(S.DEPTH_SIZE[k]),
                           markerfacecolor=S.MUTED, markeredgecolor=S.SURFACE,
                           label=f"{k} digits") for k in (3, 4, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.055))

    S.subtitle(fig,
               "The effect holds inside each tokenizer, so it is not a quantiser artefact",
               "Within RQ-KMeans and MQ, varying only depth and codebook size moves "
               "collisions and reported HR@10 together. RQ-VAE cannot test the claim: "
               "its collision rate barely varies.")
    fig.tight_layout(rect=[0, 0.14, 1, 0.86])
    S.footnote(fig, "Marker size encodes SID depth; deeper SIDs and larger "
                    "codebooks both sit at the low-collision, low-score end.")
    S.save(fig, "fig03_within_tokenizer")


if __name__ == "__main__":
    main()
