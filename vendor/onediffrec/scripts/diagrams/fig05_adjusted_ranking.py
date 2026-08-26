"""Fig 5 - the leaderboard after discounting the collision-attributable score.

The adjustment is deliberately simple and model-based: take the fitted
log10(collision) slope from fig04 and evaluate every configuration at the
irreducible floor rate instead of its own. This is an ESTIMATE of what the
grid would look like if all tokenizers addressed items equally uniquely -- not
a re-measurement. It is shown to answer "does the ranking survive?", not to
supply a corrected number for publication.
"""

import numpy as np
import pandas as pd

import style as S
from fig02_collision_vs_performance import load_core
from fig04_regression import fit


def main():
    S.apply_base_style()
    import matplotlib.pyplot as plt

    d = load_core().copy()
    beta, se, crit, r2, dof = fit(d)
    slope = beta[0]  # HR@10 points per decade of collision rate

    d["adjusted"] = d["HR@10"] - slope * (
        np.log10(d.collision_pct) - np.log10(S.FLOOR_PCT))
    d["config"] = (d.tokenizer.map(S.tok_label) + "  " + d.depth.astype(str)
                   + "d x " + d.codebook.astype(str))

    # Average the two paradigms so each row is one tokenizer configuration.
    agg = (d.groupby(["config", "tokenizer"], as_index=False)
             .agg(reported=("HR@10", "mean"), adjusted=("adjusted", "mean"),
                  collision=("collision_pct", "mean"))
             .sort_values("reported"))

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    y = np.arange(len(agg))

    for yi, row in zip(y, agg.itertuples()):
        c = S.TOKENIZER_COLOR[row.tokenizer]
        ax.plot([row.adjusted, row.reported], [yi, yi], color=c, lw=1.6,
                alpha=0.4, zorder=1, solid_capstyle="round")
        ax.scatter([row.adjusted], [yi], s=46, facecolor=S.SURFACE,
                   edgecolors=c, linewidths=1.8, zorder=3)
        ax.scatter([row.reported], [yi], s=52, color=c,
                   edgecolors=S.SURFACE, linewidths=1.6, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(agg.config, fontsize=7.6)
    for tick, tok in zip(ax.get_yticklabels(), agg.tokenizer):
        tick.set_color(S.TOKENIZER_COLOR[tok])
    ax.set_xlabel("HR@10 (mean of the two paradigms)")
    ax.set_ylim(-0.9, len(agg) - 0.1)
    ax.grid(axis="y", visible=False)
    S.despine(ax, keep=("bottom",))

    handles = [
        plt.Line2D([], [], marker="o", ls="", markersize=7.5,
                   markerfacecolor=S.MUTED, markeredgecolor=S.SURFACE,
                   label="Reported"),
        plt.Line2D([], [], marker="o", ls="", markersize=7,
                   markerfacecolor=S.SURFACE, markeredgecolor=S.MUTED,
                   markeredgewidth=1.8,
                   label="Estimated at the collision floor"),
    ]
    handles += [plt.Line2D([], [], marker="s", ls="", markersize=7,
                           markerfacecolor=S.TOKENIZER_COLOR[t],
                           markeredgecolor=S.SURFACE, label=S.tok_label(t))
                for t in S.TOKENIZER_ORDER]
    ax.legend(handles=handles, loc="lower right", ncol=1)

    spread_before = agg.reported.max() - agg.reported.min()
    spread_after = agg.adjusted.max() - agg.adjusted.min()
    ax.annotate(f"spread across configurations\n"
                f"{spread_before:.1f} pts reported  ->  {spread_after:.1f} pts adjusted",
                xy=(0.015, 0.965), xycoords="axes fraction", ha="left", va="top",
                fontsize=8, color=S.INK,
                bbox=dict(boxstyle="round,pad=0.45", fc=S.SURFACE, ec=S.GRID,
                          lw=0.8))

    S.subtitle(fig,
               "Discounting collisions compresses the gap between configurations",
               "Open markers estimate each configuration's HR@10 if it addressed items "
               f"as uniquely as the best case ({S.FLOOR_PCT:.2f}%), using the fitted "
               f"{slope:+.2f} pts/decade slope.")
    fig.tight_layout(rect=[0, 0.055, 1, 0.875])
    S.footnote(fig, "Model-based estimate, not a re-measurement: it assumes the "
                    "fitted linear relationship holds down to the floor. Treat the "
                    "direction and the compression as the finding, not the exact values.")
    S.save(fig, "fig05_adjusted_ranking")

    print(f"\n  slope = {slope:+.3f} HR@10 pts per decade of collision rate")
    print(f"  reported spread {spread_before:.2f} -> adjusted {spread_after:.2f}")
    print(agg.sort_values("adjusted", ascending=False)
            .head(5)[["config", "reported", "adjusted"]].to_string(index=False))


if __name__ == "__main__":
    main()
