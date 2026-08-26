"""Fig 7 - backbone scaling, at the one cell where several backbones were run.

Only the codebook-256 / 3-digit cells carry more than the two core models, so
this is the only place the grid can speak to backbone choice. The comparison is
within a fixed tokenizer, so collisions are constant down each panel.
"""

import numpy as np
import pandas as pd

import style as S

# Ordered smallest to largest so the x-axis reads as a capacity axis.
ORDER = ["Qwen2.5-1.5B", "Qwen2.5-1.5B-grpo", "Qwen3-1.7B", "Qwen3.5-2B",
         "Qwen2.5-3B", "DiffGRM", "Mask Diffusion"]


def main():
    S.apply_base_style()
    import matplotlib.pyplot as plt

    m = pd.read_csv(S.DATA / "joined.csv")
    d = m[(m.codebook == 256) & (m.depth == 3)].copy()
    d["order"] = d.model.map({k: i for i, k in enumerate(ORDER)})
    d = d.sort_values(["tokenizer", "order"])

    toks = [t for t in S.TOKENIZER_ORDER if t in set(d.tokenizer)]
    fig, axes = plt.subplots(1, len(toks), figsize=(10.4, 4.6), sharey=True)
    if len(toks) == 1:
        axes = [axes]

    for ax, tok in zip(axes, toks):
        g = d[d.tokenizer == tok]
        x = np.arange(len(g))
        colour = S.TOKENIZER_COLOR[tok]
        # One series, one colour; the diffusion model is set apart by hatch,
        # not by a second hue, because it is a paradigm not a category.
        colours = [colour] * len(g)
        hatches = ["///" if f == "Masked diffusion" else "" for f in g.family]

        bars = ax.bar(x, g["HR@10"], width=0.62, color=colours,
                      edgecolor=S.SURFACE, linewidth=2)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
                b.set_edgecolor(S.SURFACE)

        for xi, v in zip(x, g["HR@10"]):
            ax.annotate(f"{v:.2f}", xy=(xi, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7.6,
                        color=S.INK)

        ax.set_xticks(x)
        ax.set_xticklabels(g.model, rotation=32, ha="right", fontsize=7.4)
        coll = g.collision_pct.iloc[0]
        ax.set_title(f"{S.tok_label(tok)}   ({coll:.2f}% collisions)", color=colour)
        ax.grid(axis="x", visible=False)
        S.despine(ax)

    axes[0].set_ylabel("HR@10")
    axes[0].set_ylim(0, d["HR@10"].max() * 1.22)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=S.MUTED, edgecolor=S.SURFACE,
                      label="Autoregressive"),
        plt.Rectangle((0, 0), 1, 1, facecolor=S.MUTED, edgecolor=S.SURFACE,
                      hatch="///", label="Masked diffusion"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.005))

    S.subtitle(fig,
               "Backbone choice moves the score far less than the tokenizer does",
               "Codebook 256, 3-digit SIDs -- the only cell with more than two models. "
               "Within a panel the SIDs are identical, so these differences are real "
               "model differences.")
    fig.tight_layout(rect=[0, 0.075, 1, 0.865])
    S.footnote(fig, "Scaling 1.5B -> 3B buys +0.8 HR@10 on RQ-VAE and +1.7 on "
                    "RQ-KMeans; switching RQ-VAE for MQ buys +5.5 at the same "
                    "backbone. Architecture beats parameter count -- Qwen3-1.7B is "
                    "the strongest RQ-VAE backbone, ahead of the 3B. Empty workbook "
                    "cells are omitted.", y=0.045)
    S.save(fig, "fig07_backbone_sweep")

    print()
    print(d[["tokenizer", "model", "collision_pct", "HR@10", "NDCG@10"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
