"""Fig 1 - how collision rate varies across the tokenizer design space."""

import matplotlib.pyplot as plt
import pandas as pd

import style as S


def main():
    S.apply_base_style()
    df = pd.read_csv(S.DATA / "collisions.csv")

    datasets = ["Industrial", "Office"]
    codebooks = [128, 256, 512]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.1), sharey=True, sharex=True)

    for i, ds in enumerate(datasets):
        for j, cbk in enumerate(codebooks):
            ax = axes[i, j]
            sub = df[(df.dataset == ds) & (df.codebook == cbk)]
            for tok in S.TOKENIZER_ORDER:
                g = sub[sub.tokenizer == tok].sort_values("depth")
                ax.plot(g.depth, g.collision_pct, marker="o", markersize=6,
                        color=S.TOKENIZER_COLOR[tok],
                        markeredgecolor=S.SURFACE, markeredgewidth=2,
                        label=S.tok_label(tok) if (i == 0 and j == 0) else None)
            ax.axhline(S.FLOOR_PCT, color=S.MUTED, lw=0.8, ls=(0, (1, 2)))
            ax.set_yscale("log")
            ax.set_xticks([3, 4, 5])
            ax.set_title(f"{ds}  |  codebook {cbk}", fontsize=9)
            S.despine(ax)
            if j == 0:
                ax.set_ylabel("Collision rate (%, log)")
            if i == 1:
                ax.set_xlabel("SID depth (digits)")

    # Direct labels on the top-left panel carry identity without the legend.
    ax0 = axes[0, 0]
    for tok, y in [("MQ", 40.1), ("RQ-KMeans", 29.3), ("RQ-VAE", 1.16)]:
        ax0.annotate(S.tok_label(tok), xy=(3, y), xytext=(3.06, y * 1.45),
                     color=S.TOKENIZER_COLOR[tok], fontsize=7.6,
                     fontweight="bold")
    axes[0, 2].annotate("irreducible floor\n(duplicate catalogue items)",
                        xy=(4, S.FLOOR_PCT), xytext=(3.35, 0.9),
                        fontsize=7, color=S.MUTED,
                        arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=0.7))

    fig.legend(loc="upper right", bbox_to_anchor=(0.995, 0.965), ncol=3)
    S.subtitle(fig,
               "Collision rate spans two orders of magnitude across tokenizers",
               "Share of catalogue items that do not receive a unique semantic ID. "
               "Deeper SIDs and larger codebooks reduce collisions, but the choice of "
               "quantiser dominates both.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    S.footnote(fig, "Source: data/CollisionRates_GenRec_Paradigms.xlsx  |  "
                    "Industrial n=3,105 items, Office n=17,696 items.")
    S.save(fig, "fig01_collision_landscape")


if __name__ == "__main__":
    main()
