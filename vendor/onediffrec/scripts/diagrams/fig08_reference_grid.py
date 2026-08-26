"""Fig 8 - the full grid as an annotated reference, collisions beside scores.

This is the table-view twin for the scatter figures: every value that fig 2 and
fig 3 encode positionally is printed here, so nothing depends on reading a
colour or a position off an axis.
"""

import numpy as np
import pandas as pd

import style as S
from fig02_collision_vs_performance import load_core


def main():
    S.apply_base_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    d = load_core()
    d = d.assign(fam=np.where(d.family == "Autoregressive", "AR", "Diff"))

    rows = [(t, k) for t in S.TOKENIZER_ORDER for k in (128, 256, 512)]
    cols = [(dp, f) for dp in (3, 4, 5) for f in ("AR", "Diff")]

    hr = np.full((len(rows), len(cols)), np.nan)
    coll = np.full(len(rows), np.nan)
    for i, (t, k) in enumerate(rows):
        for j, (dp, f) in enumerate(cols):
            sel = d[(d.tokenizer == t) & (d.codebook == k)
                    & (d.depth == dp) & (d.fam == f)]
            if len(sel):
                hr[i, j] = sel["HR@10"].iloc[0]
        c = d[(d.tokenizer == t) & (d.codebook == k)]
        if len(c):
            coll[i] = c.collision_pct.iloc[0]

    fig, (axc, ax) = plt.subplots(
        1, 2, figsize=(10.0, 5.4), sharey=True,
        gridspec_kw=dict(width_ratios=[1, 6], wspace=0.06))

    # --- left strip: collision rate, sequential single-hue ramp ----------
    axc.imshow(coll[:, None], cmap="Blues", aspect="auto",
               norm=LogNorm(vmin=0.3, vmax=45))
    for i, v in enumerate(coll):
        axc.text(0, i, f"{v:.2f}%", ha="center", va="center", fontsize=7.8,
                 color=S.SURFACE if v > 8 else S.INK, fontweight="bold")
    axc.set_xticks([0]); axc.set_xticklabels(["collisions"], fontsize=8)
    axc.set_title("Collision rate", fontsize=9)
    axc.tick_params(length=0); axc.grid(False)
    for sp in axc.spines.values():
        sp.set_visible(False)

    # --- right: HR@10, its own sequential ramp --------------------------
    im = ax.imshow(hr, cmap="Oranges", aspect="auto",
                   norm=Normalize(vmin=np.nanmin(hr), vmax=np.nanmax(hr)))
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(hr[i, j]):
                continue
            hot = hr[i, j] > np.nanpercentile(hr, 68)
            ax.text(j, i, f"{hr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.8, color=S.SURFACE if hot else S.INK)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{dp}d\n{f}" for dp, f in cols], fontsize=7.8)
    ax.set_title("Reported HR@10", fontsize=9)
    ax.tick_params(length=0); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for j in range(1, len(cols)):
        if j % 2 == 0:
            ax.axvline(j - 0.5, color=S.SURFACE, lw=3)

    axc.set_yticks(range(len(rows)))
    axc.set_yticklabels([f"{S.tok_label(t)}  x{k}" for t, k in rows], fontsize=7.8)
    for tick, (t, _) in zip(axc.get_yticklabels(), rows):
        tick.set_color(S.TOKENIZER_COLOR[t])
    for i in range(1, len(rows)):
        if i % 3 == 0:
            for a in (axc, ax):
                a.axhline(i - 0.5, color=S.SURFACE, lw=3)

    # Colour bar gets an explicit axes: a colorbar attached via ax= is not
    # compatible with tight_layout, which this figure needs for the header.
    cax = fig.add_axes([0.925, 0.14, 0.014, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("HR@10", fontsize=8, color=S.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=0, colors=S.MUTED)

    S.subtitle(fig,
               "Full grid: the two ramps darken together",
               "Rows are tokenizer x codebook, columns are SID depth x paradigm. "
               "The darkest HR@10 cells sit in the rows with the darkest collision "
               "rates, at the top-left of each tokenizer block.")
    fig.subplots_adjust(left=0.155, right=0.90, top=0.80, bottom=0.135)
    S.footnote(fig, "Two separate single-hue sequential ramps, one per measure -- "
                    "they are not on a shared scale and are placed side by side only "
                    "so the row ordering can be compared. Every plotted value is "
                    "printed; diagrams/data/joined.csv holds the full table.")
    S.save(fig, "fig08_reference_grid")


if __name__ == "__main__":
    main()
