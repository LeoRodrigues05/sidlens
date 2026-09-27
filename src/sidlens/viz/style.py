"""One visual system for every SidLens figure.

Colour is assigned by the job it does, never by taste, and every categorical
palette below was run through a CVD/contrast validator before it was written
here (see docs/results/semantic_mapping/SEMANTIC_MAPPING_TAKEAWAYS.md for the run). The rules that
matter, because they are the ones a hurried figure breaks:

  * A quantizer keeps its colour in every figure. RQ-KMeans is blue, RQ-VAE is
    orange, MQ is aqua, whether or not the other two are on the same axes.
    Colour follows the entity, not the row it happens to occupy in one plot.
  * Ordered factors (codebook width 128<256<512, prefix depth 1..5) take a
    one-hue lightness ramp, light = small, so the order is visible in the ink.
    They never take categorical hues -- three unrelated hues would make the
    reader memorise which is which.
  * Text never wears a series colour. Labels, ticks and legends stay in ink
    tokens; the coloured mark beside them carries identity.
  * Gridlines are hairlines one step off the surface, solid. A null or
    reference line is drawn in the muted ink with its own label, so the reader
    never has to guess what a grey line means.

Figures are saved twice: PDF (vector, for the paper) and PNG at 200 dpi (for
docs and slides). Both share the same rcParams so they cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- ink and chrome (light surface) -----------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NULL_BAND = "#ececea"       # permutation / random-subdivision null envelope
OTHER = "#c3c2b7"           # the "everything else" fold, never a series hue

# --- identity: one fixed hue per quantizer -----------------------------------
# Slots 1-3 of the reference palette validate all-pairs (any two can touch).
QUANTIZER_COLOR = {
    "rqkmeans": "#2a78d6",  # blue
    "rqvae":    "#eb6834",  # orange
    "MQ":       "#1baf7a",  # aqua
}
QUANTIZER_LABEL = {"rqkmeans": "RQ-KMeans", "rqvae": "RQ-VAE", "MQ": "MQ"}
QUANTIZER_ORDER = ("rqkmeans", "rqvae", "MQ")

# --- identity: categorical slots for up to 8 classes (adjacent-validated) ----
CATEGORY_SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# --- order: one-hue ramps, light = small -------------------------------------
WIDTH_RAMP = {128: "#86b6ef", 256: "#3987e5", 512: "#1c5cab"}
DEPTH_RAMP = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#1c5cab", 5: "#0d366b"}
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]

# --- mark specs ---------------------------------------------------------------
LINE_W = 1.6
MARKER = 4.0          # radius >= 4 px at 200 dpi is the floor; paper figures are small
RING = SURFACE        # the 2 px surface ring around overlapping markers


def _font_family() -> list[str]:
    """Prefer a Helvetica-metric sans if the node has one; DejaVu otherwise."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Nimbus Sans", "Helvetica", "Arial", "Liberation Sans"):
        if name in have:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def apply() -> None:
    """Install the rcParams. Idempotent; call once per process before plotting."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": _font_family(),
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "medium",
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "xtick.major.width": 0.6,
        "lines.linewidth": LINE_W,
        "lines.solid_joinstyle": "round",
        "lines.solid_capstyle": "round",
        "lines.markersize": MARKER,
        "lines.markeredgewidth": 0.8,
        "lines.markeredgecolor": RING,
        "pdf.fonttype": 42,           # embed TrueType so text stays editable
        "ps.fonttype": 42,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save(fig, stem: Path | str) -> list[Path]:
    """Write <stem>.pdf and <stem>.png and return both paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    for p in out:
        fig.savefig(p)
    plt.close(fig)
    return out


def null_line(ax, y, label: str, xmin=None, xmax=None):
    """A reference/null level in muted ink with a right-edge label."""
    ax.axhline(y, color=MUTED, linewidth=0.9, zorder=1)
    x = ax.get_xlim()[1] if xmax is None else xmax
    ax.annotate(label, (x, y), xytext=(2, 2), textcoords="offset points",
                ha="right", va="bottom", fontsize=6.5, color=MUTED)


def legend_outside(ax, **kw):
    """Legend to the right of the axes, so it never covers a mark."""
    return ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     borderaxespad=0.0, **kw)


def panel_label(ax, text: str):
    """Bold panel letter, top-left outside the axes, as journals expect."""
    ax.text(-0.02, 1.06, text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="right", va="bottom", color=INK)
