"""Shared figure style for the collision-rate analysis.

Palette is the validated reference instance from the dataviz skill, light mode.
Only the first three categorical slots are used (blue / orange / aqua): that
subset is the documented all-pairs-safe prefix, which matters because most of
these figures are scatter plots where every pair of series sits side by side.

Tokenizer identity is carried by hue, model family by marker shape, and SID
depth by marker size -- so no distinction ever rests on colour alone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "diagrams" / "data"
OUTDIR = REPO / "diagrams"

# --- palette -------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TOKENIZER_COLOR = {
    "RQ-VAE": "#2a78d6",     # slot 1, blue
    "RQ-KMeans": "#eb6834",  # slot 2, orange
    "MQ": "#1baf7a",         # slot 3, aqua
}
TOKENIZER_ORDER = ["RQ-VAE", "RQ-KMeans", "MQ"]
TOKENIZER_LABEL = {"MQ": "MQ (parallel SID)"}

FAMILY_MARKER = {"Autoregressive": "o", "Masked diffusion": "^"}
FAMILY_ORDER = ["Autoregressive", "Masked diffusion"]
DEPTH_SIZE = {3: 42, 4: 74, 5: 118}

CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

# The 11-item floor: 11/3105 items on Industrial share a semantic ID under
# every tokenizer, so this is the irreducible rate, not a real "zero".
FLOOR_PCT = 0.3543


def tok_label(t):
    return TOKENIZER_LABEL.get(t, t)


def apply_base_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "figure.dpi": 160,
    })


def despine(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def subtitle(fig, title, sub, y=0.985):
    """Bold headline over a lighter explanatory line, both left-aligned.

    The sub-line is wrapped to the figure width so it can never run off the
    canvas as the caption text grows.
    """
    import textwrap
    fig.text(0.012, y, title, ha="left", va="top",
             fontsize=13, fontweight="bold", color=INK)
    # ~2.05 characters per point of figure width at 8.8pt in this face.
    width = max(60, int(fig.get_size_inches()[0] * 13.2))
    fig.text(0.012, y - 0.042, textwrap.fill(sub, width), ha="left", va="top",
             fontsize=8.8, color=INK_2, linespacing=1.45)


def footnote(fig, text, y=0.012):
    import textwrap
    width = max(70, int(fig.get_size_inches()[0] * 18))
    fig.text(0.012, y, textwrap.fill(text, width), ha="left", va="bottom",
             fontsize=7.4, color=MUTED, linespacing=1.5)


def save(fig, name):
    check_layout(fig, name)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote diagrams/{name}.png / .pdf")


def check_layout(fig, name=""):
    """Programmatic stand-in for the eyeball pass.

    Reports text that overflows the canvas and pairs of text artists that
    overlap each other -- the two layout faults that actually show up in
    rendered matplotlib output.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.canvas.get_width_height()

    items = []
    for ax in fig.axes:
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t.get_text().strip():
                items.append((t, ax))
    for t in fig.texts:
        if t.get_text().strip():
            items.append((t, None))

    problems = []
    boxes = []
    for t, ax in items:
        try:
            bb = t.get_window_extent(renderer=rend)
        except Exception:
            continue
        label = t.get_text().replace("\n", " ")[:38]
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fw + 1 or bb.y1 > fh + 1:
            problems.append(f"OVERFLOW  {label!r}")
        boxes.append((label, bb, ax))

    for i in range(len(boxes)):
        for k in range(i + 1, len(boxes)):
            (la, ba, axa), (lb, bb2, axb) = boxes[i], boxes[k]
            if axa is not axb:
                continue  # cross-axes near-misses are usually just padding
            if ba.overlaps(bb2):
                problems.append(f"OVERLAP   {la!r} <-> {lb!r}")

    tag = name or "figure"
    if problems:
        print(f"  [layout] {tag}: {len(problems)} issue(s)")
        for p in problems:
            print(f"      {p}")
    else:
        print(f"  [layout] {tag}: clean")
    return problems
