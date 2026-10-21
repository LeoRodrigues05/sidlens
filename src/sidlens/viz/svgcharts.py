"""Inline-SVG charts for self-contained HTML reports (no JavaScript library, no network).

Colours are CSS variables defined by the page (`--q-rqkmeans`, `--q-rqvae`,
`--q-MQ`, ink and grid tokens), so one chart serves light and dark themes.
Quantizer identity follows `viz/style.py` (blue / orange / aqua, validated
for CVD as the first three categorical slots). Every mark carries `data-tip`
text for the page's tooltip script, and every chart returns its table view.

Traps, each with the guard that closes it
-----------------------------------------
1. **Identity by colour alone.** Every multi-series chart gets a legend, and
   width (128 vs 512) is encoded by fill (open vs filled), not by a new hue.
2. **Labels that are data.** Every string is escaped with `html.escape`.
3. **A scale that hides zero.** Interval charts always include 0 in the domain
   when a value crosses or touches it, and draw the zero line.
"""

from __future__ import annotations

import html
import math

QUANTIZERS = ("rqkmeans", "rqvae", "MQ")
Q_LABEL = {"rqkmeans": "RQ-KMeans", "rqvae": "RQ-VAE", "MQ": "MQ"}
FONT = 12


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=mag * 10)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v: float, pct: bool = False, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if pct:
        return f"{v * 100:+.1f}" if signed else f"{v * 100:.1f}"
    a = abs(v)
    s = f"{v:+.2f}" if signed else f"{v:.2f}"
    if a < 0.1 and a > 0:
        s = f"{v:+.3f}" if signed else f"{v:.3f}"
    return s


def _domain(values, include_zero=True, pad=0.06):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    if include_zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        hi = lo + 1
    p = (hi - lo) * pad
    return lo - p, hi + p


def legend(items: list[tuple[str, str]], width_key: bool = False) -> str:
    """items: (css colour var, label). width_key adds the open/filled key."""
    parts = ['<div class="legend">']
    for var, label in items:
        parts.append(f'<span class="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" '
                     f'fill="var({var})"/></svg>{esc(label)}</span>')
    if width_key:
        parts.append('<span class="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" '
                     'fill="var(--surface)" stroke="var(--ink-2)" stroke-width="2"/></svg>width 128</span>')
        parts.append('<span class="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" '
                     'fill="var(--ink-2)"/></svg>width 512</span>')
    parts.append("</div>")
    return "".join(parts)


def dumbbell(rows: list[dict], *, title: str, x_label: str, pct: bool = False, signed: bool = False,
             ref: float | None = None, ref_label: str = "", include_zero: bool = True, label_w: int = 150,
             width: int = 560) -> str:
    """One row per quantizer x depth; open dot = width 128, filled = 512, with 95 % intervals.

    rows: {"quantizer", "depth", "w128": (est, lo, hi) | None, "w512": ...}. Values in data units.
    """
    row_h, top, bottom = 22, 8, 34
    h = top + bottom + row_h * len(rows)
    vals = [v for r in rows for k in ("w128", "w512") if r.get(k) for v in r[k]]
    if ref is not None:
        vals.append(ref)
    lo, hi = _domain(vals, include_zero=include_zero)
    x0, x1 = label_w, width - 16

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    s = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    for t in _ticks(lo, hi):
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - bottom}" class="grid"/>')
        s.append(f'<text x="{X(t):.1f}" y="{h - bottom + 14}" class="tick" text-anchor="middle">{esc(_fmt(t, pct))}</text>')
    if lo < 0 < hi:
        s.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top}" y2="{h - bottom}" class="zero"/>')
    if ref is not None:
        s.append(f'<line x1="{X(ref):.1f}" x2="{X(ref):.1f}" y1="{top}" y2="{h - bottom}" class="ref"/>'
                 f'<text x="{X(ref) + 3:.1f}" y="{top + 9}" class="tick">{esc(ref_label)}</text>')
    s.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{h - 4}" class="axis-label" text-anchor="middle">{esc(x_label)}</text>')
    prev_q = None
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        q = r["quantizer"]
        if prev_q is not None and q != prev_q:
            s.append(f'<line x1="4" x2="{width - 4}" y1="{y - row_h / 2:.1f}" y2="{y - row_h / 2:.1f}" class="sep"/>')
        prev_q = q
        s.append(f'<text x="{label_w - 8}" y="{y + 4:.1f}" class="row-label" text-anchor="end">'
                 f'{esc(Q_LABEL.get(q, q))} · depth {r["depth"]}</text>')
        pts = [(k, r[k]) for k in ("w128", "w512") if r.get(k) and not math.isnan(r[k][0])]
        if len(pts) == 2:
            s.append(f'<line x1="{X(pts[0][1][0]):.1f}" x2="{X(pts[1][1][0]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="link"/>')
        for k, (est, elo, ehi) in pts:
            dy = -3 if k == "w128" else 3
            yy = y + dy
            tip = f"{Q_LABEL.get(q, q)} {r['depth']}×{k[1:]}: {_fmt(est, pct, signed)}"
            if elo is not None and not math.isnan(elo):
                tip += f" [{_fmt(elo, pct, signed)}, {_fmt(ehi, pct, signed)}]"
                s.append(f'<line x1="{X(elo):.1f}" x2="{X(ehi):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                         f'stroke="var(--q-{q})" stroke-width="1.5" opacity="0.7"/>')
            fill = f"var(--q-{q})" if k == "w512" else "var(--surface)"
            s.append(f'<circle cx="{X(est):.1f}" cy="{yy:.1f}" r="4.5" fill="{fill}" stroke="var(--q-{q})" '
                     f'stroke-width="2" class="mark" tabindex="0" data-tip="{esc(tip)}"/>')
    s.append("</svg>")
    return f'<figure><figcaption>{esc(title)}</figcaption>{"".join(s)}</figure>'


def profiles(panels: list[dict], *, title: str, x_labels: list[str], y_label: str, pct: bool = False,
             y_domain: tuple[float, float] | None = None, panel_w: int = 200, panel_h: int = 150) -> str:
    """Small multiples. panels: {"title", "series": [{"quantizer", "label", "dashed", "points": [(x_index, est, lo, hi)]}]}."""
    cols = len(panels)
    left, top, gap, bottom = 58, 22, 14, 30
    W = left + cols * panel_w + (cols - 1) * gap + 8
    H = top + panel_h + bottom
    vals = [v for p in panels for se in p["series"] for pt in se["points"] for v in pt[1:] if v is not None]
    lo, hi = y_domain or _domain(vals, include_zero=True)
    nx = len(x_labels)
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(title)}">']
    for c, p in enumerate(panels):
        px = left + c * (panel_w + gap)

        def X(i):
            return px + 10 + i / max(nx - 1, 1) * (panel_w - 20)

        def Y(v):
            return top + panel_h - (v - lo) / (hi - lo) * panel_h

        s.append(f'<text x="{px + panel_w / 2:.1f}" y="{top - 8}" class="panel-title" text-anchor="middle">{esc(p["title"])}</text>')
        for t in _ticks(lo, hi, 4):
            s.append(f'<line x1="{px}" x2="{px + panel_w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
            if c == 0:
                s.append(f'<text x="{px - 5}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{esc(_fmt(t, pct))}</text>')
        if lo < 0 < hi:
            s.append(f'<line x1="{px}" x2="{px + panel_w}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" class="zero"/>')
        for i, xl in enumerate(x_labels):
            s.append(f'<text x="{X(i):.1f}" y="{top + panel_h + 14}" class="tick" text-anchor="middle">{esc(xl)}</text>')
        for se in p["series"]:
            q = se["quantizer"]
            pts = [(i, e, a, b) for i, e, a, b in se["points"] if e is not None and not math.isnan(e)]
            if not pts:
                continue
            d = " ".join(f"{'M' if j == 0 else 'L'}{X(i):.1f},{Y(e):.1f}" for j, (i, e, _, _) in enumerate(pts))
            dash = ' stroke-dasharray="5 3"' if se.get("dashed") else ""
            s.append(f'<path d="{d}" fill="none" stroke="var(--q-{q})" stroke-width="2" stroke-linejoin="round"{dash}/>')
            for i, e, a, b in pts:
                tip = f"{se['label']}, {x_labels[i]}: {_fmt(e, pct)}"
                if a is not None and not math.isnan(a):
                    tip += f" [{_fmt(a, pct)}, {_fmt(b, pct)}]"
                fill = "var(--surface)" if se.get("dashed") else f"var(--q-{q})"
                s.append(f'<circle cx="{X(i):.1f}" cy="{Y(e):.1f}" r="4" fill="{fill}" stroke="var(--q-{q})" stroke-width="2" '
                         f'class="mark" tabindex="0" data-tip="{esc(tip)}"/>')
    s.append(f'<text x="12" y="{top + panel_h / 2:.1f}" class="axis-label" transform="rotate(-90 12 {top + panel_h / 2:.1f})" '
             f'text-anchor="middle">{esc(y_label)}</text>')
    s.append("</svg>")
    return f'<figure><figcaption>{esc(title)}</figcaption>{"".join(s)}</figure>'


def hbars(items: list[dict], *, title: str, x_label: str, pct: bool = False, label_w: int = 190, width: int = 560,
          signed: bool = False) -> str:
    """Horizontal bars with optional intervals. items: {"label", "est", "lo", "hi", "var"} (var = css colour var)."""
    row_h, top, bottom = 24, 8, 34
    h = top + bottom + row_h * len(items)
    vals = [v for it in items for v in (it["est"], it.get("lo"), it.get("hi")) if v is not None]
    lo, hi = _domain(vals)
    x0, x1 = label_w, width - 16

    def X(v):
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)

    s = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{esc(title)}">']
    for t in _ticks(lo, hi):
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{h - bottom}" class="grid"/>')
        s.append(f'<text x="{X(t):.1f}" y="{h - bottom + 14}" class="tick" text-anchor="middle">{esc(_fmt(t, pct))}</text>')
    s.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{h - 4}" class="axis-label" text-anchor="middle">{esc(x_label)}</text>')
    for i, it in enumerate(items):
        y = top + row_h * i
        a, b = sorted((X(0), X(it["est"])))
        tip = f"{it['label']}: {_fmt(it['est'], pct, signed)}"
        if it.get("lo") is not None:
            tip += f" [{_fmt(it['lo'], pct, signed)}, {_fmt(it['hi'], pct, signed)}]"
        s.append(f'<text x="{label_w - 8}" y="{y + 16}" class="row-label" text-anchor="end">{esc(it["label"])}</text>')
        s.append(f'<rect x="{a:.1f}" y="{y + 6}" width="{max(b - a, 1):.1f}" height="12" rx="3" fill="var({it.get("var", "--q-rqkmeans")})" '
                 f'class="mark" tabindex="0" data-tip="{esc(tip)}"/>')
        if it.get("lo") is not None:
            s.append(f'<line x1="{X(it["lo"]):.1f}" x2="{X(it["hi"]):.1f}" y1="{y + 12}" y2="{y + 12}" class="whisker"/>')
    s.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{top}" y2="{h - bottom}" class="zero"/>')
    s.append("</svg>")
    return f'<figure><figcaption>{esc(title)}</figcaption>{"".join(s)}</figure>'


def table(columns: list[str], rows: list[list], *, caption: str = "", numeric_from: int = 1) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if j >= numeric_from else ""}">{esc(v)}</td>' for j, v in enumerate(r)) + "</tr>" for r in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="table-wrap"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
