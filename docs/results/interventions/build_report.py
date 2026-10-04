#!/usr/bin/env python
"""Build the SidLens intervention report (one self-contained HTML page).

Every number drawn or quoted in a chart is read from an experiment summary's
`estimates.csv` (or its per-row tables), never typed in by hand; the few
fixed facts in the prose (hash matches, audit counts) are cited from the
experiments' validation records. Charts are inline SVG with a hover layer and
a table-view twin; colors are theme tokens (quantizer hues fixed as in
`sidlens.viz.style`, validated with the dataviz six-checks script).

    python docs/results/interventions/build_report.py --out docs/results/interventions/sidlens_interventions.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from sidlens import paths                                                # noqa: E402

D = paths.DERIVED
S = {
    "exp3": D / "controlled/exp3_ar_history_patching/summary-280961",
    "exp4": D / "controlled/exp4_ar_copy_circuit/summary-281261",
    "exp4t": D / "controlled/exp4_ar_copy_circuit/summary-281394",
    "exp4cells": D / "controlled/exp4_ar_copy_circuit/281261",
    "exp5": D / "controlled/exp5_ar_next_two_conditioning/summary-281263",
    "exp6": None,                                                        # set in main()
    "exp7": D / "controlled/exp7_diffusion_history_use/summary-283124-r2",
    "exp8": D / "controlled/exp8_diffusion_copy_route/summary-283159",
    "rep": D / "representation/exp1_ar_digit_decoding/summary-281264-281411",
}
Q = {"rqkmeans_3codebook_128": ("rqk", "RQ-KMeans 3×128"), "rqvae_4codebook_128": ("rqvae", "RQ-VAE 4×128")}
esc = html.escape


def est(path: Path) -> pd.DataFrame:
    return pd.read_csv(path / "estimates.csv")


def f2(x, d=2, sign=False):
    return (f"{x:+.{d}f}" if sign else f"{x:.{d}f}").replace("-", "−")


# ------------------------------------------------------------------ charts --
class Scale:
    def __init__(self, d0, d1, r0, r1):
        self.d0, self.d1, self.r0, self.r1 = d0, d1, r0, r1

    def __call__(self, v):
        return self.r0 + (v - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(round(t, 10))
        t += step
    return out


def fmt_tick(v):
    if abs(v) < 1e-12:
        return "0"
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


def line_panel(title, series, xdom, ydom, xlab, ylab, xticks=None, yticks=None, w=460, h=280,
               end_labels=True, band=True, ref_lines=(), xlabels=None):
    """series: list of dict(name, cls, pts=[(x, y, lo, hi)], dash=False, tipfmt).

    xlabels maps tick values to category names, for an ordered categorical axis."""
    ml, mr, mt, mb = 48, 108 if end_labels else 16, 14, 42
    X = Scale(xdom[0], xdom[1], ml, w - mr)
    Y = Scale(ydom[0], ydom[1], h - mb, mt)
    xticks = xticks or nice_ticks(*xdom)
    yticks = yticks or nice_ticks(*ydom)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    for t in yticks:
        y = Y(t)
        g.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        g.append(f'<text class="tick" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt_tick(t)}</text>')
    for v, lab in ref_lines:
        y = Y(v)
        g.append(f'<line class="ref" x1="{ml}" x2="{w - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        g.append(f'<text class="tick" x="{w - mr + 4}" y="{y + 4:.1f}">{esc(lab)}</text>')
    g.append(f'<line class="axis" x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}"/>')
    for t in xticks:
        x = X(t)
        lab = esc(xlabels[t]) if xlabels and t in xlabels else fmt_tick(t)
        g.append(f'<text class="tick" x="{x:.1f}" y="{h - mb + 16}" text-anchor="middle">{lab}</text>')
    g.append(f'<text class="axlab" x="{(ml + w - mr) / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlab)}</text>')
    g.append(f'<text class="axlab" x="12" y="{mt + (h - mb - mt) / 2:.1f}" text-anchor="middle" '
             f'transform="rotate(-90 12 {mt + (h - mb - mt) / 2:.1f})">{esc(ylab)}</text>')
    labels = []
    for s in series:
        pts = [p for p in s["pts"] if p[1] is not None and not np.isnan(p[1])]
        if not pts:
            continue
        if band and all(p[2] is not None and not np.isnan(p[2]) for p in pts):
            up = " ".join(f"{X(p[0]):.1f},{Y(min(max(p[3], ydom[0]), ydom[1])):.1f}" for p in pts)
            dn = " ".join(f"{X(p[0]):.1f},{Y(min(max(p[2], ydom[0]), ydom[1])):.1f}" for p in reversed(pts))
            g.append(f'<polygon class="band {s["cls"]}" points="{up} {dn}"/>')
        path = " ".join(f"{'M' if i == 0 else 'L'}{X(p[0]):.1f},{Y(p[1]):.1f}" for i, p in enumerate(pts))
        g.append(f'<path class="ln {s["cls"]}{" dash" if s.get("dash") else ""}" d="{path}"/>')
        for p in pts:
            tip = s["tipfmt"](p)
            g.append(f'<circle class="hit" cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="11" data-tip="{esc(tip)}"/>')
        last = pts[-1]
        g.append(f'<circle class="dot {s["cls"]}" cx="{X(last[0]):.1f}" cy="{Y(last[1]):.1f}" r="4"/>')
        labels.append([Y(last[1]), s["short"], s["cls"]])
    if end_labels and labels:
        labels.sort()
        for i in range(1, len(labels)):                      # keep end labels 14px apart
            labels[i][0] = max(labels[i][0], labels[i - 1][0] + 14)
        floor = h - mb - 10                                  # and clear of the last x tick below them
        if labels[-1][0] > floor:
            labels[-1][0] = floor
            for i in range(len(labels) - 2, -1, -1):
                labels[i][0] = min(labels[i][0], labels[i + 1][0] - 14)
        for y, txt, cls in labels:
            g.append(f'<text class="endlab" x="{w - mr + 8}" y="{y + 4:.1f}">{esc(txt)}</text>')
    g.append("</svg>")
    return "".join(g)


def forest(rows, series, xdom, xlab, note=""):
    """rows: list of (label, {series_key: (est, lo, hi, n)}); series: list of (key, cls, name)."""
    X = Scale(xdom[0], xdom[1], 2, 298)
    ticks = nice_ticks(*xdom, n=5)
    out = ['<div class="forest" role="table">']
    for label, vals in rows:
        svg = [f'<svg viewBox="0 0 300 {12 + 12 * len(series)}" class="strip" preserveAspectRatio="none">']
        for t in ticks:
            svg.append(f'<line class="{"zero" if abs(t) < 1e-12 else "grid"}" x1="{X(t):.1f}" x2="{X(t):.1f}" '
                       f'y1="0" y2="{12 + 12 * len(series)}"/>')
        svg.append("</svg>")
        marks = []
        for i, (k, cls, name) in enumerate(series):
            if k not in vals or vals[k] is None:
                continue
            e, lo, hi, n = vals[k]
            y = 12 + 12 * i
            l, r, c = (max(xdom[0], min(xdom[1], v)) for v in (lo, hi, e))
            tip = f"{f2(e, 3, True)} nats\n95% CI {f2(lo, 3, True)} to {f2(hi, 3, True)}\n{name} · {label} · n={n:,}"
            pl, pr, pc = ((X(v) - 2) / 296 * 100 for v in (l, r, c))
            marks.append(f'<span class="ci {cls}" style="left:{pl:.2f}%;width:{max(pr - pl, 0.3):.2f}%;top:{y - 1}px"></span>'
                         f'<span class="pt {cls}" style="left:{pc:.2f}%;top:{y - 5}px" tabindex="0" data-tip="{esc(tip)}"></span>')
        out.append(f'<div class="frow" role="row"><div class="flab" role="rowheader">{esc(label)}</div>'
                   f'<div class="fplot" role="cell" style="height:{12 + 12 * len(series)}px">{"".join(svg)}{"".join(marks)}</div></div>')
    axis = "".join(f'<span style="left:{(X(t) - 2) / 296 * 100:.2f}%">{fmt_tick(t)}</span>' for t in ticks)
    out.append(f'<div class="frow faxis"><div class="flab"></div><div class="fplot"><div class="ticks">{axis}</div>'
               f'<div class="axlab-h">{esc(xlab)}</div></div></div>')
    out.append("</div>")
    return "".join(out)


def heatmap(grid, marks, title, vmax):
    """grid [28 layers x 12 heads] of mean Δ; sequential blue from 0 to vmax; negative = neutral."""
    L, H = grid.shape
    cw, ch, ml, mt = 16, 11, 34, 22
    w, h = ml + H * cw + 8, mt + L * ch + 28
    g = [f'<svg viewBox="0 0 {w} {h}" class="chart heat" role="img" aria-label="{esc(title)}">']
    for L_ in range(L):
        y = mt + (L - 1 - L_) * ch
        if L_ % 3 == 0 or L_ == L - 1:
            g.append(f'<text class="tick" x="{ml - 6}" y="{y + 8.5:.1f}" text-anchor="end">{L_}</text>')
        for h_ in range(H):
            v = grid[L_, h_]
            cls = "cellneg" if v <= 0 else f"h{min(13, max(1, 1 + int(round(v / vmax * 12))))}"
            g.append(f'<rect class="cell {cls}" x="{ml + h_ * cw + 1}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" '
                     f'rx="1.5" data-tip="{esc(f"{f2(v, 4, True)} nats{chr(10)}layer {L_}, head {h_}")}"/>')
    for L_, h_ in marks:
        y = mt + (L - 1 - L_) * ch
        g.append(f'<rect class="sel" x="{ml + h_ * cw}" y="{y}" width="{cw}" height="{ch}" rx="2"/>')
    for h_ in range(0, H, 2):
        g.append(f'<text class="tick" x="{ml + h_ * cw + cw / 2:.1f}" y="{mt + L * ch + 14}" text-anchor="middle">{h_}</text>')
    g.append(f'<text class="axlab" x="{ml + H * cw / 2:.1f}" y="{h - 2}" text-anchor="middle">head</text>')
    g.append(f'<text class="axlab" x="{ml - 24}" y="{mt - 8}">layer</text>')
    g.append("</svg>")
    return "".join(g)


def table(cols, rows, num_from=1):
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f'<td class="{"n" if i >= num_from else ""}">{esc(str(v))}</td>'
                                   for i, v in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(fid, title, sub, body, legend, tbl, caption):
    leg = "".join(f'<span class="lg"><i class="key {cls} {kind}"></i>{esc(name)}</span>' for cls, name, kind in legend)
    # kind may be "line dash" for a dashed key
    return (f'<figure id="{fid}"><figcaption class="ft"><span class="fttl">{esc(title)}</span>'
            f'<span class="fsub">{esc(sub)}</span></figcaption>'
            f'{f"<div class=legend>{leg}</div>" if legend else ""}{body}'
            f'<p class="cap">{caption}</p>'
            f'<details><summary>Table view</summary>{tbl}</details></figure>')


def verdict(ok, text):
    icon = "✓" if ok == "held" else ("✕" if ok == "failed" else "◐")
    return f'<span class="vd {ok}"><b aria-hidden="true">{icon}</b>{esc(text)}</span>'


# ------------------------------------------------------------------ data ----
def recency_ar():
    e = est(S["exp3"])
    q = e[(e.part == "A") & (e.measure == "delta_logp_codes") & (e.m == 0) & (e.digit == 0)]
    return {v: [(int(r.recency), r.est, r.lo, r.hi) for r in g.sort_values("recency").itertuples()]
            for v, g in q.groupby("variant")}


def recency_diff():
    e = est(S["exp7"])
    q = e[(e.estimand == "delta S_full") & (e.m == 0) & (e.digit == 0)]
    return {v: [(int(r.r), r.est, r.lo, r.hi) for r in g.sort_values("r").itertuples()]
            for v, g in q.groupby("variant")}


def patching_ar():
    e = est(S["exp3"])
    q = e[(e.part == "B") & (e.measure == "recovery") & (e.digit == 0) & e.group.isin(["item", "header"])]
    return {(v, g_): [(int(r.layer), r.est, r.lo, r.hi) for r in gg.sort_values("layer").itertuples()]
            for (v, g_), gg in q.groupby(["variant", "group"])}


def pick(e, **kw):
    q = e
    for k, v in kw.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise KeyError(f"{kw}: {len(q)} rows")
    r = q.iloc[0]
    return (float(r.est), float(r.lo), float(r.hi), int(r.n_rows))


def by_q(e, **kw):
    return {Q[v][0]: pick(e[e.variant == v], **kw) for v in Q if (e.variant == v).any()}


# ------------------------------------------------------- 2026-09-30 sections --
# Time (retrospective exp6 + exp9) and SAE features (representation exp2). Every
# number is read from those experiments' estimate tables.
T = {
    "retro6": D / "retrospective/exp6_time_structure/287189/result",
    "exp9": D / "controlled/exp9_ar_order_vs_time/summary-287182",
    "exp9v": D / "controlled/exp9_ar_order_vs_time/summary-287183",
    "sae": D / "representation/exp2_ar_sae/summary-287222-287231",
    "saea1": D / "representation/exp2_ar_sae/summary-a1-287380-287415",
    "a1sel": (D / "representation/exp2_ar_sae/amend_select_within_d-287374/cell-00/a1_selection.csv",
              D / "representation/exp2_ar_sae/amend_select_within_d-287395/cell-01/a1_selection.csv"),
}
QD = {"diff-next1-rqkmeans-3cb-128": ("rqk", "RQ-KMeans 3×128"), "diff-next1-rqvae-4cb-128": ("rqvae", "RQ-VAE 4×128")}
BINS = ["same day", "1-30 d", "31-365 d", ">365 d"]
BIN_LABEL = {"same day": "same day", "1-30 d": "1–30 days", "31-365 d": "31–365 days", ">365 d": "over a year"}
AGE_CLS = {"same day": "s-age0", "1-30 d": "s-age1", "31-365 d": "s-age2", ">365 d": "s-age3"}
STRATA9 = [("U", "Most recent item on a later day than the one before"),
           ("T_later", "Both on the same day, target on a later day"),
           ("T_same", "Both, and the target, on the same day")]


def panel(title, body, sub=""):
    return (f'<div class="panel"><div class="ptitle">{esc(title)}'
            f'{f"<span>{esc(sub)}</span>" if sub else ""}</div>{body}</div>')


def share_forest(rows, series, xdom, xlab, unit=""):
    """forest() for a non-nats quantity: the tooltip carries `unit` instead of 'nats'."""
    return forest(rows, series, xdom, xlab).replace(" nats\n95% CI", f"{unit}\n95% CI")


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def new_sections(QS):
    r6 = pd.read_csv(T["retro6"] / "estimates.csv")
    e9 = pd.read_csv(T["exp9"] / "estimates.csv")
    e9v = pd.read_csv(T["exp9v"] / "estimates.csv")
    sc = pd.read_csv(T["sae"] / "causal_estimates.csv")
    so = pd.read_csv(T["sae"] / "observational_estimates.csv")
    sf = pd.read_csv(T["sae"] / "fidelity.csv")
    sa1 = pd.read_csv(T["saea1"] / "causal_estimates.csv")
    a1sel = pd.concat([pd.read_csv(p) for p in T["a1sel"]])
    figs, out = {}, {}
    g = lambda **kw: pick(r6, **kw)                                              # noqa: E731
    KV = list(Q.items())

    # ---- T1: informativeness by position and age ----------------------------
    panels, trows = [], []
    for v, (k, lab) in KV:
        q = r6[(r6.part == "D3") & (r6.cohort == "AR test") & (r6.variant == v) & (r6.estimand == "P(match0)")]
        ser = []
        for b in BINS:
            s = q[(q.gap_bin == b) & (q.n_rows >= 30)].sort_values("recency")
            pts = [(int(r.recency), float(r.est), None, None) for r in s.itertuples()]
            ser.append({"name": BIN_LABEL[b], "short": BIN_LABEL[b], "cls": AGE_CLS[b], "pts": pts,
                        "tipfmt": lambda p, b=b, lab=lab: f"{100 * p[1]:.1f}% share the target's first digit\n"
                                                          f"{lab} · item {p[0]} back · {BIN_LABEL[b]} before the target"})
            trows += [(lab, BIN_LABEL[b], int(r.recency), f"{100 * r.est:.1f}%", f"{int(r.n_rows):,}") for r in s.itertuples()]
        panels.append(panel(lab, line_panel(lab, ser, (1, 10), (0, 0.6), "history item (1 = most recent)",
                                            "shares the target's first digit", xticks=[1, 2, 4, 6, 8, 10],
                                            yticks=[0, 0.2, 0.4, 0.6], band=False, end_labels=False), "AR test split"))
    r2 = {k: (g(part="D3", cohort="AR test", variant=v, estimand="R2 gain position beyond time")[0],
              g(part="D3", cohort="AR test", variant=v, estimand="R2 gain time beyond position")[0]) for v, (k, _) in KV}
    figs["t1"] = figure(
        "fig-age", "In the data, an item's age matters and its position barely does",
        "How often a history item shares the target's first SID digit, by how far back it sits and how long before the target it was reviewed",
        f'<div class="multi">{"".join(panels)}</div>',
        [(AGE_CLS[b], f"{BIN_LABEL[b]} before the target", "line") for b in BINS],
        table(["SID", "item age", "item back", "shares first digit", "pairs"], trows, num_from=2),
        f"Every (row, history item) pair of the AR test split; cells with fewer than 30 pairs are left out. At a fixed age "
        f"the lines are nearly flat in position. In a logistic model, adding position to age explains "
        f"{f2(r2['rqk'][0], 3)} (RQ-KMeans) and {f2(r2['rqvae'][0], 3)} (RQ-VAE) more of the deviance (McFadden R²); "
        f"adding age to position explains {f2(r2['rqk'][1], 2)} and {f2(r2['rqvae'][1], 2)}.")

    # ---- T2: copy calibration -----------------------------------------------
    panels, trows, rho = [], [], {}
    for coh, title, VM in (("AR test", "Qwen AR", Q), ("DiffGRM", "DiffGRM", QD)):
        ser = []
        for v, (k, lab) in VM.items():
            rho[(coh, k)] = g(part="M1", cohort=coh, variant=v, estimand="rho", gap_bin="same/>365")
            for estn, what, dash in (("copy rate c", "model", False), ("data rate v", "data", True)):
                pts = []
                for i, b in enumerate(BINS):
                    e, lo, hi, n = g(part="M1", cohort=coh, variant=v, estimand=estn, gap_bin=b)
                    pts.append((i, e, lo, hi))
                    trows.append((title, lab, "model copies it" if not dash else "target shares it (data)",
                                  BIN_LABEL[b], f"{100 * e:.1f}%", f"[{100 * lo:.1f}, {100 * hi:.1f}]", f"{n:,}"))
                ser.append({"name": f"{lab} {what}", "short": f"{lab.split()[0]} {what}", "cls": f"s-{k}", "dash": dash,
                            "pts": pts,
                            "tipfmt": lambda p, lab=lab, dash=dash, title=title: (
                                f"{100 * p[1]:.1f}% of rows\n95% CI {100 * p[2]:.1f} to {100 * p[3]:.1f}\n"
                                f"{title} · {lab} · {'the target shares the most recent item' if dash else 'the model copies the most recent item'}"
                                f"'s first digit · {BIN_LABEL[BINS[p[0]]]}")})
        panels.append(panel(title, line_panel(title, ser, (0, 3), (0, 0.75), "time from the most recent item to the target",
                                              "share of rows", xticks=[0, 1, 2, 3], yticks=[0, 0.25, 0.5, 0.75],
                                              xlabels={0: "same day", 1: "1–30 d", 2: "31–365 d", 3: "> 1 year"}),
                            "AR test split" if coh == "AR test" else "leave-last-out cohort"))
    rq = lambda coh, k: rho[(coh, k)]                                            # noqa: E731
    figs["t2"] = figure(
        "fig-calibration", "The models copy the most recent item whatever its age",
        "Solid: the model's top first digit is the most recent item's first digit. Dashed: the target's first digit is (how often copying would be right).",
        f'<div class="multi">{"".join(panels)}</div>',
        [("s-rqk", "RQ-KMeans, model", "line"), ("s-rqk", "RQ-KMeans, data", "line dash"),
         ("s-rqvae", "RQ-VAE, model", "line"), ("s-rqvae", "RQ-VAE, data", "line dash")],
        table(["model", "SID", "rate", "age of the most recent item", "share", "95% CI", "rows"], trows, num_from=4),
        "ρ = (copy rate on same-day rows − over a year) / (data rate on same-day rows − over a year): 1 means the model "
        "copies as the data warrant, 0 means it ignores age. AR: "
        f"{f2(rq('AR test', 'rqk')[0], 2)} [{f2(rq('AR test', 'rqk')[1], 2)}, {f2(rq('AR test', 'rqk')[2], 2)}] and "
        f"{f2(rq('AR test', 'rqvae')[0], 2)} [{f2(rq('AR test', 'rqvae')[1], 2)}, {f2(rq('AR test', 'rqvae')[2], 2)}]; "
        f"DiffGRM: {f2(rq('DiffGRM', 'rqk')[0], 2)} and {f2(rq('DiffGRM', 'rqvae')[0], 2)}. No model is given a timestamp, "
        "so any dependence on age comes from content. The cohorts differ; compare shapes.")

    # ---- T3: exp9 position vs item ------------------------------------------
    p9 = lambda est_, s_, v: pick(e9[(e9.variant == v) & (e9.part == "TF")], stratum=s_, estimand=est_)  # noqa: E731
    body, trows = [], []
    for est_, ttl in (("A position effect (d0)", "A · prefers whatever sits last"),
                      ("B item effect (d0)", "B · prefers the item that really is later")):
        rows = [(lab, {Q[v][0]: p9(est_, s_, v) for v in Q}) for s_, lab in STRATA9]
        body.append(panel(ttl, share_forest(rows, QS, (-0.1, 0.45), "difference in share of rows (top-1 first digit)")))
        for s_, lab in STRATA9:
            for v in Q:
                x = p9(est_, s_, v)
                da = p9("data asymmetry P(t=c1)-P(t=c2)", s_, v)
                trows.append((ttl.split(" · ")[0], lab, Q[v][1], f2(x[0], 3, True), f"[{f2(x[1], 3, True)}, {f2(x[2], 3, True)}]",
                              f2(da[0], 3, True), f"{x[3]:,}"))
    P2 = {Q[v][0]: pick(e9[e9.variant == v], stratum="T_later - U", estimand="P2 A(T_later) - A(U)") for v in Q}
    figs["t3"] = figure(
        "fig-swap", "Swapping the two most recent items: the preference follows the position",
        "exp9. The same two items in both orders; strata are fixed from their review days before any forward",
        f'<div class="multi">{"".join(body)}</div>', [(c, n, "dot") for _, c, n in QS],
        table(["effect", "stratum", "SID", "estimate", "95% CI", "data asymmetry", "rows"], trows, num_from=3),
        "A = ½[(top-1 is r1's digit − top-1 is r2's digit) in the original order + (top-1 is r2's digit − r1's) after the swap]; "
        "B is the same with r1 counted in both orders. r1 is the most recent item and r2 the one before. Rows where r1 and r2 "
        "have different first digits, AR test split. In the middle stratum the data give no reason to prefer either item "
        f"(data asymmetry {f2(p9('data asymmetry P(t=c1)-P(t=c2)', 'T_later', 'rqkmeans_3codebook_128')[0], 3, True)}). "
        f"A on tied rows minus A on untied rows: {f2(P2['rqk'][0], 3, True)} [{f2(P2['rqk'][1], 3, True)}, {f2(P2['rqk'][2], 3, True)}] "
        f"(RQ-KMeans) and {f2(P2['rqvae'][0], 3, True)} [{f2(P2['rqvae'][1], 3, True)}, {f2(P2['rqvae'][2], 3, True)}] (RQ-VAE).")

    # ---- T4: exp9 flips and tie averaging -----------------------------------
    rows_mix = [(lab, {Q[v][0]: p9("P4 delta_mix SID (tie average - clean)", s_, v) for v in Q}) for s_, lab in STRATA9]
    rows_flip = [(lab, {Q[v][0]: p9("P3 flip rate top1 d0", s_, v) for v in Q}) for s_, lab in STRATA9]
    trows = []
    for s_, lab in STRATA9:
        for v in Q:
            mix, flip = p9("P4 delta_mix SID (tie average - clean)", s_, v), p9("P3 flip rate top1 d0", s_, v)
            d1 = pick(e9[(e9.variant == v) & (e9.part == "D")], stratum=s_, estimand="D top1_changed")
            dh = pick(e9[(e9.variant == v) & (e9.part == "D")], stratum=s_, estimand="D HR@10 swap - clean")
            trows.append((lab, Q[v][1], f2(mix[0], 3, True), f"[{f2(mix[1], 3, True)}, {f2(mix[2], 3, True)}]",
                          pct(flip[0]), pct(d1[0]), f"{f2(100 * dh[0], 1, True)} pp"))
    body = [panel("Tie average − original order (nats, whole SID)", forest(rows_mix, QS, (-0.06, 0.06), "Δ log p of the correct SID")),
            panel("First-digit decisions the swap flips", share_forest(rows_flip, QS, (0, 0.6), "share of rows"))]
    mixdiff = {Q[v][0]: pick(e9[e9.variant == v], stratum="T_later - U", estimand="delta_mix(T_later) - delta_mix(U)") for v in Q}
    figs["t4"] = figure(
        "fig-tieavg", "Half of RQ-KMeans' first-digit decisions on tied rows depend on ASIN order",
        "exp9. Left: averaging the model over both orders of the two items. Right: how often the swap changes the top first digit",
        f'<div class="multi">{"".join(body)}</div>', [(c, n, "dot") for _, c, n in QS],
        table(["stratum", "SID", "tie average − original", "95% CI", "first digit flips", "top-1 SID changes (beam search)",
               "HR@10 swap − original"], trows, num_from=2),
        "Averaging is only licensed where the timestamps cannot order the two items (the lower two strata). For RQ-KMeans it "
        f"gains on tied rows and loses where the order is real; the difference is {f2(mixdiff['rqk'][0], 3, True)} nats "
        f"[{f2(mixdiff['rqk'][1], 3, True)}, {f2(mixdiff['rqk'][2], 3, True)}]. The beam-search columns come from plain beam "
        "search on the test split (Part D, whose original order reproduces exp6's lists for 3,680 of 3,681 rows).")

    # ---- T5: exp6 by target type --------------------------------------------
    TT = [("dup", "Duplicate review record (same item, same day)"), ("repeat_same_day", "Repeat SID, same day"),
          ("repeat_later", "Repeat SID, later day"), ("new_same_day", "New item, same day"), ("new_later", "New item, later day")]
    m3 = r6[(r6.part == "M3") & (r6.decoder == "archived")]
    rows5, trows = [], []
    for t_, lab in TT:
        vals = {}
        for v, (k, name) in KV:
            x = pick(m3[m3.variant == v], target_type=t_, estimand="dHR@10 C_all - B")
            vals[k] = (100 * x[0], 100 * x[1], 100 * x[2], x[3])
            base = pick(m3[m3.variant == v], target_type=t_, estimand="HR@10 baseline")
            sh = pick(m3[m3.variant == v], target_type=t_, estimand="share of rows")
            hits = pick(m3[m3.variant == v], target_type=t_, estimand="share of baseline hits")
            trows.append((lab, name, pct(sh[0], 1), pct(base[0], 1), f"{f2(100 * x[0], 1, True)} pp",
                          f"[{f2(100 * x[1], 1, True)}, {f2(100 * x[2], 1, True)}]", pct(hits[0], 0)))
        rows5.append((lab, vals))
    lo5 = min(v[1] for _, vals in rows5 for v in vals.values())
    fx = forest(rows5, QS, (math.floor(lo5 / 10) * 10, 5), "Δ HR@10 when copying is blocked (percentage points)")
    fx = fx.replace(" nats\n95% CI", " pp\n95% CI")
    sd = {Q[v][0]: pick(m3[m3.variant == v], target_type="gap1 = 0", estimand="X4 share of all-rows HR@10 loss") for v in Q}
    figs["t5"] = figure(
        "fig-where", "The copy benefit lives on same-day rows",
        "exp6 re-read by review day: the change in HR@10 when the copy reads are blocked, by kind of target",
        fx, [(c, n, "dot") for _, c, n in QS],
        table(["target", "SID", "share of rows", "HR@10 as run", "Δ HR@10", "95% CI", "share of all hits"], trows, num_from=2),
        f"Archived decoder, AR test split. Rows whose target falls on the most recent item's day carry {pct(sd['rqk'][0])} "
        f"(RQ-KMeans) and {pct(sd['rqvae'][0])} (RQ-VAE) of the all-rows HR@10 loss. Repeat SIDs on later days are rare for "
        "RQ-VAE (7 rows), so that row is noisy.")

    # ---- numbers for the prose ----------------------------------------------
    d1 = lambda coh, est_: g(part="D1", cohort=coh, variant="rqkmeans_3codebook_128", estimand=est_)[0]   # noqa: E731
    asin = pick(r6, part="D2", estimand="same-day consecutive pairs ASIN ascending")[0]
    cr = {b: g(part="M1", cohort="AR test", variant="rqkmeans_3codebook_128", estimand="copy rate c", gap_bin=b)[0] for b in BINS}
    dr = {b: g(part="M1", cohort="AR test", variant="rqkmeans_3codebook_128", estimand="data rate v", gap_bin=b)[0] for b in BINS}
    hits_dup = {Q[v][0]: pick(m3[m3.variant == v], target_type="dup", estimand="share of baseline hits")[0] for v in Q}
    m4 = {(Q[v][0], s_): pick(r6[(r6.part == "M4") & (r6.variant == v)], stratum=s_, estimand="ratio H1 valid / test")
          for v in Q for s_ in ("raw", "post-stratified")}
    A = {(Q[v][0], s_): p9("A position effect (d0)", s_, v) for v in Q for s_ in ("U", "T_later")}
    Av = {Q[v][0]: pick(e9v[(e9v.variant == v) & (e9v.part == "TF")], stratum="T_later", estimand="A position effect (d0)") for v in Q}
    flipT = {Q[v][0]: p9("P3 flip rate top1 d0", "T_later", v) for v in Q}
    top1T = {Q[v][0]: pick(e9[(e9.variant == v) & (e9.part == "D")], stratum="T_later", estimand="D top1_changed") for v in Q}
    mixT = {Q[v][0]: p9("P4 delta_mix SID (tie average - clean)", "T_later", v) for v in Q}
    Bu = {Q[v][0]: p9("B item effect (d0)", "U", v) for v in Q}
    tiles = [
        (f"{pct(d1('AR test', 'share gap1 same day'))} <span class=\"sep\">/</span> {pct(d1('AR train', 'share gap1 same day'))}",
         "of targets fall on the same day as the most recent history item", "AR test / train split"),
        (pct(asin), "of same-day item pairs are simply in ASIN order", "upstream breaks day ties by raw-file order"),
        (f"{f2(r2['rqk'][0], 3)} <span class=\"sep\">vs</span> {f2(r2['rqk'][1], 2)}",
         "what position adds once age is known, vs what age adds to position", "McFadden R², RQ-KMeans, AR test"),
        (f"{pct(d1('AR test', 'share dup_target'), 1)} <span class=\"sep\">→</span> {pct(hits_dup['rqk'])} / {pct(hits_dup['rqvae'])}",
         "duplicate-record rows, and their share of all HR@10 hits", "AR test · RQ-KMeans / RQ-VAE"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="tv">{v}</div><div class="tl">{esc(l)}</div><div class="ts">{esc(s)}</div></div>'
                        for v, l, s in tiles)

    sec_time = f'''<section id="s9">
  <div class="head"><div class="sno">9 · Time · retrospective exp6, exp9</div><h2>In the data, "recent" means the same day</h2></div>
  <p>No model is given a timestamp. The AR prompt lists the history "in chronological order", and DiffGRM has a recency
    position, so both see only order. The frozen reviews keep a day for every event, and <code>sidlens.data.timestamps</code>
    puts it back on all 43,102 of them. That exposes three facts about the data. Reviews come in same-day bursts. Within a
    day, the order is ASIN order, because upstream breaks ties by raw-file order. And some reviews are recorded twice.</p>
  <div class="tiles">{tile_html}</div>
  <p>Split by time, the data say something simple. Whether a history item shares the target's first digit depends on how
    long before the target it was reviewed, and hardly at all on where it sits in the list. A same-day item seven places back
    is about as informative as the most recent one; an item from over a year ago is uninformative wherever it sits.</p>
  {figs["t1"]}
  <p>The models cannot see this. RQ-KMeans AR copies the most recent item's first digit in {pct(cr["same day"])} of same-day
    rows and in {pct(cr[">365 d"])} of rows where that item is over a year old. The data justify the copy in {pct(dr["same day"])}
    and {pct(dr[">365 d"])} of them. The other three models follow a quarter to two fifths of the drop, which can only come
    from content cues.</p>
  {figs["t2"]}
  <p>exp9 tests order directly. It swaps the two most recent items and keeps everything else fixed. When both were reviewed
    on the same day and the target came later, the swap leaves every timestamp as it was, and the data treat the two items
    as exchangeable. RQ-KMeans still prefers whichever item is last, by {f2(A[("rqk", "T_later")][0], 2)} on those rows against
    {f2(A[("rqk", "U")][0], 2)} where the last item really is later ({f2(Av["rqk"][0], 2)} on the valid split). RQ-VAE is only
    weakly positional ({f2(A[("rqvae", "T_later")][0], 2)} and {f2(A[("rqvae", "U")][0], 2)}), and it follows the truly later item
    about as strongly ({f2(Bu["rqvae"][0], 2)}).</p>
  {figs["t3"]}
  <p>So an arbitrary order moves real decisions. On tied rows the swap flips {pct(flipT["rqk"][0])} of RQ-KMeans' first-digit
    choices and changes its top-1 recommended SID in {pct(top1T["rqk"][0])}, though HR@10 stays level. Averaging the model over
    the two orders the timestamps cannot tell apart gains {f2(mixT["rqk"][0], 3, True)} nats on those rows. It is a free,
    timestamp-aware inference rule, and it helps only where it is licensed.</p>
  {figs["t4"]}
  <p>Time also changes how section 4 reads. The benefit of copying sits almost entirely on rows whose target falls on the
    most recent item's day. It does nothing for a new item reviewed on a later day, the largest kind of target. Part of the
    benefit is an artefact: duplicate review records are {pct(d1("AR test", "share dup_target"), 1)} of test rows, but they supply
    {pct(hits_dup["rqk"])} (RQ-KMeans) and {pct(hits_dup["rqvae"])} (RQ-VAE) of all HR@10 hits. Duplicates also explain part of
    the gap between valid and test effect sizes (section 12). Valid has twice test's share of them, and reweighting test to valid's mix moves the exp4
    ratio from {f2(m4[("rqk", "raw")][0], 2)} to {f2(m4[("rqk", "post-stratified")][0], 2)} (RQ-KMeans) and from
    {f2(m4[("rqvae", "raw")][0], 2)} to {f2(m4[("rqvae", "post-stratified")][0], 2)} (RQ-VAE).</p>
  {figs["t5"]}
  <p class="expts">Associational where it re-reads earlier outputs; exp9 is the controlled test of order. Review days are not
    purchase days. The DiffGRM cohort (leave-last-out) is not a time split: its targets span 2013 to 2018.</p>
</section>'''

    # ---- SAE section ---------------------------------------------------------
    cs = lambda v, part, est_, **kw: pick(sc[(sc.variant == v) & (sc.part == part)], estimand=est_, **kw)   # noqa: E731
    panels, trows = [], []
    for est_, part, ttl, ydom, yt, ylab in (
            ("CC1 dCopyScore(copy) - dCopyScore(control)", "CC", "Copy latents at the first-digit readout",
             (-0.05, 0.45), [0, 0.1, 0.2, 0.3, 0.4], "copy score lost vs control (nats)"),
            ("CM1 dS(match) - dS(nonmatch)", "CM", "Prefix-match latents at the digit-d readout",
             (-0.05, 0.1), [-0.05, 0, 0.05, 0.1], "match − non-match (nats)")):
        ser = []
        for v, (k, lab) in KV:
            pts = [(L, *cs(v, part, est_, layer=L)[:3]) for L in (12, 16, 20, 24)]
            ser.append({"name": lab, "short": lab.split()[0], "cls": f"s-{k}", "pts": pts,
                        "tipfmt": lambda p, lab=lab: f"{f2(p[1], 3, True)} nats\n95% CI {f2(p[2], 3, True)} to {f2(p[3], 3, True)}\n{lab} · layer {p[0]}"})
            trows += [(ttl, lab, p[0], f2(p[1], 3, True), f"[{f2(p[2], 3, True)}, {f2(p[3], 3, True)}]") for p in pts]
        panels.append(panel(ttl, line_panel(ttl, ser, (12, 24), ydom, "layer", ylab, xticks=[12, 16, 20, 24], yticks=yt,
                                            ref_lines=[(0, "")])))
    figs["s1"] = figure(
        "fig-sae", "Copy features are causal from layer 20; match detectors are not read at the readout",
        "Error-preserving ablation of SAE latents at one layer and one position, held-out users, AR test split",
        f'<div class="multi">{"".join(panels)}</div>', [(c, n, "line") for _, c, n in QS],
        table(["ablation", "SID", "layer", "estimate", "95% CI"], trows, num_from=2),
        "Left: remove the active latents that fire for the recent item's first digit, minus a matched control in the same row; "
        "the copy score is the log-probability of copying that digit. Right: remove the 8 latents that best separate matching "
        "from non-matching rows, and compare the cost to the correct digit in matching and non-matching rows. An amendment "
        "(A1, post hoc) that selects latents within each digit gives the same picture. All self-patch controls are bit-exact.")

    ob = lambda v, q_, est_, L: pick(so[(so.variant == v) & (so.q == q_)], estimand=est_, layer=L)   # noqa: E731
    fid = {Q[v][0]: sf[sf.variant == v] for v in Q}
    rec = {Q[v][0]: sc[(sc.variant == v) & (sc.part == "F2") & sc.estimand.str.startswith("recovered")].est.min() for v in Q}
    a1best = {Q[v][0]: a1sel[a1sel.variant == v].groupby("layer").heldout_within_d_auc.max() for v in Q}
    a1cm = {Q[v][0]: sa1[(sa1.variant == v) & (sa1.estimand == "CM1 dS(match) - dS(nonmatch)")] for v in Q}
    t1z = {Q[v][0]: so[(so.variant == v) & (so.estimand == "T1 Z: R2(Z+B0) - R2(B0) (new)") & so.layer.isin([12, 16])].est for v in Q}
    t2 = {Q[v][0]: so[(so.variant == v) & (so.estimand == "T2 AUC(R+content) - AUC(content)")] for v in Q}
    rng = lambda s_, d=2: f"{f2(s_.min(), d)}–{f2(s_.max(), d)}"             # noqa: E731
    rngp = lambda s_: f"{f2(s_.min(), 3, True)} to {f2(s_.max(), 3, True)}"    # noqa: E731

    def srow(label, a, b):
        return f'<tr><td>{label}</td><td class="n">{a}</td><td class="n">{b}</td></tr>'
    srows = [
        srow("Reconstruction error on test tokens (fraction of variance unexplained, layers 12–24)",
             rng(fid["rqk"].fvu_test_all, 3), rng(fid["rqvae"].fvu_test_all, 3)),
        srow("… at the first-digit readout only, where rows differ least", rng(fid["rqk"].fvu_test_readout0), rng(fid["rqvae"].fvu_test_readout0)),
        srow("Spliced into the model: share of the output kept (vs mean-ablation)", f"≥ {pct(rec['rqk'])}", f"≥ {pct(rec['rqvae'])}"),
        srow("Copy latents' share of readout activation, layer 24",
             pct(ob(Q_rev("rqk"), "Q1", "O1b share of readout(0) activation on on-copy latents", 24)[0], 1),
             pct(ob(Q_rev("rqvae"), "Q1", "O1b share of readout(0) activation on on-copy latents", 24)[0], 1)),
        srow("Removing them: copy score lost vs control, layer 20 / 24",
             f"{f2(cs(Q_rev('rqk'), 'CC', 'CC1 dCopyScore(copy) - dCopyScore(control)', layer=20)[0], 2)} / "
             f"{f2(cs(Q_rev('rqk'), 'CC', 'CC1 dCopyScore(copy) - dCopyScore(control)', layer=24)[0], 2)} nats",
             f"{f2(cs(Q_rev('rqvae'), 'CC', 'CC1 dCopyScore(copy) - dCopyScore(control)', layer=20)[0], 2)} / "
             f"{f2(cs(Q_rev('rqvae'), 'CC', 'CC1 dCopyScore(copy) - dCopyScore(control)', layer=24)[0], 2)} nats"),
        srow("Best prefix-match latent per layer, held-out within-digit AUC (A1; range over layers)",
             rng(a1best["rqk"]), rng(a1best["rqvae"])),
        srow("Removing match latents: match − non-match cost, declared / A1",
             f"{rngp(sc[(sc.variant == Q_rev('rqk')) & (sc.estimand == 'CM1 dS(match) - dS(nonmatch)')].est)} / {rngp(a1cm['rqk'].est)}",
             f"{rngp(sc[(sc.variant == Q_rev('rqvae')) & (sc.estimand == 'CM1 dS(match) - dS(nonmatch)')].est)} / {rngp(a1cm['rqvae'].est)}"),
        srow("Target content beyond the history (R² gain from the latents, new targets, layers 12–16)",
             rngp(t1z["rqk"]), rngp(t1z["rqvae"])),
        srow("Same-day burst beyond SID overlap (AUC gain, layers 12–24)", f"{rngp(t2['rqk'].est)}", f"{rngp(t2['rqvae'].est)}"),
    ]
    sae_table = ('<div class="prose-table"><table><thead><tr><th>SAEs, per model (layers 12, 16, 20, 24)</th>'
                 '<th>RQ-KMeans 3×128</th><th>RQ-VAE 4×128</th></tr></thead><tbody>' + "".join(srows) + "</tbody></table></div>")
    sec_sae = f'''<section id="s10">
  <div class="head"><div class="sno">10 · Features · representation exp2</div><h2>What sparse autoencoders find inside the AR models</h2></div>
  <p>The probes in section 3 read the model through questions we chose in advance. A sparse autoencoder (SAE) instead learns
    its own basis: each activation is rewritten as a sum of a few directions, called latents, out of 12,288. We fitted
    one SAE per model at layers 12, 16, 20 and 24, on train-split activations, and read the test split in that basis. A latent
    can then be removed from the real activation without touching the SAE's reconstruction error, which makes it a unit
    for a causal test.</p>
  {sae_table}
  <p>Four things come out.</p>
  <ul class="plain">
    <li><b>Copy features exist and are causal, late.</b> A handful of code-specific latents at the first-digit readout fire
      on the recent item's first digit. Removing them costs the copy most at layers 20 and 24, where exp3's patching and the
      logit lens put the copy decision. They carry little of the readout's activation, though. Earlier, the digit is
      decodable (section 3) but not held in features whose removal matters.</li>
    <li><b>The model detects the prefix match, but the output does not read it there.</b> RQ-KMeans has sharp latents that
      tell whether the recent item matches the prefix so far. Removing them at the readout changes nothing. With exp4's
      single-edge result, this points to the match being computed inside attention's query–key comparison. That narrows
      the open question of where the match is computed.</li>
    <li><b>Nothing about the target beyond the history.</b> Neither the residual nor the latents predict the target item's
      content embedding better than a linear map of the history's items.</li>
    <li><b>The model does sense same-day bursts.</b> Its readout predicts whether the most recent item was reviewed on the
      target's day better than five SID-overlap features do. RQ-KMeans' copying ignores that signal (section 9).</li>
  </ul>
  {figs["s1"]}
  <p class="expts">One SAE per model and layer, fitted on about 0.5M tokens with nothing tuned. The dead-latent share is high in
    RQ-KMeans' early layers. An ablation at one layer and one position leaves later layers free to re-read the history,
    so a null effect bounds that position's role, not the whole computation.</p>
</section>'''
    out.update({"sec_time": sec_time, "sec_sae": sec_sae})
    # §1 chips, lead paragraphs
    out["v_time"] = verdict("held", "5 of 6") + " " + verdict("partial", "age-flat reliance")
    out["v_exp9"] = verdict("held", "RQ-KMeans 5 of 5") + " " + verdict("failed", "RQ-VAE 2 of 5")
    out["v_sae"] = (verdict("held", "4 of 6") + " " + verdict("failed", "copy share") + " "
                    + verdict("partial", "match latents"))
    out["lead_time"] = esc(
        f"No model sees a timestamp. In the data, the item that predicts the target is one from the same day, "
        f"wherever it sits in the list; the models copy whatever sits last. RQ-KMeans AR copies the most recent item's "
        f"first digit in {pct(cr['same day'])} of same-day rows and {pct(cr['>365 d'])} of rows where it is over a year old, "
        f"against {pct(dr['same day'])} and {pct(dr['>365 d'])} that the data justify. Same-day rows carry "
        f"{pct(sd['rqk'][0])} and {pct(sd['rqvae'][0])} of the copy benefit, and duplicate review records supply "
        f"{pct(hits_dup['rqk'])} and {pct(hits_dup['rqvae'])} of all HR@10 hits.")
    out["lead_sae"] = esc(
        "Sparse autoencoders find code-specific copy features that are causal from layer 20, and prefix-match detectors "
        "that the output does not read at the readout. They find no target information beyond the history, but a "
        "same-day burst signal the copy ignores.")
    return out


def Q_rev(k):
    return next(v for v, (kk, _) in Q.items() if kk == k)



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exp6", type=Path, required=True, help="exp6 summary directory")
    ap.add_argument("--exp6-plain", type=Path, default=None, help="summary of the corrected plain-decoder re-run")
    ap.add_argument("--exp6-verdict", default="", help="HTML verdict chips for exp6, from its RESULTS")
    ap.add_argument("--standalone", type=Path, default=None,
                    help="also write the page as a complete HTML document (for opening from disk)")
    args = ap.parse_args(argv)
    global V6
    V6 = args.exp6_verdict
    S["exp6"] = args.exp6
    QS = [("rqk", "s-rqk", "RQ-KMeans 3×128"), ("rqvae", "s-rqvae", "RQ-VAE 4×128")]
    figs = {}

    # F1 recency (AR vs DiffGRM)
    ra, rd = recency_ar(), recency_diff()
    ymax = max(max(p[3] for p in v) for d in (ra, rd) for v in d.values())
    panels, trows = [], []
    for name, data, cohort in (("Qwen AR", ra, "AR test cohort, 3,681 rows"), ("DiffGRM", rd, "diffusion cohort, 6,297 users")):
        ser = []
        for v, (k, lab) in Q.items():
            pts = data[v]
            ser.append({"name": lab, "short": lab.split(" ")[0], "cls": f"s-{k}", "pts": pts,
                        "tipfmt": lambda p, lab=lab, name=name: f"{f2(p[1], 3, True)} nats\n95% CI {f2(p[2], 3, True)} to {f2(p[3], 3, True)}\n{lab} · {name} · item {p[0]} back"})
            trows += [(name, lab, p[0], f2(p[1], 3, True), f"[{f2(p[2], 3, True)}, {f2(p[3], 3, True)}]") for p in pts]
        panels.append(f'<div class="panel"><div class="ptitle">{esc(name)}<span>{esc(cohort)}</span></div>'
                      + line_panel(name, ser, (1, 10), (0, math.ceil(ymax * 10) / 10), "replaced item (1 = most recent)",
                                   "Δ log p, first digit (nats)", xticks=[1, 2, 4, 6, 8, 10]) + "</div>")
    figs["recency"] = figure(
        "fig-recency", "Both paradigms lean on the most recent item",
        "Cost to the correct first digit of replacing one past item with a catalogue item that has a different first digit",
        f'<div class="multi">{"".join(panels)}</div>', [(c, n, "line") for _, c, n in QS],
        table(["model", "SID", "item back", "Δ nats", "95% CI"], trows, num_from=2),
        "AR: teacher-forced (exp3). DiffGRM: all digits masked, the first decoding step (exp7). The cohorts differ, "
        "so compare shapes, not rows. Bands are paired user-bootstrap 95% intervals.")

    # F2 AR patching
    pa = patching_ar()
    panels, trows = [], []
    for v, (k, lab) in Q.items():
        ser = [{"name": "item's own tokens", "short": "item tokens", "cls": "s-ctx", "pts": pa[(v, "item")],
                "tipfmt": lambda p: f"{f2(p[1], 2)} recovered\nlayer {p[0]} · item's own tokens"},
               {"name": "answer position", "short": "answer position", "cls": f"s-{k}", "pts": pa[(v, "header")],
                "tipfmt": lambda p, lab=lab: f"{f2(p[1], 2)} recovered\nlayer {p[0]} · answer position · {lab}"}]
        for g_, nm in (("item", "item tokens"), ("header", "answer position")):
            trows += [(lab, nm, p[0], f2(p[1], 3), f"[{f2(p[2], 3)}, {f2(p[3], 3)}]") for p in pa[(v, g_)]]
        panels.append(f'<div class="panel"><div class="ptitle">{esc(lab)}</div>'
                      + line_panel(lab, ser, (0, 27), (0, 1.05), "layer", "share of the effect restored",
                                   xticks=[0, 5, 10, 15, 20, 25, 27], yticks=[0, 0.25, 0.5, 0.75, 1.0]) + "</div>")
    figs["patching"] = figure(
        "fig-patching", "The item's evidence moves late, in layers 19–24",
        "Copying the clean activation back into the run with the most recent item replaced: share of the first-digit effect restored",
        f'<div class="multi">{"".join(panels)}</div>',
        [("s-ctx", "at the item's own tokens", "line"), ("s-rqk", "at the answer position (RQ-KMeans)", "line"),
         ("s-rqvae", "at the answer position (RQ-VAE)", "line")],
        table(["model", "positions", "layer", "recovery", "95% CI"], trows, num_from=2),
        "exp3, test split. Recovery = Σ(patched − corrupted) / Σ(clean − corrupted). The separator tokens between "
        "the item and the answer never carry it (≈ 0 at every layer; table in the experiment's RESULTS).")

    # F3 exp4 forest
    e4 = est(S["exp4"])
    rows4 = [
        ("Change the recent item's digits from d on, when it matches the target so far", by_q(e4, estimand="delta | match (pooled)")),
        ("Same change when it does not match", by_q(e4, estimand="delta | nonmatch (pooled)")),
        ("Block the one link: answer position → that item's digit-d token, all layers", by_q(e4, estimand="H2a K1 | match (pooled d>=1)")),
        ("Block the link in layers 0–13 only", by_q(e4, estimand="K4_next_early | match (pooled d>=1)")),
        ("Block the link in layers 14–27 only", by_q(e4, estimand="K5_next_late | match (pooled d>=1)")),
        ("Block every token of that item", by_q(e4, estimand="K2_item | match (pooled d>=1)")),
        ("Link minus the same link to a non-matching item (paired)", by_q(e4, estimand="H2b K1 - K3 | match (paired, pooled d>=1)")),
        ("Top-5 heads minus layer-matched random heads (held-out users)", by_q(e4, estimand="H3 S - mean(control sets) | held-out match (pooled d>=1)")),
    ]
    trows = [(lab, name, f2(v[k][0], 3, True), f"[{f2(v[k][1], 3, True)}, {f2(v[k][2], 3, True)}]", f"{v[k][3]:,}")
             for lab, v in rows4 for k, _, name in QS if k in v]
    figs["copy"] = figure(
        "fig-copy", "One attention link carries the AR copy, and only late",
        "Cost to the correct digit d ≥ 2, in rows where the recent item already matches the target's first d digits",
        forest(rows4, QS, (-0.1, 1.6), "Δ log p (nats); positive = the change hurt the correct digit"),
        [(c, n, "dot") for _, c, n in QS], table(["condition", "SID", "Δ nats", "95% CI", "rows"], trows, num_from=2),
        "exp4, valid split (3,680 rows, 1,608 users), which played no part in forming the hypothesis. The top-5 heads "
        "carry 8–9% of the all-heads effect; no single head exceeds 0.007 nats.")

    # F4 head heatmaps
    heat, trows = [], []
    for cell, (v, (k, lab)) in zip(("cell-00", "cell-01"), Q.items()):
        sc = pd.read_parquet(S["exp4cells"] / cell / "part_h_screen.parquet", columns=["layer", "head", "delta"])
        grid = sc.groupby(["layer", "head"]).delta.mean().unstack().to_numpy()
        sel = json.loads((S["exp4cells"] / cell / "heads.json").read_text())["S"]
        heat.append(f'<div class="panel hp"><div class="ptitle">{esc(lab)}</div>{heatmap(grid, sel, lab, 0.007)}</div>')
        trows += [(lab, L_, h_, f2(grid[L_, h_], 4, True), "yes" if [L_, h_] in sel else "")
                  for L_ in range(grid.shape[0]) for h_ in range(grid.shape[1]) if grid[L_, h_] > 0.003]
    figs["heads"] = figure(
        "fig-heads", "No head carries the copy alone",
        "Knocking out the copy link for one head in one layer: mean cost on screening users (outlined: the 5 heads selected)",
        '<div class="scalelegend"><span>0</span><span class="ramp">' + "".join(f'<i style="background:var(--h{i})"></i>' for i in range(1, 14))
        + '</span><span>0.007 nats or more</span></div>' + f'<div class="multi heats">{"".join(heat)}</div>',
        [], table(["SID", "layer", "head", "Δ nats", "selected"], trows, num_from=1),
        'Shading follows the scale above, from 0 to 0.007 nats; cells at or below 0 are left unshaded. The table lists heads above 0.003 nats. '
        'For scale, blocking the link for all heads at once costs 0.39–0.43 nats.')

    # F5 logit lens
    er = est(S["rep"])
    panels, trows = [], []
    for v, (k, lab) in Q.items():
        s = er[(er.variant == v) & (er.kind == "lens") & (er.digit == 0)]
        gold = [(int(r.layer), r.est, r.lo, r.hi) for r in s[s.what == "golden top-1"].sort_values("layer").itertuples()]
        cp = [(int(r.layer), r.est, r.lo, r.hi) for r in s[s.what == "recent item's code top-1"].sort_values("layer").itertuples()]
        ser = [{"name": "correct first digit", "short": "correct digit", "cls": "s-ctx", "pts": gold,
                "tipfmt": lambda p: f"{f2(100 * p[1], 1)}% of rows\nlayer {p[0]} · correct first digit is top-1"},
               {"name": "recent item's first digit", "short": "recent item's digit", "cls": f"s-{k}", "pts": cp,
                "tipfmt": lambda p, lab=lab: f"{f2(100 * p[1], 1)}% of rows\nlayer {p[0]} · recent item's digit is top-1 · {lab}"}]
        trows += [(lab, "correct first digit", p[0], f"{100 * p[1]:.1f}%") for p in gold]
        trows += [(lab, "recent item's first digit", p[0], f"{100 * p[1]:.1f}%") for p in cp]
        panels.append(f'<div class="panel"><div class="ptitle">{esc(lab)}</div>'
                      + line_panel(lab, ser, (0, 28), (0, 0.85), "layer (28 = final norm)", "rows where it is the top code",
                                   xticks=[0, 7, 14, 21, 28], yticks=[0, 0.2, 0.4, 0.6, 0.8]) + "</div>")
    figs["lens"] = figure(
        "fig-lens", "The model's first digit is mostly a copy of the recent item's",
        "Logit lens at the position that predicts the first digit: how often each code is the top-1 code, by layer",
        f'<div class="multi">{"".join(panels)}</div>',
        [("s-ctx", "correct first digit", "line"), ("s-rqk", "recent item's first digit (RQ-KMeans)", "line"),
         ("s-rqvae", "recent item's first digit (RQ-VAE)", "line")],
        table(["model", "code", "layer", "top-1 rate"], trows, num_from=2),
        "Observational, test-split captures. A linear probe reads the recent item's first digit at this position from "
        "layer 0 (84% and 93%), long before the output prefers it; no probe reads the correct first digit better than "
        "copying (16.4% and 29.0%).")

    # F6 exp6: primary conditions from the main run; plain-decoder rows only from the corrected re-run
    e6 = est(S["exp6"])
    e6 = e6[~(e6.contrast.fillna("").str.contains("plain") | e6.condition.fillna("").str.contains("plain"))]
    have_plain = args.exp6_plain is not None
    if have_plain:
        e6 = pd.concat([e6, est(args.exp6_plain)], ignore_index=True)
    conds = [("C_all - B", "Block copy reads at every digit (first digit included)"),
             ("C_later - B", "Block copy reads at digits 2+ only"),
             ("N_later - B", "Control: block as many non-matching reads"),
             ("C_all_plain - B_plain", "Every digit, under plain beam search")][: 4 if have_plain else 3]
    rows6, trows = [], []
    for st, stn in (("repeat", "repeat targets"), ("new", "new targets")):
        for c, cn in conds:
            vals = {}
            for v, (k, lab) in Q.items():
                q = e6[(e6.variant == v) & (e6.estimand == "delta") & (e6.contrast == c) & (e6.metric == "hr10") & (e6.stratum == st)]
                if len(q):
                    r = q.iloc[0]
                    vals[k] = (100 * r.est, 100 * r.lo, 100 * r.hi, int(r.n_rows))
                    trows.append((stn, cn, lab, f2(100 * r.est, 2, True), f"[{f2(100 * r.lo, 2, True)}, {f2(100 * r.hi, 2, True)}]", f"{int(r.n_rows):,}"))
            rows6.append((f"{cn} · {stn}", vals))
    lo6 = min(min(v[1] for v in vals.values()) for _, vals in rows6)
    hi6 = max(max(v[2] for v in vals.values()) for _, vals in rows6)
    fx = forest(rows6, QS, (math.floor(lo6 / 5) * 5, max(2, math.ceil(hi6))), "Δ HR@10 (percentage points)")
    fx = fx.replace(" nats\n95% CI", " pp\n95% CI")
    figs["decode"] = figure(
        "fig-decode", "What copying does to real recommendations",
        "Change in exact-SID HR@10 when the copy reads are blocked inside beam-search decoding (50 beams)",
        fx, [(c, n, "dot") for _, c, n in QS],
        table(["stratum", "condition", "SID", "Δ HR@10 (pp)", "95% CI", "rows"], trows, num_from=3),
        "exp6, AR test split. Repeat = the target's SID is already in the user's history. Knockouts act in layers 14–27 "
        "on each beam's own prefix. Each row is a paired difference from the unmodified run of the same decoder: "
        "the archived decoder (beam sampling, see section 8)"
        + (", or plain beam search for the rows marked so." if have_plain else "."))

    # F7 exp5
    e5 = est(S["exp5"])
    b = e5[(e5.estimand == "B recovery") & (e5.d == 0)]
    ser, trows = [], []
    for g_, nm, cls in (("slot0", "item 1's own tokens", "s-ctx"), ("sep", "separator (predicts item 2)", "s-mq")):
        pts = [(int(r.layer), r.est, r.lo, r.hi) for r in b[b.group == g_].sort_values("layer").itertuples()]
        ser.append({"name": nm, "short": "item 1 tokens" if g_ == "slot0" else "separator", "cls": cls, "pts": pts,
                    "tipfmt": lambda p, nm=nm: f"{f2(p[1], 2)} recovered\nlayer {p[0]} · {nm}"})
        trows += [(nm, p[0], f2(p[1], 3), f"[{f2(p[2], 3)}, {f2(p[3], 3)}]") for p in pts]
    a1 = pick(e5, estimand="A1 item1 m=0 -> slot1", d=0)
    h1 = pick(e5, estimand="hist1 m=0 -> slot1", d=0)
    a2 = pick(e5, estimand="A2 item1 - hist1 (paired) -> slot1", d=0)
    bars = forest([("Replace item 1", {"mq": a1}), ("Replace the most recent history item", {"mq": h1}),
                   ("Difference (paired)", {"mq": a2})], [("mq", "s-mq", "MQ 4×256")], (0, 0.8),
                  "Δ log p of item 2's first digit (nats)")
    figs["next2"] = figure(
        "fig-next2", "Item 2 is built from item 1",
        "Next-two model (MQ 4×256): what item 2's first digit depends on, and where item 1's information flows",
        f'<div class="multi"><div class="panel"><div class="ptitle">What item 2 depends on</div>{bars}</div>'
        f'<div class="panel"><div class="ptitle">Share of the item-1 effect restored, by layer</div>'
        + line_panel("next-two patching", ser, (0, 27), (0, 1.05), "layer", "share restored",
                     xticks=[0, 5, 10, 15, 20, 25, 27], yticks=[0, 0.25, 0.5, 0.75, 1.0]) + "</div></div>",
        [("s-ctx", "item 1's own tokens", "line"), ("s-mq", "separator (predicts item 2)", "line")],
        table(["positions", "layer", "recovery", "95% CI"], trows, num_from=1),
        f"exp5, two-item test split (3,452 rows). Replacing item 1: {f2(a1[0], 2)} nats [{f2(a1[1], 2)}, {f2(a1[2], 2)}]; "
        f"replacing the most recent history item: {f2(h1[0], 2)} [{f2(h1[1], 2)}, {f2(h1[2], 2)}].")

    # F8 DiffGRM: exp7 + exp8 forest
    e7, e8 = est(S["exp7"]), est(S["exp8"])
    rowsD = [
        ("Change the recent item's digits from d on, when it matches the target so far", by_q(e7, estimand="delta | match (pooled)")),
        ("Same change when it does not match", by_q(e7, estimand="delta | nonmatch (pooled)")),
        ("Decoder route: digit d cannot read the item's slot (exp7)", by_q(e7, estimand="H2a K1 | match (pooled d>=1)")),
        ("Decoder route: no digit can read the item's slot (exp8)", by_q(e8, estimand="dec | match (pooled d>=1)")),
        ("Encoder route: other history slots cannot read the item", by_q(e8, estimand="enc | match (pooled d>=1)")),
        ("Both routes blocked", by_q(e8, estimand="H1 both | match (pooled d>=1)")),
        ("Both routes blocked for a non-matching item (control)", by_q(e8, estimand="both_ctrl | match (pooled d>=1)")),
        ("Both minus the sum of the single routes", by_q(e8, estimand="H2 both - (enc + dec) | match (pooled d>=1)")),
    ]
    trows = [(lab, name, f2(v[k][0], 3, True), f"[{f2(v[k][1], 3, True)}, {f2(v[k][2], 3, True)}]", f"{v[k][3]:,}")
             for lab, v in rowsD for k, _, name in QS if k in v]
    figs["diff"] = figure(
        "fig-diff", "DiffGRM copies too, but routes it through the encoder",
        "Cost to the correct digit d ≥ 2 with the golden prefix revealed, in rows where the recent item matches the target so far",
        forest(rowsD, QS, (-0.1, 1.1), "Δ log p (nats)"), [(c, n, "dot") for _, c, n in QS],
        table(["condition", "SID", "Δ nats", "95% CI", "rows"], trows, num_from=2),
        "exp7 and exp8, diffusion cohort (6,297 users), fp32. A slot is one whole history item: DiffGRM encodes each "
        "item as one token, and its single encoder layer mixes items into each other's slots.")

    # ------------------------------------------------------------- numbers --
    k3 = est(S["exp3"])
    first = {Q[v][0]: pick(k3[k3.variant == v], part="A", recency=1, m=0, digit=0, measure="delta_logp_codes") for v in Q}
    flip = {Q[v][0]: pick(k3[k3.variant == v], part="A", recency=1, m=0, digit=0, measure="top1_flip_rate") for v in Q}
    lens_final = {Q[v][0]: pick(er[(er.variant == v) & (er.kind == "lens")], what="recent item's code top-1", layer=28, digit=0) for v in Q}
    link = by_q(e4, estimand="H2a K1 | match (pooled d>=1)")
    both = by_q(e8, estimand="H1 both | match (pooled d>=1)")
    dm = {Q[v][0]: pick(e7[e7.variant == v], estimand="delta S_full", r=1, m=0, digit=0) for v in Q}
    d6 = {(Q[v][0], st): pick(e6[(e6.variant == v) & (e6.metric == "hr10")], estimand="delta", contrast="C_all - B", stratum=st)
          for v in Q for st in ("repeat", "new", "all")}
    lv6 = {(Q[v][0], c): pick(e6[(e6.variant == v) & (e6.metric == "hr10")], estimand="level", condition=c, stratum="all")
           for v in Q for c in ("B", "C_all") + (("B_plain", "C_all_plain") if have_plain else ())}
    rep_lv = {Q[v][0]: pick(e6[(e6.variant == v) & (e6.metric == "hr10")], estimand="level", condition="B", stratum="repeat") for v in Q}
    new_lv = {Q[v][0]: pick(e6[(e6.variant == v) & (e6.metric == "hr10")], estimand="level", condition="B", stratum="new") for v in Q}
    cr = {(Q[v][0], c): pick(e6[(e6.variant == v)], estimand="level", condition=c, stratum="all", metric="top1_is_history")
          for v in Q for c in ("B", "C_all")}
    v6 = json.loads((S["exp6"] / "result.json").read_text())
    nrep = int(e6[(e6.variant == "rqkmeans_3codebook_128") & (e6.estimand == "level") & (e6.condition == "B") &
                  (e6.metric == "hr10") & (e6.stratum == "repeat")].n_rows.iloc[0])
    pp = lambda x: f2(100 * x, 1, True)                                          # noqa: E731

    def pair(d, fmt):
        return f'{fmt(d["rqk"])} <span class="sep">/</span> {fmt(d["rqvae"])}'

    tiles = [
        (pair(first, lambda x: f2(x[0], 2)), "nats", "cost to the AR first digit when the most recent item is replaced", "exp3"),
        (pair(lens_final, lambda x: f"{100 * x[0]:.0f}%"), "", "AR first-digit predictions that copy the most recent item", "logit lens"),
        (pair(link, lambda x: f2(x[0], 2)), "nats", "carried by one attention link, read only in layers 14–27", "exp4"),
        (pair(both, lambda x: f2(x[0], 2)), "nats", "DiffGRM copy removed by blocking its encoder and decoder routes", "exp8"),
    ]
    tile_html = "".join(f'<div class="tile"><div class="tv">{v}<small>{u}</small></div><div class="tl">{esc(l)}</div>'
                        f'<div class="ts">{esc(s)} · RQ-KMeans / RQ-VAE</div></div>' for v, u, l, s in tiles)

    # recency shares removed by keeping the first digit (AR exp3 A3; DiffGRM exp7 m=1 vs m=0)
    keep_ar = {Q[v][0]: 1 - pick(k3[k3.variant == v], part="A3", recency=1, m=1, digit=0,
                                   measure="retained_fraction_all_available")[0] for v in Q}
    keep_d = {}
    for v in Q:
        q7 = e7[(e7.variant == v) & (e7.estimand == "delta S_full") & (e7.r == 1) & (e7.digit == 0)]
        keep_d[Q[v][0]] = 1 - float(q7[q7.m == 1].est.iloc[0]) / float(q7[q7.m == 0].est.iloc[0])
    v1 = json.loads((sorted(S["exp6"].parent.glob("283114/cell-00/validation.json"))[0]).read_text())["V1_reproduction"]
    v1_rows = f"{round(v1['identical_50'] * v1['rows']):,} of {v1['rows']:,} rows with identical 50-SID lists"

    def sig(x):                                   # interval excludes zero?
        return x[1] > 0 or x[2] < 0

    def ppci(x):
        return f"{pp(x[0])} pp [{pp(x[1])}, {pp(x[2])}]"
    rep, new, al = ({k: d6[(k, st)] for k in ("rqk", "rqvae")} for st in ("repeat", "new", "all"))
    direction = lambda x: ("lowers" if x[0] < 0 else "raises") if sig(x) else "does not measurably change"   # noqa: E731
    decode_summary = (
        f"Blocking the copy reads inside the archived beam-search decoder {direction(al['rqk'])} HR@10 for RQ-KMeans "
        f"({ppci(al['rqk'])}) and {direction(al['rqvae'])} it for RQ-VAE ({ppci(al['rqvae'])}). "
        f"On repeat targets, the SID the user already had, the change is {ppci(rep['rqk'])} and {ppci(rep['rqvae'])}; "
        f"on new targets, {ppci(new['rqk'])} and {ppci(new['rqvae'])}.")
    def hr_row(label, a, b):
        return f'<tr><td>{label}</td><td class="n">{a}</td><td class="n">{b}</td></tr>'

    pct = lambda x: f"{100 * x[0]:.1f}%"                                          # noqa: E731
    rows_ = [hr_row("Archived decoder, as run (<code>B</code>)", pct(lv6[("rqk", "B")]), pct(lv6[("rqvae", "B")])),
             hr_row("… repeat targets / new targets", f"{pct(rep_lv['rqk'])} / {pct(new_lv['rqk'])}",
                    f"{pct(rep_lv['rqvae'])} / {pct(new_lv['rqvae'])}"),
             hr_row("Copy reads blocked at every digit (<code>C_all</code>)", pct(lv6[("rqk", "C_all")]),
                    pct(lv6[("rqvae", "C_all")]))]
    if have_plain:
        rows_ += [hr_row("Plain beam search, no sampling or repetition penalty (<code>B_plain</code>)",
                         pct(lv6[("rqk", "B_plain")]), pct(lv6[("rqvae", "B_plain")])),
                  hr_row("… with copy reads blocked (<code>C_all_plain</code>)",
                         pct(lv6[("rqk", "C_all_plain")]), pct(lv6[("rqvae", "C_all_plain")]))]
    rows_.append(hr_row("Top-1 SID already in the history: as run / blocked",
                        f"{pct(cr[('rqk', 'B')])} / {pct(cr[('rqk', 'C_all')])}",
                        f"{pct(cr[('rqvae', 'B')])} / {pct(cr[('rqvae', 'C_all')])}"))
    decode_text = ('<div class="prose-table"><table><thead><tr><th>HR@10, exact SID</th><th>RQ-KMeans 3×128</th>'
                   '<th>RQ-VAE 4×128</th></tr></thead><tbody>' + "".join(rows_) + "</tbody></table></div>")

    # takeaway: where the hits come from, whether any block helps new targets, control, manipulation
    h6 = e6[e6.metric == "hr10"]
    d6x = {(Q[v][0], c, st): pick(h6[h6.variant == v], estimand="delta", contrast=c, stratum=st)
           for v in Q for c in ("C_all - B", "C_later - B", "N_later - B") for st in ("repeat", "new", "all")}
    share = {k: rep_lv[k][0] * rep_lv[k][3] / (lv6[(k, "B")][0] * lv6[(k, "B")][3]) for k in ("rqk", "rqvae")}
    helps_new = [c for (k, c, st), x in d6x.items() if st == "new" and x[1] > 0]
    later_new = {k: d6x[(k, "C_later - B", "new")] for k in ("rqk", "rqvae")}
    ctrl = {k: d6x[(k, "N_later - B", "all")] for k in ("rqk", "rqvae")}
    t = [f"Copying helps, and most of all on repeats. Repeat targets are "
         f"{100 * rep_lv['rqk'][3] / lv6[('rqk', 'B')][3]:.0f}% (RQ-KMeans) and "
         f"{100 * rep_lv['rqvae'][3] / lv6[('rqvae', 'B')][3]:.0f}% (RQ-VAE) of rows, but they supply "
         f"{100 * share['rqk']:.0f}% and {100 * share['rqvae']:.0f}% of the baseline's HR@10 hits.",
         "No block raises new-target HR@10 beyond noise." if not helps_new else
         f"Some blocks raise new-target HR@10 ({', '.join(sorted(set(helps_new)))}).",
         f"Blocking only the later-digit reads costs RQ-VAE's new targets {ppci(later_new['rqvae'])}, so copying a matching "
         f"item's next digit also finds new items that share a prefix with the history; for RQ-KMeans the same block "
         f"gives {ppci(later_new['rqk'])}.",
         "The control, blocking as many non-matching reads, moves HR@10 by "
         + " and ".join(f"{f2(100 * x[0], 2, True)} pp [{f2(100 * x[1], 2, True)}, {f2(100 * x[2], 2, True)}]"
                        for x in (ctrl["rqk"], ctrl["rqvae"])) + ".",
         f"The knockout reaches the output: the share of top-1 SIDs already in the history falls from "
         f"{pct(cr[('rqk', 'B')])} to {pct(cr[('rqk', 'C_all')])} and from {pct(cr[('rqvae', 'B')])} to "
         f"{pct(cr[('rqvae', 'C_all')])}."]
    if have_plain:
        dp = {(Q[v][0], st): pick(h6[h6.variant == v], estimand="delta", contrast="C_all_plain - B_plain", stratum=st)
              for v in Q for st in ("repeat", "new")}
        t.append(f"Under plain beam search, the same block changes repeat-target HR@10 by {ppci(dp[('rqk', 'repeat')])} "
                 f"and {ppci(dp[('rqvae', 'repeat')])}, and new-target HR@10 by {ppci(dp[('rqk', 'new')])} and "
                 f"{ppci(dp[('rqvae', 'new')])}.")
    # any top-1 gain on new targets is stated, so the HR@10 framing cannot hide it
    g1 = e6[(e6.metric == "hr1") & (e6.estimand == "delta") & (e6.stratum == "new") & (e6.lo > 0)]
    for r in g1.itertuples():
        dec = "plain beam search" if "plain" in r.contrast else "the archived decoder"
        what = {"C_all": "copy reads at every digit", "C_later": "copy reads at digits 2+",
                "N_later": "the control reads"}[r.contrast.split(" - ")[0].removesuffix("_plain")]
        t.append(f"One gain does appear, at top-1: under {dec}, blocking {what} raises "
                 f"{Q[r.variant][1]}'s new-target HR@1 by {ppci((r.est, r.lo, r.hi))}. With copying off, the top slot "
                 f"sometimes goes to a correct new item; within the top-10 the loss outweighs it.")
    decode_takeaway = " ".join(t)
    V = {"exp3": verdict("held", "recency decays; first digit carries it"),
         "exp4": verdict("held", "H1–H3 held") + " " + verdict("failed", "a few heads carry it"),
         "rep": verdict("held", "2 of 3") + " " + verdict("failed", "late rise, RQ-VAE"),
         "exp6": V6,
         "exp5": verdict("held", "item 2 depends on item 1"),
         "exp7": verdict("held", "copying") + " " + verdict("failed", "direct link") + " " + verdict("failed", "slower decay"),
         "exp8": verdict("held", "both routes, super-additive") + " " + verdict("partial", "encoder alone not small")}
    from datetime import date
    extra = new_sections(QS)
    subs = {**extra, "tiles": tile_html, "f_recency": figs["recency"], "f_patching": figs["patching"], "f_copy": figs["copy"],
            "f_heads": figs["heads"], "f_lens": figs["lens"], "f_decode": figs["decode"], "f_next2": figs["next2"],
            "f_diff": figs["diff"], "first_rqk": f2(first["rqk"][0], 2), "first_rqvae": f2(first["rqvae"][0], 2),
            "flip_rqk": f"{100 * flip['rqk'][0]:.0f}", "flip_rqvae": f"{100 * flip['rqvae'][0]:.0f}",
            "dm_rqk": f2(dm["rqk"][0], 2), "dm_rqvae": f2(dm["rqvae"][0], 2),
            "keep_ar_rqk": f"{100 * keep_ar['rqk']:.0f}", "keep_ar_rqvae": f"{100 * keep_ar['rqvae']:.0f}",
            "keep_d_rqk": f"{100 * keep_d['rqk']:.0f}", "keep_d_rqvae": f"{100 * keep_d['rqvae']:.0f}",
            "decode_summary": esc(decode_summary), "decode_text": decode_text, "nrep": f"{nrep:,}",
            "nrep_pct": f"{100 * nrep / 3681:.1f}", "v1_rows": esc(v1_rows),
            "plain_sentence": (f"Plain beam search, with no sampling and no repetition penalty, gives HR@10 "
                               f"{100 * lv6[('rqk', 'B_plain')][0]:.1f}% and {100 * lv6[('rqvae', 'B_plain')][0]:.1f}%, against "
                               f"{100 * lv6[('rqk', 'B')][0]:.1f}% and {100 * lv6[('rqvae', 'B')][0]:.1f}% for the archived decoder."
                               if have_plain else "A plain beam-search comparison is being re-run (see exp6, amendment 2)."),
            "decode_takeaway": esc(decode_takeaway),
            "exp6_plain_dir": (f", <code>{esc(args.exp6_plain.name)}/</code> (plain decoder)" if have_plain else ""),
            "exp6_dir": S["exp6"].name, "built": date.today().isoformat(), **{f"v_{k}": v for k, v in V.items()}}
    html_out = PAGE
    for k, v in subs.items():
        html_out = html_out.replace(f"@@{k}@@", v)
    if "@@" in html_out:
        import re
        left = sorted(set(re.findall(r"@@(\w+)@@", html_out)))
        raise RuntimeError(f"unfilled markers: {left}")
    args.out.write_text(html_out)
    print(f"wrote {args.out} ({len(html_out) / 1024:.0f} KB)")
    if args.standalone:
        args.standalone.write_text(standalone(html_out))
        print(f"wrote {args.standalone} (standalone document)")
    return 0


def standalone(page: str) -> str:
    """The same page as a complete document, for opening from disk or sending as a file.

    The published page is a fragment: the artifact host wraps it in a doctype,
    head and small reset (zero body margin, light color-scheme). Opened from disk
    without that wrapper, a browser renders it in quirks mode with an 8px body
    margin, so this adds the same wrapper. Title, font links and styles go in the
    head; everything after the stylesheet is the body.
    """
    cut = page.index("</style>") + len("</style>")
    reset = ("<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}"
             "[hidden]{display:none!important}</style>")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{reset}\n{page[:cut]}\n</head>\n<body>\n{page[cut:]}\n</body>\n</html>\n")


PAGE = Path(__file__).with_name("report_template.html").read_text()
V6 = ""

if __name__ == "__main__":
    raise SystemExit(main())
