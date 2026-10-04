#!/usr/bin/env python
"""Estimates, paired user-bootstrap intervals and figures for exp3 (AR history patching).

Recomputes every number from the per-row tables the cells wrote; nothing is
read from a cell's own summaries except its validation record.

Why every estimand is a ratio of per-user sums
----------------------------------------------
"Mean over rows" and "ratio of sums over rows" both reduce to
sum_u w_u * num_u / sum_u w_u * den_u, where u is a user and w_u is how often
the bootstrap draw picked that user. One (draws x users) weight matrix then
serves every estimand, which is exactly what "all rows, conditions and cells of
a sampled user stay together" requires: two estimands' intervals come from the
same resampled cohorts, so their difference is paired.

    python summarize.py --cells <run>/cell-00 <run>/cell-01 --out <run>/../summary-<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.provenance.hashing import sha256_file                       # noqa: E402

SEED, DRAWS = 20260927, 2000
GROUPS = ("item", "between", "header", "target")


class Boot:
    """Paired cluster bootstrap over users shared by every estimand."""

    def __init__(self, users: np.ndarray, draws: int, seed: int):
        self.users = np.sort(np.unique(users))
        self.index = {u: i for i, u in enumerate(self.users)}
        rng = np.random.default_rng(seed)
        n = len(self.users)
        self.W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)])

    def ratio(self, user: pd.Series, num: pd.Series, den: pd.Series) -> dict:
        """Point estimate and percentile 95% interval of sum(num) / sum(den)."""
        u = user.map(self.index).to_numpy()
        nu = np.bincount(u, weights=num.to_numpy(float), minlength=len(self.users))
        de = np.bincount(u, weights=den.to_numpy(float), minlength=len(self.users))
        if de.sum() == 0:
            return {"est": np.nan, "lo": np.nan, "hi": np.nan, "n_rows": int(len(num))}
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (self.W @ nu) / (self.W @ de)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return {"est": float(nu.sum() / de.sum()), "lo": float(lo), "hi": float(hi),
                "n_rows": int(len(num)), "n_users": int((de != 0).sum())}

    def mean(self, user, x) -> dict:
        return self.ratio(user, x, pd.Series(np.ones(len(x)), index=x.index))

    def ratio_of_means(self, user_a, xa, user_b, xb) -> dict:
        """mean(xa) / mean(xb) with both row counts resampled with the users."""
        def sums(user, x):
            u = user.map(self.index).to_numpy()
            return (np.bincount(u, weights=x.to_numpy(float), minlength=len(self.users)),
                    np.bincount(u, minlength=len(self.users)).astype(float))
        sa, na = sums(user_a, xa)
        sb, nb = sums(user_b, xb)
        if not na.sum() or not nb.sum() or not sb.sum():
            return {"est": np.nan, "lo": np.nan, "hi": np.nan, "n_rows": int(len(xa))}
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = ((self.W @ sa) / (self.W @ na)) / ((self.W @ sb) / (self.W @ nb))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        return {"est": float((sa.sum() / na.sum()) / (sb.sum() / nb.sum())), "lo": float(lo),
                "hi": float(hi), "n_rows": int(len(xa)), "n_users": int((na != 0).sum())}


def load_cell(d: Path, allow_pilot: bool) -> dict:
    status = (d / "status.txt").read_text().split()[0] if (d / "status.txt").exists() else "missing"
    if status != "complete":
        raise RuntimeError(f"{d}: status {status!r}; only complete cells are summarized")
    inputs = json.loads((d / "inputs.json").read_text())
    val = json.loads((d / "validation.json").read_text())
    if inputs.get("limited") and not allow_pilot:
        raise RuntimeError(f"{d}: a pilot (--limit {inputs['limited']}); pass --allow-pilot")
    return {"dir": d, "inputs": inputs, "validation": val, "variant": inputs["variant"],
            "quantizer": inputs["variant"].split("_")[0]}


def part_a(cell: dict, boot: Boot) -> tuple[list[dict], pd.DataFrame]:
    d = cell["dir"]
    clean = pd.read_parquet(d / "clean.parquet")[["example_id", "digit", "logp_codes", "top1"]]
    a = pd.read_parquet(d / "part_a.parquet").merge(
        clean.rename(columns={"logp_codes": "clean_logp", "top1": "clean_top1"}),
        on=["example_id", "digit"], validate="many_to_one")
    a["delta"] = a.clean_logp - a.logp_codes
    a["flip"] = (a.top1 != a.clean_top1).astype(float)
    out = []
    n = cell["inputs"]["n_digits"]
    tag = {"variant": cell["variant"], "quantizer": cell["quantizer"]}
    whole = a.groupby(["example_id", "user_id", "k", "recency", "m"], as_index=False).delta.sum()
    for (r, m), g in a.groupby(["recency", "m"]):
        for dgt, gd in g.groupby("digit"):
            out.append({**tag, "part": "A", "recency": int(r), "m": int(m), "digit": int(dgt),
                        "measure": "delta_logp_codes", **boot.mean(gd.user_id, gd.delta)})
            if r == 1 and m == 0:
                out.append({**tag, "part": "A", "recency": 1, "m": 0, "digit": int(dgt),
                            "measure": "top1_flip_rate", **boot.mean(gd.user_id, gd.flip)})
        gw = whole[(whole.recency == r) & (whole.m == m)]
        out.append({**tag, "part": "A", "recency": int(r), "m": int(m), "digit": -1,
                    "measure": "delta_whole_sid", **boot.mean(gw.user_id, gw.delta)})
    # A3 retained fraction, as declared (all available rows per m) and on the
    # common subset of rows that have a control at every m (sensitivity).
    r1 = a[a.recency == 1]
    full = r1.groupby("example_id").m.nunique()
    common = set(full[full == n].index)
    for dgt in range(n):
        base = r1[(r1.m == 0) & (r1.digit == dgt)]
        for m in range(n):
            gm = r1[(r1.m == m) & (r1.digit == dgt)]
            for subset, gm_, base_ in (("all_available", gm, base),
                                       ("common_rows", gm[gm.example_id.isin(common)],
                                        base[base.example_id.isin(common)])):
                est = boot.ratio_of_means(gm_.user_id, gm_.delta, base_.user_id, base_.delta)
                out.append({**tag, "part": "A3", "recency": 1, "m": m, "digit": dgt,
                            "measure": f"retained_fraction_{subset}", **est})
    cov = a.drop_duplicates(["example_id", "k", "m"]).groupby("m").size()
    miss = pd.read_csv(d / "part_a_no_control.csv")
    coverage = pd.DataFrame({"with_control": cov,
                             "no_control": miss.groupby("m").size()}).fillna(0).astype(int)
    coverage["variant"] = cell["variant"]
    return out, coverage.reset_index()


def part_b(cell: dict, boot: Boot) -> list[dict]:
    b = pd.read_parquet(cell["dir"] / "part_b.parquet")
    base = b[b.condition.isin(["clean", "corrupted"])].pivot_table(
        index=["example_id", "digit"], columns="condition", values="logp_codes")
    p = b[b.condition == "patch"].merge(base.reset_index(), on=["example_id", "digit"])
    p["num"] = p.logp_codes - p.corrupted
    p["den"] = p.clean - p.corrupted
    tag = {"variant": cell["variant"], "quantizer": cell["quantizer"]}
    out = []
    for (L, grp, dgt), g in p.groupby(["layer", "group", "digit"]):
        out.append({**tag, "part": "B", "layer": int(L), "group": grp, "digit": int(dgt),
                    "measure": "recovery", **boot.ratio(g.user_id, g.num, g.den)})
        out.append({**tag, "part": "B", "layer": int(L), "group": grp, "digit": int(dgt),
                    "measure": "mean_patched_minus_corrupted", **boot.mean(g.user_id, g.num)})
        big = g[g.den.abs() >= 0.5]
        out.append({**tag, "part": "B", "layer": int(L), "group": grp, "digit": int(dgt),
                    "measure": "median_row_recovery_den_ge_0.5",
                    "est": float((big.num / big.den).median()) if len(big) else np.nan,
                    "lo": np.nan, "hi": np.nan, "n_rows": int(len(big))})
    # Readout = header + target, each patched separately, summed as declared.
    for (L, dgt), g in p[p.group.isin(["header", "target"])].groupby(["layer", "digit"]):
        h = g[g.group == "header"].set_index("example_id")
        t = g[g.group == "target"].set_index("example_id").reindex(h.index)
        out.append({**tag, "part": "B", "layer": int(L), "group": "header+target", "digit": int(dgt),
                    "measure": "recovery",
                    **boot.ratio(h.user_id, h.num + t.num, h.den)})
    effect = base.reset_index().merge(b[["example_id", "user_id"]].drop_duplicates(), on="example_id")
    effect["d"] = effect.clean - effect.corrupted
    for dgt, g in effect.groupby("digit"):
        out.append({**tag, "part": "B", "layer": -1, "group": "corruption", "digit": int(dgt),
                    "measure": "clean_minus_corrupted", **boot.mean(g.user_id, g.d)})
    return out


def figures(res: pd.DataFrame, out: Path) -> list[str]:
    from sidlens.viz import style
    import matplotlib.pyplot as plt

    style.apply()
    made = []
    cells = list(dict.fromkeys(res.variant))
    ramp = style.DEPTH_RAMP

    # Fig 1: recency profile (m = 0), one panel per cell, one line per digit.
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], cells):
        s = res[(res.variant == v) & (res.part == "A") & (res.measure == "delta_logp_codes") & (res.m == 0)]
        for dgt in sorted(s.digit.unique()):
            q = s[s.digit == dgt].sort_values("recency")
            ax.errorbar(q.recency, q.est, yerr=[q.est - q.lo, q.hi - q.est], color=ramp[dgt + 1],
                        marker="o", ms=style.MARKER, lw=style.LINE_W, capsize=0, label=f"digit {dgt + 1}")
        style.null_line(ax, 0, "no effect")
        ax.set_title(v, fontsize=8)
        ax.set_xlabel("replaced item recency (1 = most recent)")
    axes[0][0].set_ylabel("Δ log p(golden code)  [nats]")
    style.legend_outside(axes[0][-1])
    made += [str(p) for p in style.save(fig, out / "fig1_recency")]

    # Fig 2: prefix sharing at recency 1.
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], cells):
        s = res[(res.variant == v) & (res.part == "A") & (res.measure == "delta_logp_codes") & (res.recency == 1)]
        for dgt in sorted(s.digit.unique()):
            q = s[s.digit == dgt].sort_values("m")
            ax.errorbar(q.m, q.est, yerr=[q.est - q.lo, q.hi - q.est], color=ramp[dgt + 1],
                        marker="o", ms=style.MARKER, lw=style.LINE_W, capsize=0, label=f"digit {dgt + 1}")
        style.null_line(ax, 0, "no effect")
        ax.set_title(v, fontsize=8)
        ax.set_xlabel("leading digits shared with the replaced item (m)")
        ax.set_xticks(sorted(s.m.unique()))
    axes[0][0].set_ylabel("Δ log p(golden code)  [nats]")
    style.legend_outside(axes[0][-1])
    made += [str(p) for p in style.save(fig, out / "fig2_prefix_sharing")]

    # Fig 3: patching recovery by layer; rows = cells, columns = digits.
    if "group" not in res or not (res.part == "B").any():      # a Part-A-only run
        plt.close("all")
        return made
    b = res[(res.part == "B") & (res.measure == "recovery") & (res.group.isin(GROUPS))]
    nd = int(b.digit.max()) + 1
    fig, axes = plt.subplots(len(cells), nd, figsize=(2.5 * nd, 2.2 * len(cells)),
                             sharex=True, sharey=True, squeeze=False)
    for i, v in enumerate(cells):
        for dgt in range(nd):
            ax = axes[i][dgt]
            s = b[(b.variant == v) & (b.digit == dgt)]
            if s.empty:                      # a 3-digit cell in a 4-column grid
                ax.set_axis_off()
                continue
            for j, grp in enumerate(GROUPS):
                q = s[s.group == grp].sort_values("layer")
                if q.empty:
                    continue
                ax.plot(q.layer, q.est, color=style.CATEGORY_SLOTS[j], lw=style.LINE_W, label=grp)
                ax.fill_between(q.layer, q.lo, q.hi, color=style.CATEGORY_SLOTS[j], alpha=0.18, lw=0)
            style.null_line(ax, 0, "")
            style.null_line(ax, 1, "")
            ax.set_title(f"{v}  digit {dgt + 1}", fontsize=7)
            if i == len(cells) - 1:
                ax.set_xlabel("layer L (patched block output)")
        axes[i][0].set_ylabel("recovery")
    # Legend beside the last populated panel; an empty panel has no handles.
    style.legend_outside(next(ax for ax in axes[:, -1] if ax.get_legend_handles_labels()[0]))
    made += [str(p) for p in style.save(fig, out / "fig3_patching_recovery")]
    plt.close("all")
    return made


def md_table(df: pd.DataFrame) -> str:
    """A markdown table without the optional `tabulate` dependency."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def report(res: pd.DataFrame, cells: list[dict], coverage: pd.DataFrame, out: Path,
           boot: Boot, draws: int, seed: int) -> str:
    L = ["# exp3 AR history patching: summary", "",
         f"Recomputed from per-row tables. Paired user bootstrap ({len(boot.users):,} users, "
         f"{draws:,} draws, seed {seed}); 95% percentile intervals exclude training-seed variance and control-draw "
         "variance. Teacher-forced conditional digit scores, not HR@10.", ""]
    L += ["## Cells and numerical controls", "",
          "| variant | rows | users | template | controls exact / n | max |diff| | batch invariance |",
          "|---|---|---|---|---|---|---|"]
    for c in cells:
        v, i = c["validation"], c["inputs"]
        ctrl = v.get("controls", {})
        ex = sum(x["n_exact"] for x in ctrl.values())
        n = sum(x["n"] for x in ctrl.values())
        mx = max((x["max_abs_diff_code_logits"] for x in ctrl.values()), default=float("nan"))
        L.append(f"| {c['variant']} | {i['n_rows']} | {i['n_users']} | {i['template']} | {ex} / {n} "
                 f"| {mx:.3g} | {json.dumps(v.get('batch_invariance', {}))} |")

    def fmt(r):
        return f"{r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"

    L += ["", "## A1: replace the most recent item (m = 0), Δ log p per digit (nats)", "",
          "| variant | digit | Δ log p | top-1 flip rate | rows |", "|---|---|---|---|---|"]
    a = res[res.part == "A"]
    for v in dict.fromkeys(a.variant):
        s = a[(a.variant == v) & (a.recency == 1) & (a.m == 0)]
        for dgt in sorted(s[s.digit >= 0].digit.unique()):
            d1 = s[(s.digit == dgt) & (s.measure == "delta_logp_codes")].iloc[0]
            f1 = s[(s.digit == dgt) & (s.measure == "top1_flip_rate")].iloc[0]
            L.append(f"| {v} | {int(dgt) + 1} | {fmt(d1)} | {f1.est:.3f} | {d1.n_rows} |")
        w = s[s.measure == "delta_whole_sid"].iloc[0]
        L.append(f"| {v} | whole SID | {fmt(w)} | | {w.n_rows} |")
    L += ["", "## A2: recency profile (m = 0), Δ log p per digit", "",
          "| variant | recency | " + " | ".join(f"digit {d + 1}" for d in range(4)) + " |",
          "|---|---|" + "---|" * 4]
    for v in dict.fromkeys(a.variant):
        s = a[(a.variant == v) & (a.m == 0) & (a.measure == "delta_logp_codes")]
        for r in sorted(s.recency.unique()):
            q = s[s.recency == r].set_index("digit")
            L.append(f"| {v} | {int(r)} | " + " | ".join(fmt(q.loc[d]) if d in q.index else "" for d in range(4)) + " |")
    L += ["", "## A3: prefix sharing at recency 1", "",
          "| variant | m | digit | Δ log p | retained fraction (all rows) | retained (common rows) |",
          "|---|---|---|---|---|---|"]
    a3 = res[res.part == "A3"]
    for v in dict.fromkeys(a.variant):
        s = a[(a.variant == v) & (a.recency == 1) & (a.measure == "delta_logp_codes")]
        for _, r in s.sort_values(["m", "digit"]).iterrows():
            ra = a3[(a3.variant == v) & (a3.m == r.m) & (a3.digit == r.digit)].set_index("measure")
            L.append(f"| {v} | {int(r.m)} | {int(r.digit) + 1} | {fmt(r)} | "
                     f"{fmt(ra.loc['retained_fraction_all_available'])} | "
                     f"{fmt(ra.loc['retained_fraction_common_rows'])} |")
    L += ["", "Control coverage (rows x items with / without a prefix-matched control):", "",
          md_table(coverage), ""]
    b = res[(res.part == "B") & (res.measure == "recovery")]
    if len(b):
        L += ["## B: patching recovery by layer (clean -> corrupted, recency-1 item, m = 0)", ""]
        corr = res[(res.part == "B") & (res.group == "corruption")]
        for v in dict.fromkeys(b.variant):
            L += [f"### {v}", "", "Corruption effect (clean − corrupted, nats): " + ", ".join(
                f"digit {int(r.digit) + 1} {fmt(r)}" for r in corr[corr.variant == v].itertuples()), ""]
            s = b[b.variant == v]
            groups = [*GROUPS, "header+target"]
            for dgt in sorted(s.digit.unique()):
                L += [f"digit {int(dgt) + 1}", "", "| layer | " + " | ".join(groups) + " |",
                      "|---|" + "---|" * len(groups)]
                for layer in range(0, int(s.layer.max()) + 1, 3):
                    q = s[(s.digit == dgt) & (s.layer == layer)].set_index("group")
                    L.append(f"| {layer} | " + " | ".join(
                        f"{q.loc[g].est:.2f} [{q.loc[g].lo:.2f}, {q.loc[g].hi:.2f}]" if g in q.index else ""
                        for g in groups) + " |")
                L.append("")
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True, help="must not exist")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-pilot", action="store_true")
    ap.add_argument("--replot", type=Path, default=None,
                    help="redraw figures from an existing summary's estimates.csv into --out; "
                         "recomputes nothing and writes nothing into the source summary")
    args = ap.parse_args(argv)
    if args.replot:
        args.out.mkdir(parents=True, exist_ok=False)
        made = figures(pd.read_csv(args.replot / "estimates.csv"), args.out)
        (args.out / "SOURCE").write_text(f"redrawn from {args.replot / 'estimates.csv'} "
                                         f"sha256 {sha256_file(args.replot / 'estimates.csv')}\n")
        print("\n".join(made))
        return 0
    cells = [load_cell(d, args.allow_pilot) for d in args.cells]
    if len({c["inputs"]["template"] for c in cells}) != 1:
        raise RuntimeError("cells mix templates; summarize eval and sft separately")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()

    users = []
    for c in cells:
        for f in ("clean.parquet", "part_b.parquet"):
            if (c["dir"] / f).exists():
                users.append(pd.read_parquet(c["dir"] / f, columns=["user_id"]).user_id.unique())
                break
    boot = Boot(np.concatenate(users), args.draws, args.seed)
    rows, cov = [], []
    for c in cells:
        if (c["dir"] / "part_a.parquet").exists():
            r, cv = part_a(c, boot)
            rows += r
            cov.append(cv)
        if (c["dir"] / "part_b.parquet").exists():
            rows += part_b(c, boot)
    res = pd.DataFrame(rows)
    res.to_csv(args.out / "estimates.csv", index=False)
    coverage = pd.concat(cov) if cov else pd.DataFrame()
    coverage.to_csv(args.out / "control_coverage.csv", index=False)
    figs = figures(res, args.out / "figures")
    report(res, cells, coverage, args.out, boot, args.draws, args.seed)
    hashes = {str(c["dir"] / f): sha256_file(c["dir"] / f) for c in cells
              for f in ("clean.parquet", "part_a.parquet", "part_b.parquet", "controls.parquet",
                        "validation.json", "inputs.json") if (c["dir"] / f).exists()}
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells], "template": cells[0]["inputs"]["template"],
        "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": args.seed,
                      "interval": "percentile 95%", "excludes": ["training-seed variance",
                                                                 "control-draw variance"]},
        "validation": {str(c["dir"]): {k: c["validation"].get(k) for k in
                                       ("controls", "batch_invariance", "fp32_vs_bfloat16",
                                        "clean_top1_by_digit", "part_a")} for c in cells},
        "input_hashes": hashes, "figures": figs}, indent=1, default=str))
    print((args.out / "report.md").read_text()[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
