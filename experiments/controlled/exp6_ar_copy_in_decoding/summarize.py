#!/usr/bin/env python
"""Estimates for exp6 (copy knockout in the archived AR decoder): P1-P3, controls, copy rates.

Recomputes ranks from the stored 50-SID lists: exact-SID rank (primary) and
`calc.py`'s rank (title / item-id fallbacks through its last-writer maps), the
latter only to compare the baseline with the recorded archive metrics.
Strata come from the CSV, fixed before decoding. One paired user bootstrap.

    python summarize.py --cells <run>/cell-00 <run>/cell-01 --out <run>/../summary-<id>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402

SEED, DRAWS = 20260927, 2000
KS = (1, 5, 10, 20, 50)


def calc_maps(info_file: Path):
    """calc.py's sid->title and sid->item-id maps (last writer wins, as upstream)."""
    to_title, to_item = {}, {}
    for line in info_file.read_text().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            to_title[parts[0].strip()] = parts[1].strip()
            to_item[parts[0].strip()] = parts[2].strip()
    return to_title, to_item


def calc_rank(pred: list[str], target: str, to_title, to_item) -> int:
    for i, s in enumerate(pred):
        if s == target:
            return i
        if s in to_title and target in to_title and to_title[s] == to_title[target]:
            return i
        if s in to_item and target in to_item and to_item[s] == to_item[target]:
            return i
    return 10**6


def exact_rank(pred: list[str], target: str) -> int:
    try:
        return list(pred).index(target)
    except ValueError:
        return 10**6


def load_cell(d: Path) -> dict:
    if (d / "status.txt").read_text().split()[0] != "complete":
        raise RuntimeError(f"{d} is not complete")
    inputs = json.loads((d / "inputs.json").read_text())
    if inputs.get("limited"):
        raise RuntimeError("pilot cells are not summarized")
    val = json.loads((d / "validation.json").read_text())
    variant = inputs["variant"]
    ex = {e.row: e for e in P.load_examples(variant, "next-item", "test")}
    to_title, to_item = calc_maps(paths.WORK / inputs["info_file"])
    frames = []
    for f in sorted(d.glob("predictions_*.parquet")):
        df = pd.read_parquet(f)
        rows = []
        for r in df.itertuples():
            e = ex[r.row]
            tgt = e.target_sids[0]
            pred = list(r.predict)
            rk = exact_rank(pred, tgt)
            hist = set(e.history_sids)
            recent0 = e.history_sids[-1].split(">")[0]
            rows.append({"row": r.row, "user_id": r.user_id, "condition": r.condition, "rank": rk,
                         "calc_rank": calc_rank(pred, tgt, to_title, to_item),
                         "repeat": tgt in hist, "repeat_item": e.target_item_ids[0] in e.history_item_ids,
                         "top1_is_history": pred[0] in hist,
                         "top1_digit0_is_recent": pred[0].split(">")[0] == recent0,
                         "top10_history_share": np.mean([p in hist for p in pred[:10]]),
                         "top10_valid": sum(p.startswith("<a_") for p in pred[:10])})
        frames.append(pd.DataFrame(rows))
    t = pd.concat(frames, ignore_index=True)
    for k in KS:
        t[f"hr{k}"] = (t["rank"] < k).astype(float)
        t[f"calc_hr{k}"] = (t["calc_rank"] < k).astype(float)
    t["ndcg10"] = np.where(t["rank"] < 10, 1 / np.log2(t["rank"].clip(upper=10**5) + 2), 0.0)
    t["stratum"] = np.where(t.repeat, "repeat", "new")
    return {"dir": d, "inputs": inputs, "validation": val, "variant": variant, "t": t}


def fmt(r) -> str:
    return f"{100 * r['est']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    cells = [load_cell(d) for d in args.cells]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()
    boot = UserBootstrap(np.concatenate([c["t"].user_id.unique() for c in cells]), args.draws, SEED)
    out = []

    def add(c, name, r, **kw):
        out.append({"variant": c["variant"], "estimand": name, **kw, **r})

    for c in cells:
        t = c["t"]
        wide = t.pivot_table(index=["row", "user_id", "stratum", "repeat_item"], columns="condition",
                             values=[f"hr{k}" for k in KS] + ["ndcg10", "top1_is_history",
                                                                "top1_digit0_is_recent", "top10_history_share"])
        wide = wide.reset_index()
        conds = sorted(t.condition.unique())
        for cond in conds:                                              # levels
            for metric in [f"hr{k}" for k in KS] + ["ndcg10", "top1_is_history", "top1_digit0_is_recent",
                                                     "top10_history_share"]:
                g = t[t.condition == cond]
                add(c, "level", boot.mean(g.user_id, g[metric]), condition=cond, metric=metric, stratum="all")
                for s, gs in g.groupby("stratum"):
                    add(c, "level", boot.mean(gs.user_id, gs[metric]), condition=cond, metric=metric, stratum=s)
                gi = g[g.repeat_item]
                add(c, "level", boot.mean(gi.user_id, gi[metric]), condition=cond, metric=metric, stratum="repeat_item")
        pairs = [("C_all", "B"), ("C_later", "B"), ("N_later", "B"), ("C_later", "N_later"),
                 ("C_all_plain", "B_plain"), ("B_plain", "B")]
        for a, b in pairs:
            if a not in conds or b not in conds:
                continue
            for metric in [f"hr{k}" for k in KS] + ["ndcg10"]:
                diff = wide[(metric, a)] - wide[(metric, b)]
                add(c, "delta", boot.mean(wide.user_id, diff), contrast=f"{a} - {b}", metric=metric, stratum="all")
                for s in ("repeat", "new"):
                    m = wide.stratum == s
                    add(c, "delta", boot.mean(wide.user_id[m], diff[m]), contrast=f"{a} - {b}", metric=metric, stratum=s)
                m = wide.repeat_item
                add(c, "delta", boot.mean(wide.user_id[m], diff[m]), contrast=f"{a} - {b}", metric=metric,
                    stratum="repeat_item")
                rep, new = wide.stratum == "repeat", wide.stratum == "new"
                add(c, "interaction", boot.paired_difference((wide.user_id[rep], diff[rep]),
                                                             (wide.user_id[new], diff[new])),
                    contrast=f"{a} - {b}", metric=metric, stratum="repeat - new")
        # V1 archive-equivalent HR for the baseline
        if "B" in conds:
            g = t[t.condition == "B"]
            rec = json.loads((paths.FROZEN_RESULTS / "sweep_metrics" / "next-item" / "metrics" /
                              Path(c["inputs"]["archive"]).name.replace(".predictions.json", ".json")).read_text())["metrics"]
            c["v1_hr"] = {f"HR@{k}": {"recorded": rec.get(f"HR@{k}"), "ours_calc_rule": round(100 * g[f"calc_hr{k}"].mean(), 4),
                                      "ours_exact_sid": round(100 * g[f"hr{k}"].mean(), 4)} for k in (1, 5, 10, 20, 50)}
    res = pd.DataFrame(out)
    res.to_csv(args.out / "estimates.csv", index=False)

    import matplotlib.pyplot as plt
    from sidlens.viz import style
    style.apply()
    figs = []
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 2.8), sharey=True, squeeze=False)
    contrasts = ["C_all - B", "C_later - B", "N_later - B", "C_all_plain - B_plain"]
    for ax, c in zip(axes[0], cells):
        s = res[(res.variant == c["variant"]) & (res.estimand == "delta") & (res.metric == "hr10")]
        for j, st in enumerate(("repeat", "new", "all")):
            q = s[s.stratum == st].set_index("contrast").reindex(contrasts)
            x = np.arange(len(contrasts)) + (j - 1) * 0.25
            ax.errorbar(x, 100 * q.est, yerr=[100 * (q.est - q.lo), 100 * (q.hi - q.est)], fmt="o",
                        color=style.CATEGORY_SLOTS[j], ms=style.MARKER, capsize=0, label=st)
        style.null_line(ax, 0, "")
        ax.set_xticks(range(len(contrasts)))
        ax.set_xticklabels([x.replace(" - ", "\n− ") for x in contrasts], fontsize=6)
        ax.set_title(c["variant"], fontsize=8)
    axes[0][0].set_ylabel("Δ HR@10 (pp)")
    style.legend_outside(axes[0][-1])
    figs += [str(p) for p in style.save(fig, args.out / "figures" / "fig1_delta_hr10")]
    plt.close("all")

    L = ["# exp6: copy knockout in the archived AR decoder (beam 50)", "",
         f"Paired user bootstrap ({len(boot.users):,} users, {args.draws:,} draws, seed {SEED}); 95% percentile "
         "intervals in percentage points; they exclude training-seed variance. Exact-SID HR unless stated.", ""]
    for c in cells:
        v = c["validation"]
        L += [f"## {c['variant']}", "",
              f"V1 reproduction: {json.dumps(v.get('V1_reproduction', {}) | {'differing_rows': len(v.get('V1_reproduction', {}).get('differing_rows', []))})}",
              f"V1 HR (recorded vs ours): {json.dumps(c.get('v1_hr'))}", "",
              "| contrast | stratum | ΔHR@10 (pp) | ΔHR@1 | ΔHR@50 | ΔNDCG@10 |", "|---|---|---|---|---|---|"]
        s = res[(res.variant == c["variant"]) & res.estimand.isin(["delta", "interaction"])]
        for (con, st), g in s.groupby(["contrast", "stratum"], sort=False):
            gm = g.set_index("metric")
            L.append(f"| {con} | {st} | {fmt(gm.loc['hr10'])} | {fmt(gm.loc['hr1'])} | {fmt(gm.loc['hr50'])} | "
                     f"{fmt(gm.loc['ndcg10'])} |")
        L += ["", "Levels (all rows): HR@10 %, top-1 is a history SID %, top-1 first digit = recent item's %", "",
              "| condition | HR@10 | repeat HR@10 | new HR@10 | top1 history | top1 digit0=recent |",
              "|---|---|---|---|---|---|"]
        lv = res[(res.variant == c["variant"]) & (res.estimand == "level")]
        for cond in sorted(lv.condition.unique()):
            q = lv[lv.condition == cond]
            get = lambda m, st: q[(q.metric == m) & (q.stratum == st)].est.iloc[0] * 100   # noqa: E731
            L.append(f"| {cond} | {get('hr10', 'all'):.2f} | {get('hr10', 'repeat'):.2f} | {get('hr10', 'new'):.2f} | "
                     f"{get('top1_is_history', 'all'):.1f} | {get('top1_digit0_is_recent', 'all'):.1f} |")
        n_rep = int(c["t"][c["t"].condition == c["t"].condition.iloc[0]].repeat.sum())
        L += ["", f"Rows: {len(c['t']) // c['t'].condition.nunique()}; repeat {n_rep}.", ""]
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells],
        "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": SEED},
        "v1_hr": {c["variant"]: c.get("v1_hr") for c in cells},
        "input_hashes": {str(f): sha256_file(f) for c in cells for f in sorted(c["dir"].glob("predictions_*.parquet"))},
        "figures": figs}, indent=1, default=str))
    print("\n".join(L[:60]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
