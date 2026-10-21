#!/usr/bin/env python
"""Summary of N3 (copy dose-response) across grid runs.

Per run and alpha, from each cell's `scores.parquet`: SID and item (uniform)
Hit@10 on repeat targets (retrospective exp7's `same_item` and `sid_partner`)
and new targets, their change from alpha = 1 (paired by user), and the share
of history SIDs in the top 10. Primary: the change in new-target SID Hit@10 at
alpha = 0.5. Paired user bootstrap per run, 2,000 draws, seed 20261021;
intervals exclude training-seed variance.

    python summarize.py --cells <cell dir> [...] --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import grid                                                  # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                      # noqa: E402

SEED, DRAWS = 20261021, 2000
GROUPS = {"repeat": ("same_item", "sid_partner"), "new": ("new",), "all": None}


def cell_estimates(d: Path) -> list[dict]:
    inp = json.loads((d / "inputs.json").read_text())
    s = pd.read_parquet(d / "scores.parquet")
    s = s[s.eligible]
    tag = {"checkpoint": inp["checkpoint"], **grid.cell_info(inp["checkpoint"])}
    boot = UserBootstrap(s.user_id, draws=DRAWS, seed=SEED)
    w = s.pivot_table(index=["example_id", "user_id", "cls"], columns="alpha",
                      values=["sid_hit", "item_uniform", "history_share"]).reset_index()
    out = []
    for g, classes in GROUPS.items():
        m = w.cls.isin(classes) if classes else pd.Series(True, index=w.index)
        if not m.any():
            continue
        for metric in ("sid_hit", "item_uniform", "history_share"):
            for a in sorted(s.alpha.unique()):
                out.append({**tag, "group": g, "metric": metric, "alpha": a, "estimand": "level",
                            **boot.mean(w.user_id[m], w[(metric, a)][m])})
                if a != 1.0:
                    out.append({**tag, "group": g, "metric": metric, "alpha": a, "estimand": "change from alpha 1",
                                **boot.mean(w.user_id[m], (w[(metric, a)] - w[(metric, 1.0)])[m])})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, skipped = [], []
    for d in args.cells:
        if not (d / "validation.json").exists() or not (d / "scores.parquet").exists():
            skipped.append(str(d))
            continue
        rows += cell_estimates(d)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    L = ["# N3: copy dose-response", "", f"Runs: {len(args.cells) - len(skipped)}; skipped: {len(skipped)}.", "",
         "| dataset | paradigm | map | new-target SID HR@10 change at alpha 0.5 | at alpha 0 | repeat-target change at 0 |",
         "|---|---|---|---|---|---|"]
    if len(est):
        c = est[(est.estimand == "change from alpha 1") & (est.metric == "sid_hit")]
        f = (lambda q: "" if q.empty else f"{q.est.iloc[0]:+.4f} [{q.lo.iloc[0]:+.4f}, {q.hi.iloc[0]:+.4f}]")  # noqa: E731
        for k, g in c.groupby(["dataset", "paradigm", "map"]):
            L.append(f"| {' | '.join(k)} | {f(g[(g.group == 'new') & (g.alpha == 0.5)])} | "
                     f"{f(g[(g.group == 'new') & (g.alpha == 0.0)])} | {f(g[(g.group == 'repeat') & (g.alpha == 0.0)])} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "inputs.json").write_text(json.dumps({"cells": [str(c) for c in args.cells], "skipped": skipped,
                                                      "seed": SEED, "draws": DRAWS}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
