#!/usr/bin/env python
"""Estimates for controlled exp1 and exp2 on grid weight states (one or more cells).

`summarize.py` (exp1) and exp2's summary are bound to the 18 frozen cells and
their archive. Grid cells (`run.py --ckpt`, `../exp2_fixed_orders/run.py
--ckpt`) carry the same `predictions.npz`; this reads any number of them and
reports, per cell and beam, at SID and item level (@10):

    level                 every condition's mean
    confidence - fixed    exp1: the seed-42 order; exp2: each user's mean over the six orders
    confidence - grid     the run's own search (exp1 `grid_l2r`; exp2 `grid_fixed_012`)
    fixed - grid_fixed    exp2: matched order, path search vs the run's catalog-legal search

Paired user bootstrap, 2,000 draws, seed 20260909; intervals exclude
training-seed variance.

    python summarize_grid.py --cells <cell dir> [...] --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import grid                                                  # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                      # noqa: E402

SEED, DRAWS = 20260909, 2000
METRICS = ("sid_hit@10", "ndcg@10", "item_lower@10", "item_uniform@10", "item_upper@10")


def labels(z) -> list[str]:
    pol = z["condition_policies"].tolist()
    if "condition_orders" not in z.files:                  # exp1: one fixed (seed-42) order
        return pol
    out = []
    for p, o in zip(pol, z["condition_orders"].tolist()):
        out.append(p if o[0] < 0 else f"{p}_" + "".join(map(str, o)))
    return out


def cell_rows(d: Path) -> list[dict]:
    res = json.loads((d / "result.json").read_text())
    ck = res["checkpoint"]
    tag = {"experiment": res["experiment"], "checkpoint": ck, **(grid.cell_info(ck) if grid.is_grid(ck) else {})}
    with np.load(d / "predictions.npz", allow_pickle=False) as z:
        lab, beams = labels(z), z["condition_beams"].tolist()
        names = z["metric_names"].tolist()
        M = z["metrics"]                                    # [condition, user, metric]
        users = z["users"]
    boot = UserBootstrap(users, draws=DRAWS, seed=SEED)
    idx = {(b, l): i for i, (b, l) in enumerate(zip(beams, lab))}
    out = []
    for b in sorted(set(beams)):
        here = {l: i for (bb, l), i in idx.items() if bb == b}
        for metric in METRICS:
            j = names.index(metric)

            def v(lbls):
                return np.mean([M[here[l], :, j] for l in lbls], axis=0)

            for l, i in here.items():
                out.append({**tag, "beam": b, "metric": metric, "estimand": "level", "condition": l,
                            **boot.mean(users, M[i, :, j])})
            fixed = [l for l in here if l == "fixed" or l.startswith("fixed_")]
            gfix = [l for l in here if l.startswith("grid_fixed_")]
            gref = "grid_l2r" if "grid_l2r" in here else ("grid_fixed_012" if "grid_fixed_012" in here else None)
            pairs = []
            if "confidence" in here and fixed:
                pairs.append(("confidence - fixed", ["confidence"], fixed))
            if "confidence" in here and gref:
                pairs.append(("confidence - grid", ["confidence"], [gref]))
            if fixed and gref:
                pairs.append(("fixed - grid", fixed, [gref] if not gfix else gfix))
            for name, a, c in pairs:
                out.append({**tag, "beam": b, "metric": metric, "estimand": name, "condition": "",
                            **boot.mean(users, v(a) - v(c))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, skipped = [], []
    for d in args.cells:
        if not (d / "result.json").exists() or not (d / "predictions.npz").exists():
            skipped.append(str(d))
            continue
        rows += cell_rows(d)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    L = ["# controlled exp1/exp2 on grid states", "",
         f"Cells: {len(args.cells) - len(skipped)}; skipped: {len(skipped)}. Paired user bootstrap, "
         f"{DRAWS:,} draws, seed {SEED}; intervals exclude training-seed variance.", "",
         "| checkpoint | beam | estimand | SID HR@10 | item HR@10 (uniform) |", "|---|---|---|---|---|"]
    if len(est):
        con = est[est.estimand != "level"]
        for (ck, b, e), g in con.groupby(["checkpoint", "beam", "estimand"], sort=False):
            gm = g.set_index("metric")
            f = lambda m: f"{gm.loc[m, 'est']:+.4f} [{gm.loc[m, 'lo']:+.4f}, {gm.loc[m, 'hi']:+.4f}]"  # noqa: E731
            L.append(f"| {ck} | {b} | {e} | {f('sid_hit@10')} | {f('item_uniform@10')} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "inputs.json").write_text(json.dumps({"cells": [str(c) for c in args.cells], "skipped": skipped,
                                                      "seed": SEED, "draws": DRAWS}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
