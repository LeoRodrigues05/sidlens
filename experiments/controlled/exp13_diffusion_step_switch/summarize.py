#!/usr/bin/env python
"""Summarize controlled/exp13: F(t), G(t), the F(0) - F(n-1) contrast, HR@10 per schedule.

    python summarize.py --run <run dir> [<run dir> ...] --out <new dir>

Intervals: paired user bootstrap, 2,000 draws, seed 20261014; they exclude
training-seed uncertainty.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402

SEED = 20261014


def cell_info(ckpt: str) -> dict:
    from sidlens import grid
    if grid.is_grid(ckpt):
        return grid.cell_info(ckpt)
    q, depth, width = re.match(r"diff-next1-(\w+)-(\d)cb-(\d+)", ckpt).groups()
    return {"quantizer": q, "depth": int(depth), "width": int(width)}


def markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def greedy_sid(z, name):
    order, codes = z[f"greedy_{name}_order"].astype(np.int64), z[f"greedy_{name}_codes"].astype(np.int64)
    return np.take_along_axis(codes, np.argsort(order, 1), 1)


def summarize_cell(path: Path) -> list[dict]:
    inputs = json.loads((path / "inputs.json").read_text())
    base = {"checkpoint": inputs["checkpoint"], **cell_info(inputs["checkpoint"])}
    W = max(inputs.get("beams", (1, 64)))              # the wide beam: 64 (frozen) or the grid run's
    n = inputs["n_digit"]
    users = pd.read_csv(path / "users.csv.gz")
    z = np.load(path / "decodes.npz")
    boot = UserBootstrap(users.user_index.values, draws=2000, seed=SEED)
    elig = users.eligible.values
    rows = []
    for beam in (W, 1):
        if beam == W:
            top = {s: z[f"b{W}_{s}_codes"][:, 0].astype(np.int64) for s in inputs["schedules"]}
        else:
            top = {s: greedy_sid(z, s) for s in inputs["schedules"]}

        def eq(a, b):
            return (top[a] == top[b]).all(1)
        disc = elig & ~eq("none", "all")
        u = users.user_index.values[disc]
        rows.append({**base, "beam": beam, "estimand": "discordant_share", "t": -1,
                     **boot.mean(users.user_index.values[elig], disc[elig].astype(float))})
        F = {t: eq(f"only_{t}", "all")[disc].astype(float) for t in range(n)}
        G = {t: eq(f"but_{t}", "none")[disc].astype(float) for t in range(n)}
        for t in range(n):
            rows.append({**base, "beam": beam, "estimand": "F", "t": t, **boot.mean(u, F[t])})
            rows.append({**base, "beam": beam, "estimand": "G", "t": t, **boot.mean(u, G[t])})
            changed = (~eq(f"only_{t}", "none"))[disc]
            rows.append({**base, "beam": beam, "estimand": "only_t_changed_top1", "t": t, **boot.mean(u, changed.astype(float))})
        rows.append({**base, "beam": beam, "estimand": "F0_minus_Flast", "t": -1, **boot.mean(u, F[0] - F[n - 1])})
        rows.append({**base, "beam": beam, "estimand": "F_sum_minus_1", "t": -1,
                     **boot.mean(u, sum(F.values()) - 1)})
        if beam == 1:
            o_none, c_none = z["greedy_none_order"], z["greedy_none_codes"]
            for t in range(n):
                o_t, c_t = z[f"greedy_only_{t}_order"], z[f"greedy_only_{t}_codes"]
                changed = disc & ~eq(f"only_{t}", "none")
                pos = changed & (o_t[:, t] != o_none[:, t])
                code = changed & (o_t[:, t] == o_none[:, t]) & (c_t[:, t] != c_none[:, t])
                uu = users.user_index.values[changed]
                if changed.sum():
                    rows.append({**base, "beam": 1, "estimand": "changed_by_position", "t": t, **boot.mean(uu, pos[changed].astype(float))})
                    rows.append({**base, "beam": 1, "estimand": "changed_by_code", "t": t, **boot.mean(uu, code[changed].astype(float))})
    ue = users.user_index.values[elig]
    for s in inputs["schedules"]:
        rows.append({**base, "beam": W, "estimand": "sid_hr10", "t": s, **boot.mean(ue, (users[f"b{W}_{s}_sid_rank"].values[elig] > 0).astype(float))})
        rows.append({**base, "beam": 1, "estimand": "greedy_sid_hit", "t": s, **boot.mean(ue, users[f"b1_{s}_greedy_hit"].values[elig].astype(float))})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, nargs="*", default=[], help="run dirs holding cell-*/")
    ap.add_argument("--cells", type=Path, nargs="*", default=[], help="cell dirs (grid layout)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, skipped, cells = [], [], []
    for path in [p for run in args.run for p in sorted(run.glob("cell-*"))] + list(args.cells):
        if not (path / "validation.json").exists():
            skipped.append(str(path))
            continue
        r = summarize_cell(path)
        rows += r
        cells.append(r[0]["checkpoint"])
        print(f"{path}: {len(r)} estimates", flush=True)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    (args.out / "inputs.json").write_text(json.dumps({"runs": [str(r) for r in args.run], "cells": cells,
                                                      "skipped": skipped, "seed": SEED}, indent=1))
    table = []
    for ck in cells:
        e = est[(est.checkpoint == ck) & (est.beam == est[est.checkpoint == ck].beam.max())]
        i = cell_info(ck)
        F = e[e.estimand == "F"].sort_values("t")
        c = e[e.estimand == "F0_minus_Flast"].iloc[0]
        table.append({"cell": f"{i['quantizer']} {i['depth']}x{i['width']}",
                      "discordant": f"{e[e.estimand == 'discordant_share'].iloc[0].est:.2f}",
                      "F(t), t = 0..n-1": " / ".join(f"{v:.2f}" for v in F.est),
                      "F(0) - F(last)": f"{c.est:.2f} [{c.lo:.2f}, {c.hi:.2f}]"})
    lines = ["# controlled/exp13 summary (beam 64)", "", f"Cells: {len(cells)}; skipped: {len(skipped)}.", "",
             markdown(pd.DataFrame(table)), ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
