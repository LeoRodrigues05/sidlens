#!/usr/bin/env python
"""Summarize controlled/exp14: copy block (PB), order swap (PO1), position shift (PO2).

    python summarize.py --run <run dir> [<run dir> ...] --out <new dir>
    python summarize.py --cells <grid cell dir> [...] --out <new dir>     # grid layout

Intervals: paired user bootstrap, 2,000 draws, seed 20261014; they exclude
training-seed uncertainty. SID and item levels are reported apart.
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
LEVELS = {"sid": "{c}_b{b}_sid_rank", "item_lo": "{c}_b{b}_item_hit_lo", "item_hi": "{c}_b{b}_item_hit_hi"}


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


def hit(users, cond, beam, level):
    v = users[LEVELS[level].format(c=cond, b=beam)].values
    return (v > 0).astype(float) if level == "sid" else v.astype(float)


def summarize_cell(path: Path) -> list[dict]:
    inputs = json.loads((path / "inputs.json").read_text())
    base = {"checkpoint": inputs["checkpoint"], **cell_info(inputs["checkpoint"])}
    users = pd.read_csv(path / "users.csv.gz")
    boot = UserBootstrap(users.user_index.values, draws=2000, seed=SEED)
    uid = users.user_index.values
    rows = []
    for beam in sorted(inputs.get("beams", (64, 256)), reverse=True):     # frozen 256, 64; grid: the run's
        for level in LEVELS:
            clean = hit(users, "clean", beam, level)
            rows.append({**base, "beam": beam, "level": level, "estimand": "hr10_clean", "group": "all", **boot.mean(uid, clean)})
            sel = users.eligible_block_ctrl.values
            d_r1 = hit(users, "block_r1", beam, level) - clean
            d_c = hit(users, "block_ctrl", beam, level) - clean
            rows.append({**base, "beam": beam, "level": level, "estimand": "PB", "group": "L>=2 with control",
                         **boot.mean(uid[sel], (d_r1 - d_c)[sel])})
            rows.append({**base, "beam": beam, "level": level, "estimand": "dHR_block_r1", "group": "L>=2 with control",
                         **boot.mean(uid[sel], d_r1[sel])})
            rows.append({**base, "beam": beam, "level": level, "estimand": "dHR_block_ctrl", "group": "L>=2 with control",
                         **boot.mean(uid[sel], d_c[sel])})
            el = users.eligible_block_r1.values
            for rel in ("repeat", "same_sid", "same_d0", "new"):
                g = el & (users.relation.values == rel)
                if g.sum():
                    rows.append({**base, "beam": beam, "level": level, "estimand": "dHR_block_r1", "group": rel,
                                 **boot.mean(uid[g], d_r1[g])})
                    rows.append({**base, "beam": beam, "level": level, "estimand": "hr10_clean", "group": rel,
                                 **boot.mean(uid[g], clean[g])})
                    rows.append({**base, "beam": beam, "level": level, "estimand": "share_of_users", "group": rel,
                                 **boot.mean(uid[el], g[el].astype(float))})
            for cond in ("swap", "shift1", "shift5"):
                e = users[f"eligible_{cond}"].values
                rows.append({**base, "beam": beam, "level": level, "estimand": f"dHR_{cond}", "group": "eligible",
                             **boot.mean(uid[e], (hit(users, cond, beam, level) - clean)[e])})
        for cond in ("block_r1", "block_ctrl", "swap", "shift1", "shift5"):
            e = users[f"eligible_{cond}"].values
            rows.append({**base, "beam": beam, "level": "sid", "estimand": "top1_changed", "group": cond,
                         **boot.mean(uid[e], (~users[f"{cond}_b{beam}_top1_same_as_clean"].values[e]).astype(float))})
        d = users.eligible_swap.values & (users.r1_d0.values != users.r2_d0.values)
        follow = {}
        for cond in ("clean", "swap"):
            t = users[f"{cond}_b{beam}_top1_d0"].values
            follow[cond] = (t == users.r2_d0.values).astype(float) - (t == users.r1_d0.values).astype(float)
            rows.append({**base, "beam": beam, "level": "sid", "estimand": f"top1_d0_is_r1_{cond}", "group": "r1,r2 differ",
                         **boot.mean(uid[d], (t == users.r1_d0.values)[d].astype(float))})
            rows.append({**base, "beam": beam, "level": "sid", "estimand": f"top1_d0_is_r2_{cond}", "group": "r1,r2 differ",
                         **boot.mean(uid[d], (t == users.r2_d0.values)[d].astype(float))})
        rows.append({**base, "beam": beam, "level": "sid", "estimand": "PO1", "group": "r1,r2 differ",
                     **boot.mean(uid[d], (follow["swap"] - follow["clean"])[d])})
        # Post hoc (added 2026-10-15 after PO1 turned out mis-scaled; see RESULTS.md): among users whose clean top-1
        # first code copies one of the two most recent items, does it follow that item to its new slot, or stay
        # with the slot (now holding the other item)?
        t_clean, t_swap = users[f"clean_b{beam}_top1_d0"].values, users[f"swap_b{beam}_top1_d0"].values
        for src, other, name in (("r1_d0", "r2_d0", "copied_r1"), ("r2_d0", "r1_d0", "copied_r2")):
            g = d & (t_clean == users[src].values)
            if g.sum():
                rows.append({**base, "beam": beam, "level": "sid", "estimand": "posthoc_follow_position", "group": name,
                             **boot.mean(uid[g], (t_swap == users[other].values)[g].astype(float))})
                rows.append({**base, "beam": beam, "level": "sid", "estimand": "posthoc_stay_with_item", "group": name,
                             **boot.mean(uid[g], (t_swap == users[src].values)[g].astype(float))})
    for cond in ("block_r1", "block_ctrl", "swap", "shift1", "shift5"):
        e = users[f"eligible_{cond}"].values
        rows.append({**base, "beam": 0, "level": "-", "estimand": "kl0", "group": cond, **boot.mean(uid[e], users[f"kl0_{cond}"].values[e])})
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
        ck = json.loads((path / "inputs.json").read_text())["checkpoint"]
        if ck in cells:                                      # a later run of the same cell: the first run listed wins
            skipped.append(f"{path} (duplicate of an earlier run)")
            continue
        r = summarize_cell(path)
        rows += r
        cells.append(r[0]["checkpoint"])
        print(f"{path}: {len(r)} estimates", flush=True)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    (args.out / "inputs.json").write_text(json.dumps({"runs": [str(r) for r in args.run], "cells": cells,
                                                      "skipped": skipped, "seed": SEED}, indent=1))

    def f(e, **kw):
        s = e
        for k, v in kw.items():
            s = s[s[k] == v]
        if s.empty:
            return ""
        r = s.iloc[0]
        return f"{r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"

    table = []
    for ck in cells:
        e = est[est.checkpoint == ck]
        i = cell_info(ck)
        B = int(e.beam.max())                                # widest beam: 256 (frozen) or the grid run's
        table.append({"cell": f"{i['quantizer']} {i['depth']}x{i['width']}", "beam": B,
                      "PB sid": f(e, estimand="PB", beam=B, level="sid"),
                      "dHR new": f(e, estimand="dHR_block_r1", beam=B, level="sid", group="new"),
                      "dHR repeat": f(e, estimand="dHR_block_r1", beam=B, level="sid", group="repeat"),
                      "PO1": f(e, estimand="PO1", beam=B),
                      "shift5 top1 changed": f(e, estimand="top1_changed", beam=B, group="shift5")})
    lines = ["# controlled/exp14 summary", "", f"Cells: {len(cells)}; skipped: {len(skipped)}.", "",
             markdown(pd.DataFrame(table)), ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
