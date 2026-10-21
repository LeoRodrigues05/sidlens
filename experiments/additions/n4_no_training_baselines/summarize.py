#!/usr/bin/env python
"""Summary of N4 across grid runs: one table of the declared primaries per run.

Reads the `estimates.csv` of every cell given (each already carries its run's
identity and bootstrap intervals) and writes them together, plus a report of
the two primaries: model minus popularity on new targets (SID HR@10) and the
model's top-1 first-code agreement with the copy rule.

    python summarize.py --cells <cell dir> [...] --out <new dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    frames, skipped = [], []
    for c in args.cells:
        if not (c / "validation.json").exists() or not (c / "estimates.csv").exists():
            skipped.append(str(c))
            continue
        frames.append(pd.read_csv(c / "estimates.csv"))
    est = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    est.to_csv(args.out / "estimates.csv", index=False)
    L = ["# N4: no-training baselines", "", f"Runs: {len(frames)}; skipped: {len(skipped)}. "
         "Paired user bootstrap per run (2,000 draws, seed 20261021); intervals exclude training-seed variance.", "",
         "| dataset | task | paradigm | map | model - pop, SID HR@10, new | top-1 first code = copy rule |",
         "|---|---|---|---|---|---|"]
    if len(est):
        key = ["dataset", "task", "paradigm", "map"]
        for k, g in est.groupby(key, sort=True):
            a = g[(g.estimand == "model - pop, sid_hit@10") & (g.target_class == "new")]
            b = g[(g.estimand == "model top-1 first code = copy rule's") & (g.target_class == "all")]
            f = (lambda r: f"{r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")                      # noqa: E731
            L.append(f"| {' | '.join(map(str, k))} | {f(a.iloc[0]) if len(a) else ''} | "
                     f"{f(b.iloc[0]) if len(b) else ''} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "inputs.json").write_text(json.dumps({"cells": [str(c) for c in args.cells], "skipped": skipped},
                                                     indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
