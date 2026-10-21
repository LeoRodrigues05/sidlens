#!/usr/bin/env python
"""Summary of N1 (prefix conditioning) across grid runs.

Per run and replacement source (catalog / synthetic), from each cell's
`records.parquet`: support(full), support(suffix), rho = mean support(suffix) /
mean support(full) (primary, catalog rows), and the prefix move (the log-prob
gain of h''s digit-d code under `prefix`, and the drop of h's own code). Paired
user bootstrap per run, 2,000 draws, seed 20261021; intervals exclude
training-seed variance. MQ runs are flagged by `quantizer` and reported apart.

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


def cell_estimates(d: Path) -> list[dict]:
    inp = json.loads((d / "inputs.json").read_text())
    r = pd.read_parquet(d / "records.parquet")
    tag = {"checkpoint": inp["checkpoint"], **grid.cell_info(inp["checkpoint"])}
    key = ["example_id", "user_id", "d"]
    w = r.pivot_table(index=key, columns="condition", values=["logp_h", "logp_alt"])
    w.columns = [f"{v}|{c}" for v, c in w.columns]
    # a design's source is that of its suffix replacement (the primary's numerator)
    synth = r[r.condition == "suffix"].set_index(key).synthetic
    w = w.join(synth).reset_index()
    sup_full = w["logp_h|full"] - w["logp_h|none"]
    sup_suffix = w["logp_h|suffix"] - w["logp_h|none"]
    measures = {"support(full)": sup_full, "support(suffix)": sup_suffix,
                "prefix move: h' code gain": w["logp_alt|prefix"] - w["logp_alt|none"],
                "prefix move: h code drop": w["logp_h|prefix"] - w["logp_h|full"]}
    boot = UserBootstrap(w.user_id, draws=DRAWS, seed=SEED)
    out = []
    for source, m in (("catalog", ~w.synthetic), ("synthetic", w.synthetic),
                      ("all", pd.Series(True, index=w.index))):
        if not m.any():
            continue
        for dd in [None] + sorted(w.d[m].unique().tolist()):
            md = m if dd is None else m & (w.d == dd)
            extra = {} if dd is None else {"d": int(dd)}
            for name, x in measures.items():
                out.append({**tag, "source": source, **extra, "estimand": name, **boot.mean(w.user_id[md], x[md])})
            out.append({**tag, "source": source, **extra, "estimand": "rho = support(suffix) / support(full)",
                        **boot.ratio(w.user_id[md], sup_suffix[md], sup_full[md])})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, skipped = [], []
    for d in args.cells:
        if not (d / "validation.json").exists() or not (d / "records.parquet").exists():
            skipped.append(str(d))
            continue
        rows += cell_estimates(d)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    L = ["# N1: prefix conditioning", "", f"Runs: {len(args.cells) - len(skipped)}; skipped: {len(skipped)}.", "",
         "| dataset | paradigm | map | rho (catalog) | support(full) | support(suffix) |", "|---|---|---|---|---|---|"]
    if len(est):
        f = (lambda q: "" if q.empty else f"{q.est.iloc[0]:.3f} [{q.lo.iloc[0]:.3f}, {q.hi.iloc[0]:.3f}]")  # noqa: E731
        c = est[(est.source == "catalog") & est["d"].isna()] if "d" in est else est[est.source == "catalog"]
        for k, g in c.groupby(["dataset", "paradigm", "map"]):
            L.append(f"| {' | '.join(k)} | {f(g[g.estimand.str.startswith('rho =')])} | "
                     f"{f(g[g.estimand == 'support(full)'])} | {f(g[g.estimand == 'support(suffix)'])} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "inputs.json").write_text(json.dumps({"cells": [str(c) for c in args.cells], "skipped": skipped,
                                                      "seed": SEED, "draws": DRAWS}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
