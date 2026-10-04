#!/usr/bin/env python
"""Estimates for exp8 (DiffGRM copy route): H1 (both routes blocked), H2 (super-additivity).

`match_d` is rebuilt from the frozen cohort. One paired user bootstrap.

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

from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

SEED, DRAWS = 20260927, 2000
CONDS = ("enc", "dec", "both", "both_ctrl")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    cells = []
    for d in args.cells:
        if (d / "status.txt").read_text().split()[0] != "complete":
            raise RuntimeError(f"{d} is not complete")
        inp = json.loads((d / "inputs.json").read_text())
        if inp.get("limited"):
            raise RuntimeError("pilot cells are not summarized")
        cohort = load_eval_cohort(load_runtime()[inp["checkpoint"]])
        L = cohort.history_lengths
        recent = cohort.histories[np.arange(len(cohort)), L - 1]
        shared = np.zeros(len(cohort), dtype=int)
        for k in range(inp["n_digit"]):
            shared += np.all(recent[:, :k + 1] == cohort.target_sids[:, :k + 1], axis=1)
        sc = pd.read_parquet(d / "scores.parquet")
        base = sc[sc.cond == "base"][["user_row", "d", "logp"]].rename(columns={"logp": "base"})
        w = sc[sc.cond != "base"].merge(base, on=["user_row", "d"], validate="many_to_one")
        w["delta"] = w.base - w.logp
        w["match"] = shared[w.user_row.to_numpy()] >= w.d.to_numpy()
        cells.append({"dir": d, "inputs": inp, "variant": inp["sem_ids"], "w": w,
                      "validation": json.loads((d / "validation.json").read_text())})
    args.out.mkdir(parents=True, exist_ok=False)
    boot = UserBootstrap(np.concatenate([c["w"].user.unique() for c in cells]), args.draws, SEED)
    out = []

    def add(c, name, est, **kw):
        out.append({"variant": c["variant"], "estimand": name, **kw, **est})

    for c in cells:
        w = c["w"]
        piv = w.pivot_table(index=["user_row", "user", "d", "match"], columns="cond", values="delta").reset_index()
        m = piv[(piv.d >= 1) & piv.match]
        add(c, "H1 both | match (pooled d>=1)", boot.mean(m.user, m["both"]))
        mc = m.dropna(subset=["both_ctrl"])
        add(c, "H1 both - both_ctrl | match (paired, pooled d>=1)", boot.mean(mc.user, mc["both"] - mc["both_ctrl"]))
        add(c, "H2 both - (enc + dec) | match (pooled d>=1)", boot.mean(m.user, m["both"] - m["enc"] - m["dec"]))
        for cond in CONDS:
            g = m.dropna(subset=[cond])
            add(c, f"{cond} | match (pooled d>=1)", boot.mean(g.user, g[cond]))
            nm = piv[(piv.d >= 1) & ~piv.match].dropna(subset=[cond])
            add(c, f"{cond} | nonmatch (pooled d>=1)", boot.mean(nm.user, nm[cond]))
            f0 = piv[piv.d == 0].dropna(subset=[cond])
            add(c, f"{cond} | digit 1 at S_full (all rows)", boot.mean(f0.user, f0[cond]), d=0)
            for dd, g2 in piv[(piv.d >= 1) & piv.match].dropna(subset=[cond]).groupby("d"):
                add(c, f"{cond} | match", boot.mean(g2.user, g2[cond]), d=int(dd))
        f0 = piv[piv.d == 0]
        add(c, "H2-style both - (enc + dec) | digit 1 at S_full", boot.mean(f0.user, f0["both"] - f0["enc"] - f0["dec"]), d=0)
    res = pd.DataFrame(out)
    res.to_csv(args.out / "estimates.csv", index=False)
    L = ["# exp8: DiffGRM copy route (encoder spread vs direct decoder read)", "",
         f"Paired user bootstrap ({len(boot.users):,} users, {args.draws:,} draws, seed {SEED}); 95% intervals exclude "
         "training-seed variance. Δ = base − condition golden log-prob (nats).", ""]
    for c in cells:
        L += [f"## {c['variant']}", "", f"Validation: {json.dumps({k: c['validation'][k] for k in ('V2_noop_encoder_mask', 'V2_noop_cross_mask', 'V3_batch_invariance')})}", "",
              "| estimand | digit | estimate [95% CI] | rows |", "|---|---|---|---|"]
        for r in res[res.variant == c["variant"]].itertuples():
            dd = "" if pd.isna(getattr(r, "d", np.nan)) else int(r.d) + 1
            L.append(f"| {r.estimand} | {dd} | {r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.n_rows} |")
        L.append("")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells],
        "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": SEED},
        "input_hashes": {str(c["dir"] / "scores.parquet"): sha256_file(c["dir"] / "scores.parquet") for c in cells}},
        indent=1))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
