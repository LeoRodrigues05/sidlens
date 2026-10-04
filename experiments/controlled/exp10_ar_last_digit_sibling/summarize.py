#!/usr/bin/env python
"""Estimands for controlled exp10 (protocol.md), from one or more run groups.

    python experiments/controlled/exp10_ar_last_digit_sibling/summarize.py \
        --run test=<group dir> --run valid=<group dir> --out <new dir>

Each group dir holds cell-00 (RQ-KMeans 3x128) and cell-01 (RQ-VAE 4x128).
Refuses to summarise a cell whose no-op controls are not bit-exact or whose
clean scores disagree with exp4 (test split).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sidlens.analysis.bootstrap import UserBootstrap

CELL_NAMES = {0: "rqkmeans_3codebook_128", 1: "rqvae_4codebook_128"}
DRAWS, SEED = 2000, 20260927


def load_cell(d: Path) -> pd.DataFrame:
    v = json.loads((d / "validation.json").read_text())
    for name, c in v["controls"].items():
        if c["n_exact"] != c["n"]:
            raise RuntimeError(f"{d}: control {name} not bit-exact ({c})")
    if "clean_vs_exp4" in v and not v["clean_vs_exp4"]["ok"]:
        raise RuntimeError(f"{d}: clean scores disagree with exp4 ({v['clean_vs_exp4']})")
    s = pd.read_parquet(d / "scores.parquet")
    wide = s.pivot_table(index=["i", "example_id", "user_id", "subset", "nd", "h_is_r1", "h_rebought",
                                "has_control"], columns="condition",
                         values=["lh", "lt", "top1_is_h", "top1_is_t"], aggfunc="first")
    wide.columns = [f"{m}_{c}" for m, c in wide.columns]
    return wide.reset_index()


def estimands(w: pd.DataFrame, tag: dict) -> list[dict]:
    out = []
    boot = UserBootstrap(w.user_id, draws=DRAWS, seed=SEED)
    for m in ("lh", "lt"):
        for k in ("KH_late", "KH_all", "KC_late"):
            w[f"d_{m}_{k}"] = w[f"{m}_{k}"] - w[f"{m}_clean"]
        w[f"p_{m}_late"] = w[f"d_{m}_KH_late"] - w[f"d_{m}_KC_late"]
        w[f"p_{m}_all"] = w[f"d_{m}_KH_all"] - w[f"d_{m}_KC_late"]
    for t in ("top1_is_h", "top1_is_t"):
        w[f"d_{t}_KH_late"] = w[f"{t}_KH_late"].astype(float) - w[f"{t}_clean"].astype(float)
    strata = {"all": np.ones(len(w), bool), "S_same": (w.subset == "S_same").to_numpy(),
              "S_diff": (w.subset == "S_diff").to_numpy(),
              "S_diff nd": ((w.subset == "S_diff") & w.nd).to_numpy(),
              "S_diff not nd": ((w.subset == "S_diff") & ~w.nd).to_numpy(),
              "h rebought": w.h_rebought.to_numpy(), "h not rebought": (~w.h_rebought).to_numpy(),
              "S_diff h=r1": ((w.subset == "S_diff") & w.h_is_r1).to_numpy()}
    ctl = w.has_control.to_numpy()
    for sname, m in strata.items():
        if not m.any():
            continue
        t = {**tag, "stratum": sname}
        mc = m & ctl
        names = {"P1 (lh) KH_late - KC_late": ("p_lh_late", mc), "P2 (lt) KH_late - KC_late": ("p_lt_late", mc),
                 "lh KH_all - KC_late": ("p_lh_all", mc), "lt KH_all - KC_late": ("p_lt_all", mc),
                 "d lh KH_late": ("d_lh_KH_late", m), "d lh KC_late": ("d_lh_KC_late", mc),
                 "d lt KH_late": ("d_lt_KH_late", m), "d lt KC_late": ("d_lt_KC_late", mc),
                 "clean lh": ("lh_clean", m), "clean lt": ("lt_clean", m),
                 "clean top1 = h": ("top1_is_h_clean", m), "clean top1 = t": ("top1_is_t_clean", m),
                 "d top1 = h KH_late": ("d_top1_is_h_KH_late", m),
                 "d top1 = t KH_late": ("d_top1_is_t_KH_late", m)}
        for est, (col, mm) in names.items():
            if not mm.any():
                continue
            out.append({**t, "estimand": est, **boot.mean(w.user_id[mm], w[col][mm].astype(float))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="split=group_dir")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    res, sources = [], {}
    for spec in args.run:
        split, root = spec.split("=", 1)
        for cell, name in CELL_NAMES.items():
            d = Path(root) / f"cell-{cell:02d}"
            w = load_cell(d)
            sources[f"{split}/{name}"] = str(d)
            res += estimands(w, {"split": split, "variant": name})
    est = pd.DataFrame(res)
    est.to_csv(args.out / "estimates.csv", index=False)
    (args.out / "sources.json").write_text(json.dumps(sources, indent=1))
    L = ["# exp10: last-digit sibling read -- generated summary", "",
         "Paired user bootstrap (2,000 draws, seed 20260927), 95% intervals, nats unless a rate; "
         "intervals exclude training-seed variance.", ""]
    for (split, v), g in est.groupby(["split", "variant"], sort=False):
        L += [f"## {v}, {split}", "", "| stratum | estimand | est [95% CI] | rows | users |", "|---|---|---|---|---|"]
        for r in g.itertuples():
            L.append(f"| {r.stratum} | {r.estimand} | {r.est:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | {r.n_rows} | {r.n_users} |")
        L.append("")
    (args.out / "report.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
