#!/usr/bin/env python
"""Summary of N2 across AR grid runs (protocol.md estimands), plus (f) induction vs copy heads.

Per run, from `records.parquet` and `attention.npz`:
    share of support(full) removed by the late key / value replacement (primary)
    attention from the readout to h's digit token, mean over late heads: full - none, suffix - none
    rho with and without the early block, and their difference
    query swap: attention to h2's token and log p(h2's code), minus the full input's
    DLA: share of the summed late-head DLA carried by the top 5 heads
(f), with `--induction` and `--exp4-root`: Spearman correlation across heads of
the base model's induction score with the run's exp4 head-screen copy effect
(`part_h_screen.parquet`, mean delta per head), the exp4 cell found under
`--exp4-root` at the same `<dataset>/<task>/<paradigm>/<map>/<state>` path.

Paired user bootstrap per run, 2,000 draws, seed 20261021; intervals exclude
training-seed variance.

    python summarize.py --cells <cell> [...] [--induction <dir> --exp4-root <dir>] --out <new dir>
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

SEED, DRAWS = 20261021, 2000


def ratio_draws(boot, u, num, den):
    sn, sd = boot._sum(u, np.asarray(num, float)), boot._sum(u, np.asarray(den, float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return sn.sum() / sd.sum(), (boot.W @ sn) / (boot.W @ sd)


def cell_estimates(d: Path, induction=None, exp4_root: Path | None = None) -> list[dict]:
    inp = json.loads((d / "inputs.json").read_text())
    r = pd.read_parquet(d / "records.parquet").sort_values("design")
    z = np.load(d / "attention.npz")
    late = z["late"].tolist()
    tag = {"checkpoint": inp["checkpoint"], **grid.cell_info(inp["checkpoint"])}
    out = []
    for source, m in (("catalog", ~r.synthetic), ("all", pd.Series(True, index=r.index))):
        q = r[m]
        if q.empty:
            continue
        u = q.user_id
        boot = UserBootstrap(u, draws=DRAWS, seed=SEED)
        sup = q.logp_full - q.logp_none

        def add(name, est_bs=None, x=None, **kw):
            if est_bs is not None:
                est, bs = est_bs
                out.append({**tag, "source": source, "estimand": name, **kw,
                            **UserBootstrap._summ(est, bs, len(q), u.nunique())})
            else:
                out.append({**tag, "source": source, "estimand": name, **kw, **boot.mean(u, x)})

        add("support(full)", x=sup)
        add("share removed by late key from suffix", ratio_draws(boot, u, q.logp_full - q.logp_k_from_suffix, sup))
        add("share removed by late value from suffix", ratio_draws(boot, u, q.logp_full - q.logp_v_from_suffix, sup))
        rho = ratio_draws(boot, u, q.logp_suffix - q.logp_none, sup)
        rho_ko = ratio_draws(boot, u, q.logp_suffix_early_ko - q.logp_none_early_ko,
                             q.logp_full_early_ko - q.logp_none_early_ko)
        add("rho", rho)
        add("rho under early block", rho_ko)
        add("rho under early block - rho", (rho_ko[0] - rho[0], rho_ko[1] - rho[1]))
        add("support(full) under early block", x=q.logp_full_early_ko - q.logp_none_early_ko)
        idx = q.design.to_numpy()
        att = {k: np.nanmean(z[f"attn_{k}"][idx][:, late], axis=(1, 2)) for k in ("full", "suffix", "none")}
        add("attention to h, full - none (late mean)", x=att["full"] - att["none"])
        add("attention to h, suffix - none (late mean)", x=att["suffix"] - att["none"])
        dn = q.has_donor.to_numpy()
        if dn.any():
            a_sw = np.nanmean(z["attn_qswap_h2"][idx][:, late], axis=(1, 2))
            a_fu = np.nanmean(z["attn_full_h2"][idx][:, late], axis=(1, 2))
            out.append({**tag, "source": source, "estimand": "query swap: attention to h2 - full",
                        **boot.mean(u[dn], (a_sw - a_fu)[dn])})
            out.append({**tag, "source": source, "estimand": "query swap: log p(h2 code) - full",
                        **boot.mean(u[dn], (q.logp_qswap_h2 - q.logp_full_h2)[dn])})
            out.append({**tag, "source": source, "estimand": "query swap: log p(h code) - full",
                        **boot.mean(u[dn], (q.logp_qswap_h - q.logp_full)[dn])})
        dla = np.nanmean(z["dla"][idx], axis=0)                     # [late, H]
        flat = np.sort(dla.ravel())[::-1]
        out.append({**tag, "source": source, "estimand": "DLA share of top 5 late heads",
                    "est": float(flat[:5].sum() / dla.sum()) if dla.sum() else np.nan, "n_rows": len(q)})
    if induction is not None and exp4_root is not None:
        e4 = exp4_root / grid.map_dir(inp["checkpoint"]) / "part_h_screen.parquet"
        if e4.exists():
            from scipy.stats import spearmanr
            sc = pd.read_parquet(e4).groupby(["layer", "head"]).delta.mean()
            ind = induction["induction"]
            x = [ind[L, h] for (L, h) in sc.index]
            rho_s = spearmanr(x, sc.values)
            out.append({**tag, "source": "heads", "estimand": "Spearman(base induction score, exp4 copy effect)",
                        "est": float(rho_s.statistic), "p_value": float(rho_s.pvalue), "n_rows": len(sc)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--induction", type=Path, help="induction.py output dir")
    ap.add_argument("--exp4-root", type=Path, help="results root of controlled exp4 (grid layout)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    ind = np.load(args.induction / "induction.npz") if args.induction else None
    rows, skipped = [], []
    for d in args.cells:
        if not (d / "validation.json").exists():
            skipped.append(str(d))
            continue
        rows += cell_estimates(d, ind, args.exp4_root)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    L = ["# N2: how the AR model matches the prefix", "", f"Runs: {len(args.cells) - len(skipped)}; "
         f"skipped: {len(skipped)}.", "", "| dataset | map | key removes | value removes | rho early-block change |",
         "|---|---|---|---|---|"]
    if len(est):
        f = (lambda g: "" if g.empty else f"{g.est.iloc[0]:.3f} [{g.lo.iloc[0]:.3f}, {g.hi.iloc[0]:.3f}]")  # noqa: E731
        c = est[est.source == "catalog"]
        for k, g in c.groupby(["dataset", "map"]):
            L.append(f"| {' | '.join(k)} | {f(g[g.estimand == 'share removed by late key from suffix'])} | "
                     f"{f(g[g.estimand == 'share removed by late value from suffix'])} | "
                     f"{f(g[g.estimand == 'rho under early block - rho'])} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "inputs.json").write_text(json.dumps({"cells": [str(c) for c in args.cells], "skipped": skipped,
                                                      "induction": str(args.induction), "exp4_root": str(args.exp4_root),
                                                      "seed": SEED, "draws": DRAWS}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
