#!/usr/bin/env python
"""Estimates for exp7 (DiffGRM history use): A1-A3, H1 (copy), H2 (cross-attention knockout).

Recomputes every estimate from the per-row tables; the `match_d` strata are
rebuilt from the frozen cohort (most recent history item vs target), not read
from the run. One paired user bootstrap over the 6,297 diffusion-cohort users.

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


def load_cell(d: Path) -> dict:
    if (d / "status.txt").read_text().split()[0] != "complete":
        raise RuntimeError(f"{d} is not complete")
    inputs = json.loads((d / "inputs.json").read_text())
    if inputs.get("limited"):
        raise RuntimeError("pilot cells are not summarized")
    cohort = load_eval_cohort(load_runtime()[inputs["checkpoint"]])
    n = inputs["n_digit"]
    L = cohort.history_lengths
    recent = cohort.histories[np.arange(len(cohort)), L - 1]              # (N, n)
    shared = np.zeros(len(cohort), dtype=int)                             # leading digits shared
    for k in range(n):
        shared += np.all(recent[:, :k + 1] == cohort.target_sids[:, :k + 1], axis=1)
    return {"dir": d, "inputs": inputs, "validation": json.loads((d / "validation.json").read_text()),
            "variant": inputs["sem_ids"], "n": n, "shared": shared,
            "clean": pd.read_parquet(d / "clean.parquet"), "a": pd.read_parquet(d / "part_a.parquet"),
            "k": pd.read_parquet(d / "part_k.parquet")}


def fmt(r) -> str:
    return f"{r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    cells = [load_cell(d) for d in args.cells]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()
    boot = UserBootstrap(np.concatenate([c["clean"].user.unique() for c in cells]), args.draws, SEED)
    out = []

    def add(c, part, name, est, **kw):
        out.append({"variant": c["variant"], "part": part, "estimand": name, **kw, **est})

    for c in cells:
        n = c["n"]
        clean = c["clean"].rename(columns={"logp": "clean"})[["user_row", "state", "d", "digit", "clean"]]
        a = c["a"].merge(clean, on=["user_row", "state", "d", "digit"], validate="many_to_one")
        a["delta"] = a.clean - a.logp
        full = a[a.state == "full"]
        for (r, m, dg), g in full.groupby(["r", "m", "digit"]):
            add(c, "A", "delta S_full", boot.mean(g.user, g.delta), r=int(r), m=int(m), digit=int(dg))
        pref = a[(a.state == "pref") & (a.r == 1)]
        for (m, dg), g in pref.groupby(["m", "digit"]):
            add(c, "A", "delta S_pref(d) r=1", boot.mean(g.user, g.delta), r=1, m=int(m), digit=int(dg))
        # H1: r = 1, m = d, S_pref(d), digit d, split by whether k* shares the target's first d digits
        h = pref[pref.m == pref.digit].copy()
        h = h[h.digit >= 1]
        h["match"] = c["shared"][h.user_row.to_numpy()] >= h.digit.to_numpy()
        hm, hn = h[h.match], h[~h.match]
        add(c, "C", "H1 match - nonmatch (pooled d>=1)", boot.paired_difference((hm.user, hm.delta), (hn.user, hn.delta)))
        add(c, "C", "delta | match (pooled)", boot.mean(hm.user, hm.delta))
        add(c, "C", "delta | nonmatch (pooled)", boot.mean(hn.user, hn.delta))
        for dg, g in h.groupby("digit"):
            gm, gn = g[g.match], g[~g.match]
            add(c, "C", "delta | match", boot.mean(gm.user, gm.delta), digit=int(dg))
            add(c, "C", "delta | nonmatch", boot.mean(gn.user, gn.delta), digit=int(dg))
        # H2: knockouts relative to the all-ones baseline of the same (user, d)
        k = c["k"]
        base = k[k.cond == "K0"][["user_row", "d", "logp"]].rename(columns={"logp": "k0"})
        k = k[k.cond != "K0"].merge(base, on=["user_row", "d"], validate="many_to_one")
        k["delta"] = k.k0 - k.logp
        k1 = k[(k.cond == "K1") & (k.d >= 1)]
        k1m, k1n = k1[k1.match], k1[~k1.match]
        add(c, "K", "H2a K1 | match (pooled d>=1)", boot.mean(k1m.user, k1m.delta))
        pair = k1m.merge(k[k.cond == "K3"][["user_row", "d", "delta"]], on=["user_row", "d"], suffixes=("", "_k3"))
        add(c, "K", "H2b K1 - K3 | match (paired, pooled d>=1)", boot.mean(pair.user, pair.delta - pair.delta_k3))
        add(c, "K", "H2c K1 match - K1 nonmatch (pooled d>=1)",
            boot.paired_difference((k1m.user, k1m.delta), (k1n.user, k1n.delta)))
        for (cond, d, mt), g in k.groupby(["cond", "d", "match"]):
            add(c, "K", f"{cond} | {'match' if mt else 'nonmatch'}", boot.mean(g.user, g.delta), digit=int(d))
    res = pd.DataFrame(out)
    res.to_csv(args.out / "estimates.csv", index=False)

    import matplotlib.pyplot as plt
    from sidlens.viz import style
    style.apply()
    figs = []
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cells):
        s = res[(res.variant == c["variant"]) & (res.estimand == "delta S_full") & (res.m == 0)]
        for dg in sorted(s.digit.dropna().unique()):
            q = s[s.digit == dg].sort_values("r")
            ax.errorbar(q.r, q.est, yerr=[q.est - q.lo, q.hi - q.est], color=style.DEPTH_RAMP[int(dg) + 1],
                        marker="o", ms=style.MARKER, lw=style.LINE_W, capsize=0, label=f"digit {int(dg) + 1}")
        style.null_line(ax, 0, "no effect")
        ax.set_title(f"{c['variant']} (DiffGRM, all masked)", fontsize=8)
        ax.set_xlabel("replaced item recency (1 = most recent)")
    axes[0][0].set_ylabel("Δ log p(golden code)  [nats]")
    style.legend_outside(axes[0][-1])
    figs += [str(p) for p in style.save(fig, args.out / "figures" / "fig1_recency")]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cells):
        s = res[(res.variant == c["variant"]) & (res.part == "K") & res.estimand.str.startswith("K4_")
                & res.estimand.str.endswith("| match")]
        for d in sorted(s.digit.dropna().unique()):
            q = s[s.digit == d].assign(layer=lambda x: x.estimand.str.extract(r"K4_(\d)")[0].astype(int)).sort_values("layer")
            ax.errorbar(q.layer, q.est, yerr=[q.est - q.lo, q.hi - q.est], color=style.DEPTH_RAMP[int(d) + 1],
                        marker="o", ms=style.MARKER, lw=style.LINE_W, capsize=0, label=f"digit {int(d) + 1}")
        style.null_line(ax, 0, "")
        ax.set_title(f"{c['variant']}: block digit d → most recent item\nin one decoder block (matching rows)", fontsize=7)
        ax.set_xlabel("decoder block")
        ax.set_xticks(range(4))
    axes[0][0].set_ylabel("Δ log p (nats)")
    style.legend_outside(axes[0][-1])
    figs += [str(p) for p in style.save(fig, args.out / "figures" / "fig2_knockout_by_block")]
    plt.close("all")

    L = ["# exp7: what the DiffGRM networks use from history", "",
         f"Paired user bootstrap ({len(boot.users):,} users, {args.draws:,} draws, seed {SEED}); 95% percentile "
         "intervals exclude training-seed variance. Δ = clean − intervened golden log-prob over the digit's K "
         "codes (nats). S_full = all digits masked; S_pref(d) = golden digits < d revealed.", ""]
    for c in cells:
        v = c["validation"]
        L += [f"## {c['variant']}", "", f"Validation: V2 {v.get('V2_all_ones_mask_is_noop')}, V3 {v.get('V3_batch_invariance')}, "
              f"missing controls {v.get('part_a_no_control')}", "",
              "| part | estimand | r | m | digit | estimate [95% CI] | rows |", "|---|---|---|---|---|---|---|"]
        s = res[res.variant == c["variant"]]
        s = s[~((s.estimand == "delta S_full") & (s.r > 1) & (s.m > 0))]
        for r in s.itertuples():
            g = lambda x: "" if pd.isna(x) else int(x)                                     # noqa: E731
            dig = "" if pd.isna(r.digit) else int(r.digit) + 1
            L.append(f"| {r.part} | {r.estimand} | {g(getattr(r, 'r', np.nan))} | {g(getattr(r, 'm', np.nan))} | "
                     f"{dig} | {fmt(r._asdict())} | {r.n_rows} |")
        L.append("")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells],
        "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": SEED},
        "input_hashes": {str(c["dir"] / f): sha256_file(c["dir"] / f) for c in cells
                         for f in ("clean.parquet", "part_a.parquet", "part_k.parquet", "validation.json")},
        "figures": figs}, indent=1))
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
