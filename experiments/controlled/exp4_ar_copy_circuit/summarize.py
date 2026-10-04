#!/usr/bin/env python
"""Estimates for exp4 (prefix-matched copying): H1-H3, descriptive tables, figures.

Recomputes everything from the per-row tables; all intervals come from one
paired user bootstrap per split (`sidlens.analysis.bootstrap.UserBootstrap`),
so contrasts between conditions and cells stay paired. H3 resamples held-out
users only; the head set S was fixed by run.py on screening users.

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
from sidlens.provenance.hashing import sha256_file                       # noqa: E402

SEED, DRAWS = 20260927, 2000
KCONDS = ("K1_next", "K2_item", "K3_ctrl", "K4_next_early", "K5_next_late")


def load(d: Path) -> dict:
    status = (d / "status.txt").read_text().split()[0]
    if status != "complete":
        raise RuntimeError(f"{d}: status {status!r}")
    c = {"dir": d, "inputs": json.loads((d / "inputs.json").read_text()),
         "validation": json.loads((d / "validation.json").read_text())}
    c["variant"] = c["inputs"]["variant"]
    clean = pd.read_parquet(d / "clean.parquet")[["i", "example_id", "user_id", "digit", "logp_codes"]]
    c["clean"] = clean.rename(columns={"logp_codes": "clean_logp"})
    return c


def with_delta(df: pd.DataFrame, clean: pd.DataFrame, on_digit="d") -> pd.DataFrame:
    """Rows scored at their own target digit, with delta = clean - intervened."""
    x = df[df.digit == df[on_digit]] if on_digit else df
    x = x.merge(clean[["i", "digit", "clean_logp"]], on=["i", "digit"], validate="many_to_one")
    return x.assign(delta=x.clean_logp - x.logp_codes)


def fmt(r: dict) -> str:
    return f"{r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def summarize_cell(c: dict, boot: UserBootstrap) -> list[dict]:
    out, d_ = [], c["dir"]
    tag = {"variant": c["variant"], "split": c["inputs"]["split"]}

    def add(part, name, r, **kw):
        out.append({**tag, "part": part, "estimand": name, **kw, **r})

    # ---- H1 -----------------------------------------------------------------
    if (d_ / "part_r.parquet").exists():
        r = with_delta(pd.read_parquet(d_ / "part_r.parquet"), c["clean"])
        m, nm = r[r.match == True], r[r.match == False]                 # noqa: E712
        add("R", "H1 match - nonmatch (pooled d>=1)", boot.paired_difference((m.user_id, m.delta),
                                                                           (nm.user_id, nm.delta)))
        add("R", "delta | match (pooled)", boot.mean(m.user_id, m.delta))
        add("R", "delta | nonmatch (pooled)", boot.mean(nm.user_id, nm.delta))
        for dd, g in r.groupby("d"):
            gm, gn = g[g.match == True], g[g.match == False]             # noqa: E712
            add("R", "delta | match", boot.mean(gm.user_id, gm.delta), d=int(dd))
            add("R", "delta | nonmatch", boot.mean(gn.user_id, gn.delta), d=int(dd))
    # ---- H2 -----------------------------------------------------------------
    if (d_ / "part_k.parquet").exists():
        kraw = pd.read_parquet(d_ / "part_k.parquet")
        k = with_delta(kraw[kraw.d >= 0], c["clean"])
        k1 = k[(k.condition == "K1_next") & (k.d >= 1)]
        k1m, k1n = k1[k1.match == True], k1[k1.match == False]           # noqa: E712
        add("K", "H2a K1 | match (pooled d>=1)", boot.mean(k1m.user_id, k1m.delta))
        pair = k1m.merge(k[(k.condition == "K3_ctrl")][["i", "d", "delta"]], on=["i", "d"],
                         suffixes=("", "_k3"))
        add("K", "H2b K1 - K3 | match (paired, pooled d>=1)",
            boot.mean(pair.user_id, pair.delta - pair.delta_k3))
        add("K", "H2c K1 match - K1 nonmatch (pooled d>=1)",
            boot.paired_difference((k1m.user_id, k1m.delta), (k1n.user_id, k1n.delta)))
        for (cond, dd, mt), g in k.groupby(["condition", "d", "match"]):
            add("K", f"{cond} | {'match' if mt else 'nonmatch'}", boot.mean(g.user_id, g.delta), d=int(dd))
        for cond in ("K4_next_early", "K5_next_late", "K2_item"):
            g = k[(k.condition == cond) & (k.d >= 1) & (k.match == True)]  # noqa: E712
            add("K", f"{cond} | match (pooled d>=1)", boot.mean(g.user_id, g.delta))
        k6 = with_delta(kraw[kraw.condition == "K6_readout_item"], c["clean"], on_digit=None)
        for dd, g in k6.groupby("digit"):
            add("K", "K6_readout_item (all rows)", boot.mean(g.user_id, g.delta), d=int(dd))
    # ---- H3 -----------------------------------------------------------------
    if (d_ / "part_h_test.parquet").exists():
        heads = json.loads((d_ / "heads.json").read_text())
        t = with_delta(pd.read_parquet(d_ / "part_h_test.parquet"), c["clean"])
        s = t[t.condition == "H_S"][["i", "d", "user_id", "match", "delta"]]
        ctl = t[t.condition.str.startswith("H_C")].groupby(["i", "d"]).delta.mean().rename("ctrl")
        s = s.join(ctl, on=["i", "d"])
        k1all = with_delta(pd.read_parquet(d_ / "part_k.parquet").query("condition == 'K1_next'"),
                           c["clean"])[["i", "d", "delta"]].rename(columns={"delta": "k1"})
        s = s.merge(k1all, on=["i", "d"])
        sm = s[s.d >= 1]
        add("H", "H3 S - mean(control sets) | held-out match (pooled d>=1)",
            boot.mean(sm.user_id, sm.delta - sm.ctrl))
        add("H", "S | held-out match (pooled d>=1)", boot.mean(sm.user_id, sm.delta))
        add("H", "control sets | held-out match (pooled d>=1)", boot.mean(sm.user_id, sm.ctrl))
        add("H", "K1 all heads | held-out match (pooled d>=1)", boot.mean(sm.user_id, sm.k1))
        add("H", "share S / K1 all heads | held-out match (pooled d>=1)",
            boot.ratio_of_means((sm.user_id, sm.delta), (sm.user_id, sm.k1)))
        s0 = s[s.d == 0]
        add("H", "S | held-out d=0 (all rows)", boot.mean(s0.user_id, s0.delta), d=0)
        add("H", "control sets | held-out d=0", boot.mean(s0.user_id, s0.ctrl), d=0)
        add("H", "K1 all heads | held-out d=0", boot.mean(s0.user_id, s0.k1), d=0)
        sets = t[(t.d >= 1) & t.condition.str.startswith("H_C")].groupby("condition").delta.mean()
        out.append({**tag, "part": "H", "estimand": "control set means (descriptive)",
                    "est": float(sets.mean()), "lo": float(sets.min()), "hi": float(sets.max()),
                    "n_rows": int(len(sets)), "S": json.dumps(heads["S"])})
    return out


def figures(res: pd.DataFrame, cells: list[dict], out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sidlens.viz import style

    style.apply()
    made = []
    colors = {"match": style.CATEGORY_SLOTS[0], "nonmatch": style.CATEGORY_SLOTS[3]}
    # Fig 1: H1 and K1 per digit, match vs non-match.
    fig, axes = plt.subplots(2, len(cells), figsize=(3.4 * len(cells), 4.6), squeeze=False)
    for j, c in enumerate(cells):
        v = c["variant"]
        for i, (part, pat, ttl) in enumerate([("R", "delta | {}", "replace digits >= d of k*"),
                                              ("K", "K1_next | {}", "knock out readout(d) -> tok(k*, d)")]):
            ax = axes[i][j]
            for s_, (lab, off) in enumerate([("match", -0.18), ("nonmatch", 0.18)]):
                q = res[(res.variant == v) & (res.part == part) & (res.estimand == pat.format(lab)) & res.d.notna()]
                q = q[q.d >= (1 if part == "R" else 0)].sort_values("d")
                ax.errorbar(q.d + 1 + off, q.est, yerr=[q.est - q.lo, q.hi - q.est], fmt="o",
                            color=colors[lab], ms=style.MARKER, capsize=0, label=lab)
            style.null_line(ax, 0, "")
            ax.set_title(f"{v}\n{ttl}", fontsize=7)
            ax.set_xlabel("output digit")
        axes[0][0].set_ylabel("Δ log p (nats)")
        axes[1][0].set_ylabel("Δ log p (nats)")
    style.legend_outside(axes[0][-1])
    made += [str(p) for p in style.save(fig, out / "fig1_match_vs_nonmatch")]
    # Fig 2: head screen heatmaps + attention to tok(k*, d) (match minus non-match).
    fig, axes = plt.subplots(2, len(cells), figsize=(3.2 * len(cells), 7.0), squeeze=False)
    for j, c in enumerate(cells):
        if not (c["dir"] / "part_h_screen.parquet").exists():
            continue
        sc = pd.read_parquet(c["dir"] / "part_h_screen.parquet")
        grid = sc.groupby(["layer", "head"]).delta.mean().unstack().to_numpy()
        ax = axes[0][j]
        im = ax.imshow(grid, aspect="auto", cmap="Blues", origin="lower")
        heads = json.loads((c["dir"] / "heads.json").read_text())["S"]
        for L, h in heads:
            ax.plot(h, L, marker="s", mfc="none", mec=style.INK, ms=6)
        ax.set_title(f"{c['variant']}\nhead screen: mean Δ (screening users)", fontsize=7)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
        z = np.load(c["dir"] / "attention_map.npz")
        pn, match = z["p_next"], z["match"]                  # (N, n, L, H), (N, n)
        diffs = []
        for d in range(1, pn.shape[1]):
            mm = match[:, d].astype(bool)
            if mm.any() and (~mm).any():
                diffs.append(np.nanmean(pn[mm, d], 0) - np.nanmean(pn[~mm, d], 0))
        ax = axes[1][j]
        im = ax.imshow(np.mean(diffs, 0), aspect="auto", cmap="RdBu_r", origin="lower",
                       vmin=-np.nanmax(np.abs(diffs)), vmax=np.nanmax(np.abs(diffs)))
        for L, h in heads:
            ax.plot(h, L, marker="s", mfc="none", mec=style.INK, ms=6)
        ax.set_title("attention readout(d)->tok(k*,d)\nmatch minus non-match (observational)", fontsize=7)
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
    made += [str(p) for p in style.save(fig, out / "fig2_heads")]
    plt.close("all")
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    cells = [load(d) for d in args.cells]
    if len({c["inputs"]["split"] for c in cells}) != 1:
        raise RuntimeError("summarize one split at a time")
    if any(c["inputs"].get("limited") for c in cells):
        raise RuntimeError("pilot cells are not summarized")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()
    users = np.concatenate([c["clean"].user_id.unique() for c in cells])
    boot = UserBootstrap(users, args.draws, SEED)
    # H3 resamples held-out users only: its estimands have no screening-half rows,
    # so screening users contribute zero weight to them automatically.
    res = pd.DataFrame([r for c in cells for r in summarize_cell(c, boot)])
    res.to_csv(args.out / "estimates.csv", index=False)
    figs = figures(res, cells, args.out / "figures")
    L = [f"# exp4 prefix-matched copying: {cells[0]['inputs']['split']} split", "",
         f"Paired user bootstrap ({len(boot.users):,} users, {args.draws:,} draws, seed {SEED}); "
         "95% percentile intervals exclude training-seed variance. Δ = clean − intervened "
         "golden log-prob over the digit's codes (nats), teacher-forced.", ""]
    for c in cells:
        v = c["validation"]
        L += [f"## {c['variant']}", "",
              f"Controls: {json.dumps(v.get('controls'))}", "",
              f"Matching rows by digit: {c['inputs']['n_match_by_digit']}; "
              f"control (K3) keys by digit: {c['inputs']['n_ctrl_key_by_digit']}", "",
              "| part | estimand | d | estimate [95% CI] | rows | users |", "|---|---|---|---|---|---|"]
        for r in res[res.variant == c["variant"]].itertuples():
            dd = "" if pd.isna(getattr(r, "d", np.nan)) else int(r.d) + 1
            L.append(f"| {r.part} | {r.estimand} | {dd} | {fmt(r._asdict())} | {r.n_rows} | "
                     f"{'' if pd.isna(r.n_users) else int(r.n_users)} |")
        if (c["dir"] / "heads.json").exists():
            L += ["", f"Selected heads S (layer, head): {json.loads((c['dir'] / 'heads.json').read_text())['S']}"]
        L.append("")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells], "split": cells[0]["inputs"]["split"],
        "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": SEED},
        "input_hashes": {str(c["dir"] / f): sha256_file(c["dir"] / f) for c in cells
                         for f in ("clean.parquet", "part_r.parquet", "part_k.parquet", "part_h_screen.parquet",
                                   "part_h_test.parquet", "heads.json", "validation.json")
                         if (c["dir"] / f).exists()},
        "figures": figs}, indent=1))
    print((args.out / "report.md").read_text()[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
