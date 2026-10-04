#!/usr/bin/env python
"""Estimates for exp9 (order vs time): the 2x2 position/item effects, flips, tie averaging, Part D.

All estimands are declared in `protocol.md`. Strata were fixed in each cell's
`rows.parquet` before any forward. One paired user bootstrap per split serves
every estimand of both cells, so the difference P2 keeps its pairing.

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

SEED, DRAWS = 20260930, 2000
STRATA = ("U", "T_later", "T_same")


def wide(cell: Path) -> pd.DataFrame:
    """One row per eligible example: clean/swap top-1 at digit 0 and golden log-probs."""
    rows = pd.read_parquet(cell / "rows.parquet")
    tf = pd.read_parquet(cell / "part_tf.parquet")
    tf = tf.merge(rows[["example_id", "eligible"]], on="example_id", validate="many_to_one")
    tf = tf[tf.eligible]
    n = int(tf.digit.max()) + 1
    lp = tf.pivot_table(index="example_id", columns=["condition", "digit"], values="logp_codes", aggfunc="first")
    t1 = tf[tf.digit == 0].pivot_table(index="example_id", columns="condition", values="top1", aggfunc="first")
    w = rows[rows.eligible].set_index("example_id")
    for c in ("clean", "swap"):
        w[f"top1_{c}"] = t1[c]
        w[f"lp0_{c}"] = lp[(c, 0)]
        w[f"lpsid_{c}"] = sum(lp[(c, d)] for d in range(n))
    if w[["top1_clean", "top1_swap", "lpsid_clean", "lpsid_swap"]].isna().any().any():
        raise ValueError(f"{cell}: an eligible row lacks a clean or swap score")
    w["n_digits"] = n
    return w.reset_index()


def estimands(w: pd.DataFrame, boot: UserBootstrap, tag: dict) -> list[dict]:
    out = []

    def add(stratum, name, r, **kw):
        out.append({**tag, "stratum": stratum, "estimand": name, **r, **kw})

    for s in STRATA + ("all",):
        x = w if s == "all" else w[w.stratum == s]
        dd = x[x.c1 != x.c2]
        a_clean = (dd.top1_clean == dd.c1).astype(float) - (dd.top1_clean == dd.c2)
        a_swap = (dd.top1_swap == dd.c2).astype(float) - (dd.top1_swap == dd.c1)
        A = 0.5 * (a_clean + a_swap)
        B = 0.5 * (((dd.top1_clean == dd.c1).astype(float) + (dd.top1_swap == dd.c1))
                   - ((dd.top1_clean == dd.c2).astype(float) + (dd.top1_swap == dd.c2)))
        add(s, "A position effect (d0)", boot.mean(dd.user_id, A))
        add(s, "B item effect (d0)", boot.mean(dd.user_id, B))
        add(s, "data asymmetry P(t=c1)-P(t=c2)", boot.mean(dd.user_id, (dd.ct == dd.c1).astype(float)
                                                           - (dd.ct == dd.c2)))
        for lab, cond, code in (("clean top1=c1", "clean", "c1"), ("clean top1=c2", "clean", "c2"),
                                ("swap top1=c2", "swap", "c2"), ("swap top1=c1", "swap", "c1")):
            add(s, f"P({lab})", boot.mean(dd.user_id, (dd[f"top1_{cond}"] == dd[code]).astype(float)))
        add(s, "P3 flip rate top1 d0", boot.mean(dd.user_id, (dd.top1_clean != dd.top1_swap).astype(float)))
        add(s, "golden top1 d0 clean", boot.mean(x.user_id, (x.top1_clean == x.ct).astype(float)))
        add(s, "golden top1 d0 swap", boot.mean(x.user_id, (x.top1_swap == x.ct).astype(float)))
        d0 = x.lp0_clean - x.lp0_swap
        dsid = x.lpsid_clean - x.lpsid_swap
        add(s, "delta d0 (clean - swap)", boot.mean(x.user_id, d0))
        add(s, "|delta| d0", boot.mean(x.user_id, d0.abs()))
        add(s, "delta SID (clean - swap)", boot.mean(x.user_id, dsid))
        add(s, "|delta| SID", boot.mean(x.user_id, dsid.abs()))
        mix = np.logaddexp(x.lpsid_clean, x.lpsid_swap) - np.log(2) - x.lpsid_clean
        add(s, "P4 delta_mix SID (tie average - clean)", boot.mean(x.user_id, mix))
        mix0 = np.logaddexp(x.lp0_clean, x.lp0_swap) - np.log(2) - x.lp0_clean
        add(s, "delta_mix d0", boot.mean(x.user_id, mix0))

    def a_series(s):
        dd = w[(w.stratum == s) & (w.c1 != w.c2)]
        A = 0.5 * (((dd.top1_clean == dd.c1).astype(float) - (dd.top1_clean == dd.c2))
                   + ((dd.top1_swap == dd.c2).astype(float) - (dd.top1_swap == dd.c1)))
        return dd.user_id, A

    add("T_later - U", "P2 A(T_later) - A(U)", boot.paired_difference(a_series("T_later"), a_series("U")))
    add("T_same - U", "A(T_same) - A(U)", boot.paired_difference(a_series("T_same"), a_series("U")))

    def mix_series(s):
        x = w[w.stratum == s]
        return x.user_id, np.logaddexp(x.lpsid_clean, x.lpsid_swap) - np.log(2) - x.lpsid_clean

    add("T_later - U", "delta_mix(T_later) - delta_mix(U)",
        boot.paired_difference(mix_series("T_later"), mix_series("U")))
    return out


def part_d(cell: Path, w: pd.DataFrame, boot: UserBootstrap, tag: dict) -> list[dict]:
    p = cell / "part_d.parquet"
    if not p.exists():
        return []
    d = pd.read_parquet(p)
    rows = pd.read_parquet(cell / "rows.parquet").set_index("example_id")
    from sidlens.data import ar_prompts as P
    ex = {e.example_id: e for e in P.load_examples(tag["variant"], "next-item", "test")}
    wide_ = d.pivot_table(index="example_id", columns="condition", values="predict", aggfunc="first")
    x = rows.loc[wide_.index].copy()
    x["target"] = [ex[i].target_sids[0] for i in x.index]
    hit = lambda preds, t, k: float(t in list(preds)[:k])                # noqa: E731
    x["hit10_clean"] = [hit(p, t, 10) for p, t in zip(wide_.clean_plain, x.target)]
    x["hit10_swap"] = [hit(p, t, 10) for p, t in zip(wide_.swap_plain, x.target)]
    x["top10_changed"] = [float(list(a)[:10] != list(b)[:10]) for a, b in zip(wide_.clean_plain, wide_.swap_plain)]
    x["top1_changed"] = [float(list(a)[0] != list(b)[0]) for a, b in zip(wide_.clean_plain, wide_.swap_plain)]
    x = x.reset_index()
    out = []
    for s in STRATA + ("all",):
        q = x[x.eligible] if s == "all" else x[x.eligible & (x.stratum == s)]
        for m in ("hit10_clean", "hit10_swap", "top10_changed", "top1_changed"):
            out.append({**tag, "part": "D", "stratum": s, "estimand": f"D {m}", **boot.mean(q.user_id, q[m])})
        out.append({**tag, "part": "D", "stratum": s, "estimand": "D HR@10 swap - clean",
                    **boot.mean(q.user_id, q.hit10_swap - q.hit10_clean)})
    return out


def figures(res: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sidlens.viz import style

    style.apply()
    made = []
    variants = list(dict.fromkeys(res.variant))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    xs = np.arange(len(STRATA))
    for j, (name, ttl) in enumerate([("A position effect (d0)", "A: preference for the last position"),
                                     ("B item effect (d0)", "B: preference for the later item (r1)"),
                                     ("P4 delta_mix SID (tie average - clean)", "tie average − clean (nats, SID)")]):
        ax = axes[j]
        for k, v in enumerate(variants):
            q = res[(res.variant == v) & (res.estimand == name)].set_index("stratum").loc[list(STRATA)]
            col = style.QUANTIZER_COLOR["rqkmeans" if v.startswith("rqkmeans") else "rqvae"]
            off = (k - 0.5) * 0.22
            ax.errorbar(xs + off, q.est, yerr=[q.est - q.lo, q.hi - q.est], fmt="o", color=col,
                        ms=style.MARKER, capsize=0, label=v)
            if j < 2:
                dq = res[(res.variant == v) & (res.estimand == "data asymmetry P(t=c1)-P(t=c2)")] \
                    .set_index("stratum").loc[list(STRATA)]
                ax.plot(xs + off, dq.est, marker="_", ls="none", color=style.MUTED, ms=9)
        style.null_line(ax, 0, "")
        ax.set_xticks(xs, STRATA)
        ax.set_title(ttl, fontsize=8)
    axes[0].set_ylabel("share of rows (top-1, digit 0)")
    style.legend_outside(axes[-1])
    made += [str(p) for p in style.save(fig, out / "fig1_position_item_mix")]
    plt.close("all")
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    cells, ws = [], []
    for c in args.cells:
        inp = json.loads((c / "inputs.json").read_text())
        val = json.loads((c / "validation.json").read_text())
        ctl = val["controls_identical_sid_swap"]
        if ctl["n"] and ctl["n_exact"] != ctl["n"]:
            raise RuntimeError(f"{c}: identical-SID swap controls not bit-exact: {ctl}")
        cells.append({"dir": c, "variant": inp["variant"], "split": inp["split"], "validation": val})
        ws.append(wide(c))
    splits = {c["split"] for c in cells}
    if len(splits) != 1:
        raise ValueError("summarise one split at a time")
    boot = UserBootstrap(pd.concat([w.user_id for w in ws]), draws=DRAWS, seed=SEED)
    res = []
    for c, w in zip(cells, ws):
        tag = {"variant": c["variant"], "split": c["split"]}
        res += [{**r, "part": "TF"} for r in estimands(w, boot, tag)]
        res += part_d(c["dir"], w, boot, tag)
    res = pd.DataFrame(res)
    res.to_csv(args.out / "estimates.csv", index=False)
    figs = figures(res[res.part == "TF"], args.out / "figures")
    key = ["A position effect (d0)", "B item effect (d0)", "data asymmetry P(t=c1)-P(t=c2)",
           "P3 flip rate top1 d0", "delta d0 (clean - swap)", "|delta| d0", "delta SID (clean - swap)",
           "P4 delta_mix SID (tie average - clean)", "P2 A(T_later) - A(U)",
           "delta_mix(T_later) - delta_mix(U)", "D hit10_clean", "D HR@10 swap - clean", "D top10_changed",
           "D top1_changed"]
    L = [f"# exp9 order vs time: {sorted(splits)[0]} split", "",
         "Paired user bootstrap, 2,000 draws, seed 20260930; intervals exclude training-seed variance.", ""]
    for c in cells:
        v = c["validation"]
        L.append(f"- {c['variant']}: identical-SID swap controls {v['controls_identical_sid_swap']}; "
                 f"V-D0 {v.get('V_D0_exp6_plain_reproduction', 'n/a')}")
    L += ["", "| variant | stratum | estimand | est | 95% CI | rows | users |", "|---|---|---|---|---|---|---|"]
    for _, r in res[res.estimand.isin(key)].iterrows():
        L.append(f"| {r.variant} | {r.stratum} | {r.estimand} | {r.est:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | "
                 f"{r.n_rows} | {r.n_users} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells], "figures": figs, "seed": SEED, "draws": DRAWS,
        "inputs_sha256": {str(p): sha256_file(p) for c in cells for p in sorted(c["dir"].glob("*.parquet"))}},
        indent=1))
    (args.out / "summarize.py.source").write_text(Path(__file__).read_text())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
