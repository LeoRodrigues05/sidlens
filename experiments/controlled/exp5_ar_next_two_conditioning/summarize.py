#!/usr/bin/env python
"""Estimates for exp5 (next-two conditioning): A1-A3, copy check, patching curves.

Recomputed from per-row tables with one paired user bootstrap
(`sidlens.analysis.bootstrap.UserBootstrap`).

    python summarize.py --cell <run>/cell-00 --out <run>/../summary-<id>
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
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402

SEED, DRAWS = 20260927, 2000
GROUPS = ("slot0", "sep", "slot1")


def fmt(r) -> str:
    return f"{r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    args = ap.parse_args(argv)
    d = args.cell
    if (d / "status.txt").read_text().split()[0] != "complete":
        raise RuntimeError(f"{d} is not complete")
    inputs = json.loads((d / "inputs.json").read_text())
    if inputs.get("limited"):
        raise RuntimeError("pilot cells are not summarized")
    val = json.loads((d / "validation.json").read_text())
    n = inputs["n_digits"]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()

    clean = pd.read_parquet(d / "clean.parquet")[["i", "user_id", "slot", "digit", "logp_codes"]] \
        .rename(columns={"logp_codes": "clean_logp"})
    boot = UserBootstrap(clean.user_id.unique(), args.draws, SEED)
    a = pd.read_parquet(d / "part_a.parquet").merge(clean[["i", "slot", "digit", "clean_logp"]],
                                                    on=["i", "slot", "digit"], validate="many_to_one")
    a["delta"] = a.clean_logp - a.logp_codes
    out = []

    def add(name, r, **kw):
        out.append({"estimand": name, **kw, **r})

    it0 = a[(a.condition == "item1") & (a.m == 0)]
    h0 = a[a.condition == "hist1"]
    for dd in range(n):
        g = it0[(it0.slot == 1) & (it0.digit == dd)]
        add("A1 item1 m=0 -> slot1", boot.mean(g.user_id, g.delta), slot=1, d=dd)
        g2 = h0[(h0.slot == 1) & (h0.digit == dd)]
        add("hist1 m=0 -> slot1", boot.mean(g2.user_id, g2.delta), slot=1, d=dd)
        pr = g.merge(g2[["i", "delta"]], on="i", suffixes=("", "_h"))
        add("A2 item1 - hist1 (paired) -> slot1", boot.mean(pr.user_id, pr.delta - pr.delta_h), slot=1, d=dd)
        g3 = h0[(h0.slot == 0) & (h0.digit == dd)]
        add("hist1 m=0 -> slot0 (reference)", boot.mean(g3.user_id, g3.delta), slot=0, d=dd)
    for (m, dd), g in a[(a.condition == "item1") & (a.slot == 1)].groupby(["m", "digit"]):
        add("A3 item1 m -> slot1", boot.mean(g.user_id, g.delta), slot=1, d=int(dd), m=int(m))
    # copy check: does item 1 already share item 2's first d digits?
    ex = {e.example_id: e for e in P.load_examples(inputs["variant"], "two-item", "test")}
    share = {}
    for eid, e in ex.items():
        s0, s1 = (P.parse_sid(s, n) for s in e.target_sids)
        k = 0
        while k < n and s0[k] == s1[k]:
            k += 1
        share[eid] = k
    a["shared_item1_item2"] = a.example_id.map(share)
    for dd in range(1, n):
        g = a[(a.condition == "item1") & (a.m == dd) & (a.slot == 1) & (a.digit == dd)]
        gm, gn = g[g.shared_item1_item2 >= dd], g[g.shared_item1_item2 < dd]
        add("copy: item1 m=d -> slot1 digit d | item1 prefix = item2 prefix", boot.mean(gm.user_id, gm.delta), slot=1, d=dd)
        add("copy: item1 m=d -> slot1 digit d | prefixes differ", boot.mean(gn.user_id, gn.delta), slot=1, d=dd)
    share_counts = pd.Series(share).value_counts().sort_index().to_dict()

    # Part B recovery
    b = pd.read_parquet(d / "part_b.parquet")
    base = b[b.condition.isin(["clean", "corrupted"])].pivot_table(
        index=["i", "slot", "digit"], columns="condition", values="logp_codes")
    p = b[b.condition == "patch"].merge(base.reset_index(), on=["i", "slot", "digit"])
    p = p[p.slot == 1]
    for (L, grp, dd), g in p.groupby(["layer", "group", "digit"]):
        add("B recovery", boot.ratio(g.user_id, g.logp_codes - g.corrupted, g.clean - g.corrupted),
            slot=1, d=int(dd), layer=int(L), group=grp)
    res = pd.DataFrame(out)
    res.to_csv(args.out / "estimates.csv", index=False)

    import matplotlib.pyplot as plt
    from sidlens.viz import style
    style.apply()
    figs = []
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.3), sharey=True)
    rb = res[res.estimand == "B recovery"]
    for dd in range(n):
        ax = axes[dd]
        for j, grp in enumerate(GROUPS):
            q = rb[(rb.d == dd) & (rb.group == grp)].sort_values("layer")
            ax.plot(q.layer, q.est, color=style.CATEGORY_SLOTS[j], lw=style.LINE_W, label=grp)
            ax.fill_between(q.layer, q.lo, q.hi, color=style.CATEGORY_SLOTS[j], alpha=0.18, lw=0)
        style.null_line(ax, 0, ""); style.null_line(ax, 1, "")
        ax.set_title(f"item 2, digit {dd + 1}", fontsize=7)
        ax.set_xlabel("layer L")
    axes[0].set_ylabel("recovery")
    style.legend_outside(axes[-1])
    figs += [str(x) for x in style.save(fig, args.out / "figures" / "fig1_item1_patching")]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    r3 = res[res.estimand == "A3 item1 m -> slot1"]
    for dd in range(n):
        q = r3[r3.d == dd].sort_values("m")
        ax.errorbar(q.m, q.est, yerr=[q.est - q.lo, q.hi - q.est], color=style.DEPTH_RAMP[dd + 1],
                    marker="o", ms=style.MARKER, lw=style.LINE_W, capsize=0, label=f"item 2 digit {dd + 1}")
    style.null_line(ax, 0, "no effect")
    ax.set_xlabel("leading digits the clamped item 1 shares with the true item 1 (m)")
    ax.set_ylabel("Δ log p (nats)")
    style.legend_outside(ax)
    figs += [str(x) for x in style.save(fig, args.out / "figures" / "fig2_item1_prefix")]
    plt.close("all")

    L = ["# exp5 next-two conditioning (two-item_best, MQ 4x256)", "",
         f"Paired user bootstrap ({len(boot.users):,} users, {args.draws:,} draws, seed {SEED}); 95% "
         "percentile intervals exclude training-seed variance. Δ = clean − intervened golden "
         "log-prob over the digit's codes (nats), teacher-forced on 'sid1 ||| sid2'.", "",
         f"Controls: {json.dumps(val.get('controls'))}; batch invariance {val.get('batch_invariance_clean_vs_part_a')}",
         f"Rows by leading digits shared between item 1 and item 2: {share_counts}", "",
         "| estimand | slot | digit | m | layer | group | estimate [95% CI] | rows |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res[res.estimand != "B recovery"].itertuples():
        L.append(f"| {r.estimand} | {r.slot + 1} | {int(r.d) + 1} | {'' if pd.isna(getattr(r, 'm', np.nan)) else int(r.m)} "
                 f"| | | {fmt(r._asdict())} | {r.n_rows} |")
    for r in rb[rb.layer % 3 == 0].itertuples():
        L.append(f"| B recovery | 2 | {int(r.d) + 1} | | {int(r.layer)} | {r.group} | {fmt(r._asdict())} | {r.n_rows} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cell": str(d), "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": SEED},
        "input_hashes": {f: sha256_file(d / f) for f in ("clean.parquet", "part_a.parquet", "part_b.parquet",
                                                          "validation.json")},
        "share_counts": share_counts, "figures": figs}, indent=1, default=str))
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
