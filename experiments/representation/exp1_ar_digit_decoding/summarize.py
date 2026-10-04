#!/usr/bin/env python
"""Layer curves for exp1 (AR digit decoding): logit lens and probes, with user-bootstrap intervals.

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
        cells.append({"dir": d, "inputs": inp, "variant": inp["variant"],
                      "validation": json.loads((d / "validation.json").read_text()),
                      "lens": pd.read_parquet(d / "logit_lens.parquet"),
                      "probes": pd.read_parquet(d / "probes.parquet")})
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "figures").mkdir()
    boot = UserBootstrap(np.concatenate([c["lens"].user_id.unique() for c in cells]), args.draws, SEED)
    rows = []
    for c in cells:
        v = c["variant"]
        for (layer, d), g in c["lens"].groupby(["layer", "digit"]):
            rows.append({"variant": v, "kind": "lens", "what": "golden top-1", "layer": int(layer), "digit": int(d),
                         **boot.mean(g.user_id, (g.golden_rank == 0).astype(float))})
            rows.append({"variant": v, "kind": "lens", "what": "golden log-prob", "layer": int(layer), "digit": int(d),
                         **boot.mean(g.user_id, g.golden_logp)})
            if d == 0:
                rows.append({"variant": v, "kind": "lens", "what": "recent item's code top-1", "layer": int(layer),
                             "digit": 0, **boot.mean(g.user_id, (g.recent_rank == 0).astype(float))})
        pr = c["probes"]
        for (probe, kind, layer), g in pr.groupby(["probe", "kind", "layer"]):
            rows.append({"variant": v, "kind": kind, "what": probe, "layer": int(layer), "digit": -1,
                         **boot.mean(g.user_id, g.correct)})
        # descriptive: P1 split by whether the target's first digit equals the recent item's
        p1 = pr[(pr.probe == "P1_target_d0_at_readout0") & (pr.kind == "probe")]
        cp = pr[(pr.kind == "copy_recent_d0")][["example_id", "pred"]].rename(columns={"pred": "recent"})
        p1 = p1.merge(cp, on="example_id")
        for same, g0 in p1.groupby(p1.label == p1.recent):
            for layer, g in g0.groupby("layer"):
                rows.append({"variant": v, "kind": "probe_split",
                             "what": f"P1 given target d0 {'==' if same else '!='} recent d0",
                             "layer": int(layer), "digit": -1, **boot.mean(g.user_id, g.correct)})
    res = pd.DataFrame(rows)
    res.to_csv(args.out / "estimates.csv", index=False)

    import matplotlib.pyplot as plt
    from sidlens.viz import style
    style.apply()
    figs = []
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cells):
        s = res[(res.variant == c["variant"]) & (res.kind == "lens")]
        for d in sorted(s.digit.unique()):
            q = s[(s.what == "golden top-1") & (s.digit == d)].sort_values("layer")
            ax.plot(q.layer, q.est, color=style.DEPTH_RAMP[d + 1], lw=style.LINE_W, label=f"target digit {d + 1}")
            ax.fill_between(q.layer, q.lo, q.hi, color=style.DEPTH_RAMP[d + 1], alpha=0.2, lw=0)
        q = s[s.what == "recent item's code top-1"].sort_values("layer")
        ax.plot(q.layer, q.est, color=style.CATEGORY_SLOTS[3], lw=style.LINE_W, ls="--",
                label="recent item's digit 1 (at readout 1)")
        ax.set_title(f"{c['variant']}: logit lens", fontsize=8)
        ax.set_xlabel("layer (28 = final norm)")
    axes[0][0].set_ylabel("top-1 rate")
    style.legend_outside(axes[0][-1])
    figs += [str(x) for x in style.save(fig, args.out / "figures" / "fig1_logit_lens")]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.4 * len(cells), 2.6), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cells):
        s = res[(res.variant == c["variant"]) & (res.kind == "probe")]
        for j, p in enumerate(sorted(s.what.unique())):
            q = s[s.what == p].sort_values("layer")
            ax.plot(q.layer, q.est, color=style.CATEGORY_SLOTS[j], lw=style.LINE_W, label=p)
            ax.fill_between(q.layer, q.lo, q.hi, color=style.CATEGORY_SLOTS[j], alpha=0.2, lw=0)
            sh = res[(res.variant == c["variant"]) & (res.kind == "shuffled") & (res.what == p)].sort_values("layer")
            ax.plot(sh.layer, sh.est, color=style.CATEGORY_SLOTS[j], lw=1, ls=":", marker=".")
        ax.set_title(f"{c['variant']}: probes (dotted = shuffled labels)", fontsize=8)
        ax.set_xlabel("layer (28 = final norm)")
    axes[0][0].set_ylabel("held-out top-1 accuracy")
    style.legend_outside(axes[0][-1])
    figs += [str(x) for x in style.save(fig, args.out / "figures" / "fig2_probes")]
    plt.close("all")

    L = ["# AR digit decoding: logit lens and probes (test-split captures)", "",
         f"User bootstrap ({len(boot.users):,} users, {args.draws:,} draws, seed {SEED}); 95% percentile "
         "intervals. Observational: decodable is not used.", ""]
    for c in cells:
        v = c["variant"]
        L += [f"## {v}", "", f"Validation: {json.dumps(c['validation'])}", "",
              "Logit lens, golden top-1 by layer (every 3rd layer + final norm = 28):", "",
              "| layer | " + " | ".join(f"digit {d + 1}" for d in range(c["inputs"]["n_digits"])) +
              " | recent item's digit-1 code top-1 (readout 1) |",
              "|---|" + "---|" * (c["inputs"]["n_digits"] + 1)]
        s = res[(res.variant == v) & (res.kind == "lens")]
        for layer in [*range(0, 28, 3), 27, 28]:
            q = s[s.layer == layer]
            if q.empty:
                continue
            cells_ = [f"{r.est:.3f}" for r in q[q.what == "golden top-1"].sort_values("digit").itertuples()]
            rec = q[q.what == "recent item's code top-1"]
            L.append(f"| {layer} | " + " | ".join(cells_) + f" | {rec.est.iloc[0]:.3f} |")
        L += ["", "Probe held-out accuracy [95% CI] by layer; baselines at layer -1:", "",
              "| probe | kind | layer | accuracy |", "|---|---|---|---|"]
        s = res[(res.variant == v) & res.kind.isin(["probe", "shuffled", "majority", "copy_recent_d0", "probe_split"])]
        for r in s.sort_values(["what", "kind", "layer"]).itertuples():
            if r.kind in ("probe", "probe_split") and r.layer not in (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28):
                continue
            L.append(f"| {r.what} | {r.kind} | {r.layer} | {r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}] |")
        L.append("")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "cells": [str(c["dir"]) for c in cells],
        "bootstrap": {"unit": "user", "users": len(boot.users), "draws": args.draws, "seed": SEED},
        "input_hashes": {str(c["dir"] / f): sha256_file(c["dir"] / f) for c in cells
                         for f in ("logit_lens.parquet", "probes.parquet", "validation.json")},
        "figures": figs}, indent=1))
    print("\n".join(L[:30]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
