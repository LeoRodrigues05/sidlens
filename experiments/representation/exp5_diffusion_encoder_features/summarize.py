#!/usr/bin/env python
"""Summarize representation/exp5 over cells: probe accuracies, SAE fidelity and census, CS1/CS2, description autoencoder.

Reads `<run>/cell-XX/` (cells whose validation.json exists) and writes
`estimates.csv`, `census_shares.csv`, `report.md` into a new directory.
Intervals: paired user bootstrap (`UserBootstrap`, 2,000 draws, seed 20261014);
they exclude training-seed uncertainty of the recommenders and of the probes,
SAEs and reconstructors.

    python summarize.py --run <run dir> [<run dir> ...] --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402

SEED = 20261014


def cell_info(ckpt: str) -> dict:
    from sidlens import grid
    if grid.is_grid(ckpt):
        return grid.cell_info(ckpt)
    q, depth, width = re.match(r"diff-next1-(\w+)-(\d)cb-(\d+)", ckpt).groups()
    return {"quantizer": q, "depth": int(depth), "width": int(width)}


def markdown(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def flip(r: dict) -> dict:
    return {**r, "est": 1 - r["est"], "lo": 1 - r["hi"], "hi": 1 - r["lo"]}


def summarize_cell(path: Path) -> tuple[list[dict], list[dict], dict]:
    inputs = json.loads((path / "inputs.json").read_text())
    base = {"checkpoint": inputs["checkpoint"], **cell_info(inputs["checkpoint"])}
    n = inputs["n_digit"]
    slots = np.load(path / "test_slots.npz")
    users_tab = pd.read_csv(path / "users.csv")
    boot = UserBootstrap(users_tab.user_index.values, draws=2000, seed=SEED)
    rows = []

    # ---- Part P ---------------------------------------------------------------------
    probes = pd.read_csv(path / "probes.csv")
    z = np.load(path / "probe_correct.npz")
    hits = {}
    for key in {k.rsplit("|", 1)[0] for k in z.files}:
        pid, site, subset = key.split("|")
        r = z[f"{key}|rows"]
        user = r if pid.startswith("P5") else slots["hist"][r]
        hits[(pid, site, subset)] = (user, z[f"{key}|correct"].astype(float), r)
        meta = probes[(probes.probe == pid) & (probes.site == site) & (probes.subset == subset)].iloc[0]
        rows.append({**base, "estimand": "probe_acc", "probe": pid, "site": site, "subset": subset,
                     "majority_acc": meta.majority_acc, "nig": meta.nig, **boot.mean(user, hits[(pid, site, subset)][1])})
    for (pid, site, subset), (user, acc, r) in hits.items():
        if site == "input":
            continue
        ref = hits.get((pid, "input", subset))
        if ref is None or not np.array_equal(ref[2], r):
            continue
        rows.append({**base, "estimand": f"probe_acc_{site}_minus_input", "probe": pid, "site": site, "subset": subset,
                     **boot.mean(user, acc - ref[1])})
    for subset, sel in (("all", np.ones(len(users_tab), bool)), ("new", users_tab.new_target.values)):
        u = users_tab.user_index.values[sel]
        rows.append({**base, "estimand": "reference_copy_rule", "probe": "P5_tgt_d0", "site": "-", "subset": subset,
                     **boot.mean(u, users_tab.copy_rule_d0.values[sel].astype(float))})
        rows.append({**base, "estimand": "reference_model_top1", "probe": "P5_tgt_d0", "site": "-", "subset": subset,
                     **boot.mean(u, users_tab.model_top1_d0_correct.values[sel].astype(float))})
        known = sel & users_tab.tgt_cat_known.values
        rows.append({**base, "estimand": "reference_copy_rule", "probe": "P5_tgt_cat_l2", "site": "-", "subset": subset,
                     **boot.mean(users_tab.user_index.values[known], users_tab.copy_rule_cat_l2.values[known].astype(float))})

    # ---- conditions -------------------------------------------------------------------
    cond = pd.read_csv(path / "conditions.csv.gz")
    kl_cols = [f"kl_{d}" for d in range(n)]
    by = {c: g.set_index("user_index").sort_index() for c, g in cond.groupby("cond")}
    u_all = by["clean"].index.values
    kl = {c: g[kl_cols].sum(1).values for c, g in by.items()}
    for c in ("sae", "desc", "oracle") + tuple(k for k in by if k.startswith("drop_")):
        rows.append({**base, "estimand": "splice_recovered", "probe": c, "site": "all digits", "subset": "all",
                     **flip(boot.ratio(u_all, kl[c], kl["mean"]))})
        for d in range(n):
            rows.append({**base, "estimand": "splice_recovered", "probe": c, "site": f"digit {d}", "subset": "all",
                         **flip(boot.ratio(u_all, by[c][f"kl_{d}"].values, by["mean"][f"kl_{d}"].values))})
        rows.append({**base, "estimand": "splice_top1_agree_d0", "probe": c, "site": "digit 0", "subset": "all",
                     **boot.mean(u_all, (by[c].top1_d0.values == by["clean"].top1_d0.values).astype(float))})
    for k in (k for k in by if k.startswith("drop_")):
        rows.append({**base, "estimand": "field_drop_kl_increase", "probe": k[5:], "site": "all digits", "subset": "all",
                     **boot.ratio(u_all, kl[k] - kl["desc"], kl["mean"])})
    for key, ctrl in (("cs1", "cs1_ctrl"), ("cs2", "cs2_ctrl")):
        g, c = by[key], by[ctrl]
        has = (g.n_ablated.values > 0)
        u = u_all[has]
        dcopy = by["clean"]["copy"].values - g["copy"].values
        dcopy_c = by["clean"]["copy"].values - c["copy"].values
        dgold = by["clean"].gold.values - g.gold.values
        dgold_c = by["clean"].gold.values - c.gold.values
        rows.append({**base, "estimand": f"{key}_share_users_with_latents", "probe": key, "site": "-", "subset": "L>=2",
                     **boot.mean(u_all[users_tab.set_index("user_index").loc[u_all].L.values >= 2], has[users_tab.set_index("user_index").loc[u_all].L.values >= 2].astype(float))})
        rows.append({**base, "estimand": f"{key}_dcopy", "probe": key, "site": "-", "subset": "with latents", **boot.mean(u, dcopy[has])})
        rows.append({**base, "estimand": f"{key}_dcopy_ctrl", "probe": key, "site": "-", "subset": "with latents", **boot.mean(u, dcopy_c[has])})
        rows.append({**base, "estimand": f"{key.upper()}", "probe": key, "site": "-", "subset": "with latents",
                     **boot.mean(u, dcopy[has] - dcopy_c[has])})
        rows.append({**base, "estimand": f"{key}_dgold_minus_ctrl", "probe": key, "site": "-", "subset": "with latents",
                     **boot.mean(u, dgold[has] - dgold_c[has])})
        rows.append({**base, "estimand": f"{key}_mean_latents_ablated", "probe": key, "site": "-", "subset": "with latents",
                     **boot.mean(u, g.n_ablated.values[has].astype(float))})
        enc = by["enc_block"]
        rows.append({**base, "estimand": f"enc_block_dcopy", "probe": key, "site": "-", "subset": "with latents",
                     **boot.mean(u, (by["clean"]["copy"].values - enc["copy"].values)[has])})

    census = pd.read_csv(path / "census.csv")
    live = census[census.assigned != "dead"]
    shares = [{**base, "family": fam, "share_of_live": cnt / len(live), "n": int(cnt), "n_live": len(live),
               "n_dead": int((census.assigned == "dead").sum())} for fam, cnt in live.assigned.value_counts().items()]
    extra = {"fidelity": json.loads((path / "sae_fidelity.json").read_text()),
             "nla": json.loads((path / "nla.json").read_text())}
    for key in ("fvu_test", "dead_frac_test", "fvu_fit_val"):
        rows.append({**base, "estimand": f"sae_{key}", "probe": "sae", "site": "output", "subset": "all",
                     "est": extra["fidelity"][key]})
    for key in ("fvu_desc", "fvu_oracle", "fvu_sae", "fvu_mean"):
        rows.append({**base, "estimand": f"nla_{key}", "probe": "nla", "site": "output", "subset": "all",
                     "est": extra["nla"][key]})
    for field, v in extra["nla"]["verbalizer"].items():
        rows.append({**base, "estimand": "verbalizer_acc", "probe": field, "site": "output", "subset": "all",
                     "est": v["acc_test"], "majority_acc": v["majority_test"]})
    return rows, shares, extra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, nargs="*", default=[], help="run dirs holding cell-*/")
    ap.add_argument("--cells", type=Path, nargs="*", default=[], help="cell dirs (grid layout)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rows, shares, skipped, cells = [], [], [], []
    for path in [p for run in args.run for p in sorted(run.glob("cell-*"))] + list(args.cells):
        if not (path / "validation.json").exists():
            skipped.append(str(path))
            continue
        r, s, _ = summarize_cell(path)
        rows += r
        shares += s
        cells.append(r[0]["checkpoint"])
        print(f"{path}: {len(r)} estimates", flush=True)
    est = pd.DataFrame(rows)
    est.to_csv(args.out / "estimates.csv", index=False)
    pd.DataFrame(shares).to_csv(args.out / "census_shares.csv", index=False)
    (args.out / "inputs.json").write_text(json.dumps({"runs": [str(r) for r in args.run], "cells": cells,
                                                      "skipped": skipped, "seed": SEED}, indent=1))

    def fmt(sel):
        if sel.empty:
            return ""
        r = sel.iloc[0]
        return f"{r.est:.3f}" if pd.isna(r.get("lo")) else f"{r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}]"

    table = []
    for ck in cells:
        e = est[est.checkpoint == ck]
        i = cell_info(ck)

        def g(estimand, probe=None, site=None, subset=None):
            s = e[e.estimand == estimand]
            for col, v in (("probe", probe), ("site", site), ("subset", subset)):
                if v is not None:
                    s = s[s[col] == v]
            return fmt(s)
        table.append({"cell": f"{i['quantizer']} {i['depth']}x{i['width']}",
                      "own d1 (output)": g("probe_acc", "P1_own_d0", "output", "all"),
                      "recency out-in": g("probe_acc_output_minus_input", "P3_rec", "output", "all"),
                      "spread out-in": g("probe_acc_output_minus_input", "P4a_r1_d0", "output", "all"),
                      "target new out-in": g("probe_acc_output_minus_input", "P5_tgt_d0", "output", "new"),
                      "SAE FVU": g("sae_fvu_test"), "splice rec.": g("splice_recovered", "sae", "all digits"),
                      "CS1": g("CS1"), "desc FVU": g("nla_fvu_desc"),
                      "desc splice rec.": g("splice_recovered", "desc", "all digits")})
    lines = ["# representation/exp5 summary", "", f"Cells: {len(cells)}; skipped: {len(skipped)}.", "",
             markdown(pd.DataFrame(table)), ""]
    (args.out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
