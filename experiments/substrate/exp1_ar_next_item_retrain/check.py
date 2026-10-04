#!/usr/bin/env python
"""Acceptance check for retrained AR next-item models (protocol.md A1-A3).

For each evaluated native variant, compare the retrained model's archived-style
predictions with the deleted model's archived predictions on the same test
rows. Arm B (`__zcr`) variants have no archive; their own scores are listed
for reference only. CPU, seconds.

    python check.py --run-root $SIDLENS_WORK/runs/ar_next_item_retrain/r20261004 \
        --out $SIDLENS_WORK/derived/substrate/exp1_ar_next_item_retrain/summary-r20261004

Traps this guards against
-------------------------
* **Scoring with a different rule.** Hits use calc.py's rule (exact SID, else
  same title, else same item id via the info file), and NDCG = 1/log2(rank+2),
  as in every archived metric. Strict exact-SID Hit@10 is reported beside it.
* **Comparing different rows.** Row order is the only join key (ar_prompts
  trap 6). Both prediction files must have the test CSV's row count, or the
  eval limit's, and the same targets row by row.
* **Treating users as independent rows.** Intervals come from the shared
  paired user bootstrap (2,000 draws, seed 20261004).
* **Moving the goalposts.** The limits (|Δ Hit@10| <= 1.5 points, |Δ NDCG@10|
  <= 1.0 point) are constants here, as declared in protocol.md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sidlens import paths
from sidlens.analysis.bootstrap import UserBootstrap

import run

LIMIT_HR10, LIMIT_NDCG10 = 0.015, 0.010
CALIBRATION = "rqkmeans_3codebook_128"


def info_maps(info_path: str):
    to_title, to_item = {}, {}
    for ln in Path(info_path).read_text().splitlines():
        parts = ln.strip().split("\t")
        if len(parts) >= 3:
            to_title[parts[0].strip()] = parts[1].strip()
            to_item[parts[0].strip()] = parts[2].strip()
    return to_title, to_item


def ranks(preds: list[dict], to_title: dict, to_item: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """calc.py's first-match rank per row (inf = miss), the strict exact-SID rank, targets."""
    calc_r, strict_r, targets = [], [], []
    for s in preds:
        p = [x.strip("\"\n").strip() for x in s["predict"]]
        t = (s["output"][0] if isinstance(s["output"], list) else s["output"]).strip(" \n\"")
        targets.append(t)
        r = math.inf
        for i, x in enumerate(p):
            if x == t or (x in to_title and t in to_title and to_title[x] == to_title[t]) \
                    or (x in to_item and t in to_item and to_item[x] == to_item[t]):
                r = i
                break
        calc_r.append(r)
        strict_r.append(next((i for i, x in enumerate(p) if x == t), math.inf))
    return np.array(calc_r, float), np.array(strict_r, float), targets


def hit(r, k=10):
    return (r < k).astype(float)


def ndcg(r, k=10):
    out = np.zeros_like(r)
    m = r < k
    out[m] = 1.0 / np.log2(r[m] + 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rows, registry = [], []
    for vdir in sorted(p for p in a.run_root.iterdir() if (p / "status.json").exists()):
        st = json.loads((vdir / "status.json").read_text())
        if st["step"] != "evaluated":
            rows.append({"variant": vdir.name, "status": st["step"]})
            continue
        man = json.loads((vdir / "manifest.json").read_text())
        test = pd.read_csv(man["inputs"]["test"]["path"], dtype=str, keep_default_na=False)
        new = json.loads((vdir / "eval" / "predictions.json").read_text())
        n = len(new)
        if st.get("eval_limit"):
            test = test.head(st["eval_limit"])
        if n != len(test):
            raise SystemExit(f"{vdir.name}: {n} predictions for {len(test)} test rows")
        to_title, to_item = info_maps(man["inputs"]["info"]["path"])
        r_new, s_new, t_new = ranks(new, to_title, to_item)
        users = test["user_id"].to_numpy()
        bs = UserBootstrap(users, draws=a.draws, seed=a.seed)
        row = {"variant": vdir.name, "status": "evaluated", "rows": n,
               "train_gpus": man.get("train_gpus"), "train_seconds": man.get("train_seconds"),
               "hr10_new": hit(r_new).mean(), "ndcg10_new": ndcg(r_new).mean(),
               "strict_hr10_new": hit(s_new).mean()}
        ck = sorted((vdir / "checkpoints").glob("*/trainer_state.json"))
        if ck:
            tsd = json.loads(ck[-1].read_text())
            row["best_eval_loss_new"] = tsd.get("best_metric")
        base = vdir.name.removesuffix(run.ZCR_SUFFIX)
        if vdir.name == base:
            method, cb, sz = run.parse_variant(base)
            stem = paths.FROZEN_RESULTS / "sweep_metrics/next-item/metrics" / \
                f"nextitem__{run.CATEGORY}__{method}__{cb}cb__{sz}"
            old = json.loads(Path(f"{stem}.predictions.json").read_text())[:n]
            rec = json.loads(Path(f"{stem}.json").read_text())
            r_old, s_old, t_old = ranks(old, to_title, to_item)
            if t_old != t_new:
                raise SystemExit(f"{vdir.name}: archived and retrained targets differ row by row")
            d_hr = bs.paired_difference((users, hit(r_new)), (users, hit(r_old)))
            d_nd = bs.paired_difference((users, ndcg(r_new)), (users, ndcg(r_old)))
            d_st = bs.paired_difference((users, hit(s_new)), (users, hit(s_old)))
            top1 = np.mean([x["predict"][0].strip() == y["predict"][0].strip() for x, y in zip(new, old)])
            # share of the archived top-10's distinct SIDs that the retrained top-10 also holds
            ov10 = np.mean([len({p.strip() for p in x["predict"][:10]} & set(b)) / max(len(b), 1)
                            for x, y in zip(new, old)
                            for b in [{p.strip() for p in y["predict"][:10]}]])
            accepted = abs(d_hr["est"]) <= LIMIT_HR10 and abs(d_nd["est"]) <= LIMIT_NDCG10
            row.update(hr10_old=hit(r_old).mean(), ndcg10_old=ndcg(r_old).mean(),
                       strict_hr10_old=hit(s_old).mean(),
                       d_hr10=d_hr["est"], d_hr10_lo=d_hr["lo"], d_hr10_hi=d_hr["hi"],
                       d_ndcg10=d_nd["est"], d_ndcg10_lo=d_nd["lo"], d_ndcg10_hi=d_nd["hi"],
                       d_strict_hr10=d_st["est"], top1_agree=top1, top10_overlap=ov10,
                       best_eval_loss_old=rec.get("best_eval_loss"), accepted=bool(accepted),
                       calibration=(base == CALIBRATION))
            if accepted:
                registry.append({"variant": vdir.name, "weights": str(vdir / "weights" / "final"),
                                 "weights_sha256": man.get("weights_sha256", {}).get("model.safetensors")})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out / "estimates.csv", index=False)
    lines = ["# Retrain acceptance (protocol.md A1-A3)", "",
             f"Run: `{a.run_root}`. Limits: |Δ Hit@10| <= {100*LIMIT_HR10:.1f} pts, "
             f"|Δ NDCG@10| <= {100*LIMIT_NDCG10:.1f} pts. Intervals: paired user bootstrap, "
             f"{a.draws} draws, seed {a.seed}; they exclude training-seed variance.", ""]
    ev = df[df.get("status") == "evaluated"] if "status" in df else df
    if "d_hr10" in ev:
        cal = ev[ev.get("calibration", False) == True]   # noqa: E712
        if len(cal):
            c = cal.iloc[0]
            lines.append(f"**A1 calibration ({CALIBRATION}):** Δ Hit@10 {100*c.d_hr10:+.2f} "
                         f"[{100*c.d_hr10_lo:+.2f}, {100*c.d_hr10_hi:+.2f}] pts, Δ NDCG@10 "
                         f"{100*c.d_ndcg10:+.2f} pts, top-1 agreement {c.top1_agree:.3f}.")
        nat = ev.dropna(subset=["d_hr10"])
        if len(nat) >= 5:
            rho = nat["hr10_new"].rank().corr(nat["hr10_old"].rank())
            lines.append(f"**A3 grid:** Spearman ρ of Hit@10 across {len(nat)} maps = {rho:.2f}.")
        lines.append(f"**A2:** {int(nat['accepted'].sum())} of {len(nat)} native variants accepted.")
    try:                       # tabulate is not in every venv
        table = ev.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        table = "```\n" + ev.to_string(index=False) + "\n```"
    lines += ["", table]
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    (a.out / "registry.json").write_text(json.dumps(registry, indent=1))
    print("\n".join(lines[:8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
