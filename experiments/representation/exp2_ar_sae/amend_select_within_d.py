#!/usr/bin/env python
"""Amendment A1 (post hoc, 2026-09-30): pick prefix-match latents by within-digit AUC.

The declared Q2 rule ranks latents by the AUC of their activation for match_d,
pooled over the readouts d >= 1. The match rate falls with d (RQ-VAE 4x128:
0.29 / 0.14 / 0.08 at d = 1 / 2 / 3). A latent that only marks "this is the
digit-1 readout" therefore scores AUC > 0.5 without encoding any match.

This rule instead ranks each latent by the mean over d of its AUC within that
d. That removes the position confound. Like the declared rule, it uses the
screening half only. It writes a `selection.json` whose CM part (S, control
sets) follows this rule, and whose copy plan is copied unchanged from the
declared analysis, so `causal.py --parts CM` can run on it.

    python amend_select_within_d.py --cell 1 --sae-run <train cell> --analysis <analyze cell> --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from analyze import CELLS, LAYERS, N_CTRL_SETS, N_S, SEED, auc_columns, dense_acts, load_latents  # noqa: E402


def within_d_auc(A, y, d, mask) -> np.ndarray:
    parts = []
    for dd in np.unique(d):
        m = mask & (d == dd)
        if y[m].any() and (~y[m]).any():
            parts.append(auc_columns(A[m], y[m]))
    return np.mean(parts, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--sae-run", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    for nm in ("sae_run", "analysis"):
        if (getattr(args, nm) / f"cell-{args.cell:02d}").is_dir():          # a run group: take this cell
            setattr(args, nm, getattr(args, nm) / f"cell-{args.cell:02d}")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "selection.json").exists():
        raise FileExistsError(f"{args.out} already holds a selection")
    base = json.loads((args.analysis / "selection.json").read_text())
    if base["cell"] != CELLS[args.cell]:
        raise ValueError("analysis is for another cell")
    pos = pd.read_parquet(args.analysis / "positions.parquet")
    p1 = pos[pos.d >= 1].reset_index(drop=True)
    y, d = p1.match.to_numpy(), p1.d.to_numpy()
    s0, s1 = (p1.half == 0).to_numpy(), (p1.half == 1).to_numpy()
    rows = p1.store_row.to_numpy()
    out = {**base, "amendment": "A1 within-d AUC selection (post hoc)", "layers": {}}
    report = []
    for L in LAYERS:
        idx, val = load_latents(args.sae_run, L)
        m = int(idx.max()) + 1
        support = np.bincount(idx[rows][(val[rows] > 0) & s0[:, None]].ravel(), minlength=m)
        cand = np.flatnonzero(support >= 20)
        A = dense_acts(idx, val, rows, cand)
        auc = within_d_auc(A, y, d, s0)
        order = np.argsort(-auc)
        S = [int(cand[j]) for j in order[:N_S]]
        jS = [int(order[k]) for k in range(N_S)]
        meanact = np.array([A[s0][:, j][A[s0][:, j] > 0].mean() if (A[s0][:, j] > 0).any() else 0
                            for j in range(len(cand))])
        pool = [j for j in range(len(cand)) if abs(auc[j] - 0.5) < 0.02 and cand[j] not in S]
        rng = np.random.default_rng(SEED + L)
        ctrl = []
        for _ in range(N_CTRL_SETS):
            used, cs = set(), []
            for j0 in jS:
                near = [j for j in sorted(pool, key=lambda j: abs(meanact[j] - meanact[j0])) if j not in used][:20]
                pick = int(rng.choice(near))
                used.add(pick)
                cs.append(int(cand[pick]))
            ctrl.append(cs)
        auc_ho = within_d_auc(A[:, jS], y, d, s1)
        lay = dict(base["layers"][str(L)])
        lay.update({"S": S, "S_auc_screen": [float(auc[j]) for j in jS], "S_auc_heldout": auc_ho.tolist(),
                    "control_sets": ctrl, "n_candidates": int(len(cand)), "rule": "mean within-d AUC"})
        out["layers"][str(L)] = lay
        for f, a_s, a_h in zip(S, lay["S_auc_screen"], auc_ho):
            report.append({"variant": base["variant"], "layer": L, "latent": f, "screen_within_d_auc": a_s,
                           "heldout_within_d_auc": float(a_h)})
        print(f"[L{L}] S={S} heldout within-d AUC {np.round(auc_ho, 3).tolist()}", flush=True)
    (args.out / "selection.json").write_text(json.dumps(out))
    pd.DataFrame(report).to_csv(args.out / "a1_selection.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
