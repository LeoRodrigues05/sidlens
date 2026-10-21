#!/usr/bin/env python
"""Estimates for exp11 (core copy estimates on the retrained AR grid): per cell, G1-G3, figures.

    python summarize.py --groups $SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/<array-id> \\
        --out $SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/summary-<array-id>

Per-cell estimators are the source experiments' own (exp4 `summarize_cell`,
exp6's rank rules), imported from their files, so a grid number means exactly
what the two-checkpoint number meant. The original checkpoints are re-scored
here from their result directories with the same code and the same draws.

Traps this guards against
-------------------------
* **Unpaired cells.** One `UserBootstrap` per split (test, valid), seed
  20260927, 2,000 draws, shared by every cell and by the originals, so G2
  differences use the same resampled users on both sides.
* **Counting a cell whose controls failed.** G1 drops a cell from an
  estimand when its exp4 no-op controls are not bit-exact or its exp6
  baseline reproduces < 99% of its own evaluation lists (protocol).
* **Superseded plain-decoder outputs.** The original exp6 job 283114 holds
  `*_plain` files from before amendment 2; the originals' plain conditions are
  read from 283586 / 283621 instead.
* **Mixing a re-run with its first attempt.** When a stage completed in more
  than one group, the later group wins and the choice is recorded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402


def _module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parents[1] / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E4 = _module("exp4_ar_copy_circuit/summarize.py", "_exp4_summarize")
E6 = _module("exp6_ar_copy_in_decoding/summarize.py", "_exp6_summarize")

SEED, DRAWS = 20260927, 2000
V1_MIN = 0.99
CTRL = paths.DERIVED / "controlled"
ORIGINAL = {
    "rqkmeans_3codebook_128": {
        "label": "next-item_best (original)",
        "exp3_A": CTRL / "exp3_ar_history_patching/280961/cell-00",
        "exp4_valid": CTRL / "exp4_ar_copy_circuit/281261/cell-00",
        "exp4_test": CTRL / "exp4_ar_copy_circuit/281394/cell-00",
        "exp6": [(CTRL / "exp6_ar_copy_in_decoding/283114/cell-00", ("B", "C_later", "C_all", "N_later")),
                 (CTRL / "exp6_ar_copy_in_decoding/283586/cell-00", ("B_plain", "C_all_plain"))]},
    "rqvae_4codebook_128": {
        "label": "oneoff_rqvae4cb128 (original; not the sweep model)",
        "exp3_A": CTRL / "exp3_ar_history_patching/280961/cell-01",
        "exp4_valid": CTRL / "exp4_ar_copy_circuit/281261/cell-01",
        "exp4_test": CTRL / "exp4_ar_copy_circuit/281394/cell-01",
        "exp6": [(CTRL / "exp6_ar_copy_in_decoding/283114/cell-01", ("B", "C_later", "C_all", "N_later")),
                 (CTRL / "exp6_ar_copy_in_decoding/283621/cell-01", ("B_plain", "C_all_plain"))]},
}
STAGES = ("exp4_valid", "exp3_A", "exp4_test", "exp6")
G1_ESTIMANDS = {  # name: (stage, estimand, sign the original checkpoints showed)
    "A1": ("exp3_A", "A1 delta digit 0 (r=1, m=0)", +1),
    "H1": ("exp4_valid", "H1 match - nonmatch (pooled d>=1)", +1),
    "H2b": ("exp4_valid", "H2b K1 - K3 | match (paired, pooled d>=1)", +1),
    "P1": ("exp6", "P1 C_all - B | repeat (HR@10)", -1),
}


def quantizer(v: str) -> str:
    return v.split("_")[0]


def depth_width(v: str) -> tuple[int, int]:
    m = re.match(r".+_(\d)codebook_(\d+)$", v)
    return int(m[1]), int(m[2])


# ------------------------------------------------------------------ inputs --
def find_stages(groups: list[Path], allow_pilot: bool) -> tuple[dict, list[dict]]:
    """{(variant, stage): dir} for every stage a cell's status.txt marks complete."""
    found, log = {}, []
    for g in groups:
        for cell in sorted(g.glob("cell-*")):
            lines = (cell / "status.txt").read_text().splitlines()
            variant = re.search(r"variant=(\S+)", lines[0])[1]
            for ln in lines[1:]:
                m = re.match(r"(complete|failed) stage=(\S+)", ln)
                if not m:
                    continue
                d = cell / m[2]
                rec = {"group": g.name, "cell": cell.name, "variant": variant, "stage": m[2], "state": m[1]}
                log.append(rec)
                if m[1] != "complete":
                    continue
                inp = json.loads((d / "inputs.json").read_text())
                if inp.get("limited") and not allow_pilot:
                    raise RuntimeError(f"{d}: a pilot stage; pass --allow-pilot")
                if (variant, m[2]) in found:
                    rec["note"] = f"supersedes {found[(variant, m[2])]}"
                found[(variant, m[2])] = d
    return found, log


def exp6_table(sources: list[tuple[Path, tuple]], variant: str) -> pd.DataFrame:
    """exp6's per-row table (exact-SID ranks, strata, copy rates) from chosen condition files."""
    ex = {e.row: e for e in P.load_examples(variant, "next-item", "test")}
    rows = []
    for d, conds in sources:
        for cond in conds:
            df = pd.read_parquet(d / f"predictions_{cond}.parquet")
            for r in df.itertuples():
                e = ex[r.row]
                tgt, pred, hist = e.target_sids[0], list(r.predict), set(e.history_sids)
                rows.append({"row": r.row, "user_id": r.user_id, "condition": cond,
                             "rank": E6.exact_rank(pred, tgt), "repeat": tgt in hist,
                             "top1_is_history": float(pred[0] in hist),
                             "top1_digit0_is_recent": float(pred[0].split(">")[0]
                                                           == e.history_sids[-1].split(">")[0])})
    t = pd.DataFrame(rows)
    for k in (1, 10, 50):
        t[f"hr{k}"] = (t["rank"] < k).astype(float)
    t["ndcg10"] = np.where(t["rank"] < 10, 1 / np.log2(t["rank"].clip(upper=10**5) + 2), 0.0)
    return t


# --------------------------------------------------------------- estimates --
class Cell:
    """All estimates of one model, with bootstrap draws kept for the G2 contrasts."""

    def __init__(self, variant: str, model: str, boots: dict):
        self.variant, self.model, self.boots = variant, model, boots
        self.rows: list[dict] = []
        self.draws: dict[str, np.ndarray] = {}
        self.valid: dict[str, bool] = {}
        self.notes: dict[str, object] = {}

    def add(self, stage, name, r: dict, draws=None, **kw):
        self.rows.append({"variant": self.variant, "model": self.model, "stage": stage, "estimand": name,
                          **kw, **r})
        if draws is not None:
            self.draws[f"{stage}|{name}"] = draws

    def mean(self, stage, name, split, user, x, **kw):
        b = self.boots[split]
        est, bs, nu = b.mean_draws(user, x)
        self.add(stage, name, b._summ(est, bs, len(np.asarray(x)), nu), bs, **kw)

    def diff(self, stage, name, split, a, b_, **kw):
        b = self.boots[split]
        ea, ba, na = b.mean_draws(*a)
        eb, bb, nb = b.mean_draws(*b_)
        self.add(stage, name, b._summ(ea - eb, ba - bb, len(np.asarray(a[1])) + len(np.asarray(b_[1])),
                                      max(na, nb)), ba - bb, **kw)

    # ---- exp3 Part A -------------------------------------------------------
    def exp3(self, d: Path):
        clean = pd.read_parquet(d / "clean.parquet")[["example_id", "digit", "logp_codes", "top1"]]
        a = pd.read_parquet(d / "part_a.parquet").merge(
            clean.rename(columns={"logp_codes": "clean_logp", "top1": "clean_top1"}),
            on=["example_id", "digit"], validate="many_to_one")
        a["delta"] = a.clean_logp - a.logp_codes
        a["flip"] = (a.top1 != a.clean_top1).astype(float)
        r1 = a[a.recency == 1]
        base = r1[(r1.m == 0) & (r1.digit == 0)]
        self.mean("exp3_A", "A1 delta digit 0 (r=1, m=0)", "test", base.user_id, base.delta, d=0)
        self.mean("exp3_A", "top-1 flip digit 0 (r=1, m=0)", "test", base.user_id, base.flip, d=0)
        w = r1[r1.m == 0].groupby(["example_id", "user_id"], as_index=False).delta.sum()
        self.mean("exp3_A", "A1 delta whole SID (r=1, m=0)", "test", w.user_id, w.delta)
        m1 = r1[(r1.m == 1) & (r1.digit == 0)]
        b = self.boots["test"]
        ea, ba, na = b.mean_draws(m1.user_id, m1.delta)
        eb, bb, _ = b.mean_draws(base.user_id, base.delta)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.add("exp3_A", "A3 retained fraction digit 0 (m=1)", b._summ(ea / eb, ba / bb, len(m1), na), d=0)
        for r in range(1, 6):
            g = a[(a.recency == r) & (a.m == 0) & (a.digit == 0)]
            self.mean("exp3_A", "A2 delta digit 0 by recency", "test", g.user_id, g.delta, recency=r, d=0)
        for dg, g in r1[r1.m == 0].groupby("digit"):
            self.mean("exp3_A", "A1 delta by digit (r=1, m=0)", "test", g.user_id, g.delta, d=int(dg))
        self.valid["exp3_A"] = True

    # ---- exp4 Parts R, K ---------------------------------------------------
    def exp4(self, d: Path, stage: str):
        split = "valid" if stage == "exp4_valid" else "test"
        inputs = json.loads((d / "inputs.json").read_text())
        val = json.loads((d / "validation.json").read_text())
        ok = all(v["n_exact"] == v["n"] for v in val.get("controls", {}).values()) and bool(val.get("controls"))
        self.valid[stage] = ok
        self.notes[f"{stage}_controls"] = val.get("controls")
        clean = pd.read_parquet(d / "clean.parquet")[["i", "example_id", "user_id", "digit", "logp_codes"]]
        c = {"dir": d, "inputs": inputs, "validation": val, "variant": inputs["variant"],
             "clean": clean.rename(columns={"logp_codes": "clean_logp"})}
        mine = {G1_ESTIMANDS["H1"][1], G1_ESTIMANDS["H2b"][1], "H2a K1 | match (pooled d>=1)"}
        for r in E4.summarize_cell(c, self.boots[split]):
            r = dict(r)
            name = r.pop("estimand")
            if name in mine:                     # recomputed below with the same estimator, keeping draws
                continue
            r.pop("variant"), r.pop("split"), r.pop("part")
            self.add(stage, name, {k: r.pop(k) for k in ("est", "lo", "hi", "n_rows", "n_users") if k in r},
                     **{k: v for k, v in r.items() if k == "d"})
        rr = E4.with_delta(pd.read_parquet(d / "part_r.parquet"), c["clean"])
        m, nm = rr[rr.match == True], rr[rr.match == False]                # noqa: E712
        self.diff(stage, "H1 match - nonmatch (pooled d>=1)", split, (m.user_id, m.delta),
                  (nm.user_id, nm.delta))
        k = E4.with_delta(pd.read_parquet(d / "part_k.parquet").query("d >= 0"), c["clean"])
        k1m = k[(k.condition == "K1_next") & (k.d >= 1) & (k.match == True)]      # noqa: E712
        self.mean(stage, "H2a K1 | match (pooled d>=1)", split, k1m.user_id, k1m.delta)
        pair = k1m.merge(k[k.condition == "K3_ctrl"][["i", "d", "delta"]], on=["i", "d"], suffixes=("", "_k3"))
        self.mean(stage, "H2b K1 - K3 | match (paired, pooled d>=1)", split, pair.user_id,
                  pair.delta - pair.delta_k3)
        k5 = k[(k.condition == "K5_next_late") & (k.d >= 1) & (k.match == True)]  # noqa: E712
        b = self.boots[split]
        e5, b5, n5 = b.mean_draws(k5.user_id, k5.delta)
        e1, b1, _ = b.mean_draws(k1m.user_id, k1m.delta)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.add(stage, "late share K5 / K1 | match (pooled d>=1)", b._summ(e5 / e1, b5 / b1, len(k5), n5))
        k0 = k[(k.condition == "K1_next") & (k.d == 0)]
        self.mean(stage, "K1 digit 0 (all rows)", split, k0.user_id, k0.delta, d=0)

    # ---- exp6 --------------------------------------------------------------
    def exp6(self, sources: list[tuple[Path, tuple]], val: dict | None):
        t = exp6_table(sources, self.variant)
        v1 = (val or {}).get("V1_reproduction", {})
        self.notes["V1"] = {k: v1.get(k) for k in ("rows", "identical_50", "identical_top10")}
        self.valid["exp6"] = val is None or (v1.get("identical_50") or 0) >= V1_MIN
        w = t.pivot_table(index=["row", "user_id", "repeat"], columns="condition",
                          values=["hr1", "hr10", "hr50", "ndcg10", "top1_is_history", "top1_digit0_is_recent"])
        w = w.reset_index()
        conds = set(t.condition)
        rep, new = w.repeat.to_numpy(bool), ~w.repeat.to_numpy(bool)
        for metric in ("hr10", "hr1", "ndcg10"):
            for a, bb in (("C_all", "B"), ("C_later", "B"), ("N_later", "B"), ("C_later", "N_later"),
                          ("C_all_plain", "B_plain")):
                if not {a, bb} <= conds:
                    continue
                diff = w[(metric, a)] - w[(metric, bb)]
                lab = metric.upper().replace("HR", "HR@").replace("NDCG", "NDCG@")
                self.mean("exp6", f"{a} - {bb} | all ({lab})", "test", w.user_id, diff)
                self.mean("exp6", f"{a} - {bb} | repeat ({lab})", "test", w.user_id[rep], diff[rep])
                self.mean("exp6", f"{a} - {bb} | new ({lab})", "test", w.user_id[new], diff[new])
                self.diff("exp6", f"{a} - {bb} | repeat - new ({lab})", "test",
                          (w.user_id[rep], diff[rep]), (w.user_id[new], diff[new]))
        for name, src in (("P1 C_all - B | repeat (HR@10)", "C_all - B | repeat (HR@10)"),
                          ("P2 C_all - B | new (HR@10)", "C_all - B | new (HR@10)"),
                          ("P3 repeat - new (HR@10)", "C_all - B | repeat - new (HR@10)")):
            r = next(x for x in self.rows if x["stage"] == "exp6" and x["estimand"] == src)
            self.add("exp6", name, {k: r[k] for k in ("est", "lo", "hi", "n_rows", "n_users")},
                     self.draws[f"exp6|{src}"])
        for cond in sorted(conds):
            for metric in ("hr10", "hr1", "top1_is_history", "top1_digit0_is_recent"):
                self.mean("exp6", f"level {metric}", "test", w.user_id, w[(metric, cond)], condition=cond)
                if metric == "hr10":
                    self.mean("exp6", "level hr10 | repeat", "test", w.user_id[rep], w[(metric, cond)][rep],
                              condition=cond)
                    self.mean("exp6", "level hr10 | new", "test", w.user_id[new], w[(metric, cond)][new],
                              condition=cond)
        self.notes["exp6_rows"] = int(len(w))
        self.notes["exp6_repeat_rows"] = int(rep.sum())


# ----------------------------------------------------------------- figures --
def figures(res: pd.DataFrame, g2: pd.DataFrame, g3: dict, out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sidlens.viz import style as S
    S.apply()
    files = []
    grid = res[res.model.str.startswith("retrain/")]
    variants = sorted(grid.variant.unique(), key=lambda v: (S.QUANTIZER_ORDER.index(quantizer(v)),) + depth_width(v))
    ypos = {v: i for i, v in enumerate(variants)}
    label = {v: f"{S.QUANTIZER_LABEL[quantizer(v)]} {depth_width(v)[0]}×{depth_width(v)[1]}" for v in variants}

    def forest(ax, stage, name, scale=1.0, title="", unit=""):
        for _, r in res[(res.stage == stage) & (res.estimand == name)].iterrows():
            if r.variant not in ypos:
                continue
            orig = not r.model.startswith("retrain/")
            y = ypos[r.variant] + (0.28 if orig else 0)
            col = S.QUANTIZER_COLOR[quantizer(r.variant)]
            ax.errorbar(scale * r.est, y, xerr=[[scale * (r.est - r.lo)], [scale * (r.hi - r.est)]],
                        fmt="D" if orig else "o", ms=S.MARKER + (0 if orig else 1), color=col,
                        mfc="white" if orig else col, capsize=0, lw=1.2)
        ax.axvline(0, color=S.AXIS, lw=0.9)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(unit, fontsize=7)
        ax.set_ylim(len(variants) - 0.5, -0.7)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([label[v] for v in variants], fontsize=7)

    key = [Line2D([], [], marker="o", ls="", color=S.INK, label="retrained (this experiment)"),
           Line2D([], [], marker="D", ls="", color=S.INK, mfc="white", label="original checkpoint")]

    # fig1: the four G1 estimands
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.6), sharey=True)
    for ax, (st, nm, sc, ti, un) in zip(axes, [
            ("exp3_A", G1_ESTIMANDS["A1"][1], 1, "A1: replace most recent item", "Δ log p, golden digit 1 (nats)"),
            ("exp4_valid", G1_ESTIMANDS["H1"][1], 1, "H1: prefix-matched copying", "match − non-match (nats)"),
            ("exp4_valid", G1_ESTIMANDS["H2b"][1], 1, "H2b: single-edge knockout", "K1 − control edge (nats)"),
            ("exp6", G1_ESTIMANDS["P1"][1], 100, "P1: copy block in decoding", "ΔHR@10, repeat targets (pp)")]):
        forest(ax, st, nm, sc, ti, un)
    axes[-1].legend(handles=key, fontsize=6.5, frameon=False, loc="lower left")
    files += [str(p) for p in S.save(fig, out / "fig1_core_estimates")]

    # fig2: where the edge acts and how much the item matters
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.6), sharey=True)
    forest(axes[0], "exp4_valid", "H2a K1 | match (pooled d>=1)", 1, "H2a: edge knockout, matching rows", "nats")
    forest(axes[1], "exp4_valid", "late share K5 / K1 | match (pooled d>=1)", 1, "Share from layers 14–27",
           "K5 / K1")
    axes[1].axvline(1, color=S.AXIS, lw=0.9, ls=":")
    forest(axes[2], "exp3_A", "A3 retained fraction digit 0 (m=1)", 1, "A3: effect left when first digit kept",
           "fraction of A1")
    files += [str(p) for p in S.save(fig, out / "fig2_edge_layers_prefix")]

    # fig3: decoding, repeat vs new, with the matched control
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.6), sharey=True)
    forest(axes[0], "exp6", "C_all - B | repeat (HR@10)", 100, "Copy block: repeat targets", "ΔHR@10 (pp)")
    forest(axes[1], "exp6", "C_all - B | new (HR@10)", 100, "Copy block: new targets", "ΔHR@10 (pp)")
    forest(axes[2], "exp6", "N_later - B | all (HR@10)", 100, "Matched control block (all)", "ΔHR@10 (pp)")
    files += [str(p) for p in S.save(fig, out / "fig3_decoding")]

    # fig4: G3, mechanism strength against accuracy across the 12 maps
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    base = grid[(grid.stage == "exp6") & (grid.estimand == "level hr10") & (grid.condition == "B")] \
        .set_index("variant").est
    for ax, (st, nm, sc, yl) in zip(axes, [("exp4_valid", "H2a K1 | match (pooled d>=1)", 1, "H2a (nats)"),
                                           ("exp6", G1_ESTIMANDS["P1"][1], 100, "P1 (pp)")]):
        y = grid[(grid.stage == st) & (grid.estimand == nm)].set_index("variant").est * sc
        for v in y.index.intersection(base.index):
            dd, wd = depth_width(v)
            ax.scatter(100 * base[v], y[v], color=S.QUANTIZER_COLOR[quantizer(v)], s=18 + 6 * dd,
                       marker="o" if wd == 128 else "s", edgecolor="white", lw=0.6, zorder=3)
        ax.set_xlabel("baseline HR@10 (%)")
        ax.set_ylabel(yl)
        rho = g3.get(nm)
        ax.set_title(f"Spearman ρ = {rho:.2f}" if rho is not None else "", fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=S.QUANTIZER_COLOR[q], label=S.QUANTIZER_LABEL[q])
               for q in S.QUANTIZER_ORDER] + \
              [Line2D([], [], marker="o", ls="", color=S.MUTED, label="width 128"),
               Line2D([], [], marker="s", ls="", color=S.MUTED, label="width 512")]
    S.legend_outside(axes[-1], handles=handles, fontsize=6.5, frameon=False)
    files += [str(p) for p in S.save(fig, out / "fig4_mechanism_vs_accuracy")]

    # fig5: G2, the same map trained twice
    if len(g2):
        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        for i, r in enumerate(g2.itertuples()):
            sc = 100 if r.estimand.startswith("P1") else 1
            ax.errorbar(sc * r.est, i, xerr=[[sc * (r.est - r.lo)], [sc * (r.hi - r.est)]], fmt="o",
                        color=S.QUANTIZER_COLOR["rqkmeans"], ms=S.MARKER, capsize=0)
        ax.axvline(0, color=S.AXIS, lw=0.9)
        ax.set_yticks(range(len(g2)))
        ax.set_yticklabels([f"{r.estimand.split(' ')[0]} ({'pp' if r.estimand.startswith('P1') else 'nats'})"
                            for r in g2.itertuples()], fontsize=7)
        ax.set_ylim(len(g2) - 0.5, -0.5)
        ax.set_xlabel("retrained − original, RQ-KMeans 3×128")
        files += [str(p) for p in S.save(fig, out / "fig5_retrained_vs_original")]
    plt.close("all")
    return files


# -------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groups", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True, help="must not exist")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--allow-pilot", action="store_true")
    ap.add_argument("--no-originals", action="store_true")
    args = ap.parse_args(argv)
    found, stage_log = find_stages(args.groups, args.allow_pilot)
    if not found:
        raise SystemExit("no completed stages found")
    args.out.mkdir(parents=True, exist_ok=False)
    v0 = next(iter({v for v, _ in found}))
    boots = {s: UserBootstrap([e.user_id for e in P.load_examples(v0, "next-item", s)], args.draws, SEED)
             for s in ("test", "valid")}

    cells: list[Cell] = []
    for v in sorted({v for v, _ in found}):
        c = Cell(v, f"retrain/r20261004/{v}", boots)
        for st in STAGES:
            d = found.get((v, st))
            if d is None:
                continue
            if st == "exp3_A":
                c.exp3(d)
            elif st.startswith("exp4"):
                c.exp4(d, st)
            else:
                conds = tuple(sorted(p.stem.removeprefix("predictions_") for p in d.glob("predictions_*.parquet")))
                c.exp6([(d, conds)], json.loads((d / "validation.json").read_text()))
        cells.append(c)
        print(f"{v}: stages {[s for s in STAGES if (v, s) in found]}", flush=True)
    if not args.no_originals:
        for v, o in ORIGINAL.items():
            c = Cell(v, o["label"], boots)
            c.exp3(o["exp3_A"])
            c.exp4(o["exp4_valid"], "exp4_valid")
            c.exp4(o["exp4_test"], "exp4_test")
            c.exp6(o["exp6"], None)
            cells.append(c)
    res = pd.DataFrame([r for c in cells for r in c.rows])
    res["quantizer"] = res.variant.map(quantizer)
    res[["depth", "width"]] = pd.DataFrame(res.variant.map(depth_width).tolist(), index=res.index)
    res.to_csv(args.out / "estimates.csv", index=False)

    # ---- G1 ----------------------------------------------------------------
    grid = [c for c in cells if c.model.startswith("retrain/")]
    g1 = []
    for key, (st, nm, sign) in G1_ESTIMANDS.items():
        have = [c for c in grid if any(r["stage"] == st and r["estimand"] == nm for r in c.rows)]
        ok = [c for c in have if c.valid.get(st)]
        hits = [c.variant for c in ok
                for r in c.rows if r["stage"] == st and r["estimand"] == nm
                and (r["lo"] > 0 if sign > 0 else r["hi"] < 0)]
        g1.append({"estimand": key, "stage": st, "direction": ">0" if sign > 0 else "<0",
                   "cells_with_result": len(have), "cells_validated": len(ok), "cells_excluding_0": len(hits),
                   "not_excluding_0": sorted(c.variant for c in ok if c.variant not in hits),
                   "failed_validation": sorted(c.variant for c in have if c not in ok)})
    g1 = pd.DataFrame(g1)
    g1.to_csv(args.out / "g1_counts.csv", index=False)

    # ---- G2 ----------------------------------------------------------------
    g2 = []
    re_ = next((c for c in grid if c.variant == "rqkmeans_3codebook_128"), None)
    og = next((c for c in cells if c.model == ORIGINAL["rqkmeans_3codebook_128"]["label"]), None)
    if re_ is not None and og is not None:
        for st, nm in (("exp3_A", G1_ESTIMANDS["A1"][1]), ("exp4_valid", G1_ESTIMANDS["H1"][1]),
                       ("exp4_valid", "H2a K1 | match (pooled d>=1)"), ("exp6", G1_ESTIMANDS["P1"][1])):
            k = f"{st}|{nm}"
            if k in re_.draws and k in og.draws:
                ra = next(r for r in re_.rows if r["stage"] == st and r["estimand"] == nm)
                ro = next(r for r in og.rows if r["stage"] == st and r["estimand"] == nm)
                bs = re_.draws[k] - og.draws[k]
                bs = bs[np.isfinite(bs)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                g2.append({"estimand": nm, "stage": st, "retrained": ra["est"],
                           "original": ro["est"], "est": ra["est"] - ro["est"], "lo": lo, "hi": hi})
    g2 = pd.DataFrame(g2)
    g2.to_csv(args.out / "g2_retrained_vs_original.csv", index=False)

    # ---- G3 ----------------------------------------------------------------
    gr = res[res.model.str.startswith("retrain/")]
    base = gr[(gr.stage == "exp6") & (gr.estimand == "level hr10") & (gr.condition == "B")].set_index("variant").est
    g3 = {}
    for st, nm in (("exp4_valid", "H2a K1 | match (pooled d>=1)"), ("exp6", G1_ESTIMANDS["P1"][1]),
                   ("exp4_valid", G1_ESTIMANDS["H1"][1]), ("exp3_A", G1_ESTIMANDS["A1"][1])):
        y = gr[(gr.stage == st) & (gr.estimand == nm)].set_index("variant").est
        idx = y.index.intersection(base.index)
        g3[nm] = float(pd.Series(y[idx]).rank().corr(pd.Series(base[idx]).rank())) if len(idx) >= 4 else None

    figs = figures(res, g2, g3, args.out / "figures")

    # ---- report ------------------------------------------------------------
    def f(r, sc=1.0, nd=3):
        return f"{sc * r['est']:+.{nd}f} [{sc * r['lo']:+.{nd}f}, {sc * r['hi']:+.{nd}f}]"

    def pick(c, st, nm, **kw):
        for r in c.rows:
            if r["stage"] == st and r["estimand"] == nm and all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    L = ["# exp11: core copy estimates on the retrained Qwen-AR grid", "",
         f"Paired user bootstrap ({DRAWS:,} draws, seed {SEED}; test {len(boots['test'].users):,} users, valid "
         f"{len(boots['valid'].users):,}); 95% percentile intervals; they exclude training-seed variance. "
         "SID-level metrics.", "", "## G1: cells whose interval excludes 0 in the original direction", "",
         "| estimand | direction | cells with result | validated | excluding 0 | not excluding 0 | failed validation |",
         "|---|---|---|---|---|---|---|"]
    for r in g1.itertuples():
        L.append(f"| {r.estimand} | {r.direction} | {r.cells_with_result} | {r.cells_validated} | "
                 f"{r.cells_excluding_0} | {', '.join(r.not_excluding_0) or '–'} | "
                 f"{', '.join(r.failed_validation) or '–'} |")
    L += ["", "## Per cell", "",
          "| model | A1 (nats) | A3 | H1 valid | H2a valid | H2b valid | late share | P1 (pp) | P2 (pp) | "
          "N_later (pp) | HR@10 B | V1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        g = lambda st, nm, sc=1.0, nd=3, **kw: (f(r, sc, nd) if (r := pick(c, st, nm, **kw)) else "–")  # noqa: E731
        hr = pick(c, "exp6", "level hr10", condition="B")
        hr = "–" if hr is None else f"{100 * hr['est']:.1f}"
        v1 = c.notes.get("V1", {}).get("identical_50")
        v1 = "–" if v1 is None else f"{v1:.4f}"
        L.append(f"| {c.model} | {g('exp3_A', G1_ESTIMANDS['A1'][1])} | {g('exp3_A', 'A3 retained fraction digit 0 (m=1)', nd=2)} | "
                 f"{g('exp4_valid', G1_ESTIMANDS['H1'][1])} | {g('exp4_valid', 'H2a K1 | match (pooled d>=1)')} | "
                 f"{g('exp4_valid', G1_ESTIMANDS['H2b'][1])} | "
                 f"{g('exp4_valid', 'late share K5 / K1 | match (pooled d>=1)', nd=2)} | "
                 f"{g('exp6', G1_ESTIMANDS['P1'][1], 100, 2)} | {g('exp6', 'P2 C_all - B | new (HR@10)', 100, 2)} | "
                 f"{g('exp6', 'N_later - B | all (HR@10)', 100, 2)} | "
                 f"{hr} | {v1} |")
    L += ["", "Test-split exp4 (secondary):", "", "| model | H1 test | H2a test | H2b test |", "|---|---|---|---|"]
    for c in cells:
        g = lambda st, nm: (f(r) if (r := pick(c, st, nm)) else "–")       # noqa: E731
        L.append(f"| {c.model} | {g('exp4_test', G1_ESTIMANDS['H1'][1])} | "
                 f"{g('exp4_test', 'H2a K1 | match (pooled d>=1)')} | {g('exp4_test', G1_ESTIMANDS['H2b'][1])} |")
    L += ["", "## G2: retrained − original, RQ-KMeans 3×128 (same rows, same draws)", ""]
    for r in g2.itertuples():
        sc = 100 if r.stage == "exp6" else 1
        L.append(f"- {r.estimand}: retrained {sc * r.retrained:+.3f}, original {sc * r.original:+.3f}, "
                 f"difference {sc * r.est:+.3f} [{sc * r.lo:+.3f}, {sc * r.hi:+.3f}]")
    L += ["", "## G3: Spearman ρ across the retrained cells with baseline HR@10 (descriptive)", ""]
    L += [f"- {k}: {'–' if v is None else f'{v:.2f}'}" for k, v in g3.items()]
    L += ["", "## Validation", ""]
    for c in cells:
        L.append(f"- {c.model}: valid {c.valid}; V1 {c.notes.get('V1')}")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({
        "groups": [str(g) for g in args.groups], "stages": stage_log,
        "bootstrap": {"unit": "user", "draws": args.draws, "seed": SEED},
        "g1": g1.to_dict("records"), "g2": g2.to_dict("records"), "g3_spearman_vs_hr10": g3,
        "validity": {c.model: c.valid for c in cells}, "notes": {c.model: c.notes for c in cells},
        "source_sha256": sha256_file(HERE), "figures": figs}, indent=1, default=str))
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
