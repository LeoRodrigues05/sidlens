#!/usr/bin/env python
"""Time in the data and in existing intervention outputs (CPU only).

Protocol: `protocol.md` beside this file, declared before any model output was
split by time. Part D reads the frozen data only. Part M re-reads the per-row
outputs of exp3, exp4, exp6 and exp7, whose sha256s are recorded, and splits
them by the timestamps from `sidlens.data.timestamps`.

Traps, each with its guard
--------------------------
* **Joining by row number across cohorts.** AR rows join the time table
  through `example_id` and DiffGRM rows through `user`, and each join is
  checked to be one-to-one and complete.
* **A stale results directory.** Every result file read is hashed into
  `inputs.json`, and a missing one raises. There is no fallback to another run.
* **Ratios with small denominators.** rho divides two differences of rates. Its
  interval comes from the same bootstrap draws as its parts, so an unstable
  denominator shows up as a wide interval.

    python run.py --out $SIDLENS_WORK/derived/retrospective/exp6_time_structure/<job-id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.data import timestamps as T                                 # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402

SEED, DRAWS = 20260930, 2000
BINS = [name for _, name in T.GAP_BINS]
AR_CELLS = {"rqkmeans_3codebook_128": 0, "rqvae_4codebook_128": 1}
DIFF_CELLS = {"diff-next1-rqkmeans-3cb-128": 0, "diff-next1-rqvae-4cb-128": 1}
C = paths.DERIVED / "controlled"
SOURCES = {
    "exp3_test": (C / "exp3_ar_history_patching" / "280961", ("clean.parquet", "part_a.parquet")),
    "exp4_valid": (C / "exp4_ar_copy_circuit" / "281261", ("clean.parquet", "part_r.parquet")),
    "exp4_test": (C / "exp4_ar_copy_circuit" / "281394", ("clean.parquet", "part_r.parquet")),
    "exp6_archived": (C / "exp6_ar_copy_in_decoding" / "283114",
                      ("predictions_B.parquet", "predictions_C_all.parquet")),
    "exp7": (C / "exp7_diffusion_history_use" / "283124", ("clean.parquet", "part_a.parquet")),
}
EXP6_PLAIN = {0: C / "exp6_ar_copy_in_decoding" / "283586" / "cell-00",
              1: C / "exp6_ar_copy_in_decoding" / "283621" / "cell-01"}
HASHES: dict[str, str] = {}


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required result {path} is missing; no fallback")
    HASHES[str(path.relative_to(paths.WORK))] = sha256_file(path)
    return pd.read_parquet(path)


def d0(sid: str, n: int) -> int:
    return P.parse_sid(sid, n)[0]


# ------------------------------------------------------------------ tables --
def ar_tables(variant: str, split: str, ev):
    """Row table (time + SID facts) and history-pair table for one AR variant x split."""
    ex = P.load_examples(variant, "next-item", split)
    n = int(variant.split("_")[1][0])
    rows, hist = T.ar_row_times(ex, ev)
    by = {e.example_id: e for e in ex}
    rows["gap1_bin"] = T.gap_bin(rows.gap1_days)
    rows["ct"] = [d0(by[i].target_sids[0], n) for i in rows.example_id]
    rows["c1"] = [d0(by[i].history_sids[-1], n) for i in rows.example_id]
    rows["match0_r1"] = rows.ct == rows.c1
    rows["repeat_sid"] = [by[i].target_sids[0] in by[i].history_sids for i in rows.example_id]
    rows["target_type"] = np.select(
        [rows.dup_target, rows.repeat_sid & (rows.gap1_days == 0), rows.repeat_sid,
         rows.gap1_days == 0], ["dup", "repeat_same_day", "repeat_later", "new_same_day"], "new_later")
    hist["gap_bin"] = T.gap_bin(hist.gap_days)
    hc = [d0(by[i].history_sids[k], n) for i, k in zip(hist.example_id, hist.k)]
    ht = [by[i].target_sids[0] for i in hist.example_id]
    hs = [by[i].history_sids[k] for i, k in zip(hist.example_id, hist.k)]
    hist["match0"] = np.array(hc) == np.array([d0(s, n) for s in ht])
    hist["sid_eq"] = np.array(hs) == np.array(ht)
    hist = hist.merge(rows[["example_id", "user_id"]], on="example_id", validate="many_to_one")
    return ex, rows, hist


def diff_tables(ckpt: str, ev):
    from sidlens.data.diffusion_eval import load_eval_cohort
    from sidlens.registry.diffusion import load_runtime
    cohort = load_eval_cohort(load_runtime()[ckpt])
    rows, hist = T.diffusion_row_times(cohort, ev)
    L = cohort.history_lengths
    rows["gap1_bin"] = T.gap_bin(rows.gap1_days)
    rows["ct"] = cohort.target_sids[:, 0]
    rows["c1"] = cohort.histories[np.arange(len(L)), L - 1, 0]
    rows["match0_r1"] = rows.ct == rows.c1
    tgt = {r: tuple(cohort.target_sids[r]) for r in range(len(L))}
    rows["repeat_sid"] = [any(tuple(cohort.histories[r, k]) == tgt[r] for k in range(L[r])) for r in range(len(L))]
    ur = dict(zip(rows.user, rows.user_row))
    hist["user_row"] = hist.user.map(ur)
    hist["gap_bin"] = T.gap_bin(hist.gap_days)
    hist["match0"] = [cohort.histories[r, k, 0] == cohort.target_sids[r, 0] for r, k in zip(hist.user_row, hist.k)]
    return cohort, rows, hist


# ------------------------------------------------------------------ helpers --
def logit_r2(df: pd.DataFrame, y: str, factors: list[str]) -> float:
    """McFadden R^2 of an unpenalised logistic model on one-hot factors."""
    from sklearn.linear_model import LogisticRegression
    yv = df[y].to_numpy().astype(int)
    p0 = yv.mean()
    ll0 = len(yv) * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    X = pd.get_dummies(df[factors].astype(str), drop_first=True).to_numpy(float)
    m = LogisticRegression(C=np.inf, max_iter=2000).fit(X, yv)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    ll = float(np.sum(yv * np.log(p) + (1 - yv) * np.log(1 - p)))
    return 1.0 - ll / ll0


def rho(boot: UserBootstrap, df: pd.DataFrame, c: str, v: str) -> dict:
    """(c_same - c_>365) / (v_same - v_>365), all four means from the same draws."""
    a, b = df[df.gap1_bin == "same day"], df[df.gap1_bin == ">365 d"]
    parts = [boot.mean_draws(q.user_id, q[col].astype(float)) for q in (a, b) for col in (c, v)]
    (ca, ca_b, _), (va, va_b, _), (cb, cb_b, _), (vb, vb_b, _) = parts
    with np.errstate(invalid="ignore", divide="ignore"):
        est, bs = (ca - cb) / (va - vb), (ca_b - cb_b) / (va_b - vb_b)
    return UserBootstrap._summ(est, bs, len(a) + len(b), df.user_id.nunique())


# ------------------------------------------------------------------ parts --
def part_d(ev, out_rows: list, tables: dict):
    for variant in AR_CELLS:
        for split in ("train", "valid", "test"):
            _, rows, hist = tables[(variant, split)]
            tag = {"part": "D1", "cohort": f"AR {split}", "variant": variant}
            for b in BINS:
                out_rows.append({**tag, "estimand": f"share gap1 {b}", "est": float((rows.gap1_bin == b).mean()),
                                 "n_rows": len(rows)})
            for k in ("tie12", "dup_target", "repeat_item", "repeat_sid", "same_day_as_target"):
                out_rows.append({**tag, "estimand": f"share {k}", "est": float(rows[k].mean()), "n_rows": len(rows)})
            for t_, g in rows.groupby("target_type"):
                out_rows.append({**tag, "estimand": f"share target_type {t_}", "est": len(g) / len(rows),
                                 "n_rows": len(rows)})
            out_rows.append({**tag, "estimand": "median gap1 days", "est": float(rows.gap1_days.median()),
                             "n_rows": len(rows)})
            if split in ("train", "test"):
                for (r, b), g in hist[hist.recency <= 10].groupby(["recency", "gap_bin"]):
                    out_rows.append({"part": "D3", "cohort": f"AR {split}", "variant": variant, "recency": int(r),
                                     "gap_bin": b, "estimand": "P(match0)", "est": float(g.match0.mean()),
                                     "n_rows": len(g)})
                    out_rows.append({"part": "D3", "cohort": f"AR {split}", "variant": variant, "recency": int(r),
                                     "gap_bin": b, "estimand": "P(sid_eq)", "est": float(g.sid_eq.mean()),
                                     "n_rows": len(g)})
                r_pos = logit_r2(hist, "match0", ["recency"])
                r_time = logit_r2(hist, "match0", ["gap_bin"])
                r_both = logit_r2(hist, "match0", ["recency", "gap_bin"])
                for name, v in (("R2 recency", r_pos), ("R2 gap_bin", r_time), ("R2 both", r_both),
                                ("R2 gain time beyond position", r_both - r_pos),
                                ("R2 gain position beyond time", r_both - r_time),
                                ("X1 ratio gain_time / R2 recency", (r_both - r_pos) / r_pos)):
                    out_rows.append({"part": "D3", "cohort": f"AR {split}", "variant": variant,
                                     "estimand": name, "est": float(v), "n_rows": len(hist)})
    for ckpt in DIFF_CELLS:
        _, rows, hist = tables[ckpt]
        tag = {"part": "D1", "cohort": "DiffGRM", "variant": ckpt}
        for b in BINS:
            out_rows.append({**tag, "estimand": f"share gap1 {b}", "est": float((rows.gap1_bin == b).mean()),
                             "n_rows": len(rows)})
        for k in ("tie12", "dup_target", "repeat_item", "repeat_sid"):
            out_rows.append({**tag, "estimand": f"share {k}", "est": float(rows[k].mean()), "n_rows": len(rows)})
        r_pos, r_time = logit_r2(hist, "match0", ["recency"]), logit_r2(hist, "match0", ["gap_bin"])
        r_both = logit_r2(hist, "match0", ["recency", "gap_bin"])
        for name, v in (("R2 recency", r_pos), ("R2 gap_bin", r_time), ("R2 both", r_both),
                        ("R2 gain time beyond position", r_both - r_pos),
                        ("X1 ratio gain_time / R2 recency", (r_both - r_pos) / r_pos)):
            out_rows.append({"part": "D3", "cohort": "DiffGRM", "variant": ckpt, "estimand": name,
                             "est": float(v), "n_rows": len(hist)})
    # D2: within-day ASIN order
    from sidlens.data.sids import load_item2id
    asin = {i: a for a, i in load_item2id(T.CATEGORY).items()}
    n = asc = 0
    for u, seq in ev.sequences.items():
        t = ev.times[u]
        for p in range(1, len(seq)):
            if t[p] == t[p - 1] and seq[p] != seq[p - 1]:
                n += 1
                asc += asin[seq[p - 1]] < asin[seq[p]]
    out_rows.append({"part": "D2", "cohort": "all events", "estimand": "same-day consecutive pairs ASIN ascending",
                     "est": asc / n, "n_rows": n})


def part_m1(tables, res: list):
    """Copy calibration: model copy rate vs data rate by gap1 bin."""
    for variant, cell in AR_CELLS.items():
        for src, split in (("exp3_test", "test"), ("exp4_valid", "valid")):
            root, _ = SOURCES[src]
            clean = read(root / f"cell-{cell:02d}" / "clean.parquet")
            c0 = clean[clean.digit == 0].set_index("example_id")
            _, rows, _ = tables[(variant, split)]
            x = rows.set_index("example_id").join(c0[["top1", "logp_codes"]], how="inner")
            if len(x) != len(rows):
                raise ValueError(f"{src} cell {cell}: {len(x)} of {len(rows)} rows joined")
            x = x.reset_index()
            x["copied"] = x.top1 == x.c1
            x["golden_top1"] = x.top1 == x.ct
            boot = UserBootstrap(x.user_id, draws=DRAWS, seed=SEED)
            tag = {"part": "M1", "cohort": f"AR {split}", "variant": variant}
            for b in BINS + ["all"]:
                q = x if b == "all" else x[x.gap1_bin == b]
                for name, col in (("copy rate c", "copied"), ("data rate v", "match0_r1"),
                                  ("golden top1", "golden_top1")):
                    res.append({**tag, "gap_bin": b, "estimand": name, **boot.mean(q.user_id, q[col].astype(float))})
                res.append({**tag, "gap_bin": b, "estimand": "over-copy c - v",
                            **boot.mean(q.user_id, q.copied.astype(float) - q.match0_r1)})
                res.append({**tag, "gap_bin": b, "estimand": "golden logp d0", **boot.mean(q.user_id, q.logp_codes)})
            res.append({**tag, "gap_bin": "same/>365", "estimand": "rho", **rho(boot, x, "copied", "match0_r1")})
    for ckpt, cell in DIFF_CELLS.items():
        root, _ = SOURCES["exp7"]
        clean = read(root / f"cell-{cell:02d}" / "clean.parquet")
        c0 = clean[(clean.state == "full") & (clean.digit == 0)].set_index("user_row")
        _, rows, _ = tables[ckpt]
        x = rows.set_index("user_row").join(c0[["top1", "logp"]], how="inner").reset_index()
        if len(x) != len(rows):
            raise ValueError(f"exp7 cell {cell}: {len(x)} of {len(rows)} users joined")
        x["user_id"] = x.user
        x["copied"] = x.top1 == x.c1
        x["golden_top1"] = x.top1 == x.ct
        boot = UserBootstrap(x.user_id, draws=DRAWS, seed=SEED)
        tag = {"part": "M1", "cohort": "DiffGRM", "variant": ckpt}
        for b in BINS + ["all"]:
            q = x if b == "all" else x[x.gap1_bin == b]
            for name, col in (("copy rate c", "copied"), ("data rate v", "match0_r1"), ("golden top1", "golden_top1")):
                res.append({**tag, "gap_bin": b, "estimand": name, **boot.mean(q.user_id, q[col].astype(float))})
            res.append({**tag, "gap_bin": b, "estimand": "over-copy c - v",
                        **boot.mean(q.user_id, q.copied.astype(float) - q.match0_r1)})
            res.append({**tag, "gap_bin": b, "estimand": "golden logp d0", **boot.mean(q.user_id, q.logp)})
        res.append({**tag, "gap_bin": "same/>365", "estimand": "rho", **rho(boot, x, "copied", "match0_r1")})


def part_m2(tables, res: list):
    """Replacement effect at recency 1-2 (m = 0, digit 0) by the item's age, split by match0."""
    for variant, cell in AR_CELLS.items():
        root, _ = SOURCES["exp3_test"]
        clean = read(root / f"cell-{cell:02d}" / "clean.parquet")
        a = read(root / f"cell-{cell:02d}" / "part_a.parquet")
        a = a[(a.m == 0) & (a.digit == 0) & (a.recency <= 2)]
        c0 = clean[clean.digit == 0].set_index("example_id").logp_codes.rename("clean")
        a = a.join(c0, on="example_id")
        a["delta"] = a.clean - a.logp_codes
        _, _, hist = tables[(variant, "test")]
        a = a.merge(hist[["example_id", "recency", "gap_bin", "match0"]], on=["example_id", "recency"],
                    validate="many_to_one")
        boot = UserBootstrap(a.user_id, draws=DRAWS, seed=SEED)
        _m2_estimands(a, boot, {"part": "M2", "cohort": "AR test", "variant": variant}, res)
    for ckpt, cell in DIFF_CELLS.items():
        root, _ = SOURCES["exp7"]
        clean = read(root / f"cell-{cell:02d}" / "clean.parquet")
        a = read(root / f"cell-{cell:02d}" / "part_a.parquet")
        a = a[(a.m == 0) & (a.state == "full") & (a.digit == 0) & (a.r <= 2)]
        c0 = clean[(clean.state == "full") & (clean.digit == 0)].set_index("user_row").logp.rename("clean")
        a = a.join(c0, on="user_row")
        a["delta"] = a.clean - a.logp
        a["recency"] = a.r
        _, _, hist = tables[ckpt]
        a = a.merge(hist[["user_row", "recency", "gap_bin", "match0"]], on=["user_row", "recency"],
                    validate="many_to_one")
        a["user_id"] = a.user
        boot = UserBootstrap(a.user_id, draws=DRAWS, seed=SEED)
        _m2_estimands(a, boot, {"part": "M2", "cohort": "DiffGRM", "variant": ckpt}, res)


def _m2_estimands(a, boot, tag, res):
    for (r, mt), g in a.groupby(["recency", "match0"]):
        for b in BINS + ["all"]:
            q = g if b == "all" else g[g.gap_bin == b]
            if len(q):
                res.append({**tag, "recency": int(r), "match0": bool(mt), "gap_bin": b,
                            "estimand": "delta d0", **boot.mean(q.user_id, q.delta)})
        s, o = g[g.gap_bin == "same day"], g[g.gap_bin == ">365 d"]
        if len(s) and len(o):
            res.append({**tag, "recency": int(r), "match0": bool(mt), "gap_bin": "same - >365",
                        "estimand": "delta(same day) - delta(>365 d)",
                        **boot.paired_difference((s.user_id, s.delta), (o.user_id, o.delta))})


def part_m3(tables, res: list):
    """exp6: baseline HR@10 and the copy-knockout change by target type."""
    for variant, cell in AR_CELLS.items():
        _, rows, _ = tables[(variant, "test")]
        ex = {e.example_id: e for e in P.load_examples(variant, "next-item", "test")}
        rows = rows.set_index("example_id")
        for decoder, files in (("archived", (SOURCES["exp6_archived"][0] / f"cell-{cell:02d}" / "predictions_B.parquet",
                                             SOURCES["exp6_archived"][0] / f"cell-{cell:02d}" / "predictions_C_all.parquet")),
                               ("plain", (EXP6_PLAIN[cell] / "predictions_B_plain.parquet",
                                          EXP6_PLAIN[cell] / "predictions_C_all_plain.parquet"))):
            b, c = (read(f).set_index("example_id").predict for f in files)
            x = rows.loc[b.index].copy()
            tg = [ex[i].target_sids[0] for i in x.index]
            x["hit_B"] = [float(t in list(p)[:10]) for p, t in zip(b, tg)]
            x["hit_C"] = [float(t in list(p)[:10]) for p, t in zip(c.loc[b.index], tg)]
            x["dhit"] = x.hit_C - x.hit_B
            x = x.reset_index()
            if len(x) != 3681:
                raise ValueError(f"exp6 {decoder} cell {cell}: {len(x)} rows")
            boot = UserBootstrap(x.user_id, draws=DRAWS, seed=SEED)
            tag = {"part": "M3", "cohort": "AR test", "variant": variant, "decoder": decoder}
            for t_ in ("all", "dup", "repeat_same_day", "repeat_later", "new_same_day", "new_later"):
                q = x if t_ == "all" else x[x.target_type == t_]
                res.append({**tag, "target_type": t_, "estimand": "HR@10 baseline", **boot.mean(q.user_id, q.hit_B)})
                res.append({**tag, "target_type": t_, "estimand": "dHR@10 C_all - B", **boot.mean(q.user_id, q.dhit)})
                res.append({**tag, "target_type": t_, "estimand": "share of rows", "est": len(q) / len(x),
                            "n_rows": len(q)})
                if t_ != "all":
                    res.append({**tag, "target_type": t_, "estimand": "share of all-rows HR@10 loss",
                                **boot.ratio(x.user_id, x.dhit * (x.target_type == t_), x.dhit)})
                    res.append({**tag, "target_type": t_, "estimand": "share of baseline hits",
                                **boot.ratio(x.user_id, x.hit_B * (x.target_type == t_), x.hit_B)})
            sd = x.gap1_days == 0
            res.append({**tag, "target_type": "gap1 = 0", "estimand": "X4 share of all-rows HR@10 loss",
                        **boot.ratio(x.user_id, x.dhit * sd, x.dhit)})
            res.append({**tag, "target_type": "gap1 = 0", "estimand": "share of baseline hits",
                        **boot.ratio(x.user_id, x.hit_B * sd, x.hit_B)})


def part_m4(tables, res: list):
    """exp4 H1 on valid vs test, raw and post-stratified to valid's composition."""
    for variant, cell in AR_CELLS.items():
        per = {}
        for src, split in (("exp4_valid", "valid"), ("exp4_test", "test")):
            root, _ = SOURCES[src]
            clean = read(root / f"cell-{cell:02d}" / "clean.parquet")
            r = read(root / f"cell-{cell:02d}" / "part_r.parquet")
            r = r[(r.digit == r.d) & (r.d >= 1)]
            cl = clean[["i", "digit", "logp_codes"]].rename(columns={"logp_codes": "clean_logp"})
            r = r.merge(cl, on=["i", "digit"], validate="many_to_one")
            r["delta"] = r.clean_logp - r.logp_codes
            _, rows, _ = tables[(variant, split)]
            r = r.merge(rows[["example_id", "gap1_days", "dup_target"]], on="example_id", validate="many_to_one")
            r["stratum"] = np.select([r.dup_target, r.gap1_days == 0], ["dup", "same_day"], "later")
            per[split] = r
        boots = {s: UserBootstrap(per[s].user_id, draws=DRAWS, seed=SEED) for s in per}
        h1 = {}
        for split, r in per.items():
            b = boots[split]
            tag = {"part": "M4", "cohort": f"AR {split}", "variant": variant}
            m, nm = r[r.match], r[~r.match]
            h1[split] = b.paired_difference((m.user_id, m.delta), (nm.user_id, nm.delta))
            res.append({**tag, "stratum": "all", "estimand": "H1 match - nonmatch", **h1[split]})
            for s_, g in r.groupby("stratum"):
                gm, gn = g[g.match], g[~g.match]
                res.append({**tag, "stratum": s_, "estimand": "share of match pairs", "est": len(gm) / max(len(m), 1),
                            "n_rows": len(gm)})
                if len(gm) and len(gn):
                    res.append({**tag, "stratum": s_, "estimand": "H1 match - nonmatch",
                                **b.paired_difference((gm.user_id, gm.delta), (gn.user_id, gn.delta))})
                if len(gm):
                    res.append({**tag, "stratum": s_, "estimand": "delta | match", **b.mean(gm.user_id, gm.delta)})
        # post-stratify test to valid's (stratum | match) composition; weights fixed at valid's point shares
        v, t = per["valid"], per["test"]
        bt = boots["test"]
        est_parts, draw_parts = [], []
        for mt, sign in ((True, 1.0), (False, -1.0)):
            w = v[v.match == mt].stratum.value_counts(normalize=True)
            for s_, ws in w.items():
                q = t[(t.match == mt) & (t.stratum == s_)]
                if not len(q):
                    raise ValueError(f"post-stratification: test has no {s_} pairs with match={mt}")
                e, d_, _ = bt.mean_draws(q.user_id, q.delta)
                est_parts.append(sign * ws * e)
                draw_parts.append(sign * ws * d_)
        ps = UserBootstrap._summ(sum(est_parts), sum(draw_parts), len(t), t.user_id.nunique())
        res.append({"part": "M4", "cohort": "AR test", "variant": variant, "stratum": "post-stratified to valid",
                    "estimand": "H1 match - nonmatch", **ps})
        ev_, bv, _ = boots["valid"].mean_draws(v[v.match].user_id, v[v.match].delta)
        ev2, bv2, _ = boots["valid"].mean_draws(v[~v.match].user_id, v[~v.match].delta)
        et, bt_, _ = bt.mean_draws(t[t.match].user_id, t[t.match].delta)
        et2, bt2, _ = bt.mean_draws(t[~t.match].user_id, t[~t.match].delta)
        hv, hv_b = ev_ - ev2, bv - bv2
        for name, (e, b_) in (("raw", (et - et2, bt_ - bt2)), ("post-stratified", (ps["est"], sum(draw_parts)))):
            with np.errstate(invalid="ignore", divide="ignore"):
                res.append({"part": "M4", "cohort": "AR valid/test", "variant": variant, "stratum": name,
                            "estimand": "ratio H1 valid / test",
                            **UserBootstrap._summ(hv / e, hv_b / b_, len(v) + len(t), 0)})


def figures(res: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sidlens.viz import style
    style.apply()
    made = []
    col = lambda v: style.QUANTIZER_COLOR["rqvae" if "rqvae" in v else "rqkmeans"]      # noqa: E731
    ramp = dict(zip(BINS, [style.SEQUENTIAL[-1], style.SEQUENTIAL[-4], style.SEQUENTIAL[-7], style.SEQUENTIAL[1]]))
    # fig1: data informativeness by recency x gap bin
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, v in zip(axes, AR_CELLS):
        q = res[(res.part == "D3") & (res.cohort == "AR test") & (res.variant == v) & (res.estimand == "P(match0)")]
        for b in BINS:
            s = q[(q.gap_bin == b) & (q.n_rows >= 30)].sort_values("recency")
            ax.plot(s.recency, s.est, marker="o", ms=style.MARKER, color=ramp[b], label=b)
        ax.set_title(f"{v}: history item shares target's first digit", fontsize=7)
        ax.set_xlabel("recency (1 = most recent)")
    axes[0].set_ylabel("P(match0), test split")
    style.legend_outside(axes[-1], title="item age")
    made += [str(p) for p in style.save(fig, out / "fig1_informativeness_position_time")]
    # fig2: copy calibration
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9), sharey=True)
    for ax, coh in zip(axes, ("AR test", "AR valid", "DiffGRM")):
        q = res[(res.part == "M1") & (res.cohort == coh)]
        for v in q.variant.unique():
            for name, ls in (("copy rate c", "-"), ("data rate v", ":")):
                s = q[(q.variant == v) & (q.estimand == name)].set_index("gap_bin").loc[BINS]
                ax.errorbar(range(len(BINS)), s.est, yerr=[s.est - s.lo, s.hi - s.est], color=col(v), ls=ls,
                            marker="o", ms=style.MARKER, capsize=0, label=f"{v} {name.split()[0]}")
        ax.set_xticks(range(len(BINS)), BINS, fontsize=6)
        ax.set_title(f"{coh}: model copies r1's first digit vs data", fontsize=7)
        ax.set_xlabel("gap between most recent item and target")
    axes[0].set_ylabel("share of rows")
    style.legend_outside(axes[-1], fontsize=5)
    made += [str(p) for p in style.save(fig, out / "fig2_copy_calibration")]
    # fig3: exp6 by target type
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    types = ["dup", "repeat_same_day", "repeat_later", "new_same_day", "new_later"]
    for ax, name in zip(axes, ("HR@10 baseline", "dHR@10 C_all - B")):
        for k, v in enumerate(AR_CELLS):
            s = res[(res.part == "M3") & (res.variant == v) & (res.decoder == "archived") &
                    (res.estimand == name)].set_index("target_type").loc[types]
            ax.errorbar(np.arange(len(types)) + (k - 0.5) * 0.25, s.est, yerr=[s.est - s.lo, s.hi - s.est],
                        fmt="o", color=col(v), ms=style.MARKER, capsize=0, label=v)
        style.null_line(ax, 0, "")
        ax.set_xticks(range(len(types)), types, fontsize=6, rotation=20)
        ax.set_title(f"exp6 (archived decoder): {name}", fontsize=7)
    style.legend_outside(axes[-1])
    made += [str(p) for p in style.save(fig, out / "fig3_exp6_by_target_type")]
    plt.close("all")
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    ev = T.load_event_times()
    tables = {(v, s): ar_tables(v, s, ev) for v in AR_CELLS for s in ("train", "valid", "test")}
    tables.update({c: diff_tables(c, ev) for c in DIFF_CELLS})
    for key, tb in tables.items():
        name = "__".join(key) if isinstance(key, tuple) else key
        tb[1].to_parquet(args.out / f"time_rows__{name}.parquet", index=False)
    res: list[dict] = []
    part_d(ev, res, tables)
    print(f"[D] {time.time() - t0:.0f}s", flush=True)
    part_m1(tables, res)
    print(f"[M1] {time.time() - t0:.0f}s", flush=True)
    part_m2(tables, res)
    print(f"[M2] {time.time() - t0:.0f}s", flush=True)
    part_m3(tables, res)
    print(f"[M3] {time.time() - t0:.0f}s", flush=True)
    part_m4(tables, res)
    print(f"[M4] {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(res)
    res.to_csv(args.out / "estimates.csv", index=False)
    figs = figures(res, args.out / "figures")
    (args.out / "inputs.json").write_text(json.dumps({"timestamps": ev.report, "results_sha256": HASHES,
                                                      "seed": SEED, "draws": DRAWS, "gap_bins": T.GAP_BINS},
                                                     indent=1))
    (args.out / "result.json").write_text(json.dumps({"figures": figs, "seconds": round(time.time() - t0, 1)},
                                                     indent=1))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
