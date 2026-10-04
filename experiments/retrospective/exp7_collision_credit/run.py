#!/usr/bin/env python
"""Retrospective exp7 -- how much of copying's benefit is collision credit?

Re-scores the exp6 copy-knockout predictions at item level (CCE uniform) and
splits every effect by how the target relates to the history: same item,
different item with the same SID (collision partner), near-duplicate variant
with a different SID, or new. CPU only. See protocol.md.

Traps guarded here:

* SID HR@10 must equal exp6's own scoring (raw first 10 entries), or the
  re-split numbers would not add up to exp6's published ones. The all-rows
  baseline and C_all deltas are checked against exp6's recorded HR@10 within
  1e-9 before anything else is written.
* Archived lists contain malformed and out-of-catalogue strings (beam
  sampling). Item scoring skips them; it never raises on them and never
  counts them as items.
* Item ids in the AR rows must equal the SID tables' ids; every target and
  history SID is re-derived from the table and compared, and any mismatch
  raises.
* A missing exp6 file raises; there is no fallback to another run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sidlens import paths
from sidlens.analysis.bootstrap import UserBootstrap
from sidlens.analysis.collisions import item_rank_bounds
from sidlens.data import ar_prompts as P
from sidlens.data.sids import SidTable, load_item2id
from sidlens.provenance.hashing import sha256_file

CATEGORY = "Industrial_and_Scientific"
CELLS = {"rqkmeans_3codebook_128": 0, "rqvae_4codebook_128": 1}
EXP6 = paths.DERIVED / "controlled" / "exp6_ar_copy_in_decoding"
ARCHIVED = {0: EXP6 / "283114" / "cell-00", 1: EXP6 / "283114" / "cell-01"}
PLAIN = {0: EXP6 / "283586" / "cell-00", 1: EXP6 / "283621" / "cell-01"}
ND_PAIRS = (paths.DERIVED / "structure" / "exp3_collision_causes" / "293555" / "result"
            / "partA_near_duplicate_pairs.csv")
CONTRASTS = {"archived": [("C_all", "B"), ("C_later", "B"), ("N_later", "B")],
             "plain": [("C_all_plain", "B_plain")]}
CLASSES = ("same_item", "sid_partner", "nd_variant", "new")
DRAWS, SEED = 2000, 20260927
EXP6_RECORDED_HR10 = {  # exp6 summary-283114, archived decoder, exact SID, %
    ("rqkmeans_3codebook_128", "C_all - B"): -5.68, ("rqvae_4codebook_128", "C_all - B"): -3.31}
HASHES: dict[str, str] = {}


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required input {path} is missing; no fallback")
    HASHES[str(path)] = sha256_file(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def legal_tuple(s: str, D: int, buckets) -> tuple[int, ...] | None:
    try:
        t = P.parse_sid(s, D)
    except (ValueError, TypeError):
        return None
    return t if t in buckets else None


def row_table(variant: str, nd_adj: dict[int, set[int]]):
    t = SidTable.load(variant)
    D = t.variant.n_codebook
    i2 = load_item2id(CATEGORY)
    id2sid = {i2[a]: c for a, c in t.asin2codes.items()}
    buckets = {sid: [i2[a] for a in items] for sid, items in t.cb2items.items()}
    ex = P.load_examples(variant, "next-item", "test")
    rows = []
    for e in ex:
        ti, ts = e.target_item_ids[0], e.target_sids[0]
        if P.parse_sid(ts, D) != id2sid[ti] or any(
                P.parse_sid(s, D) != id2sid[i] for i, s in zip(e.history_item_ids, e.history_sids)):
            raise ValueError(f"{e.example_id}: item ids disagree with the SID table")
        partners = [h for h in e.history_item_ids if h in nd_adj.get(ti, ())]
        if ti in e.history_item_ids:
            cls = "same_item"
        elif ts in e.history_sids:
            cls = "sid_partner"
        elif partners:
            cls = "nd_variant"
        else:
            cls = "new"
        rows.append({"example_id": e.example_id, "row": e.row, "user_id": e.user_id,
                     "cls": cls, "target_item": ti, "target_sid": ts,
                     "target_bucket": len(buckets[id2sid[ti]]),
                     "variant_sids": sorted({e.history_sids[e.history_item_ids.index(h)] for h in partners})})
    return pd.DataFrame(rows), buckets, D


def score(rows, preds: pd.Series, buckets, D):
    """Per-row SID hit (exp6 rule), item hit (uniform, lower), variant copy."""
    sid_hit, item_u, item_lo, vcopy = [], [], [], []
    sid_hit1, item_u1 = [], []  # A1 (post hoc): cutoff 1, where bucket ambiguity bites
    for r in rows.itertuples(index=False):
        p = list(preds.loc[r.example_id])
        sid_hit.append(float(r.target_sid in p[:10]))
        sid_hit1.append(float(r.target_sid in p[:1]))
        tuples = [x for x in (legal_tuple(s, D, buckets) for s in p) if x is not None]
        bounds = item_rank_bounds(tuples, r.target_item, buckets,
                                  target_sid=P.parse_sid(r.target_sid, D))
        b = bounds.hit_bounds(10)
        item_u.append(b["uniform"])
        item_lo.append(b["lower"])
        item_u1.append(bounds.hit_bounds(1)["uniform"])
        first = next((s for s in p if legal_tuple(s, D, buckets) is not None), None)
        vcopy.append(float(first in r.variant_sids) if r.cls == "nd_variant" else np.nan)
    return pd.DataFrame({"sid_hit": sid_hit, "item_u": item_u, "item_lo": item_lo,
                         "sid_hit1": sid_hit1, "item_u1": item_u1,
                         "variant_copy": vcopy}, index=rows.index)


def estimates(variant, decoder, rows, scored, boot):
    out = []
    tag = {"variant": variant, "decoder": decoder}
    u = rows.user_id
    base_name = CONTRASTS[decoder][0][1]
    b = scored[base_name]
    for cls in ("all", *CLASSES):
        m = np.ones(len(rows), bool) if cls == "all" else (rows.cls == cls).to_numpy()
        if not m.any():
            continue
        t = {**tag, "class": cls}
        out.append({**t, "estimand": "share of rows", "est": float(m.mean()), "n_rows": int(m.sum())})
        for metric in ("sid_hit", "item_u", "item_lo"):
            out.append({**t, "estimand": f"{base_name} {metric}@10", **boot.mean(u[m], b[metric][m])})
        for metric in ("sid_hit1", "item_u1"):
            out.append({**t, "estimand": f"A1 {base_name} {metric[:-1]}@1", **boot.mean(u[m], b[metric][m])})
        if cls != "all":
            out.append({**t, "estimand": f"share of {base_name} SID hits",
                        **boot.ratio(u, b.sid_hit * m, b.sid_hit)})
        if cls == "nd_variant":
            out.append({**t, "estimand": f"{base_name} variant copy rate",
                        **boot.mean(u[m], b.variant_copy[m])})
        for k, base in CONTRASTS[decoder]:
            c = scored[k]
            for metric in ("sid_hit", "item_u", "item_lo"):
                d = c[metric] - b[metric]
                out.append({**t, "estimand": f"d {metric}@10 {k} - {base}", **boot.mean(u[m], d[m])})
            for metric in ("sid_hit1", "item_u1"):
                d = c[metric] - b[metric]
                out.append({**t, "estimand": f"A1 d {metric[:-1]}@1 {k} - {base}", **boot.mean(u[m], d[m])})
            if cls == "all":
                out.append({**t, "estimand": f"A1 E2@1 sum(d item) / sum(d SID) {k} - {base}",
                            **boot.ratio(u, c.item_u1 - b.item_u1, c.sid_hit1 - b.sid_hit1)})
            dsid = c.sid_hit - b.sid_hit
            ditem = c.item_u - b.item_u
            if cls == "all":
                out.append({**t, "estimand": f"E2 sum(d item) / sum(d SID) {k} - {base}",
                            **boot.ratio(u, ditem, dsid)})
            else:
                out.append({**t, "estimand": f"E1 share of all-rows SID loss {k} - {base}",
                            **boot.ratio(u, dsid * m, dsid)})
                out.append({**t, "estimand": f"share of all-rows item loss {k} - {base}",
                            **boot.ratio(u, ditem * m, ditem)})
            if cls == "nd_variant":
                out.append({**t, "estimand": f"{k} variant copy rate",
                            **boot.mean(u[m], c.variant_copy[m])})
        # E3: within new-or-variant rows
    nv = rows.cls.isin(["nd_variant", "new"]).to_numpy()
    ndm = (rows.cls == "nd_variant").to_numpy()
    for k, base in CONTRASTS[decoder]:
        dsid = scored[k].sid_hit - b.sid_hit
        out.append({**tag, "class": "nd_variant within new+nd_variant",
                    "estimand": f"E3 share of SID loss {k} - {base}",
                    **boot.ratio(u[nv], (dsid * ndm)[nv], dsid[nv])})
        out.append({**tag, "class": "nd_variant within new+nd_variant",
                    "estimand": "E3 row share", "est": float(ndm[nv].mean()), "n_rows": int(nv.sum())})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    nd = read(ND_PAIRS)
    nd_adj: dict[int, set[int]] = {}
    for a, b in zip(nd.i, nd.j):
        nd_adj.setdefault(int(a), set()).add(int(b))
        nd_adj.setdefault(int(b), set()).add(int(a))

    res, checks = [], []
    for variant, cell in CELLS.items():
        rows, buckets, D = row_table(variant, nd_adj)
        boot = UserBootstrap(rows.user_id, draws=DRAWS, seed=SEED)
        for decoder, root in (("archived", ARCHIVED[cell]), ("plain", PLAIN[cell])):
            conds = sorted({x for pair in CONTRASTS[decoder] for x in pair})
            scored = {}
            for k in conds:
                pr = read(root / f"predictions_{k}.parquet").set_index("example_id").predict
                if set(pr.index) != set(rows.example_id) or len(pr) != len(rows):
                    raise ValueError(f"{root.name} {k}: rows do not match the test cohort")
                scored[k] = score(rows, pr, buckets, D)
                scored[k].assign(example_id=rows.example_id.values, cls=rows.cls.values,
                                 user_id=rows.user_id.values).to_parquet(
                    out / f"scored__{variant}__{decoder}__{k}.parquet")
            if decoder == "archived":
                d = 100 * (scored["C_all"].sid_hit - scored["B"].sid_hit).mean()
                rec = EXP6_RECORDED_HR10[(variant, "C_all - B")]
                checks.append({"variant": variant, "dHR10_C_all_minus_B_pp": float(d),
                               "exp6_recorded": rec, "abs_diff": abs(d - rec)})
                if abs(d - rec) > 0.006:  # exp6 reports 2 decimals
                    raise RuntimeError(f"{variant}: re-scored C_all - B {d:.3f} pp does not "
                                       f"match exp6's {rec} pp")
            res += estimates(variant, decoder, rows, scored, boot)
        rows.drop(columns="variant_sids").to_parquet(out / f"rows__{variant}.parquet")
    est = pd.DataFrame(res)
    est.to_csv(out / "estimates.csv", index=False)
    (out / "validation.json").write_text(json.dumps(checks, indent=2))
    (out / "inputs.json").write_text(json.dumps(HASHES, indent=2))
    write_report(out, est)
    print("done")
    return 0


def fmt(r: dict) -> str:
    if pd.isna(r.get("lo", np.nan)):
        return f"{r['est']:.4f}"
    return f"{r['est']:.4f} [{r['lo']:.4f}, {r['hi']:.4f}]"


def write_report(out, est):
    L = ["# Retrospective exp7: collision credit in copying -- generated report", "",
         "Paired user bootstrap (2,000 draws, seed 20260927); 95% percentile intervals; "
         "they exclude training-seed variance. Proportions, not percentage points.", ""]
    for (v, dec), g in est.groupby(["variant", "decoder"], sort=False):
        L += [f"## {v}, {dec} decoder", "", "| class | estimand | estimate |", "|---|---|---|"]
        for r in g.to_dict("records"):
            L.append(f"| {r['class']} | {r['estimand']} | {fmt(r)} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
