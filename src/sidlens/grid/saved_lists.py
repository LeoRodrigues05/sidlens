"""Retrospective analyses of a grid run's saved test lists (both paradigms, both tasks).

The retrospective experiments read archived outputs whose formats differ per
paradigm and per sweep (retrospective exp2-exp4, exp6). Every grid run instead
saves one schema (`evaluation/predictions.jsonl`, OneDiffRec's
`ordered_targets_v1`): ranked lists of ordered SID tuples, keyed by
`example_id`, with each row's events. This module builds one row table per run
and computes, from it, the headline estimands of those experiments, so the
same numbers exist for every grid run:

    exp2  where the top-1 first goes wrong: share of rows whose first wrong
          digit is d (n = exact), per target slot
    exp3  SID-level hits against item-level bounds (lower / uniform tie-break /
          upper) at k = 1, 5, 10; hit rates of collided vs singleton targets
    exp4  next-two: P(slot 2 right | slot 1 right) - P(slot 2 right | slot 1
          wrong) for the top-1 pair, and slot-2 minus slot-1 hit@10 (association,
          not dependence, as in exp4)
    exp6  hits and the top-1's copy of the most recent item's first code by the
          review-day gap to it, against the data rate (the target's first code
          equals it); the share of hits supplied by duplicate records
    all   every metric by retrospective exp7's target class

Traps, each with its guard
--------------------------
1. **SID hits passed off as item hits.** Item metrics come only from
   `collisions.item_rank_bounds` over each SID's bucket in the run's own table.
2. **Rows joined by position.** Days, classes and saved rows join by
   `example_id`, one to one, or raise.
3. **Slot confusion in next-two lists.** A candidate is an ordered tuple; slot
   s is position s of it, and the pair is right only when both are.
4. **Unsettled days.** Office's one unsettled position marks its rows
   `time_ambiguous`; time splits drop them.
"""

from __future__ import annotations

import numpy as np

from sidlens.analysis.collisions import item_rank_bounds
from sidlens.data import ar_prompts as P
from sidlens.data import timestamps as T
from sidlens.grid import events, inputs as GI, predictions as GP

KS = (1, 5, 10)
SEED, DRAWS = 20261021, 2000


def row_table(e: dict):
    """One row per saved test row of the run, with its list parsed and every split variable."""
    import pandas as pd

    saved = GP.load(e)
    table = GI.sid_table(e)
    n = table.variant.n_codebook
    items = GI.item2id(e)
    buckets = {sid: tuple(sorted(items[a] for a in asins)) for sid, asins in table.cb2items.items()}
    code_of = {items[a]: c for a, c in table.asin2codes.items()}
    ids = list(saved)
    cls = events.target_classes(e, "test", ids).set_index("example_id")
    days, _ = events.row_times(e, "test", ids)
    days = days.set_index("example_id")
    rows = events.cohort_rows(e, "test")
    by_id = {r["example_id"]: r for r in rows}
    out = []
    for eid in ids:
        s, r = saved[eid], by_id[eid]
        hist = [code_of[items[a]] for a in r["history_item_ids"]]
        tgt = [code_of[items[a]] for a in r["target_item_ids"]]
        preds = [[P.parse_sid(x, n) for x in c["sids"]] for c in s["predictions"]]
        if [len(p) for p in preds] != [len(tgt)] * len(preds):
            raise ValueError(f"{eid}: a candidate does not have one SID per target slot")
        rec = {"example_id": eid, "user_id": r["user_id"], "n_slots": len(tgt), "n_digit": n, "hist_len": len(hist),
               "n_list": len(preds)}
        pair_rank = next((k + 1 for k, p in enumerate(preds) if p == tgt), 0)
        rec["pair_rank"] = pair_rank
        for slot, t in enumerate(tgt):
            sfx = "" if slot == 0 else str(slot + 1)
            col = [p[slot] for p in preds]
            uniq = list(dict.fromkeys(col))
            rank = next((k + 1 for k, p in enumerate(col) if p == t), 0)
            top = col[0] if col else None
            first_wrong = n if top == t else (next(d for d in range(n) if top[d] != t[d]) if top else -1)
            b = item_rank_bounds(uniq, items[r["target_item_ids"][slot]], buckets, target_sid=t)
            rec.update({f"rank{sfx}": rank, f"first_wrong{sfx}": first_wrong,
                        f"bucket{sfx}": len(buckets[t]),
                        **{f"item_{w}@{k}{sfx}": v for k in KS for w, v in b.hit_bounds(k).items()}})
        r1 = hist[-1]
        top = preds[0][0] if preds else None
        rec.update({"top1_d0_is_r1": bool(top is not None and top[0] == r1[0]),
                    "data_d0_is_r1": bool(tgt[0][0] == r1[0]),
                    "top1_in_history": bool(top is not None and top in set(hist)),
                    "top10_history_share": float(np.mean([p[0] in set(hist) for p in preds[:10]])) if preds else 0.0})
        out.append(rec)
    t = pd.DataFrame(out).set_index("example_id")
    t = t.join(cls.drop(columns=["user_id", "row"]), how="inner").join(
        days[["gap1_days", "same_day_as_target", "dup_target", "repeat_item", "time_ambiguous"]], how="inner")
    if len(t) != len(out):
        raise ValueError("class or day tables do not cover every saved row")
    t["gap1_bin"] = T.gap_bin(t.gap1_days.clip(lower=0))
    for k in KS:
        t[f"sid_hit@{k}"] = ((t["rank"] > 0) & (t["rank"] <= k)).astype(float)
        if (t.n_slots == 2).any():
            t[f"sid_hit@{k}2"] = ((t["rank2"] > 0) & (t["rank2"] <= k)).astype(float)
            t[f"pair_hit@{k}"] = ((t.pair_rank > 0) & (t.pair_rank <= k)).astype(float)
    return t.reset_index()


def _add(out, tag, boot, user, x, **kw):
    out.append({**tag, **kw, **boot.mean(user, np.asarray(x, dtype=float))})


def estimands(t, experiment: str, tag: dict) -> list[dict]:
    """The declared estimands of one retrospective experiment on a row table (module doc)."""
    from sidlens.analysis.bootstrap import UserBootstrap

    boot = UserBootstrap(t.user_id, draws=DRAWS, seed=SEED)
    u = t.user_id
    out: list[dict] = []
    slots = [""] + (["2"] if "rank2" in t else [])
    if experiment in ("exp2", "all"):
        n = int(t.n_digit.iloc[0])                 # the depth, not the data's maximum
        for sfx in slots:
            for d in range(n + 1):
                _add(out, tag, boot, u, t[f"first_wrong{sfx}"] == d, experiment="exp2",
                     estimand="share first wrong digit" if d < n else "share top-1 exact",
                     slot=1 if not sfx else 2, d=d)
    if experiment in ("exp3", "all"):
        for k in KS:
            _add(out, tag, boot, u, t[f"sid_hit@{k}"], experiment="exp3", estimand=f"SID HR@{k}")
            for w in ("lower", "uniform", "upper"):
                _add(out, tag, boot, u, t[f"item_{w}@{k}"], experiment="exp3", estimand=f"item {w} HR@{k}")
            _add(out, tag, boot, u, t[f"sid_hit@{k}"] - t[f"item_uniform@{k}"], experiment="exp3",
                 estimand=f"SID - item uniform HR@{k}")
        col = t.bucket > 1
        if col.any() and (~col).any():
            out.append({**tag, "experiment": "exp3", "estimand": "SID HR@10 collided - singleton",
                        **boot.paired_difference((u[col], t["sid_hit@10"][col]), (u[~col], t["sid_hit@10"][~col]))})
            _add(out, tag, boot, u, col, experiment="exp3", estimand="share collided targets")
    if experiment in ("exp4", "all") and "rank2" in t:
        n = int(t.n_digit.iloc[0])
        p1, p2 = t.first_wrong == n, t.first_wrong2 == n
        if p1.any() and (~p1).any():
            out.append({**tag, "experiment": "exp4", "estimand": "P(p2 right | p1 right) - P(p2 right | p1 wrong)",
                        **boot.paired_difference((u[p1], p2[p1].astype(float)), (u[~p1], p2[~p1].astype(float)))})
        for name, m in (("P(p2 right | p1 right)", p1), ("P(p2 right | p1 wrong)", ~p1)):
            if m.any():
                _add(out, tag, boot, u[m], p2[m], experiment="exp4", estimand=name)
        _add(out, tag, boot, u, t["sid_hit@102"] - t["sid_hit@10"], experiment="exp4",
             estimand="slot-2 minus slot-1 SID HR@10")
        _add(out, tag, boot, u, t["pair_hit@10"], experiment="exp4", estimand="pair HR@10")
    if experiment in ("exp6", "all"):
        ok = ~t.time_ambiguous
        for b in [name for _, name in T.GAP_BINS] + ["all"]:
            m = ok & ((t.gap1_bin == b) if b != "all" else True)
            if not m.any():
                continue
            for name, col in (("SID HR@10", "sid_hit@10"), ("item uniform HR@10", "item_uniform@10"),
                              ("top-1 copies r1 first code", "top1_d0_is_r1"),
                              ("data: target first code = r1's", "data_d0_is_r1")):
                _add(out, tag, boot, u[m], t[col][m], experiment="exp6", estimand=name, gap_bin=b)
            _add(out, tag, boot, u[m], t["top1_d0_is_r1"][m].astype(float) - t["data_d0_is_r1"][m],
                 experiment="exp6", estimand="over-copy (model - data)", gap_bin=b)
            if b != "all":
                _add(out, tag, boot, u[ok], (t.gap1_bin == b)[ok], experiment="exp6", estimand="share of rows",
                     gap_bin=b)
        hits = t["sid_hit@10"] > 0
        if hits.any():
            _add(out, tag, boot, u[hits], t.dup_target[hits], experiment="exp6",
                 estimand="share of SID HR@10 hits from duplicate records")
        _add(out, tag, boot, u, t.dup_target, experiment="exp6", estimand="share duplicate-record targets")
    if experiment in ("exp7", "all"):
        for c in events.CLASSES:
            m = t.cls == c
            if not m.any():
                continue
            for name, col in (("SID HR@10", "sid_hit@10"), ("item uniform HR@10", "item_uniform@10"),
                              ("item lower HR@10", "item_lower@10")):
                _add(out, tag, boot, u[m], t[col][m], experiment="classes", estimand=name, target_class=c)
            _add(out, tag, boot, u, m, experiment="classes", estimand="share of rows", target_class=c)
    return out


def exp7_on_exp6(cell, entry: dict) -> list[dict]:
    """Retrospective exp7 on a grid exp6 cell: each condition's SID and item HR@10
    by target class, and its change from B, paired by user."""
    import json
    from pathlib import Path

    import pandas as pd
    from sidlens.analysis.bootstrap import UserBootstrap

    cell = Path(cell)
    rows = pd.read_parquet(cell / "rows.parquet").set_index("example_id")
    table, items = GI.sid_table(entry), GI.item2id(entry)
    n = table.variant.n_codebook
    buckets = {sid: tuple(sorted(items[a] for a in asins)) for sid, asins in table.cb2items.items()}
    scored = {}
    for f in sorted(cell.glob("predictions_*.parquet")):
        d = pd.read_parquet(f).set_index("example_id")
        cond = f.stem.split("predictions_", 1)[1]
        recs = []
        for eid, r in rows.iterrows():
            pred = [P.parse_sid(s, n) for s in d.loc[eid, "predict"]]
            tgt = tuple(r.target_codes)
            b = item_rank_bounds(list(dict.fromkeys(pred)), int(r.target_item), buckets, target_sid=tgt).hit_bounds(10)
            recs.append({"example_id": eid, "sid": float(tgt in pred[:10]), "item_u": b["uniform"],
                         "item_lo": b["lower"]})
        scored[cond] = pd.DataFrame(recs).set_index("example_id")
    boot = UserBootstrap(rows.user_id, draws=DRAWS, seed=SEED)
    tag = {"cell": str(cell), **{k: json.loads((cell / "inputs.json").read_text()).get(k)
                                 for k in ("grid_key", "grid_state", "variant", "dataset")}}
    out = []
    for cond, s in scored.items():
        for c in list(events.CLASSES) + ["all"]:
            m = (rows.cls == c) if c != "all" else pd.Series(True, index=rows.index)
            for col in ("sid", "item_u", "item_lo"):
                _add(out, tag, boot, rows.user_id[m], s[col][m], estimand=f"{col} HR@10", condition=cond,
                     target_class=c)
                if cond != "B" and "B" in scored:
                    _add(out, tag, boot, rows.user_id[m], (s[col] - scored["B"][col])[m],
                         estimand=f"{col} HR@10 change from B", condition=cond, target_class=c)
    return out


def main(argv=None) -> int:
    """python -m sidlens.grid.saved_lists --ckpt 'grid:<key>@best' --experiment exp3 --out <new dir>
    python -m sidlens.grid.saved_lists --ckpt 'grid:<key>@best' --experiment exp7 --exp6-cell <exp6 grid cell> --out <dir>
    """
    import argparse
    import json
    from pathlib import Path

    import pandas as pd
    from sidlens import grid
    from sidlens.grid import registry
    from sidlens.provenance.hashing import sha256_file

    ap = argparse.ArgumentParser(description="Retrospective analyses of a grid run's saved test lists")
    ap.add_argument("--ckpt", required=True, help="grid:<key>@best (only the best state has saved lists)")
    ap.add_argument("--experiment", required=True, choices=("exp2", "exp3", "exp4", "exp6", "exp7", "all"))
    ap.add_argument("--exp6-cell", type=Path, help="exp7: the run's controlled exp6 grid cell")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if grid.split_name(args.ckpt)[1] != grid.BEST:
        ap.error("only the best state was evaluated; intermediate states have no saved lists")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "estimates.csv").exists():
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    e = registry.load().entry(args.ckpt)
    tag = {"checkpoint": args.ckpt, **grid.cell_info(args.ckpt)}
    if args.experiment == "exp7":
        if args.exp6_cell is None:
            ap.error("exp7 re-scores controlled exp6: pass --exp6-cell")
        inp = json.loads((args.exp6_cell / "inputs.json").read_text())
        if inp.get("grid_key") != e["key"]:
            raise ValueError(f"{args.exp6_cell} is not this run's exp6 cell")
        est = pd.DataFrame([{**tag, **r} for r in exp7_on_exp6(args.exp6_cell, e)])
        sources = {str(p): sha256_file(p) for p in sorted(args.exp6_cell.glob("*.parquet"))}
    else:
        if args.experiment == "exp4" and e["task"] != "two-item":
            ap.error("exp4 is defined for next-two runs")
        t = row_table(e)
        t.to_parquet(args.out / "rows.parquet", index=False)
        est = pd.DataFrame(estimands(t, args.experiment, tag))
        p = Path(e["run_dir"]) / "evaluation" / "predictions.jsonl"
        sources = {str(p): sha256_file(p)}
    est.to_csv(args.out / "estimates.csv", index=False)
    (args.out / "inputs.json").write_text(json.dumps({"checkpoint": args.ckpt, "experiment": args.experiment,
                                                      "grid_key": e["key"], "sources_sha256": sources,
                                                      "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "user"}},
                                                     indent=1))
    (args.out / "validation.json").write_text(json.dumps({"rows_joined_one_to_one": True,
                                                          "n_estimates": len(est)}, indent=1))
    print(f"wrote {args.out} ({len(est)} estimates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
