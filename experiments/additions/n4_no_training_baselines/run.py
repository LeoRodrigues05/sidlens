#!/usr/bin/env python
"""N4: no-training baselines (copy rule, popularity) against one grid run's saved test lists.

Protocol: `protocol.md` beside this file, declared before any baseline was
computed. CPU only; one invocation = one grid run at its best state.

Traps, each with its guard
--------------------------
1. **A baseline built from test data.** Popularity counts only the run's
   training cohort targets; the copy rule reads only each row's own history.
2. **Item credit for SID hits.** Item metrics come from
   `collisions.item_rank_bounds` over the run's own SID buckets.
3. **Rows joined by position.** Saved lists, classes and cohort rows join by
   `example_id`, one to one, or raise.
4. **Ties.** Every ordering is total (popularity, then code order), so a
   baseline list is a function of its inputs alone.

    python run.py --ckpt 'grid:<key>@best' --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import grid                                                  # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                      # noqa: E402
from sidlens.analysis.collisions import item_rank_bounds                  # noqa: E402
from sidlens.data import ar_prompts as P                                  # noqa: E402
from sidlens.grid import events, inputs as GI, predictions as GP, registry  # noqa: E402
from sidlens.provenance.hashing import sha256_file                        # noqa: E402

SEED, DRAWS, K = 20261021, 2000, 10
BASELINES = ("copy", "pop", "pop_r1")


class Ranker:
    """Catalog SIDs indexed by prefix, each list in popularity order (trap 4)."""

    def __init__(self, sids, pop: Counter, n: int):
        self.n = n
        self.order = sorted(sids, key=lambda s: (-pop.get(s, 0), s))
        self.by_prefix: dict[tuple, list] = {}
        for s in self.order:
            for l in range(n + 1):
                self.by_prefix.setdefault(s[:l], []).append(s)

    def copy(self, hist: list[tuple]) -> list[tuple]:
        """History LCP first, then the recency of the item giving it, then popularity."""
        out, seen = [], set()
        for l in range(self.n, -1, -1):
            for h in (reversed(hist) if l else [()]):
                for s in self.by_prefix.get(h[:l], ()):
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
                        if len(out) == K:
                            return out
        return out

    def pop(self) -> list[tuple]:
        return self.order[:K]

    def pop_r1(self, r1: tuple) -> list[tuple]:
        first = self.by_prefix.get(r1[:1], [])[:K]
        rest = [s for s in self.order if s[:1] != r1[:1]][:K - len(first)]
        return first + rest


def metrics(lst: list[tuple], target: tuple, target_item: int, buckets) -> dict:
    rank = next((k + 1 for k, s in enumerate(lst[:K]) if s == target), 0)
    b = item_rank_bounds(list(dict.fromkeys(lst[:K])), target_item, buckets, target_sid=target).hit_bounds(K)
    return {"sid_hit": float(rank > 0), "ndcg": 1 / math.log2(rank + 1) if rank else 0.0,
            "item_uniform": b["uniform"], "item_lower": b["lower"], "item_upper": b["upper"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="grid:<key>@best")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if grid.split_name(args.ckpt)[1] != grid.BEST:
        ap.error("N4 compares with saved test lists, which exist only for the best state")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t0 = time.time()
    e = registry.load().entry(args.ckpt)
    table, items = GI.sid_table(e), GI.item2id(e)
    n = table.variant.n_codebook
    code_of = {items[a]: c for a, c in table.asin2codes.items()}
    buckets = {sid: tuple(sorted(items[a] for a in asins)) for sid, asins in table.cb2items.items()}
    pop = Counter(code_of[items[a]] for r in events.cohort_rows(e, "train") for a in r["target_item_ids"])
    R = Ranker(list(buckets), pop, n)
    saved = GP.load(e)
    rows = {r["example_id"]: r for r in events.cohort_rows(e, "test")}
    if set(saved) - set(rows):
        raise ValueError("saved rows missing from the prepared cohort")
    ids = list(saved)
    cls = events.target_classes(e, "test", ids).set_index("example_id")

    recs = []
    for eid in ids:
        r = rows[eid]
        hist = [code_of[items[a]] for a in r["history_item_ids"]]
        tgt, ti = code_of[items[r["target_item_ids"][0]]], items[r["target_item_ids"][0]]
        lists = {"copy": R.copy(hist), "pop": R.pop(), "pop_r1": R.pop_r1(hist[-1])}
        model = [P.parse_sid(c["sids"][0], n) for c in saved[eid]["predictions"]]
        lists["model"] = list(dict.fromkeys(model))
        rec = {"example_id": eid, "user_id": r["user_id"], "cls": cls.at[eid, "cls"],
               "model_top1_d0_eq_copy": bool(model and model[0][0] == lists["copy"][0][0]),
               "model_top1_eq_copy": bool(model and model[0] == lists["copy"][0]),
               "copy_top1_lcp": max((next((d for d in range(n) if h[d] != lists["copy"][0][d]), n)
                                     for h in hist), default=0)}
        for name, lst in lists.items():
            rec.update({f"{name}_{k}": v for k, v in metrics(lst, tgt, ti, buckets).items()})
        recs.append(rec)
    t = pd.DataFrame(recs)
    t.to_parquet(args.out / "rows.parquet", index=False)

    boot = UserBootstrap(t.user_id, draws=DRAWS, seed=SEED)
    tag = {"checkpoint": args.ckpt, **grid.cell_info(args.ckpt)}
    est = []

    def add(m, x, **kw):
        if m.any():
            est.append({**tag, **kw, **boot.mean(t.user_id[m], np.asarray(x)[m.to_numpy()].astype(float))})

    groups = {"all": pd.Series(True, index=t.index), **{c: t.cls == c for c in events.CLASSES}}
    for g, m in groups.items():
        for name in BASELINES + ("model",):
            for metric in ("sid_hit", "ndcg", "item_uniform", "item_lower", "item_upper"):
                add(m, t[f"{name}_{metric}"], estimand=f"{metric}@{K}", list=name, target_class=g)
        for b in BASELINES:
            for metric in ("sid_hit", "item_uniform"):
                add(m, t[f"model_{metric}"] - t[f"{b}_{metric}"], estimand=f"model - {b}, {metric}@{K}",
                    list="model", target_class=g)
        add(m, t.model_top1_d0_eq_copy, estimand="model top-1 first code = copy rule's", list="model",
            target_class=g)
        add(m, t.model_top1_eq_copy, estimand="model top-1 SID = copy rule's", list="model", target_class=g)
    pd.DataFrame(est).to_csv(args.out / "estimates.csv", index=False)
    pred = Path(e["run_dir"]) / "evaluation" / "predictions.jsonl"
    (args.out / "inputs.json").write_text(json.dumps({
        "checkpoint": args.ckpt, "grid_key": e["key"], "n_rows": len(t), "n_users": int(t.user_id.nunique()),
        "n_catalog_sids": len(buckets), "n_train_targets": int(sum(pop.values())), "k": K,
        "predictions_sha256": sha256_file(pred), "sid_index_sha256": e["sid_index"]["sha256"],
        "bootstrap": {"unit": "user", "draws": DRAWS, "seed": SEED}, **grid.cell_info(args.ckpt)}, indent=1))
    (args.out / "validation.json").write_text(json.dumps({
        "rows_joined_one_to_one": len(t) == len(saved), "copy_lists_full": bool((t.copy_sid_hit >= 0).all()),
        "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"wrote {args.out}: {len(t)} rows, {len(est)} estimates, {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
