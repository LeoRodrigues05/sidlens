"""Exact ordered-target ranking metrics, shared by corrected AR/Diff evaluation.

An ordered SID tuple is one relevant result: partial tuples earn zero exact
credit. Per-slot diagnostics use positions in the original tuple ranking.
SID equivalence is not item identity when quantization has collisions.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def _scores(rank, ks):
    return {
        f"{metric}@{k}": (
            (1.0 if metric == "hit" else 1.0 / rank if metric == "mrr"
             else 1.0 / math.log2(rank + 1))
            if rank is not None and rank <= k else 0.0
        )
        for k in ks for metric in ("hit", "mrr", "ndcg")
    }


def ordered_target_metrics(records: Iterable[Mapping], ks=(1, 5, 10)) -> dict:
    """Score records with ``target_sids`` and ranked ``predictions[*].sids``.

    Optional ``target_candidate_item_ids`` and prediction
    ``candidate_item_ids`` contain one collision group per target slot.
    All examples, including empty prediction lists, stay in the denominator.
    Repeated tuple predictions are rejected; decoders must deduplicate them.
    """
    ks = tuple(sorted(set(ks)))
    if not ks or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    rows = list(records)
    if not rows:
        raise ValueError("Cannot score an empty evaluation cohort")
    horizon = len(rows[0]["target_sids"])
    if horizon < 1:
        raise ValueError("Targets must contain at least one SID")
    sums = {"exact_sid_tuple": _scores(None, ks)}
    sums.update({f"slot_{slot + 1}": _scores(None, ks) for slot in range(horizon)})
    singleton_sums = _scores(None, ks)
    singleton_count = collision_known_count = collision_target_count = 0
    returned = 0
    colliding_predictions = 0
    known_predictions = 0
    for row in rows:
        target = tuple(row["target_sids"])
        if len(target) != horizon or any(not isinstance(sid, str) or not sid for sid in target):
            raise ValueError("Every example must have the same positive target horizon and string SIDs")
        predictions = row["predictions"]
        ranking = [tuple(pred["sids"]) for pred in predictions]
        if any(len(pred) != horizon or any(not isinstance(sid, str) or not sid for sid in pred)
               for pred in ranking):
            raise ValueError("Prediction horizon or SID format differs from target")
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate ordered tuple predictions")
        returned += len(ranking)
        exact_rank = next((i for i, pred in enumerate(ranking, 1) if pred == target), None)
        scores = _scores(exact_rank, ks)
        for key, value in scores.items():
            sums["exact_sid_tuple"][key] += value
        for slot in range(horizon):
            rank = next((i for i, pred in enumerate(ranking, 1) if pred[slot] == target[slot]), None)
            for key, value in _scores(rank, ks).items():
                sums[f"slot_{slot + 1}"][key] += value
        groups = row.get("target_candidate_item_ids")
        if groups is not None:
            if len(groups) != horizon or any(not group for group in groups):
                raise ValueError("Target collision groups must be nonempty and match the horizon")
            target_ids = row.get("target_item_ids")
            if target_ids is not None and (len(target_ids) != horizon or any(
                    str(item) not in {str(candidate) for candidate in group}
                    for item, group in zip(target_ids, groups))):
                raise ValueError("Target item identity is not present in its SID collision group")
            collision_known_count += 1
            colliding = any(len(set(map(str, group))) > 1 for group in groups)
            collision_target_count += int(colliding)
            if not colliding:
                singleton_count += 1
                for key, value in scores.items():
                    singleton_sums[key] += value
        for prediction in predictions:
            groups = prediction.get("candidate_item_ids")
            if groups is not None:
                if len(groups) != horizon or any(not group for group in groups):
                    raise ValueError("Prediction collision groups must be nonempty and match the horizon")
                known_predictions += 1
                colliding_predictions += int(any(len(set(map(str, group))) > 1 for group in groups))
    return {
        "schema_version": "ordered_targets_v1",
        "examples": len(rows),
        "horizon": horizon,
        "ks": list(ks),
        **{name: {key: value / len(rows) for key, value in values.items()}
           for name, values in sums.items()},
        "collision_diagnostics": {
            "semantics": "SID tuple equivalence; no item identity or within-collision ranking assumed",
            "targets_with_known_groups": collision_known_count,
            "targets_with_any_collision": collision_target_count,
            "predictions_with_known_groups": known_predictions,
            "predictions_with_any_collision": colliding_predictions,
            "singleton_target_examples": singleton_count,
            "singleton_target_exact_tuple": (
                {key: value / singleton_count for key, value in singleton_sums.items()}
                if singleton_count else None
            ),
        },
        "mean_returned_tuples": returned / len(rows),
        "slot_rank_semantics": "first matching slot within the original ranked tuples, without projection/deduplication",
    }
