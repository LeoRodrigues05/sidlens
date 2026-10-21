"""Strict single-target CCE for the current AR/Diff prediction-record schema.

The complete catalog determines collision groups. Saved candidate lists are
checked against it, never used as a substitute for that catalog. Short/empty
completed rankings are scored as returned and all examples stay in the mean.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence

from metrics.collision import CollisionEvaluator, collision_stats, parse_sid


def current_cce_metrics(
    records: Iterable[Mapping], item_to_sid: Mapping, ks: Sequence[int], *,
    retained_top_k: int, include_records: bool = False,
) -> dict:
    """Score one-target ordered records with an explicitly retained rank budget.

    ``retained_top_k`` is the decoder's configured export limit. It may exceed
    a particular row's returned count (e.g. catalog exhaustion or a failed
    decode), but must cover every requested cutoff. Pair CCE is not defined.
    """
    if isinstance(retained_top_k, bool) or not isinstance(retained_top_k, int) or retained_top_k < 1:
        raise ValueError("retained_top_k must be a positive integer")
    if not isinstance(item_to_sid, Mapping) or not item_to_sid:
        raise ValueError("CCE requires a nonempty complete item-to-SID catalog")
    catalog = {str(item): parse_sid(sid) for item, sid in item_to_sid.items()}
    if len(catalog) != len(item_to_sid):
        raise ValueError("Catalog item IDs are not unique after string conversion")
    groups = {}
    for item, sid in catalog.items():
        groups.setdefault(sid, set()).add(item)
    if len({len(sid) for sid in groups}) != 1:
        raise ValueError("Catalog SID depths must be consistent")
    group_sizes = {sid: len(items) for sid, items in groups.items()}
    scorer = CollisionEvaluator(group_sizes, ks, beam_policy="strict")
    if max(scorer.ks) > retained_top_k:
        raise ValueError("Requested K exceeds the retained decoder top_k; rerun inference with a larger export limit")
    rows = list(records)
    if not rows:
        raise ValueError("Cannot score an empty evaluation cohort")
    sums, beam_totals = Counter(), Counter()
    details, seen_ids = [], set()
    short_rows = empty_rows = 0

    def validate_group(raw, sid, context):
        if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list):
            raise ValueError(f"{context} collision groups must contain exactly one item list")
        members = [str(item) for item in raw[0]]
        if len(members) != len(set(members)) or set(members) != groups[sid]:
            raise ValueError(f"{context} collision group differs from the complete catalog")

    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Record {position} must be an object")
        target = row.get("target_sids")
        if not isinstance(target, list) or len(target) != 1:
            raise ValueError("CCE supports exactly one target item; pair CCE is not defined")
        if not isinstance(target[0], str) or not target[0]:
            raise ValueError("Current-schema target SIDs must be nonempty strings")
        predictions = row.get("predictions")
        if not isinstance(predictions, list):
            raise ValueError("Current-schema predictions must be a list")
        if len(predictions) > retained_top_k:
            raise ValueError("Prediction count exceeds the recorded retained decoder top_k")
        beam = []
        for prediction in predictions:
            if not isinstance(prediction, Mapping):
                raise ValueError("Every prediction must be an object containing sids")
            sids = prediction.get("sids")
            if not isinstance(sids, list) or len(sids) != 1 or not isinstance(sids[0], str) or not sids[0]:
                raise ValueError("Each prediction must contain exactly one nonempty SID string")
            beam.append(sids[0])
        result = scorer.score(beam, target[0])  # Reject invalid, unknown, or duplicate SIDs.
        target_sid = parse_sid(target[0])
        if "example_id" in row:
            example_id = row["example_id"]
            if not isinstance(example_id, str) or not example_id or example_id in seen_ids:
                raise ValueError("Example IDs must be nonempty unique strings")
            seen_ids.add(example_id)
        if "target_item_ids" in row:
            ids = row["target_item_ids"]
            if not isinstance(ids, list) or len(ids) != 1 or catalog.get(str(ids[0])) != target_sid:
                raise ValueError("Target item identity does not match the supplied SID catalog")
        if "target_candidate_item_ids" in row:
            validate_group(row["target_candidate_item_ids"], target_sid, "Target")
        for prediction, sid in zip(predictions, result.ranked_sids):
            if "candidate_item_ids" in prediction:
                validate_group(prediction["candidate_item_ids"], sid, "Prediction")
        sums.update(result.metrics)
        beam_totals.update(result.beam.to_dict())
        short_rows += int(len(beam) < max(scorer.ks))
        empty_rows += int(not beam)
        if include_records:
            detail = {"row": position, **result.to_dict()}
            if "example_id" in row:
                detail["example_id"] = row["example_id"]
            details.append(detail)
    report = {
        "metric_protocol": "paper-cce-v1",
        "collision_protocol": "paper-collision-item-rate-v1",
        "score_scale": "fraction", "target_cardinality": 1,
        "input_schema": "ordered_targets_v1", "ks": list(scorer.ks),
        "samples": len(rows), "metric_denominator": len(rows),
        "retained_top_k": retained_top_k,
        "metrics": {name: value / len(rows) for name, value in sorted(sums.items())},
        "catalog": collision_stats(group_sizes).to_dict(),
        "semantics": "Expand complete collision groups in SID rank order; uniformly average target position within its group. K counts expanded items for ItemHit/ItemNDCG and SID ranks for Hit/NDCG.",
        "diagnostics": {
            "target_status": {"ok": len(rows), "invalid": 0, "unknown": 0},
            "beam_entries": dict(beam_totals),
            "samples_with_beam_issue": {"invalid": 0, "unknown": 0, "duplicate": 0},
            "beam_policy": {"mode": "strict", "invalid": "error", "unknown": "error",
                            "duplicate": "error", "bad_target": "error", "padding": "ordinary_entry"},
            "short_rankings": short_rows, "empty_rankings": empty_rows,
            "short_ranking_policy": "score returned ranking; retain example in denominator; never infer unreturned candidates",
        },
    }
    if include_records:
        report["records"] = details
    return report
