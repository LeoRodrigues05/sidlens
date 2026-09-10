#!/usr/bin/env python3
"""Retrospective Experiment 4 -- next-two conditional-association audit.

This is deliberately an *artifact audit*, not a causal intervention.  The
retained AR JSON files contain an ordered list of first-item beams and one
greedy second item generated after each first item.  They contain neither the
first-pass sequence scores nor a distribution/score for the second pass.  The
analysis can therefore measure order-specific retrieval asymmetry and beam-pair
association, but it cannot identify how much changing item 1 causes item 2 to
change.

Every prediction row is joined by position to its exact frozen two-item split
and checked against the one-to-many SID catalogue in the matching info file.
The script fails on target/alignment drift, while illegal generated strings are
kept as observed model failures and counted.  Its primary uncertainty unit is
the user, because the chronological global split contains repeated users.

Outputs (under $SIDLENS_WORK/derived/retrospective/exp4_next_two):

``artifact_validation.csv``
    Per-cell schemas, hashes, row alignment, legality, collisions, and split
    overlap.  All ten retained AR prediction cells must validate.
``configuration.csv``
    Slot-1, slot-2, ordered-pair, reverse-pair, unordered-pair, and one-item-
    representative collision sensitivities at k in {1,3,5,10,20}, for all
    rows and one final test row per user, before/after singleton filtering.
``inference.csv``
    User-cluster percentile-bootstrap intervals for the analysis-designated contrasts:
    the top-pair and all-pair conditional p2 risk differences, slot2-slot1
    hit@10, ordered-reversed pair hit@20, and beam association.
``beam_association.csv``
    Pairing-specific compatibility with training transitions.  For each row,
    the observed p1->p2 log-count is compared with every off-diagonal within-
    row re-pairing.  Training counts exclude that test user's training rows.
``row_metrics.csv``
    Auditable per-row ranks, diversity, collision flags, and primary outcomes.
``result.json`` and ``report.md``
    Availability/source-semantics audit, cohort alignment, provenance, and the
    claim boundary separating association from causal dependence.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from sidlens import paths
from sidlens.provenance import hashing


CATEGORY = "Industrial_and_Scientific"
CUTOFFS = (1, 3, 5, 10, 20)
FILE_RE = re.compile(
    r"^twoitem__(?P<category>.+)__(?P<quantizer>MQ|rqvae|rqkmeans)__"
    r"(?P<depth>\d+)cb__(?P<width>\d+)\.predictions\.json$"
)
TOKEN_RE = re.compile(r"<([a-z])_(\d+)>")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_sid(raw: object, depth: int) -> tuple[int, ...]:
    """Parse exactly one full SID; reject prose, separators, and wrong depth."""

    text = str(raw).strip().strip('"').replace(" ", "")
    matches = TOKEN_RE.findall(text)
    rebuilt = "".join(f"<{letter}_{number}>" for letter, number in matches)
    if rebuilt != text:
        raise ValueError(f"malformed SID: {raw!r}")
    expected = [chr(ord("a") + i) for i in range(depth)]
    letters = [letter for letter, _ in matches]
    if letters != expected:
        raise ValueError(
            f"expected consecutive {expected}, got {letters} in {raw!r}")
    return tuple(int(number) for _, number in matches)


def parse_pair(raw: object, depth: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    parts = str(raw).split("|||")
    if len(parts) != 2:
        raise ValueError(f"expected exactly two SIDs separated by |||: {raw!r}")
    return parse_sid(parts[0], depth), parse_sid(parts[1], depth)


def parse_item_pair(raw: object) -> tuple[int, int]:
    parts = str(raw).split("|||")
    if len(parts) != 2:
        raise ValueError(f"expected exactly two item IDs: {raw!r}")
    return int(parts[0].strip()), int(parts[1].strip())


def safe_sid(raw: object, depth: int) -> tuple[int, ...] | None:
    try:
        return parse_sid(raw, depth)
    except (TypeError, ValueError):
        return None


def split_path(category: str, split: str, quantizer: str,
               depth: int, width: int) -> Path:
    root = paths.FROZEN_DATA / "splits" / "two-item" / split
    candidates = sorted(root.glob(
        f"{category}_{quantizer}_{depth}codebook_{width}*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {split} split for {quantizer} {depth}x{width}; "
            f"found {candidates}")
    return candidates[0]


def info_path(category: str, quantizer: str, depth: int, width: int) -> Path:
    root = paths.FROZEN_SIDS / "info" / "two-item"
    candidates = sorted(root.glob(
        f"{category}_{quantizer}_{depth}codebook_{width}*.txt"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one info file for {quantizer} {depth}x{width}; "
            f"found {candidates}")
    return candidates[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_info(path: Path, depth: int) -> tuple[
        dict[int, tuple[int, ...]], dict[tuple[int, ...], list[int]]]:
    """Load ``SID<TAB>title<TAB>zero-based item_id`` losslessly."""

    item_to_sid: dict[int, tuple[int, ...]] = {}
    by_sid: dict[tuple[int, ...], list[int]] = defaultdict(list)
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no}: expected >=3 tab fields")
            sid = parse_sid(parts[0], depth)
            item = int(parts[-1])
            if item in item_to_sid:
                raise ValueError(f"{path}:{line_no}: duplicate item {item}")
            item_to_sid[item] = sid
            by_sid[sid].append(item)
    for items in by_sid.values():
        items.sort()
    return item_to_sid, dict(by_sid)


def identity(row: dict[str, str]) -> tuple[str, str, str]:
    """Representation-independent row identity shared by all 27 split cells."""

    return row["user_id"], row["history_item_id"], row["item_id"]


def final_row_indices(rows: Sequence[dict]) -> set[int]:
    """Last row position per user; CSV order is chronological in this substrate."""

    last: dict[str, int] = {}
    for index, row in enumerate(rows):
        last[str(row["user_id"])] = index
    return set(last.values())


def first_rank(pairs: Sequence[tuple[tuple[int, ...] | None,
                                    tuple[int, ...] | None]],
               target: tuple[tuple[int, ...], tuple[int, ...]]) -> int | None:
    for rank, pair in enumerate(pairs, start=1):
        if pair == target:
            return rank
    return None


def slot_rank(pairs: Sequence[tuple[tuple[int, ...] | None,
                                   tuple[int, ...] | None]],
              slot: int, target: tuple[int, ...]) -> int | None:
    for rank, pair in enumerate(pairs, start=1):
        if pair[slot] == target:
            return rank
    return None


def hit(rank: int | None, cutoff: int) -> float:
    return float(rank is not None and rank <= cutoff)


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def normalized_entropy(values: Sequence[object]) -> float:
    """Empirical entropy / log(n); zero for fewer than two observations."""

    n = len(values)
    if n < 2:
        return 0.0
    counts = Counter(values)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    return float(entropy / math.log(n))


def load_training_transitions(path: Path, depth: int) -> tuple[
        Counter, dict[str, Counter], set[str]]:
    global_counts: Counter = Counter()
    by_user: dict[str, Counter] = defaultdict(Counter)
    users: set[str] = set()
    with path.open(newline="") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            try:
                pair = parse_pair(row["item_sid"], depth)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            user = row["user_id"]
            global_counts[pair] += 1
            by_user[user][pair] += 1
            users.add(user)
    return global_counts, dict(by_user), users


def coupling_lift(
    pairs: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
    global_counts: Counter,
    heldout_counts: Counter | None = None,
) -> float | None:
    """Observed vs all off-diagonal p1/p2 re-pairings within one row.

    Compatibility is ``log(1 + training_count(p1,p2))``.  Counts contributed by
    the test user's training rows are removed.  The off-diagonal reference
    preserves the row's p1 and p2 marginals but breaks their stored pairing.
    This is a behavioral association statistic, not a causal estimand.
    """

    if len(pairs) < 2:
        return None
    heldout_counts = heldout_counts or Counter()

    def score(pair: tuple[tuple[int, ...], tuple[int, ...]]) -> float:
        count = global_counts[pair] - heldout_counts[pair]
        if count < 0:
            raise ValueError("held-out transition count exceeds global count")
        return math.log1p(count)

    observed = mean(score(pair) for pair in pairs)
    shuffled = mean(
        score((p1, p2_other))
        for i, (p1, _) in enumerate(pairs)
        for j, (_, p2_other) in enumerate(pairs)
        if i != j
    )
    return observed - shuffled


def summarize_rows(records: list[dict], cutoffs: Sequence[int]) -> dict:
    out: dict[str, object] = {
        "n_rows": len(records),
        "n_users": len({r["user"] for r in records}),
        "n_swap_eligible": sum(r["swap_eligible"] for r in records),
        "mean_beams": mean(r["n_pairs"] for r in records),
        "mean_valid_pairs": mean(r["n_valid_pairs"] for r in records),
        "mean_p2_unique_fraction": mean(r["p2_unique_fraction"] for r in records),
        "mean_p2_entropy_normalized": mean(r["p2_entropy_normalized"] for r in records),
        "mean_self_pair_fraction": mean(r["self_pair_fraction"] for r in records),
        "ordered_mrr": mean(r["ordered_rr"] for r in records),
        "reversed_mrr_swap_eligible": mean(
            r["reversed_rr"] for r in records if r["swap_eligible"]),
        "coupling_lift": mean(
            r["coupling_lift"] for r in records
            if r["coupling_lift"] is not None),
    }
    for cutoff in cutoffs:
        for key in (
            "slot1_hit", "slot2_hit", "ordered_pair_hit", "reversed_pair_hit",
            "unordered_pair_hit", "representative_item_pair_lower",
            "representative_item_pair_uniform", "representative_item_pair_upper",
        ):
            eligible = (r for r in records
                        if key != "reversed_pair_hit" or r["swap_eligible"])
            out[f"{key}@{cutoff}"] = mean(r[f"{key}@{cutoff}"] for r in eligible)
        out[f"slot2_minus_slot1@{cutoff}"] = (
            float(out[f"slot2_hit@{cutoff}"])
            - float(out[f"slot1_hit@{cutoff}"]))
        out[f"ordered_minus_reversed@{cutoff}"] = mean(
            r[f"ordered_pair_hit@{cutoff}"] - r[f"reversed_pair_hit@{cutoff}"]
            for r in records if r["swap_eligible"])
    for prefix in ("top1", "all_pairs"):
        for key, value in conditional_counts(records, prefix).items():
            out[f"{prefix}_{key}"] = value
    return out


def cluster_bootstrap(values: Sequence[tuple[str, float]], n_boot: int,
                      seed: int) -> dict:
    """Percentile bootstrap over users, retaining every row within a user."""

    usable = [(str(user), float(value)) for user, value in values
              if math.isfinite(float(value))]
    if not usable:
        return {"estimate": None, "ci_low": None, "ci_high": None,
                "n_rows": 0, "n_users": 0, "n_boot": n_boot, "seed": seed}
    grouped: dict[str, list[float]] = defaultdict(list)
    for user, value in usable:
        grouped[user].append(value)
    users = sorted(grouped)
    sums = np.asarray([sum(grouped[user]) for user in users], dtype=float)
    counts = np.asarray([len(grouped[user]) for user in users], dtype=float)
    estimate = float(sums.sum() / counts.sum())
    if n_boot <= 0:
        return {"estimate": estimate, "ci_low": None, "ci_high": None,
                "n_rows": len(usable), "n_users": len(users),
                "n_boot": n_boot, "seed": seed}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(n_boot, len(users)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "n_rows": len(usable),
        "n_users": len(users),
        "n_boot": n_boot,
        "seed": seed,
    }


def conditional_counts(records: Sequence[dict], prefix: str) -> dict:
    """P(slot2 correct | slot1 state), preserving explicit denominators."""

    usable = [row for row in records if row["conditional_pair_eligible"]]
    n_correct = sum(int(row[f"{prefix}_n_p1_correct"]) for row in usable)
    y_correct = sum(int(row[f"{prefix}_p2_correct_p1_correct"]) for row in usable)
    n_wrong = sum(int(row[f"{prefix}_n_p1_wrong"]) for row in usable)
    y_wrong = sum(int(row[f"{prefix}_p2_correct_p1_wrong"]) for row in usable)
    rate_correct = y_correct / n_correct if n_correct else float("nan")
    rate_wrong = y_wrong / n_wrong if n_wrong else float("nan")
    return {
        "n_p1_correct": n_correct,
        "p2_correct_p1_correct": y_correct,
        "p2_rate_given_p1_correct": rate_correct,
        "n_p1_wrong": n_wrong,
        "p2_correct_p1_wrong": y_wrong,
        "p2_rate_given_p1_wrong": rate_wrong,
        "conditional_risk_difference": rate_correct - rate_wrong,
        "conditional_risk_ratio": (
            rate_correct / rate_wrong if rate_wrong > 0 else float("nan")),
    }


def cluster_bootstrap_conditional(
    records: Sequence[dict], prefix: str, n_boot: int, seed: int,
    *, standardize_by: str | None = None,
) -> dict:
    """User-cluster bootstrap for a conditional risk difference.

    When ``standardize_by`` is supplied, a risk difference is computed inside
    each stratum and those differences receive equal weight.  This avoids a
    pooled configuration mixture turning general cell difficulty into an
    apparent p1/p2 association.
    """

    if n_boot < 0:
        raise ValueError("bootstrap replicate count must be non-negative")
    usable = [row for row in records if row["conditional_pair_eligible"]]
    strata = (["pooled"] if standardize_by is None else
              sorted({str(row[standardize_by]) for row in usable}))
    stratum_index = {value: index for index, value in enumerate(strata)}
    grouped: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((len(strata), 4), dtype=float))
    for row in usable:
        stratum = "pooled" if standardize_by is None else str(row[standardize_by])
        grouped[str(row["user"])][stratum_index[stratum]] += np.asarray([
            row[f"{prefix}_n_p1_correct"],
            row[f"{prefix}_p2_correct_p1_correct"],
            row[f"{prefix}_n_p1_wrong"],
            row[f"{prefix}_p2_correct_p1_wrong"],
        ], dtype=float)
    users = sorted(grouped)
    if not users:
        return {
            "estimate": None, "ci_low": None, "ci_high": None,
            "p2_rate_given_p1_correct": None,
            "p2_rate_given_p1_wrong": None,
            "n_p1_correct": 0, "p2_correct_p1_correct": 0,
            "n_p1_wrong": 0, "p2_correct_p1_wrong": 0,
            "pooled_risk_difference": None,
            "standardization": standardize_by or "none",
            "n_strata": 0, "valid_bootstrap_replicates": 0,
            "n_rows": 0, "n_users": 0, "n_boot": n_boot, "seed": seed,
        }
    matrix = np.stack([grouped[user] for user in users])
    total = matrix.sum(axis=0)
    if np.any(total[:, 0] == 0) or np.any(total[:, 2] == 0):
        missing = [strata[index] for index in range(len(strata))
                   if total[index, 0] == 0 or total[index, 2] == 0]
        raise ValueError(f"conditional contrast lacks both p1 groups in strata {missing}")
    stratum_rate1 = total[:, 1] / total[:, 0]
    stratum_rate0 = total[:, 3] / total[:, 2]
    stratum_rd = stratum_rate1 - stratum_rate0
    estimate = float(stratum_rd.mean())
    pooled = total.sum(axis=0)
    n1, y1, n0, y0 = pooled
    rate1, rate0 = y1 / n1, y0 / n0
    pooled_rd = rate1 - rate0
    ci_low = ci_high = None
    valid_replicates = 0
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(users), size=(n_boot, len(users)))
        totals = matrix[draws].sum(axis=1)
        valid = np.all((totals[:, :, 0] > 0) & (totals[:, :, 2] > 0), axis=1)
        boot_totals = totals[valid]
        boot = (boot_totals[:, :, 1] / boot_totals[:, :, 0]
                - boot_totals[:, :, 3] / boot_totals[:, :, 2]).mean(axis=1)
        valid_replicates = len(boot)
        if len(boot):
            ci_low = float(np.quantile(boot, 0.025))
            ci_high = float(np.quantile(boot, 0.975))
    return {
        "estimate": finite_or_none(float(estimate)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p2_rate_given_p1_correct": float(stratum_rate1.mean()),
        "p2_rate_given_p1_wrong": float(stratum_rate0.mean()),
        "n_p1_correct": int(n1),
        "p2_correct_p1_correct": int(y1),
        "n_p1_wrong": int(n0),
        "p2_correct_p1_wrong": int(y0),
        "pooled_risk_difference": float(pooled_rd),
        "pooled_p2_rate_given_p1_correct": float(rate1),
        "pooled_p2_rate_given_p1_wrong": float(rate0),
        "standardization": standardize_by or "none",
        "n_strata": len(strata),
        "valid_bootstrap_replicates": valid_replicates,
        "n_rows": len(usable),
        "n_users": len(users),
        "n_boot": n_boot,
        "seed": seed,
    }


def stable_identity_hash(rows: Sequence[dict]) -> str:
    return hashing.stable_json_hash([identity(row) for row in rows])


def fixed_cell_heterogeneity(inference_rows: Sequence[dict], metric: str) -> dict:
    """Describe the finite archived cells without treating them as replicates."""

    rows = [
        row for row in inference_rows
        if row["variant"] != "pooled"
        and row["row_scope"] == "all_rows"
        and row["target_scope"] == "collision_clean"
        and row["metric"] == metric
        and row.get("estimate") is not None
    ]
    if not rows:
        raise ValueError(f"no per-cell inference rows for {metric}")
    values = [float(row["estimate"]) for row in rows]
    by_quantizer: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_quantizer[str(row["variant"]).split("_", 1)[0]].append(
            float(row["estimate"]))
    return {
        "metric": metric,
        "n_cells": len(values),
        "equal_cell_mean": mean(values),
        "min": min(values),
        "max": max(values),
        "n_positive": sum(value > 0 for value in values),
        "n_negative": sum(value < 0 for value in values),
        "n_zero": sum(value == 0 for value in values),
        "quantizer_equal_cell_means": {
            quantizer: mean(group) for quantizer, group in sorted(by_quantizer.items())
        },
        "scope": "descriptive finite-cell heterogeneity; cells are not stochastic replicates",
    }


def analyze_file(pred_path: Path, category: str,
                 canonical_identity_hash: str | None = None) -> tuple[
                     dict, list[dict], list[dict], str]:
    match = FILE_RE.match(pred_path.name)
    if not match or match["category"] != category:
        raise ValueError(f"unexpected prediction filename: {pred_path.name}")
    quantizer = match["quantizer"]
    depth, width = int(match["depth"]), int(match["width"])
    variant = f"{quantizer}_{depth}codebook_{width}"
    test_path = split_path(category, "test", quantizer, depth, width)
    train_path = split_path(category, "train", quantizer, depth, width)
    valid_path = split_path(category, "valid", quantizer, depth, width)
    catalogue_path = info_path(category, quantizer, depth, width)
    metric_path = pred_path.with_name(
        pred_path.name.replace(".predictions.json", ".json"))

    test_rows = read_csv(test_path)
    predictions = json.loads(pred_path.read_text())
    if len(test_rows) != len(predictions):
        raise ValueError(
            f"{variant}: {len(test_rows)} test rows != "
            f"{len(predictions)} prediction rows")
    row_hash = stable_identity_hash(test_rows)
    if canonical_identity_hash is not None and row_hash != canonical_identity_hash:
        raise ValueError(f"{variant}: row identity/order differs across cells")

    item_to_sid, items_by_sid = load_info(catalogue_path, depth)
    global_transitions, transitions_by_user, train_users = (
        load_training_transitions(train_path, depth))
    valid_users = {row["user_id"] for row in read_csv(valid_path)}
    test_users = {row["user_id"] for row in test_rows}
    final_indices = final_row_indices(test_rows)
    records: list[dict] = []

    invalid_p1 = invalid_p2 = empty_p1 = empty_p2 = 0
    duplicate_raw_p1 = duplicate_raw_pairs = 0
    duplicate_normalized_p1 = duplicate_normalized_pairs = 0
    gt_alignment = info_alignment = 0
    total_pairs = rows_lt20 = 0

    for row_index, (split, pred) in enumerate(zip(test_rows, predictions)):
        if not isinstance(pred, dict) or not {"gt1", "gt2", "top_pairs"} <= set(pred):
            raise ValueError(f"{variant} prediction row {row_index}: bad schema")
        gt1, gt2 = parse_pair(split["item_sid"], depth)
        pred_gt1, pred_gt2 = parse_sid(pred["gt1"], depth), parse_sid(pred["gt2"], depth)
        if (pred_gt1, pred_gt2) != (gt1, gt2):
            raise ValueError(
                f"{variant} row {row_index}: prediction/split GT mismatch")
        gt_alignment += 1
        item1, item2 = parse_item_pair(split["item_id"])
        if item_to_sid.get(item1) != gt1 or item_to_sid.get(item2) != gt2:
            raise ValueError(
                f"{variant} row {row_index}: split target/info mismatch")
        info_alignment += 1

        raw_pairs = pred["top_pairs"]
        if not isinstance(raw_pairs, list):
            raise ValueError(f"{variant} row {row_index}: top_pairs is not a list")
        if not 1 <= len(raw_pairs) <= 20:
            raise ValueError(
                f"{variant} row {row_index}: expected 1..20 top_pairs, "
                f"found {len(raw_pairs)}")
        parsed: list[tuple[tuple[int, ...] | None, tuple[int, ...] | None]] = []
        raw_p1_seen: set[str] = set()
        raw_pair_seen: set[tuple[str, str]] = set()
        normalized_p1_seen: set[str] = set()
        normalized_pair_seen: set[tuple[str, str]] = set()
        valid_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        valid_p2: list[tuple[int, ...]] = []
        singleton_valid_pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self_pairs = 0
        for beam_index, pair in enumerate(raw_pairs):
            if not isinstance(pair, dict) or not {"pred1", "pred2"} <= set(pair):
                raise ValueError(
                    f"{variant} row {row_index} beam {beam_index}: bad schema")
            raw1, raw2 = str(pair["pred1"]), str(pair["pred2"])
            if raw1 in raw_p1_seen:
                duplicate_raw_p1 += 1
            raw_p1_seen.add(raw1)
            if (raw1, raw2) in raw_pair_seen:
                duplicate_raw_pairs += 1
            raw_pair_seen.add((raw1, raw2))
            if not raw1.strip():
                empty_p1 += 1
            if not raw2.strip():
                empty_p2 += 1
            p1, p2 = safe_sid(raw1, depth), safe_sid(raw2, depth)
            if p1 is None or p1 not in items_by_sid:
                invalid_p1 += 1
                p1 = None
            if p2 is None or p2 not in items_by_sid:
                invalid_p2 += 1
                p2 = None
            normalized1 = "" if p1 is None else repr(p1)
            normalized2 = "" if p2 is None else repr(p2)
            if normalized1 in normalized_p1_seen:
                duplicate_normalized_p1 += 1
            normalized_p1_seen.add(normalized1)
            if (normalized1, normalized2) in normalized_pair_seen:
                duplicate_normalized_pairs += 1
            normalized_pair_seen.add((normalized1, normalized2))
            parsed.append((p1, p2))
            if p1 is not None and p2 is not None:
                valid_pairs.append((p1, p2))
                valid_p2.append(p2)
                self_pairs += int(p1 == p2)
                if len(items_by_sid[p1]) == len(items_by_sid[p2]) == 1:
                    singleton_valid_pairs.append((p1, p2))

        total_pairs += len(raw_pairs)
        rows_lt20 += int(len(raw_pairs) < 20)
        ordered_rank = first_rank(parsed, (gt1, gt2))
        reversed_rank = first_rank(parsed, (gt2, gt1))
        slot1 = slot_rank(parsed, 0, gt1)
        slot2 = slot_rank(parsed, 1, gt2)
        multiplicity1 = len(items_by_sid[gt1])
        multiplicity2 = len(items_by_sid[gt2])
        collision_clean = multiplicity1 == multiplicity2 == 1
        swap_eligible = (item1 != item2 and gt1 != gt2)
        p2_unique_fraction = (
            len(set(valid_p2)) / len(valid_p2) if valid_p2 else 0.0)
        user = split["user_id"]
        heldout = transitions_by_user.get(user, Counter())
        assoc = coupling_lift(valid_pairs, global_transitions, heldout)
        assoc_singleton = coupling_lift(
            singleton_valid_pairs, global_transitions, heldout)
        top_pair = parsed[0] if parsed else (None, None)
        top_observed = int(bool(parsed))
        top_p1_correct = int(top_pair[0] == gt1)
        top_p2_correct = int(top_pair[1] == gt2)
        all_p1_correct = [int(p1 == gt1) for p1, _ in parsed]
        all_p2_correct = [int(p2 == gt2) for _, p2 in parsed]
        all_n_p1_correct = sum(all_p1_correct)
        all_n_p1_wrong = len(parsed) - all_n_p1_correct
        all_p2_p1_correct = sum(
            p1_ok * p2_ok for p1_ok, p2_ok
            in zip(all_p1_correct, all_p2_correct))
        all_p2_p1_wrong = sum(
            (1 - p1_ok) * p2_ok for p1_ok, p2_ok
            in zip(all_p1_correct, all_p2_correct))
        record: dict[str, object] = {
            "variant": variant,
            "quantizer": quantizer,
            "depth": depth,
            "width": width,
            "row_index": row_index,
            "user": user,
            "is_final_test_row": row_index in final_indices,
            "history_len": len(ast.literal_eval(split["history_item_id"])),
            "item1": item1,
            "item2": item2,
            "same_target_item": item1 == item2,
            "same_target_sid": gt1 == gt2,
            "same_sid_distinct_items": item1 != item2 and gt1 == gt2,
            "target1_multiplicity": multiplicity1,
            "target2_multiplicity": multiplicity2,
            "collision_clean": collision_clean,
            "swap_eligible": swap_eligible,
            "conditional_pair_eligible": swap_eligible,
            "n_pairs": len(raw_pairs),
            "n_valid_pairs": len(valid_pairs),
            "n_singleton_valid_pairs": len(singleton_valid_pairs),
            "p2_unique_fraction": p2_unique_fraction,
            "p2_entropy_normalized": normalized_entropy(valid_p2),
            "self_pair_fraction": self_pairs / len(valid_pairs) if valid_pairs else 0.0,
            "ordered_rank": ordered_rank,
            "reversed_rank": reversed_rank,
            "slot1_rank": slot1,
            "slot2_rank": slot2,
            "ordered_rr": reciprocal_rank(ordered_rank),
            "reversed_rr": reciprocal_rank(reversed_rank),
            "coupling_lift": assoc,
            "coupling_lift_singleton_beams": assoc_singleton,
            "top1_n_p1_correct": top_p1_correct,
            "top1_p2_correct_p1_correct": top_p1_correct * top_p2_correct,
            "top1_n_p1_wrong": top_observed - top_p1_correct,
            "top1_p2_correct_p1_wrong": (
                top_observed - top_p1_correct) * top_p2_correct,
            "all_pairs_n_p1_correct": all_n_p1_correct,
            "all_pairs_p2_correct_p1_correct": all_p2_p1_correct,
            "all_pairs_n_p1_wrong": all_n_p1_wrong,
            "all_pairs_p2_correct_p1_wrong": all_p2_p1_wrong,
        }
        for cutoff in CUTOFFS:
            ordered = hit(ordered_rank, cutoff)
            reversed_ = hit(reversed_rank, cutoff)
            record[f"slot1_hit@{cutoff}"] = hit(slot1, cutoff)
            record[f"slot2_hit@{cutoff}"] = hit(slot2, cutoff)
            record[f"ordered_pair_hit@{cutoff}"] = ordered
            record[f"reversed_pair_hit@{cutoff}"] = reversed_
            record[f"unordered_pair_hit@{cutoff}"] = max(ordered, reversed_)
            # A SID hit only identifies a Cartesian bucket.  With no within-
            # bucket score, exact-item pair accuracy is bounded as follows.
            record[f"representative_item_pair_upper@{cutoff}"] = ordered
            record[f"representative_item_pair_lower@{cutoff}"] = (
                ordered if collision_clean else 0.0)
            record[f"representative_item_pair_uniform@{cutoff}"] = (
                ordered / (multiplicity1 * multiplicity2))
        records.append(record)

    # Exact legacy partial-credit HR validation.  This catches rank-order drift
    # without adopting the legacy metric as a dependence outcome.
    recorded_metrics = json.loads(metric_path.read_text()).get("metrics", {})
    legacy_validation = {}
    for cutoff in CUTOFFS[1:]:
        exact = mean(
            max(r[f"slot1_hit@{cutoff}"], r[f"slot2_hit@{cutoff}"]) / 2
            + r[f"ordered_pair_hit@{cutoff}"] / 2
            for r in records) * 100
        # Algebra: none=0, one slot=.5, both ordered=1.  `max/2 + joint/2`
        # reproduces that maximum over stored pairs only when slot hits can
        # occur in different pairs? It can overstate in that case, so compute
        # the exact per-pair legacy value below if needed.
        exact_values = []
        for split, pred in zip(test_rows, predictions):
            gt = parse_pair(split["item_sid"], depth)
            rels = []
            for pair in pred["top_pairs"][:cutoff]:
                p1, p2 = safe_sid(pair["pred1"], depth), safe_sid(pair["pred2"], depth)
                rels.append((int(p1 == gt[0]) + int(p2 == gt[1])) / 2)
            exact_values.append(max(rels, default=0.0))
        exact = mean(exact_values) * 100
        old = recorded_metrics.get(f"HR@{cutoff}")
        difference = exact - float(old) if old is not None else None
        if difference is not None and abs(difference) > 0.011:
            raise ValueError(
                f"{variant}: recomputed legacy HR@{cutoff}={exact:.6f} vs {old}")
        legacy_validation[f"HR@{cutoff}"] = {
            "recomputed_percent": exact,
            "recorded_percent": old,
            "difference_pp": difference,
        }

    validation = {
        "variant": variant,
        "quantizer": quantizer,
        "depth": depth,
        "width": width,
        "n_rows": len(records),
        "n_users": len(test_users),
        "repeated_user_rows": len(records) - len(test_users),
        "max_rows_per_user": max(Counter(r["user"] for r in records).values()),
        "n_final_rows": len(final_indices),
        "gt_rows_aligned": gt_alignment,
        "info_target_rows_aligned": info_alignment,
        "row_identity_sha256": row_hash,
        "n_catalogue_items": len(item_to_sid),
        "n_unique_sids": len(items_by_sid),
        "max_sid_bucket": max(map(len, items_by_sid.values())),
        "target_collision_rows": sum(not r["collision_clean"] for r in records),
        "same_target_item_rows": sum(r["same_target_item"] for r in records),
        "same_sid_distinct_item_rows": sum(r["same_sid_distinct_items"] for r in records),
        "n_beam_pairs": total_pairs,
        "rows_with_fewer_than_20_pairs": rows_lt20,
        "invalid_pred1": invalid_p1,
        "invalid_pred2": invalid_p2,
        "empty_pred1": empty_p1,
        "empty_pred2": empty_p2,
        "duplicate_raw_pred1_within_row": duplicate_raw_p1,
        "duplicate_raw_pair_within_row": duplicate_raw_pairs,
        "duplicate_after_invalid_normalization_pred1": duplicate_normalized_p1,
        "duplicate_after_invalid_normalization_pair": duplicate_normalized_pairs,
        "train_users": len(train_users),
        "valid_users": len(valid_users),
        "test_users": len(test_users),
        "train_test_user_overlap": len(train_users & test_users),
        "valid_test_user_overlap": len(valid_users & test_users),
        "prediction_sha256": hashing.sha256_file(pred_path),
        "metric_sha256": hashing.sha256_file(metric_path),
        "test_sha256": hashing.sha256_file(test_path),
        "train_sha256": hashing.sha256_file(train_path),
        "valid_sha256": hashing.sha256_file(valid_path),
        "info_sha256": hashing.sha256_file(catalogue_path),
        "legacy_metric_validation": legacy_validation,
    }
    association_rows = [
        {
            "variant": r["variant"], "row_index": r["row_index"],
            "user": r["user"], "is_final_test_row": r["is_final_test_row"],
            "n_valid_pairs": r["n_valid_pairs"],
            "n_singleton_valid_pairs": r["n_singleton_valid_pairs"],
            "coupling_lift": r["coupling_lift"],
            "coupling_lift_singleton_beams": r["coupling_lift_singleton_beams"],
            "p2_unique_fraction": r["p2_unique_fraction"],
            "p2_entropy_normalized": r["p2_entropy_normalized"],
            "self_pair_fraction": r["self_pair_fraction"],
        }
        for r in records
    ]
    return validation, records, association_rows, row_hash


def cohort_alignment(rows: Sequence[dict]) -> dict:
    """Map AR's zero-based A/item IDs to the raw Diffusion sequence cohort."""

    sequence_path = paths.FROZEN_DATA / "sequences" / "all_item_seqs.json"
    user_map_path = paths.FROZEN_DATA / "id_maps" / f"{CATEGORY}.user2id"
    item_map_path = paths.FROZEN_DATA / "id_maps" / f"{CATEGORY}.item2id"
    sequences = json.loads(sequence_path.read_text())

    def inverse_tsv(path: Path) -> dict[int, str]:
        answer: dict[int, str] = {}
        with path.open() as fh:
            for line_no, line in enumerate(fh, start=1):
                raw, number = line.rstrip("\n").split("\t")
                if int(number) in answer:
                    raise ValueError(f"{path}:{line_no}: duplicate numeric ID")
                answer[int(number)] = raw
        return answer

    users, items = inverse_tsv(user_map_path), inverse_tsv(item_map_path)
    final = final_row_indices(rows)
    mapped = contiguous = final_pair = 0
    final_users: set[str] = set()
    for index, row in enumerate(rows):
        user_number = int(row["user_id"].removeprefix("A"))
        raw_user = users.get(user_number)
        hist = [items[int(item)] for item in ast.literal_eval(row["history_item_id"])]
        item1, item2 = parse_item_pair(row["item_id"])
        pair = [items[item1], items[item2]]
        if raw_user not in sequences:
            continue
        mapped += 1
        seq = sequences[raw_user]
        full = hist + pair
        if any(seq[start:start + len(full)] == full
               for start in range(len(seq) - len(full) + 1)):
            contiguous += 1
        if index in final:
            final_users.add(raw_user)
            final_pair += int(seq[-2:] == pair)
    lengths = [len(seq) for seq in sequences.values()]
    # Mirror AbstractDataset._sliding_train exactly.  With two targets, its
    # implemented window condition admits t=2 (three-item window), hence one
    # history item despite the configured/documented min_hist=2.
    diffusion_train_history_lengths = [
        t - 1
        for length in lengths
        for t in range(2, min(length - 3, 50) + 1)
        if t + 1 >= 3
    ]
    return {
        "ar_test_rows": len(rows),
        "ar_test_users": len({r["user_id"] for r in rows}),
        "rows_mapped_to_diffusion_raw_sequences": mapped,
        "rows_contiguous_in_diffusion_raw_sequences": contiguous,
        "ar_final_rows": len(final),
        "ar_final_rows_matching_diffusion_final_pair": final_pair,
        "ar_final_users_in_diffusion_cohort": len(final_users),
        "diffusion_test_users": len(sequences),
        "diffusion_interactions": sum(map(len, sequences.values())),
        "diffusion_train_examples_implemented": len(diffusion_train_history_lengths),
        "diffusion_valid_examples": sum(length >= 4 for length in lengths),
        "diffusion_test_examples": sum(length >= 3 for length in lengths),
        "diffusion_train_history_min_implemented": min(diffusion_train_history_lengths),
        "diffusion_train_history_median_implemented": float(np.median(
            diffusion_train_history_lengths)),
        "diffusion_train_history_max_implemented": max(diffusion_train_history_lengths),
        "diffusion_min_hist_configuration": 2,
        "diffusion_min_hist_implementation_mismatch": True,
        "sequence_sha256": hashing.sha256_file(sequence_path),
        "user_map_sha256": hashing.sha256_file(user_map_path),
        "item_map_sha256": hashing.sha256_file(item_map_path),
        "interpretation": (
            "One last AR test row per AR test user maps to the same raw final "
            "two-item target used by diffusion, but retained diffusion metrics "
            "are aggregate over its full user cohort and cannot be subset."),
    }


def diffusion_source_audit(repo: Path) -> dict:
    """Static audit of the exact retained next-two training/evaluation path."""

    model_path = repo / "vendor/diffgrm_new/genrec/models/DIFF_GRM/model.py"
    beam_path = repo / "vendor/diffgrm_new/genrec/models/DIFF_GRM/beam.py"
    trainer_path = repo / "vendor/diffgrm_new/genrec/models/DIFF_GRM/trainer.py"
    tokenizer_path = repo / "vendor/diffgrm_new/genrec/models/DIFF_GRM/tokenizer.py"
    dataset_path = repo / "vendor/diffgrm_new/genrec/dataset.py"
    sources = {p.name: p.read_text() for p in
               (model_path, beam_path, trainer_path, tokenizer_path, dataset_path)}

    beam_tree = ast.parse(sources["beam.py"])
    fast = next(node for node in beam_tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "fast_beam_search_for_eval")
    iterative = next(node for node in beam_tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "iterative_mask_decode")
    iterative_assigns_model_n_digit = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "n_digit"
                for target in node.targets)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "model"
        and node.value.attr == "n_digit"
        for node in ast.walk(iterative))
    max_len_loads = sum(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and node.id == "max_len" for node in ast.walk(fast))
    decoder_is_bidirectional = (
        "attention_mask=None,  # 不使用因果掩码" in sources["model.py"])
    trainer_appends_pred1 = "_augment_history_with_pred(pass1_batch, pred1_i)" in sources["trainer.py"]
    trainer_slices_first = "preds1[:, :, :n_digit]" in sources["trainer.py"]
    trainer_slices_last = "p2[:, :, -n_digit:]" in sources["trainer.py"]
    tokenizer_concatenates_targets = (
        "decoder_input_all.extend(cb_input)" in sources["tokenizer.py"]
        and "target_items = item_seq[-n_target_items:]" in sources["tokenizer.py"])
    return {
        "training_decoder_positions": "2*n_digit for n_target_items=2",
        "training_target_items_concatenated": tokenizer_concatenates_targets,
        "decoder_self_attention": "bidirectional/no causal mask",
        "decoder_bidirectional_verified": decoder_is_bidirectional,
        "inference_iterative_length_assignment": "model.n_digit",
        "inference_hardcodes_one_sid_verified": iterative_assigns_model_n_digit,
        "fast_beam_max_len_runtime_loads": max_len_loads,
        "fast_beam_max_len_is_ignored": max_len_loads == 0,
        "trainer_takes_first_sid_in_pass1": trainer_slices_first,
        "trainer_appends_pass1_sid_to_history": trainer_appends_pred1,
        "trainer_takes_last_slice_in_pass2": trainer_slices_last,
        "effective_retained_evaluation": (
            "Generate one SID from decoder positions 0..n_digit-1; append it "
            "to history; generate one SID from the same positions again. The "
            "second slice is the whole one-SID output, so it is the immediate "
            "next item, not a horizon+2 item."),
        "mechanistic_consequence": (
            "Training couples both items through bidirectional masked self-"
            "attention, but retained evaluation never directly decodes positions "
            "n_digit..2*n_digit-1. It is outer autoregression with the first-slot "
            "decoder reused, not a retained joint-block generation condition."),
        "source_sha256": {
            str(path.relative_to(repo)): hashing.sha256_file(path)
            for path in (model_path, beam_path, trainer_path, tokenizer_path, dataset_path)
        },
    }


def availability_audit(pred_paths: Sequence[Path]) -> dict:
    ar_cells = []
    for path in pred_paths:
        match = FILE_RE.match(path.name)
        if match:
            ar_cells.append({
                "quantizer": match["quantizer"],
                "depth": int(match["depth"]),
                "width": int(match["width"]),
                "per_example_pairs": True,
                "scores": False,
            })
    label_path = (paths.FROZEN_CKPT / "ar" / "best_variant_labels" /
                  "two-item" / "best" / f"{CATEGORY}.variant")
    ar_weight_cell = label_path.read_text().strip()
    registry_path = paths.MANIFESTS / "registry.diffusion.json"
    registry = json.loads(registry_path.read_text())
    diffusion = []
    for key, record in registry.items():
        if record.get("task") != "next2":
            continue
        ckpt = Path(record["ckpt_path"])
        transcript = Path(record["transcript_path"])
        if not ckpt.is_file() or not transcript.is_file():
            raise ValueError(f"registry path missing for {key}")
        if hashing.sha256_file(ckpt) != record["ckpt_sha256"]:
            raise ValueError(f"checkpoint hash drift for {key}")
        if hashing.sha256_file(transcript) != record["transcript_sha256"]:
            raise ValueError(f"transcript hash drift for {key}")
        diffusion.append({
            "cell": key,
            "quantizer": record["quantizer"],
            "depth": record["n_codebook"],
            "width": record["codebook_size"],
            "status": record["status"],
            "per_example_pairs": False,
            "aggregate_metrics": bool(record.get("recorded_metrics")),
            "checkpoint": str(ckpt),
        })
    ar_representations = {
        (cell["quantizer"], cell["depth"], cell["width"]) for cell in ar_cells}
    diff_representations = {
        (cell["quantizer"], cell["depth"], cell["width"])
        for cell in diffusion if cell["status"] == "trained"}
    return {
        "ar_prediction_cells": ar_cells,
        "ar_prediction_cell_count": len(ar_cells),
        "ar_retained_weight_cell": ar_weight_cell,
        "diffusion_next2_cells": diffusion,
        "diffusion_trained_cell_count": sum(
            cell["status"] == "trained" for cell in diffusion),
        "representation_overlap": [
            {"quantizer": q, "depth": d, "width": w}
            for q, d, w in sorted(ar_representations & diff_representations)
        ],
        "fully_matched_mechanistic_cell": False,
        "why_not_matched": (
            "The retained AR weight is MQ 4x256, whereas trained diffusion "
            "weights are rqvae 3x256 and 4x256. Historical AR predictions do "
            "overlap the rqvae representations, but their weights are absent; "
            "diffusion has no per-example pair export."),
        "registry_sha256": hashing.sha256_file(registry_path),
        "ar_label_sha256": hashing.sha256_file(label_path),
    }


def render_report(result: dict, configuration_rows: list[dict],
                  inference_rows: list[dict]) -> str:
    validation = result["validation_summary"]
    cohort = result["cohort_alignment"]
    availability = result["availability"]
    heterogeneity = result["cell_heterogeneity"]
    top1_heterogeneity = heterogeneity["top1_conditional_p2_risk_difference"]
    quantizer_means = ", ".join(
        f"{name} {100 * value:+.2f} pp"
        for name, value in top1_heterogeneity["quantizer_equal_cell_means"].items())
    primary = {row["metric"]: row for row in inference_rows
               if row["variant"] == "pooled"
               and row["row_scope"] == "all_rows"
               and row["target_scope"] == "collision_clean"}

    def effect(name: str) -> str:
        row = primary.get(name)
        if not row or row.get("estimate") is None:
            return "not estimable"
        return (f"{row['estimate']:+.4f} (user-cluster bootstrap 95% CI "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}])")

    def conditional_effect(name: str) -> str:
        row = primary.get(name)
        if not row or row.get("estimate") is None:
            return "not estimable"
        return (
            f"equal-cell P(p2 correct | p1 correct)="
            f"{100 * row['p2_rate_given_p1_correct']:.2f}% "
            f"versus P(p2 correct | p1 wrong)="
            f"{100 * row['p2_rate_given_p1_wrong']:.2f}% "
            f"(risk difference **{100 * row['estimate']:+.2f} "
            f"percentage points** (user-cluster bootstrap 95% CI "
            f"[{100 * row['ci_low']:+.2f}, {100 * row['ci_high']:+.2f}])). "
            f"The raw pooled contingency is "
            f"{100 * row['pooled_p2_rate_given_p1_correct']:.2f}% "
            f"({row['p2_correct_p1_correct']}/{row['n_p1_correct']}) versus "
            f"{100 * row['pooled_p2_rate_given_p1_wrong']:.2f}% "
            f"({row['p2_correct_p1_wrong']}/{row['n_p1_wrong']}), RD "
            f"{100 * row['pooled_risk_difference']:+.2f} points")

    return f"""# Retrospective Experiment 4: next-two conditional-association audit

## Result

All {validation['n_ar_cells']} retained AR cells passed row-by-row target and
target-to-catalogue alignment ({validation['n_prediction_rows']:,} cell-example rows;
{validation['n_unique_test_rows']:,} unique chronological examples).  The
primary collision-clean, all-event descriptive summaries across cells
are:

- **Primary conditional contingency (top-ranked pair):**
  {conditional_effect('top1_conditional_p2_risk_difference')}.
- All-retained-pair conditional sensitivity:
  {conditional_effect('all_pairs_conditional_p2_risk_difference')}.
- slot-2 minus slot-1 hit@10: **{effect('slot2_minus_slot1_hit@10')}**;
- ordered minus reverse-pair hit@20: **{effect('ordered_minus_reversed_pair_hit@20')}**;
- stored-pair mean log1p training-transition-count advantage over off-diagonal
  within-row re-pairing, restricted to singleton predicted SID buckets:
  **{effect('coupling_lift_singleton_beams')}**
  (all valid SID beams: **{effect('coupling_lift')}**).

The top-pair conditional association is heterogeneous: only
{top1_heterogeneity['n_positive']}/{top1_heterogeneity['n_cells']} cell estimates
are positive, spanning {100 * top1_heterogeneity['min']:+.2f} to
{100 * top1_heterogeneity['max']:+.2f} points; quantizer-level equal-cell means
are {quantizer_means}.  The non-conditional pooled contrasts above are
event-weighted descriptions of this fixed, unbalanced archive.  For example,
ordered-minus-
reversed hit@20 is {100 * heterogeneity['ordered_minus_reversed_pair_hit@20']['equal_cell_mean']:+.2f}
points under equal-cell weighting versus
{100 * primary['ordered_minus_reversed_pair_hit@20']['estimate']:+.2f} points
under event weighting.

Of {validation['n_beam_pairs']:,} stored pairs,
{validation['invalid_pred1_total']:,} ({validation['invalid_pred1_fraction']:.2%})
have an invalid first SID and {validation['invalid_pred2_total']:,} have an
invalid second SID.  Of the invalid first SIDs,
{validation['empty_pred1_total']:,} are empty and
{validation['nonempty_invalid_pred1_total']:,} are nonempty malformed or
catalogue-unknown outputs.  Their positions are retained: an invalid first SID
is a slot-1 and exact-pair miss, while a valid second SID remains eligible for
slot-2 retrieval and enters the p1-wrong conditional group.  Coupling and
diversity summaries exclude any pair containing an invalid SID.

These are associations, not estimates of causal dependence.  In AR, each row
contains up to 20 first-pass outputs deduplicated by their raw decoded strings,
one greedy second item per first item, and no scores.  Pair order is inherited
exclusively from the first pass;
there is no counterfactual distribution of item 2 under controlled item-1
interventions.  Moreover, conditioning on p1 correctness selects easier rows
and is not a randomized intervention.  `configuration.csv` reports the full cutoff curves and
`row_metrics.csv` preserves the audit trail.

## Collision and repeated-user controls

If each ranked SID pair is converted to one representative item pair, exact-SID
hits give lower/uniform/upper target-item-pair sensitivity under adversarial,
uniform, or oracle within-bucket selection.  These are not bounds for a full
ranked Cartesian item expansion, where earlier collision buckets consume
additional ranks.  Primary
contrasts exclude collided target buckets; the swap contrast additionally
excludes equal-SID and equal-item pairs.  The conditional contingencies use
that same distinct-target exclusion so repeated ground-truth items cannot make
slot agreement tautological.
Uncertainty resamples {validation['n_unique_test_users']:,} users—not rows—and
keeps all their repeated events and representation cells together.  These
intervals condition on the fixed one-seed cells and do not cover training-seed
or configuration uncertainty.  A second scope keeps only the last test row per
user.

## Cross-paradigm availability

There are {availability['ar_prediction_cell_count']} AR per-example prediction
cells, but only `{availability['ar_retained_weight_cell']}` retains AR weights.
Diffusion retains {availability['diffusion_trained_cell_count']} trained next-two
checkpoints and aggregate transcripts, with no per-example pairs or scores.
Although {cohort['ar_final_rows_matching_diffusion_final_pair']:,}/
{cohort['ar_final_rows']:,} final AR rows map to the same raw final pair in the
diffusion sequence cohort, the aggregate diffusion results cannot be restricted
to those users.  There is therefore no fully matched AR/joint-diffusion
mechanistic comparison in the snapshot.

## Diffusion source-semantic audit

The diffusion model trains on two concatenated SIDs with bidirectional masked
decoder attention.  Its retained beam path, however, hard-codes generation to
`model.n_digit`; the `max_len=2*n_digit` argument is unused.  Evaluation thus
generates one SID, appends it to history, and generates one SID again from the
first-slot positions.  This correctly targets the immediate following item,
but it does **not** directly evaluate the trained second-half positions or a
joint two-item block.  Source hashes and executable static checks are recorded
in `result.json`.

## What would establish causal dependence

Rerun the retained MQ 4x256 AR checkpoint with item 1 clamped to ground truth,
its top prediction, matched-popularity controls, and valid derangements; export
item-2 logits/scores and compare NLL, JS divergence, and rank changes within the
same history.  For diffusion, export one-pass 2*d masked-decoder trajectories,
reveal/replace each item while holding history, seed, and mask schedule fixed,
and measure directional item1->item2 versus item2->item1 influence.  Use one
final row per user as the primary cohort, user-cluster inference for all-event
sensitivity, singleton SID targets, and a prospectively declared primary
contrast.
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)

    metrics_root = (paths.FROZEN_RESULTS / "sweep_metrics" /
                    "two-item" / "metrics")
    pred_paths = sorted(metrics_root.glob(
        f"twoitem__{args.category}__*.predictions.json"))
    if not pred_paths:
        raise SystemExit(f"no retained two-item predictions under {metrics_root}")

    validations: list[dict] = []
    records: list[dict] = []
    association_rows: list[dict] = []
    canonical_hash: str | None = None
    canonical_rows: list[dict] | None = None
    for pred_path in pred_paths:
        validation, cell_records, cell_association, row_hash = analyze_file(
            pred_path, args.category, canonical_hash)
        if canonical_hash is None:
            canonical_hash = row_hash
            match = FILE_RE.match(pred_path.name)
            assert match is not None
            canonical_rows = read_csv(split_path(
                args.category, "test", match["quantizer"],
                int(match["depth"]), int(match["width"])))
        validations.append(validation)
        records.extend(cell_records)
        association_rows.extend(cell_association)
        print(
            f"{validation['variant']:27s} rows={validation['n_rows']} "
            f"users={validation['n_users']} invalid="
            f"{validation['invalid_pred1']}/{validation['invalid_pred2']} "
            f"collision-clean="
            f"{1-validation['target_collision_rows']/validation['n_rows']:.1%}")

    expected = {
        ("MQ", 3, 256), ("MQ", 3, 512), ("MQ", 4, 256), ("MQ", 5, 256),
        *( (q, d, 256) for q in ("rqkmeans", "rqvae") for d in (3, 4, 5) ),
    }
    observed = {(v["quantizer"], v["depth"], v["width"])
                for v in validations}
    if observed != expected:
        raise ValueError(
            f"retained two-item grid changed; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}")

    configuration_rows: list[dict] = []
    inference_rows: list[dict] = []
    variants = [v["variant"] for v in validations]
    row_scopes = {
        "all_rows": lambda row: True,
        "final_row_per_user": lambda row: bool(row["is_final_test_row"]),
    }
    target_scopes = {
        "all_sid_targets": lambda row: True,
        "collision_clean": lambda row: bool(row["collision_clean"]),
    }

    for variant in variants:
        cell = [row for row in records if row["variant"] == variant]
        for row_scope, row_keep in row_scopes.items():
            for target_scope, target_keep in target_scopes.items():
                selected = [row for row in cell if row_keep(row) and target_keep(row)]
                summary = summarize_rows(selected, CUTOFFS)
                configuration_rows.append({
                    "variant": variant, "row_scope": row_scope,
                    "target_scope": target_scope, **summary,
                })

                contrast_specs = {
                    "slot2_minus_slot1_hit@10": [
                        (r["user"], r["slot2_hit@10"] - r["slot1_hit@10"])
                        for r in selected],
                    "ordered_minus_reversed_pair_hit@20": [
                        (r["user"], r["ordered_pair_hit@20"] - r["reversed_pair_hit@20"])
                        for r in selected if r["swap_eligible"]],
                    "coupling_lift": [
                        (r["user"], r["coupling_lift"])
                        for r in selected if r["coupling_lift"] is not None],
                    "coupling_lift_singleton_beams": [
                        (r["user"], r["coupling_lift_singleton_beams"])
                        for r in selected
                        if r["coupling_lift_singleton_beams"] is not None],
                }
                for offset, (metric, values) in enumerate(contrast_specs.items()):
                    boot = cluster_bootstrap(
                        values, args.bootstrap,
                        args.seed + 1009 * variants.index(variant) + offset)
                    inference_rows.append({
                        "variant": variant, "row_scope": row_scope,
                        "target_scope": target_scope, "metric": metric, **boot,
                        "interpretation": "descriptive association, not causal dependence",
                    })
                for offset, prefix in enumerate(("top1", "all_pairs"), start=10):
                    boot = cluster_bootstrap_conditional(
                        selected, prefix, args.bootstrap,
                        args.seed + 1009 * variants.index(variant) + offset)
                    inference_rows.append({
                        "variant": variant, "row_scope": row_scope,
                        "target_scope": target_scope,
                        "metric": f"{prefix}_conditional_p2_risk_difference",
                        **boot,
                        "interpretation": (
                            "P(p2 SID correct | p1 SID correct) minus P(p2 SID "
                            "correct | p1 SID wrong); descriptive, not causal"),
                    })

    # Pooled inference preserves representation cells inside the user cluster.
    for row_scope, row_keep in row_scopes.items():
        for target_scope, target_keep in target_scopes.items():
            selected = [row for row in records if row_keep(row) and target_keep(row)]
            specs = {
                "slot2_minus_slot1_hit@10": [
                    (r["user"], r["slot2_hit@10"] - r["slot1_hit@10"])
                    for r in selected],
                "ordered_minus_reversed_pair_hit@20": [
                    (r["user"], r["ordered_pair_hit@20"] - r["reversed_pair_hit@20"])
                    for r in selected if r["swap_eligible"]],
                "coupling_lift": [
                    (r["user"], r["coupling_lift"])
                    for r in selected if r["coupling_lift"] is not None],
                "coupling_lift_singleton_beams": [
                    (r["user"], r["coupling_lift_singleton_beams"])
                    for r in selected
                    if r["coupling_lift_singleton_beams"] is not None],
            }
            for offset, (metric, values) in enumerate(specs.items()):
                boot = cluster_bootstrap(
                    values, args.bootstrap, args.seed + 90001 + offset)
                inference_rows.append({
                    "variant": "pooled", "row_scope": row_scope,
                    "target_scope": target_scope, "metric": metric, **boot,
                    "interpretation": (
                        "descriptive association pooled over fixed cells; cells "
                        "are not independent training replicates"),
                })
            for offset, prefix in enumerate(("top1", "all_pairs"), start=10):
                boot = cluster_bootstrap_conditional(
                    selected, prefix, args.bootstrap,
                    args.seed + 90001 + offset,
                    standardize_by="variant")
                inference_rows.append({
                    "variant": "pooled", "row_scope": row_scope,
                    "target_scope": target_scope,
                    "metric": f"{prefix}_conditional_p2_risk_difference",
                    **boot,
                    "interpretation": (
                        "P(p2 SID correct | p1 SID correct) minus P(p2 SID "
                            "correct | p1 SID wrong), standardized with equal "
                            "weight over fixed cells; descriptive, not causal"),
                })

    assert canonical_rows is not None
    availability = availability_audit(pred_paths)
    source_audit = diffusion_source_audit(paths.REPO)
    cohort = cohort_alignment(canonical_rows)
    heterogeneity_metrics = (
        "top1_conditional_p2_risk_difference",
        "all_pairs_conditional_p2_risk_difference",
        "slot2_minus_slot1_hit@10",
        "ordered_minus_reversed_pair_hit@20",
        "coupling_lift_singleton_beams",
    )
    cell_heterogeneity = {
        metric: fixed_cell_heterogeneity(inference_rows, metric)
        for metric in heterogeneity_metrics
    }
    n_beam_pairs = sum(v["n_beam_pairs"] for v in validations)
    invalid_pred1 = sum(v["invalid_pred1"] for v in validations)
    invalid_pred2 = sum(v["invalid_pred2"] for v in validations)
    empty_pred1 = sum(v["empty_pred1"] for v in validations)
    empty_pred2 = sum(v["empty_pred2"] for v in validations)
    validation_summary = {
        "n_ar_cells": len(validations),
        "n_prediction_rows": len(records),
        "n_unique_test_rows": len(canonical_rows),
        "n_unique_test_users": len({r["user_id"] for r in canonical_rows}),
        "all_cells_same_row_identity_and_order": True,
        "all_gt_rows_aligned": all(v["gt_rows_aligned"] == v["n_rows"]
                               for v in validations),
        "all_info_targets_aligned": all(
            v["info_target_rows_aligned"] == v["n_rows"] for v in validations),
        "n_beam_pairs": n_beam_pairs,
        "invalid_pred1_total": invalid_pred1,
        "invalid_pred1_fraction": invalid_pred1 / n_beam_pairs,
        "invalid_pred2_total": invalid_pred2,
        "invalid_pred2_fraction": invalid_pred2 / n_beam_pairs,
        "empty_pred1_total": empty_pred1,
        "empty_pred2_total": empty_pred2,
        "nonempty_invalid_pred1_total": invalid_pred1 - empty_pred1,
        "nonempty_invalid_pred2_total": invalid_pred2 - empty_pred2,
    }
    snapshot = (paths.MANIFESTS / "CURRENT").read_text().strip()
    result = {
        "experiment": "retrospective-exp4-next-two-conditional-association-audit",
        "recovered_plan_label": "next-two conditional-dependence audit",
        "snapshot": snapshot,
        "category": args.category,
        "claim_boundary": {
            "identified": [
                "slot-specific retrieval differences",
                "ordered-versus-reversed pair rank association",
                "training-transition compatibility of stored beam pairing",
            ],
            "not_identified": [
                "causal effect of item 1 on item 2",
                "joint pair likelihood or calibrated probability",
                "item identity inside a collided SID bucket",
                "matched AR-versus-joint-diffusion mechanism difference",
            ],
            "reason": (
                "Archived AR results have one p2 per unique p1 and no scores or "
                "controlled p1 interventions; diffusion has aggregate metrics only."),
        },
        "ar_artifact_semantics": {
            "targets": (
                "gt1 and gt2 are consecutive chronological interactions; the "
                "two-item converter moves the last original history item to "
                "gt1 and keeps the original next-item target as gt2"),
            "row_join": (
                "Prediction JSON has no user/item IDs; it is joined by row "
                "position only after exact gt1/gt2 equality and common cross-cell "
                "identity/order are verified"),
            "pass1": (
                "50 constrained AR beams, decoded then deduplicated by exact "
                "string in rank order, truncated to at most 20 unique p1 values"),
            "pass2": (
                "For each retained p1, append p1 plus literal separator ' ||| ' "
                "to the original prompt and perform one constrained greedy decode"),
            "stored_rank": (
                "Pass1 beam order only; p2 has no stored score and pairs are not "
                "reranked by a joint or conditional score"),
            "absent_fields": [
                "user_id", "history", "item_id", "pass1_score", "pass2_score",
                "joint_score", "alternative p2 beams",
            ],
        },
        "validation_summary": validation_summary,
        "configurations": validations,
        "availability": availability,
        "cohort_alignment": cohort,
        "diffusion_source_semantics": source_audit,
        "cell_heterogeneity": cell_heterogeneity,
        "primary_inference": [
            row for row in inference_rows
            if row["variant"] == "pooled"
            and row["row_scope"] == "all_rows"
            and row["target_scope"] == "collision_clean"
        ],
        "method": {
            "primary_estimand": (
                "At the top-ranked stored pair, P(p2 exact SID correct | p1 "
                "exact SID correct) minus P(p2 exact SID correct | p1 exact "
                "SID wrong), computed inside each representation cell and equally "
                "averaged over the 10 cells, on singleton-SID distinct-target rows"),
            "conditional_sensitivity": (
                "The same contingency over every retained beam pair; user "
                "bootstrap keeps all beams from a row together"),
            "primary_cutoffs": {
                "slot_gap": 10, "pair_swap": 20,
                "why_pair_20": "ordered exact-pair hits are sparse",
            },
            "primary_population": (
                "all chronological AR test events with singleton target SID buckets; "
                "one-final-row-per-user is an analysis-designated sensitivity"),
            "swap_exclusions": (
                "same item and same SID pairs are not orientation-identifiable"),
            "beam_association": (
                "leave-test-user-out training transition log1p count for stored "
                "p1->p2 pairs minus the exact mean over off-diagonal within-row "
                "p2 re-pairings"),
            "uncertainty": (
                f"{args.bootstrap} percentile bootstrap draws over users; all rows "
                "and fixed representation cells for a user remain together"),
            "collision_policy": (
                "singleton targets are primary; all-SID lower/uniform/upper "
                "sensitivities assume one representative item pair per ranked SID "
                "pair, not a ranked Cartesian expansion"),
        },
        "validity_threats": [
            "Rows and representation cells are repeated measurements, not independent replicates.",
            "Train/valid/test are a global chronological event split with overlapping users.",
            "Beam rank is first-pass rank; p2 did not contribute a stored reranking score.",
            "Pass1 was deduplicated and pass2 is one greedy value, limiting conditional diversity.",
            "P1 correctness is a model outcome: P(p2 correct|p1 correct) can be larger because those rows are easier, even without a causal p1-to-p2 effect.",
            "SID collisions pool multiple items and can make SID equality optimistic.",
            "Reverse chronological pairs can be plausible even when unobserved; swap is an asymmetry probe, not a universal error label.",
            "Training-transition compatibility is observational even after excluding the same user's train rows.",
            "Only one trained endpoint/seed is represented per cell.",
            "Diffusion and AR evaluations use different full cohorts and no retained per-example diffusion output exists.",
            "Diffusion's implemented sliding-train loop admits one history item despite min_hist=2 in its configuration.",
        ],
    }

    out = (Path(args.out) if args.out else paths.DERIVED /
           "retrospective" / "exp4_next_two")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "artifact_validation.csv", [
        {key: (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
         for key, value in row.items()}
        for row in validations
    ])
    write_csv(out / "configuration.csv", configuration_rows)
    write_csv(out / "inference.csv", inference_rows)
    write_csv(out / "beam_association.csv", association_rows)
    write_csv(out / "row_metrics.csv", records)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "report.md").write_text(
        render_report(result, configuration_rows, inference_rows))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
