#!/usr/bin/env python
"""Retrospective Experiment 3 -- collision-conditioned SID accuracy.

This analysis uses the 18 retained next-item AR prediction files (the balanced
3 quantizers x 3 depths x {128,512} grid).  It answers two different questions
that the historical leaderboard conflated:

1. Is a *SID* that happens to represent several items easier or harder to rank?
2. What item-retrieval claim survives when the model supplied no score or
   tie-break among the items carrying that SID?

Outputs go to ``$SIDLENS_WORK/derived/retrospective/exp3_collisions``:

``configuration.csv``
    Per configuration/cutoff, singleton and collided SID hit rates plus the
    within-configuration risk difference.
``item_bounds.csv``
    Item-level lower, uniform-tie, stable-catalogue-order, and oracle upper hit
    rates after expanding each predicted SID through the lossless map.
``frequency_strata.csv``
    Collision contrasts coarsened by training SID frequency, training item
    frequency, and history length.  This is descriptive adjustment, not a
    causal estimate.
``result.json`` and ``report.md``
    Machine-readable provenance/validation and a concise interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from sidlens import paths
from sidlens.analysis.collisions import (
    exact_rank,
    exact_sign_test,
    item_rank_bounds,
    parse_sid,
)
from sidlens.provenance import hashing


CATEGORY = "Industrial_and_Scientific"
CUTOFFS = (1, 3, 5, 10, 20, 50)
FILE_RE = re.compile(
    r"^nextitem__(?P<category>.+)__(?P<quantizer>MQ|rqvae|rqkmeans)__"
    r"(?P<depth>\d+)cb__(?P<width>\d+)\.predictions\.json$")


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


def index_path(category: str, quantizer: str, depth: int, width: int) -> Path:
    root = paths.FROZEN_SIDS / "index_json"
    if quantizer == "MQ":
        name = f"{category}.index.MQ.{depth}codebook_{width}.json"
    elif quantizer == "rqvae":
        name = f"{category}.index_{depth}codebook_{width}.json"
    else:
        name = f"{category}.rqkmeans.index_{depth}codebook_{width}.json"
    return root / name


def split_path(category: str, split: str, quantizer: str,
               depth: int, width: int) -> Path:
    root = paths.FROZEN_DATA / "splits" / "next-item" / split
    candidates = sorted(root.glob(
        f"{category}_{quantizer}_{depth}codebook_{width}*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {split} split for {quantizer} {depth}x{width}, "
            f"found {candidates}")
    return candidates[0]


def load_index(path: Path, depth: int) -> tuple[
        dict[int, tuple[int, ...]], dict[tuple[int, ...], list[int]]]:
    raw = json.loads(path.read_text())
    item_to_sid = {
        int(item): parse_sid("".join(tokens), depth) for item, tokens in raw.items()
    }
    items_by_sid: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for item, sid in item_to_sid.items():
        items_by_sid[sid].append(item)
    return item_to_sid, dict(items_by_sid)


def quantile_edges(values: list[int], n_bins: int = 5) -> np.ndarray:
    """Precompute stable coarsening edges once per configuration."""

    if not values:
        return np.asarray([], dtype=float)
    return np.quantile(
        np.asarray(values, dtype=float), np.linspace(0, 1, n_bins + 1)[1:-1])


def quantile_bin(edges: np.ndarray, value: int) -> int:
    """Assign a value to precomputed edges; repeated boundaries stay visible."""

    return int(np.searchsorted(edges, value, side="right"))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def bootstrap_user_contrasts(records: list[dict], n_boot: int,
                             seed: int) -> dict:
    """Resample users while preserving paired representation cells.

    The primary collision contrast first computes a risk difference inside
    every configuration and then gives the 18 cells equal weight.  A raw
    pooled contrast is retained only as a sensitivity because collision
    prevalence varies sharply by representation.
    """

    if n_boot < 0:
        raise ValueError("bootstrap replicate count must be non-negative")
    users = sorted({str(row["user"]) for row in records})
    variants = sorted({str(row["variant"]) for row in records})
    user_index = {user: i for i, user in enumerate(users)}
    variant_index = {variant: i for i, variant in enumerate(variants)}
    # Per user x cell: singleton count/hits, collided count/hits.
    cell = np.zeros((len(users), len(variants), 4), dtype=float)
    ambiguity_sum = np.zeros(len(users), dtype=float)
    ambiguity_by_group = np.zeros((len(users), 2), dtype=float)
    ambiguity_group_count = np.zeros((len(users), 2), dtype=float)
    row_count = np.zeros(len(users), dtype=float)
    for row in records:
        ui = user_index[str(row["user"])]
        vi = variant_index[str(row["variant"])]
        group = 2 if bool(row["collided"]) else 0
        cell[ui, vi, group] += 1.0
        cell[ui, vi, group + 1] += float(row["sid_hit@10"])
        ambiguity_sum[ui] += (
            float(row["sid_hit@10"]) - float(row["item_uniform@10"]))
        ambiguity_group = int(bool(row["collided"]))
        ambiguity_by_group[ui, ambiguity_group] += (
            float(row["sid_hit@10"]) - float(row["item_uniform@10"]))
        ambiguity_group_count[ui, ambiguity_group] += 1.0
        row_count[ui] += 1.0

    totals = cell.sum(axis=0)
    if np.any(totals[:, 0] == 0) or np.any(totals[:, 2] == 0):
        raise ValueError("every configuration must contain singleton and collided targets")
    cell_rd = totals[:, 3] / totals[:, 2] - totals[:, 1] / totals[:, 0]
    estimates = {
        "within_config_collision_rd@10": float(cell_rd.mean()),
        "raw_collision_rd@10": float(
            totals[:, 3].sum() / totals[:, 2].sum()
            - totals[:, 1].sum() / totals[:, 0].sum()),
        "sid_minus_uniform_item@10": float(
            ambiguity_sum.sum() / row_count.sum()),
        "sid_minus_uniform_item_singleton_target@10": float(
            ambiguity_by_group[:, 0].sum()
            / ambiguity_group_count[:, 0].sum()),
        "sid_minus_uniform_item_collided_target@10": float(
            ambiguity_by_group[:, 1].sum()
            / ambiguity_group_count[:, 1].sum()),
    }

    samples: dict[str, np.ndarray] = {}
    if n_boot:
        rng = np.random.default_rng(seed)
        weights = rng.multinomial(
            len(users), np.full(len(users), 1.0 / len(users)), size=n_boot)
        boot = np.einsum("bu,uvg->bvg", weights, cell, optimize=True)
        valid = np.all((boot[:, :, 0] > 0) & (boot[:, :, 2] > 0), axis=1)
        if not np.any(valid):
            raise ValueError("no bootstrap draw retained both collision groups in every cell")
        boot = boot[valid]
        samples["within_config_collision_rd@10"] = (
            boot[:, :, 3] / boot[:, :, 2]
            - boot[:, :, 1] / boot[:, :, 0]).mean(axis=1)
        samples["raw_collision_rd@10"] = (
            boot[:, :, 3].sum(axis=1) / boot[:, :, 2].sum(axis=1)
            - boot[:, :, 1].sum(axis=1) / boot[:, :, 0].sum(axis=1))
        boot_ambiguity = weights @ ambiguity_sum
        boot_count = weights @ row_count
        samples["sid_minus_uniform_item@10"] = boot_ambiguity / boot_count
        boot_group_sum = weights @ ambiguity_by_group
        boot_group_count = weights @ ambiguity_group_count
        samples["sid_minus_uniform_item_singleton_target@10"] = (
            boot_group_sum[:, 0] / boot_group_count[:, 0])
        samples["sid_minus_uniform_item_collided_target@10"] = (
            boot_group_sum[:, 1] / boot_group_count[:, 1])

    out: dict[str, object] = {
        "n_users": len(users), "n_configurations": len(variants),
        "n_boot": n_boot, "seed": seed,
    }
    for key, estimate in estimates.items():
        values = samples.get(key)
        out[key] = {
            "estimate": estimate,
            "ci_low": (float(np.quantile(values, 0.025))
                       if values is not None else None),
            "ci_high": (float(np.quantile(values, 0.975))
                        if values is not None else None),
            "valid_bootstrap_replicates": (len(values) if values is not None else 0),
        }
    return out


def load_training_counts(path: Path, depth: int) -> tuple[Counter, Counter]:
    sid_counts: Counter = Counter()
    item_counts: Counter = Counter()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            sid_counts[parse_sid(row["item_sid"], depth)] += 1
            item_counts[int(row["item_id"])] += 1
    return sid_counts, item_counts


def analyze_file(pred_path: Path, category: str) -> tuple[
        dict, list[dict], list[dict], list[dict]]:
    match = FILE_RE.match(pred_path.name)
    if not match or match["category"] != category:
        raise ValueError(f"unexpected prediction filename: {pred_path.name}")
    q = match["quantizer"]
    depth, width = int(match["depth"]), int(match["width"])
    variant = f"{q}_{depth}codebook_{width}"

    idx_path = index_path(category, q, depth, width)
    item_to_sid, items_by_sid = load_index(idx_path, depth)
    test_path = split_path(category, "test", q, depth, width)
    train_path = split_path(category, "train", q, depth, width)
    with test_path.open(newline="") as fh:
        test_rows = list(csv.DictReader(fh))
    pred_rows = json.loads(pred_path.read_text())
    if len(test_rows) != len(pred_rows):
        raise ValueError(
            f"{variant}: {len(test_rows)} test rows != {len(pred_rows)} predictions")

    train_sid_count, train_item_count = load_training_counts(train_path, depth)
    sid_freq_edges = quantile_edges(list(train_sid_count.values()))
    item_freq_edges = quantile_edges(list(train_item_count.values()) or [0])
    legal = set(items_by_sid)
    empty_beam_slots = 0
    rows_with_empty_padding = 0
    records: list[dict] = []

    for row_number, (test, pred) in enumerate(zip(test_rows, pred_rows), start=2):
        target = parse_sid(test["item_sid"], depth)
        pred_target = parse_sid(pred["output"], depth)
        item = int(test["item_id"])
        if pred_target != target or item_to_sid.get(item) != target:
            raise ValueError(
                f"{variant} row {row_number}: split/prediction/index target mismatch")

        raw_predictions = pred.get("predict")
        if not isinstance(raw_predictions, list) or len(raw_predictions) != max(CUTOFFS):
            raise ValueError(
                f"{variant} row {row_number}: expected exactly {max(CUTOFFS)} beam slots")
        parsed_predictions: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        padding_started = False
        for slot, raw in enumerate(raw_predictions, start=1):
            if raw == "":
                padding_started = True
                empty_beam_slots += 1
                continue
            if padding_started:
                raise ValueError(
                    f"{variant} row {row_number}: non-empty beam after padding at slot {slot}")
            try:
                sid = parse_sid(raw, depth)
            except ValueError as exc:
                raise ValueError(
                    f"{variant} row {row_number}: malformed beam at slot {slot}: {exc}") from exc
            if sid not in legal:
                raise ValueError(
                    f"{variant} row {row_number}: beam at slot {slot} is not in the catalogue")
            if sid in seen:
                raise ValueError(
                    f"{variant} row {row_number}: duplicate non-empty beam at slot {slot}")
            seen.add(sid)
            parsed_predictions.append(sid)
        rows_with_empty_padding += int(padding_started)

        rank = exact_rank(parsed_predictions, target)
        bounds = item_rank_bounds(
            parsed_predictions, item, items_by_sid, target_sid=target)
        multiplicity = len(items_by_sid[target])
        try:
            hist_ids = json.loads(test["history_item_id"].replace("'", '"'))
        except json.JSONDecodeError:
            import ast
            hist_ids = ast.literal_eval(test["history_item_id"])
        rec = {
            "variant": variant,
            "quantizer": q,
            "depth": depth,
            "width": width,
            "user": test["user_id"],
            "item": item,
            "multiplicity": multiplicity,
            "collided": multiplicity > 1,
            "history_len": len(hist_ids),
            "train_sid_count": int(train_sid_count[target]),
            "train_item_count": int(train_item_count[item]),
            "sid_frequency_bin": quantile_bin(
                sid_freq_edges, int(train_sid_count[target])),
            "item_frequency_bin": quantile_bin(
                item_freq_edges, int(train_item_count[item])),
            "history_bin": min(len(hist_ids) // 5, 4),
            "sid_rank": rank,
            "item_rank": asdict(bounds),
        }
        for cutoff in CUTOFFS:
            rec[f"sid_hit@{cutoff}"] = float(rank is not None and rank <= cutoff)
            for key, value in bounds.hit_bounds(cutoff).items():
                rec[f"item_{key}@{cutoff}"] = value
        records.append(rec)

    config_rows: list[dict] = []
    bound_rows: list[dict] = []
    for cutoff in CUTOFFS:
        for collided in (False, True):
            group = [r for r in records if bool(r["collided"]) == collided]
            config_rows.append({
                "variant": variant, "quantizer": q, "depth": depth,
                "width": width, "cutoff": cutoff,
                "target_group": "collided" if collided else "singleton",
                "n_examples": len(group),
                "sid_hit_rate": mean(r[f"sid_hit@{cutoff}"] for r in group),
                "mean_multiplicity": mean(r["multiplicity"] for r in group),
                "mean_train_sid_count": mean(r["train_sid_count"] for r in group),
                "mean_train_item_count": mean(r["train_item_count"] for r in group),
            })
        single = [r for r in records if not r["collided"]]
        coll = [r for r in records if r["collided"]]
        config_rows.append({
            "variant": variant, "quantizer": q, "depth": depth,
            "width": width, "cutoff": cutoff, "target_group": "contrast",
            "n_examples": len(records),
            "sid_hit_rate": "",
            "collision_risk_difference": (
                mean(r[f"sid_hit@{cutoff}"] for r in coll)
                - mean(r[f"sid_hit@{cutoff}"] for r in single)),
        })
        for target_group, group in (
            ("all", records), ("singleton", single), ("collided", coll)):
            bound_rows.append({
                "variant": variant, "quantizer": q, "depth": depth,
                "width": width, "cutoff": cutoff,
                "target_group": target_group, "n_examples": len(group),
                "sid_hit_rate": mean(r[f"sid_hit@{cutoff}"] for r in group),
                "item_lower": mean(r[f"item_lower@{cutoff}"] for r in group),
                "item_uniform": mean(r[f"item_uniform@{cutoff}"] for r in group),
                "item_catalogue": mean(
                    r[f"item_catalogue@{cutoff}"] for r in group),
                "item_upper": mean(r[f"item_upper@{cutoff}"] for r in group),
            })

    strata: dict[tuple, dict[bool, list[float]]] = defaultdict(
        lambda: {False: [], True: []})
    for rec in records:
        key = (rec["sid_frequency_bin"], rec["item_frequency_bin"],
               rec["history_bin"])
        strata[key][bool(rec["collided"])].append(rec["sid_hit@10"])
    frequency_rows = []
    for key, groups in sorted(strata.items()):
        n0, n1 = len(groups[False]), len(groups[True])
        row = {
            "variant": variant, "quantizer": q, "depth": depth,
            "width": width, "sid_frequency_bin": key[0],
            "item_frequency_bin": key[1], "history_bin": key[2],
            "n_singleton": n0, "n_collided": n1,
            "has_collision_group_overlap": bool(n0 and n1),
            "singleton_hit@10": mean(groups[False]) if n0 else None,
            "collided_hit@10": mean(groups[True]) if n1 else None,
            "risk_difference": None,
            "overlap_weight": 0.0,
        }
        if n0 and n1:
            row["risk_difference"] = (
                float(row["collided_hit@10"]) - float(row["singleton_hit@10"]))
            row["overlap_weight"] = n0 * n1 / (n0 + n1)
        frequency_rows.append(row)

    metric_path = pred_path.with_name(pred_path.name.replace(
        ".predictions.json", ".json"))
    recorded = json.loads(metric_path.read_text()).get("metrics", {})
    validation = {}
    for cutoff in CUTOFFS:
        exact = mean(r[f"sid_hit@{cutoff}"] for r in records) * 100
        old = recorded.get(f"HR@{cutoff}")
        validation[f"HR@{cutoff}"] = {
            "exact_sid_percent": exact,
            "recorded_legacy_percent": old,
            "difference_pp": exact - old if old is not None else None,
        }

    meta = {
        "variant": variant, "quantizer": q, "depth": depth, "width": width,
        "n_examples": len(records), "n_users": len({r['user'] for r in records}),
        "n_catalogue_items": len(item_to_sid), "n_unique_sids": len(items_by_sid),
        "catalogue_collision_excess_rate": 1 - len(items_by_sid) / len(item_to_sid),
        "test_targets_in_collided_buckets": mean(r["collided"] for r in records),
        "max_bucket": max(map(len, items_by_sid.values())),
        "invalid_prediction_count": 0,
        "rows_with_empty_beam_padding": rows_with_empty_padding,
        "empty_beam_slots": empty_beam_slots,
        "row_identity_sha256": hashing.stable_json_hash([
            (row["user_id"], row["history_item_id"], row["item_id"])
            for row in test_rows
        ]),
        "prediction_sha256": hashing.sha256_file(pred_path),
        "test_sha256": hashing.sha256_file(test_path),
        "train_sha256": hashing.sha256_file(train_path),
        "index_sha256": hashing.sha256_file(idx_path),
        "legacy_metric_sha256": hashing.sha256_file(metric_path),
        "legacy_metric_validation": validation,
    }
    return meta, records, config_rows, bound_rows + frequency_rows


def weighted_frequency_contrast(rows: list[dict]) -> dict:
    strata = [r for r in rows if r.get("has_collision_group_overlap")]
    weight = sum(float(r["overlap_weight"]) for r in strata)
    estimate = (sum(float(r["risk_difference"]) * float(r["overlap_weight"])
                    for r in strata) / weight if weight else float("nan"))
    total_singleton = sum(int(r["n_singleton"]) for r in rows)
    total_collided = sum(int(r["n_collided"]) for r in rows)
    supported_singleton = sum(int(r["n_singleton"]) for r in strata)
    supported_collided = sum(int(r["n_collided"]) for r in strata)
    total = total_singleton + total_collided
    supported = supported_singleton + supported_collided
    return {
        "n_total_strata": len(rows), "n_overlap_strata": len(strata),
        "overlap_weight": weight, "adjusted_risk_difference@10": estimate,
        "support": {
            "n_rows": supported, "n_rows_total": total,
            "row_fraction": supported / total if total else None,
            "n_singleton": supported_singleton,
            "n_singleton_total": total_singleton,
            "singleton_fraction": (
                supported_singleton / total_singleton if total_singleton else None),
            "n_collided": supported_collided,
            "n_collided_total": total_collided,
            "collided_fraction": (
                supported_collided / total_collided if total_collided else None),
        },
    }


def render_report(result: dict) -> str:
    adjusted = result["pooled"]["within_config_collision_rd@10"]
    raw = result["pooled"]["raw_collision_rd@10"]
    amb = result["pooled"]["sid_minus_uniform_item@10"]
    amb_single = result["pooled"]["sid_minus_uniform_item_singleton_target@10"]
    amb_collided = result["pooled"]["sid_minus_uniform_item_collided_target@10"]
    freq = result["frequency_adjusted"]
    signs = result["configuration_signs@10"]
    def ci_text(stat: dict, scale: float = 100.0) -> str:
        if stat.get("ci_low") is None or stat.get("ci_high") is None:
            return "not computed"
        return f"{scale * stat['ci_low']:+.1f}, {scale * stat['ci_high']:+.1f}"

    return f"""# Retrospective Experiment 3: collision-conditioned SID accuracy and item-rank bounds

## Result

Across {result['n_configurations']} retained AR configurations and
{result['n_prediction_rows']:,} configuration-example rows, targets in a
collided SID bucket had a **{100 * adjusted['estimate']:+.1f} percentage-point**
higher exact-SID HR@10 than singleton
targets after computing the contrast within each cell and
equally averaging the 18 cells (user-cluster bootstrap 95% interval
[{ci_text(adjusted)}] percentage
points).  The unstandardized pooled contrast is
{100 * raw['estimate']:+.1f} percentage points; it is secondary because collision
prevalence varies by representation.  The within-cell direction was positive in
{signs['positive']}/{signs['nonzero']} configurations (two-sided exact sign
test diagnostic p={signs['p_value']:.4g}; the cells are not independent
replicates).

That is not item-identification evidence.  Expanding SID beams into their
one-to-many item buckets, the exact-SID score exceeds the uniform-within-bucket
item HR@10 by **{100 * amb['estimate']:.1f} percentage points** on average
(95% interval [{ci_text(amb)}]).  The
gap is {100 * amb_collided['estimate']:.1f} points for collided targets versus
{100 * amb_single['estimate']:.1f} for singleton targets.  The latter is not a
target collision: it comes from earlier predicted collision buckets consuming
item-rank positions.  The
full lower/uniform/stable-
order/upper bounds are in `item_bounds.csv`.

After coarsening within configuration by training SID frequency, training item
frequency, and history length, the overlap-weighted collided-minus-singleton
association is **{100 * freq['adjusted_risk_difference@10']:+.1f} percentage
points** across
{freq['n_overlap_strata']} overlap strata.  Those strata cover
{freq['support']['n_rows']:,}/{freq['support']['n_rows_total']:,}
({freq['support']['row_fraction']:.1%}) rows; it is an overlap-population
sensitivity, not a full-sample adjustment.  This analysis is descriptive:
collision multiplicity, pooled label frequency, item popularity, and quantizer
geometry were not randomized.

## Interpretation

The historical SID-level leaderboard does not penalize ambiguity: several
catalogue items can satisfy the same exact string, and their training examples
pool onto that string.  The observed advantage is consistent with that pooled-
label mechanism, but is not a randomized effect.  Collision-conditioned SID
accuracy therefore does not answer
whether a recommender recovered the intended item.  Item-level claims must
either use a model-provided within-bucket score or report bounds such as these.

## Validity limits

- These are one-seed endpoint predictions, not independent training replicates.
- The 18 configurations reuse the same users; uncertainty resamples users and
  keeps all of a user's rows/configurations together.
- The uniform tie-break is a no-information reference, and catalogue order is
  merely a reproducible sensitivity analysis.  Neither is a learned decoder.
- Prefix-constrained generation makes all retained beams legal, but a correct
  SID remains unable to distinguish members of its collision bucket.
- Upstream `HR` allowed rare cross-SID title/item-ID equivalences.  This
  experiment deliberately uses literal SID equality and records the difference
  from every legacy metric in `result.json`.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args(argv)

    metrics_root = paths.FROZEN_RESULTS / "sweep_metrics" / "next-item" / "metrics"
    pred_paths = sorted(metrics_root.glob(
        f"nextitem__{args.category}__*.predictions.json"))
    if not pred_paths:
        raise SystemExit(f"no retained predictions under {metrics_root}")

    metadata, records, config_rows, bound_rows, frequency_rows = [], [], [], [], []
    for pred_path in pred_paths:
        meta, recs, configs, mixed = analyze_file(pred_path, args.category)
        metadata.append(meta)
        records.extend(recs)
        config_rows.extend(configs)
        bound_rows.extend(r for r in mixed if "item_lower" in r)
        frequency_rows.extend(r for r in mixed if "overlap_weight" in r)
        print(
            f"{meta['variant']:27s} n={meta['n_examples']}  "
            f"target-collided={meta['test_targets_in_collided_buckets']:.1%}  "
            f"invalid={meta['invalid_prediction_count']}")

    # The retained grid is intentionally the complete balanced 3 x 3 x 2 grid.
    expected = {(q, d, w) for q in ("MQ", "rqkmeans", "rqvae")
                for d in (3, 4, 5) for w in (128, 512)}
    observed = {(m["quantizer"], m["depth"], m["width"]) for m in metadata}
    if observed != expected:
        raise ValueError(
            f"retained AR grid changed; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}")
    identity_hashes = {m["row_identity_sha256"] for m in metadata}
    if len(identity_hashes) != 1:
        raise ValueError("test row identities or ordering differ across configurations")

    pooled = bootstrap_user_contrasts(records, args.bootstrap, args.seed)
    contrasts10 = [float(r["collision_risk_difference"])
                   for r in config_rows
                   if r["target_group"] == "contrast" and r["cutoff"] == 10]
    pos = sum(v > 0 for v in contrasts10)
    neg = sum(v < 0 for v in contrasts10)
    sign = {"positive": pos, "negative": neg, "zero": len(contrasts10)-pos-neg,
            "nonzero": pos+neg, "p_value": exact_sign_test(pos, neg),
            "unit": "configuration cell; descriptive because cells are fixed, shared-user, one-seed conditions"}
    freq = weighted_frequency_contrast(frequency_rows)

    snapshot = (paths.MANIFESTS / "CURRENT").read_text().strip()
    result = {
        "experiment": "retrospective-exp3-collision-conditioned-accuracy",
        "snapshot": snapshot,
        "category": args.category,
        "n_configurations": len(metadata),
        "n_prediction_rows": len(records),
        "all_configurations_share_row_identity_and_order": True,
        "cutoffs": list(CUTOFFS),
        "configuration_signs@10": sign,
        "pooled": pooled,
        "frequency_adjusted": freq,
        "configurations": metadata,
        "method": {
            "primary_outcome": "literal full-SID HR@10",
            "collision_definition": "target full SID maps to more than one catalogue item",
            "item_bounds": (
                "Expand each unique legal predicted SID to every catalogue item. "
                "Lower/upper place the target last/first in its bucket; uniform "
                "averages all within-bucket positions; catalogue uses ascending item ID."),
            "uncertainty": (
                f"{args.bootstrap} percentile bootstrap replicates clustered by user; "
                "all examples and configurations for a sampled user stay together."),
            "primary_standardization": (
                "Compute collided-minus-singleton literal SID HR@10 inside each "
                "configuration, then equally average all 18 configuration contrasts."),
            "frequency_adjustment": (
                "Within-configuration coarsened strata: quintiles of training SID "
                "target count, training item target count, and 5-interaction history bins; "
                "overlap-stratum risk differences weighted by n0*n1/(n0+n1); "
                "support coverage is reported because non-overlap strata are excluded."),
        },
        "caveats": [
            "One training seed per configuration; cells are not stochastic replicates.",
            "Collision and popularity are observational and not randomized.",
            "No model-provided score exists within a collision bucket.",
            "Literal SID equality differs slightly from the legacy evaluator's title/item-ID fallback.",
        ],
    }

    out = (Path(args.out) if args.out else
           paths.DERIVED / "retrospective" / "exp3_collisions")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "configuration.csv", config_rows)
    write_csv(out / "item_bounds.csv", bound_rows)
    write_csv(out / "frequency_strata.csv", frequency_rows)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "report.md").write_text(render_report(result))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
