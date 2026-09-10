#!/usr/bin/env python
"""Aggregate a complete matched-beam array with paired user uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


GRID = tuple((quantizer, depth, width)
             for quantizer in ("MQ", "rqkmeans", "rqvae")
             for depth in (3, 4, 5) for width in (128, 512))
CUTOFFS = (1, 3, 5, 10)
ALL_METRICS = tuple(f"{metric}@{cutoff}"
                    for metric in ("sid_hit", "ndcg", "item_lower", "item_uniform", "item_upper")
                    for cutoff in CUTOFFS)
OUTCOMES = ("sid_hit@10", "ndcg@10", "item_uniform@10")
ORDERS = {3: [0, 2, 1], 4: [2, 3, 0, 1], 5: [2, 4, 3, 0, 1]}


class SummaryError(ValueError):
    """Inputs are incomplete or fail the experiment's comparison contract."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    def reject(value):
        raise SummaryError(f"nonfinite JSON value {value} in {path}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SummaryError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def _require(condition, message):
    if not condition:
        raise SummaryError(message)


def _verified_files(folder):
    """Check the two numerical inputs against the completed job's manifest."""
    manifest = folder / "output.sha256"
    _require(manifest.is_file(), f"missing output hash manifest: {folder}")
    entries = {}
    for line in manifest.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        _require(match is not None, f"malformed output hash manifest: {manifest}")
        recorded = Path(match[2])
        path = recorded if recorded.is_absolute() else folder / recorded
        path = path.resolve()
        _require(path.parent == folder.resolve(), f"hash manifest leaves cell directory: {path}")
        _require(path.name not in entries, f"duplicate hash target: {path}")
        entries[path.name] = match[1]
    hashes = {str(manifest.resolve()): sha256(manifest)}
    for name in ("result.json", "predictions.npz"):
        path = folder / name
        _require(path.is_file() and name in entries, f"missing hashed input {path}")
        digest = sha256(path)
        _require(digest == entries[name], f"output hash mismatch: {path}")
        hashes[str(path.resolve())] = digest
    hashes[str((folder / "status.txt").resolve())] = sha256(folder / "status.txt")
    return hashes


def _prediction_checks(data, *, n, d, width, conditions, reported, label):
    k = len(conditions)
    required = {
        "users", "user_ids", "targets", "target_item_ids", "target_multiplicity",
        "condition_beams", "condition_policies", "codes", "scores", "counts",
        "ranks", "metric_names", "metrics",
    }
    _require(required <= set(data), f"{label}: prediction fields missing")
    names = tuple(data["metric_names"].tolist())
    _require(len(names) == len(set(names)) and set(names) == set(ALL_METRICS),
             f"{label}: incomplete metric set")
    _require(data["metrics"].shape == (k, n, len(names)), f"{label}: metrics shape")
    _require(np.isfinite(data["metrics"]).all()
             and ((data["metrics"] >= 0) & (data["metrics"] <= 1)).all(),
             f"{label}: metrics outside finite [0,1]")
    for name in ("user_ids", "target_item_ids", "target_multiplicity", "counts", "ranks", "codes", "targets"):
        _require(np.issubdtype(data[name].dtype, np.integer), f"{label}: {name} must be integers")
    for name in ("users", "user_ids", "target_item_ids", "target_multiplicity"):
        _require(data[name].shape == (n,), f"{label}: {name} shape")
    _require(len(set(data["users"].tolist())) == n and len(set(data["user_ids"].tolist())) == n,
             f"{label}: duplicate user identity")
    _require((data["target_multiplicity"] >= 1).all(), f"{label}: invalid target multiplicity")
    _require(data["targets"].shape == (n, d)
             and ((data["targets"] >= 0) & (data["targets"] < width)).all(),
             f"{label}: invalid target SID")
    _require(data["codes"].shape == (k, n, 10, d)
             and data["scores"].shape == (k, n, 10)
             and data["counts"].shape == data["ranks"].shape == (k, n),
             f"{label}: prediction shapes")
    _require(((data["counts"] >= 0) & (data["counts"] <= 10)).all(),
             f"{label}: invalid count")
    visible = np.arange(10)[None, None, :] < data["counts"][:, :, None]
    _require(((data["codes"][visible] >= 0) & (data["codes"][visible] < width)).all()
             and (data["codes"][~visible] == -1).all(), f"{label}: invalid SID/padding")
    for index, condition in enumerate(conditions):
        if condition[1] == "legacy_confidence":
            _require(reported[condition].get("scores_available") is False
                     and np.isnan(data["scores"][index]).all(),
                     f"{label}: legacy score unavailability must be explicit")
        else:
            _require(reported[condition].get("scores_available", True) is True
                     and np.isfinite(data["scores"][index][visible[index]]).all()
                     and np.isneginf(data["scores"][index][~visible[index]]).all(),
                     f"{label}: score/padding mismatch")
    for first in range(10):
        for second in range(first + 1, 10):
            duplicate = (data["codes"][:, :, first] == data["codes"][:, :, second]).all(-1)
            _require(not (duplicate & visible[:, :, second]).any(), f"{label}: duplicate visible SID")
    matches = (data["codes"] == data["targets"][None, :, None, :]).all(-1) & visible
    rank = np.where(matches.any(-1), matches.argmax(-1) + 1, 0)
    _require(np.array_equal(rank, data["ranks"]), f"{label}: stored ranks disagree with predictions")
    for cutoff in CUTOFFS:
        hit = (rank > 0) & (rank <= cutoff)
        gain = np.where(hit, 1 / np.log2(np.maximum(rank, 1) + 1), 0)
        _require(np.array_equal(data["metrics"][:, :, names.index(f"sid_hit@{cutoff}")], hit),
                 f"{label}: SID hit metric disagrees with rank")
        _require(np.allclose(data["metrics"][:, :, names.index(f"ndcg@{cutoff}")], gain,
                             atol=1e-12, rtol=0), f"{label}: NDCG disagrees with rank")
        lower, uniform, upper = [data["metrics"][:, :, names.index(f"item_{bound}@{cutoff}")]
                                 for bound in ("lower", "uniform", "upper")]
        _require(((lower <= uniform) & (uniform <= upper) & (upper <= hit)).all(),
                 f"{label}: inconsistent item-expansion bounds")
    return names


def load_grid(input_dir: Path, *, expected_users=6297):
    """Load all 18 cells; ``expected_users`` is injectable only for small tests."""
    input_dir = Path(input_dir)
    expected = {f"cell-{index:02d}" for index in range(len(GRID))}
    actual = {p.name for p in input_dir.glob("cell-*") if p.is_dir()}
    _require(actual == expected,
             f"incomplete grid: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    cells, values, hashes = [], [], {}
    common_users = common_ids = common_targets = common_conditions = common_signature = None
    for index, (quantizer, d, width) in enumerate(GRID):
        folder = input_dir / f"cell-{index:02d}"
        status = folder / "status.txt"
        _require(status.is_file(), f"missing completion status: {folder}")
        text = status.read_text().strip()
        _require(text.startswith("complete ") and re.search(r"(?:^|\s)exit=0(?:\s|$)", text),
                 f"incomplete or failed cell: {folder}: {text}")
        hashes.update(_verified_files(folder))
        result = _json(folder / "result.json")
        ckpt = f"diff-next1-{quantizer}-{d}cb-{width}"
        _require((result["cell_index"], result["checkpoint"], result["quantizer"],
                  result["depth"], result["codebook_size"]) == (index, ckpt, quantizer, d, width),
                 f"{folder}: cell/checkpoint identity mismatch")
        _require(result.get("full_cohort") is True and result["n_users"] == expected_users,
                 f"{folder}: must contain the full {expected_users}-user cohort")
        _require(result["experiment"] == "controlled-exp1-matched-beam"
                 and result["fixed_order_seed"] == 42 and result["fixed_order"] == ORDERS[d],
                 f"{folder}: experiment/order mismatch")
        _require(result["model_shape_validation"]["ok"] is True,
                 f"{folder}: model shape validation did not pass")
        _require(re.fullmatch(r"[0-9a-f]{64}", result["cohort_sha256"]) is not None,
                 f"{folder}: missing cohort hash")
        signature = tuple(result[key] for key in ("cohort_sha256", "snapshot", "category", "precision"))
        with np.load(folder / "predictions.npz", allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        _require(data["condition_beams"].ndim == data["condition_policies"].ndim == 1
                 and len(data["condition_beams"]) == len(data["condition_policies"]),
                 f"{folder}: condition shape mismatch")
        conditions = list(zip(data["condition_beams"].tolist(), data["condition_policies"].tolist()))
        _require(len(set(conditions)) == len(conditions), f"{folder}: duplicate condition")
        beams = sorted({beam for beam, _ in conditions})
        policies = sorted({policy for _, policy in conditions})
        wanted = [(beam, policy) for beam in beams for policy in policies]
        _require(beams in ([64], [64, 256])
                 and policies in (["confidence", "fixed"], ["confidence", "fixed", "legacy_confidence"])
                 and set(conditions) == set(wanted),
                 f"{folder}: incomplete or unsupported condition set")
        reported = {(condition["beam_width"], condition["policy"]): condition
                    for condition in result["conditions"]}
        _require(len(reported) == len(result["conditions"]) and set(reported) == set(conditions),
                 f"{folder}: reported condition set mismatch")
        names = _prediction_checks(data, n=expected_users, d=d, width=width,
                                   conditions=conditions, reported=reported, label=str(folder))
        if common_signature is None:
            common_signature, common_conditions = signature, wanted
            common_users, common_ids, common_targets = (
                data["users"], data["user_ids"], data["target_item_ids"])
        else:
            _require(signature == common_signature, f"{folder}: cohort/snapshot/precision mismatch")
            _require(wanted == common_conditions, f"{folder}: condition set differs across cells")
            for name, reference in (("users", common_users), ("user_ids", common_ids),
                                    ("target_item_ids", common_targets)):
                _require(np.array_equal(data[name], reference), f"{folder}: {name} row identity mismatch")
        for position, condition in enumerate(conditions):
            for metric_position, name in enumerate(names):
                actual_mean = data["metrics"][position, :, metric_position].mean()
                _require(abs(actual_mean - reported[condition]["metrics"][name]) <= 1e-12,
                         f"{folder}: reported mean differs from per-user metrics: {condition}/{name}")
        anchors = [("historical_fixed64_reconstruction", (64, "fixed"), "fixed64")]
        if (256, "legacy_confidence") in conditions:
            anchors.append(("historical_confidence256_reconstruction", (256, "legacy_confidence"),
                            "confidence256"))
        for key, condition, anchor_name in anchors:
            _require(key in result, f"{folder}: missing {key}")
            for outcome in ("sid_hit@10", "ndcg@10"):
                record = result[key][outcome]
                actual_mean = data["metrics"][conditions.index(condition), :, names.index(outcome)].mean()
                _require(record["comparable_full_cohort"] is True
                         and abs(actual_mean - record["new"]) <= 1e-12
                         and abs(record["delta"] - (actual_mean - record["recorded"])) <= 1e-12,
                         f"{folder}: inconsistent historical reconstruction report")
                _require(abs(actual_mean - record["recorded"]) <= 1 / expected_users + 1e-7,
                         f"{folder}: historical {anchor_name} reconstruction failed for {outcome}: "
                         f"{actual_mean-record['recorded']:+.8f}; no cell exclusion is permitted")
        positions = [conditions.index(condition) for condition in wanted]
        selected = data["metrics"][positions][:, :, [names.index(name) for name in OUTCOMES]]
        values.append(selected)
        cells.append(result)
    return {
        "cells": cells, "values": np.stack(values), "conditions": common_conditions,
        "metric_names": OUTCOMES, "users": common_users, "user_ids": common_ids,
        "input_sha256": hashes, "input_directory": str(input_dir.resolve()),
    }


def bootstrap_user_means(values, *, n_boot=2000, seed=20260909):
    """Resample users, keeping every cell, policy, beam and outcome together.

    Input is [cell, condition, user, metric]; output is bootstrap means with
    shape [replicate, cell, condition, metric]. No configuration resampling.
    """
    _require(values.ndim == 4 and values.shape[2] > 0 and n_boot >= 1,
             "bootstrap needs [cell,condition,user,metric] and positive replicates")
    n = values.shape[2]
    user_matrix = values.transpose(2, 0, 1, 3).reshape(n, -1)
    output = np.empty((n_boot, user_matrix.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, 64):
        count = min(64, n_boot-start)
        # Multinomial counts implement sampling n users with replacement.
        # One draw supplies weights for every column of the user matrix.
        weights = rng.multinomial(n, np.full(n, 1 / n), size=count)
        output[start:start + count] = weights @ user_matrix / n
    return output.reshape(n_boot, values.shape[0], values.shape[1], values.shape[3])


def _effect(cell_estimates, boot_cell_estimates):
    estimate = float(np.mean(cell_estimates))
    low, high = np.quantile(boot_cell_estimates.mean(axis=1), [0.025, 0.975])
    return {
        "difference": estimate, "ci95": [float(low), float(high)],
        "n_cells": len(cell_estimates), "positive_cells": int((cell_estimates > 0).sum()),
        "negative_cells": int((cell_estimates < 0).sum()),
        "tied_cells": int((cell_estimates == 0).sum()),
    }


def summarize(grid, *, n_boot=2000, seed=20260909):
    values, cells = grid["values"], grid["cells"]
    conditions, names = grid["conditions"], grid["metric_names"]
    means = values.mean(axis=2)
    boot = bootstrap_user_means(values, n_boot=n_boot, seed=seed)
    comparisons, configuration, changes, interactions = [], [], [], []
    legacy_comparisons, terminal_effects = [], []
    quantizers = sorted({cell["quantizer"] for cell in cells})
    beams = sorted({beam for beam, _ in conditions})
    for beam in beams:
        confidence, fixed = (conditions.index((beam, policy)) for policy in ("confidence", "fixed"))
        for metric_index, metric in enumerate(names):
            delta = means[:, confidence, metric_index] - means[:, fixed, metric_index]
            boot_delta = boot[:, :, confidence, metric_index] - boot[:, :, fixed, metric_index]
            item = {"beam_width": beam, "metric": metric,
                    "role": ("primary" if beam == 64 and metric == "sid_hit@10" else
                             "secondary" if beam == 64 and metric == "ndcg@10" else "sensitivity"),
                    "confidence_mean": float(means[:, confidence, metric_index].mean()),
                    "fixed_mean": float(means[:, fixed, metric_index].mean()),
                    **_effect(delta, boot_delta), "by_quantizer": {}}
            for quantizer in quantizers:
                subset = [i for i, cell in enumerate(cells) if cell["quantizer"] == quantizer]
                item["by_quantizer"][quantizer] = {
                    "confidence_mean": float(means[subset, confidence, metric_index].mean()),
                    "fixed_mean": float(means[subset, fixed, metric_index].mean()),
                    **_effect(delta[subset], boot_delta[:, subset]),
                }
            comparisons.append(item)
            for index, cell in enumerate(cells):
                confidence_info = next(record for record in cell["conditions"]
                                       if (record["beam_width"], record["policy"]) == (beam, "confidence"))
                fixed_info = next(record for record in cell["conditions"]
                                  if (record["beam_width"], record["policy"]) == (beam, "fixed"))
                effect = _effect(delta[index:index + 1], boot_delta[:, index:index + 1])
                configuration.append({
                    "checkpoint": cell["checkpoint"], "quantizer": cell["quantizer"],
                    "depth": cell["depth"], "codebook_size": cell["codebook_size"],
                    "beam_width": beam, "metric": metric,
                    "confidence_mean": float(means[index, confidence, metric_index]),
                    "fixed_mean": float(means[index, fixed, metric_index]),
                    "difference": effect["difference"], "ci_low": effect["ci95"][0],
                    "ci_high": effect["ci95"][1],
                    "confidence_seconds": confidence_info["seconds"],
                    "fixed_seconds": fixed_info["seconds"],
                })
    if (64, "legacy_confidence") in conditions:
        for beam in beams:
            shared, fixed, legacy = [conditions.index((beam, policy))
                                     for policy in ("confidence", "fixed", "legacy_confidence")]
            for metric_index, metric in enumerate(names):
                for left, right, destination, contrast in (
                    (legacy, fixed, legacy_comparisons, "legacy_confidence_minus_fixed"),
                    (shared, legacy, terminal_effects, "shared_confidence_minus_legacy_confidence"),
                ):
                    delta = means[:, left, metric_index] - means[:, right, metric_index]
                    boot_delta = boot[:, :, left, metric_index] - boot[:, :, right, metric_index]
                    item = {
                        "beam_width": beam, "metric": metric, "role": "diagnostic",
                        "contrast": contrast, "left_policy": conditions[left][1],
                        "right_policy": conditions[right][1],
                        "left_mean": float(means[:, left, metric_index].mean()),
                        "right_mean": float(means[:, right, metric_index].mean()),
                        **_effect(delta, boot_delta), "by_quantizer": {}, "configuration": [],
                    }
                    for quantizer in quantizers:
                        subset = [i for i, cell in enumerate(cells) if cell["quantizer"] == quantizer]
                        item["by_quantizer"][quantizer] = _effect(delta[subset], boot_delta[:, subset])
                    for index, cell in enumerate(cells):
                        item["configuration"].append({
                            "checkpoint": cell["checkpoint"],
                            "left_mean": float(means[index, left, metric_index]),
                            "right_mean": float(means[index, right, metric_index]),
                            **_effect(delta[index:index + 1], boot_delta[:, index:index + 1]),
                        })
                    destination.append(item)
    if beams == [64, 256]:
        for metric_index, metric in enumerate(names):
            for policy in sorted({policy for _, policy in conditions}):
                lower, upper = (conditions.index((beam, policy)) for beam in (64, 256))
                changes.append({
                    "policy": policy, "metric": metric,
                    "beam64_mean": float(means[:, lower, metric_index].mean()),
                    "beam256_mean": float(means[:, upper, metric_index].mean()),
                    **_effect(means[:, upper, metric_index] - means[:, lower, metric_index],
                              boot[:, :, upper, metric_index] - boot[:, :, lower, metric_index]),
                })
            c64, f64, c256, f256 = [conditions.index(key) for key in
                                  ((64, "confidence"), (64, "fixed"), (256, "confidence"), (256, "fixed"))]
            interactions.append({
                "metric": metric,
                "estimand": "(confidence-fixed) at beam256 minus (confidence-fixed) at beam64",
                **_effect((means[:, c256, metric_index] - means[:, f256, metric_index])
                          - (means[:, c64, metric_index] - means[:, f64, metric_index]),
                          (boot[:, :, c256, metric_index] - boot[:, :, f256, metric_index])
                          - (boot[:, :, c64, metric_index] - boot[:, :, f64, metric_index])),
            })
    reconstruction = [abs(record["delta"]) for cell in cells
                      for record in cell["historical_fixed64_reconstruction"].values()]
    confidence_reconstruction = [abs(record["delta"]) for cell in cells
                                 for record in cell.get("historical_confidence256_reconstruction", {}).values()]
    return {
        "experiment": "controlled-exp1-matched-beam-summary",
        "n_users": values.shape[2], "n_cells": len(cells), "beam_widths": beams,
        "cohort_sha256": cells[0]["cohort_sha256"], "snapshot": cells[0]["snapshot"],
        "category": cells[0]["category"], "input_directory": grid["input_directory"],
        "input_sha256": grid["input_sha256"],
        "bootstrap": {"replicates": n_boot, "seed": seed, "interval": "percentile 95%",
                      "unit": "user", "shared_draws_across_all_cells_and_conditions": True,
                      "configuration_resampling": False},
        "historical_fixed64_reconstruction": {
            "status": "passed", "comparisons": len(reconstruction),
            "maximum_absolute_difference": max(reconstruction),
            "tolerance": 1 / values.shape[2] + 1e-7,
        },
        "historical_confidence256_reconstruction": ({
            "status": "passed", "comparisons": len(confidence_reconstruction),
            "maximum_absolute_difference": max(confidence_reconstruction),
            "tolerance": 1 / values.shape[2] + 1e-7,
        } if confidence_reconstruction else None),
        "comparisons": comparisons, "configuration": configuration,
        "legacy_confidence_comparisons": legacy_comparisons,
        "shared_vs_legacy_confidence": terminal_effects,
        "beam_width_effects": changes, "beam_interactions": interactions,
        "runtime_seconds": {
            f"{beam}/{policy}": sum(next(record["seconds"] for record in cell["conditions"]
                                        if (record["beam_width"], record["policy"]) == (beam, policy))
                                   for cell in cells)
            for beam, policy in conditions},
        "claim_boundary": (
            "Matched reveal-policy comparison conditional on these confidence-selected checkpoints "
            "and one fixed seed42 order per SID depth. Both branches use full final-digit expansion. "
            "Equal beam caps do not imply equal compute. User intervals do not quantify training-seed, "
            "checkpoint-selection, random-order, or dataset uncertainty. No independent-cell p-values."),
    }


def _scaled(value):
    return f"{100 * value:.3f}"


def render_report(result):
    primary = next(row for row in result["comparisons"] if row["role"] == "primary")
    interval = primary["ci95"]
    direction = ("positive" if interval[0] > 0 else "negative" if interval[1] < 0
                 else "compatible with either sign under user resampling")
    lines = [
        "# Experiment 1: matched-beam decoding rerun", "",
        f"For the specified shared-search primary comparison at beam 64, confidence-guided decoding "
        f"has SID hit@10 {_scaled(primary['confidence_mean'])}% "
        f"versus {_scaled(primary['fixed_mean'])}% for the fixed seed-42 order: "
        f"a difference of {100 * primary['difference']:+.3f} percentage points "
        f"(paired user bootstrap 95% interval [{100 * interval[0]:+.3f}, {100 * interval[1]:+.3f}]). "
        f"The interval is {direction}. The configuration difference is positive in "
        f"{primary['positive_cells']}/{primary['n_cells']} cells, negative in "
        f"{primary['negative_cells']}, and tied in {primary['tied_cells']}.", "",
        f"The comparison uses the same {result['n_users']:,} users and all {result['n_cells']} frozen "
        "checkpoints. Both policies use identical beam caps, full token expansion on every digit, "
        "projected cross-attention, final legality filtering, and maximum-score SID deduplication. "
        "Only the permitted next position changes: any remaining position versus the fixed order.", "",
        "## Retrieval comparison", "",
        "Values and intervals below are multiplied by 100; hit-rate differences are percentage points. "
        "NDCG is shown on the same 0–100 scale. Configuration means have equal weight.", "",
        "| Beam | Outcome | Confidence | Fixed | Difference | 95% interval | Positive cells |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    if result["legacy_confidence_comparisons"]:
        direct = next(row for row in result["legacy_confidence_comparisons"]
                      if row["beam_width"] == 64 and row["metric"] == "sid_hit@10")
        lines[2:2] = [
            "The direct equal-beam rerun of the original confidence decoder gives "
            f"SID hit@10 {_scaled(direct['left_mean'])}% versus {_scaled(direct['right_mean'])}% "
            f"for fixed order at beam 64: {100 * direct['difference']:+.3f} percentage points "
            f"(paired user bootstrap 95% interval [{100 * direct['ci95'][0]:+.3f}, "
            f"{100 * direct['ci95'][1]:+.3f}]). This preserves the original confidence algorithm, "
            "including its greedy final digit. The separate shared-search comparison below also "
            "standardizes final-digit expansion and remains the specified primary analysis.", "",
        ]
    labels = {"sid_hit@10": "SID hit@10", "ndcg@10": "NDCG@10", "item_uniform@10": "Uniform item hit@10"}
    for row in result["comparisons"]:
        lines.append(f"| {row['beam_width']} | {labels[row['metric']]} | {_scaled(row['confidence_mean'])} "
                     f"| {_scaled(row['fixed_mean'])} | {100 * row['difference']:+.3f} "
                     f"| [{100 * row['ci95'][0]:+.3f}, {100 * row['ci95'][1]:+.3f}] "
                     f"| {row['positive_cells']}/{row['n_cells']} |")
    lines.extend(["", "SID hit@10 is the primary beam-64 outcome; NDCG@10 is secondary. "
                  "The item outcome expands each predicted SID into its item bucket and assumes a "
                  "uniform ordering within a bucket. It is a sensitivity analysis, not an observed "
                  "model preference between items sharing a SID."])
    if result["legacy_confidence_comparisons"]:
        lines.extend(["", "## Original confidence decoder at equal beams", "",
                      "This diagnostic directly reruns the archived confidence algorithm with the "
                      "same beam cap as fixed order. It retains confidence's historical greedy last "
                      "digit; the shared-search comparison above instead expands the last digit in "
                      "both policies. The primary comparison was specified before these outcomes.", "",
                      "| Beam | Outcome | Original confidence | Fixed | Difference (×100) | 95% interval |",
                      "| --- | --- | ---: | ---: | ---: | --- |"])
        for row in result["legacy_confidence_comparisons"]:
            lines.append(f"| {row['beam_width']} | {labels[row['metric']]} | {_scaled(row['left_mean'])} "
                         f"| {_scaled(row['right_mean'])} | {100 * row['difference']:+.3f} "
                         f"| [{100 * row['ci95'][0]:+.3f}, {100 * row['ci95'][1]:+.3f}] |")
        lines.extend(["", "The additional effect of shared full-final expansion relative to original "
                      "confidence (including deterministic tie handling) is:", "",
                      "| Beam | Outcome | Shared minus original confidence (×100) | 95% interval |",
                      "| --- | --- | ---: | --- |"])
        for row in result["shared_vs_legacy_confidence"]:
            lines.append(f"| {row['beam_width']} | {labels[row['metric']]} "
                         f"| {100 * row['difference']:+.3f} "
                         f"| [{100 * row['ci95'][0]:+.3f}, {100 * row['ci95'][1]:+.3f}] |")
    lines.extend(["", "## Configuration variation", "",
                  "| Beam | Quantizer | SID hit@10 difference (pp) | 95% interval | Positive / negative / tied |",
                  "| --- | --- | ---: | --- | --- |"])
    for row in result["comparisons"]:
        if row["metric"] != "sid_hit@10":
            continue
        for quantizer, group in row["by_quantizer"].items():
            lines.append(f"| {row['beam_width']} | {quantizer} | {100 * group['difference']:+.3f} "
                         f"| [{100 * group['ci95'][0]:+.3f}, {100 * group['ci95'][1]:+.3f}] "
                         f"| {group['positive_cells']} / {group['negative_cells']} / {group['tied_cells']} |")
    if result["beam_width_effects"]:
        lines.extend(["", "## Beam-width sensitivity", "",
                      "These paired changes increase the cap from 64 to 256 on the same users and "
                      "checkpoints. The interaction is the change in the confidence-minus-fixed gap.", "",
                      "| Outcome | Contrast | Change (×100) | 95% interval |",
                      "| --- | --- | ---: | --- |"])
        for row in result["beam_width_effects"] + result["beam_interactions"]:
            label = row.get("policy", "interaction")
            lines.append(f"| {labels[row['metric']]} | {label} | {100 * row['difference']:+.3f} "
                         f"| [{100 * row['ci95'][0]:+.3f}, {100 * row['ci95'][1]:+.3f}] |")
    lines.extend(["", "## Interpretation and checks", "",
                  "The results describe a decoding intervention on these fixed checkpoints and this "
                  "one fixed order per SID length. Checkpoints were originally selected by confidence "
                  "validation performance; no new selection or training was performed. We therefore "
                  "cannot generalize the result to other checkpoint selection rules, training seeds, "
                  "or arbitrary fixed orders.", "",
                  "The historical confidence decoder greedily completed its last digit. This rerun "
                  "fully expands that digit in both policies, so changes from the earlier +0.34-point "
                  "comparison in the primary shared-search result are not attributable solely to beam width. "
                  "The original-confidence diagnostic preserves the historical final step. Equal beam caps also do not "
                  "mean equal candidate counts or runtime.", "",
                  f"All 18 cells passed identity, prediction, metric, and saved-file hash checks. "
                  f"The historical fixed64 reconstruction passed {result['historical_fixed64_reconstruction']['comparisons']} "
                  f"metric comparisons within {result['historical_fixed64_reconstruction']['tolerance']:.8f} raw units "
                  "(one hit-equivalent plus numerical tolerance). No cells were excluded.", "",
                  f"Intervals use {result['bootstrap']['replicates']:,} shared bootstrap draws of users; "
                  "each sampled user carries all of their configurations and decoding conditions. "
                  "The 18 configurations are not treated as independent training replicates. "
                  "Secondary and sensitivity intervals are descriptive and not multiplicity-adjusted.", "",
                  "Recorded decoding-and-scoring wall time across all cells:", ""])
    for condition, seconds in result["runtime_seconds"].items():
        lines.append(f"- {condition}: {seconds / 60:.2f} minutes.")
    if result["historical_confidence256_reconstruction"] is not None:
        guard = result["historical_confidence256_reconstruction"]
        lines.extend(["", f"Original confidence at beam256 also reproduced its {guard['comparisons']} "
                      f"archived metrics within {guard['tolerance']:.8f} raw units. Original confidence "
                      "does not expose path scores; its saved scores are explicitly unavailable (NaN), "
                      "while predictions and retrieval metrics are retained."])
    lines.extend(["", "Per-configuration values are in `configuration.csv`; numerical results, "
                  "input hashes, and bootstrap details are in `result.json`.", ""])
    return "\n".join(lines)


def write_outputs(result, out):
    out = Path(out)
    _require(not out.exists() or not any(out.iterdir()), f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(render_report(result))
    with (out / "configuration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["configuration"][0]))
        writer.writeheader()
        writer.writerows(result["configuration"])
    (out / "environment.json").write_text(json.dumps({
        "numpy_version": np.__version__, "analysis_file": str(Path(__file__).resolve()),
        "analysis_sha256": sha256(__file__),
    }, indent=2) + "\n")
    (out / "output.sha256").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in sorted(out.iterdir()) if path.is_file()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    _require(not args.out.resolve().is_relative_to(args.input.resolve()),
             "summary output must be separate from the input array directory")
    result = summarize(load_grid(args.input), n_boot=args.bootstrap, seed=args.seed)
    write_outputs(result, args.out)
    print(f"Validated 18 full cells; wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
