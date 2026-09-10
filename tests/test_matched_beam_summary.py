"""Identity and paired-user inference checks for the controlled rerun."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "experiments/controlled/exp1_matched_beam/summarize.py"
SPEC = importlib.util.spec_from_file_location("matched_beam_summary", SOURCE)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def _rehash(folder):
    (folder / "output.sha256").write_text("".join(
        f"{SUMMARY.sha256(folder / name)}  {name}\n"
        for name in ("result.json", "predictions.npz")))


def _write_fixture(root, *, beams=(64, 256), n=4, legacy=False):
    policies = ("confidence", "fixed", "legacy_confidence") if legacy else ("confidence", "fixed")
    conditions = [(beam, policy) for beam in beams for policy in policies]
    for index, (quantizer, depth, width) in enumerate(SUMMARY.GRID):
        folder = root / f"cell-{index:02d}"
        folder.mkdir(parents=True)
        targets = np.zeros((n, depth), dtype=np.int64)
        codes = np.full((len(conditions), n, 10, depth), -1, dtype=np.int64)
        scores = np.full((len(conditions), n, 10), -np.inf)
        scores[:, :, 0] = -0.5
        for ci, (_, policy) in enumerate(conditions):
            if policy == "legacy_confidence":
                scores[ci] = np.nan
        ranks = np.zeros((len(conditions), n), dtype=np.int64)
        metrics = np.empty((len(conditions), n, len(SUMMARY.ALL_METRICS)))
        for condition_index in range(len(conditions)):
            hit = (np.arange(n) + condition_index + index) % 3 != 0
            codes[condition_index, :, 0] = 0
            codes[condition_index, ~hit, 0, 0] = 7
            ranks[condition_index] = hit.astype(np.int64)
            metrics[condition_index] = hit[:, None]
        fixed = conditions.index((64, "fixed"))
        historical = {}
        for outcome in ("sid_hit@10", "ndcg@10"):
            mean = float(metrics[fixed, :, SUMMARY.ALL_METRICS.index(outcome)].mean())
            historical[outcome] = {"new": mean, "recorded": mean, "delta": 0,
                                   "comparable_full_cohort": True, "within_one_hit_equivalent": True}
        result = {
            "experiment": "controlled-exp1-matched-beam",
            "cell_index": index, "checkpoint": f"diff-next1-{quantizer}-{depth}cb-{width}",
            "quantizer": quantizer, "depth": depth, "codebook_size": width,
            "full_cohort": True, "n_users": n, "fixed_order_seed": 42,
            "fixed_order": SUMMARY.ORDERS[depth], "model_shape_validation": {"ok": True},
            "cohort_sha256": "a" * 64, "snapshot": "fixture", "category": "fixture",
            "precision": "float32", "historical_fixed64_reconstruction": historical,
            "conditions": [{
                "beam_width": beam, "policy": policy, "seconds": 1.0,
                "scores_available": policy != "legacy_confidence",
                "metrics": {name: float(metrics[ci, :, mi].mean())
                            for mi, name in enumerate(SUMMARY.ALL_METRICS)},
            } for ci, (beam, policy) in enumerate(conditions)],
        }
        if (256, "legacy_confidence") in conditions:
            ci = conditions.index((256, "legacy_confidence"))
            result["historical_confidence256_reconstruction"] = {
                outcome: {"new": float(metrics[ci, :, SUMMARY.ALL_METRICS.index(outcome)].mean()),
                          "recorded": float(metrics[ci, :, SUMMARY.ALL_METRICS.index(outcome)].mean()),
                          "delta": 0, "comparable_full_cohort": True, "within_one_hit_equivalent": True}
                for outcome in ("sid_hit@10", "ndcg@10")
            }
        np.savez_compressed(
            folder / "predictions.npz", users=np.asarray([f"user-{i}" for i in range(n)]),
            user_ids=np.arange(n), targets=targets, target_item_ids=np.arange(n),
            target_multiplicity=np.ones(n, dtype=np.int64),
            condition_beams=np.asarray([beam for beam, _ in conditions]),
            condition_policies=np.asarray([policy for _, policy in conditions]),
            codes=codes, scores=scores, counts=np.ones((len(conditions), n), dtype=np.int64),
            ranks=ranks, metric_names=np.asarray(SUMMARY.ALL_METRICS), metrics=metrics,
        )
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "status.txt").write_text(f"complete job=1 task={index} exit=0 finished=fixture\n")
        _rehash(folder)


def _mutate_json(folder, transform):
    path = folder / "result.json"
    result = json.loads(path.read_text())
    transform(result)
    path.write_text(json.dumps(result))
    _rehash(folder)


def _mutate_npz(folder, transform):
    path = folder / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    transform(data)
    np.savez_compressed(path, **data)
    _rehash(folder)


def test_shared_bootstrap_keeps_correlated_cells_and_conditions_together():
    # Each user's gain repeats in every cell: copying a cell must not shrink
    # uncertainty, and changing neither policy must produce exact zero.
    one = np.array([[[[1.0], [0.0]], [[0.0], [1.0]]]])
    repeated = np.repeat(one, 18, axis=0)
    original = SUMMARY.bootstrap_user_means(one, n_boot=300, seed=9)
    copied = SUMMARY.bootstrap_user_means(repeated, n_boot=300, seed=9)
    original_effect = (original[:, :, 0, 0] - original[:, :, 1, 0]).mean(1)
    copied_effect = (copied[:, :, 0, 0] - copied[:, :, 1, 0]).mean(1)
    assert np.array_equal(original_effect, copied_effect)
    assert set(original_effect) == {-1, 0, 1}
    same_policies = repeated.copy()
    same_policies[:, 1] = same_policies[:, 0]
    same = SUMMARY.bootstrap_user_means(same_policies, n_boot=300, seed=9)
    assert np.array_equal(same[:, :, 0], same[:, :, 1])


def test_complete_grid_uses_equal_cell_means_and_paired_beam_interaction(tmp_path):
    source = tmp_path / "array"
    _write_fixture(source)
    grid = SUMMARY.load_grid(source, expected_users=4)
    result = SUMMARY.summarize(grid, n_boot=100, seed=11)
    assert result["n_cells"] == 18 and result["n_users"] == 4
    assert result["historical_fixed64_reconstruction"]["status"] == "passed"
    primary = next(row for row in result["comparisons"] if row["role"] == "primary")
    values = grid["values"]
    expected = (values[:, 0, :, 0] - values[:, 1, :, 0]).mean()
    assert primary["difference"] == pytest.approx(expected)
    assert primary["confidence_mean"] == pytest.approx(values[:, 0, :, 0].mean())
    assert primary["positive_cells"] + primary["negative_cells"] + primary["tied_cells"] == 18
    interaction = next(row for row in result["beam_interactions"] if row["metric"] == "sid_hit@10")
    expected_interaction = ((values[:, 2, :, 0] - values[:, 3, :, 0])
                            - (values[:, 0, :, 0] - values[:, 1, :, 0])).mean()
    assert interaction["difference"] == pytest.approx(expected_interaction)
    assert len(primary["by_quantizer"]) == 3
    assert len(result["configuration"]) == 18 * 2 * 3
    output = tmp_path / "summary"
    SUMMARY.write_outputs(result, output)
    assert json.loads((output / "result.json").read_text())["n_cells"] == 18
    assert "one fixed order" in (output / "report.md").read_text()
    for line in (output / "output.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert SUMMARY.sha256(output / name) == digest
    with pytest.raises(SUMMARY.SummaryError, match="overwrite"):
        SUMMARY.write_outputs(result, output)


def test_beam64_only_is_valid_without_inventing_sensitivity(tmp_path):
    _write_fixture(tmp_path, beams=(64,))
    result = SUMMARY.summarize(SUMMARY.load_grid(tmp_path, expected_users=4), n_boot=10)
    assert result["beam_widths"] == [64]
    assert result["beam_width_effects"] == result["beam_interactions"] == []


def test_missing_or_failed_cells_are_not_silently_excluded(tmp_path):
    (tmp_path / "cell-00").mkdir()
    with pytest.raises(SUMMARY.SummaryError, match="incomplete grid"):
        SUMMARY.load_grid(tmp_path)
    source = tmp_path / "full"
    _write_fixture(source)
    (source / "cell-06/status.txt").write_text("failed job=1 task=6 exit=1\n")
    with pytest.raises(SUMMARY.SummaryError, match="incomplete or failed"):
        SUMMARY.load_grid(source, expected_users=4)


def test_partial_cohort_is_rejected_by_production_default(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(SUMMARY.SummaryError, match="6297-user"):
        SUMMARY.load_grid(tmp_path)


@pytest.mark.parametrize("identity", ["users", "user_ids", "target_item_ids"])
def test_cross_cell_identity_order_mismatch_is_rejected(tmp_path, identity):
    _write_fixture(tmp_path)
    _mutate_npz(tmp_path / "cell-01", lambda data: data.__setitem__(identity, data[identity][::-1]))
    with pytest.raises(SUMMARY.SummaryError, match="row identity mismatch"):
        SUMMARY.load_grid(tmp_path, expected_users=4)


def test_cohort_hash_and_condition_mismatches_are_rejected(tmp_path):
    _write_fixture(tmp_path)
    _mutate_json(tmp_path / "cell-01", lambda result: result.__setitem__("cohort_sha256", "b" * 64))
    with pytest.raises(SUMMARY.SummaryError, match="cohort/snapshot/precision mismatch"):
        SUMMARY.load_grid(tmp_path, expected_users=4)
    _mutate_json(tmp_path / "cell-01", lambda result: result.__setitem__("cohort_sha256", "a" * 64))
    _mutate_npz(tmp_path / "cell-01", lambda data: data["condition_beams"].__setitem__(2, 128))
    with pytest.raises(SUMMARY.SummaryError, match="condition set"):
        SUMMARY.load_grid(tmp_path, expected_users=4)


def test_historical_reconstruction_guard_uses_actual_user_mean(tmp_path):
    _write_fixture(tmp_path)

    def change(result):
        record = result["historical_fixed64_reconstruction"]["sid_hit@10"]
        record["recorded"] += 0.75
        record["delta"] = record["new"] - record["recorded"]
        # A forged precomputed pass marker must not override recomputation.
        record["within_one_hit_equivalent"] = True

    _mutate_json(tmp_path / "cell-01", change)
    with pytest.raises(SUMMARY.SummaryError, match="historical fixed64 reconstruction failed"):
        SUMMARY.load_grid(tmp_path, expected_users=4)


def test_prediction_rank_and_stored_file_integrity_are_checked(tmp_path):
    _write_fixture(tmp_path)
    _mutate_npz(tmp_path / "cell-01", lambda data: data["ranks"].__setitem__((0, 0), 9))
    with pytest.raises(SUMMARY.SummaryError, match="stored ranks disagree"):
        SUMMARY.load_grid(tmp_path, expected_users=4)
    # Separate integrity check: changing input bytes without rehashing fails
    # before parsing its numerical data.
    (tmp_path / "cell-00/result.json").write_text("{}")
    with pytest.raises(SUMMARY.SummaryError, match="hash mismatch"):
        SUMMARY.load_grid(tmp_path, expected_users=4)


def test_legacy_policy_has_separate_equal_beam_and_final_step_contrasts(tmp_path):
    _write_fixture(tmp_path, legacy=True)
    grid = SUMMARY.load_grid(tmp_path, expected_users=4)
    result = SUMMARY.summarize(grid, n_boot=100)
    assert len(result["comparisons"]) == len(result["legacy_confidence_comparisons"]) == 6
    assert len(result["shared_vs_legacy_confidence"]) == 6
    assert result["historical_confidence256_reconstruction"]["comparisons"] == 36
    shared = next(row for row in result["comparisons"] if row["role"] == "primary")
    legacy = result["legacy_confidence_comparisons"][0]
    terminal = result["shared_vs_legacy_confidence"][0]
    assert shared["difference"] == pytest.approx(legacy["difference"] + terminal["difference"])
    assert legacy["role"] == terminal["role"] == "diagnostic"
    assert "Original confidence decoder at equal beams" in SUMMARY.render_report(result)
    _mutate_npz(tmp_path / "cell-01", lambda data: data["scores"].__setitem__((2, 0, 0), 0.0))
    with pytest.raises(SUMMARY.SummaryError, match="legacy score unavailability"):
        SUMMARY.load_grid(tmp_path, expected_users=4)


def test_legacy_confidence_reconstruction_is_also_a_gate(tmp_path):
    _write_fixture(tmp_path, legacy=True)

    def change(result):
        record = result["historical_confidence256_reconstruction"]["ndcg@10"]
        record["recorded"] += 0.75
        record["delta"] = record["new"] - record["recorded"]

    _mutate_json(tmp_path / "cell-01", change)
    with pytest.raises(SUMMARY.SummaryError, match="historical confidence256 reconstruction failed"):
        SUMMARY.load_grid(tmp_path, expected_users=4)
