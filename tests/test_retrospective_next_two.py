"""Focused tests for the retrospective next-two audit."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "experiments/retrospective/exp4_next_two/run.py"
SPEC = importlib.util.spec_from_file_location("retrospective_exp4_next_two", RUN_PATH)
assert SPEC is not None and SPEC.loader is not None
E4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E4)


def test_sid_and_pair_parser_are_strict():
    assert E4.parse_sid(' "<a_1> <b_22><c_3>" ', 3) == (1, 22, 3)
    assert E4.parse_pair("<a_1><b_2> ||| <a_3><b_4>", 2) == (
        (1, 2), (3, 4))
    assert E4.parse_item_pair("12 ||| 99") == (12, 99)
    with pytest.raises(ValueError, match="malformed"):
        E4.parse_sid("answer=<a_1><b_2>", 2)
    with pytest.raises(ValueError, match="consecutive"):
        E4.parse_sid("<a_1><c_2>", 2)
    with pytest.raises(ValueError, match="exactly two"):
        E4.parse_pair("<a_1> ||| <a_2> ||| <a_3>", 1)


def test_ranks_preserve_archived_positions_and_invalid_beams():
    pairs = [(None, (9,)), ((1,), (8,)), ((2,), (9,)), ((1,), (9,))]
    assert E4.slot_rank(pairs, 0, (1,)) == 2
    assert E4.slot_rank(pairs, 1, (9,)) == 1
    assert E4.first_rank(pairs, ((1,), (9,))) == 4
    assert E4.first_rank(pairs, ((9,), (1,))) is None
    assert E4.hit(4, 3) == 0.0
    assert E4.hit(4, 4) == 1.0


def test_final_row_scope_keeps_last_position_per_user():
    rows = [
        {"user_id": "A1"},
        {"user_id": "A2"},
        {"user_id": "A1"},
        {"user_id": "A3"},
        {"user_id": "A2"},
    ]
    assert E4.final_row_indices(rows) == {2, 3, 4}


def test_entropy_and_diversity_do_not_mistake_determinism_for_independence():
    assert E4.normalized_entropy([]) == 0.0
    assert E4.normalized_entropy(["x", "x", "x"]) == 0.0
    assert E4.normalized_entropy(["x", "y"]) == pytest.approx(1.0)


def test_coupling_lift_uses_off_diagonal_pairing_and_holds_out_user_counts():
    a, b, x, y = (1,), (2,), (8,), (9,)
    pairs = [(a, x), (b, y)]
    counts = Counter({(a, x): 9, (b, y): 4, (a, y): 0, (b, x): 0})
    expected = (pytest.approx((__import__("math").log1p(9)
                              + __import__("math").log1p(4)) / 2))
    assert E4.coupling_lift(pairs, counts) == expected

    # Removing the held-out user's copies reduces, rather than leaks, the
    # compatibility evidence associated with the same user's training rows.
    heldout = Counter({(a, x): 9, (b, y): 4})
    assert E4.coupling_lift(pairs, counts, heldout) == pytest.approx(0.0)


def test_cluster_bootstrap_uses_user_not_row_as_sampling_unit():
    values = [("u1", 1.0), ("u1", 1.0), ("u2", -1.0)]
    result = E4.cluster_bootstrap(values, n_boot=200, seed=7)
    assert result["estimate"] == pytest.approx(1 / 3)
    assert result["n_rows"] == 3
    assert result["n_users"] == 2
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_conditional_contingency_reports_rates_denominators_and_cluster_ci():
    rows = [
        {
            "user": "u1", "conditional_pair_eligible": True,
            "top1_n_p1_correct": 1, "top1_p2_correct_p1_correct": 1,
            "top1_n_p1_wrong": 0, "top1_p2_correct_p1_wrong": 0,
        },
        {
            "user": "u1", "conditional_pair_eligible": True,
            "top1_n_p1_correct": 0, "top1_p2_correct_p1_correct": 0,
            "top1_n_p1_wrong": 1, "top1_p2_correct_p1_wrong": 0,
        },
        {
            "user": "u2", "conditional_pair_eligible": True,
            "top1_n_p1_correct": 1, "top1_p2_correct_p1_correct": 0,
            "top1_n_p1_wrong": 1, "top1_p2_correct_p1_wrong": 1,
        },
        {
            "user": "u3", "conditional_pair_eligible": False,
            "top1_n_p1_correct": 1, "top1_p2_correct_p1_correct": 1,
            "top1_n_p1_wrong": 0, "top1_p2_correct_p1_wrong": 0,
        },
    ]
    counts = E4.conditional_counts(rows, "top1")
    assert counts["n_p1_correct"] == 2
    assert counts["p2_correct_p1_correct"] == 1
    assert counts["p2_rate_given_p1_correct"] == pytest.approx(0.5)
    assert counts["n_p1_wrong"] == 2
    assert counts["p2_correct_p1_wrong"] == 1
    assert counts["p2_rate_given_p1_wrong"] == pytest.approx(0.5)
    assert counts["conditional_risk_difference"] == pytest.approx(0.0)

    boot = E4.cluster_bootstrap_conditional(rows, "top1", n_boot=200, seed=3)
    assert boot["estimate"] == pytest.approx(0.0)
    assert boot["n_rows"] == 3
    assert boot["n_users"] == 2
    assert boot["n_p1_correct"] == 2
    assert boot["n_p1_wrong"] == 2


def test_pooled_conditional_primary_is_standardized_within_cell():
    rows = []
    # Cell A has RD=1 and mostly p1-correct rows; cell B has RD=0 and mostly
    # p1-wrong rows. Raw pooling is .9, while the equal-cell estimand is .5.
    for index in range(9):
        rows.append({
            "user": f"a{index}", "variant": "A",
            "conditional_pair_eligible": True,
            "top1_n_p1_correct": 1, "top1_p2_correct_p1_correct": 1,
            "top1_n_p1_wrong": 0, "top1_p2_correct_p1_wrong": 0,
        })
        rows.append({
            "user": f"b{index}", "variant": "B",
            "conditional_pair_eligible": True,
            "top1_n_p1_correct": 0, "top1_p2_correct_p1_correct": 0,
            "top1_n_p1_wrong": 1, "top1_p2_correct_p1_wrong": 0,
        })
    rows.extend((
        {"user": "a9", "variant": "A", "conditional_pair_eligible": True,
         "top1_n_p1_correct": 0, "top1_p2_correct_p1_correct": 0,
         "top1_n_p1_wrong": 1, "top1_p2_correct_p1_wrong": 0},
        {"user": "b9", "variant": "B", "conditional_pair_eligible": True,
         "top1_n_p1_correct": 1, "top1_p2_correct_p1_correct": 0,
         "top1_n_p1_wrong": 0, "top1_p2_correct_p1_wrong": 0},
    ))
    result = E4.cluster_bootstrap_conditional(
        rows, "top1", n_boot=0, seed=1, standardize_by="variant")
    assert result["estimate"] == pytest.approx(0.5)
    assert result["pooled_risk_difference"] == pytest.approx(0.9)
    assert result["standardization"] == "variant"
    assert result["n_strata"] == 2


def test_item_pair_bounds_and_swap_exclusion_in_summary():
    base = {
        "user": "u", "swap_eligible": False, "n_pairs": 1,
        "n_valid_pairs": 1, "p2_unique_fraction": 1.0,
        "p2_entropy_normalized": 0.0, "self_pair_fraction": 0.0,
        "ordered_rr": 1.0, "reversed_rr": 1.0, "coupling_lift": None,
        "conditional_pair_eligible": False,
        "top1_n_p1_correct": 0, "top1_p2_correct_p1_correct": 0,
        "top1_n_p1_wrong": 0, "top1_p2_correct_p1_wrong": 0,
        "all_pairs_n_p1_correct": 0,
        "all_pairs_p2_correct_p1_correct": 0,
        "all_pairs_n_p1_wrong": 0,
        "all_pairs_p2_correct_p1_wrong": 0,
    }
    for k in E4.CUTOFFS:
        base.update({
            f"slot1_hit@{k}": 1.0, f"slot2_hit@{k}": 1.0,
            f"ordered_pair_hit@{k}": 1.0, f"reversed_pair_hit@{k}": 1.0,
            f"unordered_pair_hit@{k}": 1.0,
            f"representative_item_pair_lower@{k}": 0.0,
            f"representative_item_pair_uniform@{k}": 0.25,
            f"representative_item_pair_upper@{k}": 1.0,
        })
    summary = E4.summarize_rows([base], E4.CUTOFFS)
    assert summary["representative_item_pair_lower@10"] == 0.0
    assert summary["representative_item_pair_uniform@10"] == 0.25
    assert summary["representative_item_pair_upper@10"] == 1.0
    assert summary["n_swap_eligible"] == 0
    assert summary["reversed_pair_hit@10"] != summary["reversed_pair_hit@10"]  # NaN


def test_diffusion_source_audit_detects_effective_one_sid_two_pass_path():
    audit = E4.diffusion_source_audit(ROOT)
    assert audit["training_target_items_concatenated"] is True
    assert audit["decoder_bidirectional_verified"] is True
    assert audit["inference_hardcodes_one_sid_verified"] is True
    assert audit["fast_beam_max_len_is_ignored"] is True
    assert audit["trainer_takes_first_sid_in_pass1"] is True
    assert audit["trainer_appends_pass1_sid_to_history"] is True
    assert audit["trainer_takes_last_slice_in_pass2"] is True
    assert "immediate next item" in audit["effective_retained_evaluation"]
    assert "not a retained joint-block" in audit["mechanistic_consequence"]
