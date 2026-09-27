"""Exact SID parsing and collision-aware item-rank bounds."""

import pytest

from sidlens.analysis.collisions import (
    exact_rank,
    exact_sign_test,
    item_rank_bounds,
    parse_sid,
)
from experiments.retrospective.exp3_collisions.run import (
    bootstrap_user_contrasts,
    weighted_frequency_contrast,
)


def test_parse_sid_is_strict_about_digit_positions_and_extra_text():
    assert parse_sid(' "<a_12> <b_3>\n" ', 2) == (12, 3)
    with pytest.raises(ValueError, match="non-consecutive"):
        parse_sid("<a_1><c_2>")
    with pytest.raises(ValueError, match="malformed"):
        parse_sid("answer=<a_1><b_2>")
    with pytest.raises(ValueError, match="expected 3"):
        parse_sid("<a_1><b_2>", 3)


def test_exact_rank_uses_first_literal_sid_hit():
    preds = [(1, 2), (3, 4), (1, 2)]
    assert exact_rank(preds, (1, 2)) == 1
    assert exact_rank(preds, (9, 9)) is None


def test_item_bounds_expand_prior_buckets_and_drop_repeated_sids():
    buckets = {(1,): [7, 2], (2,): [3, 4, 5], (3,): [9]}
    # The repeated first SID does not consume two more item slots.
    bounds = item_rank_bounds([(1,), (1,), (2,), (3,)], 4, buckets)
    assert bounds.best == 3
    assert bounds.worst == 5
    assert bounds.catalogue == 4       # sorted bucket [3,4,5]
    assert bounds.multiplicity == 3
    assert bounds.hit_bounds(2) == {
        "lower": 0.0, "uniform": 0.0, "catalogue": 0.0, "upper": 0.0}
    assert bounds.hit_bounds(3)["uniform"] == pytest.approx(1 / 3)
    assert bounds.hit_bounds(4)["uniform"] == pytest.approx(2 / 3)
    assert bounds.hit_bounds(5) == {
        "lower": 1.0, "uniform": 1.0, "catalogue": 1.0, "upper": 1.0}

    fast_bounds = item_rank_bounds(
        [(1,), (1,), (2,), (3,)], 4, buckets, target_sid=(2,))
    assert fast_bounds == bounds
    with pytest.raises(KeyError, match="absent from supplied"):
        item_rank_bounds([(2,)], 4, buckets, target_sid=(1,))


def test_missing_target_sid_has_zero_hit_bounds():
    bounds = item_rank_bounds([(1,)], 9, {(1,): [1], (2,): [9, 10]})
    assert bounds.best is None
    assert bounds.multiplicity == 2
    assert set(bounds.hit_bounds(10).values()) == {0.0}


def test_exact_sign_test_known_small_cases():
    assert exact_sign_test(0, 0) == 1.0
    assert exact_sign_test(1, 1) == 1.0
    assert exact_sign_test(5, 0) == pytest.approx(0.0625)


def test_primary_collision_contrast_is_standardized_within_configuration():
    records = []
    # Cell A is mostly collided and has RD=1; cell B is mostly singleton and
    # has RD=0. Equal-cell standardization is .5, whereas raw pooling is .9.
    for index in range(9):
        records.append({
            "user": f"u{index}", "variant": "A", "collided": True,
            "sid_hit@10": 1.0, "item_uniform@10": 0.0})
        records.append({
            "user": f"v{index}", "variant": "B", "collided": False,
            "sid_hit@10": 0.0, "item_uniform@10": 0.0})
    records.extend((
        {"user": "u9", "variant": "A", "collided": False,
         "sid_hit@10": 0.0, "item_uniform@10": 0.0},
        {"user": "v9", "variant": "B", "collided": True,
         "sid_hit@10": 0.0, "item_uniform@10": 0.0},
    ))
    result = bootstrap_user_contrasts(records, n_boot=0, seed=1)
    assert result["within_config_collision_rd@10"]["estimate"] == pytest.approx(0.5)
    assert result["raw_collision_rd@10"]["estimate"] == pytest.approx(0.9)


def test_cluster_bootstrap_is_deterministic_and_keeps_all_cells_per_user():
    records = []
    for user in ("u1", "u2", "u3"):
        for variant in ("A", "B"):
            records.extend((
                {"user": user, "variant": variant, "collided": False,
                 "sid_hit@10": 0.0, "item_uniform@10": 0.0},
                {"user": user, "variant": variant, "collided": True,
                 "sid_hit@10": 1.0, "item_uniform@10": 0.5},
            ))
    first = bootstrap_user_contrasts(records, n_boot=50, seed=9)
    second = bootstrap_user_contrasts(records, n_boot=50, seed=9)
    assert first == second
    assert first["within_config_collision_rd@10"]["estimate"] == 1.0
    assert first["within_config_collision_rd@10"]["valid_bootstrap_replicates"] == 50
    assert first["sid_minus_uniform_item_singleton_target@10"]["estimate"] == 0.0
    assert first["sid_minus_uniform_item_collided_target@10"]["estimate"] == 0.5


def test_frequency_adjustment_reports_overlap_support_coverage():
    rows = [
        {"has_collision_group_overlap": True, "n_singleton": 2,
         "n_collided": 3, "risk_difference": 0.25, "overlap_weight": 1.2},
        {"has_collision_group_overlap": False, "n_singleton": 5,
         "n_collided": 0, "risk_difference": None, "overlap_weight": 0.0},
    ]
    result = weighted_frequency_contrast(rows)
    assert result["adjusted_risk_difference@10"] == pytest.approx(0.25)
    assert result["n_total_strata"] == 2
    assert result["n_overlap_strata"] == 1
    assert result["support"]["row_fraction"] == pytest.approx(0.5)
    assert result["support"]["singleton_fraction"] == pytest.approx(2 / 7)
