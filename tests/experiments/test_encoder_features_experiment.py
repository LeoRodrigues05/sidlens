"""representation/exp5 helpers: history cuts, census assignment, matched controls (CPU, no data)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[2] / "experiments" / "representation" / "exp5_diffusion_encoder_features" / "common.py"
_spec = importlib.util.spec_from_file_location("exp5_common", _path)
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_history_cuts_follow_leave_last_out():
    seqs = {"u": ["a", "b", "c", "d", "e", "f"], "v": ["a", "c", "d"]}
    item2id = {k: i for i, k in enumerate("abcdef")}
    sids = np.arange(12).reshape(6, 2)
    test = C.build_histories(["u", "v"], seqs, item2id, sids, "test", slots=4)
    assert test.lengths.tolist() == [4, 2] and test.target_items.tolist() == [5, 3]
    assert test.items[0].tolist() == [1, 2, 3, 4]                      # the oldest item fell out of 4 slots
    fit = C.build_histories(["u", "v"], seqs, item2id, sids, "fit", slots=4)
    assert fit.target_items.tolist() == [2, 3]                          # t = 2, 3 for n = 6; none for n = 3
    assert fit.lengths.tolist() == [2, 3]
    assert (fit.sids[~fit.mask] == -1).all() and (fit.sids[0, 1] == sids[1]).all()
    table = C.slot_table(fit, sids, {})
    assert table["r"].tolist() == [2, 1, 3, 2, 1] and table["is_r1"].tolist() == [False, True, False, False, True]
    assert C.rec_bin(np.array([1, 5, 6, 10, 11, 49])).tolist() == [0, 4, 5, 5, 6, 6]


def test_census_assigns_the_better_explained_family_and_marks_dead_latents():
    rng = np.random.default_rng(0)
    rows = 400
    own = rng.integers(0, 4, rows)
    other = rng.integers(0, 4, rows)
    idx = np.stack([own, 4 + other, np.full(rows, 8)], 1)               # latent c fires on own == c
    val = np.ones_like(idx, dtype=float)
    fam = {"own": (np.ones(rows, bool), own, 4), "other": (np.ones(rows, bool), other, 4)}
    per, assigned, fires = C.census(idx, val, 10, fam)
    assert (assigned[:4] == "own").all() and (assigned[4:8] == "other").all()
    assert assigned[8] == "unassigned" and assigned[9] == "dead" and fires[9] == 0
    assert per["own"]["c_star"][2] == 2 and np.isclose(per["own"]["kappa"][2], 1.0)


def test_matched_control_takes_nearest_unused_latents_and_never_the_chosen_or_excluded():
    idx = np.array([10, 11, 12, 13, 14])
    val = np.array([5.0, 4.9, 1.0, 0.0, 3.0])
    out = C.matched_control(idx, val, np.array([10]), np.array([11]))
    assert out.tolist() == [14]                                          # 11 excluded, 13 inactive
    assert C.matched_control(idx, val, np.array([], dtype=int), np.array([], dtype=int)).tolist() == []
    with pytest.raises(ValueError):
        C.matched_control(idx, val, np.array([13]), np.array([], dtype=int))


def test_descriptions_are_one_hot_with_unknown_words():
    words = np.array([[0, 2], [1, 0]])
    X = C.one_hot_design(words, (2, 3))
    assert X.tolist() == [[1, 0, 0, 0, 1], [0, 1, 1, 0, 0]]
    assert C.with_unknown(np.array([-1, 2]), 5).tolist() == [5, 2]
    with pytest.raises(ValueError):
        C.one_hot_design(np.array([[2, 0]]), (2, 3))
