"""Design and paired inference checks independent of trained-model outputs."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).resolve().parents[2] / "experiments/controlled/exp2_fixed_orders/common.py"
_spec = importlib.util.spec_from_file_location("fixed_order_common", _path)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def test_declared_axes_cover_each_position_permutation_once():
    axes = common.conditions([64, 256])
    assert len(axes) == len(set(axes)) == 14
    assert {order for _, policy, order in axes if policy == "fixed"} == {
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)}
    for beams in ([], [64, 64], [32]):
        with pytest.raises(ValueError):
            common.conditions(beams)


def test_user_pairing_cancels_opposing_cell_effects_exactly():
    axes = common.conditions([64])
    values = np.zeros((2, 7, 5, 1))
    # The two cells have opposite effects for the SAME users. A correct paired
    # bootstrap returns zero width; independently resampling cells/users would not.
    values[0, 0, :, 0] = [1, 0, 1, 0, 1]
    values[1, 1:, :, 0] = [1, 0, 1, 0, 1]
    result = common.paired_effects(values, axes, n_boot=100)
    np.testing.assert_array_equal(result["overall"], [[0]])
    np.testing.assert_array_equal(result["overall_ci"], [[[0]], [[0]]])
    np.testing.assert_allclose(result["per_cell"], [[[.6], [-.6]]])


def test_fixed_comparator_is_mean_outcome_and_axis_permutation_invariant():
    axes = common.conditions([64])
    values = np.zeros((1, 7, 4, 1))
    values[:, 0] = 1
    values[:, 1:4] = 1
    result = common.paired_effects(values, axes, n_boot=20)
    np.testing.assert_allclose(result["overall"], [[.5]])
    permutation = [4, 2, 6, 0, 1, 5, 3]
    shuffled = common.paired_effects(values[:, permutation], [axes[i] for i in permutation], n_boot=20)
    np.testing.assert_array_equal(result["overall_ci"], shuffled["overall_ci"])
    with pytest.raises(ValueError, match="six unique"):
        common.paired_effects(values[:, :-1], axes[:-1], n_boot=10)
