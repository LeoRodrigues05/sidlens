"""The three metrics behind the structure-layer figures, on inputs small enough
to check by hand. The figure script itself needs the frozen substrate, so it is
not imported through the package; the functions are loaded from the file."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "experiments" / "structure" / "semantic_mapping" / "figures.py"


@pytest.fixture(scope="module")
def figs():
    spec = importlib.util.spec_from_file_location("semantic_figures", _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prefix_groups_counts_and_nesting(figs):
    codes = np.array([[1, 5], [1, 5], [1, 6], [2, 5]])
    g1, c1 = figs.prefix_groups(codes, 1)
    g2, c2 = figs.prefix_groups(codes, 2)
    assert sorted(c1.tolist()) == [1, 3]
    assert sorted(c2.tolist()) == [1, 1, 2]
    # depth-2 groups never straddle depth-1 groups
    for g in np.unique(g2):
        assert len(np.unique(g1[g2 == g])) == 1


def test_item_weighted_purity_ignores_missing_labels(figs):
    group = np.array([0, 0, 0, 1, 1])
    lab = np.array([3, 3, 7, -1, 4])
    # group 0: majority 3 covers 2 of 3 labelled; group 1: one labelled item
    assert figs.item_weighted_purity(group, lab) == pytest.approx(3 / 4)


def test_purity_multi_restricts_to_shared_groups(figs):
    group = np.array([0, 0, 1, 2, 3])
    counts = np.bincount(group)
    lab = np.array([1, 1, 2, 2, 2])
    out = figs.purity_multi(group, counts, lab, np.random.default_rng(0), n_perm=20)
    assert out["purity"] == pytest.approx(1.0)
    assert out["multi_item_share"] == pytest.approx(2 / 5)
    # a permutation within the two grouped items cannot change their purity
    assert out["null_mean"] == pytest.approx(1.0)


def test_between_share_bounds(figs):
    rng = np.random.default_rng(1)
    z = rng.normal(size=(40, 6))
    one = figs.between_share(z, np.zeros(40, dtype=int))
    each = figs.between_share(z, np.arange(40))
    assert one == pytest.approx(0.0, abs=1e-12)
    assert each == pytest.approx(1.0, abs=1e-12)
    # two well-separated blobs explain most of the variance
    z2 = np.vstack([rng.normal(size=(20, 6)), rng.normal(size=(20, 6)) + 20])
    assert figs.between_share(z2, np.repeat([0, 1], 20)) > 0.95
