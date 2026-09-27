"""Self-consistency of the cluster-geometry analysis (Experiment 2)."""

import numpy as np
import pytest

from sidlens.analysis import geometry as G
from sidlens.data import embeddings as E
from sidlens.data.sids import SidTable, SidVariant

CATEGORY = "Industrial_and_Scientific"


@pytest.fixture(scope="module")
def rqkmeans():
    t = SidTable.load("rqkmeans_3codebook_128")
    return t, E.load_aligned(CATEGORY, t.keys)


def test_depth_zero_is_the_whole_catalogue(rqkmeans):
    """R^2_0 == 1 by construction. If it drifts, the denominator is wrong."""
    table, z = rqkmeans
    s = G.profile(table.codes, z)[0]
    assert s.n_clusters == 1
    assert s.r2_within == pytest.approx(1.0)
    assert s.r2_multi == pytest.approx(1.0)
    assert s.largest_cluster == len(z)


def test_radius_matches_naive_definition(rqkmeans):
    """The fast path uses a centered one-pass identity; check it against the
    literal definition, since that is where precision could quietly be lost."""
    table, z = rqkmeans
    zc = z - z.mean(axis=0)
    for depth in (1, 2, 3):
        fast = G.depth_stats(table.codes, zc, depth)
        groups = {}
        for i, c in enumerate(table.codes):
            groups.setdefault(tuple(c[:depth]), []).append(i)
        radii = [float(np.sqrt(((zc[ix] - zc[ix].mean(axis=0)) ** 2).sum(axis=1).mean()))
                 for ix in groups.values() if len(ix) >= 2]
        assert fast.median_radius == pytest.approx(np.median(radii), rel=1e-10)
        assert fast.n_clusters == len(groups)


def test_residual_quantizers_refine_monotonically():
    """For a genuine residual quantizer, deeper clusters cannot be looser.

    Asserted only for rqvae and rqkmeans. MQ quantizes the full embedding
    independently per digit, so its depth axis is not a refinement hierarchy and
    monotonicity is not required of it.
    """
    for q in ("rqvae", "rqkmeans"):
        t = SidTable.load(SidVariant(q, 5, 256))
        z = E.load_aligned(CATEGORY, t.keys)
        prof = G.profile(t.codes, z)
        r2 = [s.r2_within for s in prof]
        assert all(a >= b - 1e-12 for a, b in zip(r2, r2[1:])), f"{q}: {r2}"
        radii = [s.median_radius for s in prof if s.median_radius is not None]
        assert all(a >= b - 1e-9 for a, b in zip(radii, radii[1:])), f"{q}: {radii}"


def test_r2_multi_is_not_just_r2_again():
    """R^2_multi must actually differ from R^2 once singletons appear.

    An earlier version restricted only the numerator, which made the two
    identical by construction -- singletons contribute zero -- and silently
    reported a column that answered nothing.
    """
    t = SidTable.load(SidVariant("MQ", 5, 256))
    z = E.load_aligned(CATEGORY, t.keys)
    prof = G.profile(t.codes, z)
    deep = [s for s in prof if s.frac_items_in_singletons > 0.5]
    assert deep, "expected some depth with a majority of singletons"
    for s in deep:
        assert s.r2_multi > s.r2_within * 1.5, (
            f"depth {s.depth}: r2_multi {s.r2_multi} vs r2_within {s.r2_within}")


def test_rqvae_first_digit_is_collapsed():
    """RQ-VAE's digit 0 uses a small fraction of its codebook while later
    digits use nearly all of it. This is load-bearing for how rqvae results are
    read, so it is pinned rather than left as a note."""
    t = SidTable.load(SidVariant("rqvae", 4, 256))
    util = t.codebook_utilization
    assert util[0]["n_used"] == 10, util[0]
    assert util[0]["max_count"] == 522
    for u in util[1:]:
        assert u["utilization"] > 0.95, u
    km = SidTable.load(SidVariant("rqkmeans", 4, 256)).codebook_utilization
    assert all(u["utilization"] == 1.0 for u in km)
