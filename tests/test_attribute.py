"""The conditional-AMI machinery, on synthetic tables with a known answer.

Real SID tables cannot test this: nobody knows the true conditional alignment of
`rqkmeans 3cb x 128`, so a wrong implementation would produce a plausible number
and go unnoticed. These build tables where the answer is known by construction --
a digit that carries the label, a digit that carries nothing, and a digit that
carries the label only through its parent -- and check the statistic tells them
apart. The third is the one that matters: it is exactly the confound the
conditional measure exists to remove.
"""

import numpy as np
import pytest

from sidlens.analysis import attribute as AT
from sidlens.data import labels as L


class FakeVariant:
    def __init__(self, n_codebook, codebook_size=8):
        self.n_codebook = n_codebook
        self.codebook_size = codebook_size
        self.quantizer = "rqkmeans"
        self.name = f"fake_{n_codebook}codebook_{codebook_size}"


class FakeTable:
    """The two attributes `alignment` touches: `variant` and `asin2codes`."""

    def __init__(self, codes: dict, n_codebook: int):
        self.variant = FakeVariant(n_codebook)
        self.asin2codes = codes

    @property
    def keys(self):
        return list(self.asin2codes)

    def clusters(self, d):
        out = {}
        for k, c in self.asin2codes.items():
            out.setdefault(c[:d], []).append(k)
        return out


def build(n_parents=12, per_parent=40, seed=0):
    """A 2-digit table: digit 0 is the parent, digit 1 varies inside it.

    Four label vectors, each with a known answer:
      informative   label == digit 1. Conditional must be ~1.
      noise         label independent of everything. Conditional must be ~0.
      parent_only   label is a function of the parent ALONE and is therefore
                    constant inside every parent. There is no within-parent
                    question to ask, so the conditional is undefined, not zero.
      parent_noisy  parent-driven but with 15% of labels resampled, so parents
                    do vary internally while digit 1 still explains none of it.
                    This is the realistic confound: marginal association is real,
                    conditional association must vanish.

    For that confound to exist at all, digit 1 must itself correlate with the
    parent -- otherwise there is nothing for conditioning to remove and the test
    would pass against an implementation that ignored the prefix entirely. So
    each parent prefers one digit-1 code 70% of the time and draws uniformly
    otherwise, which is also what a real residual quantizer does. The preferred
    code is keyed to `p % 3`, the same grouping the label uses -- keying it to
    `p % 4` would make the two independent by the CRT (12 parents = 3 x 4) and
    quietly destroy the confound this fixture exists to create.
    """
    rng = np.random.default_rng(seed)
    codes, informative, noise, parent_only, parent_noisy = {}, [], [], [], []
    i = 0
    for p in range(n_parents):
        for j in range(per_parent):
            k = f"A{i:05d}"
            i += 1
            d1 = (p % 3) if rng.random() < 0.7 else int(rng.integers(0, 4))
            codes[k] = (p, d1)
            informative.append(d1)
            noise.append(int(rng.integers(0, 4)))
            parent_only.append(p % 3)
            parent_noisy.append(int(rng.integers(0, 3)) if rng.random() < 0.15
                                else p % 3)
    return (FakeTable(codes, 2), list(codes), np.array(informative),
            np.array(noise), np.array(parent_only), np.array(parent_noisy))


@pytest.fixture(scope="module")
def fixture():
    return build()


def test_digit_zero_conditional_equals_marginal(fixture):
    """With no prefix there is nothing to condition on. Anything else would mean
    the two paths disagree about the same quantity."""
    table, keys, informative, *_ = fixture
    rows = AT.alignment(table, informative, keys, "informative", n_perm=0)
    assert rows[0]["digit"] == 0
    assert rows[0]["conditional_ami"] == rows[0]["marginal_ami"]
    assert rows[0]["n_items_conditioned"] == len(keys)


def test_a_digit_that_determines_the_label_scores_near_one(fixture):
    table, keys, informative, *_ = fixture
    rows = AT.alignment(table, informative, keys, "informative", n_perm=50)
    d1 = rows[1]
    assert d1["conditional_ami"] > 0.95
    assert d1["cond_z"] > 5
    assert d1["cond_p"] <= 0.02
    assert d1["conditional_supported"]


def test_an_uninformative_digit_sits_on_its_null(fixture):
    """AMI is chance-corrected, so the value must land near zero AND the
    permutation z must not flag it. A test on the value alone would pass even if
    the null were computed wrongly."""
    table, keys, _, noise, _, _ = fixture
    rows = AT.alignment(table, noise, keys, "noise", n_perm=100, seed=3)
    d1 = rows[1]
    assert abs(d1["conditional_ami"]) < 0.05
    assert abs(d1["cond_z"]) < 3
    assert d1["cond_p"] > 0.01


def test_conditioning_removes_a_parent_driven_confound(fixture):
    """The whole reason the conditional measure exists.

    The label is driven by the parent, so digit 1 adds nothing. The MARGINAL
    still sees strong association -- digit 1 co-occurs with parents that carry
    the label -- and the conditional must not."""
    table, keys, _, _, _, parent_noisy = fixture
    rows = AT.alignment(table, parent_noisy, keys, "parent_noisy",
                        n_perm=100, seed=5)
    # Thresholds are set against the chance-corrected scale, where a marginal
    # AMI near 0.3 between a 3-class label and a 4-code digit is a large
    # association -- not against an intuition calibrated on raw correlation.
    assert rows[0]["marginal_ami"] > 0.25         # the confound is really there
    d1 = rows[1]
    assert d1["marginal_ami"] > 0.25              # and it survives into digit 1
    assert abs(d1["conditional_ami"]) < 0.05      # but not past conditioning
    assert abs(d1["cond_z"]) < 3
    # The gap is the claim: conditioning must remove most of what was there.
    assert d1["marginal_ami"] - abs(d1["conditional_ami"]) > 0.2


def test_a_label_constant_within_parents_is_undefined_not_zero(fixture):
    """When the label never varies inside a parent there is no question to ask.
    Reporting 0 would read as 'digit 1 adds nothing measurable here', which is a
    different and much stronger claim than 'this cannot be measured'."""
    table, keys, _, _, parent_only, _ = fixture
    rows = AT.alignment(table, parent_only, keys, "parent_only", n_perm=0)
    d1 = rows[1]
    assert d1["conditional_ami"] is None
    assert d1["n_parent_cells"] == 0
    assert d1["conditioned_share"] == 0.0
    assert not d1["conditional_supported"]
    assert d1["marginal_ami"] > 0.25   # the marginal confound is real...
    assert rows[0]["marginal_ami"] > 0.5   # ...and the parent fully carries it


def test_null_permutes_within_parents_not_globally():
    """A global shuffle would break the parent-label link too, and would then
    call a parent-driven digit 'significant'. Pin the within-parent behaviour:
    under a parent-driven label, every permuted replicate scores near zero --
    and so does the real statistic, which is why it cannot be flagged."""
    table, keys, _, _, _, parent_noisy = build(seed=11)
    codes = np.array([table.asin2codes[k] for k in keys])
    null = AT._conditional_null(codes[:, :1], codes[:, 1], parent_noisy,
                                n_perm=40, seed=0)
    assert null.size == 40
    assert np.abs(null).max() < 0.1
    observed = AT._pooled_conditional(codes[:, :1], codes[:, 1], parent_noisy)[0]
    assert observed < null.max() + 0.05


def test_missing_labels_are_dropped_once_for_every_digit():
    """Each digit must be scored on the same items, or the rows in a column are
    not comparable."""
    table, keys, informative, *_ = build(seed=7)
    y = informative.copy()
    y[::5] = L.MISSING
    rows = AT.alignment(table, y, keys, "holey", n_perm=0)
    assert len({r["n_labelled"] for r in rows}) == 1
    assert rows[0]["n_labelled"] == int((y != L.MISSING).sum())


def test_conditioned_share_is_reported(fixture):
    """Conditional values at depth get thin. The share makes that visible
    without the reader doing the division."""
    table, keys, informative, *_ = fixture
    rows = AT.alignment(table, informative, keys, "informative", n_perm=0)
    for r in rows:
        assert 0.0 <= r["conditioned_share"] <= 1.0
    assert rows[0]["conditioned_share"] == 1.0


def test_copurchase_lift_is_one_when_edges_ignore_the_partition(monkeypatch):
    """Chance correction check. Edges drawn independently of the SID must give a
    lift near 1; if the chance rate were computed wrongly, every quantizer would
    look good (or bad) by a constant factor."""
    table, keys, *_ = build(n_parents=10, per_parent=30, seed=2)
    rng = np.random.default_rng(0)
    fake = {k: {keys[j] for j in rng.choice(len(keys), 12, replace=False)}
            for k in keys}
    monkeypatch.setattr(L, "neighbour_sets", lambda cat, ks: fake)
    rows = AT.copurchase_preservation(table, keys, "irrelevant")
    for r in rows:
        assert r["lift"] == pytest.approx(1.0, abs=0.35)
