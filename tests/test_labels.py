"""The label layer, and the gate that decides what may be probed.

The gate is the part worth testing. Everything downstream trusts it to keep an
identity column out of an alignment table, and an identity column scores high
for any quantizer -- so a gate that silently opened would not fail loudly, it
would produce impressive numbers that mean nothing.
"""

import json

import numpy as np
import pytest

from sidlens.data import labels as L
from sidlens.data.sids import SidTable

CATEGORY = "Industrial_and_Scientific"


@pytest.fixture(scope="module")
def keys():
    return SidTable.load("rqkmeans_3codebook_128").keys


def test_join_covers_the_whole_catalogue():
    """The recovered labels are useless if they cover a subset of the items the
    SIDs partition. 100% is what the manifest claims; assert it independently."""
    man = L.manifest(CATEGORY)
    assert man["join"]["join_rate"] == 1.0
    assert man["join"]["joined"] == man["join"]["catalogue_items"]
    assert man["join"]["unmatched_asins"] == []


def test_brand_is_gated_as_an_identity_column(keys):
    """~2.6 items per brand. Probing it measures item identity, not semantics."""
    ok, why = L.usable(CATEGORY, keys)["brand_ext"]
    assert not ok
    assert "identity column" in why


def test_rank_is_usable_as_deciles_but_not_as_a_class_column(keys):
    """3008 distinct ranks over 3105 items is a near-unique key. It survives as
    a continuous field and as deciles; it must never become a categorical."""
    assert L.FIELDS["rank"].kind == "continuous"
    codes, vocab = L.codes(CATEGORY, keys, "rank_decile")
    assert len(vocab) == 10
    assert set(np.unique(codes[codes != L.MISSING]).tolist()) <= set(range(10))


def test_gated_field_requires_an_explicit_force(keys):
    with pytest.raises(ValueError, match="not usable"):
        L.codes(CATEGORY, keys, "cat_l3")
    codes, vocab = L.codes(CATEGORY, keys, "cat_l3", force=True)
    assert len(vocab) > 300


def test_codes_are_row_aligned_and_mark_missing(keys):
    """-1 must mean 'no label', never class 0. A silent conflation would put
    every unlabelled item into one large fake class."""
    codes, vocab = L.codes(CATEGORY, keys, "cat_l1")
    assert len(codes) == len(keys)
    lab = L.load(CATEGORY)
    for i in (0, 17, len(keys) - 1):
        want = lab[keys[i]].get("cat_l1") or ""
        if want:
            assert vocab[codes[i]] == want
        else:
            assert codes[i] == L.MISSING
    assert (codes == L.MISSING).sum() == sum(
        1 for k in keys if not (lab[k].get("cat_l1") or ""))


def test_vocab_is_stable_across_calls(keys):
    """Two callers must agree on what class 7 is, or their numbers cannot be
    compared. The vocab is built from `keys` order, so this pins that order."""
    a_codes, a_vocab = L.codes(CATEGORY, keys, "cat_l1")
    b_codes, b_vocab = L.codes(CATEGORY, keys, "cat_l1")
    assert a_vocab == b_vocab
    assert np.array_equal(a_codes, b_codes)


def test_deciles_partition_the_covered_items_only(keys):
    """An item with no price must not land in a price bin."""
    codes, _ = L.codes(CATEGORY, keys, "price_decile")
    lab = L.load(CATEGORY)
    for i, k in enumerate(keys):
        has_price = isinstance(lab[k].get("price_usd"), (int, float))
        assert (codes[i] != L.MISSING) == has_price


def test_decile_bins_are_monotonic_in_the_underlying_value(keys):
    """Bin index must increase with price, or 'decile' is a misnomer."""
    codes, _ = L.codes(CATEGORY, keys, "price_decile")
    lab = L.load(CATEGORY)
    pairs = [(lab[k]["price_usd"], int(codes[i]))
             for i, k in enumerate(keys) if codes[i] != L.MISSING]
    pairs.sort()
    bins = [b for _, b in pairs]
    assert bins == sorted(bins)


def test_neighbour_sets_stay_inside_the_catalogue(keys):
    """also_buy points at all of Amazon. An out-of-catalogue neighbour would be
    counted as a miss in every preservation statistic and silently deflate it."""
    ns = L.neighbour_sets(CATEGORY, keys)
    inside = set(keys)
    assert set(ns) == inside
    for k, nbrs in ns.items():
        assert nbrs <= inside
    assert sum(len(v) for v in ns.values()) > 0


def test_external_labels_are_not_inside_frozen():
    """The frozen substrate is defined as the bytes the models trained on. These
    labels were pulled afterwards; joining them in would destroy that claim."""
    from sidlens import paths
    man = L.manifest(CATEGORY)
    assert paths.FROZEN not in (paths.EXTERNAL / "x").parents
    assert not str(man["labels_path"]).startswith(str(paths.FROZEN))
    assert str(man["labels_path"]).startswith(str(paths.EXTERNAL))


def test_manifest_records_the_source_bytes():
    """A label table with no recorded source cannot be re-derived or audited."""
    man = L.manifest(CATEGORY)
    assert len(man["source"]["sha256"]) == 64
    assert man["source"]["url"].startswith("https://")
    assert man["source"]["records_scanned"] > 100_000
    assert json.loads(json.dumps(man))  # round-trips; no NaN leaked in
