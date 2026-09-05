"""Invariants of the SID substrate.

These encode the consistency hazards found while auditing upstream. Each one is
a property that, if it silently broke, would corrupt interpretability results
without producing an error anywhere.
"""

import json
import re

import numpy as np
import pytest

from sidlens import paths
from sidlens.data.sids import (SID_OFFSET, SidTable, SidVariant, available,
                               check_nesting, is_nested_family, load_item2id)

CATEGORY = "Industrial_and_Scientific"
N_ITEMS = 3105


@pytest.fixture(scope="module")
def variants():
    return available("diffgrm")


def test_all_27_variants_present(variants):
    assert len(variants) == 27
    assert {v.quantizer for v in variants} == {"rqvae", "rqkmeans", "MQ"}
    assert {v.n_codebook for v in variants} == {3, 4, 5}
    assert {v.codebook_size for v in variants} == {128, 256, 512}


def test_every_variant_loads_with_valid_codes(variants):
    """SidTable's constructor rejects out-of-range codes.

    An out-of-range code is the signature of the faiss bit-packing defect, so
    this doubles as a check that no .mispacked file leaked into the live set.
    """
    for v in variants:
        t = SidTable.load(v)
        assert t.n_items == N_ITEMS, f"{v.name}: {t.n_items} items"
        assert t.codes.min() >= 0
        assert t.codes.max() < v.codebook_size


def test_token_offsets_match_model_embedding_layout(variants):
    """Digit d must occupy embedding rows [3 + d*K, 3 + (d+1)*K)."""
    for v in variants[:6]:
        t = SidTable.load(v)
        for key in t.keys[:50]:
            codes, tokens = t.asin2codes[key], t.tokens(key)
            for d, (c, tok) in enumerate(zip(codes, tokens)):
                assert tok == c + SID_OFFSET + d * v.codebook_size
                assert SID_OFFSET + d * v.codebook_size <= tok < SID_OFFSET + (d + 1) * v.codebook_size


def test_cb2items_is_lossless_where_the_pickle_is_not(variants):
    """Every item must be recoverable from an SID, including collided ones.

    Upstream's `tokens2item[tuple] = item_id` keeps only the last writer. At
    rqkmeans 3cb x 128 that is 2195 entries for 3105 items -- decoding through
    it loses 29% of the catalogue.
    """
    t = SidTable.load("rqkmeans_3codebook_128")
    assert sum(len(v) for v in t.cb2items.values()) == N_ITEMS
    assert len(t.cb2items) < N_ITEMS, "expected collisions in this variant"

    pkl = next((paths.FROZEN_SIDS / "mappings").glob(
        "tokens2item_*EXTERNAL3x7_rqkmeans_3codebook_128*.pkl"), None)
    if pkl is not None:
        import pickle
        lossy = pickle.loads(pkl.read_bytes())
        assert len(lossy) == len(t.cb2items)
        assert len(lossy) < N_ITEMS


def test_repaired_rqkmeans_index_json_agrees_with_sem_ids():
    """The 2026-08-20 repair must have landed in both representations.

    Disagreement would mean a model was trained on SIDs that differ from the
    ones we would analyse it with.
    """
    for cb in (3, 4, 5):
        for size in (128, 256, 512):
            v = SidVariant("rqkmeans", cb, size)
            sem = SidTable.load(v)
            idx = SidTable.load_index_json(CATEGORY, v)
            item2id = load_item2id(CATEGORY)
            assert len(idx.asin2codes) == len(sem.asin2codes) == N_ITEMS
            agree = sum(
                1 for asin, codes in sem.asin2codes.items()
                if idx.asin2codes.get(str(item2id[asin])) == codes)
            assert agree == N_ITEMS, f"{v.name}: only {agree}/{N_ITEMS} agree"


def test_diffgrm_and_diffgrm_new_sem_ids_are_identical(variants):
    """The two trees must not have drifted apart."""
    for v in variants:
        a = SidTable.load(v, tree="diffgrm")
        b = SidTable.load(v, tree="diffgrm_new")
        assert a.asin2codes == b.asin2codes, f"{v.name} differs between trees"


def test_repair_actually_changed_the_seven_and_nine_bit_variants():
    """The quarantined originals must differ from what replaced them.

    faiss `compute_codes()` returns a packed little-endian bit stream, which
    coincides with one-code-per-byte only at nbits == 8. So the repair had to
    change the 128 (7-bit) and 512 (9-bit) families and must have left 256
    (8-bit) untouched -- which is why only five files are quarantined and none
    of them is a 256.
    """
    root = paths.FROZEN_SIDS / "mispacked"
    files = sorted(root.glob("*.mispacked"))
    assert len(files) == 5, [f.name for f in files]

    pat = re.compile(r"rqkmeans\.index_(\d+)codebook_(\d+)\.json\.mispacked$")
    item2id = load_item2id(CATEGORY)
    id2item = {str(v): k for k, v in item2id.items()}

    for mis in files:
        m = pat.search(mis.name)
        assert m, mis.name
        cb, size = int(m.group(1)), int(m.group(2))
        assert size in (128, 512), \
            f"{mis.name}: 8-bit (size 256) packing is unaffected, should not be quarantined"

        old = json.loads(mis.read_text())
        new = SidTable.load(SidVariant("rqkmeans", cb, size))
        differing = sum(
            1 for k, toks in old.items()
            if tuple(int(t.split("_")[1].rstrip(">")) for t in toks)
            != new.asin2codes[id2item[k]])
        assert differing > 0, f"{mis.name}: repair changed nothing"


def test_duplicate_floor_is_below_every_collision_rate(variants):
    """11/3105 items are genuine catalogue duplicates.

    No quantizer can go below that, so it is the floor any collision plot must
    draw -- treating 0% as the reference would overstate every quantizer's gap.
    """
    for v in variants:
        s = SidTable.load(v).collision_stats
        assert s["collision_rate"] >= s["floor_rate"] - 1e-9, v.name


def test_mq_is_not_nested_but_residual_quantizers_are():
    """MQ's digits are parallel, not a refinement hierarchy.

    Exp2's R^2_d means something different for MQ, so this structural fact needs
    a test rather than a comment -- if MQ ever became nested, the separate
    treatment would be wrong.
    """
    def level0_agreement(q):
        a = SidTable.load(SidVariant(q, 3, 128))
        b = SidTable.load(SidVariant(q, 4, 128))
        return sum(1 for k in a.keys if a.asin2codes[k][0] == b.asin2codes[k][0])

    assert level0_agreement("rqkmeans") == N_ITEMS, "RQ-KMeans must be nested"
    assert level0_agreement("MQ") < N_ITEMS * 0.05, "MQ must not be nested"


# --- depth-family nesting ----------------------------------------------------

def test_rqkmeans_depth_variants_are_one_fit_truncated():
    """3cb must be a byte-exact prefix of 5cb, or 'add a digit' is not a clean
    manipulation and RQ3's redundancy question cannot be asked within a family."""
    for size in (128, 256):
        a = SidTable.load(f"rqkmeans_3codebook_{size}")
        e = SidTable.load(f"rqkmeans_5codebook_{size}")
        rep = check_nesting(a, e)
        assert rep["nested"], rep


def test_rqvae_and_mq_depth_variants_are_independent_fits():
    """They share nothing, so a 3cb-vs-4cb comparison changes the quantizer as
    well as the depth. Pinned so nobody reads such a comparison as pure scaling."""
    for q in ("rqvae", "MQ"):
        rep = check_nesting(SidTable.load(f"{q}_3codebook_128"),
                            SidTable.load(f"{q}_4codebook_128"))
        assert not rep["nested"]
        assert rep["agree_rate"] < 0.05, rep
        assert not is_nested_family(f"{q}_3codebook_128")


def test_rqkmeans_5cb_512_is_a_separate_lineage():
    """The mispacking exception. Its siblings agree 3105/3105; it agrees on ~0,
    which is the signature of a different fit rather than a deeper one."""
    a = SidTable.load("rqkmeans_3codebook_512")
    assert check_nesting(a, SidTable.load("rqkmeans_4codebook_512"))["nested"]
    odd = check_nesting(a, SidTable.load("rqkmeans_5codebook_512"))
    assert not odd["nested"]
    assert odd["agree_rate"] < 0.01, odd
    assert not is_nested_family("rqkmeans_5codebook_512")
    assert is_nested_family("rqkmeans_3codebook_512")


def test_check_nesting_refuses_incomparable_pairs():
    a = SidTable.load("rqkmeans_3codebook_128")
    with pytest.raises(ValueError, match="one codebook size"):
        check_nesting(a, SidTable.load("rqkmeans_4codebook_256"))
    with pytest.raises(ValueError, match="fewer digits"):
        check_nesting(SidTable.load("rqkmeans_4codebook_128"), a)
