"""The AR checkpoint's SID vocabulary -- the token-id trap, specifically.

`<a_N>` tokens occupy a contiguous id block, which invites `token_id = base + N`.
The ids inside a block are ordered by a plain string sort, so `<a_100>` lands at
base+1 and `<a_10>` at base+11. The shortcut is off by a per-code amount, raises
nothing, and returns logits for the wrong codeword. These tests pin the real
mapping and prove the shortcut is wrong, so nobody reintroduces it.

These read only tokenizer JSON and the frozen SID tables -- no weights, no GPU.
"""

import json

import numpy as np
import pytest

from sidlens import paths
from sidlens.data.sids import SidTable
from sidlens.models import ar

CKPT = "next-item_best"


@pytest.fixture(scope="module")
def vocab():
    return ar.load_vocab(CKPT)


def test_every_checkpoint_matches_the_sid_table_it_claims():
    """The tokenizer and the frozen .sem_ids are independent artifacts. If a
    code used in the table has no token, the model cannot address that item and
    generation silently decodes to something else."""
    for ckpt in ar.AR_CHECKPOINTS:
        rep = ar.validate_against(ar.load_vocab(ckpt))
        assert rep["ok"], rep


def test_contiguous_block_arithmetic_would_be_wrong(vocab):
    """The load-bearing test. If this ever passes, the ids became code-ordered
    and the abstraction is no longer earning its keep -- but until then, any
    `base + code` shortcut is a silent bug."""
    ids = sorted(vocab.id_by_digit_code[(0, c)] for c in vocab.codes(0))
    base = ids[0]
    naive = {c: base + c for c in vocab.codes(0)}
    real = {c: vocab.id_of(0, c) for c in vocab.codes(0)}
    assert naive != real
    wrong = sum(1 for c in real if naive[c] != real[c])
    assert wrong > len(real) // 2   # most codes, not an edge case


def test_ids_form_one_contiguous_block_per_digit(vocab):
    """Contiguity is real -- it is only the ORDER inside the block that differs.
    Pinning both stops a future reader from concluding the layout is arbitrary."""
    for d in vocab.digits:
        ids = sorted(vocab.id_by_digit_code[(d, c)] for c in vocab.codes(d))
        assert ids == list(range(ids[0], ids[0] + len(ids)))


def test_ids_are_returned_in_code_order(vocab):
    """`vocab.ids(d)` must be indexable by position in `codes(d)`; slicing the
    raw id block instead yields string-sort order."""
    codes = vocab.codes(0)
    ids = vocab.ids(0)
    assert len(ids) == len(codes)
    for i, c in enumerate(codes):
        assert ids[i] == vocab.id_of(0, c)
    assert not np.array_equal(ids, np.sort(ids))   # code order != id order


def test_round_trip_between_code_and_token_id(vocab):
    for d in vocab.digits:
        for c in vocab.codes(d)[:20]:
            assert vocab.code_of(vocab.id_of(d, c)) == (d, c)


def test_unused_codes_are_absent_and_say_so():
    """rqvae 4cb x 128 collapsed its first digit to 39 used codes, so the model's
    decision at digit 0 is over 39 options, not 128. Anything assuming
    `codebook_size` alternatives is wrong by a factor of three there."""
    v = ar.load_vocab("oneoff_rqvae4cb128")
    assert v.n_codes(0) == 39
    assert v.variant.codebook_size == 128
    for d in (1, 2, 3):
        assert v.n_codes(d) == 128
    missing = next(c for c in range(128) if (0, c) not in v.id_by_digit_code)
    with pytest.raises(KeyError, match="no token in this checkpoint"):
        v.id_of(0, missing)


def test_tokens_for_a_real_item_match_its_sid(vocab):
    table = SidTable.load("rqkmeans_3codebook_128")
    asin = table.keys[0]
    code = table.asin2codes[asin]
    ids = vocab.tokens_for(code)
    assert [vocab.code_of(i) for i in ids] == list(enumerate(code))


def test_digit_logits_selects_the_right_columns(vocab):
    """A fake logit vector where each SID token's value encodes its (digit,code)
    -- so a mis-selection is visible as a wrong number, not just a wrong shape."""
    n_vocab = max(vocab.digit_code_by_id) + 1
    logits = np.zeros(n_vocab)
    for (d, c), tid in vocab.id_by_digit_code.items():
        logits[tid] = d * 1000 + c
    for d in vocab.digits:
        got = ar.digit_logits(logits, vocab, d)
        want = np.array([d * 1000 + c for c in vocab.codes(d)], dtype=float)
        assert np.array_equal(got, want)


def test_matched_cells_name_real_checkpoints():
    """RQ1's paradigm comparison is only defined where an AR and a diffusion
    checkpoint share a SID. Both sides must actually exist."""
    reg = json.loads((paths.MANIFESTS / "registry.diffusion.json").read_text())
    for variant, (ar_ckpt, diff_id) in ar.MATCHED_CELLS.items():
        assert ar.AR_CHECKPOINTS[ar_ckpt] == variant
        assert diff_id in reg
        assert reg[diff_id]["sem_ids_name"] == variant
        assert reg[diff_id]["status"] == "trained"


def test_two_item_checkpoint_has_no_diffusion_match():
    """Documents the gap that blocks RQ4: the AR two-item run is on MQ 4cb x 256
    and every next2 diffusion run is rqvae. Without a retrain there is no
    paradigm comparison for the two-item task at all."""
    assert ar.AR_CHECKPOINTS["two-item_best"] == "MQ_4codebook_256"
    assert "MQ_4codebook_256" not in ar.MATCHED_CELLS
    reg = json.loads((paths.MANIFESTS / "registry.diffusion.json").read_text())
    next2 = {e["sem_ids_name"] for e in reg.values() if e["task"] == "next2"}
    assert "MQ_4codebook_256" not in next2


def test_unknown_checkpoint_raises():
    with pytest.raises(KeyError, match="unknown AR checkpoint"):
        ar.load_vocab("does-not-exist")
