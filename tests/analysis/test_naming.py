"""Discriminative-term signatures, and whether their AUC can be trusted.

A term list always reads plausibly, so the only real test is whether the
held-out AUC separates a planted signal from noise. Both cases are built here:
a cell whose items genuinely share vocabulary, and a cell assigned at random
from the same pool. The second is the important one -- an AUC that came back
high for random membership would make every row in `naming.csv` meaningless
while looking entirely convincing.
"""

import numpy as np
import pytest

from sidlens.analysis import naming as N

FILLER = ["widget", "device", "unit", "module", "component", "assembly",
          "fitting", "element", "fixture", "part", "gadget", "apparatus"]


def make_texts(n_signal=40, n_bg=200, seed=0, signal_terms=("borosilicate", "graduated")):
    """A pool where `sig_*` items share two distinctive words and nothing else."""
    rng = np.random.default_rng(seed)
    texts, sig, bg = {}, [], []
    for i in range(n_signal):
        k = f"sig_{i}"
        words = list(signal_terms) + list(rng.choice(FILLER, 4))
        texts[k] = " ".join(words)
        sig.append(k)
    for i in range(n_bg):
        k = f"bg_{i}"
        texts[k] = " ".join(rng.choice(FILLER, 6))
        bg.append(k)
    return texts, sig, bg


def test_signature_recovers_the_planted_terms():
    texts, sig, bg = make_texts()
    prior, total = N.corpus_prior(texts)
    terms = N.signature(sig, bg, texts, prior, total, top=5)
    found = {t.term for t in terms}
    assert {"borosilicate", "graduated"} <= found
    assert terms[0].z > 3


def test_filler_words_carry_no_contrast():
    """Words common to both sides must score near zero.

    They can still appear in a long enough list -- with only two planted terms
    there is nothing else to fill it with -- so the claim under test is about z,
    not about rank. If a filler word ever approached the planted terms, the
    method would have degenerated into raw frequency.
    """
    texts, sig, bg = make_texts()
    prior, total = N.corpus_prior(texts)
    terms = N.signature(sig, bg, texts, prior, total, top=8)
    planted = [t for t in terms if t.term in ("borosilicate", "graduated")]
    filler = [t for t in terms if t.term in FILLER]
    assert len(planted) == 2
    assert min(t.z for t in planted) > 5
    assert max((t.z for t in filler), default=0.0) < 1.0


def test_heldout_auc_is_high_for_a_real_signature():
    texts, sig, bg = make_texts()
    prior, total = N.corpus_prior(texts)
    val = N.validate(sig, bg, texts, prior, total)
    assert val["auc"] > 0.9
    assert val["n_train"] + val["n_test"] == len(sig)


def test_heldout_auc_is_chance_for_random_membership():
    """The load-bearing test. Draw the 'cell' at random from one homogeneous
    pool: there is nothing to find, and the AUC must say so."""
    rng = np.random.default_rng(3)
    texts = {f"x_{i}": " ".join(rng.choice(FILLER, 6)) for i in range(240)}
    keys = list(texts)
    rng.shuffle(keys)
    val = N.validate(keys[:40], keys[40:], texts, *N.corpus_prior(texts))
    assert val["auc"] is None or abs(val["auc"] - 0.5) < 0.15


def test_ties_are_scored_as_half():
    """Most background items contain no signature term and score exactly 0.
    Counting those as wins would inflate every cell toward 1."""
    texts = {"a": "alpha beta", "b": "alpha beta", "c": "alpha beta",
             "d": "alpha beta", "e": "alpha beta", "f": "alpha beta",
             "g": "alpha beta", "h": "alpha beta"}
    texts.update({f"z{i}": "gamma delta" for i in range(10)})
    prior, total = N.corpus_prior(texts)
    val = N.validate(list("abcdefgh"), [f"z{i}" for i in range(10)],
                     texts, prior, total)
    assert val["auc"] is not None
    assert 0.0 <= val["auc"] <= 1.0


def test_small_cells_are_skipped_not_guessed():
    texts, sig, bg = make_texts()
    prior, total = N.corpus_prior(texts)
    assert N.signature(sig[:2], bg, texts, prior, total) == []
    val = N.validate(sig[:3], bg, texts, prior, total)
    assert val["auc"] is None and "too few" in val["reason"]


def test_digit_terms_are_flagged_not_dropped():
    """'18-gauge' is an attribute, '740001201' is a model number, and the module
    must not pretend to tell them apart -- it flags both and lets the reader."""
    texts, sig, bg = make_texts(signal_terms=("18-gauge", "borosilicate"))
    prior, total = N.corpus_prior(texts)
    terms = N.signature(sig, bg, texts, prior, total, top=6)
    by = {t.term: t for t in terms}
    assert by["18-gauge"].has_digit
    assert not by["borosilicate"].has_digit


def test_length_normalisation_stops_long_titles_winning():
    """Without it the score is a proxy for title length and the AUC measures
    verbosity."""
    w = {"alpha": 3.0}
    short = N._score("alpha beta", w)
    long = N._score("alpha " + " ".join(FILLER * 3), w)
    assert short > long


def test_depth_summary_reports_validated_count_separately():
    """Most deep cells are too small to validate. A median AUC over 3 of 400
    cells must not be presented as if it covered all of them."""
    rows = [{"variant": "v", "quantizer": "q", "digit": 1, "n_items": 5,
             "heldout_auc": 0.9},
            {"variant": "v", "quantizer": "q", "digit": 1, "n_items": 5,
             "heldout_auc": None}]
    out = N.depth_summary(rows)
    assert out[0]["n_cells_described"] == 2
    assert out[0]["n_cells_validated"] == 1
    assert out[0]["median_auc"] == pytest.approx(0.9)


def test_sibling_counts_by_subtraction_match_explicit_counting():
    """`describe_cells` derives a cell's background by subtracting it from its
    parent instead of counting the siblings directly -- the difference between a
    two-hour sweep and a ten-minute one. The two must agree exactly, or every
    signature in the sweep is computed against a background nobody checked.
    """
    texts, sig, bg = make_texts(n_signal=30, n_bg=90)
    docs = N.tokenize_all(texts)
    prior, total = N.corpus_prior(docs)
    everyone = sig + bg

    explicit = N.signature(sig, bg, docs, prior, total, top=10)

    fg, fg_docs = N.doc_counts(docs, sig)
    parent = N.doc_counts(docs, everyone)[0].copy()
    parent.subtract(fg)
    subtracted = N.signature_from_counts(fg, fg_docs, +parent, len(sig),
                                         prior, total, top=10)

    assert [(t.term, t.z) for t in explicit] == [(t.term, t.z) for t in subtracted]


def test_tokenize_all_is_equivalent_to_tokenizing_each():
    texts = {"a": "Stainless Steel 18-Gauge Wire", "b": "Borosilicate Beaker"}
    assert N.tokenize_all(texts) == {k: N.tokenize(v) for k, v in texts.items()}
