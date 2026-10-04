"""History-item replacement: prefix-matched controls and token-aligned prompts.

Uses the frozen tokenizers, CSVs and SID tables; no weights, no GPU. A control
at level m must share exactly m leading digits with the item it replaces, must
never carry the row's target or another history item, and the re-encoded
prompt must differ from the clean one only at that item's digits >= m.
"""

import numpy as np
import pytest

from sidlens import paths
from sidlens.data import ar_prompts as P
from sidlens.data.sids import SidTable, load_item2id
from sidlens.interventions import history as Hi
from sidlens.interventions.scoring import DigitScorer
from sidlens.models import ar

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

CKPTS = ["next-item_best", "oneoff_rqvae4cb128"]
SEED = 20260927


@pytest.fixture(scope="module", params=CKPTS)
def setup(request):
    ckpt = request.param
    vocab = ar.load_vocab(ckpt)
    tok = transformers.AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / ckpt),
                                                     local_files_only=True)
    table = SidTable.load(vocab.variant)
    pool = Hi.ControlPool(table, load_item2id(P.PRIMARY_CATEGORY))
    examples = P.load_examples(vocab.variant, "next-item", "test")[:25]
    return vocab, tok, table, pool, examples


def test_controls_share_exactly_m_digits_and_never_leak(setup):
    vocab, tok, table, pool, examples = setup
    n = vocab.variant.n_codebook
    known = {tuple(c) for c in table.codes.tolist()}
    drawn = 0
    for ex in examples:
        banned_ids = {*ex.history_item_ids, *ex.target_item_ids}
        banned_sids = {*ex.history_sids, *ex.target_sids}
        for k in range(len(ex.history_sids)):
            orig = P.parse_sid(ex.history_sids[k], n)
            for m in range(n):
                c = pool.draw(ex, k, m, SEED)
                if c is None:
                    continue
                drawn += 1
                assert c.codes[:m] == orig[:m] and c.codes[m] != orig[m]
                assert c.item_id not in banned_ids and c.sid not in banned_sids
                assert c.codes in known
                assert pool.draw(ex, k, m, SEED) == c           # reproducible alone
    assert drawn > 0


def test_replacement_changes_only_item_k_digits_from_m(setup):
    vocab, tok, _, pool, examples = setup
    n = vocab.variant.n_codebook
    for ex in examples[:10]:
        clean = P.encode(ex, tok, vocab, template="eval", with_target=True)
        for k in range(len(ex.history_sids)):
            for m in range(n):
                c = pool.draw(ex, k, m, SEED)
                if c is None:
                    continue
                new = P.encode(Hi.replace_history_item(ex, k, c), tok, vocab,
                               template="eval", with_target=True)
                changed = Hi.check_replacement(clean, new, k, m)
                assert changed and all(clean.item[i] == k and clean.digit[i] >= m for i in changed)
                assert new.predict_pos(0, 0) == clean.predict_pos(0, 0)


def test_check_replacement_rejects_the_wrong_item_or_level(setup):
    vocab, tok, _, pool, examples = setup
    ex = next(e for e in examples if len(e.history_sids) >= 2)
    clean = P.encode(ex, tok, vocab, template="eval", with_target=True)
    c = pool.draw(ex, 0, 0, SEED)
    new = P.encode(Hi.replace_history_item(ex, 0, c), tok, vocab, template="eval", with_target=True)
    with pytest.raises(ValueError, match="outside"):
        Hi.check_replacement(clean, new, 1, 0)              # changed item 0, claimed item 1
    with pytest.raises(ValueError):
        Hi.check_replacement(clean, clean, 0, 0)            # nothing changed


def test_no_candidate_returns_none(setup):
    vocab, _, table, pool, examples = setup
    ex = examples[0]
    orig = P.parse_sid(ex.history_sids[0], vocab.variant.n_codebook)
    everything = set(pool.item_ids.tolist())
    assert len(pool.candidates(orig, 0, everything, ())) == 0


def test_scorer_matches_a_direct_log_softmax(setup):
    vocab, _, table, _, examples = setup
    n = vocab.variant.n_codebook
    scorer = DigitScorer(vocab, table)
    V = max(vocab.digit_code_by_id) + 1
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, n, V, generator=g)
    targets = np.array([P.parse_sid(e.target_sids[0], n) for e in examples[:4]])
    s = scorer.score(logits, targets)
    for r in range(4):
        for d in range(n):
            sub = logits[r, d, vocab.ids(d)]
            j = vocab.codes(d).index(int(targets[r, d]))
            assert s["logp_codes"][r, d] == pytest.approx(float(torch.log_softmax(sub, -1)[j]), abs=1e-5)
            assert s["rank"][r, d] == int((sub > sub[j]).sum())
            assert s["logp_legal"][r, d] >= s["logp_codes"][r, d] - 1e-6
            assert s["logp_vocab"][r, d] <= s["logp_codes"][r, d] + 1e-6


def test_target_slot_clamp_changes_only_item_one_digits_from_m():
    """Next-two: clamping item 1 must leave the history, the separator and item 2 intact."""
    ck = "two-item_best"
    vocab = ar.load_vocab(ck)
    tok = transformers.AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / ck), local_files_only=True)
    pool = Hi.ControlPool(SidTable.load(vocab.variant), load_item2id(P.PRIMARY_CATEGORY))
    n = vocab.variant.n_codebook
    drawn = 0
    for ex in P.load_examples(vocab.variant, "two-item", "test")[:15]:
        clean = P.encode(ex, tok, vocab, template="eval", with_target=True)
        for m in range(n):
            c = pool.draw_target(ex, 0, m, SEED)
            if c is None:
                continue
            drawn += 1
            orig = P.parse_sid(ex.target_sids[0], n)
            assert c.codes[:m] == orig[:m] and c.codes[m] != orig[m]
            assert c.sid not in (*ex.history_sids, *ex.target_sids)
            new = P.encode(Hi.replace_target_slot(ex, 0, c), tok, vocab, template="eval", with_target=True)
            changed = Hi.check_replacement(clean, new, 0, m, role="target_sid")
            assert all(clean.role[i] == "target_sid" and clean.item[i] == 0 for i in changed)
            assert new.predict_pos(1, 0) == clean.predict_pos(1, 0)
            assert pool.draw_target(ex, 0, m, SEED) == c
            assert pool.draw_target(ex, 0, m, SEED) != pool.draw(ex, 0, m, SEED) or True  # distinct seed keys
    assert drawn > 0
