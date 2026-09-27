"""AR prompt reconstruction: byte-equal to upstream, and every position named.

The reference is upstream's own code. The vendored `data.py` is imported by
file path and its `EvalSidDataset` (test-time prompts) and `SidSFTDataset`
(training prompts plus golden target) are run on the same frozen CSVs. Our ids
must equal theirs row for row. The text must equal every archived prediction
file. Tokenizers only: no weights, no GPU.
"""

import importlib.util
import json
import sys

import pytest

from sidlens import paths
from sidlens.data import ar_prompts as P
from sidlens.models import ar

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

CASES = [("next-item_best", "next-item"), ("oneoff_rqvae4cb128", "next-item"),
         ("two-item_best", "two-item")]


@pytest.fixture(scope="module")
def vendor_data():
    """Upstream's data.py, loaded by path without writing bytecode into vendor/."""
    spec = importlib.util.spec_from_file_location(
        "_sidlens_vendor_onediffrec_data", paths.VENDOR / "onediffrec" / "data.py")
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def ckpts():
    out = {}
    for ck, task in CASES:
        vocab = ar.load_vocab(ck)
        tok = transformers.AutoTokenizer.from_pretrained(
            str(paths.FROZEN_CKPT / "ar" / ck), local_files_only=True)
        out[ck] = (vocab, tok, P.load_examples(vocab.variant, task, "test"))
    return out


@pytest.mark.parametrize("ck,task", CASES)
def test_eval_ids_equal_upstream_evaluator(vendor_data, ckpts, ck, task):
    """What evaluate.py fed generate(), for every test row."""
    vocab, tok, exs = ckpts[ck]
    ref = vendor_data.EvalSidDataset(train_file=str(P.csv_path(vocab.variant, task)),
                                     tokenizer=tok, max_len=2560, test=True, seed=42)
    assert len(ref) == len(exs)
    for i, ex in enumerate(exs):
        e = P.encode(ex, tok, vocab, template="eval")
        assert e.input_ids == ref[i]["input_ids"], ex.example_id
        assert e.prompt_len == len(e.input_ids)


@pytest.mark.parametrize("ck,task", CASES)
def test_sft_ids_and_target_equal_upstream_training(vendor_data, ckpts, ck, task):
    """What the SFT loss saw: training wording + golden target + EOS."""
    vocab, tok, exs = ckpts[ck]
    ref = vendor_data.SidSFTDataset(train_file=str(P.csv_path(vocab.variant, task)),
                                    tokenizer=tok, max_len=2048, test=False, seed=42)
    for i, ex in enumerate(exs):
        e = P.encode(ex, tok, vocab, template="sft", with_target=True)
        assert e.input_ids == ref[i]["input_ids"], ex.example_id
        n_masked = sum(1 for x in ref[i]["labels"] if x == -100)
        assert n_masked == e.prompt_len


def test_two_templates_really_differ(ckpts):
    """If this ever fails, upstream unified the wording and trap 1 is gone."""
    vocab, tok, exs = ckpts["next-item_best"]
    a = P.encode(exs[0], tok, vocab, template="eval").input_ids
    b = P.encode(exs[0], tok, vocab, template="sft").input_ids
    assert a != b
    with pytest.raises(ValueError, match="template"):
        P.encode(exs[0], tok, vocab, template="train")


def _archived():
    root = paths.FROZEN_RESULTS / "sweep_metrics"
    out = []
    for task, stem in (("next-item", "nextitem"), ("two-item", "twoitem")):
        for f in sorted((root / task / "metrics").glob(f"{stem}__*.predictions.json")):
            _, _, q, cb, size = f.name.split(".")[0].split("__")
            out.append((task, f"{q}_{cb[:-2]}codebook_{size}"))
    return out


@pytest.mark.parametrize("task,variant", _archived())
def test_row_order_matches_every_archived_prediction_file(task, variant):
    """Archived predictions carry no user id; row order is the only join key."""
    rep = P.check_against_archive(P.load_examples(variant, task, "test"))
    assert rep["ok"], rep


@pytest.mark.parametrize("ck,task", CASES)
def test_csv_sids_match_frozen_tables(ckpts, ck, task):
    vocab, _, exs = ckpts[ck]
    rep = P.check_against_table(exs)
    assert rep["ok"], rep


def test_position_map_names_every_sid_token(ckpts):
    vocab, tok, exs = ckpts["next-item_best"]
    for ex in exs[:200]:
        e = P.encode(ex, tok, vocab, template="eval", with_target=True)
        n = vocab.variant.n_codebook
        for k, sid in enumerate(ex.history_sids):
            pos = e.positions("hist_sid", item=k)
            assert [e.input_ids[p] for p in pos] == vocab.tokens_for(P.parse_sid(sid, n))
        assert e.positions("response_header") == list(range(e.prompt_len - 3, e.prompt_len))
        assert tok.decode([e.input_ids[p] for p in e.positions("response_header")]) \
            == P.RESPONSE_HEADER
        target = P.parse_sid(ex.target_sids[0], n)
        assert e.predict_pos(0, 0) == e.prompt_len - 1
        for d in range(n):
            # the token after the predicting position is the target digit
            assert e.input_ids[e.predict_pos(0, d) + 1] == vocab.id_of(d, target[d])
        assert e.role[-1] == "target_end" and e.input_ids[-1] == tok.eos_token_id


def test_two_item_slots_and_separator(ckpts):
    vocab, tok, exs = ckpts["two-item_best"]
    e = P.encode(exs[0], tok, vocab, template="eval", with_target=True)
    n = vocab.variant.n_codebook
    for slot in (0, 1):
        code = P.parse_sid(exs[0].target_sids[slot], n)
        assert [e.code[p] for p in e.positions("target_sid", item=slot)] == list(code)
    sep = e.positions("target_sep")
    assert sep and max(e.positions("target_sid", item=0)) < min(sep) \
        < min(e.positions("target_sid", item=1))
    assert tok.decode([e.input_ids[p] for p in sep]) == P.TWO_ITEM_SEP


def test_predict_pos_beyond_digit0_needs_the_target(ckpts):
    vocab, tok, exs = ckpts["next-item_best"]
    e = P.encode(exs[0], tok, vocab, template="eval")
    assert e.predict_pos(0, 0) == e.prompt_len - 1
    with pytest.raises(KeyError, match="without target"):
        e.predict_pos(0, 1)


def test_left_padding_positions_match_generate(ckpts):
    """Left-padded rows get the position ids generate() would compute, so a
    plain forward reads every real token at its unpadded RoPE position."""
    vocab, tok, exs = ckpts["next-item_best"]
    batch = [P.encode(ex, tok, vocab, template="eval") for ex in exs[:8]]
    for side in ("left", "right"):
        c = P.collate(batch, tok.pad_token_id, side=side)
        for i, e in enumerate(batch):
            o = int(c["offset"][i])
            real = c["position_ids"][i, o:o + len(e)]
            assert real.tolist() == list(range(len(e)))
            assert c["input_ids"][i, o:o + len(e)].tolist() == e.input_ids


def test_malformed_sids_are_refused():
    assert P.parse_sid("<a_1><b_22><c_3>", 3) == (1, 22, 3)
    for bad in ("<a_1><c_3><b_22>", "<a_1><b_2>", "<a_1> <b_2><c_3>", "a_1b_2c_3"):
        with pytest.raises(ValueError):
            P.parse_sid(bad, 3)


def test_example_ids_are_stable_and_unique(ckpts):
    _, _, exs = ckpts["next-item_best"]
    ids = [e.example_id for e in exs]
    assert len(set(ids)) == len(ids)
    assert ids[5] == "next-item/test/rqkmeans_3codebook_128/5"
    json.dumps(ids[:3])
