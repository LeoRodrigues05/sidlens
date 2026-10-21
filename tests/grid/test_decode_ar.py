"""The grid AR decoder equals OneDiffRec's `decode_batch`; copy edits behave.

A tiny random Qwen2 with the frozen RQ-KMeans 3x128 tokenizer (real SID tokens,
so OneDiffRec's catalog grammar and its tokenizer checks run unchanged), CPU.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sidlens import paths  # noqa: E402
from sidlens.data import ar_prompts as P  # noqa: E402
from sidlens.data.sids import SidTable  # noqa: E402
from sidlens.grid import decode_ar, v3code  # noqa: E402
from sidlens.models import ar  # noqa: E402

CKPT = "next-item_best"


@pytest.fixture(scope="module")
def setup():
    if not (paths.FROZEN_CKPT / "ar" / CKPT / "tokenizer.json").exists():
        pytest.skip("frozen AR tokenizer not on this machine")
    tok = transformers.AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / CKPT), local_files_only=True)
    vocab = ar.load_vocab(CKPT)
    table = SidTable.load(vocab.variant)
    sids = sorted({"".join(f"<{'abc'[d]}_{c}>" for d, c in enumerate(code)) for code in table.cb2items})[:300]
    groups = {s: [str(i)] for i, s in enumerate(sids)}
    grammar = v3code.ar_eval().CatalogGrammar(tok, groups, 1)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(8.0)                       # sharper, less tie-prone distributions
    hist = [sids[3:8], sids[40:43] + sids[3:5], sids[100:101] + sids[5:7]]
    examples = [P.ArExample(f"e{i}", "next-item", "test", vocab.variant.name, i, f"u{i}", tuple(range(len(h))),
                            tuple(h), (0,), (sids[0],)) for i, h in enumerate(hist)]
    encs = [P.encode(ex, tok, vocab, template="v3") for ex in examples]
    return model, tok, vocab, grammar, encs


def _plain(model, tok, grammar, prompts, **kw):
    return v3code.ar_eval().decode_batch(model, tok, grammar, prompts, **kw)


def test_prompts_equal_onediffrec(setup):
    _, tok, _, _, encs = setup
    for e in encs:
        assert e.input_ids == decode_ar.encode_prompt(tok, e.example.history_sids)


@pytest.mark.parametrize("batch", [1, 3])
def test_equals_vendored_decoder(setup, batch):
    model, tok, _, grammar, encs = setup
    prompts = [e.input_ids for e in encs]
    for s in range(0, len(prompts), batch):
        part = prompts[s:s + batch]
        assert decode_ar.decode(model, tok, grammar, part, num_beams=6, top_k=4) == \
            _plain(model, tok, grammar, part, num_beams=6, top_k=4)


def _edit(setup, condition, alpha=None):
    model, _, vocab, _, encs = setup
    infos = [decode_ar.PromptInfo.from_encoded(e, 3) for e in encs]
    return decode_ar.CopyEdit(condition, infos, vocab, 3, 6, decode_ar.late_layers(model), alpha=alpha)


def _close(a, b, tol=1e-3):
    # The edit route scores with an explicit attention mask, a different kernel from
    # the plain decoder's: equal lists, scores equal up to float noise (V2b on real runs).
    return all([c["sids"] for c in x] == [c["sids"] for c in y] and
               all(abs(c["joint_logprob"] - d["joint_logprob"]) <= tol for c, d in zip(x, y))
               for x, y in zip(a, b))


def test_baseline_edit_route_matches_plain(setup):
    model, tok, _, grammar, encs = setup
    prompts = [e.input_ids for e in encs]
    plain = decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4)
    assert _close(decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4, edit=_edit(setup, "B")), plain)
    one = decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4, edit=_edit(setup, "C_all", alpha=1.0))
    base = decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4, edit=_edit(setup, "B", alpha=1.0))
    assert one == base                                    # alpha = 1 adds exactly nothing on the float route
    assert _close(one, plain)


def test_copy_block_moves_scores_and_alpha_zero_is_the_block(setup):
    model, tok, _, grammar, encs = setup
    prompts = [e.input_ids for e in encs]
    plain = decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4)
    edit = _edit(setup, "C_all")
    blocked = decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4, edit=edit)
    assert edit.counts["generation_edges"] > 0 and edit.counts["rescoring_edges"] > 0
    assert not _close(blocked, plain, tol=1e-6)
    zero = decode_ar.decode(model, tok, grammar, prompts, num_beams=6, top_k=4, edit=_edit(setup, "C_all", alpha=0.0))
    assert _close(zero, blocked, tol=1e-3)


def test_rescoring_edges_follow_the_candidate_prefix(setup):
    _, _, vocab, _, encs = setup
    edit = _edit(setup, "C_later")
    info = edit.infos[0]                               # history = sids[3:8]; item codes share prefixes
    codes = info.codes[0]
    gen = [vocab.id_of(d, c) for d, c in enumerate(codes)]
    edges = dict(edit.rescoring_edges(0, gen))
    for d in (1, 2):
        want = sorted(info.token_pos[(j, d)] for j in range(len(info.codes)) if info.codes[j][:d] == codes[:d])
        assert sorted(edges.get(info.length - 1 + d, [])) == want
    assert info.length - 1 not in edges              # C_later leaves digit 0 alone
