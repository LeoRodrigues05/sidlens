"""Knockout inside generate(): an empty plan changes nothing; a real plan does; guards fire.

Tiny random Qwen2, CPU, left-padded beam search. The empty-plan run is the
baseline every decoding condition is compared with, so it must equal plain
`generate()` token for token and score for score.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sidlens.interventions.generation import generation_knockout          # noqa: E402

V, B, BEAMS, NEW = 61, 2, 3, 4


@pytest.fixture(scope="module")
def setup():
    cfg = transformers.Qwen2Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   pad_token_id=0, eos_token_id=1, bos_token_id=2)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, V, (B, 9), generator=g)
    am = torch.ones(B, 9, dtype=torch.long)
    ids[1, :2], am[1, :2] = 0, 0                                     # left padding on row 1
    gc = transformers.GenerationConfig(num_beams=BEAMS, num_return_sequences=BEAMS, max_new_tokens=NEW,
                                       min_new_tokens=NEW, length_penalty=0.0, pad_token_id=0,
                                       eos_token_id=1, do_sample=False)
    return model, ids, am, gc


def gen(model, ids, am, gc):
    with torch.no_grad():
        o = model.generate(ids, attention_mask=am, generation_config=gc, return_dict_in_generate=True,
                           output_scores=True)
    return o.sequences, o.sequences_scores


def test_empty_plan_equals_plain_generate(setup):
    model, ids, am, gc = setup
    s0, sc0 = gen(model, ids, am, gc)
    with generation_knockout(model, [1, 2, 3], lambda *a: [], prompt_len=9) as st:
        s1, sc1 = gen(model, ids, am, gc)
    assert torch.equal(s0, s1) and torch.equal(sc0, sc1)
    assert st["forwards"] == NEW and st["edges"] == 0
    assert torch.equal(gen(model, ids, am, gc)[0], s0)             # hooks removed afterwards


def test_real_plan_changes_decoding(setup):
    model, ids, am, gc = setup
    s0, sc0 = gen(model, ids, am, gc)

    def plan(seqs, step, q, kv):                                     # every row: last query stops reading keys 3-5
        return [(r, q - 1, [3, 4, 5]) for r in range(seqs.shape[0])]
    with generation_knockout(model, [0, 1, 2, 3], plan, prompt_len=9) as st:
        s1, sc1 = gen(model, ids, am, gc)
    assert st["edges"] == NEW * B * BEAMS * 3
    assert not torch.equal(sc0, sc1)


def test_guards(setup):
    model, ids, am, gc = setup
    with pytest.raises(ValueError, match="already"):                 # key 0 is padding on row 1's beams
        with generation_knockout(model, [2], lambda seqs, step, q, kv: [(BEAMS, q - 1, [0])], prompt_len=9):
            gen(model, ids, am, gc)
    with pytest.raises(IndexError):
        with generation_knockout(model, [7], lambda *a: [], prompt_len=9):
            pass
