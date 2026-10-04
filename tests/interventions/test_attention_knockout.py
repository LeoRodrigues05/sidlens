"""Attention-edge knockout and the attention-probability observer on a toy Qwen2 (CPU).

The knockout is only trustworthy if (a) an empty or already-masked knockout
changes nothing, bit for bit; (b) knocking an edge out at every layer equals
running the model on a mask with that edge removed, an independent route to
the same computation; (c) it touches only its own row, head and later
positions. The observer must reproduce eager attention weights.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sidlens.hooks.attention_probs import attention_probs                    # noqa: E402
from sidlens.interventions.attention import Knockout, base_mask, knockout_attention  # noqa: E402

B, T, L, H = 3, 12, 3, 4


def _model(impl):
    cfg = transformers.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=L, num_attention_heads=H,
                                   num_key_value_heads=2, max_position_embeddings=64)
    cfg._attn_implementation = impl
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def model():
    return _model("sdpa")


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 97, (B, T), generator=g)
    am = torch.ones(B, T, dtype=torch.long)
    am[2, T - 3:] = 0                                  # row 2 right-padded
    return ids, am, base_mask(am, H)


def run(model, ids, mask, kos=()):
    with knockout_attention(model, mask, kos):
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask).logits


def test_empty_and_already_masked_knockouts_are_exact_noops(model, batch):
    ids, _, m = batch
    base = run(model, ids, m)
    assert torch.equal(run(model, ids, m, []), base)
    fut = Knockout(0, (4,), (7, 8), tuple(range(L)))              # keys after the query
    noop = Knockout(0, (4,), (7, 8), tuple(range(L)), expect_masked=True)
    assert torch.equal(run(model, ids, m, [noop]), base)
    with pytest.raises(ValueError, match="already masked"):
        run(model, ids, m, [fut])
    with pytest.raises(ValueError, match="live edge"):          # a control must BE a no-op
        run(model, ids, m, [Knockout(0, (4,), (1,), (0,), expect_masked=True)])
    real = Knockout(1, (9,), (3,), (1,))                           # guard not relaxed by the control
    mixed = run(model, ids, m, [noop, real])
    assert torch.equal(mixed[0], base[0]) and not torch.equal(mixed[1], base[1])
    pad = [Knockout(2, (8,), (T - 2,), (0,))]                      # a padding key
    with pytest.raises(ValueError, match="already masked"):
        run(model, ids, m, pad)


def test_all_layer_knockout_equals_a_globally_edited_mask(model, batch):
    ids, _, m = batch
    ko = Knockout(1, (9,), (3, 5), tuple(range(L)))
    edited = m.clone()
    edited[1, :, 9, 3] = False
    edited[1, :, 9, 5] = False
    with torch.no_grad():
        direct = model(input_ids=ids, attention_mask=edited).logits
    assert torch.equal(run(model, ids, m, [ko]), direct)


def test_knockout_is_local_to_row_head_and_later_positions(model, batch):
    ids, _, m = batch
    base = run(model, ids, m)
    out = run(model, ids, m, [Knockout(1, (9,), (3,), (1,))])
    assert torch.equal(out[[0, 2]], base[[0, 2]])                  # other rows untouched
    assert torch.equal(out[1, :9], base[1, :9])                    # earlier positions untouched
    assert not torch.equal(out[1, 9:], base[1, 9:])
    one = run(model, ids, m, [Knockout(1, (9,), (3,), (1,), heads=(2,))])
    assert not torch.equal(one[1, 9:], base[1, 9:]) and not torch.equal(one, out)


def test_guards(model, batch):
    ids, am, m = batch
    with pytest.raises(ValueError, match="self edges"):
        Knockout(0, (4,), (4,), (0,))
    other = base_mask(am, H)                                       # equal values, other object
    with pytest.raises(RuntimeError, match="different attention mask"):
        with knockout_attention(model, m, []):
            with torch.no_grad():
                model(input_ids=ids, attention_mask=other)
    with pytest.raises(RuntimeError, match="fired"):
        with knockout_attention(model, m, []):
            pass
    with pytest.raises(IndexError):
        run(model, ids, m, [Knockout(0, (4,), (1,), (L,))])


def test_observer_reproduces_eager_attention_weights(batch):
    ids, am, m = batch
    eager = _model("eager")
    rows, qs = [0, 1, 2, 2], [11, 9, 8, 3]
    with attention_probs(eager, range(L), rows, qs) as probs:
        with torch.no_grad():
            ref = eager(input_ids=ids, attention_mask=am, output_attentions=True).attentions
    for layer in range(L):
        want = ref[layer][torch.tensor(rows), :, torch.tensor(qs)]     # (P, H, T)
        assert torch.allclose(probs[layer], want, atol=1e-5), layer
    sd = _model("sdpa")
    with knockout_attention(sd, m, []), attention_probs(sd, [1], rows, qs) as p2:
        with torch.no_grad():
            sd(input_ids=ids, attention_mask=m)
    want = ref[1][torch.tensor(rows), :, torch.tensor(qs)]
    assert torch.allclose(p2[1], want, atol=1e-5)
