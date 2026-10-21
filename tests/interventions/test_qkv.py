"""Query/key/value input hooks (`interventions.qkv`) on a tiny random Qwen2, CPU.

Patching a projection with its own captured value is bit-exact; a patch at one
position moves outputs only from that position on (causality); an observer
that recomputes projections sees the patched values; a capture refuses a
second call; a patch that never fires raises.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sidlens.hooks.attention_probs import attention_probs  # noqa: E402
from sidlens.interventions import qkv  # noqa: E402


@pytest.fixture(scope="module")
def model():
    cfg = transformers.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(6.0)
    return m


def _run(m, ids):
    with torch.no_grad():
        return m(input_ids=ids).logits


IDS = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(1))
ROWS, POS = [0, 1], [5, 7]


@pytest.mark.parametrize("which", ["q", "k", "v"])
def test_self_patch_is_bit_exact(model, which):
    base = _run(model, IDS)
    with qkv.capture(model, which, [1, 2], ROWS, POS) as cap:
        _run(model, IDS)
    with qkv.patch(model, which, cap, ROWS, POS) as fired:
        out = _run(model, IDS)
    assert torch.equal(out, base)
    assert fired == {1: 1, 2: 1}


@pytest.mark.parametrize("which", ["k", "v"])
def test_patch_is_causal(model, which):
    base = _run(model, IDS)
    with qkv.capture(model, which, [1], ROWS, POS) as cap:
        _run(model, IDS)
    vals = {1: torch.zeros_like(cap[1])}
    with qkv.patch(model, which, vals, ROWS, POS):
        out = _run(model, IDS)
    for r, p in zip(ROWS, POS):
        assert torch.equal(out[r, :p], base[r, :p])          # earlier queries cannot read a later key/value
        assert not torch.allclose(out[r, p:], base[r, p:])


def test_observer_sees_patched_query(model):
    with attention_probs(model, [2], ROWS, POS) as plain:
        _run(model, IDS)
    with qkv.capture(model, "q", [2], ROWS, POS) as cap:
        _run(model, IDS)
    with qkv.patch(model, "q", {2: cap[2] * 3.0}, ROWS, POS) as fired, attention_probs(model, [2], ROWS, POS) as obs:
        _run(model, IDS)
    assert fired[2] == 2                                      # the observer's recomputation and the layer's own
    assert not torch.allclose(obs[2], plain[2])


def test_capture_refuses_a_second_call_and_unfired_patch_raises(model):
    with pytest.raises(RuntimeError, match="fired twice"):
        with qkv.capture(model, "k", [1], ROWS, POS), attention_probs(model, [1], ROWS, POS):
            _run(model, IDS)
    with pytest.raises(RuntimeError, match="never fired"):
        with qkv.patch(model, "v", {0: torch.zeros(2, 16)}, ROWS, POS):
            pass


def test_o_input_reconstructs_attention_output(model):
    with qkv.o_input(model, [1], ROWS, POS) as oi:
        _run(model, IDS)
    attn = model.model.layers[1].self_attn
    out = oi[1] @ attn.o_proj.weight.T                       # what o_proj adds to the residual (no bias)
    assert out.shape == (2, 32) and torch.isfinite(out).all()
