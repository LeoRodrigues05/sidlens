"""Residual patching: exact controls on a tiny random Qwen2, CPU only.

The experiment's acceptance rests on three identities that must hold bit for
bit when shapes are fixed: patching a site with its own value changes nothing;
patching every position of a layer with another run's values reproduces that
run downstream; and positions before the first differing token are identical
in both runs (causal masking), so patching them is a no-op. If the mechanism
cannot show these on a toy model, no recovery curve it produces means anything.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sidlens.hooks import capture                                  # noqa: E402
from sidlens.interventions.residual import Patch, patch_residual, run_patched  # noqa: E402

T, B, CHANGED = 12, 3, 5


@pytest.fixture(scope="module")
def model():
    cfg = transformers.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=3, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def inputs():
    g = torch.Generator().manual_seed(1)
    clean = torch.randint(0, 97, (B, T), generator=g)
    corr = clean.clone()
    corr[:, CHANGED] = (clean[:, CHANGED] + 7) % 97
    return clean, corr


def _fwd(model, ids):
    return lambda: model(input_ids=ids).logits


def _layer_out(model, ids, site):
    with capture(model, names=[site], to_cpu=False) as cap:
        with torch.no_grad():
            model(input_ids=ids)
    return cap[site]


def _all(site, src, rows=range(B), pos=range(T)):
    r = torch.tensor([i for i in rows for _ in pos])
    p = torch.tensor([j for _ in rows for j in pos])
    return Patch(site, r, p, src[r, p])


def test_self_patch_is_exact_noop(model, inputs):
    _, corr = inputs
    site = "model.layers.1"
    base = run_patched(model, _fwd(model, corr), [])
    own = _layer_out(model, corr, site)
    out = run_patched(model, _fwd(model, corr), [_all(site, own, pos=[CHANGED, CHANGED + 1])])
    assert torch.equal(out, base)


def test_full_layer_patch_restores_clean_run(model, inputs):
    clean, corr = inputs
    base_clean = run_patched(model, _fwd(model, clean), [])
    for layer in range(3):
        site = f"model.layers.{layer}"
        src = _layer_out(model, clean, site)
        out = run_patched(model, _fwd(model, corr), [_all(site, src)])
        assert torch.equal(out, base_clean), site


def test_positions_before_the_change_are_a_noop(model, inputs):
    clean, corr = inputs
    base = run_patched(model, _fwd(model, corr), [])
    site = "model.layers.2"
    src = _layer_out(model, clean, site)
    out = run_patched(model, _fwd(model, corr), [_all(site, src, pos=range(CHANGED))])
    assert torch.equal(out, base)


def test_patching_the_changed_token_moves_later_logits(model, inputs):
    clean, corr = inputs
    base = run_patched(model, _fwd(model, corr), [])
    site = "model.layers.0"
    src = _layer_out(model, clean, site)
    out = run_patched(model, _fwd(model, corr), [_all(site, src, pos=[CHANGED])])
    assert torch.equal(out[:, :CHANGED], base[:, :CHANGED])
    assert not torch.equal(out[:, CHANGED:], base[:, CHANGED:])


def test_hooks_are_removed_after_the_context(model, inputs):
    clean, corr = inputs
    base = run_patched(model, _fwd(model, corr), [])
    src = _layer_out(model, clean, "model.layers.0")
    run_patched(model, _fwd(model, corr), [_all("model.layers.0", src)])
    assert torch.equal(run_patched(model, _fwd(model, corr), []), base)


def test_a_patch_that_never_fires_raises(model, inputs):
    src = torch.zeros(1, 32)
    with pytest.raises(RuntimeError, match="silent no-op"):
        with patch_residual(model, [Patch("model.layers.0", torch.tensor([0]),
                                          torch.tensor([0]), src)]):
            pass                                     # no forward inside


def test_rejects_non_residual_sites_duplicates_and_bad_shapes(model, inputs):
    _, corr = inputs
    v = torch.zeros(1, 32)
    with pytest.raises(KeyError):
        with patch_residual(model, [Patch("model.layers.0.self_attn", torch.tensor([0]),
                                          torch.tensor([0]), v)]):
            pass
    dup = [Patch("model.layers.0", torch.tensor([0]), torch.tensor([3]), v)] * 2
    with pytest.raises(ValueError, match="same"):
        with patch_residual(model, dup):
            pass
    wide = Patch("model.layers.0", torch.tensor([0]), torch.tensor([3]), torch.zeros(1, 31))
    with pytest.raises(ValueError, match="width"):
        run_patched(model, _fwd(model, corr), [wide])
    far = Patch("model.layers.0", torch.tensor([0]), torch.tensor([T + 4]), v)
    with pytest.raises(IndexError):
        run_patched(model, _fwd(model, corr), [far])
