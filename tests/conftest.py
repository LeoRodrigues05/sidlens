"""Fixtures shared across test directories.

Only one lives here: a random-weight vendored DIFF_GRM. The mechanics of
tracing, patching and masking (exact no-ops, hooks that must fire, readouts
that must equal the model's own) do not depend on trained weights, so they are
tested on every machine, including one that cannot reach `$SIDLENS_WORK`. What
trained weights add (that an intervention moves a real prediction) stays in
the tests that load a frozen checkpoint.
"""

from types import SimpleNamespace

import pytest


@pytest.fixture(scope="session")
def random_diffgrm():
    """Factory: an untrained vendored DIFF_GRM with encoder states of random histories.

    Weights are scaled up from the vendor's init (std 0.02), which gives almost
    flat logits and uniform attention; a mask or patch must be able to move an
    output for "it moved" to be a test. Every history has real items and
    padding, so padding guards have something to refuse.
    """
    torch = pytest.importorskip("torch")
    from sidlens.models import diffusion
    from sidlens.registry.diffusion import DiffGRMConfig

    built = {}

    def make(n_digit=3, codebook_size=6, seed=0, batch=6, scale=6.0):
        key = (n_digit, codebook_size, seed, batch, scale)
        if key not in built:
            config = DiffGRMConfig(
                vocab_size=3 + n_digit * codebook_size, n_embd=32, n_head=4, n_inner=64, n_digit=n_digit,
                codebook_size=codebook_size, n_target_items=1, n_target_digits=n_digit, encoder_n_layer=1,
                decoder_n_layer=4, max_history_len=50, dropout=0.1, layer_norm_eps=1e-5,
                share_decoder_output_embedding=True, sid_offset=3)
            torch.manual_seed(seed)
            model = diffusion.build_model(config).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.mul_(scale)
            generator = torch.Generator().manual_seed(seed + 1)
            histories = torch.randint(0, codebook_size, (batch, 50, n_digit), generator=generator)
            lengths = torch.randint(3, 40, (batch,), generator=generator)
            mask = torch.arange(50)[None, :] < lengths[:, None]
            histories[~mask] = -1
            built[key] = SimpleNamespace(model=model, histories=histories, mask=mask, lengths=lengths,
                                         enc=diffusion.encode(model, histories, mask))
        return built[key]

    return make
