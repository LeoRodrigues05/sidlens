"""DiffGRM cross-attention knockout on the real rqkmeans 3x128 checkpoint (CPU).

The knockout is only a knockout if (a) an all-ones mask reproduces the vendor's
unmasked decoder bit for bit, (b) blocking a real history slot moves the
queried digit, (c) blocking a padded slot is refused (padded slots are
attended, so that would not remove an item), and (d) a hook that never fires
raises.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

CELL = "diff-next1-rqkmeans-3cb-128"


@pytest.fixture(scope="module")
def setup():
    entry = load_runtime()[CELL]
    cohort = load_eval_cohort(entry)
    model, _ = diffusion.load(entry, device="cpu")
    rows = [i for i in range(len(cohort)) if cohort.history_lengths[i] < cohort.histories.shape[1]][:4]
    h = torch.from_numpy(cohort.histories[rows])
    m = torch.from_numpy(cohort.history_mask[rows])
    enc = diffusion.encode(model, h, m)
    return model, cohort, rows, enc, DI.project_cross_cache(model, enc), m.numpy()


def _score(model, enc, cache, xmask=None):
    n = int(model.n_digit)
    R = enc.shape[0]
    with torch.inference_mode():
        return DI.score_states(model, enc, cache, torch.arange(R), torch.zeros((R, n), dtype=torch.long),
                               torch.ones((R, n), dtype=torch.bool), xmask=xmask)


def test_all_ones_mask_is_exactly_the_vendor_decoder(setup):
    model, _, _, enc, cache, _ = setup
    L, n, S = len(model.decoder_blocks), int(model.n_digit), enc.shape[1]
    assert torch.equal(_score(model, enc, cache), _score(model, enc, cache, torch.ones((L, enc.shape[0], n, S))))


def test_blocking_the_most_recent_item_moves_the_queried_digit(setup):
    model, cohort, rows, enc, cache, hmask = setup
    L, n, S = len(model.decoder_blocks), int(model.n_digit), enc.shape[1]
    base = _score(model, enc, cache)
    edges = [(r, range(L), [0], [int(cohort.history_lengths[i]) - 1], r) for r, i in enumerate(rows)]
    xm = torch.from_numpy(DI.knockout_mask(L, len(rows), n, S, edges, hmask))
    out = _score(model, enc, cache, xm)
    assert not torch.equal(out[:, 0], base[:, 0])


def test_padding_slots_are_refused_and_unfired_hooks_raise(setup):
    model, cohort, rows, enc, cache, hmask = setup
    L, n, S = len(model.decoder_blocks), int(model.n_digit), enc.shape[1]
    pad = int(cohort.history_lengths[rows[0]])                 # first padded slot of row 0
    with pytest.raises(ValueError, match="padded"):
        DI.knockout_mask(L, len(rows), n, S, [(0, range(L), [0], [pad], 0)], hmask)
    with pytest.raises(RuntimeError, match="fired"):
        with DI.cross_attention_masks(model, torch.ones((L, len(rows), n, S))):
            pass


def test_encoder_mask_expanded_padding_is_exact_and_knockout_moves_digits(setup):
    model, cohort, rows, enc, cache, hmask = setup
    h = torch.from_numpy(cohort.histories[rows])
    m = torch.from_numpy(cohort.history_mask[rows])
    noop = torch.from_numpy(DI.encoder_block_mask(hmask, []))
    with DI.encoder_attention_masks(model, noop):
        enc2 = diffusion.encode(model, h, m)
    assert torch.equal(enc2, enc)
    k = [(r, int(cohort.history_lengths[i]) - 1) for r, i in enumerate(rows)]
    with DI.encoder_attention_masks(model, torch.from_numpy(DI.encoder_block_mask(hmask, k))):
        enc3 = diffusion.encode(model, h, m)
    assert not torch.equal(enc3, enc)
    last = [int(cohort.history_lengths[i]) - 1 for i in rows]
    pad = int(cohort.history_lengths[rows[0]])
    with pytest.raises(ValueError, match="padded"):
        DI.encoder_block_mask(hmask, [(0, pad)])
    bad = torch.ones_like(noop)                                  # re-opens padding
    with pytest.raises(ValueError, match="re-opens"):
        with DI.encoder_attention_masks(model, bad):
            diffusion.encode(model, h, m)
    assert last  # rows have real history
