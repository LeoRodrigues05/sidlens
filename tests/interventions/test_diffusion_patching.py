"""DiffGRM patching, self-attention knockout and step switch: exact controls (CPU).

Run on a random-weight vendored DIFF_GRM (`conftest.random_diffgrm`), so they
hold wherever the code is. An intervention's estimate means nothing unless:
patching a site with its own value changes nothing; patching the last block
with another run's values reproduces that run, position by position; an
all-ones mask is the vendor's unmasked decoder; a switch that switches nothing
is the baseline decode and one that switches everything is the donor's decode,
bit for bit; and a hook that cannot fire, or a slot swap that swaps nothing,
raises rather than reading as "no effect".
"""

import pytest

torch = pytest.importorskip("torch")

from sidlens.analysis.diffusion_lens import decoder_residuals        # noqa: E402
from sidlens.analysis.matched_decode import decode                   # noqa: E402
from sidlens.interventions import diffusion as DI                    # noqa: E402
from sidlens.interventions.residual import Patch, patch_residual     # noqa: E402
from sidlens.models import diffusion                                 # noqa: E402

K = 6


@pytest.fixture(scope="module")
def setup(random_diffgrm):
    s = random_diffgrm(n_digit=3, codebook_size=K, seed=0, batch=6)
    rows = torch.arange(len(s.lengths))
    donor_histories = s.histories.clone()                            # most recent item replaced
    donor_histories[rows, s.lengths - 1] = (s.histories[rows, s.lengths - 1] + 1) % K
    enc_donor = diffusion.encode(s.model, donor_histories, s.mask)
    partial = torch.full((len(rows), 3), -1, dtype=torch.long)
    partial[:, 0] = rows % K                                         # digit 0 revealed, 1 and 2 masked
    return s.model, s, rows, enc_donor, partial


def _score(model, enc, rows, partial, **kwargs):
    with torch.inference_mode():
        return DI.score_states(model, enc, DI.project_cross_cache(model, enc), rows, partial.clamp_min(0),
                               partial < 0, **kwargs)


def _whole_block(layer, residuals, rows, positions=range(3)):
    r = torch.tensor([int(i) for i in rows for _ in positions])
    p = torch.tensor([j for _ in rows for j in positions])
    return Patch(f"decoder_blocks.{layer}", r, p, residuals[layer][r, p])


def test_self_patch_of_every_decoder_block_is_exact(setup):
    model, s, rows, _, partial = setup
    base = _score(model, s.enc, rows, partial)
    seen, residuals = decoder_residuals(model, s.enc, DI.project_cross_cache(model, s.enc), partial, rows)
    assert torch.equal(seen, base)
    assert residuals.shape == (4, len(rows), 3, 32)
    for layer in range(4):
        assert torch.equal(_score(model, s.enc, rows, partial, patches=[_whole_block(layer, residuals, rows)]), base)


def test_noop_cross_mask_self_mask_and_self_patch_compose_exactly(setup):
    model, s, rows, _, partial = setup
    R, S = len(rows), s.enc.shape[1]
    base = _score(model, s.enc, rows, partial)
    _, residuals = decoder_residuals(model, s.enc, DI.project_cross_cache(model, s.enc), partial, rows)
    together = _score(model, s.enc, rows, partial, xmask=torch.ones((4, R, 3, S)),
                      smask=torch.ones((4, R, 3, 3)), patches=[_whole_block(1, residuals, rows)])
    assert torch.equal(together, base)
    slot = [(r, range(4), [1], [int(s.lengths[r]) - 1], r) for r in range(R)]
    blocked = torch.from_numpy(DI.knockout_mask(4, R, 3, S, slot, s.mask.numpy()))
    assert not torch.equal(_score(model, s.enc, rows, partial, xmask=blocked)[:, 1], base[:, 1])


def test_last_block_patch_reproduces_the_donor_run_position_by_position(setup):
    model, s, rows, enc_donor, partial = setup
    base = _score(model, s.enc, rows, partial)
    donor, donor_residuals = decoder_residuals(model, enc_donor, DI.project_cross_cache(model, enc_donor),
                                               partial, rows)
    assert not torch.equal(donor, base)
    full = _score(model, s.enc, rows, partial, patches=[_whole_block(3, donor_residuals, rows)])
    assert torch.equal(full, donor)
    one = _score(model, s.enc, rows, partial, patches=[_whole_block(3, donor_residuals, rows, positions=[1])])
    assert torch.equal(one[:, 1], donor[:, 1])                       # the patched position is the donor's
    assert torch.equal(one[:, [0, 2]], base[:, [0, 2]])              # the others never read it
    early = _score(model, s.enc, rows, partial, patches=[_whole_block(0, donor_residuals, rows, positions=[0])])
    assert not torch.equal(early[:, 1], base[:, 1])                  # an early patch reaches other digits


def test_encoder_sites_are_refused_and_a_patch_that_cannot_fire_raises(setup):
    model, s, rows, _, partial = setup
    values = torch.zeros((len(rows), 32))
    with pytest.raises(ValueError, match="never fire"):
        _score(model, s.enc, rows, partial, patches=[Patch("encoder_blocks.0", rows, rows * 0, values)])
    with pytest.raises(RuntimeError, match="fired"):
        with patch_residual(model, [Patch("decoder_blocks.2", rows, rows * 0, values)]):
            pass


def test_all_ones_self_attention_mask_is_exact_and_a_last_block_edge_moves_only_its_reader(setup):
    model, s, rows, _, partial = setup
    R = len(rows)
    base = _score(model, s.enc, rows, partial)
    assert torch.equal(_score(model, s.enc, rows, partial, smask=torch.ones((4, R, 3, 3))), base)
    assert torch.equal(_score(model, s.enc, rows, partial, smask=torch.ones((4, R, 4, 3, 3))), base)
    edges = [(r, [3], [1], [0]) for r in range(R)]                   # last block: digit 1 stops reading digit 0
    last = _score(model, s.enc, rows, partial, smask=torch.from_numpy(DI.self_knockout_mask(4, R, 3, edges)))
    assert not torch.equal(last[:, 1], base[:, 1])
    assert torch.equal(last[:, [0, 2]], base[:, [0, 2]])
    every = [(r, range(4), [1], [0]) for r in range(R)]
    all_blocks = _score(model, s.enc, rows, partial, smask=torch.from_numpy(DI.self_knockout_mask(4, R, 3, every)))
    one_head = _score(model, s.enc, rows, partial, smask=torch.from_numpy(
        DI.self_knockout_mask(4, R, 3, [(r, range(4), [1], [0], [0]) for r in range(R)], n_head=4)))
    assert not torch.equal(all_blocks[:, 1], base[:, 1])
    assert not torch.equal(one_head[:, 1], base[:, 1]) and not torch.equal(one_head[:, 1], all_blocks[:, 1])


def test_self_attention_masks_refuse_empty_rows_wrong_shapes_and_unfired_hooks(setup):
    model, s, rows, _, partial = setup
    R = len(rows)
    with pytest.raises(ValueError, match="no key"):
        DI.self_knockout_mask(4, R, 3, [(0, [1], [2], [0, 1, 2])])
    empty = torch.ones((4, R, 3, 3))
    empty[2, 1, 0] = 0
    with pytest.raises(ValueError, match="every key"):
        _score(model, s.enc, rows, partial, smask=empty)
    with pytest.raises(ValueError, match="heads"):
        _score(model, s.enc, rows, partial, smask=torch.ones((4, R, 2, 3, 3)))
    with pytest.raises(ValueError, match="n_layers"):
        _score(model, s.enc, rows, partial, smask=torch.ones((3, R, 3, 3)))
    with pytest.raises(RuntimeError, match="fired"):
        with DI.decoder_self_attention_masks(model, torch.ones((4, R, 3, 3))):
            pass


def test_mix_encoder_slots_swaps_real_slots_and_refuses_padding(setup):
    model, s, rows, enc_donor, _ = setup
    none = torch.zeros_like(s.mask)
    assert torch.equal(DI.mix_encoder_slots(s.enc, enc_donor, none, s.mask), s.enc)
    assert torch.equal(DI.mix_encoder_slots(s.enc, enc_donor, s.mask, s.mask), enc_donor)
    own = none.clone()
    own[rows, s.lengths - 1] = True                                  # only the replaced item's own slot
    mixed = DI.mix_encoder_slots(s.enc, enc_donor, own, s.mask)
    assert torch.equal(mixed[own], enc_donor[own]) and torch.equal(mixed[~own], s.enc[~own])
    assert not torch.equal(mixed, s.enc) and not torch.equal(mixed, enc_donor)
    padded = none.clone()
    padded[0, int(s.lengths[0])] = True
    with pytest.raises(ValueError, match="padded"):
        DI.mix_encoder_slots(s.enc, enc_donor, padded, s.mask)
    shorter = s.mask.clone()
    shorter[0, int(s.lengths[0]) - 1] = False                        # claims a real slot is padding
    with pytest.raises(ValueError, match="nonzero on the padding"):
        DI.mix_encoder_slots(s.enc, enc_donor, none, shorter)
    longer = s.mask.clone()
    longer[0, int(s.lengths[0])] = True                              # claims a padded slot is real
    with pytest.raises(ValueError, match="exactly zero on a real slot"):
        DI.mix_encoder_slots(s.enc, enc_donor, none, longer)


@pytest.mark.parametrize("beam,chunk", [(1, 256), (16, 5)])
def test_history_switch_is_the_baseline_when_off_and_the_donor_when_on(setup, beam, chunk):
    model, s, rows, enc_donor, _ = setup
    options = {"beam_width": beam, "policy": "confidence", "decoder_chunk_size": chunk}
    base = decode(model, s.enc, **options)
    donor = decode(model, enc_donor, **options)
    assert not torch.equal(base.scores, donor.scores)

    def run(schedule):
        score_fn = DI.history_switch_scorer(model, s.enc, enc_donor, schedule)
        result = decode(model, s.enc, score_fn=score_fn, **options)
        assert all(count >= 1 for count in score_fn.calls.values())
        return result

    off = run(torch.zeros((len(rows), 3), dtype=torch.bool))
    on = run(torch.ones((len(rows), 3), dtype=torch.bool))
    assert torch.equal(off.codes, base.codes) and torch.equal(off.scores, base.scores)
    assert torch.equal(on.codes, donor.codes) and torch.equal(on.scores, donor.scores)
    by_row = torch.zeros((len(rows), 3), dtype=torch.bool)
    by_row[0] = True                                                 # one user switched, the rest untouched
    mixed = run(by_row)
    assert torch.equal(mixed.scores[0], donor.scores[0]) and torch.equal(mixed.scores[1:], base.scores[1:])
    first_step = torch.zeros((len(rows), 3), dtype=torch.bool)
    first_step[:, 0] = True
    single = run(first_step)
    assert not torch.equal(single.scores, base.scores) and not torch.equal(single.scores, donor.scores)


def test_history_switch_refuses_a_schedule_of_the_wrong_shape_or_type(setup):
    model, s, rows, enc_donor, _ = setup
    with pytest.raises(ValueError, match="schedule"):
        DI.history_switch_scorer(model, s.enc, enc_donor, torch.zeros((len(rows), 2), dtype=torch.bool))
    with pytest.raises(ValueError, match="schedule"):
        DI.history_switch_scorer(model, s.enc, enc_donor, torch.zeros((len(rows), 3)))


def _cross_outputs(model, enc, rows, partial):
    from sidlens.hooks import capture
    names = [f"decoder_blocks.{L}.cross_attn" for L in range(4)]
    with capture(model, names=names, to_cpu=False) as cap:
        out = _score(model, enc, rows, partial)
    assert all(cap.call_counts[n] == 1 for n in names)
    return out, [cap[n] for n in names]


def test_cross_attention_patch_is_exact_on_itself_and_is_the_only_route_of_the_history(setup):
    model, s, rows, enc_donor, partial = setup
    base, own = _cross_outputs(model, s.enc, rows, partial)
    donor, theirs = _cross_outputs(model, enc_donor, rows, partial)
    r = torch.tensor([int(i) for i in rows for _ in range(3)])
    p = torch.tensor([j for _ in rows for j in range(3)])
    every_self = {L: (r, p, own[L][r, p]) for L in range(4)}
    assert torch.equal(_score(model, s.enc, rows, partial, cross_patches=every_self), base)
    # Same state, other history: the decoder reads the history only through cross-attention, so
    # the donor's cross-attention outputs in every block and position reproduce the donor run.
    every_donor = {L: (r, p, theirs[L][r, p]) for L in range(4)}
    assert torch.equal(_score(model, s.enc, rows, partial, cross_patches=every_donor), donor)
    one = _score(model, s.enc, rows, partial, cross_patches={3: (rows, rows * 0 + 1, theirs[3][rows, 1])})
    assert not torch.equal(one[:, 1], base[:, 1])
    assert torch.equal(one[:, [0, 2]], base[:, [0, 2]])               # a last-block read stays at its digit


def test_cross_attention_patch_refuses_bad_blocks_duplicates_and_unfired_hooks(setup):
    model, s, rows, _, partial = setup
    values = torch.zeros((len(rows), 32))
    with pytest.raises(ValueError, match="no decoder blocks"):
        _score(model, s.enc, rows, partial, cross_patches={4: (rows, rows * 0, values)})
    with pytest.raises(ValueError, match="same"):
        _score(model, s.enc, rows, partial, cross_patches={0: (rows * 0, rows * 0, values)})
    with pytest.raises(IndexError):
        _score(model, s.enc, rows, partial, cross_patches={0: (rows, rows * 0 + 3, values)})
    with pytest.raises(RuntimeError, match="fired"):
        with DI.patch_cross_attention(model, {2: (rows, rows * 0, values)}):
            pass
