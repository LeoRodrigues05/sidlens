"""The grid DiffGRM decoder equals OneDiffRec's `decode_joint_catalog` bit for bit,
and its hooks change the output only when they intervene."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from sidlens.grid import decode_diff, v3code  # noqa: E402
from sidlens.interventions import diffusion as DI  # noqa: E402


def _catalog(n_digit, K, n, seed=3):
    g = torch.Generator().manual_seed(seed)
    allc = list(itertools.product(range(K), repeat=n_digit))
    pick = torch.randperm(len(allc), generator=g)[:n].tolist()
    return [allc[i] for i in pick]


@pytest.mark.parametrize("n_digit", [3, 4])
def test_equals_vendored_decoder(random_diffgrm, n_digit):
    m = random_diffgrm(n_digit=n_digit, codebook_size=6, batch=6)
    catalog = _catalog(n_digit, 6, 60)
    batch = {"history_sid": m.histories, "history_mask": m.mask}
    theirs = v3code.joint_decode().decode_joint_catalog(m.model, batch, catalog, beam_size=12, top_k=7)
    ours = decode_diff.decode(m.model, batch, catalog, beam_size=12, top_k=7)
    assert ours == theirs
    # every returned SID is in the catalog
    assert all(tuple(h["codes"][0]) in set(catalog) for hyps in ours for h in hyps)


def test_precomputed_encoder_states_are_a_noop(random_diffgrm):
    m = random_diffgrm(n_digit=3, codebook_size=6, batch=6)
    catalog = _catalog(3, 6, 60)
    batch = {"history_sid": m.histories, "history_mask": m.mask}
    enc = m.model.forward(batch, return_loss=False).hidden_states
    assert decode_diff.decode(m.model, batch, catalog, beam_size=10, top_k=5, encoded=enc) == \
        decode_diff.decode(m.model, batch, catalog, beam_size=10, top_k=5)


def test_decoder_context_all_ones_is_noop_and_block_moves_output(random_diffgrm):
    m = random_diffgrm(n_digit=3, codebook_size=6, batch=6)
    catalog = _catalog(3, 6, 60)
    batch = {"history_sid": m.histories, "history_mask": m.mask}
    n_layers, S = len(m.model.decoder_blocks), m.histories.shape[1]
    base = decode_diff.decode(m.model, batch, catalog, beam_size=10, top_k=5)

    def ones(owners, column, states):
        return DI.cross_attention_masks(m.model, torch.ones((n_layers, len(states), 3, S)))
    assert decode_diff.decode(m.model, batch, catalog, beam_size=10, top_k=5, decoder_context=ones) == base

    def block_real(owners, column, states):
        mask = torch.ones((n_layers, len(states), 3, S))
        for r, o in enumerate(owners.tolist()):
            mask[:, r, :, :int(m.lengths[o]) - 1] = 0     # keep only the most recent real slot
        return DI.cross_attention_masks(m.model, mask)
    blocked = decode_diff.decode(m.model, batch, catalog, beam_size=10, top_k=5, decoder_context=block_real)
    assert blocked != base


def test_decode_cohort_batches(random_diffgrm):
    m = random_diffgrm(n_digit=3, codebook_size=6, batch=6)
    catalog = _catalog(3, 6, 60)
    whole = decode_diff.decode(m.model, {"history_sid": m.histories, "history_mask": m.mask}, catalog,
                               beam_size=10, top_k=5)
    rows = decode_diff.decode_cohort(m.model, m.histories, m.mask, catalog, beam_size=10, top_k=5,
                                     batch_size=6, device="cpu")
    assert rows == whole


def test_explicit_order_identity_equals_left_to_right_and_others_differ(random_diffgrm):
    m = random_diffgrm(n_digit=3, codebook_size=6, batch=6)
    catalog = _catalog(3, 6, 60)
    batch = {"history_sid": m.histories, "history_mask": m.mask}
    l2r = decode_diff.decode(m.model, batch, catalog, beam_size=8, top_k=5)
    assert decode_diff.decode(m.model, batch, catalog, beam_size=8, top_k=5, reveal_order=[0, 1, 2]) == l2r
    rev = decode_diff.decode(m.model, batch, catalog, beam_size=8, top_k=5, reveal_order=[2, 1, 0])
    assert all(tuple(h["codes"][0]) in set(catalog) for hyps in rev for h in hyps)   # legality follows the order
    with pytest.raises(ValueError, match="permutation"):
        decode_diff.decode(m.model, batch, catalog, beam_size=8, top_k=5, reveal_order=[0, 0, 1])


def test_step_encoded_none_is_plain_and_all_donor_is_donor(random_diffgrm):
    m = random_diffgrm(n_digit=3, codebook_size=6, batch=6)
    catalog = _catalog(3, 6, 60)
    batch = {"history_sid": m.histories, "history_mask": m.mask}
    enc = m.model.forward(batch, return_loss=False).hidden_states
    donor_h = m.histories.roll(1, dims=0)
    donor = m.model.forward({"history_sid": donor_h, "history_mask": m.mask.roll(1, dims=0)},
                            return_loss=False).hidden_states
    kw = dict(beam_size=8, top_k=5)
    plain = decode_diff.decode(m.model, None, catalog, encoded=enc, **kw)
    assert decode_diff.decode(m.model, None, catalog, encoded=enc, step_encoded=lambda c: None, **kw) == plain
    assert decode_diff.decode(m.model, None, catalog, encoded=enc, step_encoded=lambda c: donor, **kw) == \
        decode_diff.decode(m.model, None, catalog, encoded=donor, **kw)
    first = decode_diff.decode(m.model, None, catalog, encoded=enc,
                               step_encoded=lambda c: donor if c == 0 else None, **kw)
    assert first != plain


def test_slot_scaling_alpha_one_is_plain_and_zero_is_the_mask_block(random_diffgrm):
    from sidlens.grid import diff_setup as DS
    m = random_diffgrm(n_digit=3, codebook_size=6, batch=6)
    catalog = _catalog(3, 6, 60)
    hist, mask = m.histories.numpy(), m.mask.numpy()
    L = m.mask.sum(1).numpy()
    slots = torch.as_tensor(L - 1)
    kw = dict(beam_size=8, top_k=5)
    enc1 = DS.encode_slot_scaled(m.model, hist, mask, slots.numpy(), 1.0)
    assert torch.equal(enc1, m.model.forward({"history_sid": m.histories, "history_mask": m.mask},
                                             return_loss=False).hidden_states)
    assert DS.slot_scale_context(m.model, slots, 1.0) is None
    plain = decode_diff.decode(m.model, None, catalog, encoded=enc1, **kw)
    # alpha = 0 (reweighting) against the 0/1-mask block of exp14 (encoder mask + cross-attention mask)
    enc0 = DS.encode_slot_scaled(m.model, hist, mask, slots.numpy(), 0.0)
    em = torch.from_numpy(DI.encoder_block_mask(mask, [(r, int(k)) for r, k in enumerate(slots.tolist())]))
    with DI.encoder_attention_masks(m.model, em):
        enc_blk = m.model.forward({"history_sid": m.histories, "history_mask": m.mask},
                                  return_loss=False).hidden_states
    assert torch.allclose(enc0, enc_blk, atol=1e-5)
    zero = decode_diff.decode(m.model, None, catalog, encoded=enc0,
                              decoder_context=DS.slot_scale_context(m.model, slots, 0.0), **kw)
    blk = decode_diff.decode(m.model, None, catalog, encoded=enc_blk,
                             decoder_context=DS.slot_block_context(m.model, slots, enc_blk.shape[1]), **kw)
    assert [[h["codes"] for h in u] for u in zero] == [[h["codes"] for h in u] for u in blk]
    assert all(abs(a["score"] - b["score"]) < 1e-4 for u, v in zip(zero, blk) for a, b in zip(u, v))
    assert zero != plain
