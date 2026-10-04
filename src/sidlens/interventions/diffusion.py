"""DiffGRM (next-item) interventions: digit scores at chosen decoder states, and
cross-attention knockout from digit positions to history items.

What a "state" is
-----------------
The DiffGRM decoder has one position per SID digit and no positional code:
a masked digit carries its digit's mask embedding, a revealed digit carries
its code's embedding. Self-attention over the digit positions is full
(bidirectional), and every decoder block cross-attends to the encoder's
per-ITEM states (one token per history item: its digit embeddings
concatenated through `item_mlp`, plus a recency position). So a state is a
(revealed codes, mask) pair, and the diffusion analogue of the AR copy link
is "digit-d position -> one history ITEM slot", not "-> a history digit token".

Traps, each with the guard that closes it
-----------------------------------------
1. **The unvalidated logits path.** `models.diffusion.digit_logits` runs with
   `use_cache=False`, which silently replaces the learned cross-attention K/V
   projections with raw encoder states. `score_states` always supplies the
   projected cross cache exactly as `analysis.matched_decode` does (fresh
   self-attention, reused projected K/V, explicit encoder-row mapping), the
   path exp1 validated against the archive.
2. **A knockout that does not reach the kernel.** The vendor's decoder blocks
   call `cross_attn(...)` WITHOUT an attention mask, so padded history slots
   (zero states) take part in every softmax. The knockout injects the
   `attention_mask` keyword through a pre-hook; if a future vendor version
   passes its own mask, the hook raises instead of overwriting it. Every
   forward in every condition gets a mask (all ones when nothing is knocked
   out), which is an exact no-op on the vendor math (`masked_fill` with no
   zero), so conditions share one code path.
3. **Silent no-ops.** Knocking out a slot that is padding would still change
   the softmax (padding keys are attended), so it is refused rather than
   passed off as "removing an item". Each block must fire exactly once.
4. **Row alignment.** Decoder rows name their encoder row explicitly
   (`enc_rows`), never by slicing a repeated buffer.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def project_cross_cache(model, encoder_hidden):
    """Per-block projected cross-attention (K, V), as `matched_decode` builds them."""
    width = model.n_embd
    cache = []
    for block in model.decoder_blocks:
        projected = block.cross_attn.qkv(encoder_hidden)
        cache.append((projected[..., width:2 * width], projected[..., 2 * width:]))
    return cache


def score_states(model, encoder_hidden, cross_cache, enc_rows, codes, masked, xmask=None):
    """Log-softmax digit distributions [R, n_digit, K] at R decoder states.

    enc_rows  [R] long: encoder row of each decoder row
    codes     [R, n_digit] long: revealed codes (any value where masked)
    masked    [R, n_digit] bool: True = still masked
    xmask     optional [n_layers, R, n_digit, S] cross-attention masks (1 keep, 0 block)
    """
    import torch

    rows = enc_rows
    past = [(None, (k.index_select(0, rows), v.index_select(0, rows))) for k, v in cross_cache]
    ctx = cross_attention_masks(model, xmask) if xmask is not None else _null()
    with ctx:
        out = model.forward_decoder_only(
            {"decoder_input_ids": codes.clamp_min(0), "mask_positions": masked.to(torch.float32),
             "encoder_hidden": encoder_hidden.index_select(0, rows)},
            return_loss=False, digit=None, past_key_values=past, use_cache=True)
    logits = out.logits
    if tuple(logits.shape) != (len(rows), int(model.n_digit), int(model.codebook_size)):
        raise ValueError(f"unexpected decoder logits shape {tuple(logits.shape)}")
    return torch.log_softmax(logits.float(), dim=-1)


@contextmanager
def _null():
    yield None


@contextmanager
def cross_attention_masks(model, xmask) -> Iterator[dict]:
    """Inject per-block cross-attention masks [n_layers, R, n_digit, S] (1 keep, 0 block)."""
    blocks = list(model.decoder_blocks)
    if xmask.shape[0] != len(blocks):
        raise ValueError(f"need one mask per decoder block ({len(blocks)}), got {xmask.shape[0]}")
    fired = {L: 0 for L in range(len(blocks))}
    handles = []

    def make_hook(L):
        def hook(_mod, args, kwargs):
            fired[L] += 1
            if kwargs.get("attention_mask") is not None:
                raise RuntimeError(f"decoder block {L} cross-attention already receives a mask; "
                                   f"the vendor changed and the knockout would overwrite it")
            q = args[0]
            m = xmask[L]
            if m.shape[0] != q.shape[0] or m.shape[1] != q.shape[1]:
                raise ValueError(f"block {L}: mask {tuple(m.shape)} does not match queries {tuple(q.shape)}")
            return args, {**kwargs, "attention_mask": m}
        return hook

    try:
        for L, blk in enumerate(blocks):
            handles.append(blk.cross_attn.register_forward_pre_hook(make_hook(L), with_kwargs=True))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := {L: n for L, n in fired.items() if n != 1}:
        raise RuntimeError(f"cross-attention blocks fired {bad} times, expected once each")


def knockout_mask(n_layers: int, n_rows: int, n_digit: int, n_slots: int, edges, history_mask):
    """[n_layers, R, n_digit, S] float32 numpy masks (1 keep, 0 block).

    edges: (row, layers, digit_queries, slots, enc_row). Refuses padding slots:
    a padded slot is attended (zero key), so removing it changes the softmax
    without removing any item. `history_mask` is the [N_enc, S] bool array of
    the encoder rows the decoder rows point at.
    """
    import numpy as np

    m = np.ones((n_layers, n_rows, n_digit, n_slots), dtype=np.float32)
    for row, layers, queries, slots, enc_row in edges:
        if not len(slots):
            continue
        if not bool(np.asarray(history_mask[enc_row])[list(slots)].all()):
            raise ValueError(f"row {row}: knockout of a padded history slot {list(slots)}")
        m[np.ix_(list(layers), [row], list(queries), list(slots))] = 0
    return m


@contextmanager
def encoder_attention_masks(model, mask) -> Iterator[dict]:
    """Replace the encoder self-attention mask with `mask` [B, 1, S, S] (1 keep, 0 block).

    The vendor passes a [B, 1, S, S] (expanded) padding mask to every encoder block. The
    replacement must keep every padded key masked (checked per call), so it
    only ADDS blocked (query slot, key slot) pairs; a mask that re-opened
    padding would change what "baseline" means. An all-padding-derived mask
    expanded to [B, 1, S, S] is an exact no-op (`masked_fill` sees the same
    zeros), so baseline rows share the knockout path.
    """
    import torch

    blocks = list(model.encoder_blocks)
    fired = {L: 0 for L in range(len(blocks))}
    handles = []

    def make_hook(L):
        def hook(_mod, args, kwargs):
            fired[L] += 1
            pad = kwargs.get("attention_mask")
            if pad is None or pad.dim() != 4 or pad.shape[2] not in (1, pad.shape[3]):
                raise RuntimeError(f"encoder block {L}: expected the vendor's [B, 1, S|1, S] padding mask")
            if mask.shape[0] != pad.shape[0] or mask.shape[-1] != pad.shape[-1]:
                raise ValueError(f"encoder mask {tuple(mask.shape)} does not match padding {tuple(pad.shape)}")
            reopened = (mask != 0) & (pad == 0)
            if bool(reopened.any()):
                raise ValueError(f"encoder block {L}: the replacement mask re-opens padded keys")
            return args, {**kwargs, "attention_mask": mask}
        return hook

    try:
        for L, blk in enumerate(blocks):
            handles.append(blk.attn.register_forward_pre_hook(make_hook(L), with_kwargs=True))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := {L: n for L, n in fired.items() if n != 1}:
        raise RuntimeError(f"encoder blocks fired {bad} times, expected once each")


def encoder_block_mask(history_mask, blocked):
    """[B, 1, S, S] float mask: padding-derived, plus rows' blocked (queries, key_slot).

    blocked: list of (row, key_slot) -> every OTHER real slot stops reading
    key_slot (the item's own slot still reads itself). Refuses padded key slots.
    """
    import numpy as np

    hm = np.asarray(history_mask, dtype=bool)
    B, S = hm.shape
    m = np.broadcast_to(hm[:, None, None, :].astype(np.float32), (B, 1, S, S)).copy()
    for row, key in blocked:
        if not hm[row, key]:
            raise ValueError(f"row {row}: encoder knockout of padded slot {key}")
        q = np.flatnonzero(hm[row])
        q = q[q != key]
        m[row, 0, q, key] = 0
    return m
