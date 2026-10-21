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

Patching and step-level switches
--------------------------------
DiffGRM predicts codes, not noise, and has no step input. Between two reveal
steps nothing survives but the partial SID and the beam score: self-attention
is recomputed and the cross-attention K/V depend on the history alone. So
"use another run's prediction at step t" has exactly two forms. On the same
state it is a switch of the history for that step (`history_switch_scorer`);
inside one step it is activation patching of the encoder states by slot
(`mix_encoder_slots`), of the decoder residual stream by (block, digit
position) (`score_states(patches=...)`), or of what one block's cross-attention
read at one digit (`patch_cross_attention`). Digit-to-digit reads are the
decoder's self-attention (`decoder_self_attention_masks`).

5. **A patch on a module the forward never reaches.** `score_states` runs the
   decoder only, so an `encoder_blocks.N` patch would not fire. Only
   `decoder_blocks.N` sites are accepted here; encoder states are an explicit
   input and are edited as a tensor.
6. **A slot swap that swaps nothing.** Padded slots are zero in every run, so
   taking one from the donor changes nothing. `mix_encoder_slots` refuses them,
   and refuses two runs whose padding differs.
7. **An empty attention row.** A self-attention mask that blocks every key of
   a query makes that softmax NaN, and NaN logits rank arbitrarily.
   `self_knockout_mask` and the hook both refuse it.
8. **Controls that are only approximately exact.** A no-op switch must be the
   baseline bit for bit. The two histories are therefore projected separately,
   as the baseline projects its own, and rows are selected between them; a
   projection of the two stacked would not be bit-identical on CPU.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator


def project_cross_cache(model, encoder_hidden):
    """Per-block projected cross-attention (K, V), as `matched_decode` builds them."""
    width = model.n_embd
    cache = []
    for block in model.decoder_blocks:
        projected = block.cross_attn.qkv(encoder_hidden)
        cache.append((projected[..., width:2 * width], projected[..., 2 * width:]))
    return cache


def score_states(model, encoder_hidden, cross_cache, enc_rows, codes, masked, xmask=None,
                 *, smask=None, patches=None, cross_patches=None):
    """Log-softmax digit distributions [R, n_digit, K] at R decoder states.

    enc_rows  [R] long: encoder row of each decoder row
    codes     [R, n_digit] long: revealed codes (any value where masked)
    masked    [R, n_digit] bool: True = still masked
    xmask     optional [n_layers, R, n_digit, S] cross-attention masks (1 keep, 0 block)
    smask     optional decoder self-attention masks, see `decoder_self_attention_masks`
    patches   optional `residual.Patch` list at `decoder_blocks.N` sites; rows index
              the R decoder rows and pos the digit position. Each site fires once.
    cross_patches optional {block: (rows, pos, values)} for `patch_cross_attention`
    """
    import torch

    from sidlens.interventions.residual import patch_residual

    rows = enc_rows
    past = [(None, (k.index_select(0, rows), v.index_select(0, rows))) for k, v in cross_cache]
    if patches and (bad := sorted({p.site for p in patches if not p.site.startswith("decoder_blocks.")})):
        raise ValueError(f"score_states runs the decoder only, so {bad} would never fire. "
                         f"Encoder states are an input here: edit them with mix_encoder_slots.")
    with ExitStack() as stack:
        if xmask is not None:
            stack.enter_context(cross_attention_masks(model, xmask))
        if smask is not None:
            stack.enter_context(decoder_self_attention_masks(model, smask))
        if patches:
            stack.enter_context(patch_residual(model, patches, expect_calls=1))
        if cross_patches:
            stack.enter_context(patch_cross_attention(model, cross_patches))
        out = model.forward_decoder_only(
            {"decoder_input_ids": codes.clamp_min(0), "mask_positions": masked.to(torch.float32),
             "encoder_hidden": encoder_hidden.index_select(0, rows)},
            return_loss=False, digit=None, past_key_values=past, use_cache=True)
    logits = out.logits
    if tuple(logits.shape) != (len(rows), int(model.n_digit), int(model.codebook_size)):
        raise ValueError(f"unexpected decoder logits shape {tuple(logits.shape)}")
    return torch.log_softmax(logits.float(), dim=-1)


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


@contextmanager
def decoder_self_attention_masks(model, smask) -> Iterator[dict]:
    """Inject per-block decoder self-attention masks (1 keep, 0 block).

    smask is [n_layers, R, n_digit, n_digit] (query digit, key digit), or
    [n_layers, R, n_head, n_digit, n_digit] to block single heads. The vendor
    calls `self_attn(..., attention_mask=None)`: digit positions attend to each
    other freely, masked or revealed. An all-ones mask is an exact no-op
    (`masked_fill` with no zero). If a future vendor version passes its own
    mask, the hook raises instead of overwriting it.
    """
    blocks = list(model.decoder_blocks)
    if smask.shape[0] != len(blocks) or smask.dim() not in (4, 5):
        raise ValueError(f"need [n_layers={len(blocks)}, R, (n_head,) n_digit, n_digit] masks, "
                         f"got {tuple(smask.shape)}")
    if bool((smask != 0).sum(-1).eq(0).any()):
        raise ValueError("a self-attention mask blocks every key of some query; that softmax is NaN")
    fired = {L: 0 for L in range(len(blocks))}
    handles = []

    def make_hook(L):
        def hook(mod, args, kwargs):
            fired[L] += 1
            if kwargs.get("attention_mask") is not None:
                raise RuntimeError(f"decoder block {L} self-attention already receives a mask; "
                                   f"the vendor changed and the knockout would overwrite it")
            q = args[0]
            m = smask[L]
            if m.shape[0] != q.shape[0] or m.shape[-1] != q.shape[1] or m.shape[-2] != q.shape[1]:
                raise ValueError(f"block {L}: mask {tuple(m.shape)} does not match queries {tuple(q.shape)}")
            if m.dim() == 4 and m.shape[1] != mod.n_head:
                raise ValueError(f"block {L}: per-head mask has {m.shape[1]} heads, the block has {mod.n_head}")
            return args, {**kwargs, "attention_mask": m}
        return hook

    try:
        for L, blk in enumerate(blocks):
            handles.append(blk.self_attn.register_forward_pre_hook(make_hook(L), with_kwargs=True))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := {L: n for L, n in fired.items() if n != 1}:
        raise RuntimeError(f"decoder self-attention blocks fired {bad} times, expected once each")


def self_knockout_mask(n_layers: int, n_rows: int, n_digit: int, edges, n_head: int | None = None):
    """Decoder self-attention masks as float32 numpy (1 keep, 0 block).

    edges: (row, layers, query_digits, key_digits) or, with `n_head`,
    (row, layers, query_digits, key_digits, heads). Shape is
    [n_layers, R, n_digit, n_digit] without `n_head`, else
    [n_layers, R, n_head, n_digit, n_digit]. Refuses an edge set that leaves a
    query with no key in some block (and head): that softmax would be NaN.
    """
    import numpy as np

    shape = (n_layers, n_rows, n_digit, n_digit) if n_head is None else (n_layers, n_rows, n_head, n_digit, n_digit)
    m = np.ones(shape, dtype=np.float32)
    for edge in edges:
        if n_head is None:
            row, layers, queries, keys = edge
            m[np.ix_(list(layers), [row], list(queries), list(keys))] = 0
        else:
            row, layers, queries, keys, heads = edge
            m[np.ix_(list(layers), [row], list(heads), list(queries), list(keys))] = 0
    if (m.sum(-1) == 0).any():
        raise ValueError("self-attention knockout leaves a query digit with no key")
    return m


@contextmanager
def patch_cross_attention(model, patches: dict) -> Iterator[dict]:
    """Overwrite decoder blocks' cross-attention OUTPUTS at (row, digit position) pairs.

    patches: {block L: (rows [k] long, pos [k] long, values [k, width])}. The
    value replaces what block L's cross-attention adds to the residual stream at
    that position (after its output projection, before the residual add), so a
    patch moves exactly what that block read from the history at that digit and
    nothing that self-attention or the MLP wrote. Values come from a capture of
    `decoder_blocks.L.cross_attn` in another run (`hooks.capture` returns the
    module's first output, the same tensor).

    This is a sublayer patch, deliberately not a `residual.patch_residual` site
    (that module accepts residual read points only). The same guards apply:
    every patched block must fire exactly once, indices are bounds-checked,
    duplicate (row, position) writes raise, and the write goes into a clone.
    """
    import torch

    blocks = list(model.decoder_blocks)
    if bad := sorted(L for L in patches if not 0 <= int(L) < len(blocks)):
        raise ValueError(f"no decoder blocks {bad}; the model has {len(blocks)}")
    prepared = {}
    for L, (rows, pos, values) in patches.items():
        rows, pos = torch.as_tensor(rows).long(), torch.as_tensor(pos).long()
        if rows.ndim != 1 or rows.shape != pos.shape or values.ndim != 2 or values.shape[0] != len(rows):
            raise ValueError(f"block {L}: need rows [k], pos [k], values [k, width]")
        key = rows.cpu() * (int(pos.max()) + 1) + pos.cpu()
        if len(torch.unique(key)) != len(key):
            raise ValueError(f"block {L}: two patches write the same (row, position)")
        prepared[int(L)] = (rows, pos, values)
    fired = {L: 0 for L in prepared}
    handles = []

    def make_hook(L):
        rows, pos, values = prepared[L]

        def hook(_mod, _inp, out):
            fired[L] += 1
            y = out[0]
            if int(rows.max()) >= y.shape[0] or int(pos.max()) >= y.shape[1] or int(rows.min()) < 0 \
                    or int(pos.min()) < 0:
                raise IndexError(f"block {L}: patch index outside cross-attention output {tuple(y.shape)}")
            if values.shape[1] != y.shape[2]:
                raise ValueError(f"block {L}: patch width {values.shape[1]} != {y.shape[2]}")
            new = y.clone()
            new[rows.to(y.device), pos.to(y.device)] = values.to(device=y.device, dtype=y.dtype)
            return (new, *out[1:])
        return hook

    try:
        for L in prepared:
            handles.append(blocks[L].cross_attn.register_forward_hook(make_hook(L)))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := {L: n for L, n in fired.items() if n != 1}:
        raise RuntimeError(f"cross-attention patches fired {bad} times, expected once each")


def mix_encoder_slots(clean, donor, take, history_mask):
    """Encoder states [B, S, d]: `donor`'s where `take` [B, S] is True, else `clean`'s.

    The two runs must share `history_mask` [B, S] (same history lengths, as
    with a replaced item): slot s of one run is then slot s of the other.
    Refuses a padded slot, which is zero in both runs and would swap nothing,
    and refuses states that are not zero exactly on the padding, which is how
    two runs with different padding show up here. The result is a new tensor;
    build its cross-attention cache with `project_cross_cache`.
    """
    import torch

    if not (clean.shape == donor.shape and tuple(take.shape) == tuple(clean.shape[:2]) == tuple(history_mask.shape)):
        raise ValueError(f"shapes differ: clean {tuple(clean.shape)}, donor {tuple(donor.shape)}, "
                         f"take {tuple(take.shape)}, history_mask {tuple(history_mask.shape)}")
    take, history_mask = take.to(torch.bool), history_mask.to(torch.bool)
    if bool((take & ~history_mask).any()):
        raise ValueError("mix_encoder_slots: a padded history slot was selected; it is zero in both runs")
    pad = ~history_mask
    if bool((clean[pad] != 0).any()) or bool((donor[pad] != 0).any()):
        raise ValueError("encoder states are nonzero on the padding: the two runs do not share history_mask")
    if any(bool(((e == 0).all(-1) & history_mask).any()) for e in (clean, donor)):
        raise ValueError("an encoder state is exactly zero on a real slot: history_mask marks padding as real")
    return torch.where(take[:, :, None], donor, clean)


def history_switch_scorer(model, enc_base, enc_donor, schedule):
    """`score_fn` for `matched_decode.decode`: the donor history at scheduled steps.

    schedule [B, n_digit] bool: row b's states at step t are scored with
    `enc_donor[b]` where schedule[b, t] is True and with `enc_base[b]`
    elsewhere. The state is the search's own in both cases, so this is the
    other run's prediction at the same step and the same partial SID. To run
    several schedules for one user, repeat the user's rows.

    All-False is the baseline bit for bit and all-True is an independent decode
    of the donor (trap 8): each history's K/V come from its own projection and
    rows are selected between them. The returned function counts its calls per
    step in `.calls`, so a caller can show that every step was scored through
    it.
    """
    import torch

    from sidlens.analysis.matched_decode import score_partial

    if enc_base.shape != enc_donor.shape:
        raise ValueError(f"base {tuple(enc_base.shape)} and donor {tuple(enc_donor.shape)} encoder states differ in shape")
    n = int(model.n_digit)
    if tuple(schedule.shape) != (enc_base.shape[0], n) or schedule.dtype != torch.bool:
        raise ValueError(f"schedule must be bool [B={enc_base.shape[0]}, n_digit={n}], got "
                         f"{schedule.dtype} {tuple(schedule.shape)}")
    cache_base = project_cross_cache(model, enc_base)
    cache_donor = project_cross_cache(model, enc_donor)
    calls = {t: 0 for t in range(n)}

    def score_fn(step, partial, history_rows):
        calls[step] += 1
        use_donor = schedule[history_rows, step]
        pick = use_donor[:, None, None]
        enc = torch.where(pick, enc_donor.index_select(0, history_rows), enc_base.index_select(0, history_rows))
        cache = [(torch.where(pick, kd.index_select(0, history_rows), kb.index_select(0, history_rows)),
                  torch.where(pick, vd.index_select(0, history_rows), vb.index_select(0, history_rows)))
                 for (kb, vb), (kd, vd) in zip(cache_base, cache_donor, strict=True)]
        return score_partial(model, enc, cache, partial, torch.arange(len(history_rows), device=partial.device))

    score_fn.calls = calls
    return score_fn
