"""Attention-edge knockout: forbid chosen query positions from reading chosen keys.

A knockout removes the edge q -> k in the listed layers (and heads): k is
masked out of q's softmax and the remaining attention renormalizes. It
removes a DIRECT read; information from k can still reach q through other
positions, which is why experiments pair it with wider knockouts that bound
the indirect routes.

Mechanism
---------
The caller passes an explicit 4D boolean mask (True = may attend) of shape
(B, n_heads, T, T) to the model. transformers returns a prepared 4D mask
as-is (`masking_utils._preprocess_mask_arguments`), so every layer receives
that very tensor as the `attention_mask` keyword. A forward pre-hook on each
`self_attn` swaps in a per-layer copy with the knocked-out entries set False.

Traps, each with the guard that closes it
-----------------------------------------
1. **A silently ignored mask.** If the model rebuilt or converted the mask
   (a different attention backend or transformers version), editing it would
   do nothing. Each hook checks that the mask it received IS the base mask
   object, and that every layer fired exactly `expect_calls` times.
2. **Kernel paths that differ between conditions.** A (B, 1, T, T) mask for
   the baseline and a (B, H, T, T) one for a knockout could select different
   kernels and differ numerically for reasons unrelated to the knockout. The
   base mask always has the full head axis, and `base_mask` is used for EVERY
   forward, with or without knockouts.
3. **No-op knockouts that look like null results.** Knocking out an edge that
   is already masked (a future or padding key) changes nothing. That raises.
   The exactness control that WANTS a no-op sets `expect_masked=True` on its
   own knockout, which then raises unless every edge IS already masked, so
   one control in a batch never relaxes the guard for the others.
4. **Empty softmax rows.** Removing the self edge could leave a query with no
   keys (position 0) and yield NaN. Self edges are refused.
5. **Cross-row leakage.** Knockouts name their batch row; only that row's
   mask entries change, so other rows in the batch are untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence


@dataclass(frozen=True)
class Knockout:
    """Remove edges queries x keys for one batch row, in `layers` (and `heads`)."""
    row: int
    queries: tuple[int, ...]
    keys: tuple[int, ...]
    layers: tuple[int, ...]
    heads: tuple[int, ...] | None = None      # None = every head
    expect_masked: bool = False               # a no-op control: every edge already masked

    def __post_init__(self) -> None:
        if not self.queries or not self.keys or not self.layers:
            raise ValueError("a knockout needs at least one query, key and layer")
        if set(self.queries) & set(self.keys):
            raise ValueError("self edges cannot be knocked out (a query could lose every key)")


def base_mask(attention_mask_2d, n_heads: int):
    """(B, n_heads, T, T) bool: causal AND key-not-padding. Contiguous, so every
    layer and every condition reads the same memory layout."""
    import torch

    m2 = attention_mask_2d.bool()
    B, T = m2.shape
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=m2.device))
    m = causal[None, None] & m2[:, None, None, :]
    return m.expand(B, n_heads, T, T).contiguous()


def _decoder_attn(model):
    layers = model.model.layers
    return [layer.self_attn for layer in layers]


@contextmanager
def knockout_attention(model, mask, knockouts: Sequence[Knockout] = (),
                       expect_calls: int = 1) -> Iterator[dict]:
    """Apply `knockouts` to every forward inside the context.

    `mask` must be the (B, H, T, T) tensor also passed as `attention_mask` to
    the model. Yields a dict layer -> call count; on a clean exit every layer
    must have fired exactly `expect_calls` times.
    """
    import torch

    attn = _decoder_attn(model)
    B, H, T, T2 = mask.shape
    if T != T2 or mask.dtype != torch.bool:
        raise ValueError(f"mask must be (B, H, T, T) bool, got {tuple(mask.shape)} {mask.dtype}")
    n_heads = model.config.num_attention_heads
    if H != n_heads:
        raise ValueError(f"mask has {H} heads, model has {n_heads}")
    per_layer: dict[int, list[Knockout]] = {}
    for ko in knockouts:
        if not 0 <= ko.row < B:
            raise IndexError(f"knockout row {ko.row} outside batch of {B}")
        if max(ko.queries) >= T or max(ko.keys) >= T or min(ko.queries) < 0 or min(ko.keys) < 0:
            raise IndexError(f"knockout position outside sequence of {T}")
        if ko.heads is not None and (min(ko.heads) < 0 or max(ko.heads) >= H):
            raise IndexError(f"knockout head outside 0..{H - 1}")
        if min(ko.layers) < 0 or max(ko.layers) >= len(attn):
            raise IndexError(f"knockout layer outside 0..{len(attn) - 1}")
        q = torch.as_tensor(ko.queries, device=mask.device)
        k = torch.as_tensor(ko.keys, device=mask.device)
        live = mask[ko.row, 0][q[:, None], k[None, :]]
        if ko.expect_masked and bool(live.any()):
            raise ValueError(f"row {ko.row}: a no-op control knocks out a live edge")
        if not ko.expect_masked and not bool(live.all()):
            raise ValueError(f"row {ko.row}: some knocked-out edges are already masked "
                             f"(future or padding keys); that knockout would be a no-op")
        for L in ko.layers:
            per_layer.setdefault(L, []).append(ko)

    layer_masks = {}
    for L, kos in per_layer.items():
        m = mask.clone()
        for ko in kos:
            h = torch.arange(H, device=mask.device) if ko.heads is None else \
                torch.as_tensor(ko.heads, device=mask.device)
            q = torch.as_tensor(ko.queries, device=mask.device)
            k = torch.as_tensor(ko.keys, device=mask.device)
            m[ko.row, h[:, None, None], q[None, :, None], k[None, None, :]] = False
        layer_masks[L] = m

    fired = {L: 0 for L in range(len(attn))}
    handles = []

    def make_hook(L: int):
        def hook(_mod, args, kwargs):
            fired[L] += 1
            got = kwargs.get("attention_mask")
            if got is not mask:
                raise RuntimeError(
                    f"layer {L} received a different attention mask than the one passed "
                    f"to the model ({type(got).__name__} {getattr(got, 'shape', None)}); "
                    f"editing it would not reach the attention kernel")
            if L in layer_masks:
                return args, {**kwargs, "attention_mask": layer_masks[L]}
            return None
        return hook

    try:
        for L, mod in enumerate(attn):
            handles.append(mod.register_forward_pre_hook(make_hook(L), with_kwargs=True))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := {L: n for L, n in fired.items() if n != expect_calls}:
        raise RuntimeError(f"attention layers fired {bad} times, expected {expect_calls} each")
