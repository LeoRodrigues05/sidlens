"""A Qwen2 attention layer's query, key or value input, captured or replaced at chosen tokens.

N2 asks where the AR model decides that a history item's prefix matches: in
what the readout asks for (query), in what h's digit token offers (key), or in
what it carries (value). These hooks act on the output of
`self_attn.{q,k,v}_proj` (before RoPE and the head split), so replacing it at
one position changes that token's query, key or value input to attention in
that layer only; the residual stream and every other layer are untouched.
`o_input` captures the input of `o_proj` (the heads' outputs, concatenated)
for direct logit attribution.

Traps, each with its guard
--------------------------
1. **A patch that fires nowhere.** Every listed layer must fire at least once,
   or the context raises on exit.
2. **Observers that recompute projections.** `hooks.attention_probs` calls
   `q_proj` / `k_proj` itself; `patch` patches every call, so observed
   attention is what the patched layer attended with. `capture` refuses a
   second call in one forward instead of silently keeping either.
3. **Rows and positions.** `rows[i]`, `positions[i]` index the forward's batch
   and sequence (the fixed-shape runner's job order, right padding, one length
   bucket); values are [P, dim] in that order.
4. **Patching with its own value** returns the module output bit for bit
   (`tests/interventions/test_qkv.py`).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

PROJ = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}


def _modules(model, which: str, layers: Sequence[int]):
    if which not in PROJ:
        raise ValueError(f"which is one of {sorted(PROJ)}")
    return {L: getattr(model.model.layers[L].self_attn, PROJ[which]) for L in layers}


@contextmanager
def capture(model, which: str, layers: Sequence[int], rows: Sequence[int], positions: Sequence[int]) -> Iterator[dict]:
    """Yield layer -> [P, dim] (detached, on device) of the projection output at (rows, positions)."""
    import torch

    if len(rows) != len(positions) or not len(rows):
        raise ValueError("rows and positions must be equally long and non-empty")
    out: dict[int, "torch.Tensor"] = {}
    handles = []

    def make(L):
        def hook(_m, _inp, o):
            if L in out:
                raise RuntimeError(f"layer {L} {which}_proj fired twice in one captured forward")
            r = torch.as_tensor(rows, device=o.device)
            p = torch.as_tensor(positions, device=o.device)
            out[L] = o[r, p].detach().clone()
        return hook

    try:
        for L, mod in _modules(model, which, layers).items():
            handles.append(mod.register_forward_hook(make(L)))
        yield out
    finally:
        for h in handles:
            h.remove()
    if missing := [L for L in layers if L not in out]:
        raise RuntimeError(f"{which}_proj of layers {missing} never fired")


@contextmanager
def patch(model, which: str, values: dict, rows: Sequence[int], positions: Sequence[int]) -> Iterator[dict]:
    """Replace the projection output at (rows, positions) with values[layer] [P, dim], every call."""
    import torch

    if len(rows) != len(positions) or not len(rows):
        raise ValueError("rows and positions must be equally long and non-empty")
    fired = {L: 0 for L in values}
    handles = []

    def make(L):
        v = values[L]

        def hook(_m, _inp, o):
            fired[L] += 1
            if v.shape != (len(rows), o.shape[-1]):
                raise ValueError(f"layer {L}: values {tuple(v.shape)} do not fit {len(rows)} x {o.shape[-1]}")
            o = o.clone()
            o[torch.as_tensor(rows, device=o.device), torch.as_tensor(positions, device=o.device)] = \
                v.to(o.dtype).to(o.device)
            return o
        return hook

    try:
        for L, mod in _modules(model, which, list(values)).items():
            handles.append(mod.register_forward_hook(make(L)))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := [L for L, n in fired.items() if n == 0]:
        raise RuntimeError(f"{which}_proj patches of layers {bad} never fired")


@contextmanager
def o_input(model, layers: Sequence[int], rows: Sequence[int], positions: Sequence[int]) -> Iterator[dict]:
    """Yield layer -> [P, n_heads * head_dim]: the heads' outputs entering `o_proj` at (rows, positions)."""
    import torch

    out: dict[int, "torch.Tensor"] = {}
    handles = []

    def make(L):
        def hook(_m, args):
            if L in out:
                raise RuntimeError(f"layer {L} o_proj fired twice in one captured forward")
            x = args[0]
            out[L] = x[torch.as_tensor(rows, device=x.device), torch.as_tensor(positions, device=x.device)].detach().clone()
        return hook

    try:
        for L in layers:
            handles.append(model.model.layers[L].self_attn.o_proj.register_forward_pre_hook(make(L)))
        yield out
    finally:
        for h in handles:
            h.remove()
    if missing := [L for L in layers if L not in out]:
        raise RuntimeError(f"o_proj of layers {missing} never fired")
