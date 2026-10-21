"""Scale chosen attention edges by a factor alpha (dose-response), instead of removing them.

Multiplying the unnormalised attention weight of edge q -> k by alpha and
renormalising equals adding log(alpha) to that edge's logit. alpha = 0 is a
knockout, alpha = 1 changes nothing, alpha > 1 strengthens the read.

Three mechanisms, one per code path, each with the same contract:

    scale_attention       a teacher-forced Qwen forward: an explicit (B, H, T, T)
                          float ADDITIVE mask (0 keep, -inf masked) is passed to
                          the model, and each listed layer receives a copy with
                          log(alpha) added at the chosen edges.
    generation_scale      the same inside KV-cached `generate()` beam search,
                          with edges planned from each beam's own prefix
                          (`interventions.generation` explains the plan).
    reweight_attention    DiffGRM's attention modules, which only accept 0/1
                          masks (`masked_fill(mask == 0, -inf)`): a forward hook
                          on the module's post-softmax dropout (the identity in
                          eval mode) multiplies the chosen weights by alpha and
                          renormalises those query rows.

Traps, each with the guard that closes it
-----------------------------------------
1. **alpha = 1 must be the baseline exactly.** For the Qwen paths every
   condition, alpha = 1 included, goes through the same float-mask route, and
   log(1) = 0 adds nothing. `reweight_attention` returns the module's output
   unchanged for rows with no scaled edge; with alpha = 1 it changes nothing.
2. **A float mask a backend would ignore.** transformers returns a prepared 4D
   mask as-is; each hook checks it received the very tensor passed in.
3. **Scaling a masked edge.** An edge that is already -inf (future, padding)
   would be scaled to nothing silently; it raises.
4. **alpha = 0 leaves a query with no keys.** Refused, as for knockouts.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence


def log_alpha(alpha: float) -> float:
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}")
    return -math.inf if alpha == 0 else math.log(alpha)


@dataclass(frozen=True)
class Scale:
    """Scale edges queries x keys of one batch row by alpha, in `layers`."""
    row: int
    queries: tuple[int, ...]
    keys: tuple[int, ...]
    layers: tuple[int, ...]
    alpha: float

    def __post_init__(self) -> None:
        if not self.queries or not self.keys or not self.layers:
            raise ValueError("a scaled edge needs at least one query, key and layer")
        if set(self.queries) & set(self.keys):
            raise ValueError("self edges cannot be scaled")


def float_mask(bool_mask, dtype):
    """Additive mask from a boolean one: 0 where True, -inf where False."""
    import torch
    out = torch.zeros(bool_mask.shape, dtype=dtype, device=bool_mask.device)
    return out.masked_fill_(~bool_mask, float("-inf")).contiguous()


@contextmanager
def scale_attention(model, fmask, scales: Sequence[Scale] = (), expect_calls: int = 1) -> Iterator[dict]:
    """`interventions.attention.knockout_attention` for an additive float mask.

    `fmask` (B, H, T, T) float must be the tensor passed as `attention_mask`.
    """
    import torch

    attn = [layer.self_attn for layer in model.model.layers]
    B, H, T, T2 = fmask.shape
    if T != T2 or not fmask.is_floating_point():
        raise ValueError(f"mask must be (B, H, T, T) float, got {tuple(fmask.shape)} {fmask.dtype}")
    per_layer: dict[int, list[Scale]] = {}
    for sc in scales:
        if not 0 <= sc.row < B or max(sc.queries) >= T or max(sc.keys) >= T:
            raise IndexError(f"scaled edge outside the batch/sequence: {sc}")
        q = torch.as_tensor(sc.queries, device=fmask.device)
        k = torch.as_tensor(sc.keys, device=fmask.device)
        if bool(torch.isinf(fmask[sc.row, 0][q[:, None], k[None, :]]).any()):
            raise ValueError(f"row {sc.row}: a scaled edge is already masked")
        for L in sc.layers:
            per_layer.setdefault(L, []).append(sc)
    layer_masks = {}
    for L, scs in per_layer.items():
        m = fmask.clone()
        for sc in scs:
            q = torch.as_tensor(sc.queries, device=fmask.device)
            k = torch.as_tensor(sc.keys, device=fmask.device)
            m[sc.row, :, q[:, None], k[None, :]] += log_alpha(sc.alpha)
        if bool(torch.isinf(m).all(-1).any()):
            raise ValueError(f"layer {L}: scaling leaves a query with no keys")
        layer_masks[L] = m

    fired = {L: 0 for L in range(len(attn))}
    handles = []

    def make_hook(L: int):
        def hook(_mod, args, kwargs):
            fired[L] += 1
            if kwargs.get("attention_mask") is not fmask:
                raise RuntimeError(f"layer {L} did not receive the float mask passed to the model")
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


@contextmanager
def generation_scale(model, layers: Sequence[int], plan, prompt_len: int, alpha: float) -> Iterator[dict]:
    """`interventions.generation.generation_knockout`, scaling the planned edges by alpha.

    Every forward in `layers` receives a float additive mask (materialised from
    the boolean causal/padding mask transformers builds, or from the causal
    structure when it passes none), with log(alpha) at the planned edges. The
    same route is taken for alpha = 1.
    """
    import torch

    attn = [layer.self_attn for layer in model.model.layers]
    layers = sorted(set(layers))
    la = log_alpha(alpha)
    stats = {"forwards": 0, "edges": 0, "edges_by_step": {}, "fired": {}}
    state = {"seqs": None, "forward": -1, "mask_in": None, "mask_out": None}
    orig_prepare = model.prepare_inputs_for_generation

    def prepare(input_ids, *args, **kwargs):
        state["seqs"] = input_ids
        return orig_prepare(input_ids, *args, **kwargs)

    def top_pre(_mod, args, kwargs):
        stats["forwards"] += 1
        state["forward"] += 1
        state["mask_in"] = state["mask_out"] = None
        return None

    def make_hook(L: int):
        def hook(_mod, args, kwargs):
            key = (state["forward"], L)
            if key in stats["fired"]:
                raise RuntimeError(f"layer {L} fired twice in forward {state['forward']}")
            stats["fired"][key] = 1
            m = kwargs.get("attention_mask")
            h = kwargs["hidden_states"]
            B, q = h.shape[0], h.shape[1]
            if state["mask_out"] is None:
                seqs = state["seqs"]
                if seqs is None or seqs.shape[0] != B:
                    raise RuntimeError("no running sequences recorded for this forward")
                kv = seqs.shape[1]
                if m is None:
                    i = torch.arange(q, device=h.device)[:, None] + (kv - q)
                    j = torch.arange(kv, device=h.device)[None, :]
                    mb = (j <= i)[None, None].expand(B, 1, q, kv)
                elif m.dtype == torch.bool:
                    mb = m
                else:
                    raise TypeError(f"unexpected non-boolean generation mask {m.dtype}")
                fm = float_mask(mb, h.dtype)
                edges = plan(seqs, seqs.shape[1] - prompt_len, q, kv)
                n = 0
                for row, qi, keys in edges:
                    if not keys:
                        continue
                    k = torch.as_tensor(keys, device=h.device)
                    if bool(torch.isinf(fm[row, 0, qi, k]).any()):
                        raise ValueError(f"row {row}: a scaled key is already masked")
                    fm[row, 0, qi, k] += la
                    if bool(torch.isinf(fm[row, 0, qi]).all()):
                        raise ValueError(f"row {row}: scaling left the query no keys")
                    n += len(keys)
                stats["edges"] += n
                step = seqs.shape[1] - prompt_len
                stats["edges_by_step"][step] = stats["edges_by_step"].get(step, 0) + n
                state["mask_in"], state["mask_out"] = m, fm
            elif m is not state["mask_in"]:
                raise RuntimeError(f"layer {L} received a different mask than earlier layers")
            return args, {**kwargs, "attention_mask": state["mask_out"]}
        return hook

    handles = []
    try:
        model.prepare_inputs_for_generation = prepare
        handles.append(model.model.register_forward_pre_hook(top_pre, with_kwargs=True))
        for L in layers:
            handles.append(attn[L].register_forward_pre_hook(make_hook(L), with_kwargs=True))
        yield stats
    finally:
        for hd in handles:
            hd.remove()
        del model.prepare_inputs_for_generation
    missing = [(f, L) for f in range(stats["forwards"]) for L in layers if (f, L) not in stats["fired"]]
    if missing:
        raise RuntimeError(f"hooked layers did not fire in {len(missing)} (forward, layer) slots")


@contextmanager
def reweight_attention(modules: dict, weights_fn) -> Iterator[dict]:
    """Scale DiffGRM attention weights after the softmax, per module.

    modules     {name: MultiHeadAttention} (the vendor's encoder `attn`, decoder
                `self_attn` / `cross_attn` modules)
    weights_fn  (name, att [B, H, T, T_kv]) -> multiplicative weights broadcastable
                to att, or None to leave this call unchanged

    Rows (b, h, t) whose weights are all 1 are returned bit for bit; others are
    multiplied and renormalised. Each module must fire at least once.
    """
    import torch

    fired = {name: 0 for name in modules}
    handles = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            fired[name] += 1
            w = weights_fn(name, out)
            if w is None:
                return None
            w = torch.as_tensor(w, dtype=out.dtype, device=out.device).expand_as(out)
            changed = (w != 1).any(-1, keepdim=True)
            if not bool(changed.any()):
                return None
            new = out * w
            z = new.sum(-1, keepdim=True)
            if bool((z[changed] == 0).any()):
                raise ValueError(f"{name}: reweighting left a query with no attention")
            return torch.where(changed, new / z, out)
        return hook

    try:
        for name, mod in modules.items():
            handles.append(mod.attn_dropout.register_forward_hook(make_hook(name)))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := [n for n, c in fired.items() if c == 0]:
        raise RuntimeError(f"attention modules never fired: {bad}")
