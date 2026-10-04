"""Per-head attention probabilities for chosen queries, recomputed from layer inputs.

`attention_points` sees the attention module's OUTPUT, which mixes heads and
values; SDPA never materializes the probabilities. This observer recomputes
them for selected (row, query) pairs from what the layer actually received:
its input hidden states, the rotary embeddings and the mask. Observation only:
the pre-hook returns None, so the forward is unchanged.

Traps, each with the guard that closes it
-----------------------------------------
1. **A different mask from the one the kernel used.** The probabilities are
   computed with the `attention_mask` keyword the layer received (after any
   knockout hook registered earlier), so they describe the same softmax.
2. **Grouped-query attention.** Qwen2 shares K/V between query heads; keys are
   repeated per group exactly as `repeat_kv` does before the kernel.
3. **Precision.** Scores are computed in fp32 from the bf16 projections. They
   are close to, not bit-equal with, what SDPA computed internally. That is
   fine for an observational map and must not be used as a causal quantity.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence


@contextmanager
def attention_probs(model, layers: Sequence[int], rows: Sequence[int],
                    queries: Sequence[int]) -> Iterator[dict]:
    """Yield layer -> (P, H, T) fp32 probabilities for the pairs (rows[i], queries[i]).

    Filled in as the model runs; each listed layer must fire exactly once.
    """
    import torch
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    if len(rows) != len(queries) or not len(rows):
        raise ValueError("rows and queries must be equally long and non-empty")
    attn = [layer.self_attn for layer in model.model.layers]
    out: dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(L: int):
        mod = attn[L]

        def hook(_mod, args, kwargs):
            if L in out:
                raise RuntimeError(f"layer {L} fired twice in one observed forward")
            h = kwargs["hidden_states"]
            cos, sin = kwargs["position_embeddings"]
            mask = kwargs.get("attention_mask")
            B, T, _ = h.shape
            r = torch.as_tensor(rows, device=h.device)
            qpos = torch.as_tensor(queries, device=h.device)
            with torch.no_grad():
                hd = mod.head_dim
                q = mod.q_proj(h).view(B, T, -1, hd).transpose(1, 2)
                k = mod.k_proj(h).view(B, T, -1, hd).transpose(1, 2)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = repeat_kv(k, mod.num_key_value_groups)
                qs = q[r, :, qpos].float()                       # (P, H, D)
                ks = k[r].float()                                # (P, H, T, D)
                s = torch.einsum("phd,phtd->pht", qs, ks) * mod.scaling
                if mask is None:
                    allowed = torch.arange(T, device=h.device)[None, :] <= qpos[:, None]
                    s = s.masked_fill(~allowed[:, None, :], float("-inf"))
                else:
                    sel = mask[r, :, qpos][..., :T]                 # (P, H or 1, T)
                    if mask.dtype == torch.bool:                    # sdpa: True = may attend
                        s = s.masked_fill(~sel, float("-inf"))
                    else:                                           # eager: additive 0 / -inf
                        s = s + sel.float()
                out[L] = torch.softmax(s, dim=-1).cpu()
            return None
        return hook

    try:
        for L in layers:
            handles.append(attn[L].register_forward_pre_hook(make_hook(L), with_kwargs=True))
        yield out
    finally:
        for h in handles:
            h.remove()
    if missing := [L for L in layers if L not in out]:
        raise RuntimeError(f"observed layers {missing} never fired")
