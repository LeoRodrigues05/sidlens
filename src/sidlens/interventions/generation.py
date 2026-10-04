"""Attention-edge knockout inside HF `generate()` (KV-cached, beam-searched decoding).

`interventions.attention` edits a mask the caller builds for ONE teacher-forced
forward. Free-running decoding is different in three ways, and each is a trap:

1. **The edges depend on the beam.** "Block the read of history items whose
   prefix matches the generated prefix" needs each beam's own generated digits,
   which exist only inside the decoding loop and are reordered every step.
   `generate()` hands the full running sequences to
   `prepare_inputs_for_generation` at every step (transformers 4.57
   `_beam_search`); the wrapper here records them, and a caller-supplied plan
   turns (sequences, step) into edges. The plan never sees a stale beam order.
2. **The mask is built by transformers, per forward.** With left padding it is
   a (B, 1, q, kv) boolean tensor; when a batch has no padding it can be None
   (the kernel then uses `is_causal`). The hook edits a clone of whatever
   arrives, materializing an explicit causal mask when it is None, and it does
   so in EVERY condition including the empty-plan baseline, so conditions
   differ only in the removed edges, never in the kernel path. The number of
   materialized forwards is recorded.
3. **Silent no-ops.** A key that is padding, in the future, or already masked
   would make a "knockout" do nothing. Every removed edge must be live before
   removal, the query row must keep at least one key, and every hooked layer
   must fire exactly once per model forward, or the call raises.

Only the listed layers are edited; all heads are knocked out together (the
(B, 1, q, kv) mask has no head axis). Observation of what was removed is
returned in `stats` (edges per step, forwards, materialized masks).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

# plan(sequences (B, cur_len) LongTensor, step, q_len, kv_len) ->
#     list of (row, q_index, [key kv-positions]) edges to remove in this forward
Plan = Callable[..., list]


@contextmanager
def generation_knockout(model, layers: Sequence[int], plan: Plan, prompt_len: int) -> Iterator[dict]:
    """Apply `plan`'s edge removals in `layers` on every forward of `generate()`.

    `prompt_len` is the (left-padded) prompt length, so step = cur_len - prompt_len
    is the number of tokens generated before this forward.
    """
    import torch

    attn = [layer.self_attn for layer in model.model.layers]
    layers = sorted(set(layers))
    if not layers or layers[0] < 0 or layers[-1] >= len(attn):
        raise IndexError(f"layers must lie in 0..{len(attn) - 1}")
    stats = {"forwards": 0, "materialized": 0, "edges": 0, "edges_by_step": {}, "fired": {}}
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
                    raise RuntimeError("no running sequences recorded for this forward; "
                                       "is this forward inside generate()?")
                kv = seqs.shape[1]
                if m is None:
                    i = torch.arange(q, device=h.device)[:, None] + (kv - q)
                    j = torch.arange(kv, device=h.device)[None, :]
                    m2 = (j <= i)[None, None].expand(B, 1, q, kv).clone()
                    stats["materialized"] += 1
                else:
                    if m.dtype != torch.bool or m.shape[0] != B or m.shape[2] != q:
                        raise TypeError(f"unexpected attention mask {m.dtype} {tuple(m.shape)}")
                    m2 = m.clone()
                    if m.shape[-1] != kv:
                        raise RuntimeError(f"mask spans {m.shape[-1]} keys but {kv} tokens are "
                                           f"in the running sequences")
                step = seqs.shape[1] - prompt_len
                edges = plan(seqs, step, q, kv)
                n = 0
                for row, qi, keys in edges:
                    if not keys:
                        continue
                    k = torch.as_tensor(keys, device=h.device)
                    live = m2[row, 0, qi, k]
                    if not bool(live.all()):
                        raise ValueError(f"step {step} row {row}: a knocked-out key is already "
                                         f"masked (padding, future or duplicate); no-op knockout")
                    m2[row, 0, qi, k] = False
                    if not bool(m2[row, 0, qi].any()):
                        raise ValueError(f"step {step} row {row}: knockout left the query no keys")
                    n += len(keys)
                stats["edges"] += n
                stats["edges_by_step"][step] = stats["edges_by_step"].get(step, 0) + n
                state["mask_in"], state["mask_out"] = m, m2
            elif m is not state["mask_in"]:
                raise RuntimeError(f"layer {L} received a different mask than earlier layers "
                                   f"in the same forward")
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
        del model.prepare_inputs_for_generation            # restore the class method
    missing = [(f, L) for f in range(stats["forwards"]) for L in layers if (f, L) not in stats["fired"]]
    if missing:
        raise RuntimeError(f"hooked layers did not fire in {len(missing)} (forward, layer) slots, "
                           f"e.g. {missing[:3]}")
