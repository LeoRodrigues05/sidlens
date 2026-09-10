"""Pilot checks against the frozen next-item DiffGRM inference path.

The fixed-order decoder is the useful reference: unlike confidence search,
its archived implementation already expands the final digit fully. Padding
and illegal fallback rows are removed before comparing unique SID rankings.
This checks reconstruction and implementation on a pilot, not reproduction
of the archived full-cohort metrics, which the runner checks separately.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch


class ValidationError(RuntimeError):
    """The matched experiment no longer follows its specified inference path."""


def _ranked_legal_unique(codes, allowed_sids):
    """Remove repeated padding and illegal fallback without inventing ranks."""
    rows = codes.detach().cpu().tolist() if torch.is_tensor(codes) else codes
    ranked = []
    for row in rows:
        seen, kept = set(), []
        for values in row:
            sid = tuple(values)
            if sid in allowed_sids and sid not in seen:
                seen.add(sid)
                kept.append(sid)
        ranked.append(kept)
    return ranked


def _assert_close(name, actual, expected, *, atol=1e-4, rtol=1e-4):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValidationError(f"{name}: nonfinite values")
    maximum = float((actual - expected).abs().max().item())
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValidationError(f"{name}: maximum absolute difference {maximum:g}")
    return {"max_absolute_difference": maximum, "atol": atol, "rtol": rtol}


def _decoder(model, hidden, codes, mask, *, cache=None):
    # use_cache=False is NOT equivalent in the frozen model: it falls back
    # to unprojected encoder states as K,V. Fresh projected inference requires
    # use_cache=True even when no cache is supplied/reused.
    return model.forward_decoder_only(
        {"encoder_hidden": hidden, "decoder_input_ids": codes,
         "mask_positions": mask.float()},
        return_loss=False, digit=None, use_cache=True, past_key_values=cache,
    )


def _legacy_fixed(model, hidden, allowed_sids):
    from genrec.models.DIFF_GRM.beam import iterative_mask_decode

    class LegalityTokenizer:
        mask_token = -1

        @staticmethod
        def codebooks_to_item_id(codes):
            return 1 if tuple(codes) in allowed_sids else None

    previous = model.config
    model.config = deepcopy(previous)
    model.config.update({
        "current_split": "test",
        "dedup_strategy": "simple",
        "vectorized_beam_search": {
            "beam_act": 64, "beam_max": 64, "top_k_final": 10,
            "neg_inf_fp32": -1e9, "neg_inf_fp16": -65504.0,
        },
        "random_beam": {"beam_act": 64, "beam_max": 64, "seed": 42},
    })
    devices = [hidden.device.index] if hidden.is_cuda else []
    try:
        # The legacy seed setter also changes CUDA RNG state. Restore every
        # touched generator after this reference call.
        with torch.random.fork_rng(devices=devices):
            result = iterative_mask_decode(
                model, hidden, n_return_sequences=10,
                tokenizer=LegalityTokenizer(), mode="random", rand_cfg={},
            )
    finally:
        model.config = previous
    return result[0] if isinstance(result, tuple) else result


@torch.inference_mode()
def verify_model_and_decoder(
    model, histories, history_mask, allowed_sids, decode_fn: Callable,
):
    """Verify up to eight paired users; return JSON-ready checks or raise.

    ``decode_fn`` follows ``sidlens.analysis.matched_decode.decode``. History
    tensors contain raw SID codes with -1 right-padding, plus a boolean mask.
    The model must already be loaded on its inference device in eval mode.
    """
    if model.training:
        raise ValidationError("pilot validation requires model.eval()")
    if not 1 <= len(histories) <= 8:
        raise ValidationError("pilot validation expects one to eight histories")
    if model.codebook_size < 64:
        raise ValidationError("historical fixed64 reference needs vocabulary >=64")
    allowed = {tuple(sid) for sid in allowed_sids}
    if not allowed:
        raise ValidationError("empty legal catalogue")
    device = next(model.parameters()).device
    histories = histories.to(device)
    history_mask = history_mask.to(device)
    hidden = model(
        {"history_sid": histories, "history_mask": history_mask},
        return_loss=False,
    ).hidden_states
    singles = torch.cat([
        model({"history_sid": histories[i:i + 1],
               "history_mask": history_mask[i:i + 1]},
              return_loss=False).hidden_states
        for i in range(len(histories))
    ])
    checks = {"encoder_batch_vs_single": _assert_close("encoder", hidden, singles)}

    depth = int(model.n_digit)
    codes = torch.zeros((len(histories), depth), dtype=torch.long, device=device)
    all_mask = torch.ones_like(codes, dtype=torch.bool)
    full = _decoder(model, hidden, codes, all_mask)
    full_single = torch.cat([
        _decoder(model, singles[i:i + 1], codes[i:i + 1], all_mask[i:i + 1]).logits
        for i in range(len(histories))
    ])
    checks["allmask_logits_batch_vs_single"] = _assert_close(
        "allmask logits", full.logits, full_single)
    # Different revealed codes per user make erroneous beam/history alignment
    # visible, while staying independent of evaluation targets.
    partial_codes = histories[:, 0].clamp_min(0)
    partial_mask = all_mask.clone()
    partial_mask[:, 0] = False
    fresh = _decoder(model, hidden, partial_codes, partial_mask)
    single = torch.cat([
        _decoder(model, singles[i:i + 1], partial_codes[i:i + 1],
                 partial_mask[i:i + 1]).logits
        for i in range(len(histories))
    ])
    checks["partial_logits_batch_vs_single"] = _assert_close(
        "partial logits", fresh.logits, single)
    cross_only = [(None, layer[1]) for layer in full.past_key_values]
    reused = _decoder(model, hidden, partial_codes, partial_mask, cache=cross_only)
    checks["cross_cache_vs_fresh_projected_logits"] = _assert_close(
        "cross-cache logits", reused.logits, fresh.logits)

    order = torch.randperm(depth, generator=torch.Generator().manual_seed(42)).tolist()
    kwargs = {"beam_width": 64, "policy": "fixed", "order": order,
              "top_k": 10, "allowed_sids": allowed}
    matched = decode_fn(model, hidden, decoder_chunk_size=256, **kwargs)
    split = [decode_fn(model, singles[i:i + 1], decoder_chunk_size=17, **kwargs)
             for i in range(len(histories))]
    split_codes = torch.cat([result.codes for result in split])
    if not torch.equal(matched.codes, split_codes):
        count = int((matched.codes != split_codes).flatten(1).any(1).sum().item())
        raise ValidationError(
            f"matched fixed64 batch/chunk ranking differs on {count} pilot users; "
            "inspect numerical ties or encoder/cache alignment before continuing")
    split_scores = torch.cat([result.scores for result in split])
    finite = torch.isfinite(matched.scores)
    if finite.any():
        checks["matched_scores_batch_vs_single"] = _assert_close(
            "matched scores", matched.scores[finite], split_scores[finite])
    checks["matched_rankings_batch_vs_single"] = {"identical": True}

    reference = _legacy_fixed(model, hidden, allowed)
    reference_ranked = _ranked_legal_unique(reference, allowed)
    matched_ranked = _ranked_legal_unique(matched.codes, allowed)
    differing = [i for i, (old, new) in enumerate(zip(reference_ranked, matched_ranked))
                 if old != new]
    if differing:
        raise ValidationError(
            "matched fixed64 differs from frozen fixed64 unique legal ranking "
            f"on pilot rows {differing}; investigate ties, math, or filtering")
    checks["matched_fixed64_vs_frozen_fixed64"] = {
        "identical_unique_legal_rankings": True, "users": len(histories),
        "fixed_order_seed": 42, "fixed_order_zero_based": order,
        "padding_normalization": "remove repeated final SID and illegal fallback",
    }
    return {"status": "passed", "users": len(histories), "checks": checks}
