"""Exact frozen confidence decoder with an explicitly supplied beam cap.

This retains the vendor's greedy last-digit fill. It provides the literal
beam-only confidence rerun and a confidence256 historical reconstruction
anchor alongside the shared full-expansion comparison. Vendor code exposes
no path scores or internal path statistics, so those quantities are absent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LegacyDecodeResult:
    """Raw code SIDs with -1 padding and counts; vendor scores unavailable."""

    codes: Tensor
    counts: Tensor


def _normalize_padding(raw: Tensor, allowed_sids) -> LegacyDecodeResult:
    """Remove only the frozen simple-dedup decoder's documented padding.

    The vendor returns unique legal SIDs followed by copies of the last legal
    SID. If no legal path survives, it repeats one illegal fallback instead.
    Neither normalization changes a legal target's first-hit rank. Unexpected
    internal duplicates or mixed illegal/valid rows fail instead of compacting
    away genuine rank positions.
    """
    if raw.ndim != 3 or raw.shape[1] < 1:
        raise ValueError("legacy decoder returned an unexpected prediction shape")
    allowed = {tuple(sid) for sid in allowed_sids}
    result = torch.full_like(raw, -1)
    counts = []
    for index, row in enumerate(raw.cpu().tolist()):
        sids = [tuple(sid) for sid in row]
        if sids[0] not in allowed:
            if any(sid != sids[0] for sid in sids):
                raise ValueError("legacy illegal fallback is not a repeated single SID")
            counts.append(0)
            continue
        seen = set()
        count = 0
        for rank, sid in enumerate(sids):
            if sid not in allowed:
                raise ValueError("legacy decoder mixed legal predictions with illegal paths")
            if sid in seen:
                if sid != sids[rank - 1] or any(tail != sid for tail in sids[rank:]):
                    raise ValueError("legacy decoder duplicate is not trailing last-SID padding")
                break
            seen.add(sid)
            count += 1
        result[index, :count] = raw[index, :count]
        counts.append(count)
    return LegacyDecodeResult(
        codes=result,
        counts=torch.tensor(counts, dtype=torch.long, device=raw.device),
    )


@torch.inference_mode()
def decode_legacy_confidence(
    model,
    encoder_hidden: Tensor,
    *,
    beam_width: int,
    allowed_sids,
    top_k: int = 10,
) -> LegacyDecodeResult:
    """Run frozen confidence decoding without changing checkpoint or math.

    The caller loads the next1 vendor and places the model in eval mode.
    Both active beam and allocated capacity equal ``beam_width``; all other
    search behavior is the frozen confidence implementation. The original
    configuration object and CPU/CUDA RNG states are restored even on error.
    """
    from genrec.models.DIFF_GRM.beam import iterative_mask_decode

    if model.training:
        raise ValueError("legacy confidence decoding requires model.eval()")
    if beam_width < 1 or top_k < 1 or beam_width < top_k:
        raise ValueError("legacy confidence requires beam_width >= top_k >= 1")
    if beam_width > model.n_digit * model.codebook_size:
        raise ValueError("legacy confidence beam exceeds first-step position-token count")
    allowed = {tuple(sid) for sid in allowed_sids}

    class LegalityTokenizer:
        mask_token = -1

        @staticmethod
        def codebooks_to_item_id(codes):
            return 1 if tuple(codes) in allowed else None

    previous = model.config
    model.config = deepcopy(previous)
    model.config.update({
        "current_split": "test",
        "dedup_strategy": "simple",
        "vectorized_beam_search": {
            "beam_act": beam_width,
            "beam_max": beam_width,
            "top_k_final": top_k,
            "neg_inf_fp32": -1e9,
            "neg_inf_fp16": -65504.0,
        },
    })
    devices = [encoder_hidden.device.index] if encoder_hidden.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            raw = iterative_mask_decode(
                model,
                encoder_hidden,
                n_return_sequences=top_k,
                tokenizer=LegalityTokenizer(),
                mode="confidence",
                rand_cfg={},
            )
    finally:
        model.config = previous
    if isinstance(raw, tuple):
        raw = raw[0]
    if tuple(raw.shape) != (len(encoder_hidden), top_k, model.n_digit):
        raise ValueError(f"legacy confidence returned unexpected shape {tuple(raw.shape)}")
    return _normalize_padding(raw, allowed)
