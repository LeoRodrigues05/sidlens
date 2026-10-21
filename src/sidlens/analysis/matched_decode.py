"""Matched next-item diffusion beam search for a reveal-position ablation.

Both policies use the same token probabilities, accumulated log-probability
scores, beam limit, full expansion at every step, and final filtering. Only the
allowed position differs: confidence considers every masked position; fixed
considers the next position in a supplied permutation. This is position-token
path beam search, not greedy selection of one confidence-ranked position.

Unlike the archived vendor decoder, both policies expand the last digit fully.
Decoder self-attention is recomputed on each partial SID. Projected encoder
cross-attention keys/values are reused, preserving the vendor inference math.
Every decoder row is explicitly matched to its original encoder row,
independent of chunk size.
We keep duplicate partial paths during search, matching the path-beam design;
at the end each SID receives its best surviving path score, not a sum across
orders. These scores are ranking scores, not normalized item probabilities.

Two optional arguments open the loop without forking it. ``observer`` sees
every step (the states scored, their distributions, and what was committed)
and must not change anything. ``score_fn`` replaces the per-step scorer, which
is the only way one step's prediction can come from a different run: nothing
but the partial SID and the beam score passes between steps (self-attention is
recomputed, cross-attention K/V depend on the history alone). With both left
as None the search is the validated one, bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DecodeResult:
    """Ranked raw code SIDs and diagnostics, all tensors on the input device.

    ``codes`` is [batch, top_k, depth], with -1 in unused ranks. ``scores`` has
    -inf in those ranks; ``counts`` is the number of returned SIDs. Generated
    and invalid counts refer to surviving beam paths before deduplication;
    unique and legal-unique counts refer to distinct completed SIDs. Catalogue
    filtering, when requested, happens only after the final beam pruning.
    """

    codes: Tensor
    scores: Tensor
    counts: Tensor
    generated_counts: Tensor
    unique_counts: Tensor
    invalid_counts: Tensor
    legal_unique_counts: Tensor
    beam_sizes: tuple[int, ...]


@dataclass(frozen=True)
class DecodeStep:
    """One reveal step, as the search saw it. Tensors are on the input device.

    ``states`` [batch, active, depth] are the partial SIDs that were scored
    (-1 = masked) and ``state_scores`` [batch, active] their accumulated
    log-probabilities. ``log_probs`` [batch, active, depth, vocabulary] are the
    per-position distributions at those states; entries at revealed positions
    are outputs the model was never trained on and must not be read. The
    commit that followed is ``parents`` / ``positions`` / ``codes`` / ``scores``
    [batch, kept]: new path j extends ``states[:, parents[:, j]]`` by writing
    ``codes[:, j]`` at ``positions[:, j]``. The tensors are the search's own:
    an observer that keeps them must not modify them in place.
    """

    step: int
    states: Tensor
    state_scores: Tensor
    log_probs: Tensor
    parents: Tensor
    positions: Tensor
    codes: Tensor
    scores: Tensor


def _top_candidates(scores: Tensor, width: int) -> tuple[Tensor, Tensor]:
    """Top finite candidates, breaking exact ties by flattened index.

    Sorting is stable, so equal scores prefer lower parent rank, then position,
    then code. Explicit tie handling avoids torch.topk's unspecified tie order.
    """
    keep = min(width, scores.shape[1])
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :keep]
    return scores.gather(1, indices), indices


def _project_cross_cache(model, encoder_hidden: Tensor):
    """Reuse only the immutable cross-attention part of the vendor cache.

    Vendor ``use_cache=False`` with no past cache falls back to raw encoder
    states as K/V instead of learned projections, changing its predictions.
    We explicitly reproduce its ``use_cache=True`` projected K/V computation,
    then always supply None for self-attention past. A generic test model
    without these vendor attributes uses independent fresh cached forwards.
    """
    if not hasattr(model, "decoder_blocks") or not hasattr(model, "n_embd"):
        return None
    cross_cache = []
    width = model.n_embd
    for block in model.decoder_blocks:
        projected = block.cross_attn.qkv(encoder_hidden)
        cross_cache.append((projected[..., width : 2 * width], projected[..., 2 * width :]))
    return cross_cache


def score_partial(model, encoder_hidden: Tensor, cross_cache, partial: Tensor,
                  history_rows: Tensor) -> Tensor:
    """Log-softmax digit distributions [rows, depth, vocabulary] at partial SIDs.

    ``partial`` [rows, depth] holds revealed codes, -1 where masked;
    ``history_rows`` [rows] names each row's history in ``encoder_hidden`` and
    ``cross_cache`` (from ``_project_cross_cache``, or None for a generic test
    model). This is the search's own per-chunk forward; a ``score_fn`` built on
    it differs from the default only in the histories it points at. The raw
    logits are checked here, before the softmax, as the search always did.
    """
    past_key_values = None if cross_cache is None else [
        (None, (key.index_select(0, history_rows), value.index_select(0, history_rows)))
        for key, value in cross_cache
    ]
    output = model.forward_decoder_only(
        {
            "decoder_input_ids": partial.clamp_min(0),
            "mask_positions": (partial < 0).to(torch.float32),
            "encoder_hidden": encoder_hidden.index_select(0, history_rows),
        },
        return_loss=False,
        digit=None,
        past_key_values=past_key_values,
        use_cache=True,
    )
    logits = output.logits
    if tuple(logits.shape) != (partial.shape[0], partial.shape[1], int(model.codebook_size)):
        raise ValueError(f"unexpected decoder logits shape {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise ValueError("decoder produced nonfinite logits")
    return torch.log_softmax(logits.float(), dim=-1)


@torch.inference_mode()
def decode(
    model,
    encoder_hidden: Tensor,
    *,
    beam_width: int,
    policy: str,
    order: Sequence[int] | None = None,
    top_k: int = 10,
    allowed_sids: Collection[tuple[int, ...]] | None = None,
    decoder_chunk_size: int = 256,
    observer: Callable[[DecodeStep], None] | None = None,
    score_fn: Callable[[int, Tensor, Tensor], Tensor] | None = None,
) -> DecodeResult:
    """Decode one SID per history with only reveal-position choice varied.

    Set the model to eval mode before calling. ``order`` is required for fixed
    decoding and must be a permutation of range(model.n_digit). A beam is a
    maximum number of paths: if fewer finite candidates exist, only those can
    be active (e.g. fixed decoding at step one when beam_width > vocabulary).
    ``decoder_chunk_size`` limits repeated-history decoder rows in one forward.
    All logits and accumulation scores are evaluated as float32 for ranking.

    ``observer`` is called once per step with a ``DecodeStep``. ``score_fn(step,
    partial, history_rows)`` returns float32 log-probabilities [rows, depth,
    vocabulary] for one chunk and replaces ``score_partial`` on this function's
    ``encoder_hidden``, which then only sets the batch size and device. Its
    output is checked for shape and for NaN or +inf; -inf is a legal
    log-probability.
    """
    if getattr(model, "training", False):
        raise ValueError("matched decoding requires model.eval()")
    if policy not in {"confidence", "fixed"}:
        raise ValueError("policy must be 'confidence' or 'fixed'")
    if beam_width < 1 or top_k < 1 or decoder_chunk_size < 1:
        raise ValueError("beam_width, top_k, and decoder_chunk_size must be positive")
    if encoder_hidden.ndim != 3 or encoder_hidden.shape[0] < 1:
        raise ValueError("encoder_hidden must have shape [nonempty batch, sequence, embedding]")
    depth, vocabulary = int(model.n_digit), int(model.codebook_size)
    if depth < 1 or vocabulary < 1:
        raise ValueError("SID depth and codebook size must be positive")
    if policy == "fixed":
        if order is None or sorted(order) != list(range(depth)):
            raise ValueError("fixed order must be a permutation of all SID positions")
        order = tuple(int(position) for position in order)
    elif order is not None:
        raise ValueError("order is only meaningful for fixed decoding")
    catalogue = None if allowed_sids is None else {tuple(sid) for sid in allowed_sids}
    if catalogue is not None and any(
        len(sid) != depth or any(code < 0 or code >= vocabulary for code in sid)
        for sid in catalogue
    ):
        raise ValueError("allowed_sids must contain legal raw code tuples of the model depth")

    device = encoder_hidden.device
    batch_size = encoder_hidden.shape[0]
    beam_ids = torch.full((batch_size, 1, depth), -1, device=device, dtype=torch.long)
    beam_scores = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
    beam_sizes: list[int] = []
    batch_indices = torch.arange(batch_size, device=device)[:, None]
    cross_cache = None if score_fn is not None else _project_cross_cache(model, encoder_hidden)

    for step in range(depth):
        active = beam_ids.shape[1]
        flat_ids = beam_ids.reshape(-1, depth)
        state_scores = beam_scores
        chunks = []
        for start in range(0, flat_ids.shape[0], decoder_chunk_size):
            stop = min(start + decoder_chunk_size, flat_ids.shape[0])
            partial = flat_ids[start:stop]
            # Flattening [B, beam, depth] is history-major, including chunks
            # that cross a history boundary. Never slice a wider repeated
            # encoder buffer as a substitute for this explicit row mapping.
            original_rows = torch.arange(start, stop, device=device) // active
            if score_fn is None:
                chunk = score_partial(model, encoder_hidden, cross_cache, partial, original_rows)
            else:
                chunk = score_fn(step, partial, original_rows)
                if tuple(chunk.shape) != (stop - start, depth, vocabulary):
                    raise ValueError(f"score_fn returned shape {tuple(chunk.shape)}")
                if chunk.dtype != torch.float32:
                    raise ValueError(f"score_fn returned {chunk.dtype}, the search ranks in float32")
                if torch.isnan(chunk).any() or torch.isposinf(chunk).any():
                    raise ValueError("score_fn returned NaN or +inf log-probabilities")
            chunks.append(chunk)
        log_probs = torch.cat(chunks, dim=0).reshape(batch_size, active, depth, vocabulary)
        allowed_positions = beam_ids < 0
        if policy == "fixed":
            position_mask = torch.arange(depth, device=device) == order[step]
            allowed_positions = allowed_positions & position_mask
        candidates = (beam_scores[:, :, None, None] + log_probs).masked_fill(
            ~allowed_positions[:, :, :, None], -torch.inf
        )
        # The fixed first step has only K branches, regardless of beam width;
        # both policies subsequently use the same cap on available path count.
        available = active * (depth - step if policy == "confidence" else 1) * vocabulary
        beam_scores, selected = _top_candidates(
            candidates.reshape(batch_size, -1), min(beam_width, available)
        )
        parents = selected // (depth * vocabulary)
        positions = (selected // vocabulary) % depth
        codes = selected % vocabulary
        if observer is not None:
            observer(DecodeStep(step=step, states=beam_ids, state_scores=state_scores,
                                log_probs=log_probs, parents=parents, positions=positions,
                                codes=codes, scores=beam_scores))
        beam_ids = beam_ids[batch_indices, parents].clone()
        beam_ids.scatter_(2, positions[:, :, None], codes[:, :, None])
        beam_sizes.append(beam_ids.shape[1])

    result_ids = torch.full((batch_size, top_k, depth), -1, device=device, dtype=torch.long)
    result_scores = torch.full((batch_size, top_k), -torch.inf, device=device, dtype=torch.float32)
    returned, generated, unique, invalid, legal_unique = [], [], [], [], []
    # At most beam_width completed paths per history; moving once to CPU
    # avoids synchronizing the accelerator once per membership/dedup check.
    cpu_ids, cpu_scores = beam_ids.cpu().tolist(), beam_scores.cpu().tolist()
    for row, (paths, scores) in enumerate(zip(cpu_ids, cpu_scores, strict=True)):
        seen: set[tuple[int, ...]] = set()
        kept_ids, kept_scores = [], []
        n_invalid, n_legal = 0, 0
        for sid_values, score in zip(paths, scores, strict=True):
            sid = tuple(sid_values)
            legal = catalogue is None or sid in catalogue
            n_invalid += int(not legal)
            if sid in seen:
                continue
            seen.add(sid)
            n_legal += int(legal)
            if legal and len(kept_ids) < top_k:
                kept_ids.append(sid)
                kept_scores.append(score)
        count = len(kept_ids)
        if count:
            result_ids[row, :count] = torch.tensor(kept_ids, device=device, dtype=torch.long)
            result_scores[row, :count] = torch.tensor(kept_scores, device=device, dtype=torch.float32)
        returned.append(count)
        generated.append(len(paths))
        unique.append(len(seen))
        invalid.append(n_invalid)
        legal_unique.append(n_legal)

    def counts_tensor(values):
        return torch.tensor(values, device=device, dtype=torch.long)

    return DecodeResult(
        codes=result_ids,
        scores=result_scores,
        counts=counts_tensor(returned),
        generated_counts=counts_tensor(generated),
        unique_counts=counts_tensor(unique),
        invalid_counts=counts_tensor(invalid),
        legal_unique_counts=counts_tensor(legal_unique),
        beam_sizes=tuple(beam_sizes),
    )
