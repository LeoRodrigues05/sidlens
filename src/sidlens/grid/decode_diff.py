"""The grid's DiffGRM test decoder (`joint_catalog_v1`), with intervention hooks.

OneDiffRec decodes every grid DiffGRM run with
`DIFF_GRM/joint_decode.decode_joint_catalog` (vendored in
`vendor/onediffrec_v3`): digits revealed left to right, at each step only the
codes some catalog SID allows after the revealed prefix, a beam of `beam_size`
partial SIDs per user ranked by summed log-softmax, ties broken by a stable
sort (previous beam rank, then code). `decode` below is that function
statement for statement, with additions that do nothing unless used:

    encoded           encoder states to decode from, instead of running the
                      encoder on `batch` (an encoder intervention is applied by
                      the caller, e.g. `interventions.diffusion.encoder_attention_masks`)
    decoder_context   callable(owners, column, states) -> context manager entered
                      around each decoder forward (a cross-attention knockout or
                      reweighting for the rows of that step), or None
    step_encoded      callable(column) -> encoder states read at that reveal
                      step instead of `encoded` (another history's, for a
                      step-level switch), or None to keep `encoded`
    reveal_order      besides the evaluator's "left-to-right" / "random", an
                      explicit permutation of the target positions (the
                      catalog check follows it: legal codes given the
                      positions revealed so far)

With neither, the output equals the vendored decoder bit for bit
(`tests/grid/test_decode_diff.py`); the reproduction check (V2) proves it on
each run's saved test lists. Because the same function runs every condition of
an experiment, a condition differs from the baseline only by its intervention.

Traps
-----
1. **Batch shape.** GPU matmuls are not batch-shape invariant, so reproducing a
   saved list exactly needs the evaluator's batch size (8 users per call).
   `decode_cohort` keeps it.
2. **Owners.** Every decoder row belongs to one user (`owners`); a hook that
   builds a per-row mask must index by `owners`, never by row position, because
   the beam is reordered and resized at every step.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from sidlens.grid import v3code

EVAL_BATCH = 8


def _jd():
    return v3code.joint_decode()


@torch.inference_mode()
def decode(model, batch, catalog, *, beam_size=64, top_k=10, reveal_order="left-to-right", seed=42,
           encoded=None, decoder_context=None, step_encoded=None):
    """`joint_decode.decode_joint_catalog`, plus `encoded` and `decoder_context` (see module doc)."""
    jd = _jd()
    if beam_size < 1 or top_k < 1 or beam_size < top_k:
        raise ValueError("beam_size must be >= top_k >= 1")
    n_digit = int(model.n_digit)
    width = int(model.mask_emb_table.num_embeddings)
    if width % n_digit:
        raise ValueError("checkpoint target width is not divisible by n_digit")
    n_targets = width // n_digit
    codes = sorted(set(tuple(int(code) for code in row) for row in catalog))
    if not codes or any(len(row) != n_digit or any(
            code < 0 or code >= model.codebook_size for code in row) for row in codes):
        raise ValueError("catalog has missing, invalid-width, or out-of-range SID codes")
    if isinstance(reveal_order, str):
        order = jd.make_reveal_order(width, reveal_order, seed)
    else:
        order = [int(x) for x in reveal_order]
        if sorted(order) != list(range(width)):
            raise ValueError(f"reveal order {order} is not a permutation of {width} positions")
    transitions = jd._catalog_transitions(codes, n_digit, n_targets, order)
    model.eval()
    if encoded is None:
        encoded = model.forward(batch, return_loss=False).hidden_states
    if not torch.isfinite(encoded).all():
        raise ValueError("history encoder produced non-finite values (check empty histories)")
    device = encoded.device
    batch_size = encoded.shape[0]
    states = torch.full((batch_size, width), -1, dtype=torch.long, device=device)
    owners = torch.arange(batch_size, device=device)
    scores = torch.zeros(batch_size, device=device)
    paths = torch.empty((batch_size, 0), device=device)
    for column in order:
        ctx = decoder_context(owners, column, states) if decoder_context is not None else None
        memory = step_encoded(column) if step_encoded is not None else None
        memory = encoded if memory is None else memory
        with (ctx if ctx is not None else nullcontext()):
            outputs = model.forward_decoder_only({
                "decoder_input_ids": states.clamp_min(0),
                "mask_positions": states.eq(-1),
                "encoder_hidden": memory[owners],
            }, digit=column, use_cache=True)
        logits = outputs.logits.float()
        if tuple(logits.shape) != (len(states), int(model.codebook_size)):
            raise ValueError(f"unexpected decoder score shape {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("decoder produced non-finite logits")
        token_logp = logits.log_softmax(-1)
        legal = torch.zeros_like(token_logp, dtype=torch.bool)
        tree, positions = transitions[column // n_digit]
        prior = positions[:positions.index(column)]
        for row, state in enumerate(states.detach().cpu().tolist()):
            legal[row, tree[tuple(state[position] for position in prior)]] = True
        candidate_scores = (scores[:, None] + token_logp).masked_fill(~legal, -torch.inf)
        next_states, next_scores, next_owners, next_paths = [], [], [], []
        for owner in range(batch_size):
            parents = torch.where(owners == owner)[0]
            flattened = candidate_scores[parents].flatten()
            count = min(beam_size, int(torch.isfinite(flattened).sum()))
            if count == 0:
                raise RuntimeError("catalog-constrained beam exhausted unexpectedly")
            indices = torch.argsort(flattened, descending=True, stable=True)[:count]
            chosen_parents = parents[indices // model.codebook_size]
            chosen_tokens = indices % model.codebook_size
            state = states[chosen_parents].clone()
            state[:, column] = chosen_tokens
            next_states.append(state)
            next_scores.append(flattened[indices])
            next_owners.append(torch.full((count,), owner, device=device, dtype=torch.long))
            next_paths.append(torch.cat([
                paths[chosen_parents], token_logp[chosen_parents, chosen_tokens, None]
            ], dim=1))
        states, scores, owners, paths = map(torch.cat, (
            next_states, next_scores, next_owners, next_paths
        ))
    result = []
    for owner in range(batch_size):
        indices = torch.where(owners == owner)[0][:top_k]
        result.append([{
            "codes": states[index].reshape(n_targets, n_digit).cpu().tolist(),
            "score": float(scores[index]),
            "token_logprobs": paths[index].cpu().tolist(),
        } for index in indices])
    return result


def decode_cohort(model, histories, history_mask, catalog, *, beam_size=50, top_k=10, seed=42,
                  batch_size=EVAL_BATCH, encode_fn=None, context_fn=None, step_fn=None,
                  reveal_order="left-to-right", device="cuda"):
    """Decode a whole cohort in the evaluator's batches of `batch_size` users.

    histories [N, S, d] long (pad -1), history_mask [N, S] bool (numpy or torch).
    encode_fn(model, batch, rows) -> encoder states for those rows, or None to
    run the encoder as the evaluator does. context_fn(rows) -> a
    `decoder_context` for `decode` (rows are cohort row indices; map a step's
    `owners` through them), or None. step_fn(rows) -> a `step_encoded` for those
    rows, or None. Returns one hypothesis list per row.
    """
    histories = torch.as_tensor(histories)
    history_mask = torch.as_tensor(history_mask, dtype=torch.bool)
    out = []
    for start in range(0, len(histories), batch_size):
        rows = list(range(start, min(start + batch_size, len(histories))))
        batch = {"history_sid": histories[rows].to(device), "history_mask": history_mask[rows].to(device)}
        encoded = encode_fn(model, batch, rows) if encode_fn is not None else None
        ctx = context_fn(rows) if context_fn is not None else None
        step = step_fn(rows) if step_fn is not None else None
        out.extend(decode(model, batch, catalog, beam_size=beam_size, top_k=top_k, seed=seed,
                          reveal_order=reveal_order, encoded=encoded, decoder_context=ctx,
                          step_encoded=step))
    return out


def prediction_records(decoded, labels, item_codes, examples=None):
    """OneDiffRec's record schema (`joint_decode.prediction_records`)."""
    return _jd().prediction_records(decoded, labels, item_codes, examples)
