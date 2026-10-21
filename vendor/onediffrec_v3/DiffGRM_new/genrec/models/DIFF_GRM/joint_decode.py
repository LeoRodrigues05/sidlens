"""Catalog-constrained decoding of the complete trained masked target sequence.

This is an explicitly separate inference protocol. It uses the original model's
decoder with fresh projected cross-attention on each reveal, matching training.
It never reuses self-attention state across changing masked sequences. Slot-specific mask
embeddings are used while masked; the existing architecture still has no slot
embedding on revealed tokens. Nothing here changes that trained architecture.
"""
from __future__ import annotations

import random
from collections import defaultdict

import torch


PROTOCOL = "joint_catalog_v1"
SCORE_DEFINITION = (
    "sum of full-codebook log-softmax values along one fixed masked-reveal "
    "path, ranked across the complete ordered target sequence; a heuristic "
    "masked-decoding score, not a normalized diffusion joint likelihood"
)


def sid_text(codes):
    return "".join(f"<{chr(97 + digit)}_{int(code)}>" for digit, code in enumerate(codes))


def catalog_from_tokenizer(tokenizer):
    return {
        str(item): tuple(int(token) - tokenizer.sid_offset - digit * tokenizer.codebook_size
                         for digit, token in enumerate(tokens))
        for item, tokens in tokenizer.item2tokens.items()
    }


def inverse_catalog(item_codes):
    result = defaultdict(list)
    for item, codes in item_codes.items():
        result[tuple(codes)].append(str(item))
    return {codes: sorted(items) for codes, items in result.items()}


def make_reveal_order(width, mode="left-to-right", seed=42):
    order = list(range(width))
    if mode == "random":
        random.Random(seed).shuffle(order)
    elif mode != "left-to-right":
        raise ValueError(f"unknown reveal order: {mode}")
    return order


def _catalog_transitions(catalog, n_digit, n_targets, order):
    transitions = {}
    for slot in range(n_targets):
        positions = [position for position in order if position // n_digit == slot]
        local = [position % n_digit for position in positions]
        tree = defaultdict(set)
        for codes in catalog:
            for step, digit in enumerate(local):
                tree[tuple(codes[d] for d in local[:step])].add(codes[digit])
        transitions[slot] = ({key: sorted(values) for key, values in tree.items()}, positions)
    return transitions


@torch.inference_mode()
def decode_joint_catalog(model, batch, catalog, *, beam_size=64, top_k=10,
                         reveal_order="left-to-right", seed=42):
    """Return distinct complete hypotheses per example, never padded duplicates.

    Each hypothesis has ``codes`` [target slots, SID digits], ``score``, and
    per-reveal ``token_logprobs``. Every forward uses all trained target slots.
    A fixed order avoids counting multiple reveal paths as separate candidates.
    """
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
    order = make_reveal_order(width, reveal_order, seed)
    transitions = _catalog_transitions(codes, n_digit, n_targets, order)
    model.eval()
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
        outputs = model.forward_decoder_only({
            "decoder_input_ids": states.clamp_min(0),
            "mask_positions": states.eq(-1),
            "encoder_hidden": encoded[owners],
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
            # Stable sorting provides a documented deterministic tie break:
            # previous beam rank, then code index.
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


def prediction_records(decoded, labels, item_codes, examples=None):
    """Convert model outputs into the shared exact ordered-target metric schema."""
    groups = inverse_catalog(item_codes)
    n_digit = len(next(iter(groups)))
    label_rows = labels.detach().cpu().tolist() if torch.is_tensor(labels) else labels
    records = []
    for index, (hypotheses, target) in enumerate(zip(decoded, label_rows)):
        target_codes = [target[start:start + n_digit] for start in range(0, len(target), n_digit)]
        record = dict(examples[index]) if examples is not None else {}
        record.update({
            "schema": "onediffrec.ordered-target-predictions/v1",
            "evaluation_protocol": PROTOCOL,
            "score_definition": SCORE_DEFINITION,
            "target_sids": [sid_text(codes) for codes in target_codes],
            "target_candidate_item_ids": [groups.get(tuple(codes), []) for codes in target_codes],
            "predictions": [{
                "sids": [sid_text(codes) for codes in prediction["codes"]],
                "candidate_item_ids": [groups[tuple(codes)] for codes in prediction["codes"]],
                "score": prediction["score"],
                "token_logprobs": prediction["token_logprobs"],
            } for prediction in hypotheses],
        })
        records.append(record)
    return records
