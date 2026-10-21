"""Declared design and per-batch recording for the DiffGRM reveal trace.

Everything `protocol.md` says is recorded is produced by `trace_batch`, which
takes a model and tensors and touches no file, so it is tested on a
random-weight model (`tests/experiments/test_reveal_trace_experiment.py`).
`run.py` adds loading, validation gates and writing.

Conventions of the recorded arrays, all with a leading user axis:
positions that are revealed at a state carry the fill value (-1 for integers,
NaN for floats), because the model's output there was never trained; rows of a
path that was not found carry the fill value throughout.
"""

from __future__ import annotations

import torch

from sidlens.analysis import diffusion_lens as DL
from sidlens.analysis.matched_decode import decode, score_partial
from sidlens.interventions.diffusion import project_cross_cache
from sidlens.models import diffusion

EXP1_CELLS = tuple(f"diff-next1-{quantizer}-{depth}cb-{width}" for quantizer in ("MQ", "rqkmeans", "rqvae")
                   for depth in (3, 4, 5) for width in (128, 512))
CELLS = EXP1_CELLS + ("diff-next1-MQ-5cb-256", "diff-next1-rqvae-3cb-256")
BEAMS = (1, 64, 256)
PRIMARY_BEAM = 256
TOP_CODES = 8
BATCH, CHUNK = 32, 1024
BOOTSTRAP_SEED = 20261013


def _spread(value, keep, n_users, fill, dtype):
    """[found * depth, ...] per-state values -> [users, depth, ...], `fill` for users without a path."""
    depth = value.shape[0] // max(len(keep), 1)
    out = torch.full((n_users, depth) + tuple(value.shape[1:]), fill, dtype=dtype)
    out[keep.cpu()] = value.reshape(len(keep), depth, *value.shape[1:]).to(dtype).cpu()
    return out


def _trace_path(model, enc, cache, path, targets, target_rows, catalogue):
    """Re-score every state of one path per user; see protocol.md, "Recorded per path state"."""
    n_users, depth, vocabulary = enc.shape[0], int(model.n_digit), int(model.codebook_size)
    blocks, top = len(model.decoder_blocks), min(TOP_CODES, vocabulary)
    rec = {"found": path["found"], "order": path["order"].to(torch.int16),
           "codes": path["codes"].to(torch.int16), "scores": path["scores"]}
    keep = path["found"].nonzero()[:, 0].to(enc.device)
    if len(keep) == 0:
        empty = {"top_codes": ((depth, top), -1, torch.int16), "top_logp": ((depth, top), float("nan"), torch.float16),
                 "entropy": ((depth,), float("nan"), torch.float16), "target_logp": ((depth,), float("nan"), torch.float16),
                 "target_rank": ((depth,), -1, torch.int16), "guess": ((depth,), -1, torch.int16),
                 "guess_in_catalogue": ((), False, torch.bool), "catalogue_top": ((depth,), -1, torch.int16),
                 "catalogue_sid_rank": ((), -1, torch.int16), "catalogue_item_rank_lo": ((), -1, torch.int16),
                 "catalogue_item_rank_hi": ((), -1, torch.int16), "target_reachable": ((), False, torch.bool),
                 "block_top": ((blocks, depth), -1, torch.int16), "block_target_logp": ((blocks, depth), float("nan"), torch.float16),
                 "block_target_rank": ((blocks, depth), -1, torch.int16)}
        rec.update({name: torch.full((n_users, depth) + shape, fill, dtype=dtype)
                    for name, (shape, fill, dtype) in empty.items()})
        return rec, {}

    order, codes = path["order"].to(enc.device)[keep], path["codes"].to(enc.device)[keep]
    states = DL.path_partials(order, codes).reshape(-1, depth)               # [found * depth, depth]
    rows = keep.repeat_interleave(depth)
    log_probs, residuals = DL.decoder_residuals(model, enc, cache, states, rows)
    masked, target = states < 0, targets[rows]
    nan = torch.full((), float("nan"), device=enc.device)
    minus = torch.full((), -1, dtype=torch.long, device=enc.device)

    target_logp = log_probs.gather(-1, target[..., None])[..., 0]
    target_rank = (log_probs > target_logp[..., None]).sum(-1)
    top_logp, top_codes = log_probs.topk(top, dim=-1)
    guess = DL.current_guess(log_probs, states)
    scores = DL.catalogue_scores(log_probs, states, catalogue)
    ranks = DL.target_ranks(scores, target_rows[rows], catalogue.sizes)
    by_block = DL.layer_log_probs(model, residuals)                           # [blocks, rows, depth, K]
    block_target = by_block.gather(-1, target[None, ..., None].expand(blocks, -1, -1, 1))[..., 0]
    block_rank = (by_block > block_target[..., None]).sum(-1)

    def spread(value, fill, dtype):
        return _spread(value, keep, n_users, fill, dtype)

    rec.update({
        "top_codes": spread(torch.where(masked[..., None], top_codes, minus), -1, torch.int16),
        "top_logp": spread(torch.where(masked[..., None], top_logp, nan), float("nan"), torch.float16),
        "entropy": spread(torch.where(masked, torch.special.entr(log_probs.exp()).sum(-1), nan),
                          float("nan"), torch.float16),
        "target_logp": spread(torch.where(masked, target_logp, nan), float("nan"), torch.float16),
        "target_rank": spread(torch.where(masked, target_rank, minus), -1, torch.int16),
        "guess": spread(guess, -1, torch.int16),
        "guess_in_catalogue": spread(catalogue.reachable(guess), False, torch.bool),
        "catalogue_top": spread(catalogue.sids[scores.argmax(1)], -1, torch.int16),
        "catalogue_sid_rank": spread(ranks["sid_rank"], -1, torch.int16),
        "catalogue_item_rank_lo": spread(ranks["item_rank_lo"], -1, torch.int16),
        "catalogue_item_rank_hi": spread(ranks["item_rank_hi"], -1, torch.int16),
        "target_reachable": spread(ranks["reachable"], False, torch.bool),
        "block_top": spread(torch.where(masked[None], by_block.argmax(-1), minus).permute(1, 0, 2), -1, torch.int16),
        "block_target_logp": spread(torch.where(masked[None], block_target, nan).permute(1, 0, 2),
                                    float("nan"), torch.float16),
        "block_target_rank": spread(torch.where(masked[None], block_rank, minus).permute(1, 0, 2), -1, torch.int16),
    })
    found = len(keep)
    stepwise = log_probs.reshape(found, depth, depth, vocabulary)[
        torch.arange(found, device=enc.device)[:, None], torch.arange(depth, device=enc.device)[None, :], order, codes]
    check = {"v3_path_score_max_abs_diff": float((stepwise.cumsum(1).cpu() - path["scores"][keep.cpu()]).abs().max()),
             "v4_last_block_max_abs_diff": float((by_block[-1] - log_probs).abs().max())}
    return rec, check


def _lattice(model, enc, cache, targets, chunk):
    """All partial states of each user's target SID; see protocol.md, "Target lattice"."""
    n_users, depth = targets.shape
    lattice = DL.lattice_partials(targets)                                   # [users, 2^depth - 1, depth]
    n_states = lattice.shape[1]
    flat = lattice.reshape(-1, depth)
    rows = torch.arange(n_users, device=enc.device).repeat_interleave(n_states)
    log_probs = torch.cat([score_partial(model, enc, cache, flat[i:i + chunk], rows[i:i + chunk])
                           for i in range(0, len(flat), chunk)]).reshape(n_users, n_states, depth, -1)
    masked = lattice < 0
    gold = DL.gold_log_probs(log_probs, lattice, targets[:, None, :])        # NaN where revealed
    rank = torch.where(masked, (log_probs > gold[..., None]).sum(-1), torch.full_like(lattice, -1))
    entropy = torch.where(masked, torch.special.entr(log_probs.exp()).sum(-1), torch.full_like(gold, float("nan")))
    best = DL.best_reveal_path(gold)
    return {"lattice_target_logp": gold.to(torch.float16), "lattice_target_rank": rank.to(torch.int16),
            "lattice_entropy": entropy.to(torch.float16), "target_best_score": best["best_score"],
            "target_best_order": best["best_order"].to(torch.int16), "target_greedy_score": best["greedy_score"],
            "target_greedy_order": best["greedy_order"].to(torch.int16)}


def _regret(model, enc, cache, sids, found):
    """Each digit of a decoded SID re-masked in turn; see protocol.md, "Regret"."""
    n_users, depth = sids.shape
    logp = torch.full((n_users, depth), float("nan"), dtype=torch.float16)
    rank = torch.full((n_users, depth), -1, dtype=torch.int16)
    keep = found.nonzero()[:, 0].to(enc.device)
    if len(keep):
        kept = sids.to(enc.device)[keep]
        digit = torch.arange(depth, device=enc.device)
        partial = kept[:, None, :].repeat(1, depth, 1)
        partial[:, digit, digit] = -1                                        # state d: digit d masked again
        log_probs = score_partial(model, enc, cache, partial.reshape(-1, depth), keep.repeat_interleave(depth))
        own = log_probs.reshape(len(keep), depth, depth, -1)[:, digit, digit]    # the re-masked digit's distribution
        committed = own.gather(-1, kept[..., None])[..., 0]
        logp[keep.cpu()] = committed.to(torch.float16).cpu()
        rank[keep.cpu()] = (own > committed[..., None]).sum(-1).to(torch.int16).cpu()
    return {"regret_logp": logp, "regret_rank": rank}


@torch.inference_mode()
def trace_batch(model, enc, targets, catalogue, allowed, *, beams=BEAMS, primary=PRIMARY_BEAM, chunk=CHUNK):
    """Everything protocol.md records for one batch of users.

    enc [B, S, d] encoder states, targets [B, depth] target SIDs, catalogue a
    `diffusion_lens.Catalogue` on enc's device, allowed the catalogue SIDs as a
    set of tuples. Returns ({name: CPU tensor with leading axis B}, {check:
    float}). Raises on a target outside the catalogue.
    """
    if 1 not in beams or primary not in beams:
        raise ValueError(f"beams {beams} must include 1 (the greedy path) and the primary beam {primary}")
    n_users, depth, vocabulary = enc.shape[0], int(model.n_digit), int(model.codebook_size)
    cache = project_cross_cache(model, enc)
    target_rows = catalogue.rows_of(targets)
    out, checks, traces = {}, {}, {}
    for beam in beams:
        trace = DL.BeamTrace(vocabulary, catalogue)
        result = decode(model, enc, beam_width=beam, policy="confidence", top_k=10, allowed_sids=allowed,
                        decoder_chunk_size=chunk, observer=trace)
        traces[beam] = (trace, result)
        hit = (result.codes == targets[:, None, :]).all(-1)
        sizes = torch.zeros(result.codes.shape[:2], dtype=torch.int16)
        valid = (result.codes[..., 0] >= 0).cpu()
        sizes[valid] = catalogue.sizes[catalogue.rows_of(result.codes.cpu()[valid])].to(torch.int16).cpu()
        out.update({
            f"beam{beam}_codes": result.codes.to(torch.int16).cpu(), f"beam{beam}_scores": result.scores.cpu(),
            f"beam{beam}_counts": result.counts.to(torch.int16).cpu(), f"beam{beam}_list_sizes": sizes,
            f"beam{beam}_target_rank": torch.where(hit.any(1), hit.to(torch.int8).argmax(1) + 1, 0).to(torch.int16).cpu(),
            f"beam{beam}_n_paths": torch.tensor([step["n_paths"] for step in trace.steps],
                                                dtype=torch.int16).expand(n_users, depth).clone(),
            f"beam{beam}_position_counts": torch.stack([step["position_counts"] for step in trace.steps], 1).to(torch.int16),
            **{f"beam{beam}_{name}": torch.stack([step[name] for step in trace.steps], 1).to(torch.int16)
               for name in ("n_distinct", "n_reachable", "n_distinct_reachable")}})

    greedy, (primary_trace, primary_result) = traces[1][0], traces[primary]
    paths = {"greedy": greedy.path_to(greedy.final_states[:, 0]), "top1": primary_trace.path_to(primary_result.codes[:, 0])}
    for kind, path in paths.items():
        rec, check = _trace_path(model, enc, cache, path, targets, target_rows, catalogue)
        out.update({f"{kind}_{name}": value for name, value in rec.items()})
        for name, value in check.items():
            checks[name] = max(checks.get(name, 0.0), value)

    lattice = _lattice(model, enc, cache, targets, chunk)
    out.update({name: value.cpu() for name, value in lattice.items()})
    in_list = (primary_result.codes == targets[:, None, :]).all(-1)
    excess = (primary_result.scores - lattice["target_best_score"][:, None])[in_list]
    checks["v5_target_score_max_excess_over_lattice"] = float(excess.max()) if excess.numel() else float("-inf")
    out.update(_regret(model, enc, cache, primary_result.codes[:, 0], paths["top1"]["found"]))
    trainer_pass = diffusion.digit_logits(model, enc, torch.zeros((n_users, depth), dtype=torch.long, device=enc.device))
    out["trainer_confidence"] = torch.softmax(trainer_pass.float(), -1).max(-1).values.cpu()
    return out, checks
