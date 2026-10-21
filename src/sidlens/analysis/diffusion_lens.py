"""Reading a DiffGRM decode: intermediate solutions, layer readouts, the catalogue.

DiffGRM is masked (absorbing) discrete diffusion over one SID. A step takes a
partial SID and returns one distribution over codes per digit position; the
search commits one (position, code) and the next step starts from the new
partial SID alone. This module turns that process into things one can look at:

    BeamTrace           what the beam committed at every step, and how many of
                        its paths are distinct or still reachable in the catalogue
    decoder_residuals   the residual stream after each decoder block at a state
    layer_log_probs     the output head applied to any block's residuals
    catalogue_scores    every catalogue SID scored from one state's distributions
    lattice_partials    all 2^n - 1 partial states of one SID, for exact
                        statements about reveal orders

Everything here observes. Scores come from `matched_decode.score_partial`, the
projected cross-attention path validated against the archive.

Traps, each with the guard that closes it
-----------------------------------------
1. **Reading a revealed position.** The training loss covers masked positions
   only, so the distribution at a revealed position is an untrained output.
   `current_guess` keeps the revealed code there, `catalogue_scores` uses the
   revealed code as a hard constraint, and `gold_log_probs` returns NaN there.
2. **A readout that is not the model's.** Block outputs are pre-`ln_f`. The
   readout applies `ln_f` and the model's own `_compute_digit_logits`, so at
   the last block it equals the model's distribution (tested bit for bit).
   At earlier blocks it is a logit lens: an upper layer could still change
   the answer, so a readout there is a description, not a causal claim.
3. **A capture of the wrong call.** A block hooked across several forwards
   keeps the last one. `decoder_residuals` runs exactly one forward and raises
   unless every block fired once.
4. **SID hits passed off as item hits.** Several items can share a SID.
   `target_ranks` returns the SID-level rank and item-level bounds separately.
5. **Product of marginals read as a joint.** `catalogue_scores` sums per-digit
   log-probabilities. That is what one state says about each SID if nothing
   further were conditioned on; it is not a normalized probability over items.
6. **State keys that collide.** Partial SIDs are keyed as base-(K+1) integers
   with 0 for a masked digit; `_check_key_range` refuses a depth and codebook
   size whose keys would not fit in int64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from sidlens.analysis.matched_decode import DecodeStep, score_partial
from sidlens.hooks import capture


def _check_key_range(depth: int, vocabulary: int) -> None:
    if (vocabulary + 1) ** depth >= 2 ** 62:
        raise ValueError(f"partial-SID keys overflow int64 at depth {depth}, codebook size {vocabulary}")


def state_keys(partial: Tensor, vocabulary: int) -> Tensor:
    """One int64 key per partial SID [..., depth] (-1 = masked); equal keys, equal states."""
    depth = partial.shape[-1]
    _check_key_range(depth, vocabulary)
    if bool((partial < -1).any()) or bool((partial >= vocabulary).any()):
        raise ValueError("partial SIDs must hold codes in [0, K) or -1 for a masked digit")
    weights = (vocabulary + 1) ** torch.arange(depth, device=partial.device, dtype=torch.long)
    return ((partial + 1) * weights).sum(-1)


@dataclass(frozen=True)
class Catalogue:
    """The distinct catalogue SIDs and how many items share each.

    ``sids`` [U, depth] long, ``sizes`` [U] long, in sorted SID order.
    ``partial_keys`` holds the key of every partial state that some catalogue
    SID completes (all 2^depth masks of every SID), sorted and unique.
    """

    sids: Tensor
    sizes: Tensor
    vocabulary: int
    partial_keys: Tensor
    index: dict = field(repr=False, compare=False, default_factory=dict)

    @classmethod
    def from_buckets(cls, sid_buckets: dict, vocabulary: int, device="cpu") -> "Catalogue":
        """Build from ``{sid tuple: item ids}``, e.g. ``DiffusionEvalCohort.sid_buckets``."""
        if not sid_buckets:
            raise ValueError("empty catalogue")
        ordered = sorted(sid_buckets)
        depths = {len(sid) for sid in ordered}
        if len(depths) != 1:
            raise ValueError(f"catalogue SIDs differ in depth: {sorted(depths)}")
        sids = torch.tensor(ordered, dtype=torch.long, device=device)
        if bool((sids < 0).any()) or bool((sids >= vocabulary).any()):
            raise ValueError("catalogue codes outside [0, K)")
        sizes = torch.tensor([len(sid_buckets[sid]) for sid in ordered], dtype=torch.long, device=device)
        if bool((sizes < 1).any()):
            raise ValueError("a catalogue SID has no item")
        masks = subset_masks(sids.shape[1]).to(device)                       # [2^n, n] True = revealed
        partial = torch.where(masks[None, :, :], sids[:, None, :], torch.full_like(sids[:, None, :], -1))
        keys = torch.unique(state_keys(partial, vocabulary))
        return cls(sids=sids, sizes=sizes, vocabulary=int(vocabulary), partial_keys=keys,
                   index={sid: row for row, sid in enumerate(ordered)})

    @property
    def n_items(self) -> int:
        return int(self.sizes.sum())

    def rows_of(self, sids) -> Tensor:
        """Row of each SID in ``sids`` [R, depth]; raises on a SID outside the catalogue."""
        rows = []
        for sid in (sids.tolist() if torch.is_tensor(sids) else sids):
            key = tuple(int(code) for code in sid)
            if key not in self.index:
                raise KeyError(f"SID {key} is not in the catalogue")
            rows.append(self.index[key])
        return torch.tensor(rows, dtype=torch.long, device=self.sids.device)

    def reachable(self, partial: Tensor) -> Tensor:
        """Bool [...]: does some catalogue SID agree with every revealed digit?"""
        return torch.isin(state_keys(partial, self.vocabulary), self.partial_keys)


def subset_masks(depth: int) -> Tensor:
    """[2^depth, depth] bool; row s has digit d True (revealed) iff bit d of s is set."""
    if not 1 <= depth <= 12:
        raise ValueError(f"subset lattice is for small depths, got {depth}")
    return ((torch.arange(2 ** depth)[:, None] >> torch.arange(depth)[None, :]) & 1).bool()


def lattice_partials(sids: Tensor) -> Tensor:
    """[R, 2^depth - 1, depth]: every partial state of each SID with a digit still masked.

    State s reveals the digits in bit set s (see `subset_masks`); s = 0 is the
    fully masked start. The fully revealed state is left out: nothing is
    predicted there.
    """
    masks = subset_masks(sids.shape[-1]).to(sids.device)[:-1]
    return torch.where(masks[None], sids[:, None, :], torch.full_like(sids[:, None, :], -1))


def gold_log_probs(log_probs: Tensor, partial: Tensor, sids: Tensor) -> Tensor:
    """[..., depth] log p(the SID's code) at each still-masked digit, NaN where revealed.

    ``log_probs`` [..., depth, K] at states ``partial`` [..., depth]; ``sids``
    broadcasts against ``partial``.
    """
    sids = sids.expand_as(partial)
    gold = log_probs.gather(-1, sids[..., None])[..., 0]
    return torch.where(partial < 0, gold, torch.full_like(gold, float("nan")))


def best_reveal_path(lattice_gold: Tensor) -> dict:
    """The best and the greedy reveal order of one SID, from its full lattice.

    ``lattice_gold`` [R, 2^depth - 1, depth] is `gold_log_probs` on
    `lattice_partials`. ``best_score`` is the maximum over all depth! orders of
    the summed log-probabilities: the score an unbounded confidence beam gives
    this SID, since the search keeps each SID's best path. ``greedy_*`` reveals,
    at each state, the masked digit whose SID code is most probable.
    Orders are [R, depth] digit positions in reveal order.
    """
    rows, states, depth = lattice_gold.shape
    if states != 2 ** depth - 1:
        raise ValueError(f"expected the {2 ** depth - 1}-state lattice of depth {depth}, got {states} states")
    full = 2 ** depth - 1
    neg = torch.full((rows,), -torch.inf, dtype=lattice_gold.dtype, device=lattice_gold.device)
    best = [neg.clone() for _ in range(full + 1)]
    best[0] = torch.zeros_like(neg)
    back = torch.zeros((rows, full + 1), dtype=torch.long, device=lattice_gold.device)
    for state in range(1, full + 1):
        for digit in range(depth):
            if not state >> digit & 1:
                continue
            previous = state & ~(1 << digit)
            candidate = best[previous] + lattice_gold[:, previous, digit]
            better = candidate > best[state]
            best[state] = torch.where(better, candidate, best[state])
            back[:, state] = torch.where(better, torch.full_like(back[:, state], digit), back[:, state])
    best_order = torch.zeros((rows, depth), dtype=torch.long, device=lattice_gold.device)
    state = torch.full((rows,), full, dtype=torch.long, device=lattice_gold.device)
    for step in range(depth - 1, -1, -1):
        digit = back.gather(1, state[:, None])[:, 0]
        best_order[:, step] = digit
        state = state & ~(1 << digit)
    greedy_order = torch.zeros_like(best_order)
    greedy_score = torch.zeros_like(neg)
    state = torch.zeros((rows,), dtype=torch.long, device=lattice_gold.device)
    row_index = torch.arange(rows, device=lattice_gold.device)
    for step in range(depth):
        here = lattice_gold[row_index, state]                              # NaN where revealed
        value, digit = torch.where(torch.isnan(here), torch.full_like(here, -torch.inf), here).max(-1)
        greedy_order[:, step] = digit
        greedy_score = greedy_score + value
        state = state | (1 << digit)
    return {"best_score": best[full], "best_order": best_order,
            "greedy_score": greedy_score, "greedy_order": greedy_order}


def decoder_residuals(model, encoder_hidden: Tensor, cross_cache, partial: Tensor,
                      history_rows: Tensor) -> tuple[Tensor, Tensor]:
    """(log_probs [R, depth, K], residuals [n_blocks, R, depth, width]) of one decoder forward.

    Arguments are `matched_decode.score_partial`'s. ``residuals[i]`` is the
    stream after decoder block i, before `ln_f`, on the model's device.
    """
    names = [f"decoder_blocks.{i}" for i in range(len(model.decoder_blocks))]
    with capture(model, names=names, to_cpu=False) as cap:
        with torch.no_grad():
            log_probs = score_partial(model, encoder_hidden, cross_cache, partial, history_rows)
    if bad := {name: cap.call_counts.get(name, 0) for name in names if cap.call_counts.get(name, 0) != 1}:
        raise RuntimeError(f"decoder blocks fired {bad} times in one forward, expected once each")
    return log_probs, torch.stack([cap[name] for name in names])


def encoder_sites(model, history_sid: Tensor, history_mask: Tensor) -> dict[str, Tensor]:
    """The encoder's three read points for one batch of histories, [B, S, width] each.

    ``input``  the first block's input: ``item_mlp(digit embeddings) + pos_emb_enc(slot)``
    ``mid``    after the last block's self-attention residual add (where one slot
               first holds information from another)
    ``output`` the encoder output the decoder reads: ``ln_f`` of the last block,
               padding zeroed; equal to `models.diffusion.encode` bit for bit

    Trap: ``mid`` is not a module output. It is the block input plus the
    attention module's output, which is what the vendor block adds; the check
    below recomputes the block from ``mid`` and refuses the capture if it does
    not equal the block's own output, so a vendor change cannot make ``mid``
    a different quantity silently. Padding rows are zero only in ``output``.
    """
    from sidlens.models import diffusion

    blocks = list(model.encoder_blocks)
    seen: dict[str, Tensor] = {}
    counts = {"input": 0, "attn": 0, "block": 0}

    def pre(_mod, args, _kwargs):
        counts["input"] += 1
        seen["input"] = args[0].detach()

    def attn(_mod, _inp, out):
        counts["attn"] += 1
        seen["attn"] = out[0].detach()

    def block(_mod, inp, out):
        counts["block"] += 1
        seen["block_in"], seen["block"] = inp[0].detach(), out.detach()

    handles = [blocks[0].register_forward_pre_hook(pre, with_kwargs=True),
               blocks[-1].attn.register_forward_hook(attn),
               blocks[-1].register_forward_hook(block)]
    try:
        output = diffusion.encode(model, history_sid, history_mask)
    finally:
        for h in handles:
            h.remove()
    if counts != {"input": 1, "attn": 1, "block": 1}:
        raise RuntimeError(f"encoder hooks fired {counts}; expected one forward through each")
    last = blocks[-1]
    with torch.no_grad():
        mid = seen["block_in"] + seen["attn"]
        again = mid + last.mlp(last.ln_2(mid))
        final = model.ln_f(seen["block"]) * history_mask.to(seen["block"].device)[..., None].float()
    if not torch.equal(again, seen["block"]):
        raise RuntimeError("block input + attention + MLP does not reproduce the encoder block output")
    if not torch.equal(final, output):
        raise RuntimeError("ln_f of the last block output with padding zeroed is not the encoder output")
    return {"input": seen["input"], "mid": mid, "output": output}


def layer_log_probs(model, residuals: Tensor) -> Tensor:
    """[..., depth, K] float32 log-softmax of the output head on block residuals.

    ``residuals`` [..., depth, width] are decoder block outputs. The head is
    the model's own: `ln_f`, then the dot product with each digit's slice of
    the embedding table.
    """
    depth = int(model.n_digit)
    if residuals.shape[-2] != depth:
        raise ValueError(f"residuals must be [..., depth={depth}, width], got {tuple(residuals.shape)}")
    lead = residuals.shape[:-2]
    with torch.no_grad():
        hidden = model.ln_f(residuals.reshape(-1, depth, residuals.shape[-1]))
        logits = torch.stack([model._compute_digit_logits(hidden[:, digit, :], digit=digit)
                              for digit in range(depth)], dim=1)
        return torch.log_softmax(logits.float(), dim=-1).reshape(*lead, depth, -1)


def current_guess(log_probs: Tensor, partial: Tensor) -> Tensor:
    """[..., depth] the state's best complete SID: revealed codes kept, masked digits by argmax."""
    return torch.where(partial >= 0, partial, log_probs.argmax(-1))


def catalogue_scores(log_probs: Tensor, partial: Tensor, catalogue: Catalogue) -> Tensor:
    """[R, U] score of every catalogue SID at each of R states.

    The sum over still-masked digits of log p(the SID's code), and -inf where
    the SID contradicts a revealed digit. See trap 5 on what this is not.
    """
    if log_probs.ndim != 3 or partial.shape != log_probs.shape[:2] or log_probs.shape[2] != catalogue.vocabulary:
        raise ValueError(f"log_probs {tuple(log_probs.shape)} and partial {tuple(partial.shape)} do not "
                         f"describe [R, depth, K={catalogue.vocabulary}] states")
    scores = torch.zeros((log_probs.shape[0], catalogue.sids.shape[0]), dtype=torch.float32,
                         device=log_probs.device)
    zero = torch.zeros((), dtype=torch.float32, device=log_probs.device)
    for digit in range(log_probs.shape[1]):
        column = log_probs[:, digit, :].float().index_select(1, catalogue.sids[:, digit])
        agrees = partial[:, digit, None] == catalogue.sids[None, :, digit]
        revealed = torch.where(agrees, zero, torch.full_like(zero, -torch.inf))
        scores = scores + torch.where((partial[:, digit] < 0)[:, None], column, revealed)
    return scores


def target_ranks(scores: Tensor, target_rows: Tensor, sizes: Tensor) -> dict:
    """Rank of each row's target SID in ``scores`` [R, U], at SID and item level.

    ``sid_rank`` is 1 + the number of catalogue SIDs scored strictly higher.
    Items sharing a SID tie exactly, so the target ITEM's rank is only bounded:
    ``item_rank_lo`` puts it first in its bucket and ``item_rank_hi`` last.
    ``ties`` counts other SIDs with exactly the target's score. ``reachable``
    is False where the target contradicts a revealed digit; its ranks then
    count every reachable SID as better and must not be read as ranks.
    """
    target = scores.gather(1, target_rows[:, None])
    better = scores > target
    ahead_items = (better * sizes[None, :]).sum(1)
    return {"sid_rank": 1 + better.sum(1),
            "item_rank_lo": ahead_items + 1,
            "item_rank_hi": ahead_items + sizes[target_rows],
            "ties": (scores == target).sum(1) - 1,
            "reachable": torch.isfinite(target[:, 0])}


def training_reveal_order(confidence: Tensor, select: str = "least") -> Tensor:
    """[B, depth] the reveal order implied by the trainer's one-shot ranking.

    ``confidence`` [B, depth] is the per-digit confidence at the fully masked
    state from the trainer's ranking pass (`use_cache=False`, i.e.
    `models.diffusion.digit_logits`, then max probability). With
    ``guided_select='least'`` the trainer masks the t least confident digits
    for t = 1..guided_steps, so the states it visits are those of revealing
    digits most-confident-first; with 'most' the reverse. A decode-path state
    lies in the trained family iff its revealed set is a prefix of this order
    (and, where guided_steps < depth, it is not the fully masked start).
    """
    if select not in ("least", "most"):
        raise ValueError(f"guided_select must be 'least' or 'most', got {select!r}")
    return torch.argsort(confidence, dim=1, descending=(select == "least"), stable=True)


class BeamTrace:
    """`matched_decode.decode` observer: every step's commits and beam make-up.

    Keeps, per step, the parent / position / code / score of every kept path
    (CPU), and per history the number of paths, of distinct partial states,
    and, given a ``catalogue``, of paths and distinct states that some
    catalogue SID still completes. Distributions are not kept: re-score the
    states of the paths you want with `decoder_residuals`.
    """

    def __init__(self, vocabulary: int, catalogue: Catalogue | None = None):
        self.vocabulary = int(vocabulary)
        self.catalogue = catalogue
        self.steps: list[dict] = []
        self.final_states: Tensor | None = None

    def __call__(self, step: DecodeStep) -> None:
        if step.step != len(self.steps):
            raise RuntimeError(f"observer saw step {step.step} after {len(self.steps)} steps; "
                               f"use a new BeamTrace per decode")
        depth = step.states.shape[-1]
        after = step.states.gather(1, step.parents[:, :, None].expand(-1, -1, depth)).clone()
        after.scatter_(2, step.positions[:, :, None], step.codes[:, :, None])
        keys = state_keys(after, self.vocabulary)
        ordered = keys.sort(dim=1).values
        distinct = 1 + (ordered[:, 1:] != ordered[:, :-1]).sum(1)
        record = {"parents": step.parents.cpu(), "positions": step.positions.cpu(), "codes": step.codes.cpu(),
                  "scores": step.scores.cpu(), "n_paths": after.shape[1], "n_distinct": distinct.cpu(),
                  "position_counts": torch.stack([(step.positions == digit).sum(1) for digit in range(depth)], 1).cpu()}
        if self.catalogue is not None:
            ok = torch.isin(keys, self.catalogue.partial_keys)
            ordered_ok = torch.where(ok, keys, torch.full_like(keys, -1)).sort(dim=1).values
            new = torch.cat([torch.ones_like(ordered_ok[:, :1], dtype=torch.bool),
                             ordered_ok[:, 1:] != ordered_ok[:, :-1]], 1)
            record["n_reachable"] = ok.sum(1).cpu()
            record["n_distinct_reachable"] = (new & (ordered_ok >= 0)).sum(1).cpu()
        self.steps.append(record)
        self.final_states = after.cpu()

    def path_to(self, sids: Tensor) -> dict:
        """The best kept path ending at each row's SID [B, depth].

        ``found`` [B] is False where no kept path ends there (order and codes
        are then -1 and scores NaN). ``order`` [B, depth] is the digit revealed
        at each step, ``codes`` the code written, ``scores`` the accumulated
        log-probability after each step. Final paths are in descending score
        order, so the first match is the SID's best path.
        """
        if self.final_states is None:
            raise RuntimeError("no decode was observed")
        sids = sids.cpu()
        match = (self.final_states == sids[:, None, :]).all(-1)
        found = match.any(1)
        index = match.to(torch.int8).argmax(1)
        depth = self.final_states.shape[-1]
        if len(self.steps) != depth:
            raise RuntimeError(f"trace holds {len(self.steps)} steps of a depth-{depth} decode")
        order = torch.full((len(sids), depth), -1, dtype=torch.long)
        codes = torch.full((len(sids), depth), -1, dtype=torch.long)
        scores = torch.full((len(sids), depth), float("nan"), dtype=torch.float32)
        for step in range(depth - 1, -1, -1):
            record = self.steps[step]
            order[:, step] = record["positions"].gather(1, index[:, None])[:, 0]
            codes[:, step] = record["codes"].gather(1, index[:, None])[:, 0]
            scores[:, step] = record["scores"].gather(1, index[:, None])[:, 0]
            index = record["parents"].gather(1, index[:, None])[:, 0]
        order[~found], codes[~found], scores[~found] = -1, -1, float("nan")
        return {"found": found, "order": order, "codes": codes, "scores": scores}


def path_partials(order: Tensor, codes: Tensor) -> Tensor:
    """[R, depth, depth] the state before each step of a reveal path (-1 = masked).

    ``order`` and ``codes`` [R, depth] as `BeamTrace.path_to` returns them.
    State 0 is fully masked; state t has the first t commits written.
    """
    rows, depth = order.shape
    if bool((order < 0).any()):
        raise ValueError("path_partials needs complete paths; drop rows whose path was not found")
    if bool((order.sort(dim=1).values != torch.arange(depth, device=order.device)).any()):
        raise ValueError("a reveal order is not a permutation of the digit positions")
    states = torch.full((rows, depth, depth), -1, dtype=torch.long, device=order.device)
    current = torch.full((rows, depth), -1, dtype=torch.long, device=order.device)
    for step in range(depth):
        states[:, step] = current
        current = current.scatter(1, order[:, step, None], codes[:, step, None])
    return states
