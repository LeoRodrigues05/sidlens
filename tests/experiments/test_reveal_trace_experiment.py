"""The reveal trace's per-batch recording, on a random-weight model (CPU).

`trace_batch` is everything representation/exp4 records. These checks do not
depend on trained weights: the recorded states must re-score to the search's
own path scores, revealed positions must carry the fill value and never a
model output, the lattice must bound the beam's score for the target, and the
recorded final lists must be `decode`'s.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sidlens.analysis import diffusion_lens as DL                    # noqa: E402
from sidlens.analysis.matched_decode import decode                   # noqa: E402

_path = Path(__file__).resolve().parents[2] / "experiments/representation/exp4_diffusion_reveal_trace/common.py"
_spec = importlib.util.spec_from_file_location("reveal_trace_common", _path)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

K, DEPTH, BEAMS = 6, 3, (1, 8, 32)


@pytest.fixture(scope="module")
def traced(random_diffgrm):
    s = random_diffgrm(n_digit=DEPTH, codebook_size=K, seed=5, batch=6)
    generator = torch.Generator().manual_seed(11)
    sids = {tuple(sid) for sid in torch.randint(0, K, (60, DEPTH), generator=generator).tolist()}
    buckets = {sid: tuple(range(1 + index % 3)) for index, sid in enumerate(sorted(sids))}
    catalogue = DL.Catalogue.from_buckets(buckets, K)
    targets = catalogue.sids[torch.randint(0, len(buckets), (6,), generator=generator)]
    out, checks = common.trace_batch(s.model, s.enc, targets, catalogue, set(buckets), beams=BEAMS, primary=32, chunk=64)
    return s, catalogue, buckets, targets, out, checks


def test_declared_cells_are_the_twenty_next_item_checkpoints_with_exp1_first():
    assert len(common.CELLS) == len(set(common.CELLS)) == 20
    assert common.CELLS[:18] == common.EXP1_CELLS and common.EXP1_CELLS[0] == "diff-next1-MQ-3cb-128"
    assert common.EXP1_CELLS[7] == "diff-next1-rqkmeans-3cb-512" and common.EXP1_CELLS[15] == "diff-next1-rqvae-4cb-512"


def test_recorded_lists_are_the_decoder_output_and_checks_pass(traced):
    s, catalogue, buckets, targets, out, checks = traced
    for beam in BEAMS:
        plain = decode(s.model, s.enc, beam_width=beam, policy="confidence", top_k=10, allowed_sids=set(buckets),
                       decoder_chunk_size=64)
        assert torch.equal(out[f"beam{beam}_codes"].long(), plain.codes)
        assert torch.equal(out[f"beam{beam}_scores"], plain.scores)
        assert torch.equal(out[f"beam{beam}_counts"].long(), plain.counts)
        hit = (plain.codes == targets[:, None, :]).all(-1)
        assert torch.equal(out[f"beam{beam}_target_rank"].long(), torch.where(hit.any(1), hit.int().argmax(1) + 1, 0))
        sizes = out[f"beam{beam}_list_sizes"]
        for user in range(6):
            for rank in range(10):
                sid = tuple(plain.codes[user, rank].tolist())
                assert int(sizes[user, rank]) == (len(buckets[sid]) if sid in buckets else 0)
        assert (out[f"beam{beam}_n_distinct"] <= out[f"beam{beam}_n_paths"]).all()
        assert (out[f"beam{beam}_n_distinct_reachable"] <= out[f"beam{beam}_n_reachable"]).all()
        assert torch.equal(out[f"beam{beam}_position_counts"].sum(-1), out[f"beam{beam}_n_paths"])
    assert checks["v3_path_score_max_abs_diff"] < 1e-4
    assert checks["v4_last_block_max_abs_diff"] < 1e-5
    assert checks["v5_target_score_max_excess_over_lattice"] < 1e-4


def test_path_states_never_read_a_revealed_position(traced):
    _, catalogue, _, targets, out, _ = traced
    for kind in ("greedy", "top1"):
        found = out[f"{kind}_found"]
        assert found.any()
        order = out[f"{kind}_order"][found].long()                       # [F, depth]
        revealed = torch.zeros((len(order), DEPTH, DEPTH), dtype=torch.bool)   # [F, state, digit]
        for state in range(1, DEPTH):
            revealed[:, state] = revealed[:, state - 1]
            revealed[torch.arange(len(order)), state, order[:, state - 1]] = True
        assert torch.equal(out[f"{kind}_target_rank"][found] == -1, revealed)
        assert torch.equal(torch.isnan(out[f"{kind}_target_logp"][found].float()), revealed)
        assert torch.equal(torch.isnan(out[f"{kind}_entropy"][found].float()), revealed)
        assert torch.equal((out[f"{kind}_top_codes"][found] == -1).all(-1), revealed)
        blocks = out[f"{kind}_block_top"][found]                         # [F, state, block, digit]
        assert blocks.shape[2] == 4 and torch.equal((blocks == -1).all(2), revealed)
        # The last block's top code is the recorded top code, and the guess keeps revealed codes.
        assert torch.equal(blocks[:, :, -1][~revealed], out[f"{kind}_top_codes"][found][..., 0][~revealed])
        guess, codes = out[f"{kind}_guess"][found].long(), out[f"{kind}_codes"][found].long()
        for state in range(1, DEPTH):
            earlier = order[:, :state]
            assert torch.equal(guess[:, state].gather(1, earlier), codes[:, :state])
        # A reachable target is never ranked below the number of catalogue SIDs.
        reachable = out[f"{kind}_target_reachable"][found]
        assert (out[f"{kind}_catalogue_sid_rank"][found][reachable] <= len(catalogue.sids)).all()
        lo, hi = out[f"{kind}_catalogue_item_rank_lo"][found], out[f"{kind}_catalogue_item_rank_hi"][found]
        assert (lo <= hi).all() and (lo >= out[f"{kind}_catalogue_sid_rank"][found]).all()
        assert reachable[:, 0].all()                                     # nothing is revealed at state 0


def test_missing_paths_carry_fill_values_and_the_lattice_orders_are_permutations(traced):
    s, catalogue, buckets, targets, out, _ = traced
    none, _ = common.trace_batch(s.model, s.enc, targets, catalogue, set(), beams=BEAMS, primary=32, chunk=64)
    assert not none["top1_found"].any() and (none["top1_order"] == -1).all()
    assert (none["top1_guess"] == -1).all() and torch.isnan(none["top1_top_logp"].float()).all()
    assert (none["regret_rank"] == -1).all() and none["greedy_found"].all()
    assert (out["regret_rank"][out["top1_found"]] >= 0).all()
    for name in ("target_best_order", "target_greedy_order"):
        assert torch.equal(out[name].long().sort(1).values, torch.arange(DEPTH).expand(6, DEPTH))
    assert (out["target_greedy_score"] <= out["target_best_score"] + 1e-6).all()
    assert out["lattice_target_logp"].shape == (6, 2 ** DEPTH - 1, DEPTH)
    assert torch.isnan(out["lattice_target_logp"][:, 0].float()).sum() == 0     # fully masked: every digit read
    assert out["trainer_confidence"].shape == (6, DEPTH)
    assert ((out["trainer_confidence"] > 0) & (out["trainer_confidence"] <= 1)).all()
    with pytest.raises(ValueError, match="greedy"):
        common.trace_batch(s.model, s.enc, targets, catalogue, set(buckets), beams=(8, 32), primary=32)
