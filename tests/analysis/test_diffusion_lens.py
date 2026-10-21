"""Reading a DiffGRM decode: the readouts must be the model's own (CPU).

On a random-weight vendored DIFF_GRM (`conftest.random_diffgrm`). A trace is
only a record of the decode if the last-block readout is the model's
distribution, a recorded path re-scores to the search's own score, and the
lattice's best reveal order agrees with an exhaustive search. Catalogue
scores are checked against a plain loop, and SID and item ranks stay apart.
"""

from itertools import permutations

import pytest

torch = pytest.importorskip("torch")

from sidlens.analysis import diffusion_lens as DL                    # noqa: E402
from sidlens.analysis.matched_decode import _project_cross_cache, decode, score_partial  # noqa: E402

K = 4


@pytest.fixture(scope="module")
def setup(random_diffgrm):
    s = random_diffgrm(n_digit=3, codebook_size=K, seed=3, batch=4)
    return s.model, s.enc, _project_cross_cache(s.model, s.enc)


def test_last_block_readout_is_the_model_distribution(setup):
    model, enc, cache = setup
    rows = torch.arange(enc.shape[0])
    partial = torch.tensor([[-1, -1, -1], [2, -1, -1], [-1, 0, 3], [1, -1, 2]])
    log_probs, residuals = DL.decoder_residuals(model, enc, cache, partial, rows)
    with torch.no_grad():
        assert torch.equal(log_probs, score_partial(model, enc, cache, partial, rows))
    assert residuals.shape == (4, 4, 3, 32)
    assert torch.equal(DL.layer_log_probs(model, residuals[3]), log_probs)
    every = DL.layer_log_probs(model, residuals)
    assert every.shape == (4, 4, 3, K)
    assert torch.allclose(every[3], log_probs, atol=1e-6)
    assert not torch.allclose(every[0], log_probs, atol=1e-3)        # an early readout is a different thing
    guess = DL.current_guess(log_probs, partial)
    assert torch.equal(guess[partial >= 0], partial[partial >= 0])   # revealed codes are never re-read
    assert torch.equal(guess[partial < 0], log_probs.argmax(-1)[partial < 0])


def test_wide_beam_trace_counts_paths_states_and_duplicates(setup):
    model, enc, _ = setup
    trace = DL.BeamTrace(K)
    result = decode(model, enc, beam_width=512, policy="confidence", top_k=K ** 3, observer=trace)
    assert result.beam_sizes == (12, 96, 384)                        # nothing pruned: 3 * 4, * 2 * 4, * 1 * 4
    assert [step["n_paths"] for step in trace.steps] == [12, 96, 384]
    # distinct partial states: 3 * 4 singles, 3 * 16 pairs, 64 complete SIDs reached along 6 orders each
    assert [step["n_distinct"].tolist() for step in trace.steps] == [[12] * 4, [48] * 4, [64] * 4]
    assert torch.equal(trace.steps[1]["position_counts"].sum(1), torch.full((4,), 96))
    assert result.counts.tolist() == [64] * 4


def test_recorded_path_rescoring_and_lattice_agree_with_the_exhaustive_search(setup):
    model, enc, cache = setup
    B = enc.shape[0]
    trace = DL.BeamTrace(K)
    result = decode(model, enc, beam_width=512, policy="confidence", top_k=K ** 3, observer=trace)
    path = trace.path_to(result.codes[:, 0])
    assert path["found"].all()
    assert torch.equal(path["scores"][:, -1], result.scores[:, 0])
    states = DL.path_partials(path["order"], path["codes"])          # [B, 3, 3]
    assert (states[:, 0] == -1).all() and ((states[:, 2] >= 0).sum(1) == 2).all()
    with torch.no_grad():
        log_probs = score_partial(model, enc, cache, states.reshape(-1, 3),
                                  torch.arange(B).repeat_interleave(3)).reshape(B, 3, 3, K)
    stepwise = log_probs[torch.arange(B)[:, None], torch.arange(3)[None, :], path["order"], path["codes"]]
    assert torch.allclose(stepwise.cumsum(1), path["scores"], atol=1e-5)

    # The lattice of every SID gives the same best-order score as the unpruned beam.
    sids = result.codes[0]                                           # all 64 SIDs for history 0
    lattice = DL.lattice_partials(sids)                              # [64, 7, 3]
    with torch.no_grad():
        lattice_lp = score_partial(model, enc, cache, lattice.reshape(-1, 3),
                                   torch.zeros(64 * 7, dtype=torch.long)).reshape(64, 7, 3, K)
    gold = DL.gold_log_probs(lattice_lp, lattice, sids[:, None, :])
    assert torch.isnan(gold[lattice >= 0]).all() and torch.isfinite(gold[lattice < 0]).all()
    best = DL.best_reveal_path(gold)
    assert torch.allclose(best["best_score"], result.scores[0], atol=1e-5)
    assert (best["greedy_score"] <= best["best_score"] + 1e-6).all()

    def along(order):                                                # the path score of one fixed order
        state, total = torch.zeros(64, dtype=torch.long), torch.zeros(64)
        for digit in order:
            total = total + gold[torch.arange(64), state, digit]
            state = state | (1 << digit)
        return total

    by_order = torch.stack([along(order) for order in permutations(range(3))])
    assert torch.allclose(by_order.max(0).values, best["best_score"], atol=1e-6)
    for name in ("best", "greedy"):
        orders = best[f"{name}_order"]
        assert torch.equal(orders.sort(1).values, torch.arange(3).expand(64, 3))
        replay = torch.stack([along(order.tolist())[i] for i, order in enumerate(orders)])
        assert torch.allclose(replay, best[f"{name}_score"], atol=1e-6)


def test_catalogue_scores_match_a_plain_loop_and_ranks_keep_sid_and_item_apart(setup):
    model, enc, cache = setup
    buckets = {(0, 1, 2): (10, 11), (0, 1, 3): (12,), (3, 3, 3): (13,), (2, 0, 1): (14, 15, 16)}
    catalogue = DL.Catalogue.from_buckets(buckets, K)
    assert catalogue.n_items == 7 and [tuple(sid) for sid in catalogue.sids.tolist()] == sorted(buckets)
    partial = torch.tensor([[-1, -1, -1], [0, -1, -1], [0, 1, -1], [-1, -1, 3]])
    with torch.no_grad():
        log_probs = score_partial(model, enc, cache, partial, torch.arange(4))
    scores = DL.catalogue_scores(log_probs, partial, catalogue)
    for r in range(4):
        for u, sid in enumerate(catalogue.sids.tolist()):
            if any(partial[r, d] >= 0 and partial[r, d] != sid[d] for d in range(3)):
                assert scores[r, u] == -torch.inf
            else:
                expected = sum(float(log_probs[r, d, sid[d]]) for d in range(3) if partial[r, d] < 0)
                assert float(scores[r, u]) == pytest.approx(expected, abs=1e-5)
    assert catalogue.reachable(partial).tolist() == [True, True, True, True]
    assert catalogue.reachable(torch.tensor([[1, -1, -1], [0, 1, 0], [-1, 3, 3], [2, 0, -1]])).tolist() == \
        [False, False, True, True]
    assert torch.equal(catalogue.rows_of([(3, 3, 3), (0, 1, 2)]), torch.tensor([3, 0]))
    with pytest.raises(KeyError):
        catalogue.rows_of([(1, 1, 1)])

    fixed = torch.tensor([[0.0, -1.0, -2.0, -torch.inf]] * 3)        # rows of catalogue order, sizes 2, 1, 3, 1
    ranks = DL.target_ranks(fixed, torch.tensor([0, 2, 3]), catalogue.sizes)
    assert ranks["sid_rank"].tolist() == [1, 3, 4]
    assert ranks["item_rank_lo"].tolist() == [1, 4, 7] and ranks["item_rank_hi"].tolist() == [2, 6, 7]
    assert ranks["reachable"].tolist() == [True, True, False]
    assert ranks["ties"].tolist() == [0, 0, 0]


def test_beam_trace_counts_catalogue_reachable_paths(setup):
    model, enc, _ = setup
    catalogue = DL.Catalogue.from_buckets({(0, 1, 2): (1,), (0, 1, 3): (2,), (3, 3, 3): (3,)}, K)
    trace = DL.BeamTrace(K, catalogue)
    decode(model, enc, beam_width=512, policy="confidence", top_k=3, observer=trace,
           allowed_sids=set(catalogue.index))
    # Unpruned, so the counts are combinatorial: reachable single digits {0@0, 3@0, 1@1, 3@1, 2@2, 3@2}.
    assert [step["n_distinct_reachable"].tolist() for step in trace.steps] == [[6] * 4, [8] * 4, [3] * 4]
    assert trace.steps[2]["n_reachable"].tolist() == [18] * 4        # 3 SIDs along 6 orders each
    with pytest.raises(RuntimeError, match="new BeamTrace"):
        decode(model, enc, beam_width=4, policy="confidence", observer=trace)


def test_state_keys_separate_masked_from_code_zero_and_refuse_overflow():
    keys = DL.state_keys(torch.tensor([[-1, 0], [0, -1], [0, 0], [-1, -1]]), K)
    assert len(set(keys.tolist())) == 4 and keys[3] == 0
    with pytest.raises(ValueError, match="overflow"):
        DL.state_keys(torch.zeros((1, 8), dtype=torch.long), 512)
    with pytest.raises(ValueError, match="codes in"):
        DL.state_keys(torch.tensor([[K, 0]]), K)


def test_training_order_reveals_most_confident_first_under_least_masking():
    confidence = torch.tensor([[0.2, 0.9, 0.5]])
    assert DL.training_reveal_order(confidence, "least").tolist() == [[1, 2, 0]]
    assert DL.training_reveal_order(confidence, "most").tolist() == [[0, 2, 1]]
    with pytest.raises(ValueError):
        DL.training_reveal_order(confidence, "random")


def test_encoder_sites_end_at_the_encoder_output_and_start_at_item_plus_position(random_diffgrm):
    s = random_diffgrm(n_digit=3, codebook_size=6, seed=0, batch=6)
    sites = DL.encoder_sites(s.model, s.histories, s.mask)
    assert set(sites) == {"input", "mid", "output"}
    assert torch.equal(sites["output"], s.enc)
    model, h = s.model, s.histories
    ids = torch.where(h < 0, torch.zeros_like(h), h + 3 + torch.arange(3)[None, None, :] * 6)
    with torch.no_grad():
        item = model.item_mlp(model.embedding(ids).reshape(h.shape[0], h.shape[1], -1))
        expect = item + model.pos_emb_enc(torch.arange(h.shape[1]))[None]
    real = s.mask
    assert torch.allclose(sites["input"][real], expect[real], atol=1e-6)
    assert not torch.allclose(sites["mid"][real], sites["input"][real], atol=1e-3)   # attention wrote something
    assert bool((sites["output"][~real] == 0).all()) and not bool((sites["mid"][~real] == 0).all())
