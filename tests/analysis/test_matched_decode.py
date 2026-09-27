from itertools import permutations, product
from types import SimpleNamespace

import pytest
import torch

from sidlens.analysis.matched_decode import decode


class TinyModel(torch.nn.Module):
    """Deterministic logits depend on the history and all revealed digits."""

    def __init__(self, depth=3, vocabulary=3):
        super().__init__()
        self.n_digit, self.codebook_size = depth, vocabulary
        self.calls = []
        self.eval()

    def logits_for(self, codes, mask, history):
        digit = torch.arange(self.n_digit, dtype=torch.float32)[None, :, None]
        token = torch.arange(self.codebook_size, dtype=torch.float32)[None, None, :]
        context = (codes.float() * (~mask) * (digit[0, :, 0] + 1)).sum(1)
        signal = history.reshape(-1, 1, 1) * 0.73 + context[:, None, None] * 0.31
        return torch.sin((digit + 0.4) * (token + 1.3) + signal) * (digit + 1.2)

    def forward_decoder_only(self, batch, *, return_loss, digit, use_cache, past_key_values):
        assert return_loss is False and digit is None and use_cache is True
        assert past_key_values is None
        codes, mask = batch["decoder_input_ids"], batch["mask_positions"].bool()
        history = batch["encoder_hidden"][:, 0, 0]
        self.calls.append((codes.clone(), mask.clone(), history.clone()))
        return SimpleNamespace(logits=self.logits_for(codes, mask, history))


def exhaustive(model, history, allowed_orders):
    """Enumerate all complete reveal paths without calling the beam code."""
    best = {}
    for order in allowed_orders:
        for sid in product(range(model.codebook_size), repeat=model.n_digit):
            codes = torch.zeros((1, model.n_digit), dtype=torch.long)
            mask = torch.ones_like(codes, dtype=torch.bool)
            score = torch.tensor(0.0)
            for position in order:
                logits = model.logits_for(codes, mask, torch.tensor([history]))
                score += torch.log_softmax(logits, -1)[0, position, sid[position]]
                codes[0, position] = sid[position]
                mask[0, position] = False
            best[sid] = max(best.get(sid, -float("inf")), score.item())
    return best


@pytest.mark.parametrize("policy", ["confidence", "fixed"])
def test_wide_beam_matches_exhaustive_complete_paths(policy):
    model = TinyModel(depth=2, vocabulary=2)
    orders = list(permutations(range(2))) if policy == "confidence" else [(1, 0)]
    result = decode(model, torch.tensor([[[0.7]]]), beam_width=16, policy=policy,
                    order=None if policy == "confidence" else orders[0], top_k=8)
    expected = exhaustive(model, 0.7, orders)
    assert result.counts.tolist() == [4]
    actual = {tuple(sid): score for sid, score in zip(result.codes[0, :4].tolist(),
                                                      result.scores[0, :4].tolist())}
    assert actual == pytest.approx(expected, abs=1e-6)
    assert torch.all(result.codes[0, 4:] == -1)
    assert torch.isneginf(result.scores[0, 4:]).all()
    assert result.beam_sizes == ((4, 8) if policy == "confidence" else (2, 4))


@pytest.mark.parametrize("policy", ["confidence", "fixed"])
def test_batch_and_decoder_chunk_invariance(policy):
    model = TinyModel()
    histories = torch.tensor([0.1, 1.3, 2.7]).reshape(3, 1, 1)
    options = {"beam_width": 7, "policy": policy,
               "order": (2, 0, 1) if policy == "fixed" else None}
    batched = decode(model, histories, decoder_chunk_size=4, **options)
    single = [decode(model, history[None], decoder_chunk_size=1, **options) for history in histories]
    large_chunk = decode(model, histories, decoder_chunk_size=256, **options)
    assert torch.equal(batched.codes, torch.cat([r.codes for r in single]))
    assert torch.equal(batched.codes, large_chunk.codes)
    assert torch.allclose(batched.scores, torch.cat([r.scores for r in single]))
    assert torch.equal(batched.scores, large_chunk.scores)


def test_depth_one_policies_are_identical_and_width_can_exceed_vocabulary():
    model = TinyModel(depth=1, vocabulary=3)
    hidden = torch.tensor([[[0.6]]])
    guided = decode(model, hidden, beam_width=256, policy="confidence")
    fixed = decode(model, hidden, beam_width=256, policy="fixed", order=[0])
    assert torch.equal(guided.codes, fixed.codes)
    assert torch.equal(guided.scores, fixed.scores)
    assert guided.beam_sizes == fixed.beam_sizes == (3,)


def test_final_legality_filter_deduplicates_and_does_not_fabricate_padding():
    model = TinyModel(depth=2, vocabulary=2)
    result = decode(model, torch.tensor([[[0.7]]]), beam_width=16,
                    policy="confidence", allowed_sids={(0, 1)}, top_k=3)
    assert result.codes.tolist() == [[[0, 1], [-1, -1], [-1, -1]]]
    assert result.counts.tolist() == [1]
    assert result.generated_counts.tolist() == [8]
    assert result.unique_counts.tolist() == [4]
    assert result.invalid_counts.tolist() == [6]
    assert result.legal_unique_counts.tolist() == [1]
    empty = decode(model, torch.tensor([[[0.7]]]), beam_width=16,
                   policy="confidence", allowed_sids=set())
    assert empty.counts.tolist() == [0]
    assert (empty.codes == -1).all() and torch.isneginf(empty.scores).all()


def test_confidence_selects_position_by_probability_and_preserves_reveals():
    class PositionModel(TinyModel):
        def logits_for(self, codes, mask, history):
            return torch.tensor([[2.0, 0.0], [0.0, 4.0], [1.0, 0.0]])[None].expand(len(codes), -1, -1)

    guided_model = PositionModel(depth=3, vocabulary=2)
    hidden = torch.zeros(1, 1, 1)
    guided = decode(guided_model, hidden, beam_width=1, policy="confidence")
    assert guided_model.calls[1][1].tolist() == [[True, False, True]]
    assert guided_model.calls[1][0].tolist() == [[0, 1, 0]]
    assert guided_model.calls[2][1].tolist() == [[False, False, True]]
    assert guided_model.calls[2][0][0, 1].item() == 1
    fixed_model = PositionModel(depth=3, vocabulary=2)
    fixed = decode(fixed_model, hidden, beam_width=1, policy="fixed", order=[2, 1, 0])
    assert fixed_model.calls[1][1].tolist() == [[True, True, False]]
    assert guided.codes[0, 0].tolist() == fixed.codes[0, 0].tolist() == [0, 1, 0]


def test_exact_ties_have_stable_parent_position_token_order():
    class Uniform(TinyModel):
        def logits_for(self, codes, mask, history):
            return torch.zeros(len(codes), self.n_digit, self.codebook_size)

    result = decode(Uniform(depth=2, vocabulary=2), torch.zeros(1, 1, 1),
                    beam_width=3, policy="confidence", top_k=3)
    assert result.codes.tolist() == [[[0, 0], [0, 1], [1, 0]]]


def test_beam_limit_and_reveal_count_hold_at_every_step():
    model = TinyModel(depth=3, vocabulary=3)
    result = decode(model, torch.zeros(2, 1, 1), beam_width=5, policy="confidence",
                    decoder_chunk_size=256)
    assert result.beam_sizes == (5, 5, 5)
    assert [len(codes) for codes, _, _ in model.calls] == [2, 10, 10]
    for step, (_, mask, _) in enumerate(model.calls):
        assert torch.all(mask.sum(1) == model.n_digit - step)


def test_cross_cache_uses_projected_keys_values_and_never_self_past():
    class CrossCached(TinyModel):
        def __init__(self):
            super().__init__()
            self.n_embd = 1
            self.projection_calls = 0
            self.decoder_blocks = [SimpleNamespace(cross_attn=SimpleNamespace(qkv=self.project))]

        def project(self, hidden):
            self.projection_calls += 1
            return torch.cat([7 * hidden, 11 * hidden, 13 * hidden], dim=-1)

        def forward_decoder_only(self, batch, *, return_loss, digit, use_cache, past_key_values):
            assert use_cache is True
            assert len(past_key_values) == 1
            self_past, (key, value) = past_key_values[0]
            assert self_past is None
            assert torch.equal(key, 11 * batch["encoder_hidden"])
            assert torch.equal(value, 13 * batch["encoder_hidden"])
            return super().forward_decoder_only(batch, return_loss=return_loss, digit=digit,
                                                use_cache=use_cache, past_key_values=None)

    model = CrossCached()
    decode(model, torch.tensor([1.1, 2.3, 3.7]).reshape(3, 1, 1), beam_width=5,
           policy="confidence", decoder_chunk_size=4)
    assert model.projection_calls == 1


@pytest.mark.parametrize("options", [
    {"policy": "sample"}, {"policy": "fixed", "order": [0, 0, 2]},
    {"policy": "fixed"}, {"policy": "confidence", "order": [0, 1, 2]},
    {"policy": "confidence", "beam_width": 0},
])
def test_invalid_experiment_settings_fail(options):
    config = {"beam_width": 4, **options}
    with pytest.raises(ValueError):
        decode(TinyModel(), torch.zeros(1, 1, 1), **config)


def test_training_mode_rejected():
    model = TinyModel().train()
    with pytest.raises(ValueError, match="eval"):
        decode(model, torch.zeros(1, 1, 1), beam_width=4, policy="confidence")
