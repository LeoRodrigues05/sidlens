import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "experiments/controlled/exp1_matched_beam/legacy.py"
SPEC = importlib.util.spec_from_file_location("matched_legacy_test", MODULE_PATH)
legacy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)


def test_padding_normalization_preserves_true_first_hit_ranks():
    raw = torch.tensor([[[1, 2], [2, 3], [2, 3]],
                        [[8, 8], [8, 8], [8, 8]],
                        [[1, 2], [2, 3], [3, 4]]])
    result = legacy._normalize_padding(raw, {(1, 2), (2, 3), (3, 4)})
    assert result.counts.tolist() == [2, 0, 3]
    assert result.codes.tolist() == [[[1, 2], [2, 3], [-1, -1]],
                                     [[-1, -1], [-1, -1], [-1, -1]],
                                     [[1, 2], [2, 3], [3, 4]]]


@pytest.mark.parametrize("row", [
    [[1, 2], [1, 2], [2, 3]],
    [[1, 2], [8, 8], [8, 8]],
    [[8, 8], [1, 2], [1, 2]],
])
def test_unexpected_internal_duplicates_or_illegal_mixtures_fail(row):
    with pytest.raises(ValueError):
        legacy._normalize_padding(torch.tensor([row]), {(1, 2), (2, 3)})


@pytest.mark.parametrize("raises", [False, True])
def test_wrapper_restores_configuration_and_rng_even_after_error(monkeypatch, raises):
    model = SimpleNamespace(training=False, config={"original": {"unchanged": True}},
                            n_digit=3, codebook_size=128)
    original_config = model.config
    original_rng = torch.get_rng_state().clone()

    def fake_vendor(model, hidden, *, n_return_sequences, tokenizer, mode, rand_cfg):
        assert model.config["vectorized_beam_search"]["beam_act"] == 64
        assert model.config["vectorized_beam_search"]["beam_max"] == 64
        assert model.config["dedup_strategy"] == "simple"
        assert mode == "confidence" and n_return_sequences == 2
        assert tokenizer.codebooks_to_item_id([1, 2, 3]) == 1
        assert tokenizer.codebooks_to_item_id([3, 2, 1]) is None
        torch.rand(5)
        if raises:
            raise RuntimeError("sentinel")
        return torch.tensor([[[1, 2, 3], [1, 2, 3]]])

    fake_module = ModuleType("genrec.models.DIFF_GRM.beam")
    fake_module.iterative_mask_decode = fake_vendor
    monkeypatch.setitem(sys.modules, "genrec.models.DIFF_GRM.beam", fake_module)
    if raises:
        with pytest.raises(RuntimeError, match="sentinel"):
            legacy.decode_legacy_confidence(model, torch.zeros(1, 1, 1),
                                            beam_width=64, allowed_sids={(1, 2, 3)}, top_k=2)
    else:
        result = legacy.decode_legacy_confidence(model, torch.zeros(1, 1, 1),
                                                beam_width=64, allowed_sids={(1, 2, 3)}, top_k=2)
        assert result.counts.tolist() == [1]
    assert model.config is original_config
    assert model.config == {"original": {"unchanged": True}}
    assert torch.equal(torch.get_rng_state(), original_rng)
