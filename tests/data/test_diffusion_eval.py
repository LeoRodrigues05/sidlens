"""Test cohort identity and historical tokenizer semantics without a GPU."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sidlens import paths
from sidlens.data.diffusion_eval import CATEGORY, load_eval_cohort
from sidlens.provenance.hashing import sha256_file


@pytest.fixture
def cohort_fixture(tmp_path):
    root = tmp_path / "frozen"
    data = root / "data"
    (data / "sequences").mkdir(parents=True)
    (data / "id_maps").mkdir()
    sem_dir = root / "sids" / "sem_ids" / "diffgrm"
    sem_dir.mkdir(parents=True)
    # Deliberately nonnumeric user order; u2 also requires history truncation.
    sequences = {"u2": ["a", "b", "c", "d"], "u1": ["a", "c"]}
    mapping = {
        "user2id": {"[PAD]": 0, "u1": 1, "u2": 2},
        "id2user": ["[PAD]", "u1", "u2"],
        "item2id": {"[PAD]": 0, "a": 1, "b": 2, "c": 3, "d": 4},
        "id2item": ["[PAD]", "a", "b", "c", "d"],
    }
    sem_path = sem_dir / "rqvae_3codebook_128.sem_ids"
    sids = {"a": [0, 1, 2], "b": [3, 4, 5], "c": [0, 1, 2], "d": [7, 8, 9]}
    for path, obj in [(data / "sequences" / "all_item_seqs.json", sequences),
                      (data / "sequences" / "id_mapping.json", mapping),
                      (sem_path, sids)]:
        path.write_text(json.dumps(obj))
    (data / "id_maps" / f"{CATEGORY}.user2id").write_text("u1\t0\nu2\t1\n")
    (data / "id_maps" / f"{CATEGORY}.item2id").write_text("a\t0\nb\t1\nc\t2\nd\t3\n")
    entry = {
        "task": "next1", "status": "trained", "quantizer": "rqvae",
        "n_codebook": 3, "codebook_size": 128,
        "sem_ids_name": sem_path.stem, "sem_ids_path": str(sem_path),
        "sem_ids_sha256": sha256_file(sem_path),
        "ckpt_path": "ckpt/model.bin", "log_path": "ckpt/log.txt", "transcript_path": None,
        "config": {"n_digit": 3, "codebook_size": 128, "max_history_len": 2,
                   "n_target_items": 1, "n_target_digits": 3},
    }
    return root, entry, sequences, sids


def _load(fixture):
    root, entry, _, _ = fixture
    return load_eval_cohort(entry, frozen_root=root, expected_users=2, expected_items=4)


def test_identity_truncation_padding_and_lossless_collisions(cohort_fixture):
    cohort = _load(cohort_fixture)
    assert cohort.users == ("u2", "u1")
    assert cohort.user_ids.tolist() == [1, 0]
    assert cohort.target_asins == ("d", "c")
    assert cohort.target_item_ids.tolist() == [3, 2]
    assert cohort.histories.tolist() == [[[3, 4, 5], [0, 1, 2]], [[0, 1, 2], [-1, -1, -1]]]
    assert cohort.history_mask.tolist() == [[True, True], [True, False]]
    assert cohort.history_lengths.tolist() == [2, 1]
    assert cohort.sequence_lengths.tolist() == [4, 2]
    assert cohort.target_sids.tolist() == [[7, 8, 9], [0, 1, 2]]
    assert cohort.sid_buckets[(0, 1, 2)] == (0, 2)
    assert sum(map(len, cohort.sid_buckets.values())) == 4
    assert len(cohort.input_sha256) == 5
    assert cohort.histories.dtype == np.int64
    assert cohort.history_mask.dtype == np.bool_


def test_exact_equivalence_to_vendored_tokenizer_methods(cohort_fixture):
    """Execute actual vendor method ASTs without importing its cache-writing stack."""
    _, entry, sequences, sids = cohort_fixture
    source = paths.VENDOR / "diffgrm/genrec/models/DIFF_GRM/tokenizer.py"
    tree = ast.parse(source.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "DIFF_GRMTokenizer")
    wanted = {"encode_history_with_mask", "encode_decoder_input", "tokenize_function"}
    methods = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    stub = SimpleNamespace(
        config=entry["config"], n_digit=3, codebook_size=128, sid_offset=3, pad_token=0,
        item2tokens={a: [c + 3 + d * 128 for d, c in enumerate(codes)] for a, codes in sids.items()},
    )
    for name in wanted:
        setattr(stub, name, namespace[name].__get__(stub))
    cohort = _load(cohort_fixture)
    for row, user in enumerate(cohort.users):
        historical = stub.tokenize_function({"user": user, "item_seq": sequences[user]}, "test")
        assert historical["history_sid"] == cohort.histories[row].tolist()
        assert historical["history_mask"] == cohort.history_mask[row].tolist()
        assert historical["labels"] == cohort.target_sids[row].tolist()


def test_unknown_history_item_is_not_silently_padded(cohort_fixture):
    root, _, sequences, _ = cohort_fixture
    sequences["u2"][0] = "unknown-even-outside-truncated-history"
    (root / "data/sequences/all_item_seqs.json").write_text(json.dumps(sequences))
    with pytest.raises(ValueError, match="unknown item"):
        _load(cohort_fixture)


def test_wrong_sid_hash_is_rejected(cohort_fixture):
    cohort_fixture[1]["sem_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        _load(cohort_fixture)


def test_mispacked_source_cannot_replace_registered_sem_ids(cohort_fixture):
    cohort_fixture[1]["sem_ids_path"] = "sids/mispacked/rqvae_3codebook_128.sem_ids"
    with pytest.raises(ValueError, match="canonical diffgrm"):
        _load(cohort_fixture)


def test_id_mapping_shift_is_checked(cohort_fixture):
    root, _, _, _ = cohort_fixture
    path = root / "data/sequences/id_mapping.json"
    mapping = json.loads(path.read_text())
    mapping["item2id"]["a"] = 0
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="canonical IDs"):
        _load(cohort_fixture)


def test_full_cohort_cardinality_required(cohort_fixture):
    root, entry, _, _ = cohort_fixture
    with pytest.raises(ValueError, match="complete 6297-user"):
        load_eval_cohort(entry, frozen_root=root)


def test_cohort_hash_changes_with_row_order_but_not_sid_assignment(cohort_fixture):
    root, entry, sequences, sids = cohort_fixture
    before = _load(cohort_fixture)
    sids["a"] = [9, 9, 9]
    sem_path = Path(entry["sem_ids_path"])
    sem_path.write_text(json.dumps(sids))
    entry["sem_ids_sha256"] = sha256_file(sem_path)
    different_sid = _load(cohort_fixture)
    assert before.cohort_sha256 == different_sid.cohort_sha256
    assert before.input_sha256 != different_sid.input_sha256
    (root / "data/sequences/all_item_seqs.json").write_text(json.dumps(dict(reversed(list(sequences.items())))))
    reordered = _load(cohort_fixture)
    assert before.cohort_sha256 != reordered.cohort_sha256


@pytest.mark.parametrize("bad_codes", [[0, 128, 1], [0, 1], [True, 1, 2]])
def test_invalid_sid_contents_are_rejected(cohort_fixture, bad_codes):
    _, entry, _, sids = cohort_fixture
    sids["a"] = bad_codes
    path = Path(entry["sem_ids_path"])
    path.write_text(json.dumps(sids))
    entry["sem_ids_sha256"] = sha256_file(path)
    with pytest.raises(ValueError):
        _load(cohort_fixture)
