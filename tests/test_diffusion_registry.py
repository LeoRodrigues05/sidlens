"""Runtime portability for the diffusion checkpoint registry."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from sidlens.models import diffusion as model_diffusion
from sidlens.registry import diffusion as registry


def _entry(source_root: str = "/source/work/frozen") -> dict:
    return {
        "ckpt_id": "diff-next1-test",
        "ckpt_path": f"{source_root}/ckpt/diffusion/model.bin",
        "ckpt_sha256": "checkpoint-hash",
        "sem_ids_path": f"{source_root}/sids/sem_ids/diffgrm/test.sem_ids",
        "sem_ids_sha256": "sid-hash",
        "log_path": f"{source_root}/ckpt/diffusion/logs/test.log",
        "log_sha256": "log-hash",
        "transcript_path": f"{source_root}/results/transcripts/test.txt",
        "transcript_sha256": "transcript-hash",
        "recorded_metrics": {"ndcg@10": 0.25},
    }


def test_runtime_load_rebases_paths_without_rewriting_provenance(
        tmp_path, monkeypatch):
    manifest_dir = tmp_path / "repo" / "manifests"
    manifest_dir.mkdir(parents=True)
    payload = {"diff-next1-test": _entry()}
    manifest = manifest_dir / "registry.diffusion.json"
    manifest.write_text(json.dumps(payload, indent=2))
    recorded_bytes = manifest.read_bytes()
    target_frozen = tmp_path / "target-work" / "frozen"

    monkeypatch.setattr(registry.paths, "MANIFESTS", manifest_dir)
    monkeypatch.setattr(registry.paths, "FROZEN", target_frozen)

    loaded = registry.load_runtime()["diff-next1-test"]

    assert loaded["ckpt_path"] == str(
        target_frozen / "ckpt/diffusion/model.bin")
    assert loaded["sem_ids_path"] == str(
        target_frozen / "sids/sem_ids/diffgrm/test.sem_ids")
    assert loaded["log_path"] == str(
        target_frozen / "ckpt/diffusion/logs/test.log")
    assert loaded["transcript_path"] == str(
        target_frozen / "results/transcripts/test.txt")
    assert loaded["ckpt_sha256"] == "checkpoint-hash"
    assert loaded["sem_ids_sha256"] == "sid-hash"
    assert loaded["log_sha256"] == "log-hash"
    assert loaded["transcript_sha256"] == "transcript-hash"
    assert loaded["recorded_metrics"] == {"ndcg@10": 0.25}

    # Resolution is runtime-only: the tracked source record remains an exact
    # audit trail and can still be requested explicitly.
    assert manifest.read_bytes() == recorded_bytes
    assert registry.load() == payload


def test_resolve_entry_paths_does_not_mutate_input_and_allows_no_transcript(
        tmp_path):
    entry = _entry()
    entry["transcript_path"] = None
    entry["transcript_sha256"] = None
    original = deepcopy(entry)

    resolved = registry.resolve_entry_paths(entry, tmp_path / "frozen")

    assert entry == original
    assert resolved is not entry
    assert resolved["transcript_path"] is None
    assert resolved["transcript_sha256"] is None
    assert resolved["ckpt_path"] == str(
        tmp_path / "frozen/ckpt/diffusion/model.bin")


@pytest.mark.parametrize("bad_path", [
    "/source/work/frozen/results/not-a-checkpoint.bin",
    "/source/work/frozen/ckpt/../outside/model.bin",
    "/source/frozen/ckpt/nested/frozen/ckpt/model.bin",
])
def test_resolve_entry_paths_rejects_unsafe_checkpoint_paths(
        tmp_path, bad_path):
    entry = _entry()
    entry["ckpt_path"] = bad_path

    with pytest.raises(ValueError, match="ckpt_path"):
        registry.resolve_entry_paths(entry, tmp_path / "frozen")


def test_model_load_by_id_uses_registry_loader(monkeypatch):
    entry = {"ckpt_id": "diff-next1-test", "ckpt_path": "/rebased/model.bin"}
    seen = {}
    sentinel = (object(), {"ok": True})

    monkeypatch.setattr(
        model_diffusion.diffusion_registry,
        "load_runtime",
        lambda: {"diff-next1-test": entry},
    )

    def fake_model_load(got, device="cpu", strict=True):
        seen.update(entry=got, device=device, strict=strict)
        return sentinel

    monkeypatch.setattr(model_diffusion, "load", fake_model_load)

    assert model_diffusion.load_by_id(
        "diff-next1-test", device="cuda:1", strict=False) is sentinel
    assert seen == {"entry": entry, "device": "cuda:1", "strict": False}
