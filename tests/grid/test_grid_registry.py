"""Grid discovery, registry and fingerprints on a synthetic OneDiffRec work root.

The work root is built in a temporary folder with OneDiffRec's own layout and
its own provenance fingerprints (the vendored `run_provenance.fingerprint_path`),
so the registry's checks are tested against the code that wrote the records.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from sidlens import paths
from sidlens.grid import fingerprint, layout, make_name, registry, split_name, v3code


def _write(p: Path, text: str | bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_bytes if isinstance(text, bytes) else p.write_text)(text)
    return p


def _result(run_dir: Path, phase: str, artifacts: dict, status="completed"):
    prov = v3code.provenance()
    rec = {"status": status, "artifacts": {k: prov.fingerprint_path(v) for k, v in artifacts.items()}}
    _write(run_dir / "provenance" / phase / "result.json", json.dumps(rec))


def _work_root(tmp: Path) -> Path:
    root = tmp / "work"
    ds = root / "industrial-v3"
    data = ds / "data" / "onediffrec" / "Amazon18" / "Industrial_and_Scientific"
    sid = _write(data / "Industrial_and_Scientific.rqkmeans.index_3codebook_8.json",
                 json.dumps({"0": ["<a_1>", "<b_2>", "<c_3>"], "1": ["<a_1>", "<b_2>", "<c_4>"]}))
    _write(data / "Industrial_and_Scientific.item2id", "A0\t0\nA1\t1\n")
    prepared = ds / "prepared" / "disjoint_horizon_v3" / "analysis-v3" / "industrial" / "next-item" / \
        "rqkmeans_3codebook_8"
    _write(prepared / "data-manifest.json", json.dumps({"input_paths": {"sid_index": str(sid)},
                                                        "output_hashes": {}, "rows": {"test": 1}}))
    source = ds / "sources" / "abc"
    for rel in v3code.manifest()["files"]:
        dst = source / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v3code.ROOT / rel, dst)
    base = ds / "experiments" / "grid-v1" / "full" / "industrial" / "next-item"
    cell = {"dataset": "industrial", "predtype": "next-item", "method": "rqkmeans", "digits": 3, "size": 8,
            "seed": 42, "sid_assignment": "native", "assignment_id": "native", "prepared_dir": str(prepared),
            "data_root": str(ds / "data" / "onediffrec"), "cache_root": str(ds / "data" / "cache"),
            "repo": str(source), "sid_index_sha256": __import__("hashlib").sha256(sid.read_bytes()).hexdigest()}
    # AR: best + one epoch snapshot
    ar = base / "ar" / "rqkmeans_3codebook_8" / "seed-42"
    _write(ar / "cell.json", json.dumps({**cell, "paradigm": "ar"}))
    _write(ar / "training" / "final_checkpoint" / "model.safetensors", os.urandom(5 << 20))
    _write(ar / "training" / "final_checkpoint" / "config.json", "{}")
    _write(ar / "training" / "snapshots" / "epoch-0.5" / "model.safetensors", os.urandom(1000))
    _write(ar / "training" / "snapshots" / "epoch-0.5" / "training_args.bin", b"args")
    _write(ar / "training" / "snapshots" / "index.json", json.dumps(
        {"requested_epochs": [0.5, 1.5], "snapshots": [
            {"requested_epoch": 0.5, "epoch": 0.51, "global_step": 7, "eval_loss": 1.0, "path": "epoch-0.5"}]}))
    _write(ar / "evaluation" / "predictions.jsonl", "{}\n")
    _result(ar, "train", {"checkpoint": ar / "training" / "final_checkpoint",
                          "epoch_snapshots": ar / "training" / "snapshots"})
    _result(ar, "eval", {"predictions": ar / "evaluation" / "predictions.jsonl"})
    # DiffGRM: best + one snapshot
    dg = base / "diffgrm" / "rqkmeans_3codebook_8" / "seed-42"
    _write(dg / "cell.json", json.dumps({**cell, "paradigm": "diffgrm"}))
    _write(dg / "training" / "pytorch_model.bin", os.urandom(2000))
    _write(dg / "training" / "intermediate" / "epoch-0020.bin", os.urandom(2000))
    _write(dg / "training" / "resolved-config.json", json.dumps({"n_digit": 3, "max_history_len": 50}))
    _write(dg / "training" / "history.jsonl", '{"epoch": 1, "best_epoch": 1}\n{"epoch": 2, "best_epoch": 1}\n')
    _write(dg / "evaluation" / "predictions.jsonl", "{}\n")
    _result(dg, "train", {"checkpoint": dg / "training" / "pytorch_model.bin",
                          "intermediate_checkpoints": dg / "training" / "intermediate"})
    _result(dg, "eval", {"predictions": dg / "evaluation" / "predictions.jsonl"})
    return root


@pytest.fixture()
def grid_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results_remote")
    monkeypatch.setattr(paths, "CACHE", tmp_path / "cache")
    return _work_root(tmp_path)


def test_fingerprint_equals_onediffrec(grid_root):
    prov = v3code.provenance()
    for p in grid_root.rglob("*"):
        if p.is_dir() and p.name in ("final_checkpoint", "snapshots", "intermediate"):
            assert fingerprint.fingerprint(p)["sha256"] == prov.fingerprint_path(p)["sha256"]
        if p.is_file() and p.suffix in (".bin", ".jsonl"):
            assert fingerprint.fingerprint(p)["sha256"] == prov.fingerprint_path(p)["sha256"]


def test_discover_and_states(grid_root):
    runs = {r.paradigm: r for r in layout.discover(grid_root)}
    assert set(runs) == {"ar", "diffgrm"}
    assert sorted(runs["ar"].states) == ["best", "epoch-0.5"]
    assert sorted(runs["diffgrm"].states) == ["best", "epoch-0020"]
    assert runs["diffgrm"].best_epoch == 1
    assert runs["ar"].key == "grid-v1/industrial/next-item/ar/rqkmeans_3codebook_8/seed-42"


def test_identity_mismatch_raises(grid_root):
    cell = next(grid_root.rglob("ar/rqkmeans_3codebook_8/seed-42/cell.json"))
    doc = json.loads(cell.read_text())
    cell.write_text(json.dumps({**doc, "digits": 4}))
    with pytest.raises(ValueError, match="cell.json says"):
        layout.discover(grid_root)


def test_registry_build_save_and_verify(grid_root):
    doc = registry.build(grid_root, workers=2, log=None)
    for e in doc["entries"].values():
        assert e["usable"], e["problems"]
        assert set(e["checks"].values()) == {"ok"}
        assert e["source_check"]["ok"]
    path = registry.save(doc)
    assert (path.parent / "coverage.csv").is_file()
    reg = registry.load()
    name = make_name("grid-v1/industrial/next-item/ar/rqkmeans_3codebook_8/seed-42", "epoch-0.5")
    assert split_name(name)[1] == "epoch-0.5"
    sha = registry.verify_state(name)
    assert sha == reg.state(name)[1]["files"]["model.safetensors"]
    assert sorted(reg.names(paradigm="ar", states="all"))[0].endswith("@best")
    # a changed weight file is refused at load time
    weights = Path(reg.state(name)[1]["path"]) / "model.safetensors"
    weights.write_bytes(os.urandom(1000))
    with pytest.raises(RuntimeError, match="sha256"):
        registry.verify_state(name)
    # rebuilding with different content moves the old registry aside
    doc2 = registry.build(grid_root, workers=2, log=None)
    assert not any(e["usable"] for e in doc2["entries"].values() if e["paradigm"] == "ar")
    registry.save(doc2)
    assert list(path.parent.glob("registry.json.replaced-*"))


def test_source_mismatch_makes_run_unusable(grid_root):
    src = next(grid_root.rglob("sources/abc/scripts/evaluate_ar_joint.py"))
    src.write_text(src.read_text() + "\n# changed\n")
    doc = registry.build(grid_root, workers=1, log=None)
    for e in doc["entries"].values():
        assert not e["usable"]
        assert e["source_check"]["mismatch"] == ["scripts/evaluate_ar_joint.py"]
