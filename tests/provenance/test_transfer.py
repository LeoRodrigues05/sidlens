"""Profiles, profile-scoped verify, and bundle create/check/push/pull.

Everything except the first two tests runs against a synthetic WORK root and a
fake Hub client, so these tests touch neither the real substrate nor the
network. The fakes implement the four Hub calls `bundle` makes, with the same
file semantics.
"""

import fnmatch
import json
import shutil
import stat
import types
from pathlib import Path

import pytest

from sidlens import paths
from sidlens.provenance import bundle, manifest as manifest_mod, profiles, verify
from sidlens.provenance.hashing import sha256_file


# ------------------------------------------------- against the real manifest --
def test_profiles_exclude_exactly_the_weight_files():
    man = manifest_mod.load()
    weights = {f"{dest}/{rel}" for dest, sec in man["data"].items() for rel in sec["files"]
               if rel == "model.safetensors"}
    assert len(weights) == 4
    for name, n_excluded in (("core", 4), ("ar", 1), ("full", 0)):
        prof = profiles.get(name)
        hit = {f"{d}/{r}" for d, sec in man["data"].items() for r in sec["files"]
               if prof.excludes(d, r)}
        assert hit <= weights and len(hit) == n_excluded, name
    # the tokenizer that every AR prompt needs is never excluded
    assert not profiles.get("core").excludes("ckpt/ar/next-item_best", "added_tokens.json")


def test_every_named_section_exists():
    man = manifest_mod.load()
    for prof in profiles.PROFILES.values():
        for sec in prof.frozen_sections or ():
            assert sec in man["data"], (prof.name, sec)


# ---------------------------------------------------------- synthetic WORK --
FILES = {
    "sids/sem_ids/diffgrm": {"x.sem_ids": b'{"A": [1, 2, 3]}'},
    "ckpt/ar/next-item_best": {"added_tokens.json": b'{"<a_1>": 5}',
                               "model.safetensors": b"W" * 64},
    "base_models/qwen": {"model.safetensors": b"B" * 32, "config.json": b"{}"},
}


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A WORK root with a 3-section frozen tree, a manifest, and a derived tree."""
    w, repo = tmp_path / "work", tmp_path / "repo"
    (repo / "manifests").mkdir(parents=True)
    (repo / "vendor").mkdir()
    data = {}
    for dest, files in FILES.items():
        recs = {}
        for rel, blob in files.items():
            p = w / "frozen" / dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
            p.chmod(0o444)
            recs[rel] = {"sha256": sha256_file(p), "size": len(blob), "mtime_ns": 0}
        data[dest] = {"kind": "x", "n_files": len(recs), "bytes": 0, "files": recs}
    man = {"schema_version": 1, "snapshot_id": "snap-test",
           "vendor": {"trees": {}, "merkle_roots": {}}, "data": data}
    (repo / "manifests" / "provenance.snap-test.json").write_text(json.dumps(man))
    (repo / "manifests" / "CURRENT").write_text("snap-test\n")
    (w / "derived" / "exp").mkdir(parents=True)
    (w / "derived" / "exp" / "r.csv").write_text("a,b\n1,2\n")
    (w / "external" / "labels").mkdir(parents=True)
    (w / "external" / "labels" / "l.json").write_text("{}")
    (w / "external" / "amazon2018").mkdir(parents=True)
    (w / "external" / "amazon2018" / "m.json.gz").write_bytes(b"gz")
    for name, val in (("WORK", w), ("FROZEN", w / "frozen"), ("CACHE", w / "cache"),
                      ("DERIVED", w / "derived"), ("MANIFESTS", repo / "manifests"),
                      ("VENDOR", repo / "vendor")):
        monkeypatch.setattr(paths, name, val)
    yield w
    for p in w.rglob("*"):          # let tmp_path cleanup delete read-only files
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | stat.S_IWUSR)


def test_verify_full_requires_weights_but_core_does_not(work):
    (work / "frozen/ckpt/ar/next-item_best/model.safetensors").chmod(0o644)
    (work / "frozen/ckpt/ar/next-item_best/model.safetensors").unlink()
    assert not verify.verify_snapshot()["ok"]
    rep = verify.verify_snapshot(profile="core")
    assert rep["ok"] and rep["excluded_files"] == 2
    assert "NOT a check of the whole snapshot" in verify.format_report(rep)
    # ...but a non-excluded file going missing still fails the profiled check
    (work / "frozen/ckpt/ar/next-item_best/added_tokens.json").chmod(0o644)
    (work / "frozen/ckpt/ar/next-item_best/added_tokens.json").unlink()
    assert not verify.verify_snapshot(profile="core")["ok"]


def test_verify_profile_still_catches_drift_and_writability(work):
    p = work / "frozen/sids/sem_ids/diffgrm/x.sem_ids"
    p.chmod(0o644)
    assert not verify.verify_snapshot(profile="core")["ok"]      # writable
    p.write_bytes(b'{"A": [1, 2, 4]}')
    p.chmod(0o444)
    rep = verify.verify_snapshot(profile="core")
    assert not rep["ok"] and rep["sections"]["sids/sem_ids/diffgrm"]["n_changed"] == 1


def test_plan_and_create_certify_frozen_and_hash_extras(work):
    plan = bundle.plan("core")
    assert "frozen/ckpt/ar/next-item_best/model.safetensors" not in plan["files"]
    assert "frozen/ckpt/ar/next-item_best/added_tokens.json" in plan["files"]
    assert plan["files"]["derived/exp/r.csv"]["sha256"] is None
    out = bundle.create("core", bundle_id="b1")
    doc = bundle.load("b1")
    assert out == work / "bundles" / "b1.json"
    assert doc["files"]["derived/exp/r.csv"]["sha256"] == sha256_file(work / "derived/exp/r.csv")
    assert doc["totals"]["frozen_files"] == 3 and doc["totals"]["extra_files"] == 3
    assert bundle.check(doc)["ok"]
    with pytest.raises(bundle.BundleError, match="already exists"):
        bundle.create("core", bundle_id="b1")


def test_create_refuses_a_drifted_substrate(work):
    p = work / "frozen/sids/sem_ids/diffgrm/x.sem_ids"
    p.chmod(0o644)
    p.write_bytes(b'{"A": [9, 9, 9]}')
    with pytest.raises(bundle.BundleError, match="provenance manifest"):
        bundle.create("core", bundle_id="b2")
    assert not (work / "bundles" / "b2.json").exists()


def test_check_detects_a_changed_extra(work):
    bundle.create("core", bundle_id="b3")
    (work / "derived/exp/r.csv").write_text("a,b\n1,3\n")
    rep = bundle.check(bundle.load("b3"))
    assert not rep["ok"] and rep["changed"] == ["derived/exp/r.csv"]


@pytest.mark.parametrize("bad", ["../x", "/abs", "frozen/sids", "bundles", "env", "nope"])
def test_extra_paths_are_confined(work, bad):
    with pytest.raises(bundle.BundleError):
        bundle.plan("results", (bad,))


def test_results_profile_sends_only_the_named_tree(work):
    p = bundle.plan("results", ("derived/exp",))
    assert list(p["files"]) == ["derived/exp/r.csv"]


# ---------------------------------------------------------------- fake Hub --
class FakeHub:
    """Remote = a directory tree; tags = {name: snapshot dir}."""

    def __init__(self, root: Path, private=True):
        self.root, self.private = root, private
        self.tags: dict[str, Path] = {}
        self.uploaded: list[str] = []

    # HfApi surface
    def repo_info(self, repo_id, repo_type):
        return types.SimpleNamespace(private=self.private)

    def list_repo_refs(self, repo_id, repo_type):
        return types.SimpleNamespace(tags=[types.SimpleNamespace(name=t) for t in self.tags])

    def upload_large_folder(self, repo_id, repo_type, folder_path, num_workers=None):
        src = Path(folder_path)
        for f in src.rglob("*"):
            if ".cache" in f.parts or f.is_dir():
                continue
            rel = f.relative_to(src).as_posix()
            dst = self.root / "main" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)          # follows the stage symlinks
            self.uploaded.append(rel)

    def create_tag(self, repo_id, repo_type, tag, tag_message=""):
        snap = self.root / "tags" / tag
        shutil.copytree(self.root / "main", snap)
        self.tags[tag] = snap

    # module-level download functions
    def hf_hub_download(self, repo_id, filename, repo_type, revision):
        return str(self.tags[revision] / filename)

    def snapshot_download(self, repo_id, repo_type, revision, local_dir, allow_patterns):
        snap = self.tags[revision]
        for f in snap.rglob("*"):
            rel = f.relative_to(snap).as_posix()
            if f.is_file() and any(fnmatch.fnmatchcase(rel, p) for p in allow_patterns):
                dst = Path(local_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
        return local_dir


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub
    fake = FakeHub(tmp_path / "remote")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: fake)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    return fake


def test_push_uploads_exactly_the_bundle_and_tags_it(work, hub):
    pytest.importorskip("huggingface_hub")
    bundle.create("core", bundle_id="b4")
    # a stale link in the stage dir must not ride along
    stale = work / "cache/bundle-stage/b4/derived/old.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale")
    bundle.push("b4", "me/sidlens")
    doc = bundle.load("b4")
    assert sorted(hub.uploaded) == sorted([*doc["files"], "bundles/b4.json"])
    assert "b4" in hub.tags
    with pytest.raises(bundle.BundleError, match="already exists"):
        bundle.push("b4", "me/sidlens")


def test_push_refuses_a_public_repo(work, hub):
    pytest.importorskip("huggingface_hub")
    bundle.create("core", bundle_id="b5")
    hub.private = False
    with pytest.raises(bundle.BundleError, match="PUBLIC"):
        bundle.push("b5", "me/sidlens")
    assert not hub.uploaded


def test_pull_into_an_empty_work_root_then_verify(work, hub, tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    bundle.create("core", bundle_id="b6")
    bundle.push("b6", "me/sidlens")
    # a second cluster: empty WORK, same repo checkout (manifests)
    w2 = tmp_path / "work2"
    w2.mkdir()
    for name, val in (("WORK", w2), ("FROZEN", w2 / "frozen"), ("CACHE", w2 / "cache"),
                      ("DERIVED", w2 / "derived")):
        monkeypatch.setattr(paths, name, val)
    dry = bundle.pull("b6", "me/sidlens", dry_run=True)
    assert dry["to_download"] == 6 and not (w2 / "frozen").exists()
    s = bundle.pull("b6", "me/sidlens")
    assert s["check"]["ok"]
    assert (w2 / "bundles/b6.json").exists()
    tok = w2 / "frozen/ckpt/ar/next-item_best/added_tokens.json"
    assert not tok.stat().st_mode & 0o222                      # frozen made read-only
    assert not (w2 / "frozen/ckpt/ar/next-item_best/model.safetensors").exists()
    assert verify.verify_snapshot(profile="core")["ok"]
    assert not verify.verify_snapshot()["ok"]                  # weights absent
    # pulling again is a no-op
    assert bundle.pull("b6", "me/sidlens")["to_download"] == 0
    for p in w2.rglob("*"):
        p.chmod(p.stat().st_mode | stat.S_IWUSR)


def test_pull_never_overwrites_a_differing_local_file(work, hub, tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    bundle.create("core", bundle_id="b7")
    bundle.push("b7", "me/sidlens")
    w2 = tmp_path / "work3"
    (w2 / "derived/exp").mkdir(parents=True)
    (w2 / "derived/exp/r.csv").write_text("different\n")
    for name, val in (("WORK", w2), ("FROZEN", w2 / "frozen"), ("CACHE", w2 / "cache")):
        monkeypatch.setattr(paths, name, val)
    with pytest.raises(bundle.BundleError, match="refusing to overwrite"):
        bundle.pull("b7", "me/sidlens")
    assert (w2 / "derived/exp/r.csv").read_text() == "different\n"
    assert not (w2 / "frozen").exists()


def test_pull_restores_read_only_directories(work, hub, tmp_path, monkeypatch):
    """A partial frozen/ with read-only dirs (as `chmod -R a-w` leaves it) can
    still receive files, and its directory modes come back unchanged."""
    pytest.importorskip("huggingface_hub")
    bundle.create("core", bundle_id="b8")
    bundle.push("b8", "me/sidlens")
    w2 = tmp_path / "work4"
    d = w2 / "frozen/sids/sem_ids/diffgrm"
    d.mkdir(parents=True)
    d.chmod(0o555)
    for name, val in (("WORK", w2), ("FROZEN", w2 / "frozen"), ("CACHE", w2 / "cache")):
        monkeypatch.setattr(paths, name, val)
    assert bundle.pull("b8", "me/sidlens")["check"]["ok"]
    assert stat.S_IMODE(d.stat().st_mode) == 0o555
    for p in [w2, *w2.rglob("*")]:
        p.chmod(p.stat().st_mode | stat.S_IWUSR)


@pytest.mark.parametrize("bad", ["a/b", "../x", "-x", "x..y", "x.lock", "", "a b"])
def test_bundle_ids_must_be_safe_names_and_tags(bad):
    """An id becomes bundles/<id>.json and a git tag on the Hub."""
    with pytest.raises(bundle.BundleError, match="bundle id"):
        bundle._check_id(bad)


def test_create_rejects_an_unsafe_id_before_writing(work):
    with pytest.raises(bundle.BundleError, match="bundle id"):
        bundle.create("core", bundle_id="../escape")
    assert not (work / "bundles").exists() and not (work / "escape.json").exists()
