"""Activation store: round trip, identity of rows, and refusal of half-written stores."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("pyarrow")

from sidlens.hooks.store import StoreReader, StoreWriter  # noqa: E402


def _rows(start, n):
    return [{"example_id": f"ex{start + i}", "pos": i, "role": "hist_sid", "digit": i % 3}
            for i in range(n)]


def _write(root, shard_bytes=1 << 30, batches=3, n=5):
    data = {"model.layers.0": [], "model.layers.1": []}
    with StoreWriter(root, {"model": "toy", "template": "eval"},
                     shard_bytes=shard_bytes) as w:
        for b in range(batches):
            t0 = torch.arange(b * n * 4, (b + 1) * n * 4, dtype=torch.float32).reshape(n, 4)
            t1 = -t0[:, :2].to(torch.bfloat16)
            data["model.layers.0"].append(t0)
            data["model.layers.1"].append(t1)
            w.add(_rows(b * n, n), {"model.layers.0": t0, "model.layers.1": t1})
    return {k: torch.cat(v) for k, v in data.items()}


@pytest.mark.parametrize("shard_bytes", [1 << 30, 64])
def test_round_trip_across_shards(tmp_path, shard_bytes):
    """Tiny shard_bytes forces one shard per add; rows must still line up."""
    want = _write(tmp_path / "s", shard_bytes=shard_bytes)
    r = StoreReader(tmp_path / "s", verify=True)
    assert len(r.manifest["shards"]) == (1 if shard_bytes > 1000 else 3)
    assert r.sites == ["model.layers.0", "model.layers.1"]
    assert r.manifest["sites"]["model.layers.1"] == {"width": 2, "dtype": "bfloat16"}
    for site, t in want.items():
        assert torch.equal(r.load(site), t)
    pick = [14, 0, 7]
    assert torch.equal(r.load("model.layers.0", pick), want["model.layers.0"][pick])
    assert list(r.rows.loc[pick, "example_id"]) == ["ex14", "ex0", "ex7"]
    assert r.meta["provenance"]["git_head"]


def test_existing_directory_is_never_reused(tmp_path):
    _write(tmp_path / "s")
    with pytest.raises(FileExistsError):
        _write(tmp_path / "s")


def test_sites_must_cover_the_same_rows(tmp_path):
    with pytest.raises(ValueError, match=r"\(3, width\)"):
        with StoreWriter(tmp_path / "s", {"model": "toy"}) as w:
            w.add(_rows(0, 3), {"a": torch.zeros(3, 4), "b": torch.zeros(2, 4)})
    assert (tmp_path / "s" / "FAILED").exists()
    with pytest.raises(RuntimeError, match="not a complete store"):
        StoreReader(tmp_path / "s")


def test_width_and_fields_cannot_drift(tmp_path):
    w = StoreWriter(tmp_path / "s", {"model": "toy"})
    w.add(_rows(0, 2), {"a": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="changed"):
        w.add(_rows(2, 2), {"a": torch.zeros(2, 5)})
    with pytest.raises(ValueError, match="changed"):
        w.add([{"example_id": "x", "pos": 0}], {"a": torch.zeros(1, 4)})
    with pytest.raises(KeyError, match="required"):
        w.add([{"pos": 0}], {"a": torch.zeros(1, 4)})


def test_interrupted_store_is_refused(tmp_path):
    w = StoreWriter(tmp_path / "s", {"model": "toy"})
    w.add(_rows(0, 2), {"a": torch.zeros(2, 4)})
    # no close(): no manifest
    with pytest.raises(RuntimeError, match="no manifest"):
        StoreReader(tmp_path / "s")


def test_tampered_shard_is_detected(tmp_path):
    _write(tmp_path / "s")
    shard = next((tmp_path / "s" / "shards").glob("*.safetensors"))
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError, match="sha256"):
        StoreReader(tmp_path / "s", verify=True)


def test_meta_must_name_the_model(tmp_path):
    with pytest.raises(ValueError, match="model"):
        StoreWriter(tmp_path / "s", {"template": "eval"})
    assert not (tmp_path / "s").exists()
    json.dumps({"ok": True})
