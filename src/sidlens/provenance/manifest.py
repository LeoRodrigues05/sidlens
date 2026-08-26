"""The provenance manifest: what bytes produced a result.

Every artifact the project emits carries a `snapshot_id`. Resolving that id
against a manifest yields the exact upstream state, vendored code, environment,
data files, and checkpoints in play -- so any claim can be traced back to bytes.
"""

from __future__ import annotations

import getpass
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sidlens import paths
from sidlens.provenance import hashing

SCHEMA_VERSION = 1


def env_section() -> dict:
    """Versions AND resolved __file__.

    A version string alone has been known to lie -- two interpreters can report
    the same torch version while importing different builds. The resolved path
    is what disambiguates.
    """
    mods = {}
    for name in ("torch", "transformers", "numpy", "datasets",
                 "accelerate", "faiss", "scipy", "sklearn"):
        try:
            m = __import__(name)
            mods[name] = {
                "version": getattr(m, "__version__", "unknown"),
                "file": getattr(m, "__file__", None),
            }
        except Exception as exc:
            mods[name] = {"version": None, "error": f"{type(exc).__name__}: {exc}"}

    cuda = {}
    try:
        import torch
        cuda = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "devices": [torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else [],
        }
    except Exception:
        pass

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True, check=False).stdout
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "modules": mods,
        "cuda": cuda,
        "pip_freeze_sha256": hashing.sha256_bytes(freeze.encode()),
        "pip_freeze": freeze,
    }


def build(
    *,
    snapshot_id: str,
    upstream: dict,
    vendor: dict,
    data_sections: dict,
    in_flight: list,
    patches: dict | None = None,
) -> dict:
    vendor_trees = {
        name: hashing.merkle_root(
            {k: hashing.FileHash(k, v["sha256"], v["size"], 0)
             for k, v in blob["files"].items()}
        )
        for name, blob in vendor.items()
        if blob.get("files")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": upstream,
        "vendor": {"trees": vendor, "merkle_roots": vendor_trees},
        "env": env_section(),
        "data": data_sections,
        "patches": patches or {},
        "in_flight_jobs": in_flight,
        "roots": {
            "repo": str(paths.REPO),
            "work": str(paths.WORK),
            "frozen": str(paths.FROZEN),
            "upstream_repo": str(paths.UPSTREAM_REPO),
            "upstream_work": str(paths.UPSTREAM_WORK),
        },
    }


def path_for(snapshot_id: str) -> Path:
    return paths.MANIFESTS / f"provenance.{snapshot_id}.json"


def save(manifest: dict) -> Path:
    paths.MANIFESTS.mkdir(parents=True, exist_ok=True)
    p = path_for(manifest["snapshot_id"])
    p.write_text(json.dumps(manifest, indent=2, sort_keys=False))
    (paths.MANIFESTS / "CURRENT").write_text(manifest["snapshot_id"] + "\n")
    return p


def current_id() -> str:
    f = paths.MANIFESTS / "CURRENT"
    if not f.exists():
        raise FileNotFoundError(
            f"no frozen snapshot yet ({f} missing) -- run `sidlens freeze` first")
    return f.read_text().strip()


def load(snapshot_id: str | None = None) -> dict:
    return json.loads(path_for(snapshot_id or current_id()).read_text())


def summarize(manifest: dict) -> str:
    d = manifest["data"]
    n = sum(s["n_files"] for s in d.values())
    b = sum(s["bytes"] for s in d.values())
    lines = [
        f"snapshot   {manifest['snapshot_id']}",
        f"created    {manifest['created_utc']}",
        f"upstream   {manifest['upstream']['head'][:12]} "
        f"({manifest['upstream']['n_dirty_paths']} dirty paths)",
        f"frozen     {n} files, {b/1e9:.2f} GB, {len(d)} sections",
        "vendor     " + "  ".join(
            f"{k}={v[:8]}" for k, v in manifest["vendor"]["merkle_roots"].items()),
    ]
    if manifest["in_flight_jobs"]:
        lines.append("IN FLIGHT  " + "; ".join(
            f"{j['jobid']} {j['name']} ({j['elapsed']})"
            for j in manifest["in_flight_jobs"]))
    return "\n".join(lines)
