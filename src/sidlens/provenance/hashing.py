"""Content hashing with a stat-keyed cache.

Verification that nobody runs is worth nothing, so hashing 15 GB has to stay
fast enough that `sidlens verify` is cheap to put at the head of every job.
Small/text files are always hashed in full; large binaries are hashed once and
the digest cached against (size, mtime_ns, inode, device). Any stat change
invalidates the entry and forces a rehash.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

CHUNK = 1 << 20
# Below this, hashing is cheaper than a cache round-trip.
CACHE_MIN_BYTES = 4 << 20


@dataclass(frozen=True)
class FileHash:
    path: str  # relative to whatever root the caller passed
    sha256: str
    size: int
    mtime_ns: int

    def as_dict(self) -> dict:
        return {"sha256": self.sha256, "size": self.size, "mtime_ns": self.mtime_ns}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HashCache:
    """SQLite-backed digest cache keyed by identity + stat.

    Concurrency: several SLURM steps may verify at once, so the connection is
    opened per-thread and writes use the default rollback journal with a busy
    timeout rather than assuming exclusive access.
    """

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            from sidlens.paths import CACHE

            db_path = CACHE / "hashes.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS hashes (
                   path TEXT PRIMARY KEY,
                   size INTEGER NOT NULL,
                   mtime_ns INTEGER NOT NULL,
                   ino INTEGER NOT NULL,
                   dev INTEGER NOT NULL,
                   sha256 TEXT NOT NULL)"""
        )
        self._conn.commit()

    def get(self, path: Path) -> str:
        st = path.stat()
        key = str(path.resolve())
        if st.st_size < CACHE_MIN_BYTES:
            return sha256_file(path)
        row = self._conn.execute(
            "SELECT size, mtime_ns, ino, dev, sha256 FROM hashes WHERE path = ?", (key,)
        ).fetchone()
        if row and (row[0], row[1], row[2], row[3]) == (
            st.st_size,
            st.st_mtime_ns,
            st.st_ino,
            st.st_dev,
        ):
            return row[4]
        digest = sha256_file(path)
        self._conn.execute(
            "INSERT OR REPLACE INTO hashes VALUES (?,?,?,?,?,?)",
            (key, st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev, digest),
        )
        self._conn.commit()
        return digest


def walk_files(root: Path, exclude: Iterable[str] = ()) -> Iterator[Path]:
    """Yield regular files under `root`, sorted, skipping excluded name parts.

    Symlinks are NOT followed and NOT yielded: the upstream run directories are
    symlink farms into the live tree, and silently hashing through them would
    reintroduce exactly the coupling the freeze exists to break.
    """
    exclude = set(exclude) | {"__pycache__", ".git", ".ipynb_checkpoints"}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            if name in exclude:
                continue
            yield p


def hash_tree(
    root: Path, exclude: Iterable[str] = (), cache: HashCache | None = None
) -> dict[str, FileHash]:
    """Hash every regular file under `root`, keyed by POSIX-relative path."""
    cache = cache or HashCache()
    out: dict[str, FileHash] = {}
    for p in walk_files(root, exclude):
        st = p.stat()
        rel = p.relative_to(root).as_posix()
        out[rel] = FileHash(rel, cache.get(p), st.st_size, st.st_mtime_ns)
    return out


def merkle_root(hashes: dict[str, FileHash]) -> str:
    """Order-independent digest over a tree, so vendor drift is one comparison."""
    h = hashlib.sha256()
    for rel in sorted(hashes):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashes[rel].sha256.encode())
        h.update(b"\n")
    return h.hexdigest()


def tree_to_json(hashes: dict[str, FileHash]) -> dict:
    return {rel: fh.as_dict() for rel, fh in sorted(hashes.items())}


def diff_trees(
    expected: dict, actual: dict[str, FileHash]
) -> dict[str, list[str]]:
    """Compare a manifest section against a freshly hashed tree."""
    exp_keys, act_keys = set(expected), set(actual)
    changed = [
        k for k in sorted(exp_keys & act_keys)
        if expected[k]["sha256"] != actual[k].sha256
    ]
    return {
        "missing": sorted(exp_keys - act_keys),
        "added": sorted(act_keys - exp_keys),
        "changed": changed,
    }


def stable_json_hash(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())
