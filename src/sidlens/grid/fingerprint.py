"""OneDiffRec's content fingerprints, computed from sidlens's cached file hashes.

Every grid run records a fingerprint of each artifact it produced
(`provenance/<phase>/result.json`): a file's sha256, or, for a directory, a
digest over every entry's relative path, size and sha256
(`run_provenance._directory_digest`). Re-deriving that digest proves the
weights sidlens loads are the bytes the run recorded.

Calling OneDiffRec's own function would read every 3 GB weight file a second
time after sidlens hashed it; the digest depends only on per-file sha256s, so
it is rebuilt here from `HashCache`, which hashes each file once. The test
suite checks this against the vendored `fingerprint_path` on real trees.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from sidlens.provenance.hashing import HashCache

TAG = "onediffrec-directory-sha256-v1"


def _feed(digest, value: str | bytes) -> None:
    data = value if isinstance(value, bytes) else os.fsencode(value)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def directory_fingerprint(root: Path, cache: HashCache | None = None) -> dict:
    """{"sha256", "size_bytes", "file_count", "directory_count", "symlink_count"},
    equal to OneDiffRec's `_directory_digest(root)`."""
    import hashlib

    cache = cache or HashCache()
    digest = hashlib.sha256()
    _feed(digest, TAG)
    totals = {"size_bytes": 0, "file_count": 0, "directory_count": 0, "symlink_count": 0}

    def visit(directory: Path, relative: Path) -> None:
        for entry in sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name)):
            rel = (relative / entry.name).as_posix()
            st = entry.stat(follow_symlinks=False)
            kind = stat.S_IFMT(st.st_mode)
            if kind == stat.S_IFDIR:
                _feed(digest, "directory")
                _feed(digest, rel)
                totals["directory_count"] += 1
                visit(Path(entry.path), relative / entry.name)
            elif kind == stat.S_IFREG:
                _feed(digest, "file")
                _feed(digest, rel)
                _feed(digest, str(st.st_size))
                _feed(digest, cache.get(Path(entry.path)))
                totals["file_count"] += 1
                totals["size_bytes"] += st.st_size
            elif kind == stat.S_IFLNK:
                _feed(digest, "symlink")
                _feed(digest, rel)
                _feed(digest, os.readlink(entry.path))
                totals["symlink_count"] += 1
            else:
                raise ValueError(f"unsupported special file: {entry.path}")

    visit(Path(root).resolve(), Path())
    return {"sha256": digest.hexdigest(), **totals}


def fingerprint(path: Path, cache: HashCache | None = None) -> dict:
    """A file's {"kind": "file", "sha256", "size_bytes"} or a directory's digest."""
    cache = cache or HashCache()
    p = Path(path).resolve()
    if p.is_file():
        return {"kind": "file", "sha256": cache.get(p), "size_bytes": p.stat().st_size}
    if p.is_dir():
        return {"kind": "directory", **directory_fingerprint(p, cache)}
    raise FileNotFoundError(path)


def matches_record(path: Path, record: dict, cache: HashCache | None = None) -> bool:
    """Does `path` still hold the bytes a run's provenance `record` describes?"""
    got = fingerprint(path, cache)
    return got["kind"] == record.get("kind") and got["sha256"] == record.get("sha256")
