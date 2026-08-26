"""Build the frozen substrate: a hash-verified, read-only copy of everything
the interpretability work depends on.

Runs exactly once. The guarantees it establishes:

  * Every byte is verified after the copy, not assumed. A silent truncation on
    a network filesystem is the failure mode this exists to catch.
  * Symlinks are never followed. The upstream run directories are symlink farms
    pointing back into the live code tree; copying through them would
    reintroduce the exact coupling the freeze exists to break.
  * The result is chmod a-w, so a later bug cannot overwrite the substrate.
"""

from __future__ import annotations

import glob as globmod
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from sidlens import paths
from sidlens.provenance import hashing
from sidlens.provenance.spec import SPEC, VENDOR_SPEC, Entry

WILDCARD = ("*", "?", "[")


def _glob_base(pattern: str) -> str:
    """Longest leading directory prefix of `pattern` containing no wildcard.

    Destination layout mirrors the source below this point, which is how job
    ids and run timestamps survive the copy -- the registry needs them to map a
    checkpoint back to its log.
    """
    parts = pattern.split("/")
    keep = []
    for p in parts:
        if any(w in p for w in WILDCARD):
            break
        keep.append(p)
    # A trailing concrete filename is not part of the base.
    if len(keep) == len(parts):
        keep = keep[:-1]
    return "/".join(keep)


def _root_for(entry: Entry) -> Path:
    return paths.UPSTREAM_REPO if entry.root == "repo" else paths.UPSTREAM_WORK


@dataclass
class Resolved:
    entry: Entry
    pairs: list[tuple[Path, Path]]  # (source, dest)


def resolve(entry: Entry) -> Resolved:
    root = _root_for(entry)
    pairs: list[tuple[Path, Path]] = []
    seen: dict[Path, Path] = {}
    for pattern in entry.globs:
        base = _glob_base(pattern)
        recursive = "**" in pattern
        for hit in sorted(globmod.glob(str(root / pattern), recursive=recursive)):
            src = Path(hit)
            if src.is_symlink() or not src.is_file():
                continue
            rel = src.relative_to(root / base) if base else src.relative_to(root)
            dst = paths.FROZEN / entry.dest / rel
            if dst in seen and seen[dst] != src:
                raise ValueError(
                    f"destination collision under {entry.dest}: "
                    f"{seen[dst]} and {src} both map to {dst}"
                )
            if dst not in seen:
                seen[dst] = src
                pairs.append((src, dst))
    return Resolved(entry, pairs)


def _copy_verified(src: Path, dst: Path, cache: hashing.HashCache) -> dict:
    """Copy one file and prove the destination matches the source."""
    src_digest = cache.get(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, 0o644)
    shutil.copy2(src, dst)
    dst_digest = hashing.sha256_file(dst)
    if dst_digest != src_digest:
        raise IOError(
            f"copy verification FAILED\n  src {src} {src_digest}\n  dst {dst} {dst_digest}"
        )
    st = dst.stat()
    return {"sha256": dst_digest, "size": st.st_size, "source": str(src)}


def freeze_data(dry_run: bool = False, only_kind: str | None = None) -> dict:
    cache = hashing.HashCache()
    sections: dict[str, dict] = {}
    problems: list[str] = []
    total_bytes = 0
    t0 = time.time()

    for entry in SPEC:
        if only_kind and entry.kind != only_kind:
            continue
        res = resolve(entry)
        n = len(res.pairs)
        if n < entry.min_files:
            msg = (f"{entry.dest}: found {n} files, expected >= {entry.min_files}"
                   f"  [globs: {', '.join(entry.globs)}]")
            if entry.required:
                problems.append(msg)
            else:
                print(f"  SKIP (optional) {msg}")
                continue
        size = sum(s.stat().st_size for s, _ in res.pairs)
        total_bytes += size
        print(f"  {entry.dest:42s} {n:5d} files  {size/1e6:9.1f} MB  [{entry.kind}]")
        if dry_run:
            continue
        files = {}
        for src, dst in res.pairs:
            rel = dst.relative_to(paths.FROZEN / entry.dest).as_posix()
            files[rel] = _copy_verified(src, dst, cache)
        sections[entry.dest] = {
            "kind": entry.kind,
            "note": entry.note,
            "root": entry.root,
            "globs": list(entry.globs),
            "n_files": n,
            "bytes": size,
            "files": files,
        }

    if problems:
        raise SystemExit(
            "freeze aborted -- source inventory does not match the spec:\n  "
            + "\n  ".join(problems)
        )
    print(f"\n  total {total_bytes/1e9:.2f} GB in {time.time()-t0:.0f}s")
    return sections


def freeze_vendor(dry_run: bool = False) -> dict:
    """Copy upstream code byte-for-byte into vendor/.

    Copy rather than submodule: every checkpoint on disk was produced by a dirty
    working tree that exists in no commit, so a commit pin cannot express the
    state we need to preserve.
    """
    cache = hashing.HashCache()
    out: dict[str, dict] = {}
    for name, patterns in VENDOR_SPEC.items():
        dest_root = paths.VENDOR / name
        files: dict[str, dict] = {}
        count = 0
        for pattern in patterns:
            recursive = "**" in pattern
            for hit in sorted(globmod.glob(str(paths.UPSTREAM_REPO / pattern),
                                           recursive=recursive)):
                src = Path(hit)
                if src.is_symlink() or not src.is_file():
                    continue
                if "__pycache__" in src.parts or src.suffix == ".pyc":
                    continue
                rel = src.relative_to(paths.UPSTREAM_REPO)
                # Strip the DiffGRM/ DiffGRM_new/ prefix so `genrec` sits at the
                # vendor root and absolute `from genrec...` imports resolve.
                parts = rel.parts
                if parts[0] in ("DiffGRM", "DiffGRM_new"):
                    rel = Path(*parts[1:])
                dst = dest_root / rel
                if dry_run:
                    count += 1
                    continue
                files[rel.as_posix()] = _copy_verified(src, dst, cache)
                count += 1
        print(f"  vendor/{name:14s} {count:5d} files")
        out[name] = {"n_files": count, "files": files}
    return out


def upstream_state() -> dict:
    """Record the dirty tree in full -- it is the real baseline, not the commit."""
    def run(*args) -> str:
        return subprocess.run(args, cwd=paths.UPSTREAM_REPO, capture_output=True,
                              text=True, check=False).stdout

    porcelain = run("git", "status", "--porcelain")
    dirty = [line[3:] for line in porcelain.splitlines() if line.strip()]
    cache = hashing.HashCache()
    dirty_hashes = {}
    for rel in dirty:
        p = paths.UPSTREAM_REPO / rel
        if p.is_file() and not p.is_symlink():
            dirty_hashes[rel] = cache.get(p)
    return {
        "repo": str(paths.UPSTREAM_REPO),
        "head": run("git", "rev-parse", "HEAD").strip(),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "dirty": True if dirty else False,
        "n_dirty_paths": len(dirty),
        "status_porcelain": porcelain,
        "tracked_diff_sha256": hashing.sha256_bytes(run("git", "diff").encode()),
        "dirty_file_hashes": dirty_hashes,
    }


def in_flight_jobs() -> list[dict]:
    """SLURM jobs running during the freeze.

    A job still writing into a source tree can be captured mid-write, so the
    manifest records what was live rather than pretending the snapshot was quiet.
    """
    r = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i|%j|%T|%M|%N"],
        capture_output=True, text=True, check=False)
    jobs = []
    for line in r.stdout.strip().splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        jobs.append({"jobid": f[0], "name": f[1], "state": f[2],
                     "elapsed": f[3], "node": f[4] if len(f) > 4 else ""})
    return jobs


def make_read_only(root: Path) -> int:
    n = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            os.chmod(p, os.stat(p).st_mode & 0o555)
            n += 1
    return n
