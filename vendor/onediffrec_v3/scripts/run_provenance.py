#!/usr/bin/env python3
"""Create and verify immutable run-provenance records.

The command is intentionally dependency-free so that training launchers can run
it before importing a model stack.  A run has two records:

* ``run.json`` describes the inputs and configuration present at start time.
* ``result.json`` links to the exact bytes of ``run.json`` and fingerprints the
  final artifacts.

Repeating ``start`` is safe for a resumed job: the immutable fields are checked
before a new attempt is appended.  Once ``result.json`` exists, ``run.json`` is
left unchanged so that the result link cannot be invalidated. ``verify
--path-map OLD_ROOT=NEW_ROOT`` checks an unchanged completed record after its
files have been copied to a different mount root.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import getpass
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

try:  # Linux is the deployment target; the fallback still preserves atomic IO.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX systems.
    fcntl = None  # type: ignore[assignment]


RUN_SCHEMA = "onediffrec.run-provenance/v1"
RESULT_SCHEMA = "onediffrec.run-result/v1"
CONTENT_FINGERPRINT_SCHEMA = "onediffrec.content-fingerprint/v1"
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]*$")
_GIT_TIMEOUT_SECONDS = 30


class ProvenanceError(RuntimeError):
    """An invalid record, changed resume input, or failed verification."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _feed_field(digest: "hashlib._Hash", value: str | bytes) -> None:
    data = value if isinstance(value, bytes) else os.fsencode(value)
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _directory_digest(root: Path) -> dict[str, Any]:
    """Hash a tree in bytewise relative-path order without following links.

    Each entry is framed with lengths and a type tag, so neither concatenation
    ambiguity nor filesystem iteration order can affect the digest. Empty
    directories and symlink targets are represented explicitly. Timestamps and
    permission bits are excluded to keep copies reproducible across machines.
    """

    digest = hashlib.sha256()
    _feed_field(digest, "onediffrec-directory-sha256-v1")
    totals = {
        "size_bytes": 0,
        "file_count": 0,
        "directory_count": 0,
        "symlink_count": 0,
    }

    def visit(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            raise ProvenanceError(f"cannot read directory {directory}: {exc}") from exc

        for entry in entries:
            rel = relative / entry.name
            rel_text = rel.as_posix()
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ProvenanceError(f"cannot stat {entry.path}: {exc}") from exc

            kind = stat.S_IFMT(entry_stat.st_mode)
            if kind == stat.S_IFDIR:
                _feed_field(digest, "directory")
                _feed_field(digest, rel_text)
                totals["directory_count"] += 1
                visit(Path(entry.path), rel)
            elif kind == stat.S_IFREG:
                content_sha256, size = _file_digest(Path(entry.path))
                _feed_field(digest, "file")
                _feed_field(digest, rel_text)
                _feed_field(digest, str(size))
                _feed_field(digest, content_sha256)
                totals["file_count"] += 1
                totals["size_bytes"] += size
            elif kind == stat.S_IFLNK:
                try:
                    target = os.readlink(entry.path)
                except OSError as exc:
                    raise ProvenanceError(f"cannot read symlink {entry.path}: {exc}") from exc
                _feed_field(digest, "symlink")
                _feed_field(digest, rel_text)
                _feed_field(digest, target)
                totals["symlink_count"] += 1
            else:
                raise ProvenanceError(
                    f"unsupported special file in directory fingerprint: {entry.path}"
                )

    visit(root, Path())
    return {"sha256": digest.hexdigest(), **totals}


def fingerprint_path(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a deterministic SHA-256 fingerprint and absolute path metadata."""

    supplied = Path(path).expanduser().absolute()
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProvenanceError(f"cannot resolve {supplied}: {exc}") from exc

    if resolved.is_file():
        sha256, size = _file_digest(resolved)
        details: dict[str, Any] = {
            "kind": "file",
            "sha256": sha256,
            "size_bytes": size,
        }
    elif resolved.is_dir():
        details = {"kind": "directory", **_directory_digest(resolved)}
    else:
        raise ProvenanceError(f"input is neither a regular file nor a directory: {resolved}")

    return {
        "path": str(supplied),
        "resolved_path": str(resolved),
        **details,
    }


def content_fingerprint_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a location-independent record for the bytes at ``path``.

    Unlike the path records embedded in ``run.json``, this compact record omits
    both the supplied and resolved filesystem locations. It can therefore be
    archived beside evaluation outputs and compared after the underlying file
    or directory has been moved to another machine.
    """

    return {
        "schema": CONTENT_FINGERPRINT_SCHEMA,
        "fingerprint": _fingerprint_payload(fingerprint_path(path)),
    }


def _parse_pairs(values: Sequence[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ProvenanceError(f"{option} expects KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        if not _KEY_RE.fullmatch(key):
            raise ProvenanceError(f"invalid key {key!r} for {option}")
        if key in parsed:
            raise ProvenanceError(f"duplicate key {key!r} for {option}")
        parsed[key] = value
    return dict(sorted(parsed.items()))


def _parse_paths(values: Sequence[str], option: str) -> dict[str, Path]:
    pairs = _parse_pairs(values, option)
    return {role: Path(path) for role, path in pairs.items()}


def _parse_path_maps(values: Sequence[str]) -> tuple[tuple[Path, Path], ...]:
    mappings: dict[Path, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ProvenanceError(f"--path-map expects OLD_ROOT=NEW_ROOT, got {raw!r}")
        old_raw, new_raw = raw.split("=", 1)
        if not old_raw or not new_raw:
            raise ProvenanceError(f"--path-map expects non-empty roots, got {raw!r}")
        old = Path(old_raw).expanduser().absolute()
        new = Path(new_raw).expanduser().absolute()
        if old in mappings:
            raise ProvenanceError(f"duplicate --path-map root: {old}")
        mappings[old] = new
    # Apply the most specific root first when mappings overlap.
    return tuple(sorted(mappings.items(), key=lambda pair: len(pair[0].parts), reverse=True))


def _parse_metrics(values: Sequence[str]) -> dict[str, Any]:
    pairs = _parse_pairs(values, "--metric")
    metrics: dict[str, Any] = {}
    for key, raw in pairs.items():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, (dict, list)):
            raise ProvenanceError(f"--metric {key} must be a scalar JSON value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ProvenanceError(f"--metric {key} must be finite")
        metrics[key] = value
    return metrics


def _fingerprint_mapping(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {role: fingerprint_path(path) for role, path in sorted(paths.items())}


def _run_command(command: Sequence[str], cwd: Path) -> tuple[bool, bytes]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, b""
    return result.returncode == 0, result.stdout


def _git_info(repo_root: Path) -> dict[str, Any]:
    root_ok, root_raw = _run_command(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo_root
    )
    if not root_ok:
        return {"available": False, "requested_root": str(repo_root.resolve())}

    root = Path(os.fsdecode(root_raw).strip()).resolve()
    head_ok, head_raw = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    branch_ok, branch_raw = _run_command(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], cwd=root
    )
    status_ok, status_raw = _run_command(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=root
    )
    diff_ok, diff_raw = _run_command(["git", "diff", "--binary", "HEAD", "--"], cwd=root)
    info: dict[str, Any] = {
        "available": bool(head_ok),
        "repository_root": str(root),
        "head": os.fsdecode(head_raw).strip() if head_ok else None,
        "branch": os.fsdecode(branch_raw).strip() if branch_ok else None,
    }
    if status_ok:
        info.update(
            {
                "dirty": bool(status_raw),
                "status_sha256": hashlib.sha256(status_raw).hexdigest(),
            }
        )
    if diff_ok:
        info["tracked_diff_sha256"] = hashlib.sha256(diff_raw).hexdigest()
    return info


_ENV_KEYS = (
    "CONDA_DEFAULT_ENV",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_CACHE",
    "VIRTUAL_ENV",
)

_SLURM_KEYS = (
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CLUSTER_NAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_NODELIST",
    "SLURM_NTASKS",
    "SLURM_PROCID",
    "SLURM_RESTART_COUNT",
    "SLURM_SUBMIT_DIR",
)


def _selected_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}


def _scheduler_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in _SLURM_KEYS if key in os.environ}


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _gpu_environment(repo_root: Path) -> dict[str, Any]:
    ok, raw = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
    )
    if not ok:
        return {"available": False}
    return {
        "available": True,
        "query": "index,name,uuid,memory.total_mb,driver_version",
        "devices": [line for line in os.fsdecode(raw).splitlines() if line.strip()],
    }


def _attempt(repo_root: Path) -> dict[str, Any]:
    return {
        "started_at_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "user": getpass.getuser(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "installed_packages": _installed_packages(),
        "gpu": _gpu_environment(repo_root),
        "platform": platform.platform(),
        "environment": _selected_environment(),
        "scheduler": _scheduler_environment(),
        "git": _git_info(repo_root),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            # Some network filesystems do not support fsync on directories.
            with contextlib.suppress(OSError):
                os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ProvenanceError(f"missing provenance record: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProvenanceError(f"invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ProvenanceError(f"cannot read {path}: {exc}") from exc


@contextlib.contextmanager
def _locked_run_dir(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run_provenance.lock"
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_snapshot_component(role: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", role).strip("._") or "source"
    suffix = hashlib.sha256(role.encode("utf-8")).hexdigest()[:8]
    return f"{readable}-{suffix}"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _snapshot_file(snapshot_dir: Path) -> Path:
    entries = list(snapshot_dir.iterdir())
    if len(entries) != 1 or not entries[0].is_file():
        raise ProvenanceError(
            f"existing file-source snapshot has unexpected contents: {snapshot_dir}"
        )
    return entries[0]


def _copy_source_snapshot(source: Path, destination: Path) -> None:
    resolved = source.expanduser().resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_fingerprint = fingerprint_path(resolved)
        snapshot_target = _snapshot_file(destination) if resolved.is_file() else destination
        snapshot_fingerprint = fingerprint_path(snapshot_target)
        if _fingerprint_payload(source_fingerprint) != _fingerprint_payload(snapshot_fingerprint):
            raise ProvenanceError(f"existing source snapshot differs from {resolved}: {destination}")
        return

    temp_path = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temp_path.exists():
        raise ProvenanceError(f"stale source-snapshot temporary path exists: {temp_path}")
    try:
        if resolved.is_dir():
            shutil.copytree(resolved, temp_path, symlinks=True)
        elif resolved.is_file():
            temp_path.mkdir()
            shutil.copy2(resolved, temp_path / resolved.name)
        else:
            raise ProvenanceError(f"cannot snapshot special file: {resolved}")
        os.replace(temp_path, destination)
    finally:
        if temp_path.is_dir():
            shutil.rmtree(temp_path, ignore_errors=True)


def _fingerprint_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in {"path", "resolved_path"}}


def _source_records(
    sources: Mapping[str, Path], run_dir: Path, snapshot_sources: bool
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    resolved_run_dir = run_dir.resolve()
    for role, source in sorted(sources.items()):
        source_record = fingerprint_path(source)
        if snapshot_sources:
            resolved_source = Path(source_record["resolved_path"])
            if resolved_source.is_dir() and _is_relative_to(resolved_run_dir, resolved_source):
                raise ProvenanceError(
                    f"run directory {resolved_run_dir} is inside source directory {resolved_source}"
                )
            snapshot_relative = Path("source_snapshot") / _safe_snapshot_component(role)
            snapshot_path = run_dir / snapshot_relative
            _copy_source_snapshot(resolved_source, snapshot_path)
            snapshot_record = fingerprint_path(snapshot_path)
            if resolved_source.is_file():
                # File snapshots are placed in a role directory to avoid name collisions.
                expected = _fingerprint_payload(source_record)
                actual = _fingerprint_payload(fingerprint_path(_snapshot_file(snapshot_path)))
            else:
                expected = _fingerprint_payload(source_record)
                actual = _fingerprint_payload(snapshot_record)
            if actual != expected:
                raise ProvenanceError(f"source snapshot verification failed for {resolved_source}")
            source_record["snapshot"] = {
                "path": snapshot_relative.as_posix(),
                **_fingerprint_payload(snapshot_record),
            }
        records[role] = source_record
    return records


def _first_difference(expected: Any, actual: Any, path: str) -> str | None:
    if type(expected) is not type(actual):
        return f"{path} type changed from {type(expected).__name__} to {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            added = sorted(actual_keys - expected_keys)
            return f"{path} keys changed (missing={missing}, added={added})"
        for key in sorted(expected):
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} length changed from {len(expected)} to {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path} changed from {expected!r} to {actual!r}"
    return None


def _assert_equal(expected: Any, actual: Any, path: str) -> None:
    difference = _first_difference(expected, actual, path)
    if difference:
        raise ProvenanceError(f"resume mismatch: {difference}")


def _validate_run_shape(run: Any) -> dict[str, Any]:
    if not isinstance(run, dict):
        raise ProvenanceError("run.json must contain a JSON object")
    if run.get("schema") != RUN_SCHEMA:
        raise ProvenanceError(f"unsupported run.json schema: {run.get('schema')!r}")
    for key in ("identity", "config", "inputs", "sources"):
        if not isinstance(run.get(key), dict):
            raise ProvenanceError(f"run.json field {key!r} must be an object")
    if not run["identity"]:
        raise ProvenanceError("run.json identity must not be empty")
    if not isinstance(run.get("attempts"), list) or not run["attempts"]:
        raise ProvenanceError("run.json attempts must be a non-empty array")
    expected_sha256 = _canonical_sha256(_run_immutable_payload(run))
    if run.get("immutable_sha256") != expected_sha256:
        raise ProvenanceError("run.json immutable identity/config fingerprint does not match")
    return run


def _run_immutable_payload(run: Mapping[str, Any]) -> dict[str, Any]:
    # Absolute mount paths are descriptive metadata, not experiment identity.
    # Content fingerprints therefore stay comparable after moving a run to a
    # different machine or storage root.
    inputs = {
        role: _fingerprint_payload(record)
        for role, record in sorted(run.get("inputs", {}).items())
    }
    sources = {}
    for role, record in sorted(run.get("sources", {}).items()):
        source = _fingerprint_payload(record)
        snapshot = source.get("snapshot")
        if isinstance(snapshot, dict):
            source["snapshot"] = {
                key: value for key, value in snapshot.items() if key != "path"
            }
        sources[role] = source
    return {
        "schema": run.get("schema"),
        "identity": run.get("identity"),
        "config": run.get("config"),
        "inputs": inputs,
        "sources": sources,
        "snapshot_sources": run.get("snapshot_sources"),
    }


def _validate_recorded_paths(
    records: Mapping[str, Any], label: str, run_dir: Path, *, snapshots: bool = False,
    path_maps: Sequence[tuple[Path, Path]] = (),
) -> None:
    for role, stored in sorted(records.items()):
        if not isinstance(stored, dict) or "path" not in stored:
            raise ProvenanceError(f"invalid {label} record for role {role!r}")
        stored_path = Path(stored["path"]).expanduser().absolute()
        current_path = stored_path
        for old_root, new_root in path_maps:
            try:
                relative = stored_path.relative_to(old_root)
            except ValueError:
                continue
            current_path = new_root / relative
            break
        current = fingerprint_path(current_path)
        expected_main = {key: value for key, value in stored.items() if key != "snapshot"}
        _assert_equal(
            _fingerprint_payload(expected_main),
            _fingerprint_payload(current),
            f"{label}.{role}",
        )
        snapshot = stored.get("snapshot")
        if snapshots and snapshot is not None:
            if not isinstance(snapshot, dict) or "path" not in snapshot:
                raise ProvenanceError(f"invalid source snapshot for role {role!r}")
            snapshot_path = run_dir / snapshot["path"]
            current_snapshot = fingerprint_path(snapshot_path)
            expected_snapshot = {
                "path": snapshot["path"],
                **_fingerprint_payload(current_snapshot),
            }
            _assert_equal(snapshot, expected_snapshot, f"{label}.{role}.snapshot")


def start_run(
    run_dir: Path,
    identity: Mapping[str, str],
    config: Mapping[str, str],
    inputs: Mapping[str, Path],
    sources: Mapping[str, Path],
    repo_root: Path,
    snapshot_sources: bool = False,
) -> Path:
    if not identity:
        raise ProvenanceError("at least one --identity KEY=VALUE is required")
    run_dir = run_dir.expanduser().absolute()
    run_path = run_dir / "run.json"
    result_path = run_dir / "result.json"

    with _locked_run_dir(run_dir):
        proposed_inputs = _fingerprint_mapping(inputs)
        proposed_sources = _source_records(sources, run_dir, snapshot_sources)
        current_attempt = _attempt(repo_root.expanduser().absolute())

        if run_path.exists():
            run = _validate_run_shape(_load_json(run_path))
            _assert_equal(dict(sorted(identity.items())), run["identity"], "identity")
            _assert_equal(dict(sorted(config.items())), run["config"], "config")
            proposed_immutable = _run_immutable_payload({
                "inputs": proposed_inputs,
                "sources": proposed_sources,
            })
            stored_immutable = _run_immutable_payload(run)
            _assert_equal(proposed_immutable["inputs"], stored_immutable["inputs"], "inputs")
            _assert_equal(proposed_immutable["sources"], stored_immutable["sources"], "sources")
            _assert_equal(snapshot_sources, run.get("snapshot_sources", False), "snapshot_sources")
            if result_path.exists():
                verify_run(run_dir)
                return run_path
            # Files may live under different mount points on a resumed machine.
            # Their locations are descriptive metadata rather than immutable run
            # identity, so retain the verified content hashes while refreshing
            # the paths used by subsequent verification.
            run["inputs"] = proposed_inputs
            run["sources"] = proposed_sources
            if _canonical_sha256(_run_immutable_payload(run)) != run["immutable_sha256"]:
                raise ProvenanceError("relocated paths changed immutable run identity")
            current_attempt["attempt_number"] = len(run["attempts"]) + 1
            run["attempts"].append(current_attempt)
            _atomic_write_json(run_path, run)
            return run_path

        current_attempt["attempt_number"] = 1
        run = {
            "schema": RUN_SCHEMA,
            "created_at_utc": current_attempt["started_at_utc"],
            "identity": dict(sorted(identity.items())),
            "config": dict(sorted(config.items())),
            "inputs": proposed_inputs,
            "sources": proposed_sources,
            "snapshot_sources": snapshot_sources,
            "initial_git": current_attempt["git"],
            "attempts": [current_attempt],
        }
        run["immutable_sha256"] = _canonical_sha256(_run_immutable_payload(run))
        _atomic_write_json(run_path, run)
    return run_path


def _result_core(
    run_path: Path,
    status: str,
    config: Mapping[str, str],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    run_sha256, run_size = _file_digest(run_path)
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "config": dict(sorted(config.items())),
        "metrics": dict(sorted(metrics.items())),
        "artifacts": _fingerprint_mapping(artifacts),
        "run": {
            "path": "run.json",
            "sha256": run_sha256,
            "size_bytes": run_size,
        },
    }
    result["immutable_sha256"] = _canonical_sha256(result)
    return result


def finalize_run(
    run_dir: Path,
    status: str,
    config: Mapping[str, str],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Path],
) -> Path:
    if not status.strip():
        raise ProvenanceError("--status must not be empty")
    run_dir = run_dir.expanduser().absolute()
    run_path = run_dir / "run.json"
    result_path = run_dir / "result.json"
    with _locked_run_dir(run_dir):
        run = _validate_run_shape(_load_json(run_path))
        # Detect mutable data or source files before binding a result to the
        # start record. This also catches a live checkout edited while a long
        # job was running.
        _validate_recorded_paths(run["inputs"], "inputs", run_dir)
        _validate_recorded_paths(run["sources"], "sources", run_dir, snapshots=True)
        proposed = _result_core(run_path, status, config, metrics, artifacts)
        if result_path.exists():
            existing = _load_json(result_path)
            existing_core = dict(existing)
            existing_core.pop("created_at_utc", None)
            _assert_equal(proposed, existing_core, "result")
            verify_run(run_dir)
            return result_path
        result = {**proposed, "created_at_utc": _utc_now()}
        _atomic_write_json(result_path, result)
    return result_path


def _validate_result(
    run_dir: Path, result: Any,
    path_maps: Sequence[tuple[Path, Path]] = (),
) -> None:
    if not isinstance(result, dict):
        raise ProvenanceError("result.json must contain a JSON object")
    if result.get("schema") != RESULT_SCHEMA:
        raise ProvenanceError(f"unsupported result.json schema: {result.get('schema')!r}")
    for key in ("config", "metrics", "artifacts", "run"):
        if not isinstance(result.get(key), dict):
            raise ProvenanceError(f"result.json field {key!r} must be an object")
    expected_result_sha256 = _canonical_sha256(
        {key: value for key, value in result.items() if key not in {"created_at_utc", "immutable_sha256"}}
    )
    if result.get("immutable_sha256") != expected_result_sha256:
        raise ProvenanceError("result.json immutable status/config/metric fingerprint does not match")
    run_link = result["run"]
    if run_link.get("path") != "run.json":
        raise ProvenanceError("result.json must link to run.json in the same run directory")
    current_run = fingerprint_path(run_dir / "run.json")
    expected_link = {
        "path": "run.json",
        "sha256": current_run["sha256"],
        "size_bytes": current_run["size_bytes"],
    }
    _assert_equal(run_link, expected_link, "result.run")
    _validate_recorded_paths(
        result["artifacts"], "artifacts", run_dir, path_maps=path_maps
    )


def verify_run(
    run_dir: Path, path_maps: Sequence[tuple[Path, Path]] = ()
) -> tuple[Path, bool]:
    run_dir = run_dir.expanduser().absolute()
    run_path = run_dir / "run.json"
    result_path = run_dir / "result.json"
    run = _validate_run_shape(_load_json(run_path))
    _validate_recorded_paths(run["inputs"], "inputs", run_dir, path_maps=path_maps)
    _validate_recorded_paths(
        run["sources"], "sources", run_dir, snapshots=True, path_maps=path_maps
    )
    has_result = result_path.exists()
    if has_result:
        _validate_result(run_dir, _load_json(result_path), path_maps=path_maps)
    return run_path, has_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture deterministic training-run provenance and artifact fingerprints."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="create run.json or verify a resumed run")
    start.add_argument("--run-dir", type=Path, required=True)
    start.add_argument(
        "--identity", action="append", default=[], metavar="KEY=VALUE",
        help="immutable run identity; repeat for task, paradigm, variant, and seed",
    )
    start.add_argument(
        "--config", action="append", default=[], metavar="KEY=VALUE",
        help="immutable resolved configuration entry (stored as a string)",
    )
    start.add_argument(
        "--input", action="append", default=[], metavar="ROLE=PATH",
        help="input file or directory to fingerprint; repeat as needed",
    )
    start.add_argument(
        "--source", "--source-file", dest="source", action="append", default=[],
        metavar="ROLE=PATH", help="source file or tree to fingerprint; repeat as needed",
    )
    start.add_argument(
        "--snapshot-sources", action="store_true",
        help="also copy source entries beneath RUN_DIR/source_snapshot",
    )
    start.add_argument(
        "--repo-root", type=Path, default=Path.cwd(),
        help="repository used for commit and dirty-tree metadata (default: cwd)",
    )

    finalize = subparsers.add_parser("finalize", help="write an immutable result.json")
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--status", required=True)
    finalize.add_argument(
        "--config", action="append", default=[], metavar="KEY=VALUE",
        help="evaluation/result configuration entry (stored as a string)",
    )
    finalize.add_argument(
        "--metric", action="append", default=[], metavar="KEY=VALUE",
        help="metric scalar; JSON numbers, booleans, and null are typed",
    )
    finalize.add_argument(
        "--artifact", action="append", default=[], metavar="ROLE=PATH",
        help="final artifact file or directory to fingerprint; repeat as needed",
    )

    verify = subparsers.add_parser("verify", help="re-hash inputs, sources, and artifacts")
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument(
        "--path-map", action="append", default=[], metavar="OLD_ROOT=NEW_ROOT",
        help="remap a stored absolute root after copying an archive; repeat as needed",
    )

    fingerprint = subparsers.add_parser(
        "fingerprint", help="write a stable, content-only fingerprint record"
    )
    fingerprint.add_argument("--path", type=Path, required=True)
    fingerprint.add_argument(
        "--output", type=Path,
        help="atomically write JSON here; omit to write the record to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "start":
            path = start_run(
                run_dir=args.run_dir,
                identity=_parse_pairs(args.identity, "--identity"),
                config=_parse_pairs(args.config, "--config"),
                inputs=_parse_paths(args.input, "--input"),
                sources=_parse_paths(args.source, "--source"),
                repo_root=args.repo_root,
                snapshot_sources=args.snapshot_sources,
            )
            print(path)
        elif args.command == "finalize":
            path = finalize_run(
                run_dir=args.run_dir,
                status=args.status,
                config=_parse_pairs(args.config, "--config"),
                metrics=_parse_metrics(args.metric),
                artifacts=_parse_paths(args.artifact, "--artifact"),
            )
            print(path)
        elif args.command == "verify":
            path, has_result = verify_run(
                args.run_dir, path_maps=_parse_path_maps(args.path_map)
            )
            suffix = " and result.json" if has_result else ""
            print(f"verified {path}{suffix}")
        elif args.command == "fingerprint":
            record = content_fingerprint_record(args.path)
            if args.output is None:
                sys.stdout.buffer.write(_json_bytes(record))
            else:
                output = args.output.expanduser().absolute()
                _atomic_write_json(output, record)
                print(output)
        else:  # pragma: no cover - argparse enforces this.
            parser.error(f"unsupported command: {args.command}")
    except ProvenanceError as exc:
        print(f"provenance error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
