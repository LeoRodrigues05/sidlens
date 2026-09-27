"""Move a profile of $SIDLENS_WORK between clusters through a private HF dataset repo.

A bundle is a JSON manifest under `$SIDLENS_WORK/bundles/<bundle-id>.json`. It
lists every file a profile selects (WORK-relative path, sha256, size) and the
snapshot and git state it came from. The files themselves travel in a Hub
*dataset* repo that mirrors the WORK layout (`frozen/...`, `external/...`,
`derived/...`, `bundles/...`), and each bundle is pinned by a git tag equal to
its id. A pull downloads exactly the manifest's files at that tag into
`$SIDLENS_WORK` and then re-hashes them. The hash check, not the transport,
decides whether the copy is accepted.

Traps each guard exists for
---------------------------
* Publishing drifted bytes under a clean name. `create` hashes every selected
  frozen file against the provenance manifest and refuses on any mismatch, and
  `push` re-checks the bundle. A bundle therefore certifies bytes that verify
  already accepted, not whatever was on disk that day.
* A public repo. The substrate carries Amazon review text, user ids and
  unpublished checkpoints. `push` requires an existing repo, which the user
  creates, and refuses unless the Hub reports it private. It never creates or
  re-scopes a repo itself.
* Silent overwrite on arrival. `pull` compares every file already present
  locally with the manifest before it downloads anything. A file that differs
  aborts the pull and is listed, never replaced. Identical files are skipped.
* frozen/ is read-only by design. `pull` grants u+w on existing *directories*
  only for the duration of the download, restores their modes afterwards, and
  marks every downloaded frozen file read-only, as `sidlens freeze` does.
* Head drift on the Hub. Later pushes add files to the repo's main branch.
  Pulling at the bundle's tag, with the manifest as an allow-list, returns
  the bundle as it was pushed and nothing that arrived afterwards.
* Symlinks. `hashing.walk_files` neither follows nor yields them, so a
  derived tree that links into another tree is bundled without the link
  target. Copy the target in before bundling if it matters.

Side effects outside the listed files: `push` keeps its resumable upload state
in `cache/bundle-stage/<id>/.cache/`, and `pull` leaves the Hub client's
download metadata in `$SIDLENS_WORK/.cache/huggingface/`. Both can be deleted.

Where to run it: wherever the Hub is reachable. On the current cluster that is
the login node, because compute nodes have no outbound network. Hashing is
stat-cached (`hashing.HashCache`), so repeat checks of the 3 GB files are
cheap once warm.
"""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sidlens import paths
from sidlens.provenance import hashing, manifest as manifest_mod
from sidlens.provenance import profiles as profiles_mod

SCHEMA_VERSION = 1
BUNDLES = "bundles"
HUB_REPO_TYPE = "dataset"
# Upload threads when the caller does not say. The Hub client's own default is
# every core but two, which on a shared login node is a denial of service.
DEFAULT_UPLOAD_WORKERS = 4
# A bundle id is a file name under bundles/ and a git tag on the Hub.
RE_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


class BundleError(RuntimeError):
    """The bundle cannot be created, published or accepted as requested."""


# ------------------------------------------------------------------ helpers --
def _bundles_dir() -> Path:
    return paths.WORK / BUNDLES


def _manifest_rel(bundle_id: str) -> str:
    return f"{BUNDLES}/{_check_id(bundle_id)}.json"


def _check_id(bundle_id: str) -> str:
    if not RE_BUNDLE_ID.match(bundle_id) or ".." in bundle_id or bundle_id.endswith(".lock"):
        raise BundleError(f"bundle id {bundle_id!r} must match {RE_BUNDLE_ID.pattern} "
                          f"(it becomes a file name and a git tag)")
    return bundle_id


def _check_extra(tree: str) -> str:
    """A WORK-relative tree that may ride along; refuse anything that escapes.

    frozen/ is excluded because the provenance manifest already owns its
    hashes, and a second hash record for the same bytes could disagree.
    """
    p = PurePosixPath(tree.strip("/"))
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise BundleError(f"--extra must be a relative path under $SIDLENS_WORK: {tree!r}")
    if p.parts[0] in ("frozen", BUNDLES, "env", "venv", "cache", ".cache"):
        raise BundleError(f"--extra may not name {p.parts[0]}/: {tree!r}")
    if not (paths.WORK / p).exists():
        raise BundleError(f"--extra tree does not exist: {paths.WORK / p}")
    return p.as_posix()


def _hub_pattern(rel: str) -> str:
    """Exact-match fnmatch pattern for one path (the Hub filters with fnmatch)."""
    return rel.replace("[", "[[]").replace("*", "[*]").replace("?", "[?]")


def _git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(paths.REPO), *args],
                              capture_output=True, text=True, check=False).stdout.strip()
    status = run("status", "--porcelain")
    return {"head": run("rev-parse", "HEAD"),
            "dirty_paths": len(status.splitlines()) if status else 0,
            "remote": run("remote", "get-url", "origin")}


def _default_id(profile: str) -> str:
    return f"{profile}-{datetime.now().strftime('%Y%m%dT%H%M%S')}"


# --------------------------------------------------------------------- plan --
def plan(profile: str, extra: tuple[str, ...] = ()) -> dict:
    """Which files the profile selects, with expected hashes. Writes nothing.

    Frozen files take their sha256 from the provenance manifest. Extra trees
    are listed with `sha256=None`; `create` fills those in.
    """
    prof = profiles_mod.get(profile)
    man = manifest_mod.load()
    files: dict[str, dict] = {}
    excluded: list[str] = []
    for dest, section in man["data"].items():
        if not prof.selects_section(dest):
            continue
        for rel, rec in section["files"].items():
            key = f"frozen/{dest}/{rel}"
            if prof.excludes(dest, rel):
                excluded.append(key)
                continue
            files[key] = {"sha256": rec["sha256"], "size": rec["size"],
                          "origin": "frozen"}
    trees = list(dict.fromkeys(_check_extra(t) for t in (*prof.extra, *extra)))
    for tree in trees:
        for p in hashing.walk_files(paths.WORK / tree):
            key = p.relative_to(paths.WORK).as_posix()
            files[key] = {"sha256": None, "size": p.stat().st_size, "origin": "extra"}
    if not files:
        raise BundleError(f"profile {profile!r} with extra={list(extra)} selects no files")
    return {"profile": profile, "extra": trees, "snapshot_id": man["snapshot_id"],
            "excluded": {"globs": list(prof.frozen_exclude), "files": excluded},
            "files": dict(sorted(files.items()))}


def totals(files: dict[str, dict]) -> dict:
    out = {"files": len(files), "bytes": sum(r["size"] for r in files.values())}
    for origin in ("frozen", "extra"):
        sel = [r for r in files.values() if r["origin"] == origin]
        out[f"{origin}_files"] = len(sel)
        out[f"{origin}_bytes"] = sum(r["size"] for r in sel)
    return out


# ------------------------------------------------------------------- create --
def create(profile: str, extra: tuple[str, ...] = (), bundle_id: str | None = None) -> Path:
    """Hash and certify a profile, then write its bundle manifest. Never overwrites."""
    bundle_id = _check_id(bundle_id or _default_id(profile))
    out = _bundles_dir() / f"{bundle_id}.json"
    if out.exists():
        raise BundleError(f"bundle {bundle_id!r} already exists at {out}")
    p = plan(profile, extra)
    cache = hashing.HashCache()
    bad: list[str] = []
    for rel, rec in p["files"].items():
        f = paths.WORK / rel
        if not f.exists():
            bad.append(f"{rel} (missing)")
            continue
        if rec["origin"] == "frozen" and f.stat().st_size != rec["size"]:
            bad.append(f"{rel} (size differs from the provenance manifest)")
            continue
        digest = cache.get(f)
        if rec["origin"] == "frozen":
            if digest != rec["sha256"]:
                bad.append(f"{rel} (sha256 differs from the provenance manifest)")
        else:
            rec["sha256"], rec["size"] = digest, f.stat().st_size
    if bad:
        raise BundleError(
            f"{len(bad)} selected file(s) do not match the snapshot; run "
            f"`sidlens verify --strict` and fix the substrate before bundling:\n  "
            + "\n  ".join(bad[:20]))
    doc = {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_on": socket.gethostname(),
        "source_work": str(paths.WORK),
        "sidlens_git": _git_state(),
        **{k: p[k] for k in ("profile", "extra", "snapshot_id", "excluded")},
        "totals": totals(p["files"]),
        "files": p["files"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:      # "x": a concurrent create of the same id fails
        json.dump(doc, fh, indent=1)
    return out


def load(bundle: str | Path) -> dict:
    """A bundle manifest by id (looked up under WORK/bundles) or by path."""
    p = Path(bundle)
    if not p.suffix == ".json":
        p = _bundles_dir() / f"{_check_id(str(bundle))}.json"
    doc = json.loads(p.read_text())
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"{p}: schema {doc.get('schema_version')} != {SCHEMA_VERSION}")
    return doc


# -------------------------------------------------------------------- check --
def check(doc: dict, quick: bool = False) -> dict:
    """Every file present, right size, and (unless quick) right sha256."""
    cache = hashing.HashCache()
    missing, changed, checked = [], [], 0
    for rel, rec in doc["files"].items():
        f = paths.WORK / rel
        if not f.exists():
            missing.append(rel)
            continue
        if f.stat().st_size != rec["size"]:
            changed.append(rel)
            continue
        if quick:
            continue
        checked += 1
        if cache.get(f) != rec["sha256"]:
            changed.append(rel)
    return {"bundle_id": doc["bundle_id"], "ok": not missing and not changed,
            "n_files": len(doc["files"]), "checked": checked, "quick": quick,
            "missing": missing, "changed": changed}


def format_check(rep: dict) -> str:
    lines = [f"bundle {rep['bundle_id']}: {rep['n_files']} files, "
             f"{rep['checked']} hashed{' (quick: sizes only)' if rep['quick'] else ''}"]
    for rel in rep["missing"][:10]:
        lines.append(f"  missing {rel}")
    for rel in rep["changed"][:10]:
        lines.append(f"  changed {rel}")
    more = len(rep["missing"]) + len(rep["changed"]) - 20
    if more > 0:
        lines.append(f"  ... and {more} more")
    lines.append("BUNDLE OK" if rep["ok"] else
                 f"BUNDLE FAILED  missing={len(rep['missing'])} changed={len(rep['changed'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the Hub --
def _require_private_repo(api, repo_id: str, allow_public: bool) -> None:
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        info = api.repo_info(repo_id, repo_type=HUB_REPO_TYPE)
    except RepositoryNotFoundError:
        raise BundleError(
            f"dataset repo {repo_id!r} does not exist or this token cannot see it. "
            f"Create it as PRIVATE on the Hub first (`hf repo create {repo_id} "
            f"--repo-type dataset --private`); push never creates repos.") from None
    if not info.private and not allow_public:
        raise BundleError(f"{repo_id} is PUBLIC; refusing to upload the substrate there")


def push(bundle_id: str, repo_id: str, *, allow_public: bool = False,
         num_workers: int | None = None) -> str:
    """Upload a bundle's files + manifest, then tag the commit with the bundle id."""
    from huggingface_hub import HfApi

    doc = load(bundle_id)
    rep = check(doc)
    if not rep["ok"]:
        raise BundleError("local files no longer match the bundle; refusing to push\n"
                          + format_check(rep))
    api = HfApi()
    _require_private_repo(api, repo_id, allow_public)
    existing = {t.name for t in api.list_repo_refs(repo_id, repo_type=HUB_REPO_TYPE).tags}
    if bundle_id in existing:
        raise BundleError(f"tag {bundle_id!r} already exists on {repo_id}; bundle ids are unique")
    stage = _stage(doc)
    # upload_large_folder is resumable and chunks commits; its resume state
    # lives in <stage>/.cache/huggingface/, so an interrupted push is resumed
    # by running the same command again.
    api.upload_large_folder(repo_id=repo_id, repo_type=HUB_REPO_TYPE,
                            folder_path=str(stage),
                            num_workers=num_workers or DEFAULT_UPLOAD_WORKERS)
    api.create_tag(repo_id, repo_type=HUB_REPO_TYPE, tag=bundle_id,
                   tag_message=f"sidlens bundle {bundle_id} ({doc['profile']}, "
                               f"snapshot {doc['snapshot_id']}, "
                               f"git {doc['sidlens_git']['head'][:12]})")
    return bundle_id


def _stage(doc: dict) -> Path:
    """A symlink mirror of exactly the bundle's files, for upload.

    Uploading from $SIDLENS_WORK itself would make the Hub client glob the
    whole work root -- the 5.7 GB venv included -- and fnmatch every file
    against every allow-pattern. The mirror holds only the manifest's paths,
    in WORK layout, and is disposable (it lives under cache/). Links are
    rebuilt on every push, so a stale mirror cannot carry extra files.
    """
    bid = doc["bundle_id"]
    stage = paths.CACHE / "bundle-stage" / bid
    for rel in [*doc["files"], _manifest_rel(bid)]:
        link = stage / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(paths.WORK / rel)
    listed = {_manifest_rel(bid), *doc["files"]}
    extra = [p for p in stage.rglob("*")
             if (p.is_symlink() or p.is_file()) and ".cache" not in p.parts
             and p.relative_to(stage).as_posix() not in listed]
    for p in extra:
        p.unlink()
    return stage


@contextmanager
def _writable_dirs(root: Path):
    """Temporarily add u+w to read-only directories under `root`; always restore."""
    changed: list[tuple[Path, int]] = []
    try:
        if root.exists():
            for dirpath, _, _ in os.walk(root):
                d = Path(dirpath)
                mode = d.stat().st_mode
                if not mode & stat.S_IWUSR:
                    os.chmod(d, mode | stat.S_IWUSR)
                    changed.append((d, mode))
        yield
    finally:
        for d, mode in reversed(changed):
            os.chmod(d, stat.S_IMODE(mode))


def pull(bundle_id: str, repo_id: str, *, revision: str | None = None,
         dry_run: bool = False) -> dict:
    """Download one bundle into $SIDLENS_WORK at its tag, then hash-check it."""
    from huggingface_hub import hf_hub_download, snapshot_download

    rev = revision or _check_id(bundle_id)
    mpath = hf_hub_download(repo_id, _manifest_rel(bundle_id),
                            repo_type=HUB_REPO_TYPE, revision=rev)
    doc = json.loads(Path(mpath).read_text())
    if doc.get("bundle_id") != bundle_id or doc.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"{repo_id}@{rev}: manifest does not describe bundle {bundle_id!r}")

    cache = hashing.HashCache()
    todo, same, conflict = [], [], []
    for rel, rec in doc["files"].items():
        f = paths.WORK / rel
        if not f.exists():
            todo.append(rel)
        elif f.stat().st_size == rec["size"] and cache.get(f) == rec["sha256"]:
            same.append(rel)
        else:
            conflict.append(rel)
    summary = {"bundle_id": bundle_id, "repo": repo_id, "revision": rev,
               "to_download": len(todo), "already_present": len(same),
               "bytes_to_download": sum(doc["files"][r]["size"] for r in todo),
               "conflicts": conflict}
    if conflict:
        raise BundleError(
            f"{len(conflict)} local file(s) differ from bundle {bundle_id}; refusing to "
            f"overwrite them:\n  " + "\n  ".join(conflict[:20]))
    if dry_run:
        return summary

    if todo:
        with _writable_dirs(paths.FROZEN):
            snapshot_download(repo_id, repo_type=HUB_REPO_TYPE, revision=rev,
                              local_dir=str(paths.WORK),
                              allow_patterns=[_hub_pattern(r) for r in todo])
    local_manifest = _bundles_dir() / f"{bundle_id}.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    if local_manifest.exists():
        if json.loads(local_manifest.read_text()) != doc:
            raise BundleError(f"{local_manifest} exists with different content")
    else:
        local_manifest.write_text(Path(mpath).read_text())
    for rel in todo:
        if doc["files"][rel]["origin"] == "frozen":
            f = paths.WORK / rel
            os.chmod(f, f.stat().st_mode & 0o555)
    summary["check"] = check(doc)
    return summary
