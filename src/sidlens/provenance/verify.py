"""Re-hash the frozen substrate and the vendored code; report any drift.

Runs at the head of every SLURM job. If this is red, nothing downstream is
trustworthy, so it exits non-zero by default rather than warning.
"""

from __future__ import annotations

from pathlib import Path

from sidlens import paths
from sidlens.provenance import hashing, manifest as manifest_mod
from sidlens.provenance import profiles as profiles_mod


def verify_snapshot(snapshot_id: str | None = None, quick: bool = False,
                    profile: str | None = None) -> dict:
    """Check frozen/ and vendor/ against the manifest.

    `profile` limits the frozen check to that profile's files (see
    `provenance.profiles`). Vendor trees are always checked in full: they are
    in git, so they are never "not copied". A profiled report still names
    every section it skipped and counts every file it excluded, so the result
    cannot be mistaken for a check of the whole snapshot.
    """
    man = manifest_mod.load(snapshot_id)
    prof = profiles_mod.get(profile) if profile else None
    cache = hashing.HashCache()
    report: dict = {"snapshot_id": man["snapshot_id"], "sections": {}, "ok": True,
                    "profile": profile, "skipped_sections": [],
                    "excluded_files": 0}

    # --- vendor: one Merkle comparison per tree ------------------------------
    for name, expected_root in man["vendor"]["merkle_roots"].items():
        root = paths.VENDOR / name
        if not root.exists():
            report["sections"][f"vendor/{name}"] = {"ok": False, "error": "missing"}
            report["ok"] = False
            continue
        actual = hashing.hash_tree(root, exclude={"__pycache__"}, cache=cache)
        actual_root = hashing.merkle_root(actual)
        ok = actual_root == expected_root
        entry = {"ok": ok, "expected": expected_root, "actual": actual_root}
        if not ok:
            expected_files = man["vendor"]["trees"][name]["files"]
            entry["diff"] = hashing.diff_trees(expected_files, actual)
            report["ok"] = False
        report["sections"][f"vendor/{name}"] = entry

    # --- frozen data ---------------------------------------------------------
    checked_files: list = []
    for dest, section in man["data"].items():
        if prof is not None and not prof.selects_section(dest):
            report["skipped_sections"].append(dest)
            continue
        wanted = {rel: rec for rel, rec in section["files"].items()
                  if prof is None or not prof.excludes(dest, rel)}
        report["excluded_files"] += len(section["files"]) - len(wanted)
        root = paths.FROZEN / dest
        if not root.exists():
            report["sections"][dest] = {"ok": False, "error": "missing"}
            report["ok"] = False
            continue
        missing, changed, checked = [], [], 0
        for rel, rec in wanted.items():
            p = root / rel
            checked_files.append(p)
            if not p.exists():
                missing.append(rel)
                continue
            st = p.stat()
            if st.st_size != rec["size"]:
                changed.append(rel)
                continue
            if quick:
                continue
            checked += 1
            if cache.get(p) != rec["sha256"]:
                changed.append(rel)
        ok = not missing and not changed
        report["sections"][dest] = {
            "ok": ok, "kind": section["kind"], "n_files": len(wanted),
            "checked": checked, "missing": missing[:20], "changed": changed[:20],
            "n_missing": len(missing), "n_changed": len(changed),
        }
        if not ok:
            report["ok"] = False

    # --- writability: frozen/ must stay read-only ----------------------------
    # Unprofiled: every file under every section root, manifest or not.
    # Profiled: only the files this run was asked to check -- a partial copy
    # may hold a half-finished download of an excluded file, which is not this
    # run's call to fail on.
    if prof is None:
        candidates = (p for dest in man["data"] if (paths.FROZEN / dest).exists()
                      for p in hashing.walk_files(paths.FROZEN / dest))
    else:
        candidates = (p for p in checked_files if p.exists() and not p.is_symlink())
    writable = []
    for p in candidates:
        if p.stat().st_mode & 0o222:
            writable.append(str(p.relative_to(paths.FROZEN)))
            if len(writable) >= 20:
                break
    report["writable_files"] = writable
    if writable:
        report["ok"] = False
    return report


def format_report(report: dict, verbose: bool = False) -> str:
    lines = [f"snapshot {report['snapshot_id']}"]
    if report.get("profile"):
        lines.append(
            f"profile  {report['profile']}: {len(report['skipped_sections'])} "
            f"section(s) skipped, {report['excluded_files']} file(s) excluded "
            f"-- this is NOT a check of the whole snapshot")
    for name, sec in report["sections"].items():
        mark = "ok  " if sec.get("ok") else "FAIL"
        detail = ""
        if not sec.get("ok"):
            if sec.get("error"):
                detail = f"  {sec['error']}"
            else:
                detail = f"  missing={sec.get('n_missing',0)} changed={sec.get('n_changed',0)}"
                for rel in (sec.get("missing") or [])[:5]:
                    detail += f"\n         - missing {rel}"
                for rel in (sec.get("changed") or [])[:5]:
                    detail += f"\n         - changed {rel}"
                if sec.get("diff"):
                    d = sec["diff"]
                    detail += (f"\n         vendor diff: +{len(d['added'])} "
                               f"-{len(d['missing'])} ~{len(d['changed'])}")
                    for rel in (d["changed"] + d["added"] + d["missing"])[:5]:
                        detail += f"\n         - {rel}"
        elif verbose:
            detail = f"  {sec.get('n_files','')} files"
        lines.append(f"  [{mark}] {name}{detail}")
    if report.get("writable_files"):
        lines.append(f"  [FAIL] frozen/ is WRITABLE ({len(report['writable_files'])}+ files)")
        for rel in report["writable_files"][:5]:
            lines.append(f"         - {rel}")
    lines.append("")
    tag = f" (profile {report['profile']})" if report.get("profile") else ""
    lines.append(("VERIFY OK" if report["ok"] else "VERIFY FAILED") + tag)
    return "\n".join(lines)
