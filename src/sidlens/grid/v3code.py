"""The vendored OneDiffRec code the grid runs depend on (`vendor/onediffrec_v3`).

Why vendored, and why checked against each run
----------------------------------------------
The grid's prompts, decoders and metrics are defined by OneDiffRec code
(`scripts/evaluate_ar_joint.py`, `DIFF_GRM/joint_decode.py`, ...). Reproducing
a run's saved test lists, or its prompt token ids, means running *that* code,
not a re-implementation of it. The copy in `vendor/onediffrec_v3` is
byte-identical to OneDiffRec commit 7ae3512 and hash-checked by `verify`
(`manifests/vendor.onediffrec_v3.json`). Each run also carries the snapshot of
the code it actually executed (`<work>/sources/<sha>/`); `check_source` proves
the two agree before any experiment trusts the vendored copy for that run.

Traps
-----
1. **Generic package names.** OneDiffRec's modules import `metrics.*` and
   `scripts.*`. sidlens's own `scripts/` folder has no `__init__.py`, so it is
   a namespace-package candidate whenever the repo root is on `sys.path`.
   `module()` puts the vendor root first and asserts every imported module's
   file lies under it, so a stray `scripts` can never be picked up silently.
2. **`genrec` collisions.** `joint_decode.py` sits inside a `genrec` tree, and
   `sidlens.vendorpath` owns that package name for the DiffGRM model code. It is
   loaded by file path under a private module name instead.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

from sidlens import paths

NAME = "onediffrec_v3"
ROOT = paths.VENDOR / NAME
PACKAGES = ("scripts", "metrics")
_BY_FILE: dict[str, object] = {}


def manifest() -> dict:
    return json.loads((paths.MANIFESTS / f"vendor.{NAME}.json").read_text())


def _under_root(module) -> bool:
    root = ROOT.resolve()
    f = getattr(module, "__file__", None)
    if f is not None and not Path(f).resolve().is_relative_to(root):
        return False
    return all(Path(p).resolve().is_relative_to(root) for p in getattr(module, "__path__", []) or [])


def module(dotted: str):
    """Import `scripts.*` or `metrics.*` from the vendored tree, and only from it."""
    if dotted.split(".")[0] not in PACKAGES:
        raise ValueError(f"{dotted!r} is not a vendored OneDiffRec package ({PACKAGES})")
    for name in [n for n in sys.modules if n.split(".")[0] in PACKAGES]:
        if not _under_root(sys.modules[name]):
            del sys.modules[name]
    root = str(ROOT)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    mod = importlib.import_module(dotted)
    for name in [n for n in sys.modules if n.split(".")[0] in PACKAGES]:
        if not _under_root(sys.modules[name]):
            raise ImportError(f"{name} resolved outside {ROOT}: {getattr(sys.modules[name], '__file__', None)}")
    return mod


def by_file(relative: str):
    """Load one standalone vendored file (no package imports) under a private name."""
    if relative not in _BY_FILE:
        path = ROOT / relative
        name = "sidlens_v3_" + relative.replace("/", "_").removesuffix(".py")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _BY_FILE[relative] = mod
    return _BY_FILE[relative]


def ar_eval():
    """OneDiffRec's AR evaluator: prompt, catalog grammar, beam search, rescoring."""
    return module("scripts.evaluate_ar_joint")


def ar_data():
    """OneDiffRec's SFT dataset classes (the training prompt wording)."""
    return module("scripts.ar.data")


def joint_decode():
    """OneDiffRec's v3 DiffGRM decoder (`joint_catalog_v1`)."""
    return by_file("DiffGRM_new/genrec/models/DIFF_GRM/joint_decode.py")


def horizon_split():
    return by_file("DiffGRM_new/genrec/horizon_split.py")


def provenance():
    """OneDiffRec's run provenance (directory fingerprints, run records)."""
    return module("scripts.run_provenance")


def check_source(source_root: Path) -> dict:
    """Compare every vendored file with the run's own source snapshot.

    Returns {"ok", "checked", "mismatch": [...], "missing": [...]}. A run whose
    snapshot differs executed different code than the vendored copy; its prompt
    or decoder may differ, so callers refuse it rather than reuse ours.
    """
    source_root = Path(source_root)
    mismatch, missing = [], []
    files = manifest()["files"]
    for rel, rec in sorted(files.items()):
        p = source_root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            mismatch.append(rel)
    return {"ok": not mismatch and not missing, "checked": len(files),
            "mismatch": mismatch, "missing": missing, "source_root": str(source_root)}
