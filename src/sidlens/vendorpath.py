"""Guarded activation of a vendored upstream tree.

The vendored DiffGRM code imports absolutely (`from genrec.models... import`),
and the live OneDiffRec tree has a `genrec` too. Worse, `genrec` has no
`__init__.py`, so it is an implicit NAMESPACE package: Python merges every
`genrec/` directory it finds on sys.path into one package whose `__path__` spans
all of them. With both trees importable you can get

    genrec.models.DIFF_GRM.model       <- frozen vendor copy
    genrec.datasets.AmazonReviews2014  <- live, mutable tree

inside a single import graph, with nothing to indicate it happened. Checking
`__file__` does not catch this -- a namespace package's `__file__` is None, so a
`__file__`-based guard passes vacuously while both trees are merged.

So `activate()` removes competing `genrec/` roots from sys.path, imports
`genrec` itself, and asserts that every entry of its `__path__` lives under the
vendor root.

The two vendors cannot coexist in one process; they both own the name `genrec`.
Cross-task comparisons run as separate processes writing into shared derived/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from sidlens import paths

Vendor = Literal["diffgrm", "diffgrm_new", "onediffrec"]
ROOT_PACKAGES = ("genrec",)
_ACTIVE: str | None = None


class VendorLeak(RuntimeError):
    """Code was resolved from outside the frozen vendor tree."""


def active() -> str | None:
    return _ACTIVE


def purge() -> list[str]:
    """Drop every vendored package from sys.modules.

    A namespace package's `__path__` is computed at first import and cached, so
    a stale entry survives any later sys.path edit. Purging forces recomputation.
    """
    dropped = [
        name for name in list(sys.modules)
        if name in ROOT_PACKAGES or any(name.startswith(p + ".") for p in ROOT_PACKAGES)
    ]
    for name in dropped:
        del sys.modules[name]
    return dropped


def _assert_paths_under(module, root: Path) -> None:
    root = root.resolve()
    entries = [Path(p).resolve() for p in getattr(module, "__path__", [])]
    stray = [p for p in entries if not p.is_relative_to(root)]
    if stray:
        raise VendorLeak(
            f"`{module.__name__}` resolves outside the frozen vendor tree.\n"
            f"  vendor root : {root}\n"
            f"  stray paths : {', '.join(str(p) for p in stray)}\n"
            f"`genrec` is a namespace package, so these directories are MERGED "
            f"into one package and unfrozen modules can be imported with no "
            f"error at all. Remove the stray entry from sys.path/PYTHONPATH.")
    if not entries and getattr(module, "__file__", None):
        got = Path(module.__file__).resolve()
        if not got.is_relative_to(root):
            raise VendorLeak(f"{module.__name__} loaded from {got}, outside {root}")


def _strip_competing_roots(root: Path) -> list[str]:
    keep, removed = [], []
    for entry in sys.path:
        try:
            competes = bool(entry) and (Path(entry) / "genrec").is_dir()
        except OSError:
            competes = False
        if competes and Path(entry).resolve() != root.resolve():
            removed.append(entry)
        else:
            keep.append(entry)
    sys.path[:] = keep
    return removed


def activate(which: Vendor, force: bool = False) -> Path:
    """Put a vendored tree at the front of sys.path and prove it is the only one."""
    global _ACTIVE
    root = paths.VENDOR / which
    if not root.is_dir():
        raise FileNotFoundError(f"vendor tree missing: {root}")

    if _ACTIVE is not None and _ACTIVE != which and not force:
        raise VendorLeak(
            f"vendor {_ACTIVE!r} is already active in this process; {which!r} "
            f"cannot be activated too -- both define `genrec`. Run them as "
            f"separate processes, or pass force=True to purge and re-import.")

    removed = _strip_competing_roots(root)
    p = str(root)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

    # Any genrec already in sys.modules carries a cached __path__ computed under
    # the old sys.path, so it must be dropped even if nothing was removed.
    purge()

    import importlib
    mod = importlib.import_module("genrec")
    _assert_paths_under(mod, root)

    _ACTIVE = which
    return root


def assert_frozen(module) -> None:
    """Confirm an imported module came from the active vendor tree."""
    if _ACTIVE is None:
        raise VendorLeak("no vendor activated; call vendorpath.activate() first")
    _assert_paths_under(module, paths.VENDOR / _ACTIVE)
    f = getattr(module, "__file__", None)
    if f is not None:
        got = Path(f).resolve()
        if not got.is_relative_to((paths.VENDOR / _ACTIVE).resolve()):
            raise VendorLeak(f"{module.__name__} loaded from {got}")
