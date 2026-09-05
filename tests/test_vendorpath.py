"""The vendor guard, against the namespace-package hazard it exists for.

`genrec` has no __init__.py, so Python merges every genrec/ on sys.path into one
namespace package. A __file__-based guard passes vacuously in that case (a
namespace package's __file__ is None) while frozen and unfrozen modules are
silently interleaved in the same import graph.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The unfrozen upstream tree. Three cases below exist to prove the guard strips
# it out of `genrec.__path__`; they can only do that if it is actually on disk.
# It is currently absent -- the upstream checkout this project was carved out of
# is gone from this machine, which is also why `paths.UPSTREAM_REPO` no longer
# resolves. Without it those cases cannot construct the leak they test for, so
# they SKIP rather than fail: a failure would say "the guard is broken", and the
# true statement is "the guard is unverified here". The distinction matters,
# because the second one is the one that should make somebody uncomfortable.
LIVE_TREE = os.environ.get(
    "SIDLENS_LIVE_TREE", "/home/leo.rodrigues/onediffrec/OneDiffRec/DiffGRM")
needs_live_tree = pytest.mark.skipif(
    not Path(LIVE_TREE).is_dir(),
    reason=f"live upstream tree absent at {LIVE_TREE}; the vendor guard's "
           f"anti-leak behaviour is UNVERIFIED in this environment. Set "
           f"SIDLENS_LIVE_TREE to a real DiffGRM checkout to run these.")


def run_isolated(body: str, pythonpath: str) -> subprocess.CompletedProcess:
    """Each case needs a fresh interpreter: sys.modules state is the thing under test."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True,
        env={"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin",
             "HOME": str(Path.home())})


SRC = str(REPO / "src")


def test_genrec_is_a_namespace_package():
    """If this ever fails, the whole guard can be simplified -- check first."""
    assert not (REPO / "vendor/diffgrm/genrec/__init__.py").exists()


def test_clean_activation_resolves_only_vendor():
    r = run_isolated("""
        from sidlens import vendorpath
        vendorpath.activate('diffgrm')
        import genrec
        paths = list(genrec.__path__)
        assert len(paths) == 1, paths
        assert 'sidlens/vendor/diffgrm' in paths[0], paths
        print('OK')
    """, SRC)
    assert "OK" in r.stdout, r.stderr


@needs_live_tree
def test_live_tree_on_pythonpath_is_stripped():
    """The live tree must not join genrec.__path__ even when importable."""
    r = run_isolated("""
        from sidlens import vendorpath
        vendorpath.activate('diffgrm')
        import genrec
        paths = list(genrec.__path__)
        assert len(paths) == 1, f'namespace spans {len(paths)} trees: {paths}'
        assert 'OneDiffRec' not in paths[0], paths
        print('OK')
    """, f"{SRC}:{LIVE_TREE}")
    assert "OK" in r.stdout, r.stderr


@needs_live_tree
def test_pre_imported_live_genrec_is_recovered():
    """Importing the live tree first must not poison the process."""
    r = run_isolated("""
        import genrec
        assert 'OneDiffRec' in list(genrec.__path__)[0]
        from sidlens import vendorpath
        vendorpath.activate('diffgrm')
        import genrec as g2
        vendorpath.assert_frozen(g2)
        assert 'OneDiffRec' not in list(g2.__path__)[0], list(g2.__path__)
        print('OK')
    """, f"{SRC}:{LIVE_TREE}")
    assert "OK" in r.stdout, r.stderr


@needs_live_tree
def test_tampering_after_activation_raises():
    r = run_isolated("""
        import sys
        from sidlens import vendorpath
        vendorpath.activate('diffgrm')
        sys.path.insert(0, '%s')
        vendorpath.purge()
        import genrec
        try:
            vendorpath.assert_frozen(genrec)
            print('LEAK NOT CAUGHT')
        except vendorpath.VendorLeak:
            print('OK')
    """ % LIVE_TREE, SRC)
    assert "OK" in r.stdout, r.stdout + r.stderr


def test_two_vendors_cannot_coexist():
    r = run_isolated("""
        from sidlens import vendorpath
        vendorpath.activate('diffgrm')
        try:
            vendorpath.activate('diffgrm_new')
            print('GUARD MISSING')
        except vendorpath.VendorLeak:
            print('OK')
    """, SRC)
    assert "OK" in r.stdout, r.stdout + r.stderr
