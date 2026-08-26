"""The vendor guard, against the namespace-package hazard it exists for.

`genrec` has no __init__.py, so Python merges every genrec/ on sys.path into one
namespace package. A __file__-based guard passes vacuously in that case (a
namespace package's __file__ is None) while frozen and unfrozen modules are
silently interleaved in the same import graph.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIVE_TREE = "/home/leo.rodrigues/onediffrec/OneDiffRec/DiffGRM"


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
