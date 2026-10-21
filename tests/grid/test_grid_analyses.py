"""Pure-logic pieces of the grid port: N1's designs, N4's copy rule, the V2a comparison,
position-keyed day tables, grid names, and the pipeline's cell runner. No data needed."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sidlens import grid
from sidlens.data import timestamps as T
from sidlens.grid import predictions as GP

REPO = Path(__file__).resolve().parents[2]


def _load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"), REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- N1 -------------------------------------------------------------------
def test_n1_design_conditions_have_their_declared_properties():
    design = _load("experiments/additions/n1_prefix_conditioning/run.py").design
    rng = np.random.default_rng(0)
    pool = np.unique(rng.integers(0, 6, size=(400, 4)), axis=0)
    ids = np.arange(len(pool))
    digit_codes = [list(range(6))] * 4
    target = (1, 2, 3, 4)
    hist = [(0, 0, 0, 0), (1, 2, 3, 5), (5, 5, 5, 5)]        # item 1 matches the target on 3 digits
    out = design("e", hist, target, pool, ids, {0}, digit_codes)
    ds = {r["d"]: r for r in out}
    assert set(ds) == {2, 3}
    for d, r in ds.items():
        h = r["h"]
        assert h[:d] == target[:d] and r["k"] == 1
        suf, none, pre = (r["repl"][c][0] for c in ("suffix", "none", "prefix"))
        assert suf[d - 1:d + 1] == h[d - 1:d + 1] and suf[:d - 1] != target[:d - 1]
        assert none[d - 1] != target[d - 1] and none[d] != h[d]
        assert pre[:d] == target[:d] and pre[d] != h[d]
        assert all(c not in set(hist) | {target} for c in (suf, none, pre))
    # another item supporting h's code under either reading excludes the design
    hist2 = hist + [(9, 9, 3, 5)]                            # same digit 2 as the target, same digit 3 as h
    assert 3 not in {r["d"] for r in design("e", hist2, target, pool, ids, {0}, digit_codes)}


# ---- N4 -------------------------------------------------------------------
def test_n4_copy_rule_ranks_by_lcp_then_recency_then_popularity():
    Ranker = _load("experiments/additions/n4_no_training_baselines/run.py").Ranker
    sids = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1), (2, 2, 2)]
    pop = Counter({(2, 2, 2): 9, (0, 1, 0): 5, (1, 1, 1): 3})
    R = Ranker(sids, pop, 3)
    lst = R.copy([(1, 0, 0), (0, 0, 0)])                     # most recent last
    assert lst[:2] == [(0, 0, 0), (1, 0, 0)]                 # LCP 3: most recent first
    assert lst[2] == (0, 0, 1)                               # LCP 2 from the most recent item
    assert lst[3] == (0, 1, 0)                               # LCP 1 (more popular before less)
    assert lst[-1] == (2, 2, 2) and len(lst) == len(sids)    # LCP 0 last, by popularity
    assert R.pop()[0] == (2, 2, 2)
    assert R.pop_r1((1, 0, 0))[:2] == [(1, 1, 1), (1, 0, 0)]


# ---- V2a comparison ------------------------------------------------------------
def _l(*pairs):
    return [{"sids": [s], "joint_logprob": v} for s, v in pairs]


def test_compare_accepts_exact_and_tied_reorders_only():
    saved = {"a": {"predictions": _l(("x", -1.0), ("y", -2.0))},
             "b": {"predictions": _l(("x", -1.0), ("y", -1.0))}}
    same = GP.compare({"a": _l(("x", -1.0), ("y", -2.0))}, saved, "ar")
    assert same["ok"] and same["identical"] == 1
    tie = GP.compare({"b": _l(("y", -1.0), ("x", -1.0))}, saved, "ar", max_tie_share=1.0)
    assert tie["ok"] and tie["tie_reordered"] == 1
    bad = GP.compare({"a": _l(("y", -1.0), ("x", -2.0))}, saved, "ar")
    assert not bad["ok"] and bad["failed"] == 1
    drift = GP.compare({"a": _l(("x", -1.01), ("y", -2.0))}, saved, "ar")
    assert not drift["ok"]
    assert GP.compare({"z": _l(("x", -1.0))}, saved, "ar", only_saved=True)["rows"] == 0


# ---- days by position -------------------------------------------------------------
def test_position_row_times_checks_items_and_flags_unsettled_days():
    D = T.DAY
    ev = T.EventTimes({"7": (10, 11, 12, 13)}, {"7": np.array([0, D, D, 3 * D])}, {}, frozenset({("7", 2)}))
    rows, hist = T.position_row_times([{"example_id": "e", "row": 0, "user_id": "u", "user": "7",
                                        "history_positions": [0, 1, 2], "target_positions": [3],
                                        "history_item_ids": [10, 11, 12], "target_item_ids": [13]}], ev)
    r = rows.iloc[0]
    assert (r.gap1_days, r.gap12_days, r.tie12, r.time_ambiguous) == (2, 0, True, True)
    assert hist.time_ambiguous.tolist() == [False, False, True]
    with pytest.raises(ValueError, match="differ from inter.json"):
        T.position_row_times([{"example_id": "e", "row": 0, "user_id": "u", "user": "7",
                               "history_positions": [0, 1], "target_positions": [2],
                               "history_item_ids": [10, 99], "target_item_ids": [12]}], ev)


# ---- names and the cell runner ------------------------------------------------------
def test_grid_names_parse_to_results_paths():
    n = "grid:grid-v1/office/next2item/ar/rqvae_4codebook_512__zcr-0123456789abcdef/seed-42@epoch-0.5"
    i = grid.cell_info(n)
    assert (i["quantizer"], i["depth"], i["width"], i["zcr"], i["task"], i["state"]) == \
        ("rqvae", 4, 512, True, "two-item", "epoch-0.5")
    assert grid.map_dir(n) == "office/two-item/ar/rqvae_4codebook_512__zcr/epoch-0.5"
    with pytest.raises(ValueError):
        grid.cell_info("grid:grid-v1/office/next2item/ar/notamap/seed-42@best")


def test_run_cell_skips_complete_and_moves_failed_aside(tmp_path, monkeypatch):
    from sidlens import paths
    from sidlens.grid import pipeline as PL
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(PL, "_source_hashes", lambda: {})
    calls = []

    def cmds(c):
        calls.append(c.out)
        return [[sys.executable, "-c", "import sys; sys.exit(int(open(sys.argv[1]).read()))", str(tmp_path / "rc")]]
    spec = PL.Spec("additions/fake", "any", commands=cmds, single=True)
    monkeypatch.setitem(PL.SPECS, spec.name, spec)
    monkeypatch.setattr(PL, "_ctx", lambda s, ck, pilot=False: PL.Ctx("", {}, "best", PL.cell_dir(s, ck), tmp_path / "s"))
    (tmp_path / "rc").write_text("3")
    assert PL.run_cell(spec, None) == 3
    out = PL.cell_dir(spec, None)
    assert (out / "status.txt").read_text().startswith("failed exit=3")
    (tmp_path / "rc").write_text("0")
    assert PL.run_cell(spec, None) == 0
    assert PL.is_complete(out) and len(list(out.parent.glob("fake.failed-*"))) == 1
    assert json.loads((out / "output.sha256.json").read_text())
    n = len(calls)
    assert PL.run_cell(spec, None) == 0 and len(calls) == n          # complete: skipped, no command run
