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
    monkeypatch.setattr(PL, "_ctx", lambda s, ck, pilot=False, which="best":
                        PL.Ctx("", {}, "best", PL.cell_dir(s, ck), tmp_path / "s"))
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


# ---- the paper set ------------------------------------------------------------------
class _FakeRegistry:
    """Both catalogs, both paradigms and tasks; a core map at 4x256, a core map at 3x128 and a
    map outside the core (5x512), each native and ZCR; AR runs also keep an intermediate state."""
    def __init__(self):
        self.e = {}
        for ds in ("industrial", "office"):
            for pt in ("next-item", "next2item"):
                for par in ("ar", "diffgrm"):
                    for v in ("rqkmeans_4codebook_256", "rqkmeans_3codebook_128", "rqkmeans_5codebook_512"):
                        for z in ("", "__zcr-0123456789abcdef"):
                            for st in ("best", "epoch-0.5") if par == "ar" else ("best",):
                                ck = f"grid:grid-v1/{ds}/{pt}/{par}/{v}{z}/seed-42@{st}"
                                self.e[ck] = {"paradigm": par, "task": grid.TASK_OF[pt], "digits": int(v[9])}

    def names(self, states="best", **kw):
        return [k for k in self.e if states == "all" or k.endswith("@best")]

    def entry(self, ck):
        return self.e[ck]


def _maps(cells):
    return sorted({(grid.cell_info(ck)["map"].split("__zcr-")[0], grid.cell_info(ck)["zcr"]) for _, ck in cells})


def test_paper_set_runs_each_spec_on_its_declared_maps(monkeypatch):
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    by = {}
    for s, ck in PL.plan("paper"):
        by.setdefault(s.name, []).append((s, ck))
    assert all(ck is None or ck.endswith("@best") for cells in by.values() for _, ck in cells)
    every = [("rqkmeans_3codebook_128", False), ("rqkmeans_3codebook_128", True), ("rqkmeans_4codebook_256", False),
             ("rqkmeans_4codebook_256", True), ("rqkmeans_5codebook_512", False), ("rqkmeans_5codebook_512", True)]
    core = [m for m in every if m[0] != "rqkmeans_5codebook_512"]
    assert _maps(by["controlled/exp4_ar_copy_circuit"]) == every
    assert _maps(by["controlled/exp7_diffusion_history_use"]) == every
    assert _maps(by["controlled/exp6_ar_copy_in_decoding"]) == core
    assert _maps(by["representation/ar_chain"]) == [m for m in core if not m[1]]
    assert _maps(by["controlled/exp2_fixed_orders"]) == [m for m in core if m[0] == "rqkmeans_3codebook_128"]
    assert len(by["controlled/exp6_ar_copy_in_decoding"]) == 8               # 4 maps x 2 catalogs
    assert len(by["retrospective/exp6_time_structure"]) == 48                # every best run
    assert len(by["additions/n2_ar_prefix_match/induction"]) == 1
    assert {grid.cell_info(ck)["task"] for _, ck in by["controlled/exp5_ar_next_two_conditioning"]} == {"two-item"}
    best = [ck for s, ck in PL.plan("best") if s.name == "controlled/exp6_ar_copy_in_decoding"]
    assert len(best) == 12 and not any(s.name == "controlled/exp3_ar_history_patching" and "epoch" in ck
                                       for s, ck in PL.plan("best"))
    with pytest.raises(ValueError):
        PL.plan("everything")


def test_paper_set_trims_exp3_and_exp4_outside_the_core_only(monkeypatch):
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    broad = "grid:grid-v1/industrial/next-item/ar/rqkmeans_5codebook_512/seed-42@best"
    zcr = "grid:grid-v1/office/next-item/ar/rqkmeans_4codebook_256__zcr-0123456789abcdef/seed-42@best"
    e3, e4 = PL.SPECS["controlled/exp3_ar_history_patching"], PL.SPECS["controlled/exp4_ar_copy_circuit"]

    def args(spec, ck, which):
        return [" ".join(map(str, cmd[3:])) + " " for cmd in spec.commands(PL._ctx(spec, ck, which=which))]
    assert [("--split valid" in a, "--parts R,K " in a) for a in args(e4, broad, "paper")] == [(True, True)]
    assert len(args(e4, broad, "best")) == 2 and all("--parts R,K,H" in a for a in args(e4, broad, "best"))
    assert len(args(e4, zcr, "paper")) == 2 and all("--parts R,K,H" in a for a in args(e4, zcr, "paper"))
    assert "--parts A " in args(e3, broad, "paper")[0] and "--parts A,B" in args(e3, broad, "best")[0]
    assert "--parts A,B" in args(e3, zcr, "paper")[0]
    assert PL.paper_tier(broad) == "broad" and PL.paper_tier(zcr) == "core"
    assert PL.paper_tier(zcr.replace("__zcr-0123456789abcdef", "")) == "deep"


def test_a_broad_cell_counts_for_the_paper_set_and_is_superseded_by_best(tmp_path, monkeypatch):
    from sidlens import paths
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(paths, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(PL, "_source_hashes", lambda: {})
    seen = []

    def cmds(c):
        seen.append(c.full)
        return [[sys.executable, "-c", "pass"]]
    spec = PL.Spec("controlled/fake_tiered", "ar", commands=cmds, tiered=True)
    monkeypatch.setitem(PL.SPECS, spec.name, spec)
    ck = "grid:grid-v1/industrial/next-item/ar/rqkmeans_5codebook_512/seed-42@best"
    out = PL.cell_dir(spec, ck)
    assert PL.run_cell(spec, ck, which="paper") == 0 and seen == [False]
    assert json.loads((out / "cell.json").read_text())["scope"] == "broad"
    assert PL.is_complete(out) and not PL.is_complete(out, PL.needs_full("best", spec, ck))
    assert PL.run_cell(spec, ck, which="paper") == 0 and seen == [False]          # complete for the paper set
    assert PL.run_cell(spec, ck, which="best") == 0 and seen == [False, True]     # best reruns it in full
    assert len(list(out.parent.glob("best.superseded-*"))) == 1
    assert json.loads((out / "cell.json").read_text())["scope"] == "full"
    assert PL.run_cell(spec, ck, which="paper") == 0 and len(seen) == 2           # a full cell serves both


def test_case_study_is_the_native_case_maps_with_their_intermediate_states(monkeypatch):
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    cells = PL.plan("case")
    assert {grid.cell_info(ck)["map"] for _, ck in cells if ck} == {"rqkmeans_4codebook_256"}
    e3 = [ck for s, ck in cells if s.name == "controlled/exp3_ar_history_patching"]
    assert sorted(grid.split_name(ck)[1] for ck in e3) == ["best", "best", "epoch-0.5", "epoch-0.5"]
    assert all(grid.split_name(ck)[1] == "best" for s, ck in cells if s.name == "controlled/exp6_ar_copy_in_decoding")
    assert any(s.name == "representation/ar_chain" for s, _ in cells)
    assert set(PL.CASE_MAPS) == {f"{q}_{d}codebook_256" for q in ("rqkmeans", "rqvae", "MQ") for d in (3, 4)}


# ---- one job --------------------------------------------------------------------------
def test_slurm_durations_parse():
    from sidlens.grid import pipeline as PL
    assert PL._parse_slurm_time("1-02:03:04") == 93784
    assert PL._parse_slurm_time("02:03:04") == 7384 and PL._parse_slurm_time("03:04") == 184
    assert PL._parse_slurm_time("UNLIMITED") is None and PL._parse_slurm_time("") is None


def test_interleave_takes_one_cell_of_every_spec_first():
    from sidlens.grid import pipeline as PL
    a, b = PL.Spec("x/a", "any"), PL.Spec("x/b", "any")
    cells = [(a, 1), (a, 2), (a, 3), (b, 1)]
    assert [(s.name, c) for s, c in PL._interleave(cells)] == [("x/a", 1), ("x/b", 1), ("x/a", 2), ("x/a", 3)]


def _fake_launch(fail=None, sleep=0.0):
    """A launcher that plays a cell: status `running`, its start and end times, `sleep` seconds,
    then its final status (failed for `fail` = (spec, map)) and the GPU it saw."""
    import subprocess
    from sidlens.grid import pipeline as PL

    def launch(s, ck, which, env, log):
        out = PL.cell_dir(s, ck)
        bad = fail is not None and s.name == fail[0] and fail[1] in (ck or "")
        code = ("import sys, time, pathlib; d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True); "
                "(d / 'status.txt').write_text('running'); t0 = time.time(); time.sleep(float(sys.argv[4])); "
                "(d / 'gpu.txt').write_text(sys.argv[3]); (d / 'span.txt').write_text(f'{t0} {time.time()}'); "
                "(d / 'status.txt').write_text(('failed exit=1' if sys.argv[2] == '1' else 'complete exit=0 seconds=1'))")
        return subprocess.Popen([sys.executable, "-c", code, str(out), "1" if bad else "0",
                                 env.get("CUDA_VISIBLE_DEVICES", "unset"), str(sleep)], env=env,
                                start_new_session=True)
    return launch


def _batch_setup(tmp_path, monkeypatch, hours=0.01):
    from sidlens import paths
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    g = PL.Spec("t/gpu", "ar", hours=hours)
    c = PL.Spec("t/cpu", "ar", gpu=False, hours=hours)
    d = PL.Spec("t/child", "ar", gpu=False, hours=hours, after=("t/gpu",))
    for s in (g, c, d):
        monkeypatch.setitem(PL.SPECS, s.name, s)
    return PL, g, c, d


def test_batch_runs_every_cell_in_one_process_tree(tmp_path, monkeypatch):
    PL, g, c, d = _batch_setup(tmp_path, monkeypatch)
    bad = "industrial/next-item/ar/rqkmeans_5codebook_512/"
    given_up = PL.cell_dir(c, "grid:grid-v1/office/next-item/ar/rqkmeans_3codebook_128/seed-42@best")
    for k in (1, 2):                                           # two earlier failures: not rerun
        (given_up.parent / f"best.failed-{k}").mkdir(parents=True)
    res = PL.batch("best", ["t/gpu", "t/cpu", "t/child"], gpu_ids=["0", "1"], cpu_workers=1, poll=0.05,
                   launch=_fake_launch(fail=("t/gpu", bad)))
    assert res == {**res, "done": 11 + 11 + 11, "failed": 1, "blocked": 1, "gave_up": 1, "left": 0}
    for s, ck in PL.plan("best", ["t/gpu", "t/cpu"]):
        seen = PL.cell_dir(s, ck) / "gpu.txt"
        if seen.exists():
            assert seen.read_text() in (("0", "1") if s.gpu else ("",))   # CPU cells see no GPU
    blocked = PL.cell_dir(d, "grid:grid-v1/industrial/next-item/ar/rqkmeans_5codebook_512/seed-42@best")
    assert not blocked.exists()                                # its parent failed: never started
    prog = json.loads(next((tmp_path / "batch").glob("progress-best-*.json")).read_text())
    assert prog["final"] and prog["specs"]["t/gpu"]["failed"] == 1 and prog["gpu_hours_left"] == 0


def test_batch_starts_nothing_measured_to_outlast_the_job(tmp_path, monkeypatch):
    PL, g, c, d = _batch_setup(tmp_path, monkeypatch, hours=2)
    first = PL.cell_dir(g, "grid:grid-v1/industrial/next-item/ar/rqkmeans_3codebook_128/seed-42@best")
    first.mkdir(parents=True)
    (first / "status.txt").write_text("complete exit=0 seconds=7200 finished=x\n")   # t/gpu: 2 h measured
    res = PL.batch("best", ["t/gpu", "t/cpu"], gpu_ids=["0"], cpu_workers=1, seconds=3600, poll=0.05,
                   launch=_fake_launch())
    # t/gpu's 11 other cells would not fit; t/cpu has no measured time, so its upper bound
    # (hours=2) does not hold it back
    assert res["done"] == 12 and res["left"] == 11


def test_an_interrupted_cell_is_rerun_without_counting_as_a_failure(tmp_path, monkeypatch):
    from sidlens import paths
    from sidlens.grid import pipeline as PL
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setattr(PL, "_source_hashes", lambda: {})
    spec = PL.Spec("additions/fake_single", "any", commands=lambda c: [[sys.executable, "-c", "pass"]], single=True)
    monkeypatch.setitem(PL.SPECS, spec.name, spec)
    monkeypatch.setattr(PL, "_ctx", lambda s, ck, pilot=False, which="best":
                        PL.Ctx("", {}, "best", PL.cell_dir(s, ck), tmp_path / "s"))
    out = PL.cell_dir(spec, None)
    out.mkdir(parents=True)
    (out / "status.txt").write_text("interrupted finished=x\n")
    assert PL.run_cell(spec, None) == 0 and PL.is_complete(out)
    assert len(list(out.parent.glob("fake_single.interrupted-*"))) == 1 and PL._attempts(out) == 0


def test_run_queues_one_job_and_refuses_a_second(tmp_path, monkeypatch, capsys):
    from sidlens import paths
    from sidlens.grid import pipeline as PL
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    for k, v in {"BATCH_GPUS": "4", "GPU_TYPE": "h200", "SLURM_PARTITION": "gpu", "BATCH_TIME": "1-00:00:00"}.items():
        monkeypatch.setenv(k, v)
    PL.submit_batch("paper", dry=True)
    line = capsys.readouterr().out
    assert line.count("DRY sbatch ") == 1 and "--gres=gpu:h200:4" in line and "--nodes=1" in line
    assert "batch.sbatch paper 0" in line and "--array" not in line and "--partition=gpu" in line
    monkeypatch.setattr(PL, "_queued_jobs", lambda name: ["4242"] if name == "sidlens-quick" else [])
    with pytest.raises(RuntimeError, match="4242"):
        PL.submit_batch("paper")                          # a quick job is queued: no paper job beside it


def test_finish_requeues_while_progress_is_made_and_summarizes_at_the_end(monkeypatch):
    from sidlens.grid import pipeline as PL
    calls = []
    monkeypatch.setattr(PL, "submit_batch", lambda which, stages=None, dry=False, round_=0: calls.append(("q", round_)))
    monkeypatch.setattr(PL, "summarize", lambda: calls.append(("summary",)))
    monkeypatch.setattr(PL, "export", lambda: calls.append(("export",)) or "x.tar.gz")
    PL.finish_or_requeue("paper", None, {"left": 5, "done": 3}, 0)
    PL.finish_or_requeue("paper", None, {"left": 5, "done": 0}, 1)          # no progress: stop, do not loop
    PL.finish_or_requeue("paper", None, {"left": 5, "done": 3}, 9)          # round cap (BATCH_MAX_ROUNDS=10)
    PL.finish_or_requeue("paper", None, {"left": 0, "done": 3}, 2)
    assert calls == [("q", 1), ("summary",), ("export",)]


def test_attempts_count_the_last_failed_run_and_memory_sizes_parse(tmp_path):
    from sidlens.grid import pipeline as PL
    d = tmp_path / "best"
    assert PL._attempts(d) == 0
    d.mkdir()
    (d / "status.txt").write_text("failed exit=1")
    assert PL._attempts(d) == 1                                # not yet moved aside, still a failure
    (tmp_path / "best.failed-1").mkdir()
    (tmp_path / "best.interrupted-1").mkdir()                  # the job's end: not counted
    assert PL._attempts(d) == 2
    (d / "status.txt").write_text("interrupted finished=x")
    assert PL._attempts(d) == 1
    assert (PL._gb("64G"), PL._gb("512M"), PL._gb("1T"), PL._gb("2048")) == (64, 0.5, 1024, 2)


def test_batch_keeps_running_cells_inside_the_memory_budget(tmp_path, monkeypatch):
    PL, g, c, d = _batch_setup(tmp_path, monkeypatch)          # every fake spec declares 64G
    res = PL.batch("best", ["t/gpu"], gpu_ids=["0", "1", "2"], cpu_workers=0, poll=0.02, mem_gb=100,
                   launch=_fake_launch(sleep=0.2))
    assert res["done"] == 12
    spans = sorted(tuple(map(float, (PL.cell_dir(g, ck) / "span.txt").read_text().split()))
                   for _, ck in PL.plan("best", ["t/gpu"]))
    assert all(b[0] >= a[1] - 0.05 for a, b in zip(spans, spans[1:]))      # 90 GB: one 64G cell at a time


def test_sigterm_marks_running_cells_interrupted_and_leaves_them_to_run(tmp_path, monkeypatch):
    import os
    import signal
    import threading
    PL, g, c, d = _batch_setup(tmp_path, monkeypatch)
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    res = PL.batch("best", ["t/gpu"], gpu_ids=["0", "1"], cpu_workers=0, poll=0.05, mem_gb=1000,
                   launch=_fake_launch(sleep=60))
    assert res["signal"] and res["done"] == 0 and res["left"] == 12
    states = [PL._status(PL.cell_dir(g, ck)) for _, ck in PL.plan("best", ["t/gpu"])]
    assert sum(st.startswith("interrupted") for st in states) == 2 and states.count("") == 10
    assert signal.getsignal(signal.SIGTERM) is not None                 # the handler was put back


def test_sigterm_counts_cells_that_finished_before_it_as_done(tmp_path, monkeypatch):
    import os
    import signal
    import threading
    PL, g, c, d = _batch_setup(tmp_path, monkeypatch)
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    res = PL.batch("best", ["t/gpu"], gpu_ids=["0", "1"], cpu_workers=0, poll=30, mem_gb=1000,
                   launch=_fake_launch(sleep=0.1))                     # finish at once; reaped only by the stop
    assert res["signal"] and res["done"] == 2 and res["left"] == 10


def test_a_job_does_not_count_itself_when_it_requeues(monkeypatch):
    import subprocess
    from sidlens.grid import pipeline as PL
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "4242\n4243\n")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    assert PL._queued_jobs("sidlens-paper") == ["4243"]
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "4242\n")
    assert PL._queued_jobs("sidlens-paper") == []                    # only itself: the successor may queue


# ---- batched rescoring, the quick set, several nodes ------------------------------------
def test_intervention_protocol_rescores_a_rows_candidates_together():
    from sidlens.grid import decode_ar as DA
    proto = {"num_beams": 50, "top_k": 10, "batch_size": 1, "score_batch_size": 1}
    assert DA.intervention_protocol(proto)["score_batch_size"] == 50
    assert DA.intervention_protocol(proto, 1)["score_batch_size"] == 1
    assert DA.intervention_protocol(proto)["evaluator_score_batch_size"] == 1 and proto["score_batch_size"] == 1
    with pytest.raises(ValueError):
        DA.intervention_protocol(proto, -1)


def test_quick_set_is_a_broad_subset_of_the_paper_set(monkeypatch):
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    quick = PL.plan("quick")
    paper = {(s.name, ck) for s, ck in PL.plan("paper")}
    assert quick and all((s.name, ck) in paper for s, ck in quick)
    by = {}
    for s, ck in quick:
        by.setdefault(s.name, []).append(ck)
    assert "representation/ar_chain" not in by and "additions/n3_copy_dose" not in by
    assert len(by["controlled/exp4_ar_copy_circuit"]) == 12 and len(by["controlled/exp6_ar_copy_in_decoding"]) == 8
    assert len(by["controlled/exp9_ar_order_vs_time"]) == 4                  # native core maps only
    core = "grid:grid-v1/industrial/next-item/ar/rqkmeans_4codebook_256/seed-42@best"
    e4 = PL.SPECS["controlled/exp4_ar_copy_circuit"]
    cmds = [" ".join(map(str, c)) + " " for c in e4.commands(PL._ctx(e4, core, which="quick"))]
    assert len(cmds) == 1 and "--split valid" in cmds[0] and "--parts R,K " in cmds[0]   # broad even on core
    assert not PL.needs_full("quick", e4, core) and PL.needs_full("paper", e4, core)


def test_shares_keep_a_weight_state_together_and_balance_the_load():
    from sidlens.grid import pipeline as PL
    big, small = PL.Spec("x/big", "ar", hours=8), PL.Spec("x/small", "ar", hours=1)
    child = PL.Spec("x/child", "ar", gpu=False, hours=1, after=("x/big",))
    cells = [(big, f"ck{i}") for i in range(4)] + [(small, f"ck{i}") for i in range(4, 12)] + \
            [(child, f"ck{i}") for i in range(4)] + [(small, None)]
    est = lambda s: s.hours * 3600
    shares = PL.assign_shards(cells, 2, est)
    assert sorted(map(len, shares)) and sum(map(len, shares)) == len(cells)
    for ck in [f"ck{i}" for i in range(4)]:                                   # parent and child together
        assert len({i for i, sh in enumerate(shares) for s, c in sh if c == ck}) == 1
    loads = [sum(est(s) for s, _ in sh) for sh in shares]
    assert max(loads) - min(loads) <= 8 * 3600
    assert PL.assign_shards(cells, 2, est) == shares                        # deterministic


def test_coordinate_runs_one_step_per_node_and_counts_progress(tmp_path, monkeypatch):
    from sidlens import paths
    from sidlens.grid import pipeline as PL, registry
    monkeypatch.setattr(registry, "load", lambda: _FakeRegistry())
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    monkeypatch.setenv("BATCH_GPUS", "8")
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    g = PL.Spec("t/gpu", "ar", hours=1)
    monkeypatch.setitem(PL.SPECS, g.name, g)
    seen = {}

    def fake_srun(cmd, cwd=None):
        seen["cmd"] = cmd
        shares = json.loads((tmp_path / "batch" / "shares-best-777.json").read_text())
        for name, ck in shares[0]:                       # node 0 finishes its share, node 1 nothing
            d = PL.cell_dir(PL.SPECS[name], ck)
            d.mkdir(parents=True)
            (d / "status.txt").write_text("complete exit=0 seconds=60")
        return 0
    res = PL.coordinate("best", ["t/gpu"], 2, 0, run=fake_srun)
    cmd = " ".join(seen["cmd"])
    assert cmd.startswith("srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:8")
    assert "batch --set best --worker --shares" in cmd and "--stage t/gpu" in cmd
    assert res["done"] == 6 and res["left"] == 6


def test_batch_stops_at_the_deadline_like_on_sigterm(tmp_path, monkeypatch):
    PL, g, c, d = _batch_setup(tmp_path, monkeypatch)
    res = PL.batch("best", ["t/gpu"], gpu_ids=["0", "1"], cpu_workers=0, poll=0.05, mem_gb=1000, seconds=1.5,
                   launch=_fake_launch(sleep=60))
    assert res["signal"] and res["done"] == 0 and res["left"] == 12
    states = [PL._status(PL.cell_dir(g, ck)) for _, ck in PL.plan("best", ["t/gpu"])]
    assert sum(st.startswith("interrupted") for st in states) == 2
