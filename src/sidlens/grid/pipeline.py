"""The grid superscript: every sidlens experiment on every OneDiffRec grid run, as one Slurm job.

    python -m sidlens.grid.pipeline register --work-root <OneDiffRec WORK_ROOT> [--run-name grid-v1]
    python -m sidlens.grid.pipeline plan   --set S [--stage ...]
    python -m sidlens.grid.pipeline pilot  [--stage ...]
    python -m sidlens.grid.pipeline run    --set S [--stage ...] [--dry-run] [--arrays | --local]
    python -m sidlens.grid.pipeline status [--set S]
    python -m sidlens.grid.pipeline summarize
    python -m sidlens.grid.pipeline export
    python -m sidlens.grid.pipeline batch  --set S [--round N]          (inside the one job)
    python -m sidlens.grid.pipeline one    --spec X --ckpt C --set S    (one cell; started by batch)
    python -m sidlens.grid.pipeline cell   --task-file F --index I      (inside a Slurm array task)

`run` queues ONE job (`scripts/grid/batch.sbatch`, one node, `BATCH_GPUS` GPUs)
whose `batch` step runs every unfinished cell of the set: one cell per GPU at a
time, `BATCH_CPU_WORKERS` CPU cells beside them, each in its own process. When
the job's time runs out it queues its successor, so one job at a time is all
the cluster ever sees. `run --arrays` keeps the old one-array-per-spec mode.

A *cell* is one experiment spec on one grid weight state; it writes to
`results_remote/<group>/<experiment>/<dataset>/<task>/<paradigm>/<map>/<state>/`
(`grid.map_dir`). Sets:

    best   every experiment on every run, best state only (`run_best_state.sh`)
    full   best, plus the cheap set at every intermediate weight state
           (`run_full.sh`): AR exp3 Part A and exp4 Part R (test); DiffGRM exp7
    paper  best state only, sized to the paper's claims (`run_paper.sh`;
           `docs/plans/GRID_PAPER_SET.md`): each spec runs on every map, on the
           core maps (`PAPER_CORE_MAPS`, native and ZCR twin, both catalogs) or on
           the native core maps only (`Spec.paper`); outside the core, exp3 runs
           Part A only and exp4 runs R and K on the validation split only
    case   the optional case study (`run_case_study.sh`): every spec, in full, on
           the native `CASE_MAPS` (3x256 and 4x256 of each quantizer, both
           catalogs), plus the cheap set at their intermediate weight states

Traps, each with its guard
--------------------------
1. **Overwriting results.** A complete cell (status `complete`, validation.json
   present) is skipped; anything else found in its place is renamed
   `<state>.failed-<stamp>` and the cell reruns from scratch.
2. **A plan that changes under running jobs.** A submission freezes its cell
   list into a task file; array tasks read that file, never re-plan.
3. **Dependencies by hope.** A cell that needs another cell's output (retro
   exp7 needs exp6) names it; the array waits on the parent array
   (`afterany`) and the cell refuses to start unless the parent is complete.
4. **Scratch filling up.** Activation captures go to `paths.SCRATCH` per run
   and are deleted by the last command of the representation chain.
5. **Silent site defaults.** Partition, account, QoS, excluded nodes and GPU
   concurrency come from the environment (`SLURM_PARTITION`, `SLURM_ACCOUNT`,
   `SLURM_QOS`, `SLURM_EXCLUDE`, `MAX_GPUS`, `MAX_CPUS`, `CPU_PARTITION`,
   `CELLS_PER_TASK`); nothing about one cluster is hard-coded.
6. **A reduced cell passing for a full one.** A paper-set exp3 or exp4 cell
   outside the core runs fewer parts. Its `cell.json` says `scope: broad`, and
   the best and full sets do not count it as complete: it is renamed
   `<state>.superseded-<stamp>` and rerun with every part.
7. **Thousands of queued jobs.** One array task per cell put about 3,600 jobs
   in the coauthor's queue. `run` now queues one job, refuses to queue a
   second one for the same set, and the job requeues itself only after it ends.
8. **A time limit counted as a failure.** A cell cut off by the job's end is
   marked `interrupted`, moved aside as `.interrupted-<stamp>` and rerun; only
   real failures (`.failed-<stamp>`) count towards `--max-attempts`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sidlens import grid, paths

REPO = paths.REPO
PY = sys.executable
EXP = REPO / "experiments"
SETS = ("best", "full", "paper", "case")
# The paper set's core maps (GRID_PAPER_SET.md): the three quantizers at one size
# (4x256), plus the two maps of our archived matched AR/DiffGRM cells, so grid
# numbers can be set beside the draft's. Each with its ZCR twin, in both catalogs.
PAPER_CORE_MAPS = ("rqkmeans_4codebook_256", "rqvae_4codebook_256", "MQ_4codebook_256",
                   "rqkmeans_3codebook_128", "rqvae_4codebook_128")
# The case study: two maps per quantizer, native tables only, for what the paper set
# runs only on the core maps or not at all (the intermediate weight states).
CASE_MAPS = tuple(f"{q}_{d}codebook_256" for q in ("rqkmeans", "rqvae", "MQ") for d in (3, 4))


def _py(script: str, *args) -> list[str]:
    return [PY, "-u", str(EXP / script), *map(str, args)]


def _mod(module: str, *args) -> list[str]:
    return [PY, "-u", "-m", module, *map(str, args)]


@dataclass(frozen=True)
class Ctx:
    ckpt: str
    entry: dict
    state: str
    out: Path
    scratch: Path
    pilot: bool = False
    full: bool = True                           # False: a paper-set cell outside the core (trap 6)

    @property
    def best(self) -> bool:
        return self.state == grid.BEST

    def cell(self, spec_name: str) -> Path:
        """Another spec's cell on the same run (under pilot/ when this is a pilot)."""
        return cell_dir(SPECS[spec_name], self.ckpt, self.pilot)


@dataclass(frozen=True)
class Spec:
    name: str                                   # results path below results_remote/
    paradigm: str                               # ar | diffgrm | any
    tasks: tuple = ("next-item",)
    cheap: bool = False                         # also at intermediate states in the full set
    gpu: bool = True
    hours: float = 1.0                          # per cell, for --time and the plan's projection
    mem: str = "64G"
    depth3: bool = False
    after: tuple = ()                           # specs whose cell on the same run must be complete
    commands: Callable = None                   # Ctx -> list[list[str]]
    single: bool = False                        # one cell for the whole grid (no run), e.g. N2 (f)
    summary: Callable | None = None             # (cells, out) -> list[list[str]]
    paper: str | None = "all"                   # paper set: all | core | deep (native core) | None
    tiered: bool = False                        # commands depend on Ctx.full (exp3, exp4)


def _limit(c: Ctx) -> list[str]:
    return ["--limit", "64"] if c.pilot else []


# ---------------------------------------------------------------- commands --
def c_exp3(c):
    return [_py("controlled/exp3_ar_history_patching/run.py", "--ckpt", c.ckpt, "--template", "v3",
                "--parts", "A,B" if c.best and c.full else "A", "--out", c.out, *_limit(c))]


def c_exp4(c):
    """Best state: R, K and H on both splits. A paper cell outside the core: R and K
    on the validation split, exp4's declared primary split. Intermediate state: R on test."""
    def run(split, parts):
        return _py("controlled/exp4_ar_copy_circuit/run.py", "--ckpt", c.ckpt, "--split", split,
                   "--parts", parts, "--out", c.out / split, *_limit(c))
    if not c.best:
        return [run("test", "R")]
    if not c.full:
        return [run("valid", "R,K")]
    return [run("test", "R,K,H"), run("valid", "R,K,H")]


def c_exp5(c):
    return [_py("controlled/exp5_ar_next_two_conditioning/run.py", "--ckpt", c.ckpt, "--out", c.out, *_limit(c))]


def c_exp6(c):
    return [_py("controlled/exp6_ar_copy_in_decoding/run_grid.py", "--ckpt", c.ckpt, "--out", c.out, *_limit(c))]


def c_exp9(c):
    return [_py("controlled/exp9_ar_order_vs_time/run.py", "--ckpt", c.ckpt, "--split", "test", "--out", c.out,
                *_limit(c))]


def c_exp10(c):
    return [_py("controlled/exp10_ar_last_digit_sibling/run.py", "--ckpt", c.ckpt, "--split", s, "--out",
                c.out / s, *_limit(c)) for s in ("test", "valid")]


def c_rep_ar(c):
    """The AR representation chain (rep exp1-exp3) on one run, captures in scratch (trap 4)."""
    S, R = c.scratch, (paths.RESULTS / "pilot" if c.pilot else paths.RESULTS) / "representation"
    md = grid.map_dir(c.ckpt)
    r1, r2, r3 = R / "exp1_ar_digit_decoding" / md, R / "exp2_ar_sae" / md, R / "exp3_ar_information_map" / md
    cap = "scripts/activations/ar_capture.py"
    lim = ["--limit", "64"] if c.pilot else []
    sae = ["--epochs", "1"] if c.pilot else []
    stamp = time.strftime("%Y%m%dT%H%M%S")
    aside = " ".join(f"[ -e '{d}' ] && mv '{d}' '{d}.failed-{stamp}';" for d in (r1, r2, r3)) + f" rm -rf '{S}';"
    return [
        ["bash", "-c", aside + " true"],          # a rerun: the chain is incomplete, so these are a failed attempt's
        [PY, "-u", str(REPO / cap), "--ckpt", c.ckpt, "--template", "v3", "--split", "test", "--layers", "all,norm",
         "--out", str(S / "test" / "capture"), *lim],
        _py("representation/exp1_ar_digit_decoding/run.py", "--capture", S / "test", "--out", r1),
        [PY, "-u", str(REPO / cap), "--ckpt", c.ckpt, "--template", "v3", "--split", "train",
         "--layers", "12,16,20,24", "--out", str(S / "train" / "capture"), *lim],
        _py("representation/exp2_ar_sae/train.py", "--ckpt", c.ckpt, "--train-capture", S / "train",
            "--test-capture", S / "test", "--out", r2 / "train", *sae),
        _py("representation/exp2_ar_sae/analyze.py", "--ckpt", c.ckpt, "--test-capture", S / "test",
            "--sae-run", r2 / "train", "--out", r2 / "analyze", *(["--smoke"] if c.pilot else [])),
        _py("representation/exp2_ar_sae/causal.py", "--ckpt", c.ckpt, "--train-capture", S / "train",
            "--sae-run", r2 / "train", "--selection", r2 / "analyze", "--out", r2 / "causal", *lim),
        _py("representation/exp2_ar_sae/amend_select_within_d.py", "--ckpt", c.ckpt, "--sae-run", r2 / "train",
            "--analysis", r2 / "analyze", "--out", r2 / "amend_a1", *(["--smoke"] if c.pilot else [])),
        _py("representation/exp2_ar_sae/causal.py", "--ckpt", c.ckpt, "--train-capture", S / "train",
            "--sae-run", r2 / "train", "--selection", r2 / "amend_a1", "--parts", "CM", "--out", r2 / "causal_a1",
            *lim),
        ["rm", "-rf", str(S / "test"), str(S / "train")],
        [PY, "-u", str(REPO / cap), "--ckpt", c.ckpt, "--template", "v3", "--split", "test", "--layers", "all,norm",
         "--target", "self-greedy", "--out", str(S / "greedy-test"), *lim],
        [PY, "-u", str(REPO / cap), "--ckpt", c.ckpt, "--template", "v3", "--split", "train", "--layers", "all,norm",
         "--target", "self-greedy", "--out", str(S / "greedy-train"), *(["--limit", "600"] if c.pilot else [])],
        _py("representation/exp3_ar_information_map/train_sae.py", "--ckpt", c.ckpt, "--train-capture",
            S / "greedy-train", "--test-capture", S / "greedy-test", "--out", r3 / "sae", *sae),
        _py("representation/exp3_ar_information_map/census.py", "--ckpt", c.ckpt, "--test-capture",
            S / "greedy-test", "--sae-run", r3 / "sae", "--out", r3 / "census-1" / "result"),
        _py("representation/exp3_ar_information_map/lens.py", "--ckpt", c.ckpt, "--train-capture",
            S / "greedy-train", "--test-capture", S / "greedy-test", "--out", r3 / "lens"),
        ["bash", "-c", f"mkdir -p '{r3 / 'capture-test'}' && cp '{S / 'greedy-test'}'/self_greedy.csv "
                       f"'{S / 'greedy-test'}'/validation.json '{S / 'greedy-test'}'/digit_scores.csv "
                       f"'{r3 / 'capture-test'}'/"],
        ["rm", "-rf", str(S)],
    ]


def c_diff(script: str, *extra, cpu=False):
    def f(c):
        return [_py(script, "--ckpt", c.ckpt, "--out", c.out, *(["--device", "cpu"] if cpu else []), *extra,
                    *_limit(c))]
    return f


def c_exp1(c):
    return [_py("controlled/exp1_matched_beam/run.py", "--ckpt", c.ckpt, "--batch-size", "8", "--out", c.out,
                *(["--limit", "64"] if c.pilot else []))]


def c_exp2(c):
    return [_py("controlled/exp2_fixed_orders/run.py", "--ckpt", c.ckpt, "--batch-size", "8", "--out", c.out,
                *(["--limit", "64"] if c.pilot else []))]


def c_saved(experiment: str):
    def f(c):
        return [_mod("sidlens.grid.saved_lists", "--ckpt", c.ckpt, "--experiment", experiment, "--out", c.out)]
    return f


def c_retro7(c):
    return [_mod("sidlens.grid.saved_lists", "--ckpt", c.ckpt, "--experiment", "exp7",
                 "--exp6-cell", c.cell("controlled/exp6_ar_copy_in_decoding"), "--out", c.out)]


def c_n2f(c):
    base = os.environ.get("QWEN_BASE_MODEL") or _base_model()
    return [_py("additions/n2_ar_prefix_match/induction.py", "--base-model", base, "--out", c.out)]


def _base_model() -> str:
    from sidlens.grid import registry
    reg = registry.load()
    for name in reg.names(paradigm="ar", states="best"):
        cell = json.loads((Path(reg.entry(name)["run_dir"]) / "cell.json").read_text())
        if cell.get("model_dir"):
            return cell["model_dir"]
    raise RuntimeError("no AR run records its base model (cell.json model_dir); set QWEN_BASE_MODEL")


# ---------------------------------------------------------------- summaries --
def s_cells(script: str, *extra, sub: str | None = None):
    def f(cells, out):
        cs = [str(c / sub) if sub else str(c) for c in cells]
        return [_py(script, "--cells", *cs, "--out", out, *extra)]
    return f


def s_collect(cells, out):
    return [_mod("sidlens.grid.collect", "--cells", *map(str, cells), "--out", out)]


def s_exp4(cells, out):
    """exp4's two splits, summarized apart (its declared primaries are on the validation split)."""
    return [_py("controlled/exp4_ar_copy_circuit/summarize.py", "--cells", *[str(c / sp) for c in cells
                                                                             if (c / sp).is_dir()],
                "--out", out / sp) for sp in ("valid", "test") if any((c / sp).is_dir() for c in cells)]


def s_ar_chain(cells, out):
    """Representation exp1-exp3 of every finished chain: exp1 over all runs, exp2 and exp3 per run
    (their summaries take one model's stage folders)."""
    R = paths.RESULTS / "representation"
    rel = [c.relative_to(paths.RESULTS / "representation" / "ar_chain") for c in cells]
    cks = [json.loads((c / "cell.json").read_text())["checkpoint"] for c in cells]
    cmds = [_py("representation/exp1_ar_digit_decoding/summarize.py", "--cells",
                *[str(R / "exp1_ar_digit_decoding" / m) for m in rel], "--out", out / "exp1")]
    for i, (m, ck) in enumerate(zip(rel, cks)):
        e2 = R / "exp2_ar_sae" / m
        cmds.append(_py("representation/exp2_ar_sae/summarize.py", "--train", e2 / "train", "--analysis",
                        e2 / "analyze", "--causal", e2 / "causal", "--cell-names", ".", "--out", out / "exp2" / f"{i:03d}"))
        cmds.append(_py("representation/exp3_ar_information_map/summarize.py", "--cell-dir",
                        R / "exp3_ar_information_map" / m, "--ckpt", ck, "--plain",
                        paths.RESULTS / "controlled" / "exp6_ar_copy_in_decoding" / m / "predictions_B_plain.parquet",
                        "--out", out / "exp3" / f"{i:03d}"))
    return cmds


def s_per_cell(builder):
    def f(cells, out):
        return [cmd for i, c in enumerate(cells) for cmd in builder(c, out / f"{i:03d}")]
    return f


SPEC_LIST = [
    # ---- AR, controlled
    Spec("controlled/exp3_ar_history_patching", "ar", cheap=True, hours=2, commands=c_exp3, tiered=True,
         summary=s_cells("controlled/exp3_ar_history_patching/summarize.py")),
    Spec("controlled/exp4_ar_copy_circuit", "ar", cheap=True, hours=4, commands=c_exp4, tiered=True,
         summary=s_exp4),
    Spec("controlled/exp5_ar_next_two_conditioning", "ar", tasks=("two-item",), hours=2, commands=c_exp5,
         paper="core",
         summary=s_per_cell(lambda c, o: [_py("controlled/exp5_ar_next_two_conditioning/summarize.py",
                                              "--cell", c, "--out", o)])),
    Spec("controlled/exp6_ar_copy_in_decoding", "ar", hours=8, commands=c_exp6, paper="core",
         summary=s_cells("controlled/exp6_ar_copy_in_decoding/summarize.py")),
    Spec("controlled/exp9_ar_order_vs_time", "ar", hours=4, commands=c_exp9, paper="core",
         summary=s_cells("controlled/exp9_ar_order_vs_time/summarize.py")),
    Spec("controlled/exp10_ar_last_digit_sibling", "ar", hours=1, commands=c_exp10,
         summary=lambda cells, out: [_py("controlled/exp10_ar_last_digit_sibling/summarize.py", "--cells",
                                         *[str(c / s) for c in cells for s in ("test", "valid")], "--out", out)]),
    # ---- AR, representation (one chain per run; outputs in the three experiment folders)
    Spec("representation/ar_chain", "ar", hours=12, mem="128G", commands=c_rep_ar, paper="deep",
         summary=s_ar_chain),
    # ---- DiffGRM
    Spec("controlled/exp1_matched_beam", "diffgrm", hours=2, commands=c_exp1, paper="core",
         summary=s_cells("controlled/exp1_matched_beam/summarize_grid.py")),
    Spec("controlled/exp2_fixed_orders", "diffgrm", depth3=True, hours=1, commands=c_exp2, paper="core",
         summary=s_cells("controlled/exp1_matched_beam/summarize_grid.py")),
    Spec("controlled/exp7_diffusion_history_use", "diffgrm", cheap=True, gpu=False, hours=2,
         commands=c_diff("controlled/exp7_diffusion_history_use/run.py", cpu=True),
         summary=s_cells("controlled/exp7_diffusion_history_use/summarize.py")),
    Spec("controlled/exp8_diffusion_copy_route", "diffgrm", gpu=False, hours=2,
         commands=c_diff("controlled/exp8_diffusion_copy_route/run.py", cpu=True),
         summary=s_cells("controlled/exp8_diffusion_copy_route/summarize.py")),
    Spec("controlled/exp12_diffusion_activation_patching", "diffgrm", gpu=False, hours=4,
         commands=c_diff("controlled/exp12_diffusion_activation_patching/run.py", cpu=True),
         summary=s_cells("controlled/exp12_diffusion_activation_patching/summarize.py")),
    Spec("controlled/exp13_diffusion_step_switch", "diffgrm", hours=2,
         commands=c_diff("controlled/exp13_diffusion_step_switch/run.py"),
         summary=s_cells("controlled/exp13_diffusion_step_switch/summarize.py")),
    Spec("controlled/exp14_diffusion_decode_perturbations", "diffgrm", hours=2,
         commands=c_diff("controlled/exp14_diffusion_decode_perturbations/run.py"),
         summary=s_cells("controlled/exp14_diffusion_decode_perturbations/summarize.py")),
    Spec("representation/exp4_diffusion_reveal_trace", "diffgrm", hours=2, paper="core",
         commands=c_diff("representation/exp4_diffusion_reveal_trace/run.py"),
         summary=s_cells("representation/exp4_diffusion_reveal_trace/summarize.py", "--skip-failed")),
    Spec("representation/exp5_diffusion_encoder_features", "diffgrm", hours=2,
         commands=c_diff("representation/exp5_diffusion_encoder_features/run.py"),
         summary=s_cells("representation/exp5_diffusion_encoder_features/summarize.py")),
    # ---- saved lists (CPU, every run)
    Spec("retrospective/exp2_prefix", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5, mem="16G",
         commands=c_saved("exp2"), summary=s_collect),
    Spec("retrospective/exp3_collisions", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5, mem="16G",
         commands=c_saved("exp3"), summary=s_collect),
    Spec("retrospective/exp4_next_two", "any", tasks=("two-item",), gpu=False, hours=0.5, mem="16G",
         commands=c_saved("exp4"), summary=s_collect),
    Spec("retrospective/exp6_time_structure", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5, mem="16G",
         commands=c_saved("all"), summary=s_collect),
    Spec("retrospective/exp7_collision_credit", "ar", gpu=False, hours=0.5, mem="16G", commands=c_retro7, paper="core",
         after=("controlled/exp6_ar_copy_in_decoding",), summary=s_collect),
    # ---- additions
    Spec("additions/n1_prefix_conditioning", "any", hours=2,
         commands=c_diff("additions/n1_prefix_conditioning/run.py"),
         summary=s_cells("additions/n1_prefix_conditioning/summarize.py")),
    Spec("additions/n2_ar_prefix_match", "ar", hours=3, paper="core",
         commands=c_diff("additions/n2_ar_prefix_match/run.py"),
         summary=lambda cells, out: [_py("additions/n2_ar_prefix_match/summarize.py", "--cells", *map(str, cells),
                                         "--induction", paths.RESULTS / "additions" / "n2_ar_prefix_match" / "induction",
                                         "--exp4-root", paths.RESULTS / "controlled" / "exp4_ar_copy_circuit",
                                         "--out", out)]),
    Spec("additions/n2_ar_prefix_match/induction", "ar", hours=1, commands=c_n2f, single=True),
    Spec("additions/n3_copy_dose", "any", hours=8, paper="deep",
         commands=c_diff("additions/n3_copy_dose/run.py"),
         summary=s_cells("additions/n3_copy_dose/summarize.py")),
    Spec("additions/n4_no_training_baselines", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5, mem="16G",
         commands=lambda c: [_py("additions/n4_no_training_baselines/run.py", "--ckpt", c.ckpt, "--out", c.out)],
         summary=s_cells("additions/n4_no_training_baselines/summarize.py")),
]
SPECS = {s.name: s for s in SPEC_LIST}
STAGES = {
    "checks": [],
    "cpu": [s.name for s in SPEC_LIST if not s.gpu],
    "ar": [s.name for s in SPEC_LIST if s.gpu and s.paradigm == "ar"],
    "diffgrm": [s.name for s in SPEC_LIST if s.gpu and s.paradigm == "diffgrm"],
    "additions": [s.name for s in SPEC_LIST if s.name.startswith("additions/")],
}


# ---------------------------------------------------------------- planning --
def cell_dir(spec: Spec, ckpt: str | None, pilot: bool = False) -> Path:
    base = (paths.RESULTS / "pilot" if pilot else paths.RESULTS) / spec.name
    return base if spec.single else base / grid.map_dir(ckpt)


def is_complete(d: Path, full: bool = False) -> bool:
    """`run_cell` writes `complete` only after every command exited 0 (a killed job stays `running`).
    With `full`, a paper-set cell that ran fewer parts (`scope: broad`) does not count (trap 6)."""
    st = d / "status.txt"
    if not (st.exists() and st.read_text().startswith("complete")):
        return False
    return not full or _scope(d) != "broad"


def _status(d: Path) -> str:
    st = d / "status.txt"
    return st.read_text() if st.exists() else ""


def _scope(d: Path) -> str:
    cj = d / "cell.json"
    return json.loads(cj.read_text()).get("scope", "full") if cj.exists() else "full"


def paper_tier(ckpt: str) -> str:
    """deep (a core map's native table), core (its ZCR twin) or broad (any other map)."""
    i = grid.cell_info(ckpt)
    if i["map"].split("__zcr-")[0] not in PAPER_CORE_MAPS:
        return "broad"
    return "core" if i["zcr"] else "deep"


def in_paper(spec: Spec, ckpt: str | None) -> bool:
    if spec.paper is None:
        return False
    if spec.paper == "all" or ckpt is None:
        return True
    tier = paper_tier(ckpt)
    return tier == "deep" or (spec.paper == "core" and tier == "core")


def in_case(ckpt: str) -> bool:
    i = grid.cell_info(ckpt)
    return not i["zcr"] and i["map"] in CASE_MAPS


def needs_full(which: str, spec: Spec, ckpt: str | None) -> bool:
    """Whether the set needs this cell with every part (trap 6)."""
    return spec.tiered and not (which == "paper" and ckpt is not None and paper_tier(ckpt) == "broad")


def plan(which: str, stages=None) -> list[tuple[Spec, str | None]]:
    """(spec, ckpt) of every cell of the set, in spec order."""
    from sidlens.grid import registry
    if which not in SETS:
        raise ValueError(f"unknown set {which!r}; sets are {SETS}")
    reg = registry.load()
    names = [n for st in (stages or STAGES) for n in STAGES.get(st, [st])] if stages else list(SPECS)
    cells = []
    for name in dict.fromkeys(names):
        spec = SPECS[name]
        if spec.single:
            if which != "paper" or in_paper(spec, None):
                cells.append((spec, None))
            continue
        for ck in reg.names(states="all"):
            e = reg.entry(ck)
            st = grid.split_name(ck)[1]
            if spec.paradigm != "any" and e["paradigm"] != spec.paradigm:
                continue
            if e["task"] not in spec.tasks or (spec.depth3 and int(e["digits"]) != 3):
                continue
            if st != grid.BEST and not (which in ("full", "case") and spec.cheap):
                continue
            if which == "paper" and not in_paper(spec, ck):
                continue
            if which == "case" and not in_case(ck):
                continue
            cells.append((spec, ck))
    return cells


def _ctx(spec: Spec, ckpt: str | None, pilot=False, which: str = "best") -> Ctx:
    from sidlens.grid import registry
    entry = registry.load().entry(ckpt) if ckpt else {}
    out = cell_dir(spec, ckpt, pilot)
    # a pilot cell may run beside the real cell of the same map, and the AR chain
    # starts by deleting its scratch folder: the two must never share one
    scratch = (paths.SCRATCH / "pilot" if pilot else paths.SCRATCH) / spec.name / (
        grid.map_dir(ckpt) if ckpt else "single")
    full = not (which == "paper" and ckpt is not None and paper_tier(ckpt) == "broad")
    return Ctx(ckpt or "", entry, grid.split_name(ckpt)[1] if ckpt else grid.BEST, out, scratch, pilot, full)


# ---------------------------------------------------------------- one cell --
def _source_hashes() -> dict:
    from sidlens.provenance.hashing import sha256_file
    files = sorted(p for root in (REPO / "src" / "sidlens", EXP) for p in root.rglob("*.py")
                   if "__pycache__" not in p.parts)
    return {str(p.relative_to(REPO)): sha256_file(p) for p in files}


def run_cell(spec: Spec, ckpt: str | None, pilot: bool = False, which: str = "best") -> int:
    c = _ctx(spec, ckpt, pilot, which)
    if not pilot and is_complete(c.out, needs_full(which, spec, ckpt)):
        print(f"[skip] {spec.name} {ckpt}: complete", flush=True)
        return 0
    for parent in spec.after:
        if not is_complete(c.cell(parent)):
            raise RuntimeError(f"{spec.name} {ckpt}: parent cell {parent} is not complete (trap 3)")
    if c.out.exists():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        kind = ("superseded" if is_complete(c.out)                   # complete but broad: trap 6
                else "interrupted" if _status(c.out).startswith("interrupted") else "failed")   # trap 8
        c.out.rename(c.out.with_name(f"{c.out.name}.{kind}-{stamp}"))
    c.out.mkdir(parents=True)
    t0 = time.time()
    (c.out / "status.txt").write_text(f"running started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    (c.out / "cell.json").write_text(json.dumps({"spec": spec.name, "checkpoint": ckpt, "pilot": pilot,
                                                 "set": which,
                                                 "scope": "full" if c.full or not spec.tiered else "broad",
                                                 "host": os.uname().nodename,
                                                 "slurm_job": os.environ.get("SLURM_JOB_ID")}, indent=1))
    (c.out / "source.sha256.json").write_text(json.dumps(_source_hashes(), indent=1))
    rc = 0
    with open(c.out / "pipeline.log", "a") as log:
        for cmd in spec.commands(c):
            log.write(f"$ {' '.join(map(str, cmd))}\n")
            log.flush()
            rc = subprocess.call([str(x) for x in cmd], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if rc:
                break
    state = "complete" if rc == 0 else "failed"
    if rc == 0:
        from sidlens.provenance.hashing import sha256_file
        sums = {str(p.relative_to(c.out)): sha256_file(p) for p in sorted(c.out.rglob("*"))
                if p.is_file() and p.name not in ("output.sha256.json", "status.txt")}
        (c.out / "output.sha256.json").write_text(json.dumps(sums, indent=1))
    (c.out / "status.txt").write_text(f"{state} exit={rc} seconds={time.time() - t0:.0f} "
                                      f"finished={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"[{state}] {spec.name} {ckpt} in {time.time() - t0:.0f}s -> {c.out}", flush=True)
    return rc


# ---------------------------------------------------------------- Slurm --
SBATCH = REPO / "scripts" / "grid" / "cell.sbatch"


def _sbatch(spec: Spec, task_file: Path, n_tasks: int, depend: str | None, dry: bool) -> str | None:
    per = max(1, int(os.environ.get("CELLS_PER_TASK", "1")))
    n_arr = -(-n_tasks // per)
    conc = os.environ.get("MAX_GPUS" if spec.gpu else "MAX_CPUS", "16" if spec.gpu else "32")
    cmd = ["sbatch", "--parsable", f"--array=0-{n_arr - 1}%{conc}", f"--job-name=sl-{spec.name.split('/')[-1][:20]}",
           f"--time={int(spec.hours * per * 60 * 1.5) + 30}", f"--mem={spec.mem}",
           f"--output={paths.RESULTS / 'provenance' / 'slurm' / '%x-%A_%a.out'}"]
    part = os.environ.get("SLURM_PARTITION" if spec.gpu else "CPU_PARTITION", os.environ.get("SLURM_PARTITION"))
    for flag, var in (("--account", "SLURM_ACCOUNT"), ("--qos", "SLURM_QOS" if spec.gpu else "CPU_QOS"),
                      ("--exclude", "SLURM_EXCLUDE")):
        if os.environ.get(var):
            cmd.append(f"{flag}={os.environ[var]}")
    if part:
        cmd.append(f"--partition={part}")
    cmd += [f"--gres={os.environ.get('GPU_GRES', 'gpu:1')}", "--cpus-per-task=8"] if spec.gpu \
        else ["--cpus-per-task=16"]
    if depend:
        cmd.append(f"--dependency=afterany:{depend}")
    cmd += [str(SBATCH), str(task_file), str(per)]
    if dry:
        print("DRY", " ".join(cmd))
        return None
    (paths.RESULTS / "provenance" / "slurm").mkdir(parents=True, exist_ok=True)
    return subprocess.check_output(cmd, text=True).strip().split(";")[0]


def submit(which: str, stages, local: bool, dry: bool) -> None:
    cells = [(s, ck) for s, ck in plan(which, stages) if not is_complete(cell_dir(s, ck), needs_full(which, s, ck))]
    by: dict[str, list] = {}
    for s, ck in cells:
        by.setdefault(s.name, []).append(ck)
    jobs: dict[str, str] = {}
    tasks = paths.RESULTS / "registry" / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for name, cks in by.items():
        spec = SPECS[name]
        if local:
            for ck in cks:
                run_cell(spec, ck, which=which)
            continue
        tf = tasks / f"{name.replace('/', '__')}-{time.strftime('%Y%m%dT%H%M%S')}.json"
        if not dry:                                   # trap 2: the submitted cell list, frozen
            tf.write_text(json.dumps({"spec": name, "set": which, "cells": cks}, indent=1))
        dep = ":".join(jobs[p] for p in spec.after if p in jobs) or None
        jid = _sbatch(spec, tf, len(cks), dep, dry)
        if jid:
            jobs[name] = jid
        print(f"{name}: {len(cks)} cells{' -> job ' + jid if jid else ''}")


# ---------------------------------------------------------------- one job --
BATCH_SBATCH = REPO / "scripts" / "grid" / "batch.sbatch"


def _job_name(which: str) -> str:
    return f"sidlens-{which}"


def _queued_jobs(name: str) -> list[str]:
    """Job ids of this user's queued or running jobs with this name, other than this job itself."""
    try:
        out = subprocess.check_output(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", name, "-o", "%i"],
                                      text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [j for j in out.split() if j != os.environ.get("SLURM_JOB_ID")]


def submit_batch(which: str, stages=None, dry: bool = False, round_: int = 0) -> str | None:
    """Queue the whole set as ONE Slurm job (trap 7): one node, `BATCH_GPUS` GPUs,
    `BATCH_CPUS` cores, `BATCH_MEM`, `BATCH_TIME`; partition, account, QoS and excluded
    nodes as for the arrays. Refuses while a job of the same set is queued or running."""
    name = _job_name(which)
    if not dry and (q := _queued_jobs(name)):
        raise RuntimeError(f"job {q[0]} ({name}) is already queued or running; one job at a time "
                           f"(it requeues itself; `scancel {q[0]}` first to start over)")
    gpus = int(os.environ.get("BATCH_GPUS", "8"))
    gres = f"gpu:{os.environ['GPU_TYPE']}:{gpus}" if os.environ.get("GPU_TYPE") else f"gpu:{gpus}"
    cmd = ["sbatch", "--parsable", f"--job-name={name}", "--nodes=1", "--ntasks=1", f"--gres={gres}",
           f"--cpus-per-task={os.environ.get('BATCH_CPUS', '64')}", f"--mem={os.environ.get('BATCH_MEM', '0')}",
           f"--time={os.environ.get('BATCH_TIME', '2-00:00:00')}",
           f"--output={paths.RESULTS / 'provenance' / 'slurm' / f'{name}-%j.out'}"]
    for flag, var in (("--partition", "SLURM_PARTITION"), ("--account", "SLURM_ACCOUNT"), ("--qos", "SLURM_QOS"),
                      ("--exclude", "SLURM_EXCLUDE")):
        if os.environ.get(var):
            cmd.append(f"{flag}={os.environ[var]}")
    cmd += [str(BATCH_SBATCH), which, str(round_), *(stages or [])]
    if dry:
        print("DRY", " ".join(cmd))
        return None
    (paths.RESULTS / "provenance" / "slurm").mkdir(parents=True, exist_ok=True)
    jid = subprocess.check_output(cmd, text=True).strip().split(";")[0]
    print(f"queued one job {jid} ({name}, round {round_}); progress in {paths.RESULTS / 'batch'}")
    return jid


def _parse_slurm_time(s: str) -> float | None:
    """Seconds in a Slurm duration ([D-]HH:MM:SS, MM:SS); None when unlimited or unset."""
    s = s.strip()
    if not s or s.upper().startswith(("UNLIMITED", "INVALID", "NOT_SET")):
        return None
    days, _, hms = s.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    parts = [0] * (3 - len(parts)) + parts
    return ((int(days) if days else 0) * 24 + parts[0]) * 3600 + parts[1] * 60 + parts[2]


def _seconds_left() -> float | None:
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        return _parse_slurm_time(subprocess.check_output(["squeue", "-h", "-j", jid, "-o", "%L"], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _visible_gpus() -> list[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if v:
        return [g for g in v.split(",") if g]
    try:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        return []


def _interleave(cells: list) -> list:
    """One cell of every spec, then the second of every spec, and so on: a broken or slow
    experiment shows up in the job's first hours, and early timings cover every spec."""
    queues: dict = {}
    for s, ck in cells:
        queues.setdefault(s.name, []).append((s, ck))
    out, qs = [], list(queues.values())
    while qs:
        out += [q.pop(0) for q in qs]
        qs = [q for q in qs if q]
    return out


def _attempts(d: Path) -> int:
    """Failed runs of a cell: the ones moved aside, plus the cell itself if its last run failed."""
    moved = len(list(d.parent.glob(f"{d.name}.failed-*"))) if d.parent.is_dir() else 0
    return moved + _status(d).startswith("failed")


def _gb(mem: str) -> float:
    """A Slurm memory size ("64G", "500M", "1T"; plain numbers are MB) in GB."""
    unit = mem[-1].upper() if mem[-1].isalpha() else "M"
    return float(mem.rstrip("KMGTkmgt")) * {"K": 2 ** -20, "M": 2 ** -10, "G": 1, "T": 2 ** 10}[unit]


def _node_mem_gb() -> float:
    """Memory this job may use: Slurm's allocation (SLURM_MEM_PER_NODE, MB), else the machine's."""
    if os.environ.get("SLURM_MEM_PER_NODE"):
        return int(os.environ["SLURM_MEM_PER_NODE"]) / 1024
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30


def _seconds(d: Path) -> float | None:
    for tok in _status(d).split():
        if tok.startswith("seconds="):
            return float(tok.split("=", 1)[1])
    return None


def _launch_one(spec: Spec, ckpt: str | None, which: str, env: dict, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-u", "-m", "sidlens.grid.pipeline", "one", "--spec", spec.name, "--set", which,
           *(["--ckpt", ckpt] if ckpt else [])]
    with open(log, "a") as fh:
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def batch(which: str, stages=None, gpu_ids=None, cpu_workers: int = 4, seconds: float | None = None,
          max_attempts: int = 2, poll: float = 15.0, launch: Callable = _launch_one,
          mem_gb: float | None = None) -> dict:
    """Every unfinished cell of the set, inside this one allocation.

    One cell per GPU at a time (CUDA_VISIBLE_DEVICES set to that GPU) and up to
    `cpu_workers` CPU cells (no GPU visible), each in its own process (`one`), in
    `_interleave` order. A cell whose parent (`Spec.after`) is still to run waits;
    one whose parent failed is reported as blocked. A cell that already failed
    `max_attempts` times is reported, not rerun. With `seconds` (default: the Slurm
    job's time left minus 10 minutes), no cell starts that its spec's median time so
    far says would not finish. A spec with no finished cell yet always starts:
    `Spec.hours` is an upper bound for array `--time`, not an estimate, and a job
    shorter than it would otherwise never run the spec. Running cells' `Spec.mem` may
    not exceed 90 % of `mem_gb` (default: the job's memory), so several representation
    chains cannot exhaust the node; a cell larger than that still runs alone. On
    SIGTERM the running cells are stopped and marked `interrupted` (trap 8). Writes
    `results_remote/batch/progress-*` after every cell: per spec done, failed and
    left, median minutes, hours left.
    """
    import signal
    t0 = time.time()
    if seconds is None and (left := _seconds_left()) is not None:
        seconds = left - 600
    deadline = t0 + seconds if seconds else None
    gpu_ids = list(_visible_gpus() if gpu_ids is None else gpu_ids)
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 8)
    threads = str(max(1, ncpu // max(1, len(gpu_ids) + cpu_workers)))
    cells = plan(which, stages)
    todo = [(s, ck) for s, ck in cells if not is_complete(cell_dir(s, ck), needs_full(which, s, ck))]
    gave_up = [(s, ck) for s, ck in todo if _attempts(cell_dir(s, ck)) >= max_attempts]
    todo = _interleave([x for x in todo if x not in gave_up])
    if any(s.gpu for s, _ in todo) and not gpu_ids:
        raise RuntimeError("GPU cells to run but no GPU visible (CUDA_VISIBLE_DEVICES)")
    med: dict[str, list] = {}
    for s, ck in cells:
        if (sec := _seconds(cell_dir(s, ck))) is not None and is_complete(cell_dir(s, ck)):
            med.setdefault(s.name, []).append(sec)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    prog = paths.RESULTS / "batch" / f"progress-{which}-{os.environ.get('SLURM_JOB_ID', stamp)}.json"
    logs = paths.RESULTS / "batch" / "logs"
    running: dict = {}                                   # Popen -> (spec, ckpt, slot, start)
    free_gpus, free_cpu = list(gpu_ids), cpu_workers
    budget = 0.9 * (mem_gb if mem_gb is not None else _node_mem_gb())

    def mem_in_use() -> float:
        return sum(_gb(v[0].mem) for v in running.values())
    done, failed, blocked = [], [], []
    stop = {"signal": False}

    def estimate(s: Spec) -> float:
        xs = sorted(med.get(s.name, []))
        return xs[len(xs) // 2] if xs else s.hours * 3600

    def write_progress(final=False):
        by: dict = {}
        for key, rows in (("done", done), ("failed", failed), ("blocked", blocked), ("gave_up", gave_up),
                          ("left", todo), ("running", [(v[0], v[1]) for v in running.values()])):
            for s, _ in rows:
                d = by.setdefault(s.name, {"gpu": s.gpu})
                d[key] = d.get(key, 0) + 1
        for name, d in by.items():
            xs = sorted(med.get(name, []))
            d["median_minutes"] = round(xs[len(xs) // 2] / 60, 1) if xs else None
            d["hours_left"] = round((d.get("left", 0) + d.get("running", 0)) * estimate(SPECS[name]) / 3600, 1)
        gpu_h = sum(d["hours_left"] for d in by.values() if d["gpu"])
        doc = {"set": which, "job": os.environ.get("SLURM_JOB_ID"), "gpus": gpu_ids, "cpu_workers": cpu_workers,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
               "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "final": final,
               "gpu_hours_left": round(gpu_h, 1),
               "wall_hours_left_on_these_gpus": round(gpu_h / max(1, len(gpu_ids)), 1), "specs": by}
        prog.parent.mkdir(parents=True, exist_ok=True)
        prog.write_text(json.dumps(doc, indent=1))

    def on_term(signum, frame):
        stop["signal"] = True

    old = signal.signal(signal.SIGTERM, on_term)
    print(f"[batch] {which}: {len(todo)} cells to run, {len(gave_up)} given up after {max_attempts} failures, "
          f"GPUs {gpu_ids}, {cpu_workers} CPU workers, {threads} threads per cell, {budget:.0f} GB for cells, "
          f"{'no deadline' if deadline is None else f'{(deadline - t0) / 3600:.1f} h'}", flush=True)
    write_progress()
    try:
        while (todo or running) and not stop["signal"]:
            for p in [p for p in running if p.poll() is not None]:
                s, ck, slot, start = running.pop(p)
                if s.gpu:
                    free_gpus.append(slot)
                else:
                    free_cpu += 1
                ok = is_complete(cell_dir(s, ck), needs_full(which, s, ck))
                (done if ok else failed).append((s, ck))
                if ok:
                    med.setdefault(s.name, []).append(time.time() - start)
                print(f"[batch] {'done' if ok else 'FAILED'} {s.name} {ck} {(time.time() - start) / 60:.1f} min",
                      flush=True)
                write_progress()
            launched = False
            for i, (s, ck) in enumerate(list(todo)):
                if (s.gpu and not free_gpus) or (not s.gpu and free_cpu <= 0):
                    continue
                parents = [(SPECS[p], ck) for p in s.after]
                if any(not is_complete(cell_dir(ps, pck)) for ps, pck in parents):
                    if not any(x in todo or x in [(v[0], v[1]) for v in running.values()] for x in parents):
                        blocked.append((s, ck))             # its parent failed or was given up
                        todo.remove((s, ck))
                    continue
                if deadline is not None and med.get(s.name) and time.time() + estimate(s) > deadline:
                    continue                                  # measured, and would not finish
                if running and mem_in_use() + _gb(s.mem) > budget:
                    continue                                  # waits for memory; alone it always runs
                env = {**os.environ, "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}
                if s.gpu:
                    slot = free_gpus.pop(0)
                    env["CUDA_VISIBLE_DEVICES"] = slot
                else:
                    slot, free_cpu = None, free_cpu - 1
                    env["CUDA_VISIBLE_DEVICES"] = ""
                tag = f"{s.name}__{grid.map_dir(ck) if ck else 'single'}".replace("/", "__")
                running[launch(s, ck, which, env, logs / f"{tag}.log")] = (s, ck, slot, time.time())
                todo.remove((s, ck))
                launched = True
            if not running and not launched:
                break                                     # nothing fits before the deadline, or all blocked
            if not launched:
                time.sleep(poll)
    finally:
        if running:                                       # trap 8: the job's end, not the cells' fault
            for p in running:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            end = time.time() + 15                        # inside Slurm's grace before SIGKILL
            for p, (s, ck, _, _) in running.items():
                try:
                    p.wait(timeout=max(0.1, end - time.time()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                d = cell_dir(s, ck)
                if is_complete(d, needs_full(which, s, ck)):           # finished before the signal
                    done.append((s, ck))
                    continue
                if d.is_dir() and _status(d).startswith("running"):     # a cell that failed stays failed
                    (d / "status.txt").write_text(f"interrupted finished={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                todo.append((s, ck))
            running.clear()
        signal.signal(signal.SIGTERM, old)
        write_progress(final=True)
    res = {"done": len(done), "failed": len(failed), "blocked": len(blocked), "gave_up": len(gave_up),
           "left": len(todo), "progress": str(prog), "signal": stop["signal"]}
    print(f"[batch] end: {json.dumps(res)}", flush=True)
    return res


def finish_or_requeue(which: str, stages, res: dict, round_: int, auto: bool = True) -> None:
    """After `batch`: if cells are left and this round made progress, queue the next round
    (one job; trap 7); if none are left, write the summaries and the export."""
    if res["left"]:
        max_rounds = int(os.environ.get("BATCH_MAX_ROUNDS", "10"))
        if auto and res["done"] and round_ + 1 < max_rounds:
            try:
                submit_batch(which, stages, round_=round_ + 1)
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                print(f"[batch] could not queue the next job ({type(exc).__name__}: {exc}); {res['left']} cells "
                      f"left: queue it from the login node with `sidlens-grid.sh run --set {which}`", flush=True)
        else:
            print(f"[batch] {res['left']} cells left and not requeued (progress {res['done']}, round {round_}); "
                  f"queue the next one with `sidlens-grid.sh run --set {which}`", flush=True)
        return
    summarize()
    print(f"[batch] export: {export()}", flush=True)


# ---------------------------------------------------------------- reports --
def status(which: str) -> dict:
    out: dict = {}
    for s, ck in plan(which):
        d = cell_dir(s, ck)
        if is_complete(d, needs_full(which, s, ck)):
            k = "complete"
        else:                                         # a broad cell the set needs in full is still to run
            k = "failed" if (d / "status.txt").exists() and not is_complete(d) else "missing"
        out.setdefault(s.name, {"complete": 0, "failed": 0, "missing": 0})[k] += 1
    return out


def summarize() -> None:
    """Each spec's summary over its complete cells, per dataset and state kind, into summary-<stamp>/."""
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for s, cks in _complete_by_spec().items():
        spec = SPECS[s]
        if spec.summary is None:
            continue
        groups: dict = {}
        for ck in cks:
            i = grid.cell_info(ck)
            groups.setdefault((i["dataset"], i["task"], i["paradigm"],
                               "best" if i["state"] == grid.BEST else "intermediate"), []).append(cell_dir(spec, ck))
        for key, cells in groups.items():
            out = paths.RESULTS / s / "summary" / f"{'-'.join(key)}-{stamp}"
            out.parent.mkdir(parents=True, exist_ok=True)
            for cmd in spec.summary(cells, out):
                rc = subprocess.call([str(x) for x in cmd], cwd=REPO)
                print(f"[summary {'ok' if rc == 0 else 'FAILED'}] {s} {key}", flush=True)


def _complete_by_spec() -> dict:
    out: dict = {}
    for s, ck in plan("full"):
        if ck and is_complete(cell_dir(s, ck)):
            out.setdefault(s.name, []).append(ck)
    return out


def export() -> Path:
    """MANIFEST.sha256 over results_remote/ and a tarball beside it."""
    from sidlens.provenance.hashing import sha256_file
    root = paths.RESULTS
    lines = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != "MANIFEST.sha256"]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    (root / "README.md").write_text(
        f"# results_remote\n\nsidlens commit {commit or 'unknown'} (plus uncommitted changes, see provenance/),\n"
        f"OneDiffRec work root {os.environ.get('ONEDIFFREC_WORK_ROOT', '?')}, exported "
        f"{time.strftime('%Y-%m-%dT%H:%M:%S')}.\nLayout: docs/plans/COPYING_GRID_PLAN.md section 2.3.\n")
    tar = paths.WORK / "exports" / f"results_remote-{time.strftime('%Y%m%dT%H%M%S')}.tar.gz"
    tar.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["tar", "-czf", str(tar), "-C", str(root.parent), root.name])
    return tar


# ---------------------------------------------------------------- CLI --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sidlens grid", description="sidlens on the OneDiffRec grid")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register", help="discover the grid's runs and weight states; write the registry")
    r.add_argument("--work-root", type=Path, default=None, help="default: $ONEDIFFREC_WORK_ROOT")
    r.add_argument("--run-name", nargs="+", default=["grid-v1"])
    r.add_argument("--tables", default="v3")
    r.add_argument("--mode", default="full")
    r.add_argument("--no-source-check", action="store_true")
    for name in ("plan", "run", "status"):
        x = sub.add_parser(name)
        x.add_argument("--set", choices=SETS, default="best")
        x.add_argument("--stage", nargs="*", help=f"stages {sorted(STAGES)} or spec names; default all")
        if name == "run":
            x.add_argument("--local", action="store_true", help="run here, one cell after another")
            x.add_argument("--arrays", action="store_true",
                           help="one Slurm array per spec instead of one job (thousands of array tasks)")
            x.add_argument("--dry-run", action="store_true")
    b = sub.add_parser("batch", help="every unfinished cell of a set inside this one job (batch.sbatch)")
    b.add_argument("--set", choices=SETS, default="paper")
    b.add_argument("--stage", nargs="*")
    b.add_argument("--round", type=int, default=0, help="how many jobs of this set ran before this one")
    b.add_argument("--cpu-workers", type=int, default=int(os.environ.get("BATCH_CPU_WORKERS", "4")))
    b.add_argument("--max-attempts", type=int, default=int(os.environ.get("BATCH_MAX_ATTEMPTS", "2")))
    b.add_argument("--seconds", type=float, default=None, help="default: the Slurm job's time left - 10 min")
    b.add_argument("--no-requeue", action="store_true")
    o = sub.add_parser("one", help="one cell (started by batch)")
    o.add_argument("--spec", required=True)
    o.add_argument("--ckpt", default=None)
    o.add_argument("--set", choices=SETS, default="best")
    p = sub.add_parser("pilot", help="one short cell per spec, timed, under results_remote/pilot/")
    p.add_argument("--stage", nargs="*")
    p.add_argument("--ckpt-like", default="rqkmeans_3codebook_128", help="map label to pilot on")
    c = sub.add_parser("cell", help="run cells of a task file (inside a Slurm array task)")
    c.add_argument("--task-file", type=Path, required=True)
    c.add_argument("--index", type=int, required=True)
    c.add_argument("--per-task", type=int, default=1)
    sub.add_parser("summarize")
    sub.add_parser("export")
    args = ap.parse_args(argv)

    if args.cmd == "register":
        from sidlens.grid import registry
        root = args.work_root or grid.onediffrec_work_root()
        doc = registry.build(root, args.run_name if len(args.run_name) > 1 else args.run_name[0], args.tables,
                             args.mode, check_source=not args.no_source_check)
        print(f"registry: {registry.save(doc)}")
        return 0
    if args.cmd == "plan":
        cells = plan(args.set, args.stage)
        by: dict = {}
        for s, ck in cells:
            d = by.setdefault(s.name, {"cells": 0, "todo": 0, "gpu": s.gpu, "hours": 0.0})
            d["cells"] += 1
            if not is_complete(cell_dir(s, ck), needs_full(args.set, s, ck)):
                d["todo"] += 1
                d["hours"] += s.hours
        for name, d in by.items():
            print(f"{name:55s} {d['cells']:5d} cells {d['todo']:5d} to run  ~{d['hours']:7.1f} "
                  f"{'GPU' if d['gpu'] else 'CPU'}-h")
        print(f"total ~{sum(d['hours'] for d in by.values() if d['gpu']):.0f} GPU-h, "
              f"~{sum(d['hours'] for d in by.values() if not d['gpu']):.0f} CPU-h (upper bounds per cell)")
        return 0
    if args.cmd == "run":
        if args.local or args.arrays:
            submit(args.set, args.stage, args.local, args.dry_run)
        else:
            submit_batch(args.set, args.stage, args.dry_run)
        return 0
    if args.cmd == "batch":
        res = batch(args.set, args.stage, cpu_workers=args.cpu_workers, seconds=args.seconds,
                    max_attempts=args.max_attempts)
        finish_or_requeue(args.set, args.stage, res, args.round, auto=not args.no_requeue)
        return 0 if not res["failed"] else 1
    if args.cmd == "one":
        return run_cell(SPECS[args.spec], args.ckpt, which=args.set)
    if args.cmd == "status":
        for name, d in status(args.set).items():
            print(f"{name:55s} complete {d['complete']:5d}  failed {d['failed']:5d}  missing {d['missing']:5d}")
        return 0
    if args.cmd == "pilot":
        names = [n for st in (args.stage or STAGES) for n in STAGES.get(st, [st])] if args.stage else list(SPECS)
        seen, timings = set(), []
        n_cells = {}
        for s, ck in plan("full", None):
            n_cells[s.name] = n_cells.get(s.name, 0) + 1
        for s, ck in plan("best", None):
            if s.name not in names or s.name in seen or (ck and args.ckpt_like not in ck):
                continue
            seen.add(s.name)
            t0 = time.time()
            try:
                rc = run_cell(s, ck, pilot=True)
            except Exception as exc:                  # one failing spec must not hide the others' timings
                print(f"[pilot] {s.name}: {type(exc).__name__}: {exc}", flush=True)
                rc = -1
            sec = time.time() - t0
            timings.append({"spec": s.name, "checkpoint": ck, "rc": rc, "seconds": round(sec, 1), "gpu": s.gpu,
                            "cells_in_full_set": n_cells.get(s.name, 0)})
            print(f"[pilot] {s.name}: rc={rc} {sec:.0f}s", flush=True)
        out = paths.RESULTS / "pilot" / f"timings-{time.strftime('%Y%m%dT%H%M%S')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"note": "pilot cells use --limit 64 rows (600 train rows for captures); "
                                           "a full cell is roughly rows/64 times longer for row-bound parts, "
                                           "plus fixed load and check time", "timings": timings}, indent=1))
        print(f"timings: {out}")
        return 0 if all(t["rc"] == 0 for t in timings) else 1
    if args.cmd == "cell":
        doc = json.loads(args.task_file.read_text())
        spec = SPECS[doc["spec"]]
        rc = 0
        for ck in doc["cells"][args.index * args.per_task:(args.index + 1) * args.per_task]:
            rc |= run_cell(spec, ck, which=doc.get("set", "best"))
        return rc
    if args.cmd == "summarize":
        summarize()
        return 0
    if args.cmd == "export":
        print(export())
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
