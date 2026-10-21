"""The grid superscript: every sidlens experiment on every OneDiffRec grid run, as Slurm arrays.

    python -m sidlens.grid.pipeline register --work-root <OneDiffRec WORK_ROOT> [--run-name grid-v1]
    python -m sidlens.grid.pipeline plan   --set best|full [--stage S]
    python -m sidlens.grid.pipeline pilot  [--stage S] [--local]
    python -m sidlens.grid.pipeline run    --set best|full [--stage S] [--local] [--dry-run]
    python -m sidlens.grid.pipeline status [--set best|full]
    python -m sidlens.grid.pipeline summarize
    python -m sidlens.grid.pipeline export
    python -m sidlens.grid.pipeline cell   --task-file F --index I      (inside a Slurm array task)

A *cell* is one experiment spec on one grid weight state; it writes to
`results_remote/<group>/<experiment>/<dataset>/<task>/<paradigm>/<map>/<state>/`
(`grid.map_dir`). Sets:

    best   every experiment on every run, best state only (`run_best_state.sh`)
    full   best, plus the cheap set at every intermediate weight state
           (`run_full.sh`): AR exp3 Part A and exp4 Part R (test); DiffGRM exp7

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


def _limit(c: Ctx) -> list[str]:
    return ["--limit", "64"] if c.pilot else []


# ---------------------------------------------------------------- commands --
def c_exp3(c):
    return [_py("controlled/exp3_ar_history_patching/run.py", "--ckpt", c.ckpt, "--template", "v3",
                "--parts", "A,B" if c.best else "A", "--out", c.out, *_limit(c))]


def c_exp4(c):
    cmds = [_py("controlled/exp4_ar_copy_circuit/run.py", "--ckpt", c.ckpt, "--split", "test",
                "--parts", "R,K,H" if c.best else "R", "--out", c.out / "test", *_limit(c))]
    if c.best:
        cmds.append(_py("controlled/exp4_ar_copy_circuit/run.py", "--ckpt", c.ckpt, "--split", "valid",
                        "--parts", "R,K,H", "--out", c.out / "valid", *_limit(c)))
    return cmds


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
            "--analysis", r2 / "analyze", "--out", r2 / "amend_a1"),
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
    Spec("controlled/exp3_ar_history_patching", "ar", cheap=True, hours=2, commands=c_exp3,
         summary=s_cells("controlled/exp3_ar_history_patching/summarize.py")),
    Spec("controlled/exp4_ar_copy_circuit", "ar", cheap=True, hours=4, commands=c_exp4,
         summary=s_exp4),
    Spec("controlled/exp5_ar_next_two_conditioning", "ar", tasks=("two-item",), hours=2, commands=c_exp5,
         summary=s_per_cell(lambda c, o: [_py("controlled/exp5_ar_next_two_conditioning/summarize.py",
                                              "--cell", c, "--out", o)])),
    Spec("controlled/exp6_ar_copy_in_decoding", "ar", hours=8, commands=c_exp6,
         summary=s_cells("controlled/exp6_ar_copy_in_decoding/summarize.py")),
    Spec("controlled/exp9_ar_order_vs_time", "ar", hours=4, commands=c_exp9,
         summary=s_cells("controlled/exp9_ar_order_vs_time/summarize.py")),
    Spec("controlled/exp10_ar_last_digit_sibling", "ar", hours=1, commands=c_exp10,
         summary=lambda cells, out: [_py("controlled/exp10_ar_last_digit_sibling/summarize.py", "--cells",
                                         *[str(c / s) for c in cells for s in ("test", "valid")], "--out", out)]),
    # ---- AR, representation (one chain per run; outputs in the three experiment folders)
    Spec("representation/ar_chain", "ar", hours=12, mem="128G", commands=c_rep_ar,
         summary=s_ar_chain),
    # ---- DiffGRM
    Spec("controlled/exp1_matched_beam", "diffgrm", hours=2, commands=c_exp1,
         summary=s_cells("controlled/exp1_matched_beam/summarize_grid.py")),
    Spec("controlled/exp2_fixed_orders", "diffgrm", depth3=True, hours=1, commands=c_exp2,
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
    Spec("representation/exp4_diffusion_reveal_trace", "diffgrm", hours=2,
         commands=c_diff("representation/exp4_diffusion_reveal_trace/run.py"),
         summary=s_cells("representation/exp4_diffusion_reveal_trace/summarize.py", "--skip-failed")),
    Spec("representation/exp5_diffusion_encoder_features", "diffgrm", hours=2,
         commands=c_diff("representation/exp5_diffusion_encoder_features/run.py"),
         summary=s_cells("representation/exp5_diffusion_encoder_features/summarize.py")),
    # ---- saved lists (CPU, every run)
    Spec("retrospective/exp2_prefix", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5,
         commands=c_saved("exp2"), summary=s_collect),
    Spec("retrospective/exp3_collisions", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5,
         commands=c_saved("exp3"), summary=s_collect),
    Spec("retrospective/exp4_next_two", "any", tasks=("two-item",), gpu=False, hours=0.5,
         commands=c_saved("exp4"), summary=s_collect),
    Spec("retrospective/exp6_time_structure", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5,
         commands=c_saved("all"), summary=s_collect),
    Spec("retrospective/exp7_collision_credit", "ar", gpu=False, hours=0.5, commands=c_retro7,
         after=("controlled/exp6_ar_copy_in_decoding",), summary=s_collect),
    # ---- additions
    Spec("additions/n1_prefix_conditioning", "any", hours=2,
         commands=c_diff("additions/n1_prefix_conditioning/run.py"),
         summary=s_cells("additions/n1_prefix_conditioning/summarize.py")),
    Spec("additions/n2_ar_prefix_match", "ar", hours=3,
         commands=c_diff("additions/n2_ar_prefix_match/run.py"),
         summary=lambda cells, out: [_py("additions/n2_ar_prefix_match/summarize.py", "--cells", *map(str, cells),
                                         "--induction", paths.RESULTS / "additions" / "n2_ar_prefix_match" / "induction",
                                         "--exp4-root", paths.RESULTS / "controlled" / "exp4_ar_copy_circuit",
                                         "--out", out)]),
    Spec("additions/n2_ar_prefix_match/induction", "ar", hours=1, commands=c_n2f, single=True),
    Spec("additions/n3_copy_dose", "any", hours=8,
         commands=c_diff("additions/n3_copy_dose/run.py"),
         summary=s_cells("additions/n3_copy_dose/summarize.py")),
    Spec("additions/n4_no_training_baselines", "any", tasks=("next-item", "two-item"), gpu=False, hours=0.5,
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


def is_complete(d: Path) -> bool:
    """`run_cell` writes `complete` only after every command exited 0 (a killed job stays `running`)."""
    st = d / "status.txt"
    return st.exists() and st.read_text().startswith("complete")


def plan(which: str, stages=None) -> list[tuple[Spec, str | None]]:
    """(spec, ckpt) of every cell of the set, in spec order."""
    from sidlens.grid import registry
    reg = registry.load()
    names = [n for st in (stages or STAGES) for n in STAGES.get(st, [st])] if stages else list(SPECS)
    cells = []
    for name in dict.fromkeys(names):
        spec = SPECS[name]
        if spec.single:
            cells.append((spec, None))
            continue
        for ck in reg.names(states="all"):
            e = reg.entry(ck)
            st = grid.split_name(ck)[1]
            if spec.paradigm != "any" and e["paradigm"] != spec.paradigm:
                continue
            if e["task"] not in spec.tasks or (spec.depth3 and int(e["digits"]) != 3):
                continue
            if st != grid.BEST and not (which == "full" and spec.cheap):
                continue
            cells.append((spec, ck))
    return cells


def _ctx(spec: Spec, ckpt: str | None, pilot=False) -> Ctx:
    from sidlens.grid import registry
    entry = registry.load().entry(ckpt) if ckpt else {}
    out = cell_dir(spec, ckpt, pilot)
    # a pilot cell may run beside the real cell of the same map, and the AR chain
    # starts by deleting its scratch folder: the two must never share one
    scratch = (paths.SCRATCH / "pilot" if pilot else paths.SCRATCH) / spec.name / (
        grid.map_dir(ckpt) if ckpt else "single")
    return Ctx(ckpt or "", entry, grid.split_name(ckpt)[1] if ckpt else grid.BEST, out, scratch, pilot)


# ---------------------------------------------------------------- one cell --
def _source_hashes() -> dict:
    from sidlens.provenance.hashing import sha256_file
    files = sorted(p for root in (REPO / "src" / "sidlens", EXP) for p in root.rglob("*.py")
                   if "__pycache__" not in p.parts)
    return {str(p.relative_to(REPO)): sha256_file(p) for p in files}


def run_cell(spec: Spec, ckpt: str | None, pilot: bool = False) -> int:
    c = _ctx(spec, ckpt, pilot)
    if not pilot and is_complete(c.out):
        print(f"[skip] {spec.name} {ckpt}: complete", flush=True)
        return 0
    for parent in spec.after:
        if not is_complete(c.cell(parent)):
            raise RuntimeError(f"{spec.name} {ckpt}: parent cell {parent} is not complete (trap 3)")
    if c.out.exists():
        stamp = time.strftime("%Y%m%dT%H%M%S")
        c.out.rename(c.out.with_name(f"{c.out.name}.failed-{stamp}"))
    c.out.mkdir(parents=True)
    t0 = time.time()
    (c.out / "status.txt").write_text(f"running started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    (c.out / "cell.json").write_text(json.dumps({"spec": spec.name, "checkpoint": ckpt, "pilot": pilot,
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
    cells = [(s, ck) for s, ck in plan(which, stages) if not is_complete(cell_dir(s, ck))]
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
                run_cell(spec, ck)
            continue
        tf = tasks / f"{name.replace('/', '__')}-{time.strftime('%Y%m%dT%H%M%S')}.json"
        if not dry:                                   # trap 2: the submitted cell list, frozen
            tf.write_text(json.dumps({"spec": name, "set": which, "cells": cks}, indent=1))
        dep = ":".join(jobs[p] for p in spec.after if p in jobs) or None
        jid = _sbatch(spec, tf, len(cks), dep, dry)
        if jid:
            jobs[name] = jid
        print(f"{name}: {len(cks)} cells{' -> job ' + jid if jid else ''}")


# ---------------------------------------------------------------- reports --
def status(which: str) -> dict:
    out: dict = {}
    for s, ck in plan(which):
        d = cell_dir(s, ck)
        k = "complete" if is_complete(d) else ("failed" if (d / "status.txt").exists() else "missing")
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
        x.add_argument("--set", choices=("best", "full"), default="best")
        x.add_argument("--stage", nargs="*", help=f"stages {sorted(STAGES)} or spec names; default all")
        if name == "run":
            x.add_argument("--local", action="store_true", help="run here, one cell after another")
            x.add_argument("--dry-run", action="store_true")
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
            if not is_complete(cell_dir(s, ck)):
                d["todo"] += 1
                d["hours"] += s.hours
        for name, d in by.items():
            print(f"{name:55s} {d['cells']:5d} cells {d['todo']:5d} to run  ~{d['hours']:7.1f} "
                  f"{'GPU' if d['gpu'] else 'CPU'}-h")
        print(f"total ~{sum(d['hours'] for d in by.values() if d['gpu']):.0f} GPU-h, "
              f"~{sum(d['hours'] for d in by.values() if not d['gpu']):.0f} CPU-h (upper bounds per cell)")
        return 0
    if args.cmd == "run":
        submit(args.set, args.stage, args.local, args.dry_run)
        return 0
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
            rc |= run_cell(spec, ck)
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
