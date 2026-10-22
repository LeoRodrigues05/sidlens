# Running sidlens on the OneDiffRec grid (coauthor runbook)

Written 2026-10-21; one-job runs and the paper set added 2026-10-22. This is
what the coauthor runs on the cluster that holds
the finished OneDiffRec grid (`grid-v1`: two catalogs, AR and DiffGRM,
next-item and next-two, 27 native SID maps and their ZCR twins, with
intermediate weight states). The plans behind it are
`docs/plans/COPYING_GRID_PLAN.md` and `docs/plans/GRID_PAPER_SET.md`; the code
is `src/sidlens/grid/` and
`scripts/grid/`.

Nothing here writes into the OneDiffRec work root. Every result goes to
`$SIDLENS_WORK/results_remote/`, which comes back to us whole.

## 1. Get the code

Clone (or unpack) sidlens anywhere on the cluster. No commits are needed on
that side.

## 2. Get the two small inputs

They are not in git: `external/labels/` (Amazon-2018 categories, brands and
prices of both catalogs, 36 MB) and `derived/grid_inputs/near_duplicates/`
(near-duplicate item pairs of both catalogs from our structure exp4, 2 MB;
used for target classes). Unpack the inputs tarball into the sidlens work root
chosen in step 3:

```bash
tar -xzf sidlens-grid-inputs-20261021.tar.gz -C <sidlens work root>
```

Each file is hash-checked when it is read (`labels.*.manifest.json`,
`grid_inputs/near_duplicates/source.json`).

## 3. Set up (login node, once)

```bash
bash scripts/grid/sidlens-grid.sh setup --onediffrec-work /path/to/onediffrec-handoff-v1/work
```

This

- builds `<sidlens work root>/venv` on top of the Python that OneDiffRec's own
  runs recorded (`cell.json` → `python`), so the torch and transformers that
  produced the saved test lists are the ones sidlens uses; `--python` overrides;
- writes `~/.config/sidlens/grid.env` (paths, plus empty Slurm settings);
- checks the vendored OneDiffRec files (`verify --profile grid`) and runs the
  CPU test suite.

The sidlens work root defaults to a `sidlens_work/` folder beside the
OneDiffRec work root (`--sidlens-work` to change it). Then fill in the Slurm
lines of `~/.config/sidlens/grid.env`:

| Variable | Meaning |
|---|---|
| `SLURM_PARTITION`, `SLURM_ACCOUNT`, `SLURM_QOS`, `SLURM_EXCLUDE` | the GPU partition and its settings (the names OneDiffRec's scripts use) |
| `BATCH_NODES` | nodes for the job (default 1); 2 for 16 GPUs on 8-GPU nodes |
| `BATCH_GPUS` | GPUs per node (default 8) |
| `GPU_TYPE` | a GPU type, if your `--gres` needs one (e.g. `h200`) |
| `BATCH_CPUS`, `BATCH_MEM` | cores and memory for the job (defaults 64 and `0`, all of the node's memory; set a size such as `1000G` if the cluster refuses `0`) |
| `BATCH_TIME` | the longest time your partition allows (default `2-00:00:00`) |
| `BATCH_CPU_WORKERS` | CPU cells run beside the GPU cells, per node (default 4; 8 on a 96-core node) |
| `BATCH_MARGIN_MIN` | minutes before the job's end at which it stops cells and queues the next job (default 15) |

If `grid.env` was written before 2026-10-22, it lacks the `BATCH_*` lines; add
them by hand (or rerun `setup`, which backs up the old file).

## 4. Register the grid

```bash
bash scripts/grid/sidlens-grid.sh register
```

Discovers every run and weight state of `grid-v1`, hashes every weight file,
checks each against the run's own provenance (`provenance/{train,eval}`) and
its `sources/` snapshot against our vendored copy, and writes
`results_remote/registry/registry.json` plus `coverage.csv` (one row per run:
usable or not, and why). Runs that are not usable are skipped by everything
below and listed there. `run_paper.sh` registers the grid itself if no
registry exists.

## 5. Run: one job

```bash
bash scripts/grid/run_quick.sh --dry-run      # the plan, and exactly one sbatch line
bash scripts/grid/run_quick.sh                # the quick set: 9-11 h on 16 H200s
bash scripts/grid/run_paper.sh                # afterwards: the rest of the paper set
```

Each script queues **one** Slurm job, on `BATCH_NODES` nodes. Inside it, every
cell of the set (`docs/plans/GRID_PAPER_SET.md`) runs as its own process: one
cell per GPU at a time and `BATCH_CPU_WORKERS` CPU cells beside them on each
node. With several nodes, the job splits the cells between them and runs one
worker per node as a job step (`srun`); nothing else is queued. The quick set
is part of the paper set, so `run_paper.sh` skips what the quick set finished. The job starts with
one cell of every experiment, so a broken experiment shows up in its first
hour. When its time runs out it stops the running cells (marked `interrupted`,
not failed), queues its successor and ends; there is never more than one job
in the queue, and a second `run_paper.sh` is refused while one is queued. A
cell that fails twice is given up on and listed. When nothing is left, the job
writes the summaries and the export tarball (step 6).

Progress, updated after every cell:

- `results_remote/batch/progress-<set>-<job>.json` (one node) or
  `progress-<set>-<job>-node<k>.json` (one per node): per experiment, cells
  done, failed and left, median minutes per cell, and GPU-hours left. **Send
  us these files after the first two or three hours**: they are the cost
  estimate.
- `results_remote/provenance/slurm/sidlens-<set>-<job>.out`: the job's log.
- `bash scripts/grid/sidlens-grid.sh status --set <set>`: complete, failed and
  missing cells per experiment.
- A failed cell's `pipeline.log` (in its results folder) has the command and
  the error; `results_remote/batch/logs/` holds each cell's console output.

To stop: `scancel <job>`. To resume: run the same script again; finished
cells are skipped. If the job could not queue its successor (for example,
`sbatch` does not work on compute nodes), its log says so: run the script
again.

Other sets, each also one job: `run_case_study.sh` (optional, when we ask:
every experiment on the native 3x256 and 4x256 maps of each quantizer, plus
the intermediate weight states), `run_best_state.sh` and `run_full.sh` (every
experiment on every map; far larger). `sidlens-grid.sh run --arrays` keeps the
old one-array-per-experiment mode; do not use it on a queue that limits jobs.
`sidlens-grid.sh pilot` (64 rows of every experiment, one GPU) still exists
but is not needed: the first hours of the one job give the same information.

## 6. Send back

The job writes `results_remote/MANIFEST.sha256` and a tarball in
`<sidlens work root>/exports/` when every cell is done. Send the tarball (or
push it as agreed); we check it against the manifest here. To export
earlier, by hand:

```bash
bash scripts/grid/sidlens-grid.sh summarize
bash scripts/grid/sidlens-grid.sh export
```

## What runs

Per cell, everything lands in
`results_remote/<group>/<experiment>/<dataset>/<task>/<paradigm>/<map>/<state>/`.

| Group | Experiments | Runs |
|---|---|---|
| controlled | exp3 (A,B), exp4 (R,K,H; test and valid), exp6, exp9, exp10 | AR next-item |
| controlled | exp5 | AR next-two |
| controlled | exp1, exp2 (depth 3), exp7, exp8, exp12, exp13, exp14 | DiffGRM next-item |
| representation | exp1-exp3 (one chain per run: captures in scratch, deleted after) | AR next-item |
| representation | exp4, exp5 | DiffGRM next-item |
| retrospective | exp2, exp3, exp4 (next-two), exp6 on the saved test lists; exp7 on exp6 | every run (exp7: AR next-item) |
| additions | N1 prefix conditioning, N3 dose-response | next-item, both paradigms |
| additions | N2 prefix matching (+ base-model induction scores, once) | AR next-item |
| additions | N4 no-training baselines | every run |

That table is the full set (`run_best_state.sh`). The paper set runs each
experiment on every map, on the core maps or on the native core maps only
(`docs/plans/GRID_PAPER_SET.md`). At intermediate weight states
(`run_full.sh`, `run_case_study.sh`): AR exp3 Part A, exp4 Part R (test split),
DiffGRM exp7.

Industrial uses all 6,297 test users; Office a fixed sample of 6,297 users
(sha256 of a recorded seed and the user id) for every experiment that runs a
model, and full cohorts for the saved-list analyses.

## Checks every cell makes before it counts

- V1: the rows, SID table, vocabulary and prompts are the run's own, token for
  token against OneDiffRec's own code.
- V2: re-decoding experiments first reproduce the run's saved test lists with
  the run's own decoder settings (AR: plain constrained beam search plus
  raw-vocabulary rescoring; DiffGRM: left-to-right catalog-constrained beam
  search). DiffGRM experiments that score states also check that sidlens's
  scorer equals the evaluator's per-digit forward.
- V3: no-op interventions are bit-exact.

A cell that fails a check stops and is marked failed; it never writes a
result that looks complete.
