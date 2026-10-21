# Running sidlens on the OneDiffRec grid (coauthor runbook)

Written 2026-10-21. This is what the coauthor runs on the cluster that holds
the finished OneDiffRec grid (`grid-v1`: two catalogs, AR and DiffGRM,
next-item and next-two, 27 native SID maps and their ZCR twins, with
intermediate weight states). The plan behind it is
`docs/plans/COPYING_GRID_PLAN.md`; the code is `src/sidlens/grid/` and
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
| `SLURM_PARTITION`, `SLURM_ACCOUNT`, `SLURM_QOS`, `SLURM_EXCLUDE` | GPU jobs (the names OneDiffRec's scripts use) |
| `CPU_PARTITION`, `CPU_QOS` | CPU jobs (default: the GPU partition) |
| `GPU_GRES` | default `gpu:1` |
| `MAX_GPUS`, `MAX_CPUS` | array tasks running at once |
| `CELLS_PER_TASK` | cells per array task (raise it if the queue caps jobs per user) |

## 4. Register the grid

```bash
bash scripts/grid/sidlens-grid.sh register
```

Discovers every run and weight state of `grid-v1`, hashes every weight file,
checks each against the run's own provenance (`provenance/{train,eval}`) and
its `sources/` snapshot against our vendored copy, and writes
`results_remote/registry/registry.json` plus `coverage.csv` (one row per run:
usable or not, and why). Runs that are not usable are skipped by everything
below and listed there.

## 5. Pilot (one GPU job, a few hours)

```bash
bash scripts/grid/sidlens-grid.sh pilot
```

Runs one short cell (64 rows) of every experiment on one configuration and
writes `results_remote/pilot/timings-*.json`. Send us that file and the
pilot's Slurm log before the full run: it tells us whether the measured cost
matches the plan and whether any experiment fails on your cluster.

## 6. Run

```bash
bash scripts/grid/run_best_state.sh      # every experiment, every configuration, best state
bash scripts/grid/run_full.sh            # the same plus the cheap set at every intermediate state
```

`run_full.sh` alone does everything; `run_best_state.sh` first is the safe
order. Both print the plan, then submit one Slurm array per experiment.
Finished cells are skipped, so either script can be rerun after failures or
time-outs (a failed cell is moved aside to `<state>.failed-<stamp>` and rerun).
Useful options: `--dry-run` (print the `sbatch` lines), and through
`sidlens-grid.sh run --stage cpu|ar|diffgrm|additions` a subset.

```bash
bash scripts/grid/sidlens-grid.sh status --set full
```

counts complete / failed / missing cells per experiment. A failed cell's
`pipeline.log` (in its results folder) has the command and the error; Slurm
logs are in `results_remote/provenance/slurm/`.

## 7. Summarize and send back

```bash
bash scripts/grid/sidlens-grid.sh summarize
bash scripts/grid/sidlens-grid.sh export
```

`export` writes `results_remote/MANIFEST.sha256` and a tarball in
`<sidlens work root>/exports/`. Send the tarball (or push it as agreed); we
check it against the manifest here.

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

At intermediate weight states (`run_full.sh` only): AR exp3 Part A, exp4 Part R
(test split), DiffGRM exp7.

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
