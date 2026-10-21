# exp4: what a DiffGRM recommendation looks like while it is being decoded

**Status (2026-10-13): run on all 20 checkpoints; 16 accepted, 4 stopped at
the archive check.** [RESULTS.md](RESULTS.md) has the provisional numbers and
the diagnosis of the four cells (one or two users of 6,297 with a tie broken
differently on other hardware). Accepting them needs an amendment to V6 in the
protocol. Figures and the per-user viewer are not built yet.

This is the observational first step of the DiffGRM step-level work
(`docs/plans/DIFFUSION_STEP_PATCHING_PLAN.md`) and item 3 of
`docs/paper/rewrite-20261004/RUN_PLAN.md`. For each of the 20 next-item
checkpoints and each of the 6,297 cohort users it records:

- the final lists at beam 1, 64 and 256, and per step how many kept paths are
  distinct partial SIDs and how many a catalogue SID still completes;
- two paths state by state (the greedy path and the best path of the
  first-ranked SID at beam 256): what is committed, the top codes at every
  masked digit, the current guess, a catalogue ranking of the state, and the
  same readout after every decoder block;
- all partial states of the target SID (best and greedy reveal order);
- regret: each digit of the decoded SID re-masked and re-scored;
- the trainer's one-shot confidence ranking, to tell which decode states
  training visited.

[protocol.md](protocol.md) was declared first. `common.trace_batch` does the
recording; `run.py` loads, gates (V1–V6) and writes `trace.npz`,
`inputs.json` and `validation.json`. The primitives are in
`src/sidlens/analysis/diffusion_lens.py` and the `observer` argument of
`matched_decode.decode`. Tests:
`tests/experiments/test_reveal_trace_experiment.py`,
`tests/analysis/test_diffusion_lens.py`.

```bash
source ~/.config/sidlens/site.env
# pilot, two matched cells, 64 users
sbatch --array=6,14 --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/diffusion_reveal_trace.sbatch --limit 64
# all 20 checkpoints
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/diffusion_reveal_trace.sbatch
```

Cells 0–17 are checked against controlled exp1's archive (V6). If that archive
is not on the cluster, the run stops unless `--allow-missing-archive` is
passed, and its results are then provisional.

`summarize.py --run <run> --out <new dir>` writes the declared statistics;
`--skip-failed` summarizes the complete cells and names the rest.
`archive_diff.py` (wrapper: `scripts/representation/diffusion_reveal_trace_archive_diff.sbatch`)
describes how a cell that stopped at V6 differs from the archive. The CIAI
queue takes four jobs per user, so the 20 cells go in blocks; the wrapper's
header shows how to keep them in one run directory.
