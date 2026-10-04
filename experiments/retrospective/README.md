# Retrospective experiment sprint (2026-09-06)

This directory implements the short, runtime-ranked experiment list recovered
from local Codex session `01a075d2-537c-7112-a524-125efde5584f` (the session
index labels it “Evaluate cluster migration”).  These numbers are local to that
planning sprint and are separate from the six top-level research experiments
described in the project README.

1. Historical diffusion confidence-order versus inferred seed-42 fixed-order
   audit.
2. Autoregressive per-digit and prefix error decomposition.
3. Collision-conditioned SID accuracy and item-rank bounds.
4. Next-two conditional-association audit (the planned dependence audit).
5. Diffusion confidence and digit-commitment trajectories (deferred until the
   retained-output analyses above are reviewed).
6. `exp6_time_structure/` (added 2026-09-30): time in the data (same-day
   bursts, ASIN tie order, duplicate records) and the exp3/exp4/exp6/exp7
   outputs split by review day. It has its own wrapper,
   `scripts/retrospective/time_structure.sbatch`, and writes a new job
   directory, unlike the fixed-directory sprint wrapper.
7. `exp7_collision_credit/` (added 2026-10-03): the exp6 copy-knockout
   predictions re-scored at item level (CCE) and split by target–history
   relation: same item, same-SID partner, near-duplicate variant, or new.
   It has its own wrapper, `scripts/retrospective/collision_credit.sbatch`,
   and writes a new job directory.

Experiments 1–4 consume frozen, retained outputs and are CPU-only.  Submit them
as a durable array pinned to the requested node:

```bash
mkdir -p slurm_logs
sbatch scripts/retrospective/retrospective_experiments.sbatch
```

The wrapper runs the frozen-substrate gate, records repository provenance and
source hashes, and fails if a required artifact or expected configuration is
missing.  It intentionally does not reserve `--gres=gpu:1`; Experiment 5 will
need the node's GPU for new model inference.

The first experiment is not a causal ablation.  Its retained runs differ in
beam width, use only one fixed “random” permutation, and select checkpoints on
the confidence-order validation score.  Its outputs therefore label the result
as a historical pipeline comparison and record the balanced-grid primary
analysis plus an all-20-run sensitivity.
