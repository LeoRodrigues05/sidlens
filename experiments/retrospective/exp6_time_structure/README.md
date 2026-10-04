# Retrospective exp6: time structure

This is a CPU audit. It puts the frozen review days back on every event
(`sidlens.data.timestamps`), then re-reads the per-row outputs of exp3, exp4,
exp6 and exp7 split by time.

- Protocol: [protocol.md](protocol.md), declared before any model output was
  split by time.
- Results: [RESULTS.md](RESULTS.md).

Run it from the repo root on a CPU node. It takes about 1 minute.

```bash
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/retrospective/time_structure.sbatch
```

Output: `$SIDLENS_WORK/derived/retrospective/exp6_time_structure/<job-id>/result/`,
a new directory per job:

- `estimates.csv` holds every estimand, with the bootstrap interval where one
  was declared.
- `time_rows__<variant>__<split>.parquet` and `time_rows__<diffusion ckpt>.parquet`
  are the per-row time tables: gap1, tie12, duplicate record, target type.
- `figures/` holds three figures:
  - `fig1_informativeness_position_time`
  - `fig2_copy_calibration`
  - `fig3_exp6_by_target_type`
- `inputs.json` holds the timestamp report and the sha256 of every result file
  read.

The inputs are fixed in `run.py` (`SOURCES`, `EXP6_PLAIN`). A missing run
raises, and there is no fallback to another run.
