# Retrospective exp7: collision credit in copying

A CPU re-scoring of the exp6 copy-knockout predictions. It scores each
prediction list at item level (CCE uniform tie-break) beside exact SID, at
HR@10 and HR@1. It splits every effect by how the target relates to the
history:

- same item
- a different item with the same SID (collision partner)
- a near-duplicate variant with a different SID (structure exp3's
  definition)
- new

- Protocol: [protocol.md](protocol.md), with amendment A1 (HR@1) marked post
  hoc.
- Results: [RESULTS.md](RESULTS.md).

Run it from the repo root on a CPU node; it takes under a minute.

```bash
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/retrospective/collision_credit.sbatch
```

Output: `$SIDLENS_WORK/derived/retrospective/exp7_collision_credit/<job-id>/result/`:

- `estimates.csv`: every estimand, with paired user-bootstrap intervals
- `report.md`
- `rows__<variant>.parquet`: the row classes
- `scored__<variant>__<decoder>__<condition>.parquet`: per-row SID and item
  hits, and the variant-copy flag
- `validation.json`: the reproduction of exp6's `C_all − B`
- `inputs.json`: the sha256 of every file read

The inputs are fixed in `run.py` (`ARCHIVED`, `PLAIN`, `ND_PAIRS`). A missing
input raises, and there is no fallback to another run.
