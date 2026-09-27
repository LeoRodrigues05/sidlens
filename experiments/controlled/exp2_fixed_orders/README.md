# All fixed reveal orders at SID depth three

Compare confidence with each of the six fixed orders on six frozen next-item
checkpoints (MQ/RQ-KMeans/RQ-VAE × width 128/512) and the same 6,297 users.
The primary comparator averages each user's outcomes across the six fixed
orders; it does not ensemble recommendation lists. See [protocol.md](protocol.md).

## Run with the archived numerical settings

**Use batch size 32 and decoder chunk size 1024.** The current CLI defaults
match these values; the commands below retain explicit flags for reproducibility.
The original matched-beam archive was submitted with these overrides to its
runner's 16/256 defaults.

```bash
# Primary complete six-cell sweep at beam 64.
sbatch scripts/controlled/fixed_orders.sbatch \
  --beam-widths 64 --batch-size 32 --decoder-chunk-size 1024 --validate

# Small implementation/timing pilot, including the beam 256 sensitivity.
sbatch --array=6 scripts/controlled/fixed_orders.sbatch \
  --limit 64 --beam-widths 64 256 \
  --batch-size 32 --decoder-chunk-size 1024 --validate

# Replace FULL_ARRAY and PILOT_ARRAY with the returned numeric IDs.
sbatch --dependency=afterok:FULL_ARRAY scripts/controlled/fixed_orders_summary.sbatch \
  FULL_ARRAY PILOT_ARRAY
```

New output directories are under
`$SIDLENS_WORK/derived/controlled/exp2_fixed_orders/<array-id>/cell-XX`.
The summary requires all six cells and all 6,297 users. It verifies file/source
hashes, recomputes ranks, exact-SID hit rates and NDCG from predictions, and
checks consistency of item bounds. A 2,000-draw paired-user bootstrap uses seed
20260915. Exports include per-cell and per-order effects, quality/cost tables,
a report, and figures.

The unchanged decoder supplies projected cross-attention, fixed score/tie rules,
full final-digit expansion and final legal-SID deduplication. This experiment
does not implement partial-state deduplication.

## Implementation audit on 2026-09-15

Pilot 194266 passed at both beams using 64 users. Initial full attempt 194270
used 16/256 defaults. Its RQ-KMeans 512 cell failed exact confidence prediction
reconstruction against the archive, despite the small pilot passing. That
attempt is preserved as incomplete; its remaining unnecessary queued cell and
dependent summary were cancelled.

The archived 18-cell run uniformly used 32/1024. Diagnostic 194288 repeated the
full 6,297-user RQ-KMeans 512 cell with those settings and passed exact baseline
prediction/rank/metric reconstruction, plus vendor and order-invariance checks.
The accepted primary sweep, array 194295, then completed uniformly with the
archived settings; summary job 194297 validated all six cells. See
[RESULTS.md](RESULTS.md). The accepted archives preserve the executed source and
explicit 32/1024 arguments. Current CLI defaults were aligned to 32/1024 after
completion; the search implementation is unchanged.
No decoder code or numerical acceptance tolerance was changed. A small
batch/chunk pilot does not establish full-cohort ranking invariance for arbitrary
batch settings; use the recorded settings for reproduction.

Beam 256 was an implementation pilot in this sprint; the complete primary sweep
uses beam 64. Both artifacts and incomplete attempts remain explicitly labelled.
