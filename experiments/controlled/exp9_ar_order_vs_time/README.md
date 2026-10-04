# exp9: does the AR copy follow position or time?

This experiment swaps the two most recent history items and splits rows by
the items' review days (`sidlens.data.timestamps`):

- **U:** the most recent item is on a later day than the one before it.
- **T_later:** both are on the same day, and the target is on a later day.
- **T_same:** both, and the target, are on the same day.

Within a day the order is ASIN order, which the timestamps cannot justify. So
a swap in T_later leaves every day unchanged, while a swap in U makes the order
false.

- Protocol: [protocol.md](protocol.md), declared before any forward.
- Results: [RESULTS.md](RESULTS.md).

## Parts

- **TF (primary, test and valid).** A teacher-forced 2×2 of clean vs swapped
  order on the same rows, through `FixedShapeRunner`. It gives the position
  effect A, the item effect B, the top-1 flip rate, Δ log-prob, and the
  tie-averaged predictor Δ_mix. Rows whose two items have identical SIDs are
  swapped anyway. That swap is token-identical, and must be bit-exact.
- **D (secondary, test).** exp6's plain beam search on clean and swapped
  prompts, in exp6's batches. It gives HR@10, and the share of rows whose
  top-10 list or top-1 SID changes. The clean arm must reproduce exp6's
  `predictions_B_plain`.

## Running it

Run from the repo root on the CIAI cluster. Source `~/.config/sidlens/site.env`
first.

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_order_vs_time.sbatch --limit 64                  # pilot
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_order_vs_time.sbatch                             # test, TF + D
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_order_vs_time.sbatch --split valid --parts TF    # valid
python experiments/controlled/exp9_ar_order_vs_time/summarize.py --cells <run>/cell-00 <run>/cell-01 --out <run>/../summary-<id>
```

Outputs go to `$SIDLENS_WORK/derived/controlled/exp9_ar_order_vs_time/<array-id>/cell-XX/`:

- `rows.parquet`: the strata, fixed before the model is loaded
- `part_tf.parquet` and `part_d.parquet`: the records of the two parts
- `controls.parquet`
- `validation.json`

The test cell takes about 20 minutes on one A100, most of it Part D; valid TF
takes about 1 minute.
