# exp5: next-two AR, is item 2 computed from item 1?

This covers RQ4 on `two-item_best` (MQ 4×256), with teacher forcing on "sid1 |||
sid2". It has two parts:

- **Part A:** clamp item 1 to catalogue controls that share m = 0…3 leading
  digits, and replace the most recent history item. Measure item 2's
  per-digit scores.
- **Part B:** residual patching over the item-1, separator and item-2 tokens ×
  28 layers.

[protocol.md](protocol.md) was declared first. Results are in
[RESULTS.md](RESULTS.md). Target-slot clamps use
`interventions.history.replace_target_slot` / `ControlPool.draw_target`.

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_next_two_conditioning.sbatch
$SIDLENS_PYTHON experiments/controlled/exp5_ar_next_two_conditioning/summarize.py --cell <run>/cell-00 --out <summary>
```

Measured: 19 min on an A100 in bf16.
