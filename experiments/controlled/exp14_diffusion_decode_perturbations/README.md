# controlled/exp14: perturbations held through DiffGRM decoding

Part B blocks every route to the most recent item (exp8's `both`) for the
whole beam search, against the same block on a control item, and reads SID
and item HR@10 by target relation (the DiffGRM counterpart of AR exp6).
Part O swaps the two most recent items, or shifts the whole history right by
1 or 5 slots. All 20 next-item checkpoints, beams 64 and 256. Protocol:
[protocol.md](protocol.md). Results: [RESULTS.md](RESULTS.md).

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/diffusion_decode_perturbations.sbatch
$SIDLENS_PYTHON experiments/controlled/exp14_diffusion_decode_perturbations/summarize.py --run <run dir> --out <new dir>
```

Outputs per cell: `decodes.npz` (top-10 per condition and beam), `users.csv.gz` (per user:
relation, eligibility, SID rank, item-hit bounds, top-1 first code, KL at the fully masked state),
`inputs.json`, `validation.json` (V1–V4), `output.sha256`.
