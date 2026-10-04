# exp7: what the DiffGRM networks use from history

This is the diffusion half of paper RQ4, run on the two AR-matched cells
(RQ-KMeans 3×128 and RQ-VAE 4×128) over the 6,297-user diffusion cohort. It
mirrors exp3 and exp4 in three parts:

- **A:** history-item replacement by recency and shared prefix.
- **C:** prefix-matched copying.
- **K:** cross-attention knockouts from digit positions to history-item slots.

Digit scores are taken at the fully masked state and at golden-prefix states.

[protocol.md](protocol.md) was declared first. Results are in
[RESULTS.md](RESULTS.md). The primitives are in
`src/sidlens/interventions/diffusion.py` (`score_states`, the
`cross_attention_masks` knockout and `knockout_mask`). Tests are in
`tests/interventions/test_diffusion_knockout.py`.

```bash
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none --mem=64G scripts/controlled/diffusion_history_use.sbatch --device cpu
```

It takes 5–8 minutes per cell on 32 CPU cores.
