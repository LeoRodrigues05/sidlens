# representation/exp5: what the DiffGRM encoder holds

Linear probes, TopK SAEs (with causal ablation of spread and identity latents)
and a description autoencoder in the style of natural language autoencoders,
on the encoder of all 20 next-item DiffGRM checkpoints. Protocol:
[protocol.md](protocol.md), declared before any forward pass. Plan:
`docs/plans/DIFFUSION_MECHANISMS_PLAN.md` (X1). Results: [RESULTS.md](RESULTS.md).

| File | Role |
|---|---|
| `common.py` | Fit/analysis histories, slot table, SAE census, matched control latents, description design (pure; tested in `tests/experiments/test_encoder_features_experiment.py`) |
| `run.py` | One checkpoint: gates V1–V5, probes (Part P), SAE fit + census + CS1/CS2 (Part S), description autoencoder (Part N), every splice/ablation scored at the fully masked state |
| `summarize.py` | All cells → `estimates.csv`, `census_shares.csv`, `report.md` (user bootstrap, 2,000 draws, seed 20261014) |

Run (from the repo root, `site.env` sourced; one job walks the cells):

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/diffusion_encoder_features.sbatch
SIDLENS_CELLS=6,14 sbatch ... --limit 600          # pilot
$SIDLENS_PYTHON experiments/representation/exp5_diffusion_encoder_features/summarize.py --run <run dir> --out <new dir>
```

GPU is strongly preferred: the probes are full-batch L-BFGS fits on ~90k rows
(about an hour per cell on 32 CPU cores, about a minute on an A100).

Outputs per cell (`$SIDLENS_WORK/derived/representation/exp5_diffusion_encoder_features/<job>/cell-XX/`):
`probes.csv`, `probe_correct.npz`, `test_slots.npz`, `sae/` (weights), `sae_history.json`,
`sae_fidelity.json`, `census.csv`, `conditions.csv.gz` (every condition × user at the fully masked
state: copy score, golden log-prob, top-1, KL per digit), `users.csv`, `nla.json`,
`nla_examples.json`, `inputs.json`, `validation.json`, `output.sha256`.
