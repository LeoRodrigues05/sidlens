# representation/exp2: sparse autoencoders on the AR residual stream

This experiment trains TopK SAEs (`sidlens.sae.topk`) on the AR next-item
models' train-split residual captures, at layers 12, 16, 20 and 24. It then
reads the test-split captures in that basis:

- **Q1:** the copy latents at the first-digit readout
- **Q2:** prefix-match latents at the digit-d readouts
- **Q3:** target semantics beyond the history's content
- **Q4:** whether the readout represents a same-day burst

Candidate latents are then ablated causally, error-preserving, through
`FixedShapeRunner`.

- Protocol: [protocol.md](protocol.md), declared before any SAE was trained.
- Results: [RESULTS.md](RESULTS.md).

## Stages

There is one wrapper, `scripts/representation/ar_sae.sbatch <stage>`, with one
array task per cell (0 = `next-item_best`, 1 = `oneoff_rqvae4cb128`):

| Stage | Script | Hardware | Reads | Writes |
|---|---|---|---|---|
| captures | `scripts/activations/ar_capture.sbatch --split train --layers 12,16,20,24` | GPU, ~3 min | frozen checkpoint | `derived/ar_capture/287160`, `287161` |
| train | `train.py` | GPU, ~15 min | train + test stores | `sae/L*/`, `test_latents_L*.safetensors`, `fidelity.json` |
| analyze | `analyze.py` | CPU, ~85 min | test store, SAE latents, embeddings, timestamps | `observational.csv`, `selection.json`, `positions.parquet`, `q1_latent_table_L*.parquet` |
| causal | `causal.py` | GPU | SAEs, `selection.json` | `records.parquet`, `controls.parquet`, `validation.json` |
| summary | `summarize.py` | CPU, minutes | all three | `causal_estimates.csv`, `observational_estimates.csv`, `fidelity.csv`, `report.md`, `figures/` |

Run everything from the repo root on the CIAI cluster, after sourcing
`~/.config/sidlens/site.env`.

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/ar_sae.sbatch train
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none --cpus-per-task=16 scripts/representation/ar_sae.sbatch analyze \
    --sae-run $SIDLENS_WORK/derived/representation/exp2_ar_sae/train-<id>
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/ar_sae.sbatch causal \
    --sae-run .../train-<id> --selection .../analyze-<id>
python experiments/representation/exp2_ar_sae/summarize.py --train <train group> --analysis <analyze group> \
    --causal <causal group> --out $SIDLENS_WORK/derived/representation/exp2_ar_sae/summary-<ids>
```

The two cells can come from different job ids. `--sae-run` and `--selection`
accept either a run group, from which each task takes its own `cell-XX`, or a
cell directory directly.

`summarize.py` expects `cell-00` and `cell-01` side by side. When the cells
ran as separate jobs, pass directories that hold symlinks to both.
