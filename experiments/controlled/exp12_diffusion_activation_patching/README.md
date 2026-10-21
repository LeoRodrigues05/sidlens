# controlled/exp12: paired-run activation patching in DiffGRM

Two runs of one checkpoint that differ in one input: the most recent history
item replaced (Part H, exp7's controls) or one revealed digit set to a
sibling's code (Part S). Encoder slots, decoder residuals and per-block
cross-attention outputs are patched between the runs in both directions;
Part S also cuts the self-attention edge from the predicted digit to the
changed one. All 20 next-item checkpoints. Protocol: [protocol.md](protocol.md).
Results: [RESULTS.md](RESULTS.md).

```bash
SIDLENS_CELLS=0-4 sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none --cpus-per-task=32 \
    scripts/controlled/diffusion_activation_patching.sbatch --device cpu      # ~10-20 min per cell on CPU
$SIDLENS_PYTHON experiments/controlled/exp12_diffusion_activation_patching/summarize.py --run <run dirs> --out <new dir>
```

Outputs per cell: `pairs_{H,S}.csv` (pair metadata, clean/corrupted golden log-prob and KL),
`sites_{H,S}.npz` (KL, golden and copy log-prob per pair × site × direction; Part S knockouts),
`missing.csv`, `inputs.json`, `validation.json` (V1–V5), `output.sha256`.
