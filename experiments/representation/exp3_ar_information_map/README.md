# Representation exp3: an information map of the AR residual stream

This experiment asks what each layer of the AR recommender holds, at history
and readout positions, and from which layer the model's own answer can be
read out.

- **Self-forced captures.** The target slot holds the model's own
  trie-constrained greedy SID. Train and test splits, all 28 layers plus the
  final norm.
- **SAEs.** exp2's configuration at every second layer.
- **Census.** Every live latent is scored against fixed variables: format,
  the token's own code, its own item, the previous item, brand, the copy
  source, the model's answer, the golden target, prefix match, same-day.
- **Tuned lens.** At every readout.
- **SID decoders.** Small trie-constrained decoders that turn one residual
  vector into a whole SID: item identity, the previous item, and the answer
  from the first readout.

- Protocol: [protocol.md](protocol.md), declared before any capture or fit.
- Results: RESULTS.md, written after the runs.

Run it from the repo root on the CIAI cluster:

```bash
source ~/.config/sidlens/site.env
X=gpu-04,gpu-05,gpu-07,gpu-49,gpu-50,gpu-51,gpu-53,gpu-54,gpu-56
P=$(sbatch --parsable --exclude=$X scripts/representation/ar_information_map.sbatch pilot)
G=$(sbatch --parsable --exclude=$X --dependency=afterok:$P scripts/representation/ar_information_map.sbatch gpu)
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none --dependency=afterok:$G \
    scripts/representation/ar_information_map.sbatch census \
    --group $SIDLENS_WORK/derived/representation/exp3_ar_information_map/$G
python experiments/representation/exp3_ar_information_map/summarize.py \
    --group $SIDLENS_WORK/derived/representation/exp3_ar_information_map/$G --out <new dir>
```

The `gpu` stage runs the captures, SAEs and lenses in one allocation per cell
(about 3 h). Each train capture is about 45 GB.

Output: `$SIDLENS_WORK/derived/representation/exp3_ar_information_map/<array-id>/cell-XX/`

| Directory | Contents |
|---|---|
| `capture-test/`, `capture-train/` | activation stores, plus `self_greedy.csv` (answer, golden, per-digit log-probs) |
| `sae/` | SAE weights per layer, test latents, `fidelity.json` |
| `lens/` | `lens_tokens.parquet` (tuned and logit lens, every layer and readout); `decoder_tokens.parquet` and `decoder_summary.csv` |
| `census-<job>/result/` | `summary.csv` (label shares per layer × group); `latents.parquet` (every live latent's F1 per variable); `answer_beyond_copy.json` |
