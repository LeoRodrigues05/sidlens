# exp4: prefix-matched copying in the AR recommenders

This is a confirmatory test, on the valid split, of exp3's post-hoc
observation. To choose digit d, does the model read digit d of a history item
whose first d digits match the prefix it has produced? It has three parts:

- **R:** replicate the stratified replacement effect.
- **K:** attention-edge knockout. Block single edges (readout(d) → a history
  token) in all, early or late layers.
- **H:** screen all 336 heads on half of the users, then test the top 5 against
  layer-matched random heads on the other half.

[protocol.md](protocol.md) was declared first. Results are in
[RESULTS.md](RESULTS.md).

| Piece | Where |
|---|---|
| Edge knockout (explicit per-layer masks; guards for no-op edges, self edges, a replaced mask, fire-once) | `src/sidlens/interventions/attention.py` |
| Per-head attention probabilities (observation only) | `src/sidlens/hooks/attention_probs.py` |
| Shared fixed-shape runner (patches + knockouts + multi-slot scoring) | `src/sidlens/interventions/runner.py` |
| Runner / summary | `run.py`, `summarize.py` (uses `sidlens.analysis.bootstrap`) |
| Wrapper | `scripts/controlled/ar_copy_circuit.sbatch` (array over the two next-item cells) |
| Tests | `tests/interventions/test_attention_knockout.py` |

```bash
source ~/.config/sidlens/site.env
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_copy_circuit.sbatch                          # valid, R+K+H
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_copy_circuit.sbatch --split test --parts R,K # secondary
```

Measured on an A100 in bf16: 30 min (RQ-KMeans) and 44 min (RQ-VAE) per cell
for the full valid run.
