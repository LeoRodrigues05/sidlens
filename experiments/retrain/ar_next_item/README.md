# Retraining the next-item Qwen-AR models

> **Status 2026-10-04: not used.** The retraining runs through
> `experiments/substrate/exp1_ar_next_item_retrain`, which uses the same recipe
> and was written in parallel by another copy of the same session. This copy
> was never run beyond a pilot cancelled within a minute. Delete it, or first
> port three features into substrate/exp1: per-step snapshots (`Snapshotter`),
> the GPU count read from the allocation, and the pilot check that
> `next-item_best` reproduces the archived lists. Its `--dry-run` passed for
> all 18 maps.

The 2026-08 next-item sweep deleted every final checkpoint except RQ-KMeans
3×128 right after evaluation (keep-best retention). This job rebuilds the
18 configurations at widths 128 and 512 from the recorded recipe and frozen
inputs. It keeps every checkpoint the analyses need, and accepts a model only
if it reproduces its archived evaluation within tolerance. Read
[protocol.md](protocol.md) first; it was written before any run.

| File | Role |
|---|---|
| `protocol.md` | Purpose, recipe, retention, guards, acceptance (declared 2026-10-04) |
| `configs.tsv` | The 18 configurations, their role (control, recipe match, missing) and array task |
| `run.py` | Per configuration: inputs → train → export → evaluate → accept; `--dry-run` checks inputs only |
| `summarize.py` | One table across configurations from the manifests |
| `../../../scripts/retrain/ar_next_item.sbatch` | SLURM wrapper (pilot on 1 GPU; full on 4 GPUs per task) |

Commands (repo root, with `~/.config/sidlens/site.env` sourced):

```bash
PY=$SIDLENS_WORK/venv/bin/python
$PY experiments/retrain/ar_next_item/run.py --mode full --root /tmp/x --dry-run   # inputs only, CPU
sbatch --array=0 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
       --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/retrain/ar_next_item.sbatch --mode pilot
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/retrain/ar_next_item.sbatch --mode full
$PY experiments/retrain/ar_next_item/summarize.py --roots $SIDLENS_WORK/runs/ar_next_item_retrain/<group> \
    --out $SIDLENS_WORK/runs/ar_next_item_retrain/summary-<group>
```

Each configuration ends as `accepted`, `flagged` (kept, not used until
reviewed) or `failed`. Accepted weights are at
`$SIDLENS_WORK/runs/ar_next_item_retrain/<group>/<variant>/weights/final/`, with
training snapshots in `checkpoints/step-*/`.

Costs (from the sweep's records): about 70 minutes of training on 4 A100s
per configuration, as most stop early near epoch 4, plus about 3 minutes of
evaluation. That is 5–7 hours per array task. Each configuration keeps
about 3 GB of final weights and about 3 GB per snapshot.
