# substrate/exp1: retrain the Qwen-AR next-item models

Recreates the 18 Qwen-AR next-item models of the width-128/512 grid that the
August sweep deleted after evaluation (arm A). It also trains ZCR versions of
five RQ-KMeans maps with the same recipe (arm B). `protocol.md` has the
recipe, the checkpoint-retention rules and the acceptance criteria, all
declared before any training finished.

| File | Role |
|---|---|
| `protocol.md` | Recipe, retention, acceptance (A1 calibration, A2 per variant, A3 grid), arm B |
| `run.py` | Per variant: preflight hashes → train (`vendor/onediffrec/sft.py`) → retain and hash the weights → archived-style evaluation → manifest. Resumes at the first incomplete step. |
| `build_zcr_inputs.py` | Arm B inputs: rewrites the SID columns of the frozen CSVs, the index and the info file from a ZCR bundle |
| `check.py` | Acceptance against the archived predictions (CPU) |
| `../../../scripts/substrate/ar_next_item_retrain.sbatch` | Array wrapper. Task t takes `run.ORDER[t::N]`, so 4 tasks fit the CIAI 4-task submit cap. |

## Run

From the repo root, after `source ~/.config/sidlens/site.env`:

```bash
# pilot: 64 samples, 1 epoch, first 64 test rows (job 294718, submitted 2026-10-04)
RUN_NAME=pilot-20261004 sbatch --array=0 --gres=gpu:1 --mem=64G --time=02:00:00 \
  --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/substrate/ar_next_item_retrain.sbatch \
  --smoke --eval-limit 64 --variants rqkmeans_3codebook_128

# arm A, all 18 maps, 4 GPUs per task (the sweep's geometry)
RUN_NAME=r20261004 sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/substrate/ar_next_item_retrain.sbatch
#   fewer GPUs per task keeps the global batch at 1,024:  --gres=gpu:2 --mem=96G
#   resubmitting the same RUN_NAME resumes; finished variants are skipped

# arm B inputs (CPU; already built for 3x128, 4x128, 5x128, 3x512, 4x512)
python experiments/substrate/exp1_ar_next_item_retrain/build_zcr_inputs.py --variant rqkmeans_3codebook_128 \
  --bundle /l/users/leo.rodrigues/onediffrec/handoff/zcr-release-v1/zcr/industrial/rqkmeans_3codebook_128 \
  --out $SIDLENS_WORK/derived/substrate/zcr_inputs/20261004
# arm B training (after the native model of the same map passes A2)
RUN_NAME=r20261004 sbatch --array=0-1 --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/substrate/ar_next_item_retrain.sbatch \
  --variants rqkmeans_3codebook_128__zcr,rqkmeans_4codebook_128__zcr \
  --zcr-inputs $SIDLENS_WORK/derived/substrate/zcr_inputs/20261004
#   (with an explicit --variants, every task gets the same list; use one task per list)

# acceptance (CPU)
$SIDLENS_PYTHON experiments/substrate/exp1_ar_next_item_retrain/check.py \
  --run-root $SIDLENS_WORK/runs/ar_next_item_retrain/r20261004 \
  --out $SIDLENS_WORK/derived/substrate/exp1_ar_next_item_retrain/summary-r20261004
```

## On another cluster

1. Clone this repo.
2. Pull the substrate. The `full` profile carries the base weights:
   `$PY -m sidlens.cli bundle pull --profile full` (see
   `docs/clusters/CROSS_CLUSTER.md`).
3. For arm B, also send `derived/substrate/zcr_inputs/20261004` with
   `--extra`.
4. Write `~/.config/sidlens/site.env` (partition, QoS, `SIDLENS_PYTHON`).
5. Submit the commands above.

The environment must match `protocol.md`: torch 2.6.0+cu124, transformers
4.57.1, accelerate 1.10.1, datasets 4.2.0. Each job records its pip freeze.

## Cost

Measured on the August sweep: training took 3,467–4,476 s per map on 4
A100-40GB GPUs. With 1 GPU, expect about 4×. Evaluation (4 shards × 3,681
rows, 50 beams) adds about 3 minutes on 4 GPUs. Each variant keeps about
9 GB: the final weights plus two trainer checkpoints without optimizer
states.
