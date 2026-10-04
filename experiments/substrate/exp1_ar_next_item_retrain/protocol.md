# Protocol: retrain the Qwen-AR next-item models of the width-128/512 grid

Declared 2026-10-04, before any training run of this protocol.

## Why

- Interventions need model weights. The August 2026 next-item sweep kept only
  the best checkpoint and deleted every other final checkpoint right after
  evaluation (`rm -rf final_checkpoint` in the sweep runner). Of the 27
  next-item models, only RQ-KMeans 3×128 (`next-item_best`) survives.
- The scores and saved top-50 lists of the deleted runs survive. They let us
  check whether a retrained model behaves like the deleted one.
- Goal: weights for every map that also has a DiffGRM next-item checkpoint
  (3 quantizers × depths 3–5 × widths 128 and 512 = 18 maps), so that AR and
  DiffGRM can be compared on 18 maps instead of 2.
- This is a re-creation with the recorded recipe, not a new experiment. A
  retrained model counts as a stand-in for the deleted one only if it passes
  the acceptance checks below.

## Configurations

All 18 maps `{rqkmeans, rqvae, MQ} × {3, 4, 5} × {128, 512}` on Industrial.

- **`rqkmeans_3codebook_128` is the calibration cell.** Its original weights
  survive, so retrained versus original measures the retraining noise
  directly.
- **`rqvae_4codebook_128` is retrained** because the surviving
  `oneoff_rqvae4cb128` is a different run: it reproduces 0 of the 3,681
  archived RQ-VAE 4×128 lists.
- **Width 256 is excluded.** It has no saved AR next-item lists, and DiffGRM
  next-item checkpoints exist for only two of its maps.

## Recipe

Identical to the 2026-08-18–20 sweep.

**Code:** `vendor/onediffrec/{sft.py, data.py, evaluate.py, split.py,
merge.py, calc.py}`. On 2026-10-04, `sft.py`, `data.py`, `evaluate.py` and
`calc.py` were checked byte-identical to upstream commit `eae9ecc`
(2026-08-17). No change touched `sft.py` or `data.py` between 2026-08-15 and
2026-09-01.

**Environment:** `OneDiffRec/.conda`, which `$SIDLENS_PYTHON` points to:
Python 3.11, torch 2.6.0+cu124, transformers 4.57.1, accelerate 1.10.1,
datasets 4.2.0. Each job records its pip freeze.

**Inputs:** the frozen copies. Each file is checked against the sha256 in
`manifests/provenance.base-20260826.json` before training, and the job stops
on any mismatch.
- **Base model:** `base_models/qwen2.5-1.5b-instruct`, copied from the file
  the sweep used (`model.safetensors` sha256 `dd924a11…`).
- **Data:** next-item train / valid / test CSVs and `item_meta/*.item.json`.
- **SID tables:** the index JSON and the decoding info file. These are the
  post-repair tables; all six archived RQ-KMeans next-item runs generated
  95–98 % of their top-10 SIDs in the repaired catalogue and none in the
  pre-repair one.

**Training:**
- **Command:** `torchrun --standalone` on `sft.py`, with `--sample -1
  --batch_size 1024 --micro_batch_size 8 --num_epochs 10 --learning_rate 3e-4
  --cutoff_len 512 --seed 42 --train_from_scratch False --freeze_LLM False
  --save_total_limit 2`.
- **Built into `sft.py`:** early stopping with patience 3 on validation loss,
  evaluation and saving every 5 % of steps, and the best model loaded at the
  end. wandb runs offline.
- **GPUs:** 4, as in the sweep. With G GPUs, `sft.py` sets gradient
  accumulation to 128 / G, so the global batch stays 1,024. The order in
  which examples reach the ranks then differs from the sweep. The GPU count
  is recorded per variant.

**Evaluation** (the archived procedure, `scripts/eval_variant.sh` at
`eae9ecc`):
1. `split.py` cuts the test CSV into the same 4 shards.
2. `evaluate.py` runs on each shard with seed 42, 50 beams, batch 8, at most
   256 new tokens and length penalty 0. With fewer than 4 GPUs the shards run
   in sequence. Each shard process seeds itself, so the output does not
   depend on the GPU count.
3. `merge.py` joins the shards, and `calc.py` scores them.

A plain beam-search evaluation (`use_model_defaults=False`, the exp6 code
path) is a separate, later step.

## Checkpoint retention

This is the failure the protocol prevents.

- **What is deleted:** nothing, except two kinds of duplicate.
  - The trainer's duplicate top-level `model.safetensors`, after its sha256
    matches the retained final weights. On a mismatch it is kept, and the
    mismatch is recorded.
  - Optimizer, scheduler and RNG states inside the trainer checkpoints, only
    after the final weights are hashed and moved.
- **After training:**
  - `final_checkpoint/` (the best model) is renamed to `<variant>/weights/final/`
    on the same filesystem, and its sha256 list is written.
  - The trainer's retained checkpoints (best and last) move to
    `<variant>/checkpoints/checkpoint-<step>/`, keeping their weights, config
    and `trainer_state.json`.
- **Evaluation** reads only `weights/final/`.
- **Resubmission:**
  - `status.json` is updated after every step, and a resubmission resumes at
    the first incomplete step.
  - A variant whose final weights are hashed is never retrained.
  - An interrupted training attempt resumes from its newest trainer
    checkpoint.

## Outputs

`$SIDLENS_WORK/runs/ar_next_item_retrain/<run-name>/<variant>/`:
- `manifest.json`: inputs with sha256, code sha256, environment, arguments,
  GPU count and type, timings, and the sha256 of the final weights
- `weights/final/` and `checkpoints/`
- `eval/`: `predictions.json` (merged, the archived format) and `calc.log`
- `logs/`
- `attempts/<job>/`
- `status.json`

Job provenance (`source.tar.gz`, git state, pip freeze) is in
`<run-name>/jobs/<job>_<task>/`.

## Acceptance

Declared now; computed by `check.py` on CPU after evaluation.

- **A1, calibration.** Compare retrained RQ-KMeans 3×128 with the original
  `next-item_best` archived predictions: Δ Hit@10, Δ NDCG@10 and top-1 SID
  agreement. This is the measured retraining noise.
- **A2, per variant.** Δ = retrained − recorded, for Hit@10 and NDCG@10.
  - Both are scored with `calc.py`'s rule (exact SID, else same title, else
    same item id) against the archived `predictions.json` of the same variant.
  - 95 % paired-user bootstrap intervals: 2,000 draws, seed 20261004, users
    from the test CSV rows.
  - Strict exact-SID Hit@10 is reported beside each.
- **Acceptance rule:** accept a variant if |Δ Hit@10| ≤ 1.5 points and
  |Δ NDCG@10| ≤ 1.0 point. For reference, two nominally identical archived
  RQ-VAE 3×256 runs differ by 1.07 points of Hit@10.
  - If A1 itself exceeds these limits, every variant is reported as "recipe
    reproduced, metrics differ". The limits are not changed afterwards.
- **A3, grid.** The Spearman correlation of Hit@10 across the 18 maps between
  retrained and recorded models. Reported; expected ≥ 0.7.
- **Variants that fail** stay in the run directory, marked not accepted.
  Analyses that depend on matching the archive exclude them and say so.
- **No intervention or representation result uses a retrained model before
  `check.py` has run.**

## Arm B: ZCR models

Added 2026-10-04, after the arm A pilot was submitted and before any training
had finished.

- **Maps:** RQ-KMeans 3×128, 4×128, 5×128, 3×512 and 4×512. These are the
  grid maps that have a ZCR table. RQ-KMeans 5×512 has none: it is a balanced
  table with 11 collisions. Priority goes to 3×128 and 4×128, which have the
  most collisions.
- **Inputs:** `build_zcr_inputs.py` rewrites the SID columns of the frozen
  CSVs, the index JSON and the decoding info file from the ZCR bundle
  (`onediffrec/handoff/zcr-release-v1`). It first checks four things:
  - the bundle's native table equals the frozen one;
  - ZCR kept every prefix;
  - all 3,105 SIDs are distinct;
  - every SID in every CSV row matches its item id under the native table.

  Titles, item ids and row order are unchanged, and the build checks this.
  For 3×128, 910 items change their last digit, which touches 75 % of training
  rows and 88 % of test rows.
- **Recipe:** identical to arm A (same command, hyperparameters, seed and
  evaluation), on the ZCR files. Variants are named `<map>__zcr`.
- **Acceptance:** no archived ZCR run exists, so A2 does not apply. The
  native retrained model of the same map is the comparison point, and a ZCR
  model is used only after its native counterpart passes A2.
- **The native-vs-ZCR comparison** (accuracy, collision credit, last-digit
  copying, how the answer forms across layers) is a separate experiment. Its
  protocol is declared before any native-vs-ZCR comparison is computed.

## What this does not establish

- That the retrained weights are identical to the deleted ones. Multi-GPU
  bf16 training is not deterministic, and the deleted weights cannot be
  compared directly.
- Seed variance. There is one retrain per map, and the calibration cell gives
  one measurement of retraining noise.
