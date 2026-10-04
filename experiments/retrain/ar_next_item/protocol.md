# Protocol: retraining the next-item Qwen-AR models

Declared 2026-10-04, before any retraining job ran.

## Purpose

The 2026-08 next-item sweep evaluated 27 Qwen-AR configurations. It ran with
a keep-best retention policy, so it deleted every final checkpoint except
RQ-KMeans 3×128 right after evaluation. Their scores and ranked lists survive;
their weights do not, and interventions need weights. This job rebuilds the
weights from the recorded recipe and frozen inputs. A rebuilt model is used
in analyses only if it reproduces its archived evaluation within tolerance.

## Configurations (18)

All 18 configurations at widths 128 and 512 (3 quantizers × depths 3, 4, 5 ×
2 widths). These are the configurations with archived AR lists and with
DiffGRM checkpoints, so after this job every one of them has weights for both
systems. Width 256 is excluded: it has neither archived AR next-item lists
nor DiffGRM next-item checkpoints (except MQ 5×256 and RQ-VAE 3×256).

| Role | Configurations | Why |
|---|---|---|
| control | RQ-KMeans 3×128 | Its original weights survive (`frozen/ckpt/ar/next-item_best`), so retraining variability can be measured directly against the same model, including on intervention estimates later |
| recipe match | RQ-VAE 4×128 | The surviving `oneoff_rqvae4cb128` is a different training run (archived Hit@10 13.1 % against 14.5 % for the one-off); the sweep's model is rebuilt |
| missing | the other 16 | Weights deleted |

`configs.tsv` lists them and assigns each to one of four array tasks.

## Recipe (fixed; taken from the record)

- **Code:** `vendor/onediffrec`. Its `sft.py`, `data.py`, `evaluate.py`,
  `calc.py` and `LogitProcessor.py` are byte-identical to OneDiffRec commit
  `eae9ecc` (checked 2026-10-04). That is the commit the frozen snapshot
  records, and neither training file changed between 2026-08-15 and
  2026-09-01. The sweep ran on 2026-08-18 to 20. The current OneDiffRec
  `scripts/ar/sft.py` has diverged and is not used.
- **Environment:** `OneDiffRec/.conda`, the interpreter recorded in the
  provenance manifest: Python 3.11, torch 2.6.0+cu124, transformers 4.57.1,
  numpy 2.4.6, datasets 4.2.0, accelerate 1.10.1.
- **Inputs, from `frozen/`:**
  - base model `qwen2.5-1.5b-instruct`;
  - per-configuration train, valid and test CSVs;
  - SID index JSON and decoding info file;
  - item metadata.

  Each file is hashed and must equal the provenance manifest. The manifest's
  `source` path must equal the path the archived run recorded in its
  `assets.json`.
- **Training command:** the sweep's own (`vendor/onediffrec/scripts/sweep_runner.sbatch`):

  ```
  torchrun --standalone --nproc_per_node=4 sft.py --base_model <base>
    --train_file <train> --eval_file <valid> --output_dir <dir> --sample -1
    --batch_size 1024 --micro_batch_size 8 --num_epochs 10 --learning_rate 3e-4
    --cutoff_len 512 --category Industrial_and_Scientific --seed 42
    --sid_index_path <index> --item_meta_path <item> --train_from_scratch False
    --freeze_LLM False --save_total_limit 2
  ```

  `sft.py` evaluates and saves every 5 % of the planned steps, stops after 3
  evaluations without improvement, and writes the best model by validation
  loss to `final_checkpoint`. wandb runs offline.
- **GPUs:** 4 A100-40GB on one node, as in the sweep. If fewer are used,
  `sft.py` raises gradient accumulation to keep the global batch at 1,024; the
  manifest records the GPU count.
- **Known differences from the original runs:** none intended. Multi-GPU bf16
  training is not bit-reproducible, so weights will differ.

## Checkpoints kept (nothing here is ever deleted)

This follows the README's run export contract:

| Path | Content |
|---|---|
| `weights/final/` | `final_checkpoint`: the best model by validation loss, with tokenizer; hashed and load-tested |
| `checkpoints/step-NNNNN/` | Model weights, config and `trainer_state.json` of the saves at 5 %, 10 %, 20 %, 40 %, 60 %, 80 % and 100 % of the planned steps that occur before early stopping. Hard-linked as each save completes, so the trainer's rotation cannot delete them. |
| `manifest.json` | The base model (initial weights) by hash; inputs, code, environment, training summary, export checks, evaluation, acceptance |

Trainer `checkpoint-*` directories hold optimizer state. They are deleted only
after all of these succeed:
- `weights/final` is hashed;
- it loads;
- its SID tokens equal the index's tokens;
- the manifest is written.

The top-level duplicate of the model is deleted at the same point, as the
sweep did.

## Steps and guards, per configuration

1. **Inputs.**
   - Every input file exists, and its hash and source path match.
   - Every row of the train, valid and test CSVs passes
     `ar_prompts.check_against_table`.
   - The test CSV passes `check_against_archive` (row-for-row equality with
     the archived predictions).
2. **Train** with the recipe above.
3. **Export** `final_checkpoint` to `weights/final`:
   - hash every file;
   - check that the SID tokens in `added_tokens.json` equal the index's
     token set, and their count equals the archived `component_tokens`;
   - load in bf16 and run one forward pass with finite logits, and
     embedding rows equal to the tokenizer length.
4. **Evaluate** with the archived pipeline, exactly as
   `vendor/onediffrec/scripts/eval_variant.sh`:
   - `split.py` into shards 0–3;
   - `evaluate.py` per shard (seed 42, 50 beams, batch 8, max 256 new
     tokens, length penalty 0);
   - `merge.py`, then `calc.py`.

   Four shards are always used, because shard composition fixes the sampling
   RNG streams.
5. **Accept** (below).

A configuration that fails a step is recorded as failed. The task continues
with its next configuration and exits non-zero at the end.

## Acceptance (declared before any outcome)

**Primary**, using the same scorer (`calc.py`) on the retrained and the
archived predictions:
- |ΔHR@10| ≤ 1.5 points;
- |ΔNDCG@10| ≤ 1.0 point.

A model inside both bounds is **accepted**. One outside is **flagged**: it is
kept but not used in analyses until reviewed. The tolerance is set by the
archive: two nominally identical Qwen runs (RQ-VAE 3×256) differ by 1.07
points of HR@10.

**Secondary** (descriptive, all configurations):
- best validation loss, best step and stopping step against the archived
  record;
- exact-SID HR@10;
- per-row agreement with the archived lists: identical top-1 SID, and top-10
  overlap.

**Control** (RQ-KMeans 3×128): the same comparisons between the retrained
model and the surviving original. Intervention estimates on the two models
are compared in a later analysis.

**Pilot** (one GPU, before any full run):
- (A) The original `next-item_best`, evaluated by this pipeline on the full
  test split, must reproduce the archived lists for at least 3,680 of 3,681
  rows (exp6 found 3,680 with SidLens's own decoder).
- (B) A smoke training of RQ-KMeans 3×128 (64 samples per dataset, 1 epoch,
  global batch 32) must complete steps 1–5. Snapshots must appear, and
  `weights/final` must load. Its scores are not judged.

## Outputs

`$SIDLENS_WORK/runs/ar_next_item_retrain/<run-group>/`:
- one directory per configuration (`nextitem__Industrial_and_Scientific__<q>__<D>cb__<W>/`)
  with `manifest.json`, `weights/`, `checkpoints/`, `inputs.json`, `logs/`,
  `eval/` and `status.txt`;
- `task-XX/` per array task, with the source archive, environment,
  requirement freezes and provenance stamp.

`pilot-<job>/` holds the pilot. A summary goes to `summary-<groups>/`.
Directories are never reused. `--resume` skips configurations whose manifest
says accepted or flagged, and refuses to touch partial ones.

## What this does not establish

- Seed variance. Each configuration is retrained once with the recorded seed,
  as originally.
- Equality of weights. Only behaviour within tolerance is checked; the control
  measures how far a retrained model is from the original.
