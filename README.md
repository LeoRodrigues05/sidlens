# SidLens

SidLens is the mechanistic-interpretability arm of OneDiffRec. OneDiffRec
trains generative recommenders that emit multi-digit Semantic IDs (SIDs);
SidLens freezes the exact artifacts behind those runs and asks what the models
compute internally, when they compute it, and whether those computations
causally affect the recommendation.

The full research plan is in the
[project brief](data/LeoRodrigues_ProjectProposal_GenRec_MechInterpAnalysis.pdf).

## Project scope

The comparison spans four axes:

| Axis | Conditions |
|---|---|
| Recommendation paradigm | Autoregressive Qwen2.5-1.5B SFT and masked diffusion (DiffGRM) |
| SID construction | RQ-VAE, RQ-KMeans, and parallel MQ/PSE |
| SID scale and order | 3, 4, or 5 digits; codebook sizes 128, 256, or 512; coarse-to-fine, fine-to-coarse, and random AR order |
| Task | Next-item and next-two-item recommendation |

The primary data lineage is Amazon Reviews 2018. Current model analysis focuses
on `Industrial_and_Scientific` (3,105 items); an `Office` catalogue (17,696
items) is also frozen. Some upstream DiffGRM paths say `AmazonReviews2014`, but
the artifacts used here belong to the Amazon-2018 lineage. The legacy
`OneDiffRec/data/Amazon` catalogue is a different item set and must never be
joined to these SIDs.

### Research questions

- **RQ1 — Recommendation paradigm:** Keeping the SID fixed, do AR and
  diffusion models represent and causally use the same evidence from a user's
  history?
- **RQ2 — SID organization:** What semantic, collaborative, and item-identity
  information is organized by each digit under RQ-VAE, RQ-KMeans, and MQ/PSE?
- **RQ3 — Interactions, order, and scale:** How do paradigm, quantizer, digit
  order, SID depth, codebook width, and collisions interact?
- **RQ4 — Next-two computation:** Is the second item generated conditionally
  from the first, or are both items represented and refined as a joint plan?

The main methods are hidden-state and attention hooks, raw per-digit logits,
probes, sparse autoencoders (SAEs), paired crosscoders, activation patching,
feature ablation, and controlled retraining.

## Experiment plan and status

“Completed” below means that an output exists in the current `derived/` tree;
it does not mean that every model-side or causal question has been answered.

| Experiment | Purpose | Current state |
|---|---|---|
| 1. SID semantic atlas | Measure what every digit adds and which attributes, terms, and collaborative neighborhoods it organizes | Item/static half completed for all 27 Industrial variants. Model probes, SAEs, and interventions remain |
| 2. RQ geometry and refinement | Measure cluster radius, within-cluster variance, chance-corrected refinement, digit dependence, utilization, and collisions | Static geometry and refinement completed for all 27 variants. Model-side patching remains |
| 3. Paradigm × quantizer | Compare matched AR and diffusion models across SID constructions | Hook layer smoke-tested, but only two matched next-item checkpoint cells survive |
| 4. AR digit order | Compare coarse-to-fine, exact fine-to-coarse, random, and output-only reorderings under matched conditions | Requires new controlled AR fine-tunes and checkpoint retention |
| 5. Scale and collisions | Explain why greater SID depth or width can stop helping | Static measurements and historical metrics exist; checkpoint-level and causal analysis remain |
| 6. Next-two mechanism | Determine when the second item is represented and whether the two item states causally interact | Requires a matched AR/joint-block-diffusion cell. Retained diffusion runs use two-pass inference, not the intended joint-block condition |

The current implementation includes SID/data loaders, AR and DiffGRM model
loaders, observation-only hooks, the Experiment 1 item atlas, and Experiment 2
geometry/refinement. `probes/`, `sae/`, `interventions/`, and `viz/` are still
scaffolds.

## Why this is a separate repository

Three properties of the upstream setup make in-place analysis unsafe:

1. **Run code was mutable.** Every diffusion run directory symlinked model code
   back into one live tree, so a later edit could change what a checkpoint was
   assumed to contain.
2. **The training baseline was a dirty tree.** The frozen upstream state had 72
   modified or untracked paths at commit `eae9ecc`; a commit or submodule alone
   cannot reproduce it.
3. **The sweep removed artifacts needed for interpretability.** The AR sweep
   deleted non-best weights after scoring. Twenty-six completed AR cells retain
   metrics but no weights and must be retrained.

SidLens therefore separates tracked analysis code from an immutable,
hash-verified snapshot of the exact SIDs, data, model artifacts, logs, and
results.

## Data contract

A Git clone contains code, vendored source, tests, manifests, and the project
brief. It does **not** contain the bulk research substrate in `$SIDLENS_WORK`.
The authoritative frozen inventory is
[`src/sidlens/provenance/spec.py`](src/sidlens/provenance/spec.py).

The following sections are ordered by dependency: later artifacts are only
meaningful when the earlier identity, model, and example data are pinned.

### 1. Semantic-ID assignments and identity mappings

These are priority-zero artifacts because every comparison depends on the
exact item-to-SID assignment used during training.

Required:

- Index JSONs for
  `{RQ-VAE, RQ-KMeans, MQ} × {3, 4, 5 digits} × {128, 256, 512 codes}`.
- DiffGRM `.sem_ids` files.
- Item-to-token and token-to-item mappings.
- Next-item and two-item constrained-decoding information files.
- ASIN, OneDiffRec integer-ID, and DiffGRM-ID mappings.
- The pre-repair RQ-KMeans `.mispacked` files for runs trained before the
  bit-unpacking repair.

Copy these files byte-for-byte. RQ-VAE and MQ assignments cannot currently be
regenerated from the available code and quantizer weights; RQ-KMeans
regeneration is not deterministic. `tokens2item` is also lossy when items
collide on a full SID, so item-level decoding must use the one-to-many mapping.

### 2. Model weights

Here, **weights** means the parameter tensors needed to initialize or run a
model. They are distinct from the resumable training checkpoints in the next
section.

Required for the planned experiments:

- A final AR `model.safetensors` for every comparison cell.
- A final DiffGRM `pytorch_model.bin` for every comparison cell.
- The exact Qwen2.5-1.5B-Instruct base weights used to initialize AR training.
- Quantizer codebooks, centroids, learned transforms, and encoder weights when
  available.
- The exact text-embedding model and revision if item embeddings ever need to
  be rebuilt.

The current frozen snapshot contains:

- Three AR weight sets:
  - `next-item_best`: RQ-KMeans, 3 digits × 128.
  - `oneoff_rqvae4cb128`: RQ-VAE, 4 digits × 128. This is a one-off validation
    anchor, not the checkpoint behind the published sweep row.
  - `two-item_best`: MQ, 4 digits × 256.
- Twenty next-item diffusion weight files.
- Three next-two diffusion weight files; two are marked trained and one is
  marked incomplete.
- Only two matched next-item AR/diffusion cells: RQ-KMeans 3 × 128 and RQ-VAE
  4 × 128.
- No matched next-two AR/diffusion pair.

The audit finds 25 planned cells with weights, 26 completed AR cells whose
weights were deleted, and 84 cells that were never trained. It also finds the
one-off RQ-VAE AR weight as an intentional registry orphan. Run:

```bash
"${SIDLENS_WORK}/venv/bin/python" scripts/audit_checkpoints.py --missing-only
```

Exact SID tables are enough for the current static atlas and decoder-facing
experiments. They do not replace quantizer weights for studying RQ-VAE's own
latent space or reproducing SID construction. The RQ-VAE quantizer checkpoints
and encoder, fitted RQ-KMeans centroid/index state, and MQ/PSE implementation or
parameters are not in the frozen substrate.

### 3. Training checkpoints and run state

A **training checkpoint** is a resumable and time-indexed record, not just a
final weight file. Retain the following for every new or recovered training
run:

- Intermediate model weights at the initial, best, final, and agreed landmark
  steps.
- Optimizer, learning-rate scheduler, gradient-scaler, trainer, sampler, and
  random-number-generator state when continuation or exact replay matters.
- Current step and epoch, trainer state, and the selected best-checkpoint
  marker.
- Model and tokenizer configuration, AR tokenizer files, and
  `added_tokens.json`.
- Training arguments and every CLI or environment override.
- Random seeds, initialization identifier, and starting-weight hash.
- Exact train/validation/test split hashes and exact SID-table hash.
- Source commit, dirty-tree patch, dependency snapshot, and submitted job
  script.
- Training/evaluation logs, metrics, predictions, and full transcript.

Intermediate checkpoints are particularly valuable here: they let us ask when
a semantic or causal feature emerges during training. Optimizer state is not
needed for endpoint-only interpretability, but it is required for exact resume.

DiffGRM's retained weight files are bare state dictionaries. Their model
configuration is reconstructed from the run log, transcript, and vendored
SLURM launcher. `n_head` cannot be recovered from tensor shapes; a wrong value
can load successfully while changing the computation. Those text/config
artifacts are therefore checkpoint data, not optional diagnostics.

The current snapshot contains inference weights, not complete resumable
checkpoints: optimizer, RNG, sampler/data-order state, and developmental weight
trajectories do not survive. The old transfer archive has partial trainer state
for one AR run, but no optimizer state or checkpoint series.

For Experiment 4, every order condition must share starting weights, examples,
SID assignments, training budget, and decoding budget.

### 4. Training and evaluation examples

Required:

- Original timestamped interaction sequences.
- Canonical train, validation, and test interaction files.
- Per-variant next-item SFT CSVs.
- Per-variant next-two-item SFT CSVs.
- DiffGRM sequence and ID-mapping files.
- Preprocessing and split configuration, including seeds.

The snapshot keeps train, validation, and test in separate directories for all
27 Industrial SID variants in both tasks. Do not flatten those directories:
the splits contain same-named files and would silently overwrite one another.
Order experiments must derive new examples from the canonical inputs without
modifying them.

### 5. Item embeddings

The atlas, geometry, refinement, and scaling analyses require:

- Pre-quantization Qwen3-Embedding-4B item embeddings.
- The row-to-ASIN/item-ID mapping used to align the matrices.

The frozen FP16 matrices are mean-pooled over `[title, description]`, have no
PCA or L2 normalization, and have shapes `(3105, 2560)` and `(17696, 2560)`.
The `sentence-t5-base_pca256` text in some DiffGRM filenames is an upstream
naming artifact and does not describe these external embeddings.

### 6. Tokenizers, model configuration, and executable code

Required:

- The AR tokenizer and its exact SID-token vocabulary.
- Complete resolved model configuration, including values supplied only by a
  launch script.
- Frozen upstream model, dataset, evaluation, and constrained-decoding code.
- Source commit, dirty diff, Python/package versions, CUDA/PyTorch build, seeds,
  and hardware description.

Do not infer AR token IDs as `block_start + code`: added SID tokens were sorted
as strings, and unused codes were omitted. Always read `added_tokens.json`.

### 7. Item metadata and interpretability labels

Training-lineage data includes item titles, descriptions, brands, reviews,
summaries, user/item mappings, and interactions.

Interpretability-only external evidence includes the Amazon-2018 category
hierarchy, price, rank, brand, `also_buy`, `also_view`, feature, and details
fields. The raw Amazon metadata and derived label table must remain under
`$SIDLENS_WORK/external`, separate from `frozen/`: they were acquired after
training and are evidence used to interpret representations, not data the
models saw during training.

The current validated targets are category levels, store, price/rank, brand,
and co-purchase. `feature` and `details` retain raw text, but color, size,
material, and style still need an extraction and validation step before they
can be used as probe labels.

Keep both:

- `external/amazon2018/meta_Industrial_and_Scientific.json.gz`, needed to
  rebuild and validate the join.
- `external/labels/`, containing the derived table and its source manifest.

### 8. Evaluation and run records

Retain:

- Recall, NDCG, loss curves, best step, and stopping step.
- Ranked prediction beams and raw per-digit scores.
- Valid-SID and collision-aware item-level metrics.
- Diffusion reveal order, masks, confidence, denoising step, and random seed.
- Runtime, peak memory, FLOPs, energy, and inference-cost measurements when
  available.
- Scheduler job ID, allocation, node/GPU model, command, stdout, and stderr.

Historical metrics and predictions are validation targets for retrained AR
cells and for checking that a reconstructed DiffGRM configuration behaves like
the original.

### 9. Activation, probe, SAE, and intervention artifacts

Every activation capture must record:

- Example, user-history, target, item, and SID identifiers.
- Snapshot and checkpoint IDs and hashes.
- Architecture, layer, token/digit position, and diffusion step.
- Hidden state, attention output, and raw per-digit logits requested by the
  experiment.
- Mask/reveal state, denoising order, and random seed.
- Tensor dtype, capture filters, patch configuration, and active code changes.

Probe datasets must preserve train/test separation. SAE and crosscoder weights
must carry their activation-corpus specification, architecture, sparsity
objective, optimizer configuration, and seeds. Intervention outputs must retain
the clean and patched example pair, the intervention target, and output/logit
delta.

Activation storage will dominate the move once model-side work begins. The
current full residual-stream capture is roughly 550 MB in FP32 for one batch of
64 histories of length 50. An unfiltered training-set capture can approach
250 GB per checkpoint in FP32, or 125 GB in FP16, before SAE caches and
intervention variants. Begin with 0.5–1 TB of scratch, capture selected
layers/positions, write chunked arrays, and promote only analysis-ready outputs
to durable storage.

### 10. Provenance

Required:

- Frozen upstream bytes in `vendor/`.
- `manifests/CURRENT` and the snapshot manifest.
- The diffusion checkpoint registry.
- Upstream commit, dirty-tree diff, and artifact hashes.
- Job manifests, logs, configuration files, and dependency snapshots.

`vendor/` and `frozen/` are immutable. Model changes belong under
`src/sidlens/`, must carry the target/source hash and numerical-impact flag, and
must be stamped into every affected result. The intended registered-monkeypatch
mechanism is not implemented yet; numerical patches are not provenance-complete
until that registry exists.

## Data required by experiment

| Experiment | Minimum inputs | Important missing input |
|---|---|---|
| 1. Semantic atlas | Exact SIDs, item embeddings, item/ASIN maps, raw Amazon metadata, derived labels | Model weights and captures for the probe/causal half |
| 2. Geometry/refinement | Exact SIDs, embeddings, identity maps | RQ-VAE encoder latent/checkpoint for native-latent geometry |
| 3. Paradigm × quantizer | Matched AR/diffusion final weights, tokenizers/configs, identical histories/splits, logits and hidden states | Most AR grid weights |
| 4. AR order | Base AR weights, exact SIDs/splits, reordered examples, every run/checkpoint bundle | All controlled fine-tunes |
| 5. Scale/collisions | Complete depth × width final-weight grid, collision groups, metrics/predictions, captures | Most AR grid weights and causal captures |
| 6. Next-two | Matched AR and true joint-block diffusion weights, exact two-item splits, attention masks, denoising traces, seeds | A matched pair and the intended joint-block implementation/run |

## Run export contract

Upstream training must export a self-contained bundle **before** any sweep
cleanup removes checkpoints. A recommended layout is:

```text
runs/<run-id>/
  manifest.json                 # task, cell, seed, hashes, code and environment
  weights/final/                # inference-ready final model + tokenizer
  checkpoints/{initial,best,final,step-*}/
  config/                       # resolved model/data/decode config + job script
  data/                         # split and SID references with hashes, not copies
  logs/                         # trainer/evaluator transcripts
  eval/                         # metrics, predictions, digit scores
  traces/                       # optional masks, reveal order, activation index
```

An export is accepted only after the files are copied to durable storage,
hashed, load-tested, matched to the intended SID table, and registered. Cleanup
must depend on successful export rather than simply on successful evaluation.

## Repository and storage layout

```text
sidlens/
  vendor/          byte-frozen upstream code; never edit
  src/sidlens/     data, model, hook, analysis, and provenance code
  experiments/     executable experiment entry points
  scripts/         SLURM wrappers, audits, and data preparation
  manifests/       checked-in snapshot and checkpoint registries
  tests/           consistency and regression tests

$SIDLENS_WORK/
  frozen/          immutable, hash-verified training substrate
  external/        post-training metadata and labels with their own manifests
  derived/         atlases, geometry, activations, probes, SAEs, and figures
  runs/            new self-contained training exports
  cache/           disposable caches
  venv/            cluster-specific environment; rebuild, do not transfer
```

`SIDLENS_WORK` selects the bulk root. `ONEDIFFREC_REPO` and
`ONEDIFFREC_WORK` select an upstream source only when creating a new snapshot;
normal analysis must read the frozen copy.

## Verification and current commands

Every job must verify its substrate before doing work:

```bash
"${SIDLENS_WORK}/venv/bin/python" -m sidlens.cli verify --strict
"${SIDLENS_WORK}/venv/bin/python" -m pytest -q
```

Other provenance commands:

```bash
sidlens freeze --snapshot-id <id>   # create a substrate once
sidlens verify --strict             # hash-check frozen/ and vendor/
sidlens show --sections             # summarize the selected snapshot
```

After a successful freeze, make both immutable:

```bash
chmod -R a-w "${SIDLENS_WORK}/frozen" vendor
```

## Moving to another cluster

There is currently no Git remote configured. The move therefore has two
independent parts: publish or bundle the Git history, then copy the non-Git
data.

### 1. Commit and publish the Git repository

From the current cluster, replace the example URL with an empty repository you
can reach from both clusters:

```bash
cd /home/leo.rodrigues/GenRecSys/sidlens/sidlens
git status --short
git add -- README.md \
  pyproject.toml \
  scripts/audit_checkpoints.py \
  src/sidlens/models/diffusion.py \
  src/sidlens/registry/diffusion.py \
  tests/test_diffusion_registry.py
git diff --cached --check
git diff --cached
git commit -m "Document migration and make registry paths portable"

REMOTE_URL='git@your-git-host:your-group/sidlens.git'
git remote add origin "${REMOTE_URL}"
git push -u origin master
```

If `origin` is added before these commands are run, replace `git remote add`
with `git remote set-url origin "${REMOTE_URL}"`.

On the target cluster:

```bash
REMOTE_URL='git@your-git-host:your-group/sidlens.git'
git clone "${REMOTE_URL}" <TARGET_CODE_ROOT>/sidlens
cd <TARGET_CODE_ROOT>/sidlens
git switch master
```

If neither cluster can reach a shared Git host, create an offline bundle after
committing; a bundle contains Git history but still no bulk data:

```bash
git bundle create ../sidlens.bundle --all
scp ../sidlens.bundle <USER>@<TARGET_HOST>:<TRANSFER_ROOT>/
```

Then on the target:

```bash
git clone <TRANSFER_ROOT>/sidlens.bundle <TARGET_CODE_ROOT>/sidlens
```

### 2. Copy the required non-Git data

Current migration inventory:

| Source | Approximate size | Action |
|---|---:|---|
| `$SIDLENS_WORK/frozen/` | 14.75 GB logical | Required; copy in full |
| `$SIDLENS_WORK/derived/` | 57 MB logical | Recommended; preserves completed atlas, geometry/refinement, and hook reports |
| `$SIDLENS_WORK/external/amazon2018/` and `external/labels/` | Under 100 MB | Required for the labelled Experiment 1 workflow |
| `$SIDLENS_WORK/runs/` | Currently negligible | Optional now; required once it contains new exports |
| `$SIDLENS_WORK/cache/` | Negligible/regenerable | Do not copy |
| `$SIDLENS_WORK/env/` or `venv/` | About 6.6 GB logical | Do not copy; rebuild for the target CUDA stack |
| `$SIDLENS_WORK/external/onediffrec_data/` | About 5.65 GB logical | Optional raw-source fallback; current SidLens code does not read it |
| `data/OneDiffRec_data.tar.gz` | 1.67 GB | Optional ignored source archive; redundant once canonical inputs are frozen |

If the target does not already expose byte-equivalent upstream training data,
transfer either the extracted `external/onediffrec_data/` tree or its compact
archive, not both. The archive SHA-256 is
`c3ab0bedf84bd52cfc79f281e358518792d01f96e1a58ab68bbdd56c8374ce89`.
Never extract that archive over `frozen/`: five authoritative Industrial
RQ-KMeans tables contain later repairs and differ from the archived source.

The old `sidlens-transfer-20260827.zip` is not a current code source and has no
additional model weights. It mostly duplicates Git plus `frozen/`. Its small
`setup/` and `extras/` trees can be retained as optional forensic provenance;
the full 13.46 GB archive does not need to be copied when using Git and
`rsync`.

Example direct transfer from the current cluster, without deleting anything at
the destination:

```bash
ssh <USER>@<TARGET_HOST> \
  'mkdir -p <TARGET_WORK_ROOT>/sidlens/external'

rsync -rtlpHS --partial --info=progress2 \
  /l/users/leo.rodrigues/sidlens/frozen \
  /l/users/leo.rodrigues/sidlens/derived \
  <USER>@<TARGET_HOST>:<TARGET_WORK_ROOT>/sidlens/

rsync -rtlpHS --partial --info=progress2 \
  /l/users/leo.rodrigues/sidlens/external/amazon2018 \
  /l/users/leo.rodrigues/sidlens/external/labels \
  <USER>@<TARGET_HOST>:<TARGET_WORK_ROOT>/sidlens/external/
```

The immutable tree has its own hash manifest, so `rsync` transport checks are
not the final acceptance test.

### 3. Configure and verify the target

Set these variables in the login shell and in every SLURM job:

```bash
export SIDLENS_REPO=<TARGET_CODE_ROOT>/sidlens
export SIDLENS_WORK=<TARGET_WORK_ROOT>/sidlens
```

Rebuild a Python 3.11 environment against the target cluster's supported CUDA
and GPU stack. The current `pyproject.toml` names runtime dependencies but is
not a reproducibility lock, so capture a lock or container specification before
accepting new experiment runs. Do not copy the old environment directory:
virtual environments embed paths, and its PyTorch/CUDA build may not match the
new nodes.

```bash
python3.11 -m venv "${SIDLENS_WORK}/venv"
"${SIDLENS_WORK}/venv/bin/python" -m pip install --upgrade pip
# Install the target cluster's supported PyTorch/CUDA build first if required.
"${SIDLENS_WORK}/venv/bin/python" -m pip install -e "${SIDLENS_REPO}"
"${SIDLENS_WORK}/venv/bin/python" -m pip install pytest
```

Then restore immutability and run the acceptance checks:

```bash
chmod -R a-w "${SIDLENS_WORK}/frozen" "${SIDLENS_REPO}/vendor"
"${SIDLENS_WORK}/venv/bin/python" -m sidlens.cli verify --strict

printf '%s  %s\n' \
  '2d0fde57bfa563108c78076d5975515124ab6175d6a1380cc990cddd7f6f20b8' \
  "${SIDLENS_WORK}/external/amazon2018/meta_Industrial_and_Scientific.json.gz" \
  | sha256sum -c -
"${SIDLENS_WORK}/venv/bin/python" \
  "${SIDLENS_REPO}/scripts/build_labels.py" \
  --category Industrial_and_Scientific
printf '%s  %s\n' \
  'b9b25e9e98a3d3a173341176ea5e5500ab7c7186616c87ad532c04e7be9b5a50' \
  "${SIDLENS_WORK}/external/labels/Industrial_and_Scientific.labels.json" \
  | sha256sum -c -

"${SIDLENS_WORK}/venv/bin/python" -m pytest "${SIDLENS_REPO}/tests" -q
"${SIDLENS_WORK}/venv/bin/python" \
  "${SIDLENS_REPO}/scripts/audit_checkpoints.py" --missing-only
```

The label build writes only to `external/`. Re-running it on the target keeps
the derived label table deterministic while updating its manifest paths to the
new cluster.

The checkpoint audit intentionally exits with status 1 while planned cells are
missing; use its summary as an inventory, not as evidence that the copied
snapshot failed verification.

Two portability notes apply on the target:

1. `manifests/registry.diffusion.json` deliberately retains the absolute paths
   of the source snapshot as provenance. The registry loader rebases its
   checkpoint, SID, log, and transcript paths at runtime onto
   `$SIDLENS_WORK/frozen`; set `SIDLENS_WORK` before starting Python or a job.
   Do not hand-edit or regenerate the tracked manifest merely to relocate it.
2. Existing SLURM wrappers contain site-specific defaults, and vendored
   training launchers preserve old account, partition, QoS, and path values as
   provenance. Export `SIDLENS_REPO`/`SIDLENS_WORK` and create new wrappers for
   the target scheduler; do not edit `vendor/`.

Finally, do not point analyses directly at mutable training-run directories on
the new cluster. Export the required run bundle, hash it, and either register
it under `runs/` or promote a reviewed snapshot before using it as experimental
evidence.
