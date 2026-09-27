# SidLens: orientation for Claude

SidLens is the mechanistic-interpretability companion to OneDiffRec. OneDiffRec
trains generative recommenders that emit multi-digit Semantic IDs (SIDs). SidLens
asks what those models compute, when, and whether it matters causally. The
analysis code lives here. Model training happens upstream, on another cluster.
`README.md` holds the full data and provenance contract; the project brief is
`docs/plans/LeoRodrigues_ProjectProposal_GenRec_MechInterpAnalysis.pdf` (RQ1–RQ4 and
Experiments 1–6).

## Two roots and one Python

| What | Where |
|---|---|
| Code (this repo) | `/home/leo.rodrigues/GenRecSys/sidlens/sidlens` |
| Bulk data, `$SIDLENS_WORK` | `/l/users/leo.rodrigues/sidlens`, the default in `src/sidlens/paths.py` |
| Python 3.11 (torch 2.6+cu124) | `/l/users/leo.rodrigues/sidlens/venv/bin/python`, not on `PATH`. Call it explicitly. `sidlens` is installed editable. |

Under `$SIDLENS_WORK`:

| Directory | Contents |
|---|---|
| `frozen/` | Read-only, hash-pinned substrate (about 13 GB): SIDs, data, checkpoints, archived results. Pinned by `manifests/provenance.base-20260826.json`. |
| `external/` | Evidence acquired after training (labels, raw metadata, the extracted tarball) |
| `derived/` | Every result that SidLens produces |
| `runs/` | Empty. Intended for exported upstream runs. |
| `bundles/` | Transfer manifests from `sidlens bundle create/pull` (see `docs/clusters/CROSS_CLUSTER.md`) |
| `env/` | The environment; `venv` is a symlink to it |

Resolve every path through `sidlens.paths`, and never hardcode `/l/users/...`
in new code. **Never write to `frozen/` or `vendor/`.** `vendor/` holds
byte-frozen upstream code (`diffgrm`, `diffgrm_new`, `onediffrec`), and
`verify` fails if anything in it changes.

## Where the data came from

- **`frozen/`** was copied on 2026-08-26 by `sidlens freeze` from the upstream
  OneDiffRec trees on the old cluster. It is the only substrate that analysis
  code reads.
- **`data/OneDiffRec_data.tar.gz`** (gitignored, 1.67 GB) was extracted to
  `$SIDLENS_WORK/external/onediffrec_data/data/{Amazon18,two-item}`. **No SidLens
  code reads it.** It is mostly redundant with `frozen/`, with two exceptions:
  - It is the only local copy of the **Office SFT CSVs**: next-item
    `Amazon18/{train,valid,test}/Office_*.csv` and next-two
    `two-item/{train,valid,test}/Office_*.csv`.
  - Five Industrial RQ-KMeans index tables in it predate the 2026-08-20
    bit-unpacking repair and differ from `frozen/`. Never prefer them, and
    never extract the tarball over `frozen/`.
- **`external/amazon2018/` and `external/labels/`** hold post-training
  interpretability labels built by `scripts/data/build_labels.py`: category levels,
  store, brand, price and rank, and `also_buy`/`also_view`. These exist for
  **Industrial only**; Office metadata has not been downloaded.

## Substrate facts that cause silent bugs

- **Catalogues.** `Industrial_and_Scientific` has 3,105 items and is the primary
  catalogue for all analysis. `Office` has 17,696 items; its SIDs, embeddings
  and interactions are frozen, but it has no checkpoints, no `.sem_ids` and no
  labels. The legacy `OneDiffRec/data/Amazon` is a different item set; never
  join it.
- **SID variants.** Each catalogue has 27: `{rqvae, rqkmeans, MQ}` × depth
  `{3,4,5}` × width `{128,256,512}`, named like `rqkmeans_3codebook_128`. The
  ASIN-keyed `.sem_ids` file is ground truth; load it with
  `sidlens.data.sids.SidTable.load(name)`.
- **Collisions.** `tokens2item_*.pkl` is lossy (last writer wins). Decode
  through `SidTable.cb2items`, which maps one SID to many items. SID-level HR
  hides item ambiguity, so report item-level bounds
  (`sidlens.analysis.collisions`).
- **Nesting.** Within one width, the RQ-KMeans depths are truncations of a
  single fit. The exception is `rqkmeans_5codebook_512`, a separate
  post-repair fit. RQ-VAE and MQ depths are independent fits. MQ digits are
  parallel partitions, so an MQ prefix is an intersection of partitions, not a
  step down a hierarchy.
- **AR token IDs.** Read them from `added_tokens.json` via
  `sidlens.models.ar.load_vocab`. IDs are string-sorted (`<a_100>` comes before
  `<a_10>`), and unused codes are omitted; for example, digit 0 of
  `rqvae_4cb_128` has only 39 codes. Never compute `base + code`.
- **AR prompts.** Rebuild them with `sidlens.data.ar_prompts`; its ids equal
  upstream's dataset classes row for row. Test evaluation (`EvalSidDataset`,
  which produced every archived prediction) and training (`SidSFTDataset`)
  use **different wording**, so `template="eval"|"sft"` is required. Digit d is
  scored at `Encoded.predict_pos(slot, d)`: digit 0 at the ":\n" of
  "### Response:\n", digit d > 0 at the previous target digit under teacher
  forcing. Left padding needs explicit `position_ids` (`collate`). The test-CSV
  row index is the only join key to the archived AR predictions (verified
  for all 28 files).
- **DiffGRM token IDs.** `id = 3 + digit*K + code`, in code order. Histories and
  labels use raw codes, with `PAD = -1`.
- **DiffGRM runtime:**
  - `DIFF_GRM.forward(return_loss=False)` returns after the encoder. The decoder
    runs only through `forward_decoder_only`, so a hook on a decoder block fed by
    `forward()` captures nothing.
  - `models/diffusion.digit_logits` uses `use_cache=False`, which drops the
    learned cross-attention K/V projections, so it is **unvalidated**. For
    logits or decoding, use `analysis/matched_decode.py`, which applies the
    projected cross-attention cache and is validated against the archive.
  - Registry-built models cannot `generate()` because the decode config is
    missing.
- **Diffusion checkpoints** are bare state dicts. Their config comes from
  `manifests/registry.diffusion.json`, whose paths are rebased onto
  `$SIDLENS_WORK` at runtime; never edit those paths. `n_head=4` cannot be
  checked from tensor shapes; only behavioural reproduction of the recorded
  metrics confirms it.
- **Exact reproduction** of archived decoding requires batch size 32 and decoder
  chunk size 1024. Other settings have failed exact reconstruction on full
  cohorts.
- **DiffGRM training facts that bound what decoding results mean** (verified
  in `vendor/diffgrm/.../model.py`): with `masking_strategy=guided`, at each
  training step each example is trained along ONE reveal order (the model's
  own confidence ranking on the fully masked input, least-confident masked
  first, `refresh=False`), so fixed reveal orders mostly visit partial states
  training rarely covers;
  `guided_steps = min(d, 4)`, so **depth-5 models never see the fully masked
  SID during training** although every decode starts there (next-item code
  only: the next-two `diffgrm_new/.../model.py` caps at `n_target_digits`
  instead, but the joint-block `model_block.py` keeps the cap of 4); and the ranking
  pass uses `use_cache=False`, i.e. the unprojected cross-attention fallback,
  a different network from the one trained. The vendored confidence decoder
  also fills the last digit greedily (no branching) and pads short lists by
  repeating the last legal SID; `matched_decode` does neither.
- **Beam search keeps paths, not states.** Both the vendor decoder and
  `matched_decode` let several reveal orders reach the same partial SID and
  keep all of them; duplicates are 57 / 79 / 91 % of the final beam at depths
  3 / 4 / 5. Legality is checked only after the last step (no trie for
  diffusion; the AR side uses a trie at every step).

## Models and cohorts

- **AR.** Qwen2.5-1.5B SFT: 28 layers × 1536, `model.layers.N`. Three weight
  sets survive:

  | Checkpoint | SID variant |
  |---|---|
  | `next-item_best` | `rqkmeans_3codebook_128` |
  | `oneoff_rqvae4cb128` | `rqvae_4codebook_128` |
  | `two-item_best` | `MQ_4codebook_256` |

  Upstream deleted the weights for the other 26 AR runs; only their metrics
  and predictions remain.
- **Diffusion (DiffGRM).** One encoder layer and four decoder layers × 256,
  four heads, history length 50, `encoder_blocks.N` and `decoder_blocks.N`
  (each with `self_attn` and `cross_attn`).
  - next1: 20 trained checkpoints covering the full 3 × 3 × {128, 512} grid,
    plus MQ-5cb-256 and rqvae-3cb-256.
  - next2: rqvae 3cb-256 and 4cb-256 are trained; 5cb-256 is incomplete.
    Retained next2 evaluation runs in two passes. It is not a joint block.
- **Matched AR↔diffusion cells** exist only for next-item:
  `rqkmeans_3codebook_128` and `rqvae_4codebook_128`
  (`sidlens.models.ar.MATCHED_CELLS`). No next-two pair is matched.
- **Test cohorts.**
  - Diffusion next-item: 6,297 users, leave-last-out, loaded with
    `sidlens.data.diffusion_eval.load_eval_cohort`.
  - Historical AR next-item: 3,681 events from 1,606 users.
  - The AR and diffusion cohorts differ. **Never join them by row number.**

## Code map

| Location | Contents |
|---|---|
| `src/sidlens/data/` | `sids` (SID tables, nesting), `diffusion_eval` (test cohort), `ar_prompts` (exact AR input ids, per-position role/item/digit map, archive and SID-table checks, `collate`), `labels`/`meta` (Amazon labels), `embeddings` (Qwen3-Embedding-4B, `(3105, 2560)` FP16, not normalized) |
| `src/sidlens/models/` | `ar` (vocab and loader), `diffusion` (registry → model, shape checks) |
| `src/sidlens/hooks/` | Observation-only forward hooks: `residual_points`, `attention_points`, `capture`, `run_capture`. Attention hooks see module outputs, not per-head weights. Hooks must never change an activation; interventions belong in `interventions/`. `hooks/store.py`: the activation store (safetensors shards + parquet rows + manifest written last; no pickle), the one format for captures from any cluster. |
| `src/sidlens/analysis/` | `atlas`, `attribute` (conditional AMI with within-parent null), `naming`, `geometry`, `refinement`, `collisions`, `matched_decode` (the validated diffusion beam search: `policy="confidence"` or `"fixed"` with an `order`), `semantic_mapping` (CLI: `build` / `query`) |
| `src/sidlens/viz/semantic_explorer.py` | Standalone offline HTML explorer |
| `src/sidlens/viz/style.py` | The one figure style: quantizer colours (RQ-KMeans blue, RQ-VAE orange, MQ aqua, fixed everywhere), one-hue ramps for width/depth, ink tokens, `save()` writes PDF+PNG. Palettes were validated for CVD/contrast; keep using them. |
| `experiments/structure/semantic_mapping/figures.py` | The eight structure-layer figures (icicle, group sizes, geometry, purity-vs-null, per-digit information, t-SNE zoom, one-digit maps, collisions); each writes a CSV twin. t-SNE is cached in the output dir. |
| `src/sidlens/probes/`, `sae/`, `interventions/` | **Empty scaffolds** |
| `src/sidlens/provenance/`, `registry/` | freeze, verify (`--profile core\|ar\|full` checks a partial substrate and says so) and manifests; `profiles` + `bundle` (hash-certified transfer through private HF dataset repos: `sidlens bundle plan/create/check/push/pull`); the diffusion registry and log parsing |
| `experiments/` | Entry points, by task. `structure/` (static SID-map analyses: `exp1_atlas`, `exp2_geometry`, `semantic_mapping` incl. `anomaly_audit.py`). `retrospective/exp{1..4}` (CPU audits of archived outputs). `controlled/exp1_matched_beam`, `controlled/exp2_fixed_orders` (GPU decoding). `paper_www27/export_paper_data.py` packages the numbers cited in the WWW'27 draft (`docs/paper/Leo_WebConf27_GenRecSys_sidlens.zip`) with source hashes. Index: `experiments/README.md`. |
| `scripts/` | By task, mirroring `experiments/`: `common/` (`_preamble.sh`, sourced by every wrapper: `~/.config/sidlens/site.env`, verify gate honouring `SIDLENS_VERIFY_PROFILE`, offline HF, provenance stamp; `site.env.example`), `data/` (`build_labels.py`), `provenance/` (`audit_checkpoints.py`), `structure/`, `retrospective/`, `controlled/` (sbatch wrappers for the matching `experiments/` group), `activations/` (`ar_capture.{py,sbatch}`, site-neutral; `hooks_smoke.{py,sbatch}`). Submit from the repo root. Index: `scripts/README.md`. |
| `requirements/` | `lock-20260925-cu124.txt`, pip freeze of this venv (a record, not an install recipe) |
| `docs/` | `plans/` (project brief PDF, research plan, three-hour ladder), `results/semantic_mapping/` (results, takeaways, figures, offline explorer, worked prefix-14 example), `clusters/` (cross-cluster runbook, prompts for the upstream-cluster agents), `paper/` (WWW'27 drafts: original zip, SidLens revision zip, compiled and diff PDFs). Index: `docs/README.md`. |
| `tests/` | pytest, CPU only, about 1 minute, grouped like the code: `data/`, `models/`, `hooks/`, `analysis/`, `provenance/`, `experiments/`. Basenames must stay unique (no `__init__.py`). 253 passed and 3 skipped on 2026-09-27. |

## Running things

```bash
export SIDLENS_WORK=/l/users/leo.rodrigues/sidlens
PY=$SIDLENS_WORK/venv/bin/python
$PY -m sidlens.cli verify --strict --quick     # substrate gate (size check + vendor hashes)
$PY -m sidlens.cli bundle plan --profile ar    # what a transfer would carry; writes nothing
$PY -m pytest -q                               # full tests, CPU
$PY scripts/provenance/audit_checkpoints.py --missing-only   # exits 1 by design while cells are missing
$PY -m sidlens.analysis.semantic_mapping query \
  --db $SIDLENS_WORK/derived/semantic_mapping/194268/mappings.sqlite \
  --variant rqkmeans_3codebook_128 --prefix 14,16 --format csv
```

- The login node (`lo-02`) has **no GPU**. Never load models there; the AR
  model is 3 GB. Submit with `sbatch` to partition `ws-ia`. GPU wrappers pin
  `--nodelist=ws-l3-020 --gres=gpu:1`. Compute nodes have no outbound network.
- Each wrapper sources `scripts/common/_preamble.sh`, calls `sidlens_gate`, archives
  the executed source, and writes to a new job-specific directory. Example
  commands are in `experiments/controlled/exp2_fixed_orders/README.md`.
- `scripts/retrospective/retrospective_experiments.sbatch` writes into fixed
  `derived/retrospective/*` directories and would overwrite reports. Do not
  rerun it casually.

## Conventions for a new experiment

The two `controlled/` experiments are the templates.

- Create `experiments/controlled/<name>/` containing:
  - `protocol.md`, written **before** outcomes are seen: question, cohort,
    conditions, primary estimand
  - `run.py` and `summarize.py`
  - `README.md` and `RESULTS.md`

  Add a matching `scripts/controlled/<name>.sbatch`.
- Write outputs to `$SIDLENS_WORK/derived/controlled/<name>/<array-id>/cell-XX/`
  and summaries to `summary-<array>-<job>/`. Existing result directories are
  never overwritten; `mkdir` fails on a collision, by design. Keep:
  - `source.tar.gz` and source hashes
  - arguments, `inputs`/`validation.json`
  - predictions and the per-user table
  - `report.md`, `figures/`
  - `status.txt`, `output.sha256`
- Statistics: a paired-user bootstrap, 2,000 draws, fixed recorded seed, with
  all cells and conditions of a sampled user kept together. Always state that
  intervals exclude training-seed uncertainty. Keep SID-level and item-level
  metrics separate, and keep AR and diffusion cohorts separate.
- New decoders must reproduce the archived baseline exactly (predictions and
  ranks, `atol = rtol = 1e-4` on scores) before any new condition counts.
- Code style: dense "why" docstrings that name the trap each guard prevents;
  fail loudly rather than fall back. Match this style.

## Experiment status and result locations (2026-09-22)

| Result | Headline | Where |
|---|---|---|
| Exp 1 static atlas (27 variants) | Per-digit conditional AMI, naming AUC, co-purchase lift | `derived/exp1_atlas/<variant>/` |
| Exp 2 geometry and refinement (27) | Radius, R², null-corrected refinement, digit deletion | `derived/exp2_geometry/`, `derived/exp2_refinement/` |
| Retro 1: reveal order (archived) | Confidence +0.34 pp HR@10, but at unequal beams | `derived/retrospective/exp1_reveal_order/` |
| Retro 2: AR prefix errors | First mismatch at digit 1 in 82% of rows | `derived/retrospective/exp2_prefix/` |
| Retro 3: collisions | SID HR@10 exceeds item-uniform HR by 4.5 pp; collided targets +42.5 pp | `derived/retrospective/exp3_collisions/` |
| Retro 4: next-two association | P(p2 correct \| p1 correct) 4.0% vs 0.3%; associational only | `derived/retrospective/exp4_next_two/` |
| Controlled 1: matched beam (18 cells, 2026-09-09) | Confidence − fixed(seed 42): −0.95 pp at beam 64, +0.22 pp at beam 256. At beam 64 the gap moves from +0.4 to −0.7 to −2.6 pp over depths 3, 4 and 5; at beam 256 depth 5 is only −0.2 pp. | `derived/controlled/exp1_matched_beam/summary-181944-181946/` |
| Controlled 2: all six fixed orders at depth 3 (2026-09-15) | Confidence − mean fixed: +0.40 pp [0.29, 0.52]. The best order differs by quantizer. About 55% of confidence's final paths are duplicates. | `derived/controlled/exp2_fixed_orders/summary-194295-194297/` |
| Semantic mapping and explorer (2026-09-15) | 83,835 assignments; category coherence against a shuffled null. RQ-VAE 3cb-128 has an anomalous 26-item bucket `(27)(89)(7)`. | `derived/semantic_mapping/194268/` (`explorer.html`, `mappings.sqlite`), `anomaly-20260915/` |
| Structure-layer figures and takeaways (2026-09-22) | RQ digits carry semantics only given the prefix (digit 1 alone: 53 % of variance; digits 2–4: 12–16 %); MQ digits are four similar, overlapping partitions (36–38 % each); category purity saturates by digit 3; RQ-VAE under-uses digit 1 in 8 of 9 fits (2–31 % of codes) and uses its last digit as an identity digit. RQ-KMeans 5×512 gives the 11 identical-embedding pairs distinct SIDs (every other table collides them), so it was not made by the same deterministic procedure. | `derived/semantic_mapping/figures-20260922/`, `docs/results/semantic_mapping/SEMANTIC_MAPPING_TAKEAWAYS.md`, `docs/plans/RESEARCH_PLAN.md` |
| Hooks smoke test | Both paradigms capture correctly | `derived/hooks_smoke/` |
| AR capture pilot (2026-09-25) | `scripts/activations/ar_capture.sbatch`, 8 rows, CPU fp32, layers 0/13/27/norm: checkpoint sha, archive and SID-table checks pass; batch-invariance 5e-4 on activation norm ~300. Pipeline check only, not a result. | `derived/ar_capture/208116/` |

Not yet started: probes, SAEs and crosscoders, activation patching,
partial-state deduplication, full sweep at beam 256, reveal-order traces,
AR order retraining, joint-block next-two. See
`docs/plans/THREE_HOUR_EXPERIMENT_PLAN.md` for the leveled experiment ladder and its
interpretation constraints.

## Outside this repo

The upstream OneDiffRec training code is on another cluster and is **not on this
machine**. That includes branch `leo-dev-changes`, the next-2-item sweep, ZCR
for RQ-KMeans, CCE, and training-time activation logging (about 100 GB per
dataset × paradigm). Before anything from there counts as evidence, export it,
hash it and register it under `$SIDLENS_WORK/runs/`; the "Run export contract"
section of the README gives the steps. `docs/clusters/CROSS_CLUSTER.md` is the runbook
for running SidLens there: bundles through private HF dataset repos, setup, and
measured activation sizes. `docs/clusters/AR_CLUSTER_PROMPT.md` is the prompt for the
agent doing AR captures and interventions there. `docs/clusters/MAIN_PROJECT_DATA_PROMPT.md`
is the extraction prompt for that repo. `docs/results/semantic_mapping/rqkmeans_prefix14.xlsx` is the worked SID
explorer example (eSUN filaments under RQ-KMeans prefix 14).
