# Prompt for the LLM in the main project folder

Copy everything below the horizontal rule into the LLM working in the main
OneDiffRec/GenRecSys project folder. The prompt requests an executable extraction
and a data bundle, not just a list of files. It is independent of any particular
LLM product.

---

You are working in the main generative-recommendation project. Audit the project,
parse its research artifacts, and create a reproducible data bundle for SidLens,
the mechanistic analysis of Semantic-ID (SID) recommenders. Work autonomously;
make useful progress when artifacts are missing and report the missing evidence.

## Objective and research questions

We have a three-hour experimental sprint. Prioritize inputs that enable useful
experiments immediately, then inventory the larger research program:

1. **Paradigm:** with identical SIDs and histories, how do autoregressive (AR)
   and masked-diffusion recommenders use history and select output digits?
2. **Representation:** what semantic, collaborative, and identity information
   does each digit add under RQ-VAE, RQ-KMeans, and parallel MQ/PSE?
3. **Order and scale:** how do digit order, depth, width, collisions, candidate
   diversity, and decoding budget affect retrieval and internal computation?
4. **Next-two:** does item 2 depend causally on item 1, and does the actual
   diffusion implementation evaluate a joint block or two sequential passes?

The main grid is quantizer {RQ-VAE, RQ-KMeans, MQ/PSE} × depth {3,4,5} ×
codebook size {128,256,512}, with AR/diffusion and next-item/next-two tasks.
Treat AR order conditions as additional, explicitly recorded factors.

Within the first 15–20 minutes, emit a minimal P0 handoff containing discovered
roots, usable checkpoint/config/SID/cohort references, existing output paths,
and urgent blockers. Then continue the broader extraction. Do not block the
experiment worker on exhaustive workbook parsing or large-binary hashing;
mark uncomputed hashes as pending, with their verification status explicit.

## 1. Discover the real roots and current state

Read applicable project instructions, READMEs, research proposals, experiment
results, manifests, model/evaluator/tokenizer code, and launch scripts. Start
with `rg --files`; inspect referenced artifact roots and symlink targets.
Do not traverse unrelated personal directories.

On the SidLens analysis machine, the known paths are:

```text
code: /home/leo.rodrigues/GenRecSys/sidlens/sidlens
bulk: /l/users/leo.rodrigues/sidlens
bulk/{frozen,external,derived,runs}
```

Honor `SIDLENS_REPO`, `SIDLENS_WORK`, `ONEDIFFREC_REPO`, and
`ONEDIFFREC_WORK` when set; discover equivalents on this machine. Historical
upstream paths `/home/leo.rodrigues/onediffrec/OneDiffRec` and
`/l/users/leo.rodrigues/onediffrec` were absent on the analysis machine. Do not
assume they exist here or silently substitute an older project copy.

If SidLens is accessible, inspect these sources first:

```text
README.md
docs/plans/LeoRodrigues_ProjectProposal_GenRec_MechInterpAnalysis.pdf
src/sidlens/provenance/spec.py
src/sidlens/paths.py
manifests/CURRENT
manifests/provenance.base-20260826.json
manifests/registry.diffusion.json
src/sidlens/data/{sids,diffusion_eval,labels,embeddings}.py
src/sidlens/models/{ar,diffusion}.py
experiments/controlled/exp1_matched_beam/{README,RESULTS}.md
experiments/retrospective/README.md
```

The September 15, 2026 SidLens inspection found existing static analyses for
27 Industrial variants, four completed retrospective analyses, and a completed
18-cell matched-beam diffusion experiment. Locate their actual outputs under
`derived/`, including `controlled/exp1_matched_beam/181944/cell-*` and
`summary-181944-181946`. Do not count completed work as a proposed new experiment.
Treat these facts as discovery anchors; verify current local evidence.

## 2. Extract the artifacts, in priority order

### P0: identities, runnable checkpoints, and existing experimental evidence

- Exact item-to-SID index JSONs, DiffGRM `.sem_ids`, item-to-token maps,
  token-to-**all-items** maps, constrained-decoding files, and ASIN ↔ upstream
  integer ID ↔ DiffGRM item ID ↔ embedding-row maps. Preserve original bytes.
- Both repaired and pre-repair RQ-KMeans `.mispacked` assignments, with an
  explicit record of which assignment each training run actually consumed.
- Canonical train/validation/test interactions and timestamps; next-item and
  next-two SFT CSVs; DiffGRM sequence files; preprocessing/split code and seeds.
  Keep splits and variants in separate directories even when basenames match.
- A run registry covering every planned cell and every discovered run. Record
  final/best/landmark weights, tokenizer files, resolved config, base model,
  training/decode arguments, seed, selection metric, checkpoint step, source
  commit/dirty patch, logs, and submitted job script. Distinguish inference
  weights from resumable training state. Discover shard indexes and adapters
  as well as single-file weights; record the required base for any adapter.
- Retained metrics, loss curves, prediction JSON/JSONL/NPZ/CSV, ranked beams,
  raw scores, timing and memory measurements. Parse all relevant sheets in
  experiment workbooks and referenced result tables, retaining sheet/cell
  provenance. Separate recorded metrics from newly recomputed values.
- Existing SidLens atlas, geometry/refinement, prefix, collision, next-two,
  and matched-beam results; their per-example files, source archives, hashes,
  status markers, and uncertainty specifications.

Known SidLens checkpoint anchors:

| AR checkpoint | SID variant | Matched diffusion checkpoint |
|---|---|---|
| `next-item_best` | `rqkmeans_3codebook_128` | `diff-next1-rqkmeans-3cb-128` |
| `oneoff_rqvae4cb128` | `rqvae_4codebook_128` | `diff-next1-rqvae-4cb-128` |
| `two-item_best` | `MQ_4codebook_256` | No retained matched pair |

The one-off RQ-VAE AR model is not the checkpoint behind the historical sweep
row. The frozen diffusion registry has 20 trained next-item entries and three
next-two entries, of which two are trained and one incomplete. Search for
additional exports and newer runs; do not assume missing weights can be
recovered from metrics. Label each cell as weights-present, metrics-only,
incomplete, never-trained-with-evidence, or unknown; absence alone does not
prove it was never trained.

### P1: semantics, geometry, and model internals

- Original pre-quantization item embeddings and row maps, with model revision,
  input text fields, pooling, normalization, dtype, and dimensionality.
  The known frozen matrices are FP16 Qwen3-Embedding-4B outputs of shape
  `(3105,2560)` and `(17696,2560)`, mean-pooled from title/description without
  PCA or L2 normalization. Some filenames misleadingly say
  `sentence-t5-base_pca256`; verify content and provenance rather than names.
- Training-lineage item text/metadata and collaborative interactions; separately
  acquired Amazon-2018 metadata, category hierarchy, prices/ranks, brand,
  `also_buy`, `also_view`, and label-table manifests. Report label coverage and
  support per class. Color/material/size/style require validated extraction;
  they are not ready-made labels merely because those words occur in text.
- Quantizer weights: RQ-VAE encoder/decoder/codebooks, fitted RQ-KMeans
  centroids/transforms/index, and MQ/PSE implementation and parameters. Exact
  SID tables enable decoder analysis without enabling quantizer regeneration.
- Existing hidden states, attention outputs/weights, per-digit logits, reveal
  masks/orders, beam ancestry, step indices, RNG seeds, probe/SAE weights, and
  intervention traces. Inventory existing traces; do not claim inference
  traces can be reconstructed from aggregate metrics.

### P2: recovery and longer-term experiments

- Exact AR initialization weights and tokenizer, intermediate checkpoints,
  optimizer/scheduler/scaler/RNG/sampler states, and trainer-state records.
- Matched AR order-condition training exports and true joint-block next-two
  checkpoints/evaluation paths. Record availability and recovery/training cost.
- Environment locks, package versions, CUDA/GPU information, scheduler/job logs,
  and any energy/FLOP measurements already recorded.

## 3. Enforce the identity and evaluation contracts

1. The primary catalogue is Amazon-2018 Industrial_and_Scientific: 3,105 items.
   Office has 17,696 items. Some DiffGRM paths say AmazonReviews2014 even when
   the data lineage is 2018. The legacy `data/Amazon` catalogue is different;
   never join it solely by category name or numeric item ID.
2. Verify ASIN sets, item counts, SID lengths/ranges, mapping coverage,
   embedding-row alignment, split membership, and target membership. Preserve
   one-to-many SID collisions. Never choose a representative item silently.
3. For AR, read the exact `added_tokens.json`: IDs are string-sorted and unused
   codes may be omitted. Do not calculate AR token IDs as an offset plus code.
4. Record whether `.sem_ids` stores an extra collision/disambiguation column
   and whether the model consumes it. Distinguish semantic digits from any
   auxiliary uniqueness suffix.
5. Recover diffusion `n_head` and all non-shape config from run evidence.
   Tensor-shape compatibility alone does not establish behavioral equivalence.
6. Inspect the executable decoder, including scoring, final-digit expansion,
   beam cap, active candidate counts, legality filtering, deduplication,
   tie handling, history truncation, and checkpoint-selection policy.
7. Explicitly distinguish exact-SID hit/HR, legacy title-compatible HR, and
   item-level retrieval. Preserve ranked empty/invalid outputs and missing
   scores as such. Collision-aware uniform tie-breaking is an assumption;
   include lower/upper bounds and the expansion convention.
8. Never join AR and diffusion examples by row number. Historical AR next-item
   data has 3,681 events from 1,606 users per retained configuration; diffusion
   has 6,297 final-item users. Construct a crosswalk using verified user mapping,
   ordered history, target item(s), and event/time identifiers. Report overlaps,
   exclusions, and both native and common truncated histories.
9. Separate source-observed facts, recovered configurations, assumptions, and
   unresolved conflicts. Prefer raw run evidence over stale prose; retain
   disagreements with source pointers rather than silently choosing a value.
10. Do not edit frozen inputs or vendored source. Copy newly recovered artifacts
    into a new bundle; give each a hash and lineage before using it as evidence.
11. Verify whether depth variants share nested assignments or were fitted
    independently. Changing SID depth is not automatically a controlled
    extension of the same representation. Preserve training-frequency counts
    at both item and SID-bucket level for collision/popularity controls.

Known parser anchors: SFT CSVs contain `user_id`, `history_item_title`,
`item_title`, `history_item_id`, `item_id`, `history_item_sid`, and `item_sid`.
Historical AR next-item predictions use `input`, `output`, and `predict`;
next-two predictions use `gt1`, `gt2`, and `top_pairs`. These prediction rows
do not contain user IDs. Validate their targets and any available history
against the source CSV before a positional attachment of identity. Parse
serialized lists safely; never use `eval`.

## 4. Produce a reusable parser and an analysis-ready bundle

Create a documented extraction script in a new, nonconflicting location and run
it. Write into `analysis_exports/sidlens_<UTC timestamp>/` or an equivalent new
directory. Keep the source project unchanged apart from those new artifacts.
Do not start training or reserve GPUs as part of this extraction.

Required outputs, using JSONL/CSV plus NPZ or Parquet where appropriate:

| Output | Required information |
|---|---|
| `README.md` | Roots, reproduction command, discovered scope, immediate usable inputs, blockers |
| `manifest.json` | Schema version, UTC time, source roots, code revision/dirty state, arguments, file hashes and hash-verification status |
| `artifact_inventory.csv` | Artifact ID/type, run/cell, original and resolved path, bytes, SHA-256, availability, provenance source, priority |
| `runs.jsonl` | Dataset/lineage, task, paradigm, quantizer, depth/width, digit order, seeds, checkpoint/config/tokenizer/split/SID references, selection rule, decoder settings, status |
| `items.jsonl` | Catalogue-scoped ASIN and integer IDs, embedding row, metadata/label references |
| `sids.jsonl` | Variant, assignment hash, item/ASIN, raw codes, consumed digits, auxiliary suffix, collision bucket ID/size |
| `examples.jsonl` | Task/split, stable example ID, user, timestamp/event index, ordered item history, target item(s), raw row reference, history hash |
| `cohort_crosswalk.csv` | AR/diffusion example IDs, evidence for the join, common/native history hashes, eligibility and exclusion reason |
| `metrics.csv` | Run/eval/condition, metric, cutoff, value, scale, unit, denominator, split, SID/item/title semantics, weighting, interval method, source pointer |
| `predictions/` | Example/run/condition/rank, ordered raw SID(s), decoded item buckets, score/type/availability, validity and padding; condition includes beam, reveal policy, order, seed |
| `traces/index.jsonl` | Existing trace file, model/activation location, tensor shape/dtype, example/beam/step mapping, masks/reveals, RNG and hash references |
| `validation.json` | Passed/failed/unchecked joins, count/range/hash checks, metric reproductions, parser failures, explicit denominators |
| `missing_artifacts.csv` | Exact missing asset, affected experiment, searched locations, recovery route, whether new inference/training is required |
| `ready_experiments.md` | Ranked experiments possible now, minimum inputs, exact existing command or implementation needed, measured versus estimated cost |

Use references with hashes for large immutable weights/embeddings; include small
configs, mappings, logs, and normalized tables in the bundle. Record existing
versus freshly computed hashes. Make a transfer file list for missing bulk assets
and preserve relative hierarchy; do not duplicate gigabytes unnecessarily.
Use streaming/chunked parsing for large files. Load NPZ without pickle; do not
execute arbitrary serialized objects to inspect them. Keep original files so
parsing is reversible, and include row counts and parse failures per source.

## 5. Finish with an actionable handoff

Return the bundle path, extraction command, counts of usable cells and matched
pairs, validation failures, highest-priority missing files, and paths to the
tables above. Map available inputs to these immediate follow-ups:

- All six fixed reveal orders at depth 3 under matched beam caps, compared
  with confidence-guided search on the same users and checkpoints.
- Partial-state deduplication before beam pruning, to test the candidate-
  diversity explanation for confidence decoding's beam sensitivity.
- Diffusion logit/reveal trajectories, then selected-layer semantic probes
  and history interventions on the two matched next-item cells.

Flag that the existing SidLens `diffusion.digit_logits` helper uses an unvalidated
`use_cache=False` path: preserve projected cross-attention K/V as in the validated
matched decoder before proposing logit/capture results. Existing attention hooks
observe module outputs; head weights require additional instrumentation.

Do not infer a causal mechanism from a leaderboard correlation, a probe score,
or an observational next-two contingency. The handoff must state exactly what
can be run now and what needs new code, new inference, recovered weights, or
new training.
