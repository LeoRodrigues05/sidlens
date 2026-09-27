# SidLens: experiment ladder and three-hour sprint

Prepared from repository source and local results on **2026-09-15**. This is a
research plan and data handoff; the experiments proposed below have not been
launched by this audit. Companion: [upstream extraction prompt](../clusters/MAIN_PROJECT_DATA_PROMPT.md).

## Recommendation

Use the next three hours to answer **whether confidence-guided diffusion beats
the average fixed order on the retained depth-3 cells, and test its sensitivity
to beam size**. Make the
fixed-order comparison the primary deliverable; run a partial-state
deduplication intervention only if implementation, validation, and GPU time
permit. In parallel, organize the existing representation and collision
evidence. One well-controlled new result is a better three-hour commitment
than starting probes, SAEs, retraining, and next-two studies together.

The layers of analysis below are levels of experimental evidence, not just
transformer layers. Each has a different unit of analysis and claim boundary.

## What already exists

| Evidence | Current state and implication |
|---|---|
| Static atlas and geometry | Completed for 27 Industrial SID variants. Reuse `derived/exp1_atlas`, `exp2_geometry`, and `exp2_refinement`. |
| AR prefix errors | Completed: 18 configurations, 66,258 event–configuration rows, 1,606 users. First mismatch is digit 1 in 82.16% of applicable rows. This describes errors; it does not prove digit-1 causality. |
| AR collisions | Completed: exact-SID HR@10 exceeds the uniform-within-bucket item HR@10 by 4.5 percentage points on average. SID retrieval can conceal item ambiguity. |
| Historical next-two | Completed for 10 AR prediction cells. Conditional success is associated across slots; retained outputs cannot establish causal dependence. |
| Matched-beam diffusion | Completed for 18 cells × 6,297 users, at beams 64 and 256. Shared-search confidence minus fixed HR@10 is −0.949 pp at 64 and +0.224 pp at 256. |
| Model availability | Three retained AR anchors and 20 trained next-item diffusion checkpoints; only two matched next-item SID cells. Two trained and one incomplete next-two diffusion checkpoints; no matched AR/joint-block pair. |
| Mechanistic infrastructure | Loaders and observation hooks exist. `probes/`, `sae/`, and `interventions/` are scaffolds. No completed causal pipeline or registered numerical-patch mechanism. |

Sources: [controlled result](../../experiments/controlled/exp1_matched_beam/RESULTS.md),
[prefix report](/l/users/leo.rodrigues/sidlens/derived/retrospective/exp2_prefix/report.md),
[collision report](/l/users/leo.rodrigues/sidlens/derived/retrospective/exp3_collisions/report.md),
[next-two report](/l/users/leo.rodrigues/sidlens/derived/retrospective/exp4_next_two/report.md).

During this audit, `verify --strict --quick` passed for all 38 frozen inventory
sections and three vendor trees. The checkpoint audit found weights for 25
planned cells, 26 completed AR cells with lost weights, and 84 cells without
training evidence in the searched inventory. There are 26 actual weight files
including the one-off AR anchor; the 25 planned cells include one incomplete
diffusion run. These are availability counts, not 25 completed experiments.
No new full hash audit of the large frozen binaries was performed.

The original confidence decoder returned ten distinct legal predictions in
49.39% of cases at beam 64 versus 94.66% for fixed order. This motivates testing
candidate diversity, but invalid paths, final filtering, and duplicate paths
must be separated experimentally. Equal beam caps do not mean equal compute.

## Experiments at each level

“Existing” means outputs/code already exist. “Extension” requires new analysis
or inference code. Time ranges are planning allowances, not measured runtimes.

| Level | Experiments and question | Measurements and essential controls | Readiness |
|---|---|---|---|
| **0. Identity and reproducibility** | Verify checkpoint–SID–tokenizer pairing; reconstruct recorded metrics; form a common AR/diffusion cohort. | Hashes, item/ASIN coverage, split overlap, exact histories/targets, config evidence, reproduction error. Preserve repaired/mispacked variants and all collision members. | Existing audits; upstream recovery via companion prompt. First 15–20 min. |
| **1. Item, codeword, and digit semantics** | Attribute association per digit; information added after a prefix; representative items/terms; training interaction or co-purchase neighborhood agreement. Add order sensitivity for MQ. | Conditional AMI with within-parent permutation null; label coverage/class support; item-held-out term/probe evaluation; train-only collaborative counts. Avoid treating codeword integers as ordered semantic distances. | Static outputs exist. New synthesis or MQ permutation sensitivity is a CPU extension. |
| **2. Prefix geometry and identity capacity** | Radius/variance by prefix; refinement beyond random subdivision; digit-deletion information loss; utilization and full-SID collisions; matched collision-rate depth/width contrasts. | Non-singleton radius median/IQR, item-weighted within-cluster variance, null-corrected refinement, neighbor preservation, conditional entropy, bucket sizes. Report singleton mass. | Static outputs exist. Match to behavioral results now; native RQ-VAE latent geometry needs missing encoder weights. |
| **3. Retrieval and decoding** | Fixed-order robustness; beam sensitivity; partial-state deduplication; legal-prefix filtering as a separate ablation; prefix errors and collision-aware retrieval. | Exact-SID HR/NDCG, item bounds, distinct legal top-10 coverage, invalid/duplicate paths, per-step unique states, latency/decoder rows. Change one search rule at a time on paired users. | Highest-priority new inference. Decoder supports supplied fixed orders; runner extension needed. |
| **4. Generation/denoising step** | Which digits are confident first? How does order change across histories? When do correct codes become recoverable, and how does a forced wrong commitment affect later digits? | Code-aligned logits, entropy/margin, calibration/NLL, reveal order and beam ancestry. Compare free decoding and correct-prefix conditioning separately; fixed masks/seeds for counterfactuals. | New trace runner needed; numerical-path validation first. A small pilot may fit, not a complete all-cell study. |
| **5. Transformer layer and representation** | Selected-layer probes of target category, broad prefix, and later-digit identity; compare history versus target positions and matched AR/diffusion semantic stages. | User-disjoint splits for history tasks; item-disjoint attribute generalization separately; balanced accuracy/macro-F1, held-out performance, shuffled-label and input-only baselines. Capture before target exposure. | Hooks exist; probe pipeline does not. Narrow follow-up after reliable traces. |
| **6. Causal history, state, and head** | Replace recent versus older history items with frequency/category-matched controls; patch clean activations into corrupted runs; valid digit swaps; head-output ablation. | Within-example change in target log probability, rank, recovery, output attributes. Clean/corrupt/self-patch/random-site controls; hold decode schedule and other conditions fixed. Attention alone is not causal evidence. | New intervention implementation and provenance required. Input interventions are the smallest entry point. |
| **7. Sparse features and cross-model alignment** | SAE feature discovery, feature ablation, and paired crosscoder alignment after a reproducible activation dataset exists. | Reconstruction, sparsity, seed stability, held-out activation patterns and causal effect. Compare aligned examples/stages; AR and diffusion widths/layer counts differ. | Later work. Three hours is insufficient for validated feature discovery across models. |
| **8. Next-two and controlled training** | Clamp item 1 and measure item-2 changes; block-direction interventions; matched AR forward/reverse/random order fine-tunes; depth/width and training-step comparisons. | Same history, seed and mask schedule; singleton targets; teacher-forced NLL plus free-running ranks. For order training: same initialization, SIDs, examples, training/decode budgets, and auxiliary suffix position. | AR next-two single-checkpoint pilot possible with new code. Joint-block comparison and order-training grid require new/recovered artifacts; not sprint commitments. |

### Interpretation constraints that affect experiment choice

- **MQ is parallel:** prefixes intersect partitions; later positions are not
  automatically finer semantic levels. Compare marginal and conditional
  information under multiple orders before assigning an intrinsic hierarchy.
- **Geometry uses shared input embeddings:** it is not RQ-VAE encoder-latent
  geometry. Text labels and text embeddings share a source; behavioral
  neighborhoods supply an additional evidence channel.
- **Label readiness varies:** brand is gated in the current atlas because of
  sparse class support. Color, size, material, and style need extraction and
  validation. Start probes with supported category labels.
- **Late-digit support can disappear:** conditional AMI requires at least
  eight labelled items per parent, with at least 100 contributing items and
  five parents for the reported support flag. Keep unsupported estimates
  separate. AMI is not mutual information in bits. Chance-corrected refinement
  is undefined when every cluster is a singleton; do not plot that as zero.
- **Scale is not automatically a controlled intervention:** depth variants may
  be independent quantizer fits. Verify shared prefixes/nesting before
  attributing their differences solely to adding a digit.
- **Cohorts differ:** do not compare the historical AR and diffusion headline
  metrics directly. Same SID assignment does not establish identical training
  data, evaluation context, model size, or optimization. Even the two matched
  cells support comparisons of these particular systems, not an isolated
  causal effect of autoregression versus diffusion.
- **Hooks need semantic indexing:** `keep_all_calls=True` keeps repeated
  tensors, but example/beam/step/reveal mapping must be saved separately.
  `attention_points` observes attention module outputs, not head weights.
- **Scores need an explicit meaning:** conditional target-code log probability
  at a fixed decoder state is suitable for a logit intervention. Accumulated
  beam path scores are unnormalized ranking scores, not item log probabilities.
- **Diffusion logits need validation:** the helper in
  [models/diffusion.py](../../src/sidlens/models/diffusion.py) uses
  `use_cache=False`; the vendor fallback omits learned cross-attention K/V
  projections. Use the validated projected-cache computation in
  [matched_decode.py](../../src/sidlens/analysis/matched_decode.py), with fresh
  self-attention, or repair and test the helper before collecting traces.
- **Joint next-two is not established:** retained diffusion evaluation emits
  one SID, appends it to history, then emits another. It does not evaluate the
  trained second-half positions as a joint two-item block.

## Primary experiment: all fixed reveal orders at depth 3

**Question:** Is confidence guidance better than the average fixed order, or
is its comparison with the single historical seed-42 permutation misleading?

### Fixed protocol

- Six frozen next-item cells: MQ, RQ-KMeans, and RQ-VAE × codebook {128,512},
  all at depth 3. These are existing runner cell indices `0,1,6,7,12,13`.
  This balanced subgrid is chosen before new results, not by historical wins.
- Run all **six permutations** of digits `(0,1,2)` plus shared-search
  confidence on the same 6,297 users at beam 64. Beam 256 is the predeclared
  sensitivity if the runtime pilot supports it.
- Primary estimand: for each user and cell, confidence hit@10 minus the mean
  hit@10 across the six fixed orders; average cells equally, then users.
  This averages fixed-order outcomes; it does not ensemble their predictions.
- Secondary: NDCG@10, each order separately, legal top-10 coverage, list
  length, invalid/duplicate paths, actual decoder rows, peak memory, latency.
  The best fixed order on test is only a descriptive oracle. Selecting a
  deployable order requires validation-set selection and a separate test run.
- Preserve checkpoint, precision, history, scoring, final expansion, tie rules,
  legal catalogue, and final deduplication. Record all resolved settings.
- Paired bootstrap of users with all cells/orders retained together; 2,000
  draws and a fixed recorded seed. Report 95% intervals in percentage points.
  This measures test-user uncertainty conditional on the frozen checkpoints,
  not uncertainty over training seeds. Report per-cell estimates too.
  Per-order and subgroup comparisons are exploratory; do not treat many
  unadjusted intervals as independent confirmatory discoveries.

### Implementation and acceptance

Extend the runner in a **new experiment directory** to accept a declared order
list and use `matched_decode.decode(..., policy="fixed", order=...)`.
The current runner hardcodes seed 42 and has no `--orders` flag. Do not present
that option as already available. Write a summary for this six-cell design:
the current matched-beam summarizer requires all 18 full-cohort cells.

On a small implementation pilot, reproduce seed-42 baseline predictions/ranks
and confidence predictions for the same users; test batch/chunk invariance and
correct order validation. Check that all six distinct permutations are run.
Use pilot outcomes only for numerical validation and timing, not hypothesis
selection. After the protocol is frozen, the full run includes those users.

**Done:** six complete beam-64 cells; matched example/hash identities; stored
predictions and per-user outcomes; numerical checks pass; paired estimate and
interval; per-order figure/table; explicit scope if beam 256 was omitted.
A null or negative result is a completed experiment.

## Stretch: intervene on duplicate partial paths

**Question:** Does spending beam slots on multiple paths to the same partial
state cause confidence search to lose retrieval quality at a small beam?

On the same depth-3 subgrid at beam 64, compare the 2 × 2 design:

| Reveal policy | Original path retention | Unique partial-state retention |
|---|---|---|
| Fixed seed-42 order | Baseline | Negative control |
| Confidence | Baseline | Intervention |

The intervention groups candidate states **before** beam truncation by revealed
position mask plus revealed code values, takes the maximum score per state,
then keeps the best B distinct states. Final maximum-score SID deduplication
remains the same. Deduplicating only after the cap does not test the same rule.
Do not merge different masks or sum path scores. Preserve deterministic ties.

Primary outcome: confidence intervention minus baseline exact-SID hit@10.
Mechanism diagnostics: per-step path/unique-state ratio, final legal coverage,
invalid rate, returned count, and computational cost. Fixed-order predictions
should be unchanged because its partial states have only one reveal order.
Use brute-force tiny examples and this negative control to validate the new
decoder; store its source hash and numerical-change specification.

Deduplication changes search breadth and compute as well as candidate diversity.
A successful result establishes the effect of this **search intervention** on
these checkpoints; it does not establish learned semantic reasoning or an
equal-compute benefit. Timing remains a separate outcome.

Do not start this extension if it threatens the complete primary report.

## Three-hour schedule

| Elapsed | Main track | Independent work and completion gate |
|---|---|---|
| 0–20 min | Freeze protocol, inspect node availability, verify inputs and baseline outputs, run upstream extraction prompt. | Inventory missing assets; join existing geometry/collision summaries. Record all choices before new outcomes. |
| 20–50 min | Implement order runner and six-cell summary; add only necessary invariance/order tests; GPU pilot on one depth-3 cell. | Design per-user/cell/condition schema; prepare figure/report generation. Pilot must preserve baseline numerics. |
| 50–125 min | Run all six cells at beam 64; include beam 256 only if measured projection leaves reporting time. | Finish CPU synthesis; implement dedup tests only if there is spare capacity. |
| 125–155 min | Complete missing primary cells and summarize. Run validated dedup pilot only if capacity remains. | Paired intervals, per-cell/order effects, quality–cost and coverage plots; flag every partial run. |
| 155–180 min | Recompute headline results from stored per-user files; finalize report and artifact hashes. | Deliver completed result, uncertainty and limitations, exact commands, and next experiment decision. |

The previous 18-cell matched-beam GPU array took about **65 minutes**, excluding
implementation and queue time. That is a useful reference, not a guarantee for
new code. Estimate this run from measured seconds/user/condition on the pilot,
plus loading and verification overhead. Keep at least 25 minutes for checks and
reporting. At audit time SLURM was available and the user's queue was empty;
an immediate GPU allocation was not established.

If queue time or validation consumes the budget, finish a declared balanced
subset or a CPU-only synthesis and report that reduced scope. Do not silently
drop slow/failing cells or describe a partial sweep as complete.

## CPU track and fallback

Join existing atlas/refinement, collision and retrieval tables by exact
variant/assignment identity. Produce per-quantizer depth/width panels showing
utilization, conditional information, singleton mass, collision-aware item
bounds and exact-SID performance. Keep AR and diffusion cohorts separate.
Any association of collision/geometry with quality is descriptive; cells reuse
users and are not independent training replicates.

Existing matched-beam per-user files also permit coverage and collision
stratification. Define strata with training frequency or target collision
status where possible. Stratifying by a decoder's achieved list length selects
on a treatment outcome and cannot prove that coverage mediates the improvement.
The decoder stores per-user counts, but its generated/unique/invalid path
diagnostics are aggregated by condition; per-user path diagnostics need a
new inference export.

## Existing commands and new output contract

Set the correct roots before Python starts:

```bash
export SIDLENS_REPO=/home/leo.rodrigues/GenRecSys/sidlens/sidlens
export SIDLENS_WORK=/l/users/leo.rodrigues/sidlens
cd "$SIDLENS_REPO"
"$SIDLENS_WORK/venv/bin/python" -m sidlens.cli verify --strict --quick
"$SIDLENS_WORK/venv/bin/python" scripts/provenance/audit_checkpoints.py --missing-only
```

The quick gate hashes vendored source but checks frozen data presence/size,
not all frozen bytes. Use `verify --strict` for the full hash check, preferably
on the allocated compute node if cold storage reads are slow. The checkpoint
audit intentionally exits 1 while planned cells are missing.

The following **existing** command is only an optional baseline reproduction
pilot, not the new order experiment:

```bash
sbatch --array=6 --time=00:30:00 scripts/controlled/matched_beam.sbatch \
  --limit 64 --beam-widths 64 --validate
```

It writes to a job-specific directory and uses the site's existing GPU wrapper.
New order/dedup experiments need their own runners, summary code, and job
wrappers; submit only after local scheduler settings and pilots are verified.
Do not rerun the retrospective array into its default directories: it writes
over the existing reports. Do not load the 3 GB AR model on the login node.

Each new experiment must retain:

```text
protocol.md                  # hypotheses, cohort, conditions, primary outcome
arguments.json + source/     # exact executable code, diff, environment
inputs.json                  # snapshot, checkpoints, SIDs, cohort hashes
predictions.npz              # users/targets/ranks/scores and condition axes
per_user.csv                 # outcomes and diagnostics, no silent exclusions
result.json + report.md      # estimates, intervals, scope, limitations
figures/                     # exportable plots with units and denominators
validation.json              # baseline and independent recomputation checks
status.txt + output.sha256   # completion and artifact integrity
```

Write under a new `$SIDLENS_WORK/derived/controlled/<experiment>/<run-id>/`.
The end-of-sprint handoff should distinguish **previous results**, **newly
completed results**, **pilots**, and **remaining work**.
