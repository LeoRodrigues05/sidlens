# Experiment 1 follow-up: matched-beam reveal-policy comparison

This is new GPU inference on the frozen next-item DiffGRM checkpoints. It
follows the retrospective comparison in `experiments/retrospective/`.

## Question and fixed protocol

On the same checkpoint and test user, does choosing the next masked SID
position by confidence improve retrieval compared with the original fixed
seed-42 order when both searches have the same beam cap?

- Population: all 6,297 frozen leave-last-item test users, in verified order.
- Models: the complete 18-cell grid of MQ/RQ-KMeans/RQ-VAE, SID depths 3/4/5,
  and codebook sizes 128/512. No training, checkpoint selection, or tuning.
- Primary comparison: confidence minus fixed-order exact-SID hit@10 at beam
  64, equally averaged over the 18 cells.
- Secondary outcome: NDCG@10. Beam 256 is a budget sensitivity, subject to the
  GPU pilot establishing practical runtime. Report its inclusion explicitly.
- Fixed order: reproduce the original CPU Torch seed-42 permutation separately
  for each SID depth; the same order is used for every user at that depth.
- Both policies use the same inference precision, logits, accumulated log
  probabilities, beam cap, full final-token expansion, final legality filtering,
  maximum-score SID deduplication, and missing-result padding. Only the allowed
  next position differs. Confidence considers all remaining positions; fixed
  order considers its next position. No partial SID legality constraints.
- Beam width is a cap: fixed decoding at width 256/codebook size 128 has only
  128 distinct first-step expansions. Record actual active widths. Equal caps
  do not imply identical candidate counts or FLOPs; record runtime as well.
- Retain per-user predictions, scores, ranks, target identity/multiplicity,
  item-expansion bounds, and hashes. No result-dependent cell exclusions.
- Inference: paired user bootstrap, with all 18 cells for a sampled user kept
  together. Intervals describe these fixed checkpoints, not training seeds.

## Implementation and validation boundary

The frozen historical decoder has another asymmetry beyond beam width:
confidence greedily completes its final digit, whereas fixed order expands
all final-token alternatives. Both new policies use full expansion. Thus the
new absolute metrics need not equal the historical confidence metrics.

A third condition, `legacy_confidence`, additionally runs the exact frozen
confidence decoder at both beam caps, retaining its greedy final fill. Its
contrast with fixed order answers the literal beam-only rerun; its contrast
with shared confidence diagnoses final-expansion sensitivity. These are
diagnostics, not replacements for the shared-search primary comparison.
Legacy confidence256 and fixed64 must reproduce archived full-cohort metrics
within one hit-equivalent (1/6,297). The vendor exposes no path scores or
internal candidate counts: legacy scores are NaN with explicit availability
metadata, and path diagnostics are null, rather than invented.

Confidence can retain duplicate partial paths arriving at the same SID through
different reveal orders. Neither shared policy deduplicates partial paths, so
equal beam caps need not retain equally many unique SIDs. Report final unique
counts and coverage (users receiving ten distinct legal predictions).

The model's projected encoder cross-attention keys/values must be preserved.
The vendor's `use_cache=False` fallback is not equivalent to its cached/trained
path. Pilot checks compare the shared decoder with the original fixed64 path,
including batched versus individual histories, before the full sweep.

The same confidence-selected checkpoints are used for both conditions. The
result identifies a decoding-policy contrast conditional on those checkpoints
and this fixed order. It does not estimate an average over checkpoint-selection
rules, independently trained models, or all random reveal orders.

## Execution

GPU access is through `sbatch`, on `ws-l3-020` in `ws-ia`, with one GPU.
Each job writes to a new job-specific directory under
`$SIDLENS_WORK/derived/controlled/exp1_matched_beam/`. Frozen inputs are never
modified. A source archive, environment, argument manifest, status marker,
input hashes, and final output hashes accompany every run.
