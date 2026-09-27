# All six fixed reveal orders: frozen depth-three checkpoints

Declared 2026-09-15 before new inference. No training or checkpoint selection.

## Question and fixed design

Does shared-search confidence improve exact-SID hit@10 compared with the mean
outcome across all six fixed reveal orders? Use MQ, RQ-KMeans, RQ-VAE crossed
with widths 128 and 512, all at depth three: old cell indices 0,1,6,7,12,13.
All 6,297 test users are retained in every cell and condition.

Primary beam: 64. Beam 256 is a declared sensitivity if pilot timing and the
session budget permit. A 64-user implementation/timing pilot uses both beams;
its outcomes are not used to select hypotheses or cells. If beam 256 is omitted
from the full sweep, report that scope explicitly. Both beams use the same
seven conditions: shared-search confidence and 012,021,102,120,201,210. Orders
are zero-based SID positions. A fixed-order mean averages outcomes and does
not merge recommendation lists. The best test order is a descriptive oracle.

Primary outcome is confidence minus mean fixed exact-SID hit@10, averaging
cells equally and pairing users. Secondary: NDCG@10, each fixed order, final
legal top-10 coverage, returned count, duplicate/invalid final paths, active
beam sizes, decoder rows, elapsed time, peak GPU memory, collision-aware item
bounds. Per-cell and per-order comparisons are exploratory.

## Numerical controls and acceptance

Use the unchanged sidlens.analysis.matched_decode.decode implementation, FP32,
TF32 disabled, eval mode, identical legal catalogue, score accumulation,
full final-digit expansion, tie ordering, and maximum-score final deduplication.
Only the allowed next position changes. Both seed-42 fixed and confidence
predictions/ranks must reproduce the archived matched-beam run for the same
users; finite scores use atol=rtol=1e-4. The pilot additionally verifies vendor
fixed-order reconstruction and all-six-order/confidence batch/chunk invariance.
Checkpoint and cohort identities must match the archive. Hash inputs, source,
protocol and outputs; fail on collisions with previous result directories.

Store predictions, scores, target identities, per-user outcomes and final-path
diagnostics. Summary independently recomputes ranks and hit/NDCG from predictions,
checks the six complete cells and all declared conditions, user identities,
source manifests, baseline checks and file hashes. Bootstrap users with 2,000
draws (seed 20260915), retaining all cells/conditions together; report percentile
95% intervals in percentage points. No cell or user exclusions.

## Interpretation

Inference compares search rules conditional on these frozen checkpoints, which
were selected using confidence validation scores. Equal beam limits do not
establish equal compute. Exact-SID hits can conceal several items sharing one
SID. User intervals do not include training-seed uncertainty. MQ digit position
does not establish an intrinsic coarse-to-fine semantic hierarchy. The partial
state deduplication intervention is a separate experiment.
