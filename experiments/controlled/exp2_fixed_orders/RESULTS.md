# All fixed reveal orders: completed 2026-09-15

On six frozen depth-three cells, confidence exceeds the mean outcome across
all six fixed reveal orders at beam 64 by **0.402 percentage points in SID
HR@10**, with a paired-user 95% interval of **[+0.287, +0.523]**.

The grid is MQ, RQ-KMeans and RQ-VAE crossed with widths 128 and 512. Every
cell retains the same 6,297 users and all seven conditions: 264,474
user–cell–condition evaluations. No models were retrained or selected.

| Outcome | Confidence | Mean fixed | Difference | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Exact-SID HR@10 | 16.680% | 16.278% | +0.402 pp | [+0.287, +0.523] |
| NDCG@10 (0–100 scale) | 12.293 | 12.003 | +0.289 | [+0.229, +0.348] |
| Item-uniform HR@10 | 12.093% | 11.844% | +0.249 pp | [+0.149, +0.345] |

All six per-cell HR differences are positive; four individual intervals cross
zero. The strongest effects are MQ-128 (+0.908 pp) and RQ-KMeans-128 (+0.691 pp).
The best aggregate fixed order on this test cohort is digit 2 → 3 → 1, with
16.444% SID HR@10. That choice is descriptive; selecting a deployable order
requires a separate validation set.

These estimates apply to six depth-three cells and an average over all fixed
orders. The earlier matched-beam report averaged 18 cells spanning depths
three through five and used one seed-42 order.

## Validation and scope

- All six complete cohorts, condition axes, source/settings identities and
  artifact hashes passed validation. Existing confidence and seed-42 fixed
  predictions, ranks and metrics reproduced the archive exactly.
- Model/vendor checks and small all-order batch/chunk checks passed. An
  independent audit reproduced primary estimates and paired intervals from
  saved predictions/targets and verified artifact/source hashes.
- Intervals use 2,000 paired-user bootstrap draws, seed 20260915, preserving
  every cell and condition together. They exclude training-seed uncertainty.
- SID hits can conceal item collisions. The item-uniform calculation assumes
  equal probability among items sharing a SID.
- Canonical decoding/scoring consumed 5.05 GPU minutes, excluding loading,
  validation, queue and reporting. Equal beam caps do not imply equal compute.
- Beam 256 was completed only for a 64-user implementation pilot. Its full
  six-cell sensitivity and partial-state deduplication remain follow-up work.
- An initial attempt failed exact reconstruction under different batch/chunk
  settings. It was preserved, diagnosed, and replaced by a full restart with
  archived 32/1024 settings; see [README.md](README.md).

## Artifacts

- Accepted inference array: **194295**; summary: **194297**.
- [Complete report](/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/summary-194295-194297/report.md).
- [Numerical results](/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/summary-194295-194297/result.json).
- [Per-cell effects](/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/summary-194295-194297/per_cell.csv).
- [Quality and cost](/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/summary-194295-194297/quality_cost.csv).
- [Exportable figure](/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/summary-194295-194297/figures/order_effect.pdf).
- Predictions, per-user tables, source archives and input identities live in
  `/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/194295/cell-XX/`
  for cells 00, 01, 06, 07, 12 and 13.
