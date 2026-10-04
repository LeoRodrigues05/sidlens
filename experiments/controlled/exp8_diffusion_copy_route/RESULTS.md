# DiffGRM copy route: completed 2026-09-28

DiffGRM copies the most recent history item's matching digit mainly **through
its encoder**: the item's information spreads into the other history slots,
and the decoder reads it there.

- **Both routes together remove most of the copy.** Blocking the encoder spread
  and the decoder's direct read together lowers the golden digit in matching
  rows by **0.24 nats** (RQ-KMeans 3×128) and **0.60 nats** (RQ-VAE 4×128).
  That is about two thirds of exp7's replacement effect (0.35 and 0.96).
- **Control.** The same double block aimed at a non-matching item gives −0.02
  and −0.05.
- **The encoder route alone carries most of it:** 0.17 and 0.30 nats.
- **The decoder route alone carries nothing:** −0.008 and −0.004, as exp7
  found.
- **The routes back each other up.** The double block exceeds the sum of the
  single blocks by 0.08 and 0.30 (H2). The direct read of the item's own slot
  becomes useful only once the spread is blocked.

The protocol ([protocol.md](protocol.md)) was declared after exp7 and before
any exp8 forward pass. Cohort and states are as in exp7 (6,297 users, fp32,
CPU). Intervals are paired user-bootstrap 95% intervals (2,000 draws, seed
20260927) and exclude training-seed variance.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp8_diffusion_copy_route/` |
|---|---|---|
| Primary | 283159 | `283159/cell-0{0,1}`, `summary-283159/` |

## Numerical acceptance

- **V1.** exp1's pilot checks passed for both cells.
- **V2.** Both no-op masks are bit-identical to the vendor path: the
  padding-only expanded encoder mask and the all-ones cross mask.
- **V3.** Clean rows are bit-identical across batches (18,891 and 25,188
  comparisons).
- **Guards.** A mask that re-opens padding, or knocks out a padded slot, is
  refused. The tests are in `tests/interventions/test_diffusion_knockout.py`.

## Declared estimands (matching rows, pooled d ≥ 2)

| | RQ-KMeans 3×128 | RQ-VAE 4×128 | Verdict |
|---|---|---|---|
| **H1** Δ(`both`) | **+0.236** [0.206, 0.268] | **+0.598** [0.537, 0.663] | holds |
| **H1** Δ(`both`) − Δ(`both_ctrl`), paired | **+0.299** [0.261, 0.337] | **+0.702** [0.623, 0.785] | holds |
| **H2** Δ(`both`) − [Δ(`enc`) + Δ(`dec`)] | **+0.075** [0.064, 0.087] | **+0.298** [0.263, 0.333] | holds |
| Δ(`enc`), encoder route only | +0.169 [0.147, 0.194] | +0.304 [0.266, 0.347] | |
| Δ(`dec`), decoder route only | −0.008 [−0.014, −0.002] | −0.004 [−0.009, 0.001] | |
| Δ(`both`) \| non-matching rows | +0.023 | +0.054 | |
| First digit at `S_full` (all rows): both / enc / dec | 0.243 / 0.118 / −0.001 | 0.165 / 0.073 / 0.003 | |

**Expectations recorded before outcomes**

- **E1** (in matching rows, both ≥ half of exp7's replacement effect): **held**
  (0.236 / 0.351 = 67%; 0.598 / 0.955 = 63%).
- **E2** (`enc` and `dec` each small, `both` large): **held in part**. `dec` is
  negligible and the routes are super-additive, but `enc` alone is not small
  (72% and 51% of `both`).

## What this adds to the AR comparison

| | AR recommenders | DiffGRM |
|---|---|---|
| Where the matching item is read | its own digit-d token, read directly from the readout position in layers 14–27 | its information, spread by the encoder into the other history slots |
| The direct read of the item | carries the effect (0.36–0.40 nats) | a backup only (0 alone; +0.08 / +0.30 once the spread is blocked) |

## Limits

- **What `both` leaves.** The item's own encoder slot still exists: it is
  unread but still in the softmax. So `both` is close to, but not identical
  with, removing the item.
- **States.** These are fixed reveal states.
- **Seeds.** One seed per checkpoint.
