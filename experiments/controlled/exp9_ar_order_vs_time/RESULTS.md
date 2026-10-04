# exp9: does the AR copy follow position or time? Completed 2026-09-30

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp9_ar_order_vs_time/` |
|---|---|---|
| Test, TF + D (primary) | 287182 | `287182/cell-0{0,1}`, summary `summary-287182/` |
| Valid, TF (replication) | 287183 | `287183/cell-0{0,1}`, summary `summary-287183/` |
| Pilot, 64 rows | 287179 | `pilot-287179/` |

The protocol ([protocol.md](protocol.md)) was declared before any forward.
Its pilot disclosure notes that the summariser was run on the pilot with the
estimates suppressed.

## Acceptance

- **Identical-SID swaps are bit-exact.** Rows whose two most recent items
  have the same SID were swapped too, which leaves the input token-identical
  but in a different batch slot. All of them gave bit-identical code logits:
  393 / 393 and 202 / 202 (test), 375 / 375 and 241 / 241 (valid).
- **V-D0 passed.** The clean plain-beam arm reproduces exp6's
  `predictions_B_plain` 50-SID lists for 3,680 of 3,681 rows in both cells.
  The one exception is the last row of the split.

## The answer in one paragraph

**RQ-KMeans AR copies whatever sits last, and it is blind to time.**

- **Position, not item.** With the same two items in both orders, it prefers
  whichever item is last. A, the position effect, is 0.36–0.39 at digit 0. That
  is the same when the two items share a day and the target comes later (the
  data say they are exchangeable there) as when the last item really is later.
- **Arbitrary order moves half its decisions.** A swap that leaves every
  timestamp unchanged flips its first-digit decision in **50%** of those rows,
  and its top-1 recommended SID in 42%.
- **Tie averaging helps where time licenses it.** Averaging the model over the
  two orders the timestamps cannot tell apart raises the golden SID's
  log-prob by 0.02–0.03 nats in those rows. The same averaging hurts where the
  order is real (−0.02 to −0.03).

**RQ-VAE AR is only weakly positional** (A ≈ 0.07–0.10). It prefers the truly
later item about as much as the last position (B ≈ 0.07), so its recency is
partly carried by content.

## Primary estimands (test; valid in brackets)

| Estimand | Stratum | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|---|
| **A, position effect** (digit 0) | U | 0.381 [0.393] | 0.099 [0.100] |
| | T_later | **0.364** [0.354] | 0.068 [0.067] |
| | T_same | 0.281 [0.282] | 0.043 [0.044] |
| **P2** A(T_later) − A(U) | | −0.017 (−0.059, 0.027) [−0.039] | −0.031 (−0.058, −0.004) [−0.033] |
| **B, item effect** | U | 0.064 (0.039, 0.091) [0.054] | 0.068 (0.036, 0.098) [0.071] |
| | T_later | 0.020 (−0.040, 0.081) [0.063] | −0.012 [−0.021] |
| | T_same | 0.000 [0.039] | 0.065 [0.074] |
| Data asymmetry P(t = c1) − P(t = c2) | U | 0.077 [0.103] | 0.096 [0.124] |
| | T_later | 0.007 [0.000] | −0.033 [−0.022] |
| | T_same | 0.042 [0.096] | 0.091 [0.109] |
| **P3** top-1 digit-0 flip rate | T_later | **0.498** (0.452, 0.545) [0.500] | 0.183 (0.145, 0.223) [0.162] |
| | U | 0.527 | 0.224 |
| **P4** Δ_mix: tie average − clean (nats, whole SID) | T_later | **+0.027** (0.005, 0.048) [+0.020] | −0.008 (−0.024, 0.009) [−0.009] |
| | U | −0.019 (−0.030, −0.008) [−0.030] | −0.002 [−0.015] |
| | T_same | +0.003 [−0.038] | −0.026 [−0.026] |
| Δ_mix(T_later) − Δ_mix(U) | | **+0.045** (0.022, 0.069) [+0.050] | −0.006 [+0.006] |

Row counts, test, rows whose two items have distinct first digits:

| | RQ-KMeans | RQ-VAE |
|---|---|---|
| U | 1,840 | 1,650 |
| T_later | 456 | 389 |
| T_same | 524 | 364 |

Δ and Δ_mix use every eligible row: 3,059 (RQ-KMeans) and 3,250 (RQ-VAE).

## Secondary

**Golden log-prob, Δ = clean − swap, digit 0 (test):**

| | U | T_later | T_same |
|---|---|---|---|
| RQ-KMeans, mean Δ | +0.092 (0.071, 0.114) | −0.003 (−0.036, 0.030) | +0.058 |
| RQ-KMeans, mean \|Δ\| | 0.345 | 0.244 | 0.287 |
| RQ-VAE, mean Δ | +0.036 | +0.096 (0.045, 0.150) | +0.048 |
| RQ-VAE, mean \|Δ\| | 0.170 | 0.242 | 0.163 |

- **RQ-KMeans behaves as the data say it should.**
  - In U, putting the older item last hurts.
  - In T_later, the order does not matter on average, but it moves each row by
    0.24 nats.
  - In T_same, the ASIN-later item (r1) is more informative, and the swap
    hurts.
- **RQ-VAE is better with the clean order even in T_later**, where the data
  asymmetry at digit 0 is about 0 (−0.03). Its later digits and full SID say
  the same (whole-SID Δ = +0.11).

**Part D, plain beam search (test):**

| | U | T_later | T_same |
|---|---|---|---|
| RQ-KMeans, top-1 SID changes | 52% | **42%** | 34% |
| RQ-KMeans, HR@10 swap − clean | −0.7 pp (−1.6, 0.1) | +0.2 pp (−1.3, 1.6) | −1.0 pp |
| RQ-VAE, top-1 SID changes | 33% | 30% | 26% |
| RQ-VAE, HR@10 swap − clean | +0.2 pp | −0.7 pp | −0.7 pp |

- **Lists change, HR@10 does not.** The top-10 list (order or membership)
  changes in ≥ 97% of swapped rows, but HR@10 moves by at most about 1 pp in
  any stratum.
- **Baseline HR@10 differs by stratum.** It is 36.6% in T_same, 18.5% in U and
  14.4% in T_later (RQ-KMeans), in line with retrospective exp6.

## Expectations

| | Statement | RQ-KMeans | RQ-VAE |
|---|---|---|---|
| E1 | A > 0 in every stratum, CI excluding 0 | held | held |
| E2 | \|A(T_later) − A(U)\| < 0.10 (time-blind) | held (−0.02; valid −0.04) | held (−0.03). A small, significant drop on ties, still far under 0.10 |
| E3 | T_later: \|B\| < 0.05 and flip rate > 0.25 | held (0.02, 0.50) | **failed**: B held, flip rate 0.18 |
| E4 | Mean Δ d0 > 0 in U; CI includes 0 in T_later | held | **failed**: +0.10 in T_later |
| E5 | Δ_mix > 0 in T_later, and above U | held (+0.027; +0.045 vs U) | **failed** (−0.008) |

## What it means

- **"The model copies the most recent item" means "the last item in the
  prompt" for RQ-KMeans AR.**
  - Its preference moves with the position (A ≈ 0.37). It barely follows the
    item (B ≈ 0.06, about the data's own asymmetry in U).
  - It is the same whether or not the timestamps say the order is real.
  - With retrospective exp6 (the copy rate does not decay with age, ρ ≈ 0),
    that makes the RQ-KMeans model's recency a positional heuristic.
- **Within-day order is an input artefact the model is sensitive to.** Half of
  its first-digit decisions on tied rows, and 42% of its top-1
  recommendations, depend on ASIN order. Order-averaging over tied items is a
  free, timestamp-aware inference fix. It improves likelihood exactly where
  licensed, but it does not change HR@10 measurably.
- **RQ-VAE's recency is more content-based.**
  - Its first-digit choice is less tied to the last position.
  - It follows the truly later item about as strongly as the last position.
  - It is still better with the original order on ties with a later target.
    That hints at something in the ASIN order it uses, such as sibling
    listings.

  The RQ-VAE checkpoint is the one-off, not the sweep's model, and one cell
  cannot separate quantizer from training run.

## Limits

- **A swap moves two items at once.** It cannot separate "reads the last
  position" from "reads the item next to the response header".
- **Stratum differences are descriptive.** Strata differ in content too: ties
  are more often bundles. The 2×2 compares each row with itself.
- **One training seed per checkpoint.** The intervals exclude seed variance.
- **Part D uses plain beam search**, not the archived sampling decoder.
