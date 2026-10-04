# Last-digit read of a prefix-matching item: completed 2026-10-03

This experiment looks at the last digit. When a history item h matches the
target on every earlier digit, the two models do opposite things with h's
last-digit token.

- **RQ-KMeans 3×128 copies h's last digit.** Removing the attention edge from
  the last digit's readout to h's last-digit token (layers 14–27) lowers h's
  own code by **0.17 nats** [0.13, 0.22] on test and **0.34** [0.26, 0.42] on
  valid, beyond a matched control edge. It does so whether the target turns
  out to be h's SID or a sibling of h. When the target is a sibling, the copy
  competes with it: the same knockout raises the sibling target by **0.15**
  [0.06, 0.26] (valid 0.13).
- **RQ-VAE 4×128 pushes away from h's last digit when the target is a
  sibling.** In sibling rows, removing the same edge makes h's own code the
  top-1 more often, and the true sibling less often:

  | | Test | Valid |
  |---|---|---|
  | top-1 = h's code | **+11.8 pp** [5.3, 19.1] | +11.6 pp [3.7, 20.3] |
  | top-1 = the sibling target | **−10.7 pp** [−18.2, −4.1] | −9.3 pp [−18.6, −0.6] |
  | h's code, uncontrolled | +0.49 nats [0.14, 0.88] | |
  | sibling's code, uncontrolled | −0.36 nats [−0.69, −0.07] | |

  When the target *is* h's SID, the read supports it slightly (−0.03 nats
  [−0.08, −0.01]; h is already at p ≈ 0.99 there).
- **At clean, RQ-VAE rarely puts h's exact code first in sibling rows**
  (18 %, against 53 % for RQ-KMeans). This is the low "variant copy" rate of
  retrospective exp7. It depends strongly on h:

  | Top-1 = h's code at clean (test) | RQ-VAE | RQ-KMeans |
  |---|---|---|
  | h re-bought by some user in the training split | 92 % | 96 % |
  | h not re-bought | 22 % | 77 % |

**The declared primary for RQ-VAE is not met.** Pooled over all eligible rows
and control-subtracted (P1), it is +0.09 nats [−0.02, +0.25] on test and
+0.06 [−0.07, +0.25] on valid. In sibling rows it is +0.16 [−0.01, +0.43] and
+0.25 [−0.04, +0.67]. These have the exclusion sign, but every interval
includes 0. So "exclusion" for RQ-VAE rests on the secondary estimands:

- the top-1 flips, which exclude 0 on both splits
- the uncontrolled change on test

The effect varies a lot between rows. The 34 test sibling rows without a
control item carry much of the uncontrolled change. Those are rows whose
whole history shares the target's prefix, e.g. a run of filament purchases.

Protocol: [protocol.md](protocol.md), declared before any exp10 forward pass.
Intervals are paired user-bootstrap 95 % intervals (2,000 draws, seed
20260927) and exclude training-seed variance. Cells and splits are never
pooled.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp10_ar_last_digit_sibling/` |
|---|---|---|
| Test (primary) | 293619 | `293619-test/cell-0{0,1}` |
| Valid (replication) | 293620 | `293620-valid/cell-0{0,1}` |
| Summary | | `summary-293619-293620/` (`estimates.csv`, `report.md`) |
| Pilots (64 rows): 293611 failed in a post-run check (column-name clash), 293616 passed | | `pilot-29361{1,6}-test/` |

## Numerical acceptance

- **No-op controls.** `K0a_empty` and `K0b_future` are bit-identical to clean
  for every row of all four cell × split runs (635, 450, 648, 434 rows).
- **Agreement with exp4 (test).** Clean last-digit `logp_codes` agrees with
  exp4's test-split clean run for every eligible row: max difference 1.0e-6
  and 1.1e-6 nats (limit 0.05).
- **Control edge.** The matched control knockout (`KC_late`) moves ℓh by
  0.000–0.035 nats.
- **Late vs all layers.** All layers (`KH_all`) gives the same numbers as
  layers 14–27, to within 0.01 nats. As in exp4, layers 0–13 contribute
  nothing.

## Rows

| Cell, split | eligible | S_same | S_diff (near-dup) | without a control item |
|---|---|---|---|---|
| RQ-KMeans 3×128, test | 635 | 487 | 148 (51) | 90 |
| RQ-VAE 4×128, test | 450 | 178 | 272 (178) | 66 |
| RQ-KMeans 3×128, valid | 648 | 506 | 142 (50) | 84 |
| RQ-VAE 4×128, valid | 434 | 262 | 172 (121) | 66 |

## Expectations recorded before outcomes

- **X1** (P1 < 0 in both cells, interval excluding 0): **held for
  RQ-KMeans** (−0.17 test, −0.34 valid); **failed for RQ-VAE** (+0.09 test,
  +0.06 valid, both intervals including 0).
- **X2** (P2 > 0 in sibling rows, conditional on X1): **held for RQ-KMeans**
  (+0.15 [0.06, 0.26] test, +0.13 valid). For RQ-VAE, X1 failed. Its P2 has
  the opposite sign: −0.12 [−0.35, +0.02] test, −0.20 [−0.53, +0.03] valid.
  Without the control subtraction, −0.36 [−0.69, −0.07] on test. Removing
  h's read hurts the sibling.
- **X3** (clean top-1 = h in sibling rows rarer for RQ-VAE): **held** (18 %
  vs 53 % test, 19 % vs 46 % valid).

## Interpretation

- **RQ-KMeans.** Its exact-SID copy is rewarded in its data. 11 % of test
  targets share a history item's SID (retrospective exp7), and 77 % of S_*
  rows are S_same. So "copy the whole SID" scores often, and the model copies
  the last digit whatever the item.
- **RQ-VAE.** The dedup loop gave product variants distinct, arbitrary last
  digits. After a variant, the next purchase is more often a sibling than
  the same SID (272 vs 178 rows on test). The model copies the shared prefix
  and moves the last digit off h's code, except for items that users re-buy.
- **Not tested.** Whether this behaviour was learned from those training
  statistics. The re-buy split is observational.
- **Consequence.** The quantizer's post-processing changes what the
  recommender learns at the last digit, not only how it is scored.
  RQ-KMeans's collisions make exact copying pay. RQ-VAE's dedup makes
  last-digit exclusion pay.

## What this does not establish

- One edge is removed. Information about h's last digit can reach the
  readout by other positions; the RQ-VAE exclusion may be only partly
  carried by this edge.
- The re-buy split uses the training split's repeat events as a proxy for
  item type. It is a stratum, not an intervention.
- One checkpoint per cell; `oneoff_rqvae4cb128` is not the sweep's RQ-VAE
  model (CLAUDE.md).
