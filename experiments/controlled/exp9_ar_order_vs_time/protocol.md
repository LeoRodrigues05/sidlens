# exp9: does the AR copy follow position or time?

Declared 2026-09-30, before any forward of this experiment and before any
model output had been split by time.

## Why

exp3, exp4 and representation/exp1 found that both AR next-item models choose
the first digit largely by copying the most recent history item. That item's
first digit is the model's top-1 in 64% (RQ-KMeans 3×128) and 53% (RQ-VAE 4×128)
of test rows. But the models never see a timestamp. The prompt lists the history
"in chronological order", so "most recent" can only mean **last in the list**.

The frozen reviews keep a day for every event (`sidlens.data.timestamps`). What
that day shows about the data (descriptive only, computed before this protocol,
no model involved):

- **Same-day bursts are the norm.** In the test split the most recent item was
  reviewed on the target's own day in 40.8% of rows, and the two most recent
  items share a day in 43.9% of rows with at least two items.
- **Same-day order is ASIN order, not time.** Upstream stable-sorts by day, so
  ties keep raw-file order, and the raw dump is about 97% ASIN-sorted. 89% of
  same-day consecutive pairs are in ascending ASIN order.
- **Position and time come apart on ties.** Test split, rows whose two most
  recent items have different first digits:

  | Stratum | n | P(target d0 = r1 d0) | P(target d0 = r2 d0) |
  |---|---|---|---|
  | U: r1 on a later day than r2 | 1,840 | 0.123 | 0.046 |
  | T_later: r1, r2 same day, target later | 456 | 0.050 | 0.044 |
  | T_same: r1, r2, target all the same day | 524 | 0.164 | 0.122 |

  In **T_later** the data treat r1 and r2 as exchangeable: which one is "last"
  is an ASIN accident. The train split agrees (0.056 vs 0.061).

## Question

1. Does the models' reliance on the last item follow its **position**, or the
   **item**? A 2×2 design answers this: the same two items, in both orders.
2. How much do predictions depend on a within-day order that the data leave
   arbitrary?
3. Does a timestamp-aware inference rule help? The rule averages the model over
   the orders that the timestamps cannot distinguish.

## Cells, split, rows

- **Cells:** `next-item_best` (RQ-KMeans 3×128) and `oneoff_rqvae4cb128`
  (RQ-VAE 4×128). The frozen checkpoints' sha256 must equal the manifest.
- **Splits:** test is **primary** (the archived cohort, 3,681 rows). Valid
  (3,680) is a replication, run and reported separately, never pooled.
- **Wording:** eval, teacher-forced golden target, bf16.
- **Eligible rows:** history length ≥ 2 and SID(r1) ≠ SID(r2). r1 is the last
  history item and r2 the one before.
- **Strata** come from timestamps and are fixed before any forward:
  - **U:** day(r1) > day(r2)
  - **T_later:** day(r1) = day(r2) < day(target)
  - **T_same:** day(r1) = day(r2) = day(target)
- **Following estimands (A, B)** additionally need d0(r1) ≠ d0(r2).
- **Exactness controls:** rows with history length ≥ 2 and SID(r1) = SID(r2).
  This includes the duplicate records.

## Conditions

- **clean:** the row as upstream built it.
- **SWAP:** r1 and r2 exchange places. Nothing else changes: same length, same
  role map, and only the two items' SID tokens differ. The swapped example is
  rebuilt as an `ArExample` and re-encoded. A check proves the changed
  positions lie inside items L−1 and L−2.
- **SWAP on SID(r1) = SID(r2) rows:** token-identical to clean, run in a
  different batch slot. Every code logit must be bit-identical to clean. This
  is the batch-position invariance that lets a clean−SWAP difference be
  attributed to the swap.

Everything runs through `FixedShapeRunner` (fixed B × T_pad, explicit masks).

## Measures

Per digit d, at `predict_pos(0, d)`:

- `logp_codes`: the golden code's log-prob among digit d's codes
- `top1`: the model's top-1 code

Whole-SID log-prob = Σ_d logp_codes (teacher forced).

## Estimands

c1 and c2 are the digit-0 codes of r1 and r2. "Last" is whichever item sits at
the last position in a given order.

**2×2 main effects at digit 0** (rows with c1 ≠ c2), per stratum:

- **A, position effect:**
  A = ½[(1[top1_clean = c1] − 1[top1_clean = c2]) + (1[top1_swap = c2] − 1[top1_swap = c1])].
  The preference for whatever sits last. A purely positional model gives
  A > 0; an order-invariant model gives A = 0.
- **B, item effect:**
  B = ½[(1[top1_clean = c1] + 1[top1_swap = c1]) − (1[top1_clean = c2] + 1[top1_swap = c2])].
  The preference for r1, the item the timestamps call later or ASIN-later,
  wherever it is placed.

**Primary**

- **P1:** A in T_later, per cell. In T_later the data give no reason to prefer
  either position (data asymmetry +0.006).
- **P2:** A(T_later) − A(U), a paired bootstrap difference. A time-aware model
  would follow position less on ties; a time-blind one gives ≈ 0.
- **P3:** flip rate, P(top1_swap ≠ top1_clean) at digit 0, in T_later. This
  is the share of rows whose first-digit decision depends on an order the data
  cannot see.
- **P4, tie averaging:**
  Δ_mix = log(½ p_clean(golden SID) + ½ p_swap(golden SID)) − log p_clean(golden SID),
  in nats, whole SID, teacher forced. Reported in T_later and in U. The rule is
  only licensed where the timestamps say the order is arbitrary (T_later,
  T_same). U is the contrast, where the order is real.

**Secondary**

- Δ = logp_clean − logp_swap, at digit 0 and whole SID, per stratum: mean and
  mean |Δ|.
- B per stratum.
- Per-digit versions of Δ.
- Everything on the valid split.
- The data asymmetry P(target d0 = c1) − P(target d0 = c2) per stratum, beside
  A and B. This is a descriptive comparison of how the model and the data weight
  the two positions.

## Part D (secondary): plain beam search on clean and swapped prompts

Test split only. The decoder is the exp6 plain one:

- trie-constrained, 50 beams, length penalty 0, no sampling
- `use_model_defaults=False`, no repetition penalty
- exp6's batches (4 contiguous shards × 8)

The SWAP condition uses the same batches, with eligible rows swapped and the
rest unchanged.

- **V-D0 (acceptance):** the clean condition must reproduce exp6's
  `predictions_B_plain` 50-SID lists. Those are 283586/cell-00 and
  283621/cell-01, identical for ≥ 99% of rows. Otherwise Part D is reported as
  unvalidated.
- **Estimands per stratum:** exact-SID HR@10 (clean, SWAP and the difference),
  the share of rows whose top-10 list changes, and the share whose top-1 SID
  changes.

## Statistics

- A paired user bootstrap (`UserBootstrap`), 2,000 draws, seed 20260930. All
  strata, conditions and cells of a sampled user stay together within a split.
- The intervals exclude training-seed uncertainty, because each checkpoint was
  trained once.
- SID-level only; no collisions are resolved.
- AR only; no DiffGRM rows are joined.

## Expectations recorded before outcomes

- **E1:** A > 0 in every stratum and cell, with the interval excluding 0.
- **E2 (time-blind):** |A(T_later) − A(U)| < 0.10 in both cells.
- **E3:** In T_later, |B| < 0.05 and the flip rate P3 > 0.25 in both cells.
- **E4:** Mean Δ at digit 0 is > 0 in U, and its interval includes 0 in
  T_later.
- **E5:** Δ_mix > 0 in T_later, and Δ_mix(T_later) > Δ_mix(U).

## Limits

- Swapping moves two items. It does not remove the model's positional prior,
  and it cannot tell "reads position" apart from "reads the item adjacent to the
  response header".
- Strata differ in content as well as time: tied items are more often bundles.
  The 2×2 design compares each row with itself across orders, and stratum
  contrasts are descriptive of these rows.
- Teacher-forced scores are not HR; Part D is the decoding check.
- One training seed per checkpoint. The RQ-VAE AR checkpoint is the one-off,
  not the sweep's model.

## Pilot and disclosure (2026-09-30)

Pilot job 287179 ran 64 test rows per cell, parts TF and D.

- The identical-SID swap controls were bit-exact: 2 of 2 in cell 0, and none
  occurred in cell 1's first 64 rows.
- V-D0 reproduced exp6's plain-decoder lists 64 of 64 in both cells.

`summarize.py` was run on the pilot to test it. Only its shape and NaN checks
were printed, and no estimate was read. Nothing in the design changed after
the pilot.
