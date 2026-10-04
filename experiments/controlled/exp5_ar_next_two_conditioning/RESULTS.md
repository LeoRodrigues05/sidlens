# Next-two AR (two-item_best): item 2 is computed from item 1, completed 2026-09-27

Under teacher forcing on "sid1 ||| sid2", the surviving next-two AR recommender
(MQ 4×256) makes its second item depend strongly on the first:

- **Item 1 matters.** Clamping item 1 to a catalogue item with a different
  first digit lowers the golden first digit of item 2 by **0.60 nats** [0.54,
  0.66].
- **Item 1 matters more than recent history.** That is **0.40 nats more**
  [0.33, 0.46] than replacing the most recent history item, which costs 0.20.
- **The same copy mechanism links the two targets.** When item 1 already
  shares item 2's first d digits, changing item 1 from digit d onward costs
  item 2's digit d **0.43–0.73 nats**. When it does not, the cost is ~0. This
  is exp4's prefix-matched copying, operating between the two target slots.
- **Item 1's information reaches item 2 late.** It stays at item 1's own
  tokens until about layer 20, then moves in layers 20–26 into the separator
  (which predicts item 2's first digit) and item 2's tokens. The history →
  target transfer happens in the same window (exp3).

Protocol ([protocol.md](protocol.md)), declared before any forward of this
checkpoint. Cohort: 3,452 two-item test rows (1,606 users), which match the
archive (`gt1`/`gt2`, 3,452 / 3,452) and the frozen `.sem_ids`. Eval wording,
bf16. Δ is the golden code's log-prob over the digit's codes, clean minus
intervened. Intervals are paired user-bootstrap 95% percentile intervals
(2,000 draws, seed 20260927) and exclude training-seed variance.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp5_ar_next_two_conditioning/` |
|---|---|---|
| Pilot, 64 rows | 281259 | `pilot-281259/` |
| **Primary: Parts A + B** | 281263 | `281263/cell-00`, `summary-281263/` |

## Numerical acceptance

- **Patch controls:** no-op, full-layer restore and self-patch are bit-exact
  at **41,424 / 41,424** (max |Δ| = 0). Batch invariance is exact.
- **Independent path:** the clean top-1 rates per slot and digit match the
  independent `ar_capture` run of the same checkpoint (job 281055) to three
  decimals.
- **Inputs:** every intervened input was rebuilt by `encode`, and
  `check_replacement` proved that only the intended SID digits changed.

## Part A

| Item-2 digit | A1: clamp item 1 (m = 0) | Replace last history item | **A2: item 1 − history (paired)** |
|---|---|---|---|
| 1 | **+0.596** [0.535, 0.659] | +0.200 [0.168, 0.231] | **+0.396** [0.334, 0.461] |
| 2 | +0.173 [0.141, 0.204] | +0.018 [−0.002, 0.038] | +0.155 [0.119, 0.189] |
| 3 | +0.052 [0.034, 0.070] | +0.018 [0.004, 0.032] | +0.034 [0.014, 0.055] |
| 4 | +0.022 [0.013, 0.031] | +0.012 [0.005, 0.020] | +0.009 [−0.002, 0.021] |

For reference, replacing the last history item costs **item 1** 0.90 [0.84,
0.97] at its first digit. So the model leans on the last history item for item
1, and on item 1 for item 2.

**A3: how much of the effect remains when the clamp shares m leading digits
with the real item 1** (Δ at item 2's digits):

| m (rows) | digit 1 | digit 2 | digit 3 | digit 4 |
|---|---|---|---|---|
| 0 (3,452) | 0.596 | 0.173 | 0.052 | 0.022 |
| 1 (3,378) | 0.195 | 0.142 | 0.035 | 0.012 |
| 2 (2,036) | 0.065 | 0.015 | 0.046 | 0.010 |
| 3 (1,056) | 0.013 | −0.003 | 0.002 | 0.054 |

Keeping item 1's first code removes two thirds of the digit-1 effect. The
largest remaining effect at level m sits on digit m+1, the first changed digit
(the diagonal). That is again the copy signature. Missing controls:

- m = 1: 74
- m = 2: 1,416
- m = 3: 2,396

MQ digits are parallel partitions, so few items share 2–3 of item 1's four
partition codes.

**Secondary, the copy check.** Changing item 1 from digit d onward, and
reading Δ at item 2's digit d:

| Item-2 digit | Item 1 shares item 2's first d digits | Prefixes differ |
|---|---|---|
| 2 | **+0.728** [0.578, 0.882] (n = 537) | +0.031 [0.008, 0.054] (n = 2,841) |
| 3 | **+0.531** [0.357, 0.746] (182) | −0.001 [−0.014, 0.013] (1,854) |
| 4 | **+0.433** [0.251, 0.679] (125) | +0.003 [−0.006, 0.013] (931) |

In 287 of 3,452 rows (8.3%), item 1 and item 2 carry the same full SID.

## Part B: where item 1's information flows

- **Corrupted input:** item 1 clamped (m = 0).
- **Patch:** the clean residual at one layer and one position group.
- **Figure:** `summary-281263/figures/fig1_item1_patching`.

| Item-2 digit | Item-1 tokens hold ≥ 0.95 recovery through | Readout group reaches ≥ 0.5 at |
|---|---|---|
| 1 | layer 20 | layer 22 (`sep`, the separator; 0.67 → 0.91 by L25) |
| 2 | layer 20 | layer 21 (`slot1`, 0.64 → 0.96 by L23) |
| 3 | layer 8 (≥ 0.86 through L20) | layer 21 (`slot1`, 0.60) |
| 4 | layer 8 (≥ 0.73 through L20) | layer 20 (`slot1`, 0.64) |

For item 2's later digits, a small share of the information sits in item 2's
own tokens from the middle layers onward. For digit 4 that share is 0.3–0.4 by
layer 12–19. The separator carries nothing for digits 2–4.

## Expectations recorded before outcomes

- **E1** (A1 > 0.1 nats): held, at 0.60.
- **A2:** no direction was declared. Item 1 matters more than the most recent
  history item.

## Limits

- **Teacher-forced dependence.** This measures dependence on a *given* item 1.
  It is not the archived two-pass decoding, and not HR.
- **Replacement effects** mix removing item 1's evidence with adding the
  control's.
- **One checkpoint.** There is no joint-block or diffusion counterpart, so this
  says nothing about paradigms.
- **MQ prefixes** are intersections of parallel partitions.
