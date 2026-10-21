# N2: how the AR model matches the prefix (protocol)

Declared 2026-10-21, before any of these measurements was made on a grid
model. Plan: `docs/plans/COPYING_GRID_PLAN.md` §4.3. These change internal
attention inputs in their own conditions only; the prompt and the weights are
never changed, and every comparison is with the unmodified run.

## Runs and rows

Every AR next-item grid run at its best state (both catalogs); Industrial all
test rows, Office the fixed 6,297-user sample. Designs are N1's: (row, digit
d >= 2, the most recent history item h matching the target's first d digits,
no other item supporting h's digit-d code), with N1's `suffix` and `none`
replacements (catalog items where they exist; `synthetic` flagged).
Teacher-forced at the golden prefix; the readout is the position that predicts
digit d. Late layers: 14-27 of 28 (exp4: every copy read is there); early:
0-13.

## Measurements per design

- (a) Attention from the readout to h's digit-d token, per layer and head, in
  the full, suffix and none inputs.
- (b) In the full input, replace only the key input (or only the value input)
  at h's digit-d token in the late layers with the suffix input's. Outcome:
  log p(h's digit-d code). Prediction if the match is decided at the key: the
  key replacement removes most of support(full) and the value replacement
  little.
- (c) Block, in the early layers, the attention from h's digit-d token to h's
  earlier digit tokens, in the full, suffix and none inputs. Outcomes: the copy
  support under the block and rho (N1's ratio) under the block.
- (d) Donor: the same row with the golden target replaced by another history
  item h2 (h2's first d digits differ from the target's, its digit-d code from
  h's). Replace the late-layer query input at the readout with the donor's.
  Outcomes: attention to h2's vs h's digit-d token, log p(h2's code) and
  log p(h's code).
- (e) Direct logit attribution at the readout of each late head's output to
  h's digit-d code (minus the mean over digit d's codes), through the final
  norm's scale in the full input.
- (f) Induction scores of the 336 heads of the base Qwen2.5-1.5B-Instruct on
  repeated random token sequences (`induction.py`, computed once), against
  each grid run's per-head copy effect from controlled exp4's head screen.

## Estimands (paired user bootstrap, 2,000 draws, seed 20261021)

- Primary: share of support(full) removed by the key replacement, and by the
  value replacement (ratio of means), catalog designs.
- Attention to h's token: full minus none, suffix minus none, summed over late
  heads.
- rho under the early block minus rho without it.
- Query swap: change in attention to h2's token and in log p(h2's code).
- DLA: the top late heads' share of the summed DLA; Spearman correlation of
  base-model induction score with exp4's per-head copy effect (across heads).

## Not claimed

Patching one layer range's inputs shows where information is read, not a
complete circuit. Intervals exclude training-seed variance.
