# Collision credit in copying: completed 2026-10-03

exp6's copy benefit comes almost entirely from targets in the same product
line as a history item. For RQ-KMeans 3×128, most of the benefit at rank 1 is
collision credit.

- **RQ-KMeans 3×128: collision partners carry the copy effect.** 11.3 % of
  test targets are a different item that shares its SID with a history item.
  Those rows supply:
  - 42 % of the baseline's SID hits at HR@10
  - 56 % [46, 68] of the HR@10 loss when copying is blocked (`C_all`)

  At rank 1 the copied SID usually names several items. Blocking copying
  costs these rows 34.5 pp of SID HR@1 but only 6.0 pp of item-level HR@1.
  Over all rows, only **28 %** [22, 36] of the SID-level HR@1 copy benefit
  survives at item level. At HR@10 it is 83 % [67, 101], because a small
  bucket copied to rank 1 still fits in the 10 item slots.
- **RQ-VAE 4×128: near-duplicate variants carry the copy effect.** 9.9 % of
  targets are a same-brand, near-identical-title variant of a history item
  with a different SID. That different SID is the work of the last-digit
  loop (structure exp3). These rows supply:
  - **80 %** [68, 95] of the `C_all` HR@10 loss
  - 94 % of the "copying helps new targets" effect that exp6 reported

  Item and SID metrics agree (E2 = 0.99 at HR@10, 0.88 at HR@1).
- **Truly new targets gain almost nothing from copying.** These are 83 % and
  85 % of rows: no same item, no same SID, and no near-duplicate in the
  history. Blocking copying changes their HR@10 by:
  - RQ-KMeans: −1.0 pp SID, −0.2 pp item [−0.8, +0.3]
  - RQ-VAE: −0.2 pp SID, −0.1 pp item

  For RQ-KMeans it *raises* their SID HR@1 by 0.7 pp [0.1, 1.2].
- The plain beam-search decoder gives the same picture (E1 0.55, E2@1 0.30,
  RQ-VAE variant share 0.81).

So "copying helps recommendation" should be stated as: copying recovers
repurchases and product variants. For RQ-KMeans, half of what it recovers at
the top rank is credit for a SID that several items share.

Protocol: [protocol.md](protocol.md), declared after structure exp3 and before
any exp6 prediction was re-scored. Amendment A1 (HR@1) was added after the
first full run, and a 200-row smoke of the RQ-KMeans baseline was printed
before it. Both are disclosed there. Intervals are paired user-bootstrap 95 %
intervals (1,606 users, 2,000 draws, seed 20260927) and exclude training-seed
variance.

| Run | Job | Output under `$SIDLENS_WORK/derived/retrospective/exp7_collision_credit/` |
|---|---|---|
| Final, with A1 | 293597 | `293597/result/` (`estimates.csv`, `report.md`, `scored__*.parquet`, `rows__*.parquet`) |
| First full run (no HR@1; identical other numbers) | 293596 | `293596/result/` |
| Failed (report formatter bug; estimates were written) | 293592 | `293592/` |

## Numerical acceptance

- Re-scored `C_all − B` HR@10 equals exp6's published values: −5.678
  against −5.68 pp, and −3.314 against −3.31 pp.
- Every target and history item id in the AR rows maps to the SID shown in
  the prompt (both cells, all 3,681 rows).

## Row classes (data only)

| Class | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|
| same_item: target item in the history | 2.9 % | 2.9 % |
| sid_partner: target SID in the history, item not | 11.3 % | 1.9 % (the identical-input pairs) |
| nd_variant: near-duplicate of a history item, SID not in the history | 3.2 % | 9.9 % |
| new | 82.6 % | 85.3 % |

## Archived decoder, `C_all − B`, by class

| | RQ-KMeans: baseline SID / item HR@10 | Δ SID / Δ item HR@10 (pp) | RQ-VAE: baseline SID / item HR@10 | Δ SID / Δ item (pp) |
|---|---|---|---|---|
| all | 24.0 / 12.6 % | −5.7 / −4.7 | 14.4 / 14.2 % | −3.3 / −3.3 |
| same_item | 95.3 / 91.6 | −23.4 / −25.2 | 86.9 / 86.9 | −15.0 / −15.4 |
| sid_partner | 89.9 / 57.5 | −28.3 / −28.5 | 98.6 / 98.6 | −2.8 / −2.8 |
| nd_variant | 58.8 / 21.9 | −31.1 / −17.3 | 49.6 / 49.6 | −27.0 / −27.4 |
| new | 11.2 / 3.4 | −1.0 / −0.2 | 5.9 / 5.8 | −0.2 / −0.1 |

At HR@1 (A1, post hoc):

| | RQ-KMeans: baseline SID / item HR@1 | Δ SID / Δ item (pp) | RQ-VAE: baseline SID / item HR@1 | Δ SID / Δ item (pp) |
|---|---|---|---|---|
| all | 11.2 / 3.7 % | −4.4 / −1.2 | 7.2 / 5.2 % | −2.3 / −2.0 |
| same_item | 84.1 / 52.0 | −24.3 / −19.6 | 83.2 / 57.5 | −24.3 / −21.0 |
| sid_partner | 61.8 / 13.8 | −34.5 / −6.0 | 98.6 / 49.3 | −11.3 / −5.6 |
| nd_variant | 14.3 / 4.8 | −10.9 / −3.5 | 16.0 / 15.2 | −12.4 / −11.7 |
| new | 1.6 / 0.5 | **+0.7** / +0.1 | 1.6 / 1.3 | −0.2 / −0.2 |

## Expectations recorded before outcomes

- **X1** (E1 ≥ 0.5 for RQ-KMeans, < 0.15 for RQ-VAE): **held**. The values
  are 0.56 [0.46, 0.68] and 0.016 [0.00, 0.04]. The RQ-KMeans interval
  includes values below 0.5.
- **X2** (E2 ≤ 0.67 for RQ-KMeans, within [0.9, 1.1] for RQ-VAE): **failed
  for RQ-KMeans** at HR@10 (0.83 [0.67, 1.01]); held for RQ-VAE (0.99).
  Reason: item HR@10 credits a whole bucket that fits in the first 10 item
  slots. At HR@1 (A1, post hoc) the RQ-KMeans ratio is 0.28.
- **X3** (RQ-VAE nd_variant share of the `C_later` loss on new-or-variant
  rows ≥ 2 × its row share): **held**. The share is 0.94 [0.80, 1.12] against
  a row share of 0.10.
- **X4** (variant copy ≥ 20 % of nd_variant rows, lowered by `C_all`):
  **held for RQ-KMeans** (29 % → 12 %). **Failed for RQ-VAE** (8.0 % → 8.3 %):
  - The RQ-VAE model rarely outputs the history variant's exact SID first,
    yet it hits the variant target in the top 10 half the time (49.6 %).
  - Blocking copying costs those rows 27 pp.
  - So it copies the shared prefix and then rarely picks the history item's
    own last digit. How it avoids that digit is not tested here.

## What this changes

- **exp6 and the paper's RQ4.** "Copying helps new targets (2.1 / 3.0 pp)"
  should become "copying helps near-duplicate variants of history items". On
  truly new targets the effect is within ±0.5 pp at item level.
- **Collision credit is a metric problem, not a model property.** Exact-SID
  HR@1 credits the RQ-KMeans model for copying a SID that its target shares
  with a different, already-bought item. Report item-level HR beside SID HR
  wherever RQ-KMeans 3×128 is compared with a family that has few collisions.
- **RQ-VAE's arbitrary last digit is where its variant targets are decided.**
  The loop gave each variant a last digit unrelated to its content. The
  model gets the first three digits by copying and must still choose the
  last one. That choice is the open mechanistic question below.

## Follow-up: controlled exp10

Controlled exp10 tested how the RQ-VAE model avoids the history variant's own
last digit (X4). It knocked out the edge readout(D−1) → the matching item's
last-digit token. RQ-KMeans copies that digit. For RQ-VAE, removing the edge
in sibling rows makes the history item's exact code top-1 more often:
+11.8 pp test, +11.6 pp valid, both intervals excluding 0. The declared
control-subtracted primary has the same sign, but its interval includes 0.
See [../../controlled/exp10_ar_last_digit_sibling/RESULTS.md](../../controlled/exp10_ar_last_digit_sibling/RESULTS.md).
