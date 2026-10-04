# Retrospective exp6: time in the data, and in the outputs we already have

Declared 2026-09-30. At that point the timestamps had been reconstructed
(`sidlens.data.timestamps`) and the data-only facts listed under "Seen before
this protocol" had been computed. No model output from exp3, exp4, exp6, exp7
or representation/exp1 had been split by time. CPU only: this audit reads
frozen data and results that already exist.

## Why

Every intervention so far treats the history as a sequence. Both paradigms'
first-digit choice leans on "the most recent item", and copying helps
recommendations (exp6). But no model is ever given a time. The AR prompt says
"chronological", and DiffGRM has a recency position. So the question is whether
the models' recency matches the recency in the data, and which parts of the
earlier findings are really about time.

## Seen before this protocol (data only, no model)

These facts are from the test split, RQ-KMeans 3×128 SIDs:

- The splits are consecutive periods:
  - train targets up to 2017-06-11
  - valid 2017-06-13 → 2017-12-09
  - test 2017-12-11 → 2018-09-28 (1st–99th percentile)
- Same-day rows. The most recent item is on the target's day in 40.8% of test
  rows (valid 42.8%, train 50.5%). The median gap is 29 days (test), 18 (valid)
  and 0 (train).
- Tie order. Within a day the order is raw-file order, which is mostly ASIN
  order: 89% of same-day consecutive pairs.
- Duplicate records. 2,653 of 43,102 events repeat a (user, item, day) review.
  2.7% of test targets duplicate a history event (valid 5.1%, train 8.0%).
- The first-digit match with the most recent item, by gap:
  - same day: 0.326
  - 1–7 days: 0.13
  - 8–30 days: 0.10
  - 31–180 days: 0.05
  - over 180 days: 0.035
- Repeat-SID targets (14.2% of test rows) are same-day in 81% of cases, and
  duplicate records in 19%.
- With r1 and r2 on the same day and the target later, r1 and r2 are equally
  informative (0.050 vs 0.044).
- The DiffGRM cohort (leave-last-out) is not a time split. Targets run from
  2013-11 to 2018-09, and 43.4% are same-day with the most recent item.

## Inputs

**Data:**

- `frozen/data/{reviews,interactions,sequences,splits,id_maps}` via
  `sidlens.data.timestamps` (sha256s in `inputs.json`)
- the RQ-KMeans 3×128 and RQ-VAE 4×128 SID tables

**Results** (under `derived/controlled/`, sha256 of every file read recorded):

| Source | Runs | What is read |
|---|---|---|
| exp3 | `280961/cell-0{0,1}` (test) | `clean.parquet`, `part_a.parquet` |
| exp4 | `281261/cell-0{0,1}` (valid), `281394/cell-0{0,1}` (test) | `clean.parquet`, `part_r.parquet` |
| exp6 | `283114/cell-0{0,1}` | `predictions_B`, `predictions_C_all` |
| exp6 | `283586/cell-00`, `283621/cell-01` | `predictions_B_plain`, `predictions_C_all_plain` |
| exp7 | `283124/cell-0{0,1}` | `clean.parquet`, `part_a.parquet` |

## Definitions

- **Gap bins.** gap = day(target) − day(item), in days. The bins are
  `sidlens.data.timestamps.GAP_BINS`: same day, 1–30 d, 31–365 d, > 365 d.
- **r1.** The most recent history item; gap1 is its gap.
- **match0.** The item's first digit equals the target's.
- **Copy (at digit 0).** The model's top-1 first-digit code equals r1's.
- **Target types (AR)**, from timestamps and SIDs:
  - `dup`: the target duplicates a history event, i.e. the same item on the
    same day
  - `repeat_same_day`: the target SID is in the history, gap1 = 0, not a
    duplicate
  - `repeat_later`: the target SID is in the history, gap1 > 0
  - `new_same_day`: the target SID is not in the history, gap1 = 0
  - `new_later`: the target SID is not in the history, gap1 > 0

## Part D: data (no model)

- **D1.** Per split and for the DiffGRM cohort:
  - the gap1 distribution over bins
  - the tie12 rate
  - the share of history items on the target's day
  - target-type shares
- **D2.** Within-day ASIN ordering, the rate over all users.
- **D3.** Informativeness by position and time. For every (row, history item)
  pair, P(match0) and P(item SID = target SID), by recency (1–10) × gap bin,
  on the test and train splits, for both SID variants. **Time beyond position:**
  the drop in deviance when a logistic model of match0 on C(recency) gains
  C(gap bin), reported as McFadden R² for C(recency), for C(gap bin), and for
  both. The reverse direction (position beyond time) is reported too.

## Part M: the models' outputs, split by time

- **M1: copy calibration.** Per gap1 bin, for:
  - AR test: exp3 `clean`
  - AR valid: exp4 `clean`
  - DiffGRM: exp7 `clean`, state `full`, digit 0

  report the model's copy rate c, the data rate v = P(match0 of r1), the golden
  top-1 rate, and the mean golden log-prob at digit 0. **Primary:**
  ρ = (c_same − c_>365) / (v_same − v_>365). ρ ≈ 0 means the models copy
  regardless of how old the item is; ρ ≈ 1 means they copy as the data warrant.
  Also reported: the over-copy c − v per bin.
- **M2: reliance by age at fixed position.** Use exp3 A (test) and exp7 A. At
  recency 1 and 2, with m = 0, take Δ = clean − replaced golden log-prob at
  digit 0, by the replaced item's gap bin, split by the item's match0.
  - AR, exp3: `logp_codes`
  - DiffGRM, exp7: `logp` at state `full`

  **Primary:** at recency 1 within match0 = True, Δ(same day) − Δ(> 365 d).
- **M3: where the copy benefit lives (exp6).** Per target type:
  - exact-SID HR@10 at baseline
  - its change under C_all, for the archived decoder and the plain decoder

  **Primary:** the share of the all-rows HR@10 loss (Σ over rows of the hit
  change) that comes from rows with gap1 = 0, and from `dup` rows.
- **M4: the valid vs test gap (exp4 H1).** H1 is the matching mean minus the
  non-matching mean of Δ_d over d ≥ 1, as exp4 defined it. Per split, it is
  reported:
  - overall
  - within r1 same-day vs later
  - within dup vs not

  and a test estimate post-stratified to valid's joint distribution of
  (same-day, dup, match). **Primary:** the valid/test ratio of H1, raw and
  post-stratified.

## Statistics

- A paired user bootstrap, 2,000 draws, seed 20260930, within each split or
  cohort. The ratios ρ and valid/test are bootstrapped as ratios of the
  resampled estimates. For the valid/test ratio, the valid and test users are
  resampled independently, because the two splits share users, so this is a
  conservative approximation.
- The intervals exclude training-seed uncertainty.
- SID-level and item-level targets are kept apart; the AR and DiffGRM cohorts
  are never joined.

## Expectations recorded before outcomes

- **X1:** Time adds to position. In D3 the R² gain from adding C(gap bin) to
  C(recency) is at least half the R² of C(recency) alone.
- **X2:** The models are largely time-blind: ρ < 0.5 for both AR cells and
  both DiffGRM cells.
- **X3:** They over-copy old items. At > 365 d the AR copy rate exceeds the data
  rate by > 0.25 (RQ-KMeans) and > 0.15 (RQ-VAE).
- **X4:** The copy benefit is a same-day phenomenon. At least 70% of the exp6
  all-rows HR@10 loss under C_all (archived decoder) comes from gap1 = 0 rows.
- **X5:** At recency 1 with match0 = True, Δ(same day) and Δ(> 365 d) differ by
  less than 30% of their mean. The model's reliance on a matching item does not
  depend on how old it is.
- **X6:** Post-stratification moves the exp4 valid/test ratio toward 1.

## Limits

- Associational. The models' outputs are split by time strata, but the time
  strata also differ in content: same-day items are often bundles. exp9 is the
  controlled counterpart for order.
- Amazon review days, not purchase times. A review can lag a purchase by weeks.
- The exp6 archived decoder samples (see exp6); the plain decoder is reported
  beside it.

## Disclosure (2026-09-30)

Before the declared run, `run.py` was executed once into a scratch directory,
to find crashes. That run printed nothing but timings, and no estimate from it
was read. The code was fixed twice (a pandas `.copy` name clash, and the
sklearn 1.8 spelling of "no penalty"); neither fix touched an estimand.
