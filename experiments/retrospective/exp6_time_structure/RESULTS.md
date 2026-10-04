# Retrospective exp6: time in the data and in existing outputs, completed 2026-09-30

Job 287189 ran on CPU (cn node, gate passed). The output is in
`$SIDLENS_WORK/derived/retrospective/exp6_time_structure/287189/result/`:
`estimates.csv`, `figures/`, `inputs.json` (sha256 of every result file read) and
`time_rows__*.parquet` (the per-row time tables). The protocol
([protocol.md](protocol.md)) was declared before any model output was split by
time. Its disclosure records one crash-test run whose estimates were not read.

## The answer in one paragraph

**In the data, "recent" means "same day", not "last in the list".** Once an
item's age is known, its position adds almost nothing about the target. Both
models see only position, and they copy the last item regardless of its age:

- RQ-KMeans AR copies the most recent item's first digit in 67% of rows,
  whether that item is from the target's own day or more than a year older.
  The data warrant 33% and 3%.
- The copy benefit in decoding (exp6) comes almost entirely from same-day rows.
- A sixth of the RQ-VAE model's hits are duplicated review records.
- The valid/test effect-size gap (open question 4) is partly explained by
  valid having twice as many duplicate records.

## Data (no model)

**Splits are consecutive periods** (1st–99th percentile of target days):

| Split | Target days | Same-day rows | Median gap1 | Duplicate-record targets |
|---|---|---|---|---|
| train | → 2017-06-11 | 50.5% | 0 d | 8.0% |
| valid | 2017-06-13 → 2017-12-09 | 42.8% | 18 d | 5.1% |
| test | 2017-12-11 → 2018-09-28 | 40.8% | 29 d | 2.7% |
| DiffGRM cohort (leave-last-out, not a time split) | 2013-11 → 2018-09 | 43.4% | 19 d | 4.4% |

gap1 is the number of days between the most recent history item and the
target.

- **Within a day, order is ASIN order (D2).** 87.9% of the 15,670 same-day
  consecutive pairs are in ascending ASIN order. Upstream's stable sort keeps
  raw-file order on ties, and the raw file is ASIN-sorted.
- **Target types (test, RQ-KMeans SIDs):**

  | Type | Share |
  |---|---|
  | new, later day | 56.6% |
  | new, same day | 29.3% |
  | repeat SID, same day | 8.8% |
  | duplicate record | 2.7% |
  | repeat SID, later day | 2.6% |

  The model was fine-tuned on a train split where 8.0% of targets are
  duplicate records, a literal copy of a history event.

**Time explains the history's informativeness; position barely does (D3,
X1 held).** The label is P(history item shares the target's first digit), over
(row, history item) pairs. McFadden R² of logistic models:

| Cohort | C(recency) | C(gap bin) | both | time beyond position | position beyond time |
|---|---|---|---|---|---|
| AR test, RQ-KMeans | 0.034 | 0.160 | 0.161 | **0.127** | 0.001 |
| AR test, RQ-VAE | 0.015 | 0.115 | 0.116 | **0.101** | 0.001 |
| AR train, RQ-KMeans | 0.029 | 0.147 | 0.149 | 0.120 | 0.002 |
| DiffGRM, RQ-KMeans | 0.030 | 0.160 | 0.162 | 0.131 | – |

At a fixed age, informativeness is flat in recency. On the test split,
RQ-KMeans, a same-day item shares the target's first digit 0.33 at recency 1
and 0.29 at recency 7. An item older than a year shares it 0.02–0.04 at any
recency (`fig1_informativeness_position_time`).

## M1: the models copy regardless of age (X2, X3 held)

The copy rate is the model's top-1 first digit equal to the most recent
item's. The data rate is P(target first digit equal to it). By gap1:

| Cohort | Model | Rate | same day | 1–30 d | 31–365 d | > 365 d | ρ |
|---|---|---|---|---|---|---|---|
| AR test | RQ-KMeans | copy | 0.671 | 0.601 | 0.609 | 0.667 | **0.01** [−0.15, 0.16] |
| | | data | 0.326 | 0.108 | 0.051 | 0.027 | |
| AR test | RQ-VAE | copy | 0.646 | 0.501 | 0.429 | 0.482 | **0.40** [0.29, 0.52] |
| | | data | 0.498 | 0.259 | 0.146 | 0.090 | |
| AR valid | RQ-KMeans | copy | 0.673 | 0.613 | 0.627 | 0.675 | −0.01 [−0.17, 0.15] |
| AR valid | RQ-VAE | copy | 0.599 | 0.440 | 0.438 | 0.503 | 0.27 [0.14, 0.42] |
| DiffGRM | RQ-KMeans | copy | 0.410 | 0.279 | 0.247 | 0.302 | 0.33 [0.22, 0.44] |
| | | data | 0.357 | 0.104 | 0.048 | 0.030 | |
| DiffGRM | RQ-VAE | copy | 0.486 | 0.341 | 0.316 | 0.398 | 0.26 [0.15, 0.37] |

ρ = (copy_same − copy_>365) / (data_same − data_>365). A model that copies as
the data warrant gives 1; one that ignores age gives 0.

- **RQ-KMeans AR is completely time-blind** (ρ ≈ 0 on both splits). It
  over-copies items older than a year by 0.64 [0.60, 0.68] (test).
- **The others track about a quarter to 40% of the data's decay.** Since they
  see no time, that must come from content cues.
- **The copy curve is U-shaped in every cell.** It is higher at > 365 d than at
  31–365 d, so its dependence on gap is not a time signal.
- **Golden first-digit accuracy falls with age:**
  - RQ-KMeans AR: 0.35 same-day → 0.03 at > 365 d
  - RQ-VAE AR: 0.51 → 0.13

  So the models' first-digit hits are mostly same-day hits.

## M2: reliance on a matching item by its age (X5 mixed)

Take Δ = clean − replaced golden first-digit log-prob, replacing the item at
recency 1 (exp3 A / exp7 A, m = 0), when the item shares the target's first
digit:

| Cohort | same day | 1–30 d | 31–365 d | > 365 d | same − > 365 |
|---|---|---|---|---|---|
| AR RQ-KMeans | 2.27 | 1.87 | 1.66 | 1.64 (n = 18) | 0.62 [−0.10, 1.26] |
| AR RQ-VAE | 1.46 | 1.24 | 0.96 | 0.73 | **0.73** [0.48, 0.96] |
| DiffGRM RQ-KMeans | 2.12 | 1.64 | 1.40 | 1.85 | 0.27 [−0.23, 0.77] |
| DiffGRM RQ-VAE | 1.62 | 0.98 | 1.03 | 1.09 | 0.53 [0.25, 0.80] |

X5 (within 30% of the mean) **held for DiffGRM RQ-KMeans**. It was **borderline
for AR RQ-KMeans** (32%, interval includes 0) and **failed for both RQ-VAE
cells**.

- **At recency 2 the picture reverses for RQ-KMeans AR.** A matching item older
  than a year is relied on more than a same-day one: 0.77 vs 0.43,
  −0.34 [−0.64, −0.05]. The dependence on age is not a monotone time signal.
- **Content is the likely source.** Same-day histories are bursts of similar
  items, so the model is more confident when several recent items agree. This
  is associational; exp9 tests order directly.
- **A non-matching recent item also matters.** Replacing it still costs 0.1–0.4
  nats: it carries evidence beyond its first digit.

## M3: where the copy benefit lives (exp6; X4 held)

This is exact-SID HR@10 on the test split, under the archived decoder, with the
plain decoder in brackets.

| Target type | Share of rows | Baseline HR@10, RQ-KMeans | Δ blocking copies, RQ-KMeans | Baseline HR@10, RQ-VAE | Δ blocking copies, RQ-VAE |
|---|---|---|---|---|---|
| duplicate record | 2.7% | 98.0% | −23.0 pp | 91.0% | −16.0 pp |
| repeat, same day | 8.8% / 1.9% | 93.5% | −27.8 | 98.6% | −2.8 |
| repeat, later day | 2.6% / 0.2% | 75.3% | −29.9 | 29% (n = 7) | 0.0 |
| new, same day | 29.3% / 36.1% | 21.3% | **−6.4** | 19.5% | **−6.7** |
| new, later day | 56.6% / 59.0% | 8.7% | **+0.1** [−0.9, 1.1] | 4.9% | −0.7 [−1.4, 0.0] |

The shares are RQ-KMeans / RQ-VAE, where the two differ; they differ because
SID collisions differ.

- **Same-day rows supply the copy benefit.** Rows with gap1 = 0 account for
  **87%** [77, 99] (RQ-KMeans) and **88%** [77, 100] (RQ-VAE) of the all-rows
  HR@10 loss when copying is blocked. The plain decoder gives 83% and 84%.
- **They also supply most baseline hits:** 71% and 79% of all HR@10 hits,
  against 41% of rows.
- **Duplicate records alone supply 11% (RQ-KMeans) and 17% (RQ-VAE) of all
  HR@10 hits**, at 2.7% of rows.
- **Copying does nothing for a new item on a later day**, which is the largest
  target type.

## M4: the valid vs test gap in exp4 H1 (X6 held)

| | Valid H1 | Test H1 | Ratio |
|---|---|---|---|
| RQ-KMeans, raw | 1.34 | 0.89 | 1.51 [1.23, 1.83] |
| RQ-KMeans, test post-stratified to valid's composition | | 1.07 | **1.25** [1.07, 1.51] |
| RQ-VAE, raw | 1.04 | 0.63 | 1.65 [1.38, 1.96] |
| RQ-VAE, post-stratified | | 0.79 | **1.32** [1.13, 1.54] |

Post-stratification uses (duplicate record, same-day, later) × match.

- **Composition explains about half the gap.** Duplicate records are 30% of
  matching pairs on valid but 16% on test (RQ-KMeans), and 21% vs 9% (RQ-VAE).
  Their H1 is about 2 nats; same-day non-duplicates give about 0.8, and later
  days 0.14–0.83.
- **About half remains** after reweighting. Within the later-day stratum, valid
  still exceeds test (0.83 vs 0.35 RQ-KMeans, 0.36 vs 0.14 RQ-VAE).

## Expectations

| | Statement | Outcome |
|---|---|---|
| X1 | time adds ≥ half of position's R² | **held**: time adds 3.7–6.9× position's R², and position adds ≤ 0.002 beyond time |
| X2 | ρ < 0.5 in all four cells | **held**: 0.01, 0.40, 0.33, 0.26 |
| X3 | over-copy at > 365 d > 0.25 / 0.15 | **held**: 0.64 / 0.39 |
| X4 | ≥ 70% of the HR@10 loss from gap1 = 0 | **held**: 87% / 88% |
| X5 | matching-item reliance within 30% across age | **mixed**: held for DiffGRM RQ-KMeans, borderline for AR RQ-KMeans, failed for both RQ-VAE cells |
| X6 | post-stratification moves the ratio toward 1 | **held**: 1.51 → 1.25 and 1.65 → 1.32 |

## What this changes in the earlier findings

- **"Recency"** (exp3, exp7) is, in the data, **same-day co-occurrence**. The
  models implement it as "last in the list", so they apply it to year-old items
  as confidently as to same-day ones (RQ-KMeans AR).
- **"Copying helps" (exp6)** is true on same-day and repeat rows. It is neutral
  for new items on later days. Part of the benefit is **duplicated reviews**,
  a data artefact that train (8%), valid (5%) and test (3%) all contain.
  Evaluation numbers that include duplicate-record targets overstate what a
  recommender learns.
- **Open question 4** (valid effects about 2× test): about half comes from the
  duplicate-record and same-day composition of the two periods.

## Limits

- **Associational.** Time strata also differ in content.
- **Review days are not purchase days.** A review can lag a purchase.
- **Small cells.** The > 365 d matching cells at recency 1 are small (n = 18 to
  82).
- **Two SID tables**, one checkpoint each; the intervals exclude training-seed
  variance.
- **Figure legend.** In `fig2_copy_calibration` the legend names only the
  DiffGRM runs. Colours are by quantizer in every panel (blue RQ-KMeans, orange
  RQ-VAE); solid lines are model copy rates and dotted lines are data rates.
