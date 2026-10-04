# Controlled exp10: at the last digit, does the model copy a sibling's code?

Declared 2026-10-03, after retrospective exp7 and before any exp10 forward
pass. GPU, CIAI cluster. Uses the exp4 runner (fixed shapes, explicit
attention-edge masks, bit-exact no-op controls).

## Why

Retrospective exp7 found that, for RQ-VAE 4×128, near-duplicate variants of a
history item carry 80 % of exp6's copy effect. RQ-VAE's dedup loop gave those
variants different last digits (structure exp3). Yet the RQ-VAE model rarely
puts the history variant's exact SID first: 8 % of variant rows under the
archived decoder, 9 % under plain beam search, and blocking copy reads does
not change that.

exp4 showed that the readout of digit d attends to the digit-d token of a
history item that matches the target's first d digits, and that this read
supports that item's code (copying). exp4 measured this only with the
target's code as outcome. At the last digit, the target can be:

- the history item's own SID (repurchase, or a collision partner)
- a sibling: same first D − 1 digits, different last digit (often a variant)

The model cannot tell these apart at the readout: it has seen the same
prefix either way. So the question is what the read of the matching item's
last-digit token does to that item's own code. Either it pushes toward that
code (copy), or away from it (exclusion of an already-bought SID).

## Seen before this protocol

- exp4 and exp6 results, structure exp3, retrospective exp7 (above).
- Data-only counts of eligible rows (no model output):

  | Cell, split | S_same | S_diff | S_diff that are near-duplicates | S_diff with h = most recent item |
  |---|---|---|---|---|
  | RQ-KMeans 3×128, test | 487 | 148 | 51 | 91 |
  | RQ-KMeans 3×128, valid | 506 | 142 | 50 | 86 |
  | RQ-VAE 4×128, test | 178 | 272 | 178 | 141 |
  | RQ-VAE 4×128, valid | 262 | 172 | 121 | 105 |

## Cohort and cells

- Cells: `next-item_best` (RQ-KMeans 3×128, last digit d = 2) and
  `oneoff_rqvae4cb128` (RQ-VAE 4×128, d = 3).
- Splits: test (primary), valid (replication). Eval template, teacher
  forcing, as in exp4.
- **Eligible rows.** At least one history item matches the target on digits
  0..D − 2. **h** is the most recent such item.
  - **S_same:** h's last digit equals the target's (h has the target's SID).
  - **S_diff:** it differs (the target is h's sibling).
- **Control item j:** the most recent history item whose first D − 1 digits
  differ from the target's. Rows without one get no control condition and
  are excluded from the contrasts that need it. Their number is reported.

## Conditions (each eligible row)

All conditions remove edges from the last digit's readout,
q = `predict_pos(0, D − 1)`:

| Condition | Edge removed | Layers |
|---|---|---|
| `clean` | none | |
| `KH_late` | q → h's last-digit token | 14–27 |
| `KH_all` | q → h's last-digit token | 0–27 |
| `KC_late` | q → j's last-digit token | 14–27 |
| `K0a_empty` | none, run through the knockout path | no-op control |
| `K0b_future` | q → q + 1, already masked | no-op control |

Layers 14–27 are where exp4 found all of the copy effect.

## Measures (at q)

- **ℓh:** log-softmax over the digit's code tokens of h's last code.
- **ℓt:** the same for the target's last code. In S_same, ℓt = ℓh.
- Whether the top-1 code is h's code; whether it is the target's.
- The legal-set versions: codes that complete the target prefix to a
  catalogue SID.

## Estimands

Paired user bootstrap, 2,000 draws, seed 20260927. Intervals exclude
training-seed variance. Cells and splits are never pooled.

- **P1 (primary, test, each cell, S_same ∪ S_diff):**
  [ℓh(`KH_late`) − ℓh(`clean`)] − [ℓh(`KC_late`) − ℓh(`clean`)], over rows
  with a control. Negative means the read pushes toward h's own code (copy);
  positive means it pushes away (exclusion).
- **P2 (primary, test, each cell, S_diff):** the same contrast for ℓt, the
  sibling target's code. Positive means the read of h competes with the
  sibling.
- **Secondary:**
  - P1 within S_same and within S_diff, and within S_diff by near-duplicate
    or not
  - `KH_all` in place of `KH_late`
  - the change in top-1 = h and top-1 = target rates
  - clean ℓh and ℓt, and top-1 rates
  - the valid split
- **Exploratory:** P1 split by whether h's item was ever re-bought in the
  training split (same item id as both a history item and the target of one
  train row).

## Expectations, recorded before outcomes

- **X1.** P1 < 0 in both cells, with the interval excluding 0 (copy, as in
  exp4).
- **X2.** If X1 holds: P2 > 0 in both cells, since removing h's read helps a
  sibling target.
- **X3.** At clean, in S_diff, top-1 = h's code is rarer for RQ-VAE than for
  RQ-KMeans.

If P1 > 0 for RQ-VAE, the read implements exclusion of the already-bought
code, which would explain exp7's low variant-copy rate.

## Numerical acceptance

- `K0a_empty` and `K0b_future` code logits are bit-identical to `clean`
  (GPU), for every row.
- **Clean agreement with exp4.** Clean ℓt at the last digit agrees with
  exp4's test-split clean `logp_codes` for the same rows within 0.05 nats
  (bf16; batch composition differs). This is reported, and a failure stops
  interpretation. exp4 runs: `281394/cell-0{0,1}`.

## What this cannot establish

- The edge knockout removes one attention edge. Information about h's last
  digit can also reach q through other positions (exp4 found most of the
  effect at the direct edge, not all).
- "Copy" and "exclusion" describe the read's net effect on h's code
  averaged over rows. Both can operate in different rows.
