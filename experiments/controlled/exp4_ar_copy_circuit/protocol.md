# Prefix-matched copying in the AR recommenders: replication, attention-edge knockout, heads

Declared 2026-09-27, before any forward pass on the valid split and before any
attention knockout had been run. No training, no checkpoint selection.

## Origin

In exp3 (test split) we split the rows after the fact by whether the most
recent history item k\* shares the target's first d digits. Changing digits
≥ d of k\* moved the golden score at output digit d almost only in matching
rows. For RQ-KMeans digit 2 the effect was 0.99 nats in matching rows and 0.02
in non-matching rows. The hypothesis is prefix-matched copying: to choose
digit d, the model reads digit d of history items whose first d digits match
the prefix generated so far.

That observation was made on the test split, so the test split cannot confirm
it. **The valid split is primary** here. Test-split runs are reported only as
secondary evidence, and they are not confirmation.

## Fixed design

**Cells.**

- Cell 00: `next-item_best` (RQ-KMeans 3×128).
- Cell 01: `oneoff_rqvae4cb128` (RQ-VAE 4×128).
- Both use the eval wording, teacher forcing, bf16, and the fixed-shape runner
  from exp3.

**Primary cohort.** All 3,680 next-item valid rows (1,608 users). Rows must
equal the frozen `.sem_ids` (`check_against_table`). There is no archive
check, because archived predictions exist for test only. There are no row
exclusions.

**Definitions.**

- k\* is the most recent history item.
- `match_d` is true when k\*'s digits 0…d−1 equal the target's. `match_0` is
  true for every row.
- `readout(d) = predict_pos(0, d)`.
- `tok(j, d)` is the position of history item j's digit-d token.
- s_d is the golden code's log-softmax over digit d's code tokens.
- An effect is Δ_d = s_d(clean) − s_d(intervened).

### Part R: replication (H1)

For each row and each d ≥ 1, replace k\* with a control that shares exactly d
leading digits, exactly as exp3 did at m = d. Candidates are drawn with the
same exclusions as exp3, seeded by `sha256("20260927|<example_id>|<k>|<m>")`.
Missing controls are counted.

**H1.** Pooled over d ≥ 1, with every (row, d) pair weighted equally:
mean Δ_d over matching rows minus mean Δ_d over non-matching rows is > 0.

### Part K: attention-edge knockout (H2)

A knockout removes an edge (query q → key k). In the listed layers and heads,
k is masked out of q's softmax and the remaining weights renormalize. It is
implemented by editing the boolean mask each layer receives
(`sidlens.interventions.attention`). All layers and all heads unless stated.
Conditions per row and digit d:

| Id | Queries | Keys | Layers |
|---|---|---|---|
| K1 `next` | readout(d) | tok(k\*, d) | all |
| K2 `item` | readout(d) | all n tokens of k\* | all |
| K3 `ctrl` | readout(d) | tok(j, d), where j is the most recent history item other than k\* with `match_d(j)` false. For d = 0, j is the second most recent item. Missing if none exists. | all |
| K4 `next_early` | readout(d) | tok(k\*, d) | 0–13 |
| K5 `next_late` | readout(d) | tok(k\*, d) | 14–27 |
| K6 `readout_item` | every position from the first response-header token on | all tokens of k\* | all |

K6 is one condition per row, scored at every digit.

- **H2a.** Mean Δ_d(K1) over matching rows with d ≥ 1 is > 0.
- **H2b.** Over matching rows with d ≥ 1 that have a K3 key, the paired mean
  Δ_d(K1) − Δ_d(K3) is > 0.
- **H2c.** Mean Δ_d(K1 | match) − mean Δ_d(K1 | non-match), over d ≥ 1, is
  > 0.
- **Descriptive:** the same quantities per d, digit 0 (all rows), K4 vs K5,
  K2, and K6.

### Part H: heads (H3)

Users are split by the parity of `sha256("20260927|<user_id>")`: even users
are the screening half, odd users the held-out half.

**Screen.** On screening-half matching (row, d ≥ 1) pairs, for every layer L
and head h (28 × 12), knock out K1 for that head in that layer only.

**Selection.** Heads are ranked by mean Δ_d over those pairs. S is the top 5.
Ties are broken by (layer, head) order. The selection rule is fixed here and
executed inside `run.py`.

**Held-out test.** On held-out matching (row, d ≥ 1) pairs:

- S-KO: K1 for every head in S, each in its own layer.
- 20 control sets. Each replaces every head of S with a head drawn uniformly
  from the other 11 heads of the same layer (seed 20260927).
- K1 over all heads (from Part K) as the reference.

**H3.** Mean Δ(S) minus the mean over control sets of their mean Δ, on
held-out matching pairs, is > 0. The share Δ(S) / Δ(K1, all heads) is
reported.

**Descriptive:**

- S-KO at d = 0 on held-out rows.
- Attention weight from readout(d) to tok(k\*, d) for every layer and head, in
  matching vs non-matching rows. It is recomputed from each layer's inputs
  (`sidlens.hooks.attention_probs`). This is observational: attention weight
  alone is not causal evidence.

### Numerical controls and acceptance

- **Mask path.** Every forward passes an explicit (B, 12, T, T) boolean mask
  (causal ∧ key padding) to every layer, whether or not it knocks anything
  out. All forwards therefore share one kernel path.
- **Required bit-exact (digit code logits):**
  - K0a, an empty knockout, equals clean.
  - K0b, a knockout of an edge to a future position (already masked), equals
    clean.
  - Batch invariance of clean rows.
- **Guards.** A knockout whose edge is already masked raises, except in K0b.
  So does a self-edge knockout, or any layer whose hook does not fire exactly
  once.
- **Other checks:** exp3 checkpoint sha and SID-table checks, and
  `check_replacement` on every Part R input.

## Statistics

- Paired user bootstrap, 2,000 draws, seed 20260927, percentile 95%
  intervals. For H3, only held-out users are resampled. The selected set is
  fixed.
- Pooled estimates weight (row, d) pairs equally.
- Intervals exclude training-seed variance.
- Per-digit, per-layer and per-head numbers are descriptive.

## Expectations recorded before outcomes

- **E1:** H1 replicates on the valid split.
- **E2:** In matching rows, K1 lowers s_d clearly more than K3, and more than
  K1 does in non-matching rows.
- **E3:** Most of the K1 effect comes from layers 14–27. This follows exp3's
  late transfer.
- **E4:** A few heads (S) carry a clear majority of the K1 effect.

## Interpretation limits

- Knockout removes a direct read. Information can still reach readout(d)
  indirectly through other positions. K2 and K6 bound this.
- Heads are selected on screening users, and their effect is estimated on
  different users of the same split.
- Teacher-forced conditional scores are not HR@10.
- One seed per checkpoint.
- The cells differ in both quantizer and depth.
