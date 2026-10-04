# What the DiffGRM networks use from history: replacement, prefix-matched copying, cross-attention knockout

Declared 2026-09-28, before any DiffGRM forward pass on this cluster and
before any diffusion intervention anywhere. No training, no checkpoint
selection. This is the diffusion half of paper RQ4 ("what the networks use"),
designed to mirror the AR experiments exp3 and exp4.

## Questions

- **Q1.** How much does each past item move the DiffGRM digit scores, and how
  does that depend on recency and on which digits of the item change?
- **Q2.** Does the diffusion decoder show the AR models' prefix-matched copying,
  where digit d follows history items whose first d digits match the
  target's?
- **Q3.** Does it read that evidence through cross-attention from the digit-d
  position to the matching item? The AR models read it through attention
  links in layers 14–27.

## Architecture facts that shape the design

These were verified in `vendor/diffgrm/.../DIFF_GRM/model.py`.

- **History tokens.** Each history item is **one** encoder token: its digit
  embeddings are concatenated, passed through `item_mlp`, and a recency
  position is added. There is one encoder block, and padded slots are zeroed.
- **Decoder.** There is one position per SID digit: the digit's mask embedding
  if it is masked, the code's embedding if it is revealed. Self-attention
  across digits is full, and every one of the 4 blocks cross-attends to all 50
  history slots. Padded slots are not masked out (their keys are zero).
- **The copy link.** The analogue of the AR link "readout(d) → digit d of item
  j" is therefore **digit-d decoder position → item j's slot**.

## Fixed design

**Cells.** The two diffusion checkpoints that share a SID table with an AR
checkpoint (`sidlens.models.ar.MATCHED_CELLS`):

- cell 00: `diff-next1-rqkmeans-3cb-128`
- cell 01: `diff-next1-rqvae-4cb-128`

**Cohort.** The diffusion next-item test cohort (`load_eval_cohort`): 6,297
users, leave-last-out, histories of up to 50 items, the most recent at the
last filled slot. This is **not** the AR cohort, so AR and diffusion numbers
are compared only side by side, never joined.

**States.** The validated logits path is used: `interventions.diffusion`
(projected cross-attention cache, as in `matched_decode`). Two states are
scored:

- `S_full`: every digit masked, as at the first decoding step. All digits are
  scored.
- `S_pref(d)`: golden digits 0…d−1 revealed, the rest masked. Digit d is
  scored. This is the state comparable to AR teacher forcing, and
  `S_pref(0)` is the same as `S_full`.

s_d is the golden code's log-softmax over the K codes of digit d. Δ = s_d
(clean) − s_d (intervened).

### Part A: replace one history item

This mirrors exp3. For recency r = 1…10 (r ≤ history length) and
shared-prefix level m = 0…n−1, history item k (r = len − k) is replaced by a
catalogue control that equals it on digits 0…m−1 and differs at digit m.

- **Exclusions.** Candidates exclude every item id and SID in the user's
  history and the target.
- **Seed.** `sha256("20260928|<user>|<k>|<m>")`.
- **Missing controls** are counted.

Each state is re-encoded after the replacement.

- **A1:** Δ at `S_full`, digit 0, for r = 1 and m = 0.
- **A2:** the recency profile at m = 0.
- **A3:** the prefix-sharing profile at r = 1.

### Part C: prefix-matched copying (H1)

This mirrors exp4 H1. For r = 1 and m = d (d ≥ 1), Δ is taken at
`S_pref(d)`, digit d. Rows are split by `match_d`: whether the most recent
item's digits 0…d−1 equal the target's.

**H1.** Pooled over d ≥ 1: mean Δ | match − mean Δ | non-match > 0.

### Part K: cross-attention knockout (H2)

This mirrors exp4 K. At `S_pref(d)`, for each d, the knockout sets the
cross-attention score of a (digit-query, history-slot) pair to −∞ before the
softmax. k\* is the most recent item.

| Id | Queries | Slots | Blocks |
|---|---|---|---|
| K1 | digit d | k\* | all 4 |
| K2 | all digits | k\* | all 4 |
| K3 | digit d | j = most recent other item with `match_d(j)` false (for d = 0, the second most recent item); missing if none | all 4 |
| K4_L | digit d | k\* | block L only, L = 0…3 |
| K5 (d ≥ 1) | digit d | every item whose first d digits match the target | all 4 |

- **H2a:** K1 \| match, pooled over d ≥ 1, is > 0.
- **H2b:** the paired K1 − K3 over matching rows that have a K3 slot is > 0.
- **H2c:** K1 \| match − K1 \| non-match is > 0.
- **Descriptive:** d = 0 (all rows match), K2, K4 by layer, and K5.

### Numerical controls and acceptance

- **V1 (decoder path).** exp1's pilot checks run on 8 users:
  - encoder and decoder batch-vs-single agreement
  - projected cross cache equals fresh
  - matched fixed-order decode, batch vs chunk

  They must pass.
- **V2 (knockout neutrality).** An all-ones cross-attention mask is
  bit-identical to the unmasked decoder. Every forward in every condition
  passes a mask, so all conditions share one path.
- **V3 (fixed shapes).** Clean rows scored in different batches must be
  bit-identical. If they are not, the maximum deviation is reported as the
  numerical floor.
- **V4 (identity).** The checkpoint sha256 must equal the registry, and the
  cohort hashes must equal the loader's.
- **Guards.** A knockout of a padded slot is refused, and every block must fire
  once per forward.

Precision: fp32, TF32 off, eval mode (as exp1).

## Statistics

- Paired user bootstrap: 6,297 users, 2,000 draws, seed 20260927, percentile
  95% intervals.
- Intervals exclude training-seed variance.
- Per-digit, per-recency and per-layer numbers are descriptive.

## Expectations recorded before outcomes

- **E1:** A1 > 0. The most recent item moves DiffGRM's first-digit score.
- **E2:** H1 holds. Uncertain: the history is item-level, so copying, if it
  exists, has to route through whole-item slots.
- **E3:** H2a and H2b hold if E2 holds.
- **E4:** Recency effects decay more slowly than in the AR models. This
  expectation comes from DiffGRM's 50-item encoder attention; it is
  descriptive.

## Interpretation limits

- The states are fixed reveal states, not the confidence-guided path. Scores
  are conditional log-probs, not HR.
- The AR comparison is across different cohorts, different history lengths
  (≤ 10 vs ≤ 50) and different model sizes. It is descriptive only.
- A slot knockout removes a direct read of that item's encoder state. Other
  slots still carry information about the item through the encoder's
  self-attention.
- One seed per checkpoint.
