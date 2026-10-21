# controlled/exp14: perturbations inside DiffGRM decoding (copy block, order swap, position shift)

Declared 2026-10-14, before any forward pass of this experiment. Plan:
`docs/plans/DIFFUSION_MECHANISMS_PLAN.md` (X4).

## Questions

- **Part B.** Does copying the most recent item help DiffGRM's
  recommendations? This is the DiffGRM counterpart of AR exp6, where blocking
  copy reads cut HR@10 by 5.7 / 3.3 points.
- **Part O.** Does DiffGRM read recency from the list order or from absolute
  slot positions? For AR (exp9), the model followed position, not item.

## Design

- **Cells**: the 20 trained next-item checkpoints, in representation/exp4's
  order. **Cohort**: the 6,297-user diffusion test cohort.
- **Decoding**: `matched_decode.decode`, confidence policy, top 10, catalogue
  filter, batch 32, decoder chunk 1,024. Beam 256 is primary and beam 64
  secondary.

**Conditions**, each held for the whole decode:

| Id | Users | Intervention |
|---|---|---|
| `clean` | all | none |
| `block_r1` | L ≥ 2 | exp8's `both` block on r1. In the encoder, every other real slot stops reading r1's slot (`encoder_block_mask`). In the decoder, cross-attention from every digit position to r1's slot is blocked in all four blocks, at every reveal step. |
| `block_ctrl` | L ≥ 2 with a control | the same block on j, the most recent item whose first code differs from r1's |
| `swap` | L ≥ 2 | r1 and r2 exchange slots |
| `shift1`, `shift5` | L + s ≤ 50 | every item moves s = 1 or 5 slots to the right; the first s slots become padding. Order and padding count are kept; absolute positions change. |

The decoder's cross-attention mask is applied through `score_fn`. The run
checks that an all-ones mask on that path reproduces the plain decode bit for
bit (V2).

## Measures

- SID HR@10 and item-level HR@10 bounds (`collisions.item_rank_bounds`), and
  top-1 SID agreement with `clean`.
- **Target relation to r1** (Part B), assigned in this order:
  - `repeat`: the target item is r1's item;
  - `same_sid`: another item with r1's SID;
  - `same_d0`: the target shares r1's first code;
  - `new`: none of these.
- **Swap**: among users whose r1 and r2 first codes differ, the share whose
  top-1 first code equals r1's, and the share equal to r2's, under `clean`
  and under `swap`.
- **Fully masked distribution**: KL against clean at digit 1, from one
  forward per condition.

## Estimands

- **PB** (beam 256, users with L ≥ 2 and a control): ΔHR@10(`block_r1`) −
  ΔHR@10(`block_ctrl`), with Δ = condition − clean. Paired over users;
  reported at SID level and at both item bounds.
- **PB-split**: ΔHR@10(`block_r1`) per target-relation group.
- **PO1**: under `swap`, among users whose r1 and r2 first codes differ:
  [P(top-1 first code = r2's) − P(= r1's)] under swap, minus the same under
  clean. +1 is a model that follows the position, 0 one that ignores the
  order, −1 one that follows the item. Paired.
- **PO2**: for `shift1` and `shift5`, the share of users whose top-1 changes,
  and ΔHR@10.

## Numerical acceptance

- **V1**: exp1's pilot checks on 8 users.
- **V2**: the masked-score path with an all-ones mask and padding-only encoder
  masks equals the plain decode bit for bit, on the first 32 users at both
  beams.
- **V3**: the plain decode matches controlled exp1's archive at beams 64 and
  256, with the tie tolerance declared in exp13's protocol (V4 there).
  Cells 18 and 19 have no archive.
- **V4**: the checkpoint sha256 equals the registry.
- **Guards**: blocking a padded slot is refused. A shifted history is checked
  to hold the same items in the same order.

## Statistics

Paired user bootstrap, 2,000 draws, seed 20261014. Intervals exclude
training-seed uncertainty. SID-level and item-level results are kept apart.

## Expectations recorded before outcomes

- **E1**: PB < 0 at SID level, with the interval excluding 0, in both matched
  cells. Copying helps, as it does for AR.
- **E2**: the `repeat` and `same_sid` groups carry most of the loss. On `new`
  targets, ΔHR@10(`block_r1`) has an interval including 0 or is positive in
  at least 15 of 20 cells (retrospective exp7: copying does not help truly
  new targets).
- **E3**: PO1 > 0.5 in at least 15 of 20 cells. DiffGRM follows the most
  recent position, as AR RQ-KMeans did.
- **E4**: under `shift5`, the top-1 changes for fewer than 20 % of users in
  at least 15 of 20 cells. Recency is read relative to the end of the history,
  not from absolute slots. This expectation is uncertain: training always
  right-padded.

## Interpretation limits

- `block_r1` leaves r1's encoder state in place; nothing reads it. As in exp8,
  this is close to removing the item, but not identical.
- Shifted histories put padding at the start, which training never did. A
  change under shift mixes the position code with that novelty.
- The two cohorts are separate: AR numbers are quoted side by side only.
- One checkpoint per cell.
