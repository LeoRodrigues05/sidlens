# How DiffGRM routes the copied item: encoder spread vs direct decoder read

Declared 2026-09-28, after exp7 and before any exp8 forward pass.

## Origin

exp7 found three things:

- DiffGRM shows prefix-matched copying (H1: 0.35 / 0.96 nats).
- Blocking the decoder's cross-attention to the matching item's slot changes
  nothing (H2a ≈ −0.005). That holds from digit d, from all digits, and in any
  one block.
- Blocking cross-attention to *all* matching items recovers only part of it.

The candidate explanation is that the encoder's single self-attention layer
spreads each item's information into the other history slots. The decoder
can then read the item through other slots when its own slot is blocked.

This protocol tests that explanation: block the encoder route, block the
decoder route, and block both.

## Fixed design

**Cells, cohort, states and scoring** are identical to exp7:

- cells: `diff-next1-rqkmeans-3cb-128` and `diff-next1-rqvae-4cb-128`
- cohort: 6,297 users
- states: `S_full` and `S_pref(d)`
- scoring: `interventions.diffusion`, fp32

k\* is the most recent item. j is the most recent other item whose **first**
digit differs from the target's. So j is non-matching at every d ≥ 1, and it
is missing if no such item exists.

**Interventions:**

- `eX` (encoder route): in the encoder self-attention, every other real slot
  stops reading slot X. Slot X still reads itself and the others. Built with
  `encoder_block_mask`.
- `xX` (decoder route): cross-attention from every digit position to slot X
  is blocked in all 4 decoder blocks.

**Conditions**, each scored at `S_full` (digit 1) and `S_pref(d)` (digit d,
d ≥ 1):

| Id | Encoder | Decoder |
|---|---|---|
| `base` | none | none |
| `enc` | `eK` | none |
| `dec` | none | `xK` |
| `both` | `eK` | `xK` |
| `both_ctrl` | `eJ` | `xJ` |

After `both`, the decoder has no route to k\*'s encoder state at all. That
state is still computed, but no other slot and no digit position reads it.

## Estimands

Δ = s(`base`) − s(condition). Matching (`match_d`) is defined as in exp7.

- **H1:** Δ(`both`) \| match, pooled over d ≥ 1, is > 0. The paired
  Δ(`both`) − Δ(`both_ctrl`) over matching rows that have a j is also > 0.
- **H2 (redundancy):** Δ(`both`) − [Δ(`enc`) + Δ(`dec`)] \| match, pooled over
  d ≥ 1, is > 0. This is super-additivity: each route covers for the other.
- **Descriptive:**
  - the same quantities at `S_full`, digit 1, over all rows
  - per d
  - non-matching rows
  - the ratio Δ(`both`) / exp7's H1 replacement effect (a different
    intervention, so only as scale)

## Validation

- **V1:** exp1's pilot checks (as exp7).
- **V2:** the no-op encoder mask (padding only, expanded) and the all-ones
  cross mask are bit-identical to the vendor path.
- **V3:** clean rows are bit-identical across batches.
- **Guards:** knockouts of padded slots are refused, a mask that re-opens
  padding is refused, and every block fires once.

## Statistics

- Paired user bootstrap: 2,000 draws, seed 20260927, percentile 95%
  intervals.
- Intervals exclude training-seed variance.

## Expectations recorded before outcomes

- **E1:** H1 holds, and in matching rows Δ(`both`) is at least half of
  exp7's replacement effect at the same d.
- **E2:** H2 holds. `enc` and `dec` each stay small, while `both` is large.

## Limits

- Blocking every read of k\*'s encoder state still leaves k\* in the encoder's
  keys for its own slot. It also leaves the history's aggregate statistics
  (for example softmax normalization) changed rather than neutral.
- The states are fixed reveal states, not the decoding path.
- One seed per checkpoint.
