# What the DiffGRM networks use from history: completed 2026-09-28

This is the diffusion half of paper RQ4, run on the two cells matched to the AR
checkpoints. **DiffGRM behaves like the AR recommenders but does not compute the
same way:**

- **Recency.** It leans on the most recent history item. Replacing that item
  with a different-first-digit item lowers the golden first digit at the fully
  masked state by **0.56 nats** (RQ-KMeans 3×128) and **0.48 nats** (RQ-VAE
  4×128). That is close to the AR models' 0.57 and 0.51 (exp3, a different
  cohort). The effect decays as steeply with recency.
- **Copying.** It shows the same **prefix-matched copying**. With the golden
  prefix revealed, changing digit d of the most recent item costs digit d
  **0.35** and **0.96 nats** in rows where that item already shares the
  target's first d digits. In other rows the cost is 0.01 and 0.07 (H1).
- **Not the AR route.** Unlike AR, the copy does not run through a direct link
  from the digit position to the matching item. Blocking cross-attention from
  digit d to the most recent item's slot in all four decoder blocks changes
  nothing (−0.007 and −0.004 nats). Neither does blocking it from every digit
  position, or in any single block (H2a/H2b fail). Blocking digit d's
  cross-attention to *every* matching item removes only part of the effect
  (up to 0.24 of about 1.1 nats). The copied item's information is carried
  redundantly across history slots. The likely route is the encoder's
  self-attention over the 50 items, a descriptive reading that this experiment
  did not test.

The protocol ([protocol.md](protocol.md)) was declared before any DiffGRM
forward pass on this cluster.

- **Cohort.** The diffusion next-item cohort: 6,297 users, leave-last-out,
  histories of up to 50 items. It is **not** the AR cohort; AR numbers are
  quoted side by side only.
- **Precision.** fp32 on CPU (cscc-cpu-p).
- **Intervals.** Paired user bootstrap, 2,000 draws, seed 20260927. They
  exclude training-seed variance.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp7_diffusion_history_use/` |
|---|---|---|
| Pilot, 256 users | 283122 | `pilot-283122/` |
| **Primary** | 283124 | `283124/cell-0{0,1}`, `summary-283124-r2/` (`summary-283124.failed` had a summarizer bug) |

## Numerical acceptance

- **V1.** exp1's pilot checks passed on 8 users for both cells: encoder and
  decoder batch vs single, projected cross cache vs fresh, and a matched
  fixed-order decode by batch and chunk.
- **V2.** An all-ones cross-attention mask is bit-identical to the vendor's
  unmasked decoder. Every forward in every condition carries a mask.
- **V3.** Clean rows scored in different batches are bit-identical (31,485
  and 44,079 comparisons; max |Δ| = 0).
- **Guards.** Knockouts of padded slots are refused, and every block fired
  once. The CPU tests in `tests/interventions/test_diffusion_knockout.py` use
  the real checkpoint.
- **Missing controls.** Deep prefix-matched controls are often missing:
  11,572 (RQ-KMeans) and 38,294 (RQ-VAE) item-level conditions have no
  catalogue item at that level. All are counted.

## Declared estimands

| | RQ-KMeans 3×128 | RQ-VAE 4×128 | Verdict |
|---|---|---|---|
| **A1** Δ at `S_full`, digit 1, most recent item, m = 0 | **+0.556** | **+0.476** | E1 held |
| **H1** Δ \| match − Δ \| non-match (`S_pref(d)`, pooled d ≥ 2) | **+0.340** [0.303, 0.377] | **+0.883** [0.811, 0.957] | E2 held |
| … Δ \| match / Δ \| non-match | 0.351 / 0.011 | 0.955 / 0.073 | |
| **H2a** K1 (block digit d → k\*, all blocks) \| match | −0.007 [−0.012, −0.003] | −0.004 [−0.008, 0.000] | **E3 failed** |
| **H2b** K1 − K3 (control slot), paired, match | −0.014 [−0.022, −0.008] | −0.004 [−0.010, 0.003] | **E3 failed** |
| **H2c** K1 \| match − K1 \| non-match | +0.010 [0.004, 0.015] | +0.032 [0.027, 0.037] | positive only because non-matching rows go negative |

**E4** (slower recency decay than AR) **failed**. The second most recent item
moves digit 1 by 0.145 and 0.114, about a quarter of the most recent item's
effect, as in AR.

## Knockouts (Δ log p at `S_pref(d)`, matching rows; per-digit intervals in `estimates.csv`)

| | RQ-KMeans digit 2 / 3 | RQ-VAE digit 2 / 3 / 4 |
|---|---|---|
| K1: digit d → k\*, all blocks | −0.008 / −0.006 | −0.003 / −0.005 / −0.008 |
| K2: all digits → k\*, all blocks | −0.009 / −0.005 | −0.003 / −0.004 / −0.007 |
| K4: digit d → k\*, one block (0–3) | between −0.005 and 0 | between −0.004 and +0.001 |
| **K5: digit d → every matching item, all blocks** | **+0.056** / +0.012 | **+0.243** / +0.028 / **+0.196** |
| For scale, H1's Δ \| match (replace k\*'s digits ≥ d) | 0.330 / 0.407 | 1.137 / 0.563 / 0.470 |

In non-matching rows, the knockouts slightly *raise* the golden score, by
0.01–0.05. Removing a direct read of a non-matching item helps a little.

## Replacement profiles (`S_full`, m = 0)

| Recency | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| RQ-KMeans digit 1 | 0.556 | 0.145 | 0.085 | 0.057 | 0.028 |
| RQ-VAE digit 1 | 0.476 | 0.114 | 0.038 | 0.026 | 0.027 |

At recency 1, keeping the replaced item's first code (m = 1) leaves 0.147
(RQ-KMeans) and 0.177 (RQ-VAE) of the digit-1 effect. The first digit carries
most of it, as in AR.

## How this compares with the AR models (different cohorts; descriptive)

| | AR (Qwen, exp3/exp4) | DiffGRM (exp7) |
|---|---|---|
| First-digit reliance on the most recent item | 0.57 / 0.51 nats | 0.56 / 0.48 nats |
| Prefix-matched copying (match vs non-match) | 1.35 / 1.06 vs ≈ 0 | 0.35 / 0.96 vs ≈ 0 |
| Direct link to the matching item | carries 0.40 / 0.36 nats; layers 14–27 | carries nothing (≈ −0.005) |
| Where the copied item lives | in its own digit tokens, read late | spread across history slots |

## Limits

- **States.** These are fixed reveal states (all masked; golden prefix
  revealed). They are not the confidence-guided decoding path, and not HR.
- **Knockouts.** A slot knockout removes a direct read. The encoder's
  self-attention still spreads each item into other slots. Testing that route
  needs an encoder-level intervention; that is the next experiment.
- **Comparison.** The AR comparison crosses cohorts, history lengths and model
  sizes.
- **Seeds.** One seed per checkpoint.
