# Prefix-matched copying in the AR recommenders: completed 2026-09-27

exp3 raised a post-hoc hypothesis on the test split. On the **valid split**,
which was not used to form it, the hypothesis is confirmed by a replication and
by a direct causal test of the proposed route:

- **The replication holds.** Changing digit d onward of the most recent history
  item costs the golden digit d (d ≥ 2) **1.35 nats** (RQ-KMeans 3×128) and
  **1.06 nats** (RQ-VAE 4×128) when that item already shares the target's first
  d digits. Otherwise it costs 0.01–0.03.
- **One attention edge carries much of it.** In matching rows, blocking the
  single edge from the position that predicts digit d to the matching item's
  digit-d token lowers the golden digit by **0.40** and **0.36 nats**. The same
  block does nothing in non-matching rows, and nothing when aimed at a
  non-matching item's digit-d token. That reading happens entirely in **layers
  14–27**: the same knockout in layers 0–13 has an effect of 0.000.
- **It is not a small circuit.** The route is spread across many heads. The top
  5 heads, chosen on screening users, beat layer-matched random heads on
  held-out users, but carry only **8–9%** of the all-heads effect.

The protocol ([protocol.md](protocol.md)) was declared before any valid-split
forward pass or any knockout. Cohort: 3,680 next-item valid rows (1,608 users);
eval wording; teacher forcing; bf16. Δ is the golden code's log-prob, clean
minus intervened, over the digit's codes. Intervals are paired user-bootstrap
95% percentile intervals (2,000 draws, seed 20260927) and exclude
training-seed variance.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp4_ar_copy_circuit/` |
|---|---|---|
| Pilot, 64 rows (validation and timing only) | 281258 | `pilot-281258/` |
| **Primary: valid split, Parts R + K + H** | 281261 | `281261/cell-0{0,1}`, `summary-281261/` |
| Declared secondary: test split, Parts R + K | 281394 | `281394/`: see "Secondary" below |

## Numerical acceptance

- **No-op knockouts.** Both are bit-exact at 3,680 / 3,680 per cell (max |Δ| =
  0): K0a (empty) and K0b (a knockout of a future, already-masked edge). Every
  forward passes one explicit (B, 12, T, T) mask to every layer, and each
  layer's hook confirms it received that exact tensor and fired once.
- **Unit tests.** On a toy Qwen2, knocking out an edge at every layer through
  the per-layer hook equals passing a globally edited mask (bit-exact). The
  attention observer matches eager attention weights to within 1e-5
  (`tests/interventions/test_attention_knockout.py`).
- **Inputs.** The frozen `.sem_ids` match (20,783 / 20,783 SIDs per cell), and
  so does the checkpoint sha256. Every Part R input passes `check_replacement`.

## Declared hypotheses

| | RQ-KMeans 3×128 | RQ-VAE 4×128 | Verdict |
|---|---|---|---|
| **H1** replication: Δ(match) − Δ(non-match), pooled d ≥ 1 | **+1.344** [1.188, 1.507] | **+1.036** [0.915, 1.152] | holds |
| … Δ \| match / Δ \| non-match | 1.353 / 0.009 | 1.062 / 0.026 | |
| **H2a** K1 (block readout(d) → tok(k\*, d)) \| match | **+0.398** [0.323, 0.475] | **+0.362** [0.307, 0.419] | holds |
| **H2b** K1 − K3 (control item's digit-d token), paired, match | **+0.430** [0.341, 0.525] | **+0.371** [0.310, 0.438] | holds |
| **H2c** K1 \| match − K1 \| non-match | **+0.392** [0.317, 0.469] | **+0.363** [0.308, 0.421] | holds |
| **H3** S (top-5 heads) − layer-matched random sets, held-out users | **+0.025** [0.007, 0.043] | **+0.024** [0.014, 0.034] | holds |
| … share of the all-heads K1 effect carried by S | 0.089 [0.037, 0.131] | 0.082 [0.054, 0.111] | |

**Expectations recorded before outcomes**

- **E1 (H1 replicates):** held.
- **E2 (K1 in matching rows beats K3, and beats K1 in non-matching rows):**
  held.
- **E3 (most of K1 from layers 14–27):** held completely. Layers 0–13 give
  +0.000 [−0.001, 0.002] and +0.001; layers 14–27 give +0.395 and +0.355.
- **E4 (a few heads carry a clear majority):** **failed**. S carries 8–9%.

## Knockouts by digit (Δ, nats; match rows unless stated)

| Cell | Digit | K1 next | K1 non-match | K2 whole item | K3 control token | K4 layers 0–13 | K5 layers 14–27 |
|---|---|---|---|---|---|---|---|
| RQ-KMeans | 1 (all rows match) | 0.144 | — | 0.368 | 0.011 | −0.001 | 0.139 |
| | 2 (n = 660) | 0.453 | 0.008 | 0.561 | 0.000 | −0.000 | 0.448 |
| | 3 (n = 471) | 0.322 | 0.004 | 0.331 | −0.004 | 0.001 | 0.321 |
| RQ-VAE | 1 (all rows match) | 0.039 | — | 0.319 | 0.001 | 0.003 | 0.035 |
| | 2 (n = 989) | 0.445 | −0.002 | 0.649 | −0.014 | 0.003 | 0.434 |
| | 3 (n = 490) | 0.293 | −0.000 | 0.358 | 0.005 | −0.001 | 0.292 |
| | 4 (n = 338) | 0.218 | −0.001 | 0.289 | 0.002 | 0.001 | 0.214 |

Per-digit intervals are in `summary-281261/estimates.csv`.

- **Digit 1.** The model also reads the most recent item's *first* digit
  directly (K1, 0.14 nats for RQ-KMeans), and reads its whole item more (K2,
  0.32–0.37).
- **Whole item vs next token.** For later digits, blocking the whole matching
  item (K2) adds only 0.01–0.20 nats over blocking its next-digit token alone.
  That one token is the main thing read.
- **K6.** Blocking every header and target position from the whole most recent
  item costs digit 1 **0.37** (RQ-KMeans) and **0.32** (RQ-VAE) on all rows.
  exp3 on the test split found that replacing that item costs 0.57 and 0.51.
  The difference is what flows through other routes, plus the control item's
  own evidence in exp3.

## Heads (screen on 521 / 867 screening-user matching pairs; test on held-out users)

**Selected S (layer, head):**

- RQ-KMeans: (24, 10), (19, 3), (27, 9), (26, 3), (19, 1)
- RQ-VAE: (24, 10), (23, 3), (14, 0), (19, 4), (27, 3)

Head (24, 10) ranks first in both models.

**Held-out effects** on matching pairs (d ≥ 2):

| | S | Random sets | All heads |
|---|---|---|---|
| RQ-KMeans | 0.033 | 0.008 | 0.370 |
| RQ-VAE | 0.028 | 0.004 | 0.336 |

Knocking out S at digit 1 on held-out rows costs 0.003–0.005, against 0.03–0.14
for all heads.

**Descriptive, not declared.** No single head matters much: every single-head,
single-edge knockout costs ≤ 0.007 nats. Only 1 (RQ-KMeans) and 3 (RQ-VAE) of
336 heads exceed 0.005. The two models spread the read differently:

- **RQ-VAE:** the 336 single-head effects sum to about the all-heads effect
  (0.41 vs 0.39). That is an additive read spread thinly over many heads in
  layers 14–27.
- **RQ-KMeans:** they do not (−0.20 vs 0.43), which means redundancy: heads
  that compensate when one is removed.

**Attention (observational, `fig2_heads`).** Heads attend more to the matching
item's next-digit token in matching rows than in non-matching rows. The
difference appears **only from layer 14 on**, in heads of layers 14, 19 and
22–23. That matches the knockout's layer boundary.

## Secondary: test split (the split that generated the hypothesis)

Job 281394, Parts R + K. All 3,681 / 3,681 no-op controls per cell are
bit-exact. Summary: `summary-281394/`. This is supporting evidence, not
confirmation, because this split generated the hypothesis.

| | RQ-KMeans test (valid) | RQ-VAE test (valid) |
|---|---|---|
| H1 match − non-match | +0.892 [0.755, 1.036] (1.344) | +0.630 [0.546, 0.719] (1.036) |
| H2a K1 \| match | +0.201 [0.148, 0.255] (0.398) | +0.223 [0.173, 0.276] (0.362) |
| H2b K1 − K3, paired | +0.235 [0.169, 0.301] (0.430) | +0.218 [0.168, 0.271] (0.371) |
| K4 layers 0–13 / K5 layers 14–27 | −0.000 / +0.200 | −0.000 / +0.219 |

- **Direction.** Every effect points the same way on the test split, and the
  layer boundary is the same.
- **Magnitude.** The effects are about half as large as on the valid split.
  Why the two splits differ was not examined here. How the upstream splits
  order events in time has not been checked in this repository either.

## What this establishes, and what it does not

Established, for these two frozen checkpoints under teacher forcing:

- To pick digit d, the model directly attends from the position that predicts
  it to digit d of a history item whose first d digits match the prefix it has
  already produced.
- That read is causally used, it happens in the second half of the network,
  and it is distributed over many heads.
- The effect sizes are large: 0.3–0.45 nats per digit.

The unexpected exp3 pattern is this mechanism. When the matching item's later
digits change, the model copies the wrong code.

Not established:

- whether the same happens in free-running beam decoding;
- whether it helps or hurts HR (copying is right when the user repeats or
  stays in a sub-category, and wrong otherwise);
- which MLP or head features compute the prefix match.

The effect is measured with direct-edge knockouts. Indirect routes are bounded
by K2 and K6, not eliminated.
