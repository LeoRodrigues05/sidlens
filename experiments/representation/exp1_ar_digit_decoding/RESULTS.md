# Layer-wise decoding of SID digits from AR activations: completed 2026-09-28

This is an observational read of the test-split activation captures: 3,681
rows and 1,606 users per model, with 29 sites (layers 0–27 and the final
norm). Decodable does not mean used. The causal counterparts are exp3 and
exp4.

- **The first digit mostly copies the most recent item.** At the position
  that predicts the first digit, the model's own top-1 code equals the most
  recent history item's first digit in **64%** of rows (RQ-KMeans 3×128) and
  **53%** (RQ-VAE 4×128). The golden first digit is top-1 in only 18% and 34%.
  That copy preference appears in the logit lens abruptly, at **layers 20–23**
  for RQ-KMeans (0.11 → 0.79) and **layer 11** (0.10 → 0.40) plus layer 20
  (0.32 → 0.57) for RQ-VAE.
- **The information is there from the first layer, but not as output.** A
  linear probe reads the recent item's first digit at that position with 84%
  (RQ-KMeans) and 93% (RQ-VAE) accuracy after **block 0**, and 96–98% after
  block 1–3, long before the unembedding favours it. exp4 found that the
  direct edge to that item is causally inert in layers 0–13. The early copy is
  present but not what the output uses.
- **No target information beyond the copy is linearly readable.** The target's
  first digit is probed at 0.13–0.15 (RQ-KMeans) and 0.28–0.31 (RQ-VAE) at
  every layer. That never exceeds the no-training rule "predict the recent
  item's first digit" (0.164 and 0.290). Split by whether the target shares the
  recent item's first digit, the probe scores 0.51–0.58 and 0.72–0.76 when it
  does, and 0.05–0.07 and 0.11–0.13 when it does not. The latter is near the
  majority baseline (0.065 / 0.23).
- **Items are assembled early.** An item's last SID token carries that item's
  first digit at ≥ 0.91 (RQ-KMeans) by layer 1 and ≥ 0.93 (RQ-VAE) by layer 4
  (0.66 and 0.80 after block 0).

Protocol ([protocol.md](protocol.md)) declared before any decoding. It carries:

- a dated amendment made before any intermediate-layer output: the acceptance
  check compares at bf16 logit precision
- a disclosure: a smoke run with undertrained probes showed layer-0 and final
  numbers

Probes are multinomial logistic regressions, standardized, with L2 chosen on
inner training users, full-batch L-BFGS for 200 iterations, and
**user-disjoint** 5-fold cross-validation. Intervals are user-bootstrap 95%
percentile intervals (2,000 draws, seed 20260927).

| Run | Job | Output under `$SIDLENS_WORK/derived/representation/exp1_ar_digit_decoding/` |
|---|---|---|
| RQ-KMeans 3×128 (capture 281053) | 281264 | `281264/cell-00` |
| RQ-VAE 4×128 (capture 281054) | 281411 | `281411/cell-01` |
| Summary | — | `summary-281264-281411/` (`report.md`, `estimates.csv`, `figures/`) |

## Acceptance

- **Stores:** every shard sha256 re-verified. The store's checkpoint sha equals
  the frozen manifest; template eval, split test.
- **Logit lens at the final norm vs the capture's own ranks:** **99.95%** and
  **99.99%** rank agreement at bf16 logit precision, 100% top-1 agreement, max
  |Δ log p| 0.12. With fp32 logits the rank agreement is 87% and 91%; that gap
  is bf16 ties, as the amendment explains.

## Logit lens (golden top-1 at each digit's readout; `fig1_logit_lens`)

| Layer | RQ-KMeans digit 1 / 2 / 3 | Recent item's digit 1 top-1 | RQ-VAE digit 1 / 2 / 3 / 4 | Recent item's digit 1 top-1 |
|---|---|---|---|---|
| 0 | 0.05 / 0.14 / 0.04 | 0.05 | 0.07 / 0.14 / 0.05 / 0.11 | 0.07 |
| 9 | 0.04 / 0.18 / 0.12 | 0.05 | 0.04 / 0.13 / 0.14 / 0.18 | 0.08 |
| 12 | 0.05 / 0.18 / 0.13 | 0.07 | 0.27 / 0.14 / 0.16 / 0.20 | 0.41 |
| 18 | 0.06 / 0.23 / 0.21 | 0.07 | 0.22 / 0.15 / 0.19 / 0.22 | 0.34 |
| 21 | 0.13 / 0.28 / 0.29 | 0.42 | 0.27 / 0.19 / 0.23 / 0.25 | 0.52 |
| 24 | 0.17 / 0.34 / 0.42 | 0.77 | 0.29 / 0.26 / 0.43 / 0.53 | 0.60 |
| final | 0.18 / 0.42 / 0.66 | 0.64 | 0.34 / 0.27 / 0.60 / 0.79 | 0.53 |

Later digits (teacher-forced) become top-1 gradually from mid-depth and
mostly after layer 20. This matches the patching transfer in exp3 (layers
19–24).

## Probes (held-out top-1 accuracy; `fig2_probes`)

| Probe | Layer 0 | Layer 3 | Layer 18 | Final | Shuffled labels | Majority |
|---|---|---|---|---|---|---|
| **RQ-KMeans 3×128** | | | | | | |
| P1 target digit 1 at readout 1 | 0.134 | 0.14 | 0.145 | 0.141 | 0.03–0.04 | 0.065 |
| … copy baseline (no training) | 0.164 | | | | | |
| P2 recent item's digit 1 at readout 1 | 0.844 | 0.93 | 0.952 | 0.949 | 0.03–0.04 | 0.065 |
| P3 recent item's digit 1 at its last token | 0.662 | 0.96 | 0.954 | 0.988 | 0.03–0.06 | 0.065 |
| P4 target digit 2 at readout 2 | 0.427 | 0.43 | 0.427 | 0.435 | 0.18–0.22 | 0.247 |
| **RQ-VAE 4×128** | | | | | | |
| P1 target digit 1 at readout 1 | 0.290 | 0.30 | 0.294 | 0.300 | ~0.19 | 0.231 |
| … copy baseline (no training) | 0.290 | | | | | |
| P2 recent item's digit 1 at readout 1 | 0.935 | 0.98 | 0.960 | 0.897 | 0.17–0.19 | 0.221 |
| P3 recent item's digit 1 at its last token | 0.804 | 0.91 | 0.974 | 0.965 | 0.14–0.17 | 0.221 |
| P4 target digit 2 at readout 2 | 0.235 | 0.23 | 0.248 | 0.252 | 0.04–0.08 | 0.071 |

The 95% intervals are about ±0.01–0.02; see `summary-*/report.md`.

P4 is flat across layers. Under teacher forcing, readout 2 is the golden
first-digit token itself, so what it decodes is mostly the prefix → next-digit
prior, and it says little about computation.

## Expectations recorded before outcomes

- **E1** (golden top-1 near chance until about layer 18, then rises): **held
  for RQ-KMeans** (≤ 0.06 through layer 19, rising from 20). It **failed for
  RQ-VAE**, whose first rise is at layer 11 (0.05 → 0.26), with a second at
  layer 20.
- **E2** (P3 high from early layers): **held** (≥ 0.91 by layer 1 and ≥ 0.93
  by layer 4).
- **E3** (P2 decodable no later than P1): **held**. P2 reaches ≥ 0.84 at
  layer 0, while P1 never exceeds the copy baseline.

## How this fits with exp3–exp5

- **Presence vs use.** The readout position holds the recent item's first
  digit from block 0. Its direct read in layers 0–13 is causally inert (exp4
  K4 = 0.000), and it becomes the output only in the late layers where the
  causal transfer happens (exp3 layers 19–24; exp4 layers 14–27).
- **Where the first-digit prediction comes from.** Largely copying: the model
  outputs the recent item's first digit in over half of the rows. That is
  consistent with the large first-digit replacement effect in exp3 (0.57 / 0.51
  nats), and with first digits being where AR errors concentrate (retro exp2:
  the first mismatch is at digit 1 in 82% of misses).

## Limits

- **Observational.** Decodable is not used.
- **One split.** Up to 128 classes and 3,681 rows, so rare classes are hard for
  any probe.
- **Linear only.** Non-linear or distributed encodings of the target are not
  excluded.
- **Categories.** Industrial category labels are not on this cluster, so
  category probes were deferred.
