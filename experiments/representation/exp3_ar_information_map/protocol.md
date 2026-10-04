# Representation exp3: what the AR residual stream holds, layer by layer

Declared 2026-10-03, before any capture, SAE fit, census or lens of this
experiment was run. GPU for captures, SAEs and lenses; CPU for the census.
Both AR cells: `next-item_best` (RQ-KMeans 3×128) and `oneoff_rqvae4cb128`
(RQ-VAE 4×128).

## Questions

1. **What kinds of information do the residual stream's features carry, at
   each depth and kind of position?** Candidate kinds:
   - where the token is (format)
   - the token itself
   - the item it belongs to, and the item before it
   - the copy source (most recent item)
   - the model's own upcoming answer
   - the true target
   - same-day bursts
   - the item's brand
2. **From which layer can the model's own answer be read out**, digit by
   digit and as a whole SID, and does the first readout position already hold
   the later digits of the answer?
3. **Which items can be read back from a single history position**, and from
   which layer: the item there, and the item before it.

exp2 trained SAEs at 4 layers and tested specific features (copy, prefix
match). exp1 ran a logit lens and linear probes for individual digits. This
experiment covers every second layer, labels every live feature, and adds
decoders for whole SIDs.

## Seen before this protocol

- The results of exp1 (logit lens, digit probes) and exp2 (SAEs at layers
  12/16/20/24), and all of the intervention suite.
- No exp3 output of any kind. The self-greedy answers below have not been
  computed.
- Code checks run before any exp3 data existed:
  - `census.py` ran on exp2's layer-12 test latents with the golden target
    standing in for the answer. Only token counts, live-latent counts and
    timings were printed; no labels or shares.
  - The SID decoder and its batched trie beam search ran on synthetic
    inputs.

## Data: captures (stage C)

- **Self-forced captures.** The target slot holds the model's own answer,
  not the golden target. The answer is the trie-constrained greedy SID:
  - at each digit, the highest-logit code among those that extend the
    answer so far to a catalogue SID
  - digit by digit, one forward per digit; the readout of digit d is
    causal, so the placeholder digits after it cannot affect it
- **Why self-forced.** At readout positions, the activations are then the
  ones the model has when producing its own answer. History and header
  positions are identical to a golden-forced capture.
- **Recorded per row.** The greedy SID, and its agreement with the golden
  target and with exp6's plain-beam top-1 (both test cells).
- **Splits and sites.** Train and test splits; all 28 decoder-layer outputs
  plus the final norm. Roles `hist_sid`, `response_header`, `target_sid`.
  bf16, eval template.
- **Checks.** The capture script's existing checks: checkpoint sha, table and
  archive checks before the target is replaced, and the batch-invariance
  report.
- **User split.** Users are disjoint between train and test, as the existing
  splits are.

## Stage S: SAEs

- **Configuration.** TopK SAEs with exp2's configuration unchanged:
  - 12,288 latents, k = 32, AuxK 512
  - lr 2e-4, batch 4,096, 20 epochs, seed 20260930
  - a 5 % user holdout of the train split
- **Layers.** 0, 2, 4, …, 26 and 27 (15 layers).
- **Data.** Fitted on the train capture; the test capture is only encoded.
- **Fidelity.** FVU by role on the train holdout and on test, and the dead
  fraction.

## Stage A: census (each layer, test tokens)

**Token groups.** Each latent is assessed within each group separately:

| Group | Tokens |
|---|---|
| `H_d` | history tokens at digit d |
| `R_0` | the last header token, which predicts answer digit 0 |
| `R_d` (d ≥ 1) | the answer's digit-(d − 1) token, which predicts answer digit d |

**Variables**, each categorical, defined for the groups where they make sense:

| Name | Kind | Value |
|---|---|---|
| `pos_recency` | format | the history item's recency bucket (1, 2, 3, 4, 5–9, ≥ 10 from the end) |
| `own_code` | token | the token's own code (H_d, R_d with d ≥ 1) |
| `item_d0` | own item | the item's digit-0 code (H_d with d ≥ 1) |
| `item_d1` | own item | the item's digit-1 code (H_d with d ≥ 2) |
| `prev_item_d0` | previous item | digit-0 code of the item before this one (H_d) |
| `r1_d` | copy source | the most recent item's digit d (R_d) |
| `answer_d` | own answer | the model's greedy answer digit d, the digit this position predicts (R_d) |
| `answer_d0` | own answer | the answer's digit 0, at later readouts (R_d with d ≥ 1) |
| `golden_d` | true target | the golden target's digit d (R_d) |
| `match_prefix` | match | the most recent item equals the answer on digits < d (R_d with d ≥ 1; binary) |
| `same_day` | burst | the most recent item was reviewed on the target's day (R_0; binary) |
| `brand` | semantic | the item's brand, top 50 brands, others pooled (H_{D−1}) |

**Score.**

- **Per latent and value.** For a latent f and a variable V in a group, let
  a = 1 when f is active (non-zero after TopK). For each value v with at
  least 20 tokens, compute the F1 of predicting V = v from a. The latent's
  score for V is the maximum over v; v* is the best value.
- **Null.** The same maximum with V's values permuted within the group,
  5 permutations; null99 is the 99th percentile over latents and
  permutations.
- **Live latents.** Only latents active on ≥ 20 tokens of the group are
  scored.
- **Label.** A live latent's label is the variable with the highest
  F1 − null99, if that difference is > 0 and F1 ≥ 0.3. Otherwise it is
  "unexplained".

**Disambiguation at readouts.** `answer_d`, `r1_d` and `golden_d` coincide
often (the copy). So for each latent labelled `answer_d`, its F1 is also
reported on the rows where the answer ≠ r1's digit (non-copy answers), with
its own null. A latent then counts as **answer-beyond-copy** if F1 − null99
> 0 on those rows too.

**Reported.** Per layer × group:

- the share of live latents with each label
- the share of the group's total activation mass carried by each label
- the top latents per label, with F1 and v*

Brand is a check for whether the model represents product attributes that
the SID does not name.

## Stage L: lenses (GPU, trained on train, evaluated on test)

**L1, tuned lens.**

- **Translator.** Per layer L, an affine translator h + A_L h + b_L
  (A, b zero-initialised). It is trained to minimise the KL from the final
  code distribution to the lens's code distribution at readout positions.
  - Code distribution: log-softmax over digit d's code tokens of
    lm_head(norm(·)) at R_d.
  - Training: Adam, lr 1e-3, 3 epochs, weight decay 0.
- **Baseline.** The logit lens (A = b = 0).

Measures per layer and digit:

- **D1.** The share of rows where the lens's trie-constrained top-1 at R_d
  equals the model's answer digit d.
- **D2 (whole SID).** The share of rows where the per-digit
  trie-constrained lens top-1s equal the model's full answer.
- **D3.** Agreement with the golden digit, and with r1's digit.
- **D4 (per row).** The earliest layer from which D2 holds at every later
  layer. Its distribution is reported overall and split by:
  - copy answers (answer = r1's SID) vs non-copy answers
  - repeat-SID targets vs new targets

**L2, SID decoders (a learned "verbaliser" for SIDs).**

- **Architecture.** One per layer and task: a linear map from the
  standardised residual (1,536 → 512), then a GRU over digits with digit
  embeddings, emitting one code per digit.
- **Decoding.** Trie-constrained to catalogue SIDs; beam 10 for top-10.
- **Training.** Teacher-forced cross-entropy on the train capture, 10 epochs,
  Adam lr 1e-3. The best epoch is chosen on a 5 % train user holdout.
- **Evaluation.** On test.

| Task | Input | Output |
|---|---|---|
| `item_at_last` | H_{D−1} token of item k | item k's SID |
| `item_at_first` | H_0 token of item k | item k's SID (only digit 0 has been seen) |
| `prev_item_at_last` | H_{D−1} token of item k ≥ 1 | item k − 1's SID |
| `answer_at_R0` | R_0 | the model's greedy answer SID |
| `golden_at_R0` | R_0 | the golden target SID |
| `r1_at_R0` | R_0 | the most recent item's SID |

Measures: exact-SID accuracy, per-digit accuracy, and top-10 exact accuracy.

Baselines, each scored the same way:

- a constant decoder that ignores the input (the trained prior)
- for the answer and golden tasks, the copy rule (output r1's SID)
- for `item_at_first`, the most frequent SID whose digit 0 is the item's
  digit 0

## Primary estimands and expectations (recorded before outcomes)

Each cell is reported separately, on test. Intervals are paired user
bootstrap, 2,000 draws, seed 20260927, for D1/D2/L2 accuracies. They exclude
training-seed variance.

- **E1 (census).** At H_d, the largest labelled shares are format, own token
  and own item at layers ≤ 10. Previous-item labels peak at middle layers.
  At R_0, the share labelled `r1_0` or `answer_0` rises after layer 18.
- **E2 (census).** Answer-beyond-copy latents at R_0 exist (≥ 1 per layer) at
  layers ≥ 20 and are rare (≤ 1 % of live latents) before layer 16.
- **E3 (tuned lens).** D2 is below 0.2 until layer 16 and reaches ≥ 0.8 by
  layer 26. The tuned lens is ≥ 0.1 above the logit lens at layers 8–20.
- **E4 (decoders).** `answer_at_R0` exact accuracy at layer 26 exceeds the
  copy rule. If it also exceeds the copy rule on non-copy answers, the
  first readout already holds the later digits of the answer
  (look-ahead).
- **E5 (decoders).** `golden_at_R0` never exceeds the copy rule by more than
  2 pp (as exp1 and exp2 found for digit 0 and content).
- **E6 (decoders).** `item_at_last` exact accuracy ≥ 0.9 by layer 4.
  `prev_item_at_last` is above its prior at middle layers.

## Amendment A1 (2026-10-04, after the pilot, before any full-run output)

The pilot (job 294030; 64 test and 600 train rows) failed one check. The
recomputed fp32 final logits gave the stored greedy digit as their
constrained argmax on only 97.4 % and 98.4 % of readouts, below the 0.99
threshold.

The cause is bf16 ties. Greedy decoding ran on bf16 logits, which are spaced
0.125 apart near |logit| ≈ 16, so many codes tie exactly; the fp32 recompute
breaks those ties differently. exp1 met the same issue.

- **The check now.** The greedy code's recomputed logit must be within
  2^-7 · |max| of the best legal logit, on ≥ 99 % of readouts.
- **Strict fp32 agreement** is reported as the ceiling for D1 and D2, the
  value an exact lens at the final layer would reach.

No estimand changed. The pilot's lens stage stopped at this check, so its
decoders never ran; the pilot is re-run before the full run.

## What this cannot establish

- Everything here is observational: decodable is not used. The causal
  counterparts are exp3–exp6, exp9, exp10 and exp2's ablations.
- **Self-forced readouts** describe greedy decoding, not the archived beam.
  Rows where greedy and beam differ are reported.
- **Brand** is the only content label on this cluster; category labels are
  not here.
- **Single SAE seed.** Labels depend on the dictionary, and F1 against chosen
  variables can miss features that encode combinations.
