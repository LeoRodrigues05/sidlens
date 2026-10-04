# representation/exp2: sparse autoencoders on the AR residual stream

Declared 2026-09-30, after the train-split captures were written (jobs 287160
and 287161) and before any SAE was trained.

## Why

The earlier results pin down behaviour and routes, but not features:

- **Linear probes (representation/exp1).** They found the most recent item's
  first digit at the first-digit readout from layer 0. They found no
  target-digit information beyond the rule "copy the recent item".
- **Causal routes.** exp3 and exp4 located the copy read in layers 14–27, spread
  over many heads, with its evidence reaching the answer positions in layers
  19–24.

Three questions remain open:

- **Which features carry the copy?** Something at the readout has to represent
  "the recent item's first digit is c".
- **Where is the prefix match computed?** This is open question 2 of the
  interventions report. At the digit-d readout the model has to tell whether a
  history item matches the prefix generated so far.
- **Is anything else about the target there?** Probing 128 classes on 3,681
  rows is underpowered. The target item's content embedding is a continuous
  label with far more power. Also, does the readout represent a same-day burst,
  which the model cannot see but which predicts whether copying is right
  (retrospective exp6)?

SAEs give an unsupervised basis in which to answer these, and a unit (a latent)
that can be ablated causally.

## Inputs

**Captures** (bf16, eval wording, teacher-forced; roles `hist_sid`,
`response_header`, `target_sid`; sites `model.layers.{12,16,20,24}`):

- **train** (SAE fitting): `derived/ar_capture/287160` (`next-item_best`,
  RQ-KMeans 3×128) and `287161` (`oneoff_rqvae4cb128`, RQ-VAE 4×128). Both
  cover 29,444 rows. Their shard sha256s are verified on load.
- **test** (analysis only): `derived/ar_capture/281053` and `281054`, the same
  sites, 3,681 rows / 1,606 users. They are the stores used by
  representation/exp1.

**Layers.** 12 (before the copy onset in RQ-KMeans, just after it in RQ-VAE),
16, 20 and 24 (the causal window and the transfer layers).

## SAE training (fixed before any fit; no tuning on test)

- **Architecture.** TopK (`sidlens.sae.topk`), n_latents = 12,288 (8 × 1536),
  k = 32, AuxK with k_aux = 512 and coefficient 1/32. A latent is dead after
  200k tokens without firing. Decoder rows are unit-norm with the parallel
  gradient removed. Inputs are scaled by one scalar so that E‖x/s‖² = d, and
  b_dec is the coordinate median.
- **Optimisation.** Adam with lr 2e-4, batch 4,096 tokens, 20 epochs, 5% linear
  warm-up, linear decay over the last 20%, seed 20260930, fp32 math.
- **Data.**
  - **Training:** every captured token of the train-split rows whose users have
    sha256("20260930|<user>") mod 20 ≠ 0.
  - **Train-holdout:** the other 5% of train users. Its FVU is reported
    (`fvu_val`).
  - Test tokens are never seen in fitting.
- **Scale.** One SAE per (cell, layer), 8 in all.

## Evaluation rows and user halves

The test split is used throughout. User halves are
sha256("20260930|<user>") mod 2: **half 0 screens** (it selects latents) and
**half 1 is held out** (it reports). Positions:

- **readout(0):** the last `response_header` token, which predicts digit 0
- **readout(d), d ≥ 1:** the `target_sid` token of digit d−1

r1 is the most recent history item, c1 its first digit, and ct the target's.
**match_d** means r1's first d digits equal the target's.

## F: fidelity

- **F1.** On all test tokens, reported per site and per role (readout(0), other
  header tokens, history tokens, target tokens):
  - FVU
  - the dead fraction: latents never active on test tokens
  - the train-holdout FVU
- **F2 (causal, all test rows).** At one layer, the residual at every captured
  position is replaced by the SAE reconstruction. Δ = clean − spliced golden
  log-prob at each digit. The baseline replaces the same positions with the
  train-mean activation. Recovered fraction = 1 − Δ_SAE / Δ_mean, pooled per
  digit.

## Q1: copy latents at readout(0)

**Defining a copy latent** (screening half). For a latent f at layer L, among
screening rows where f is active at readout(0):

- the preferred code is c\*(f) = argmax_c P(c1 = c | active)
- the selectivity is s(f) = P(c1 = c\*(f) | active)
- the support is n(f)

f is a **copy latent** when s(f) ≥ 0.5 and n(f) ≥ 20.

**Observational** (held-out half):

- **O1a.** The number of copy latents per layer.
- **O1b.** The share of the readout(0) SAE activation, Σ val, carried by active
  copy latents whose c\* = c1 of the row.
- **O1c.** How often the rule "c1 = c\* of the most active copy latent" is right.

**Causal (CC):** at layer L and readout(0), on held-out rows, ablate every active
copy latent with c\* = c1. Ablation is error-preserving: x − Σ s·a_f·W_dec[f].

- **Measures.**
  - the copy score: the log-prob of c1 among digit-0 codes
  - the golden log-prob at digit 0
  - the copy rate: 1[top-1 = c1]
- **Control.** In the same row, ablate as many other active latents (c\* ≠ c1,
  or not copy latents), chosen greedily as the nearest in activation value. A
  row with no on-copy latent active contributes no pair.
- **CC1.** Δ copy score (copy ablation) − Δ copy score (control), paired per row.
- **CC2.** The change in copy rate, copy ablation minus control.

## Q2: prefix-match latents at readout(d), d ≥ 1

**Selection** (screening half, pooled over d ≥ 1, per layer). For each latent
with support ≥ 20 active (row, d) pairs, compute the AUC of its activation (0
when inactive) for match_d. **S_L** = the 8 latents with the largest AUC.
**Control sets:** 5 sets of 8 latents, drawn with seed 20260930 from latents
with support ≥ 20 and |AUC − 0.5| < 0.02, each matched to S_L on mean activation
by nearest rank.

**Observational** (held-out half):

- **O2a.** The held-out AUC of each S_L latent.
- **O2b.** The logit lens of each S_L decoder row: the rank of r1's digit-d
  code among digit-d codes, through the final RMSNorm weight and the tied
  unembedding, averaged over held-out match pairs. Descriptive.

**Causal (CM):** at layer L and readout(d), on held-out (row, d ≥ 1) pairs,
ablate S_L (its active members) and, separately, each control set. Δ is the
drop in golden log-prob at digit d.

- **CM1.** Δ(S_L | match) − Δ(S_L | non-match), a paired user bootstrap
  difference.
- **CM2.** Δ(S_L | match) − mean over control sets of Δ(control | match).

The reference is exp4's single-edge knockout K1 on the test split (summary-281394).

## Q3: target information beyond the history's content (dense probes on the residual and on SAE latents)

- **Label.** Y is the target item's Qwen3 embedding
  (`frozen/data/embeddings`), projected on its top 64 principal components
  (fitted on the 3,105-item catalogue) and standardised.
- **Predictors** at readout(0), layer L:
  - **B0:** the same 64-d projections of r1's embedding and of the mean
    history embedding (128-d)
  - **R_L:** the residual (1536-d)
  - **Z_L:** the SAE latent activations of latents active on ≥ 20 test rows
- **Model.** Ridge, alpha ∈ {1, 10, 100, 1000} chosen by inner 4-fold CV on
  the training folds' users, then user-disjoint 5-fold CV with folds from
  sha256("20260930|<user>") mod 5. R² is out-of-fold, pooled over all 64
  dimensions (1 − SSE / SST).
- **T1_L** = R²(R_L ⊕ B0) − R²(B0), and the same with Z_L, reported on all rows
  and on **new** rows: target SID not in the history and ct ≠ c1.

## Q4: does the readout encode a same-day burst?

- **Label.** gap1 = 0: the most recent item was reviewed on the target's day
  (`sidlens.data.timestamps`).
- **Predictors:**
  - **B_content:** 1[c1 = c2], the shared-prefix length of r1 and r2,
    1[SID(r1) = SID(r2)], the history length, and the number of distinct first
    digits in the history
  - **R_L:** the readout(0) residual
- **Model.** L2 logistic regression, C ∈ {0.01, 0.1, 1} by inner CV, the same
  user-disjoint folds, out-of-fold AUC.
- **T2_L** = AUC(R_L ⊕ B_content) − AUC(B_content).
- **Descriptive.** The 10 latents with the largest same-day AUC at readout(0)
  (screening half), with their held-out AUC and their c\* selectivity.

## Statistics

- Paired user bootstrap (`UserBootstrap`), 2,000 draws, seed 20260930, over
  held-out users (Q1 and Q2) or all test users (Q3 and Q4, out-of-fold).
- The intervals exclude training-seed uncertainty, for both the recommenders
  and the SAEs: each SAE is fitted once.
- Q3 and Q4 are observational: decodable does not mean used.

## Expectations recorded before outcomes

- **S1:** Test FVU < 0.25 at every site, and F2 recovers ≥ 70% of the
  mean-ablation loss at digit 0.
- **S2:** Copy latents exist at every layer, and at layers 20 and 24 the
  on-copy latents carry ≥ 30% of readout(0) latent activation (RQ-KMeans).
- **S3:** CC1 > 0 with the interval excluding 0 at layers 20 and 24, in both
  cells.
- **S4:** At layers ≥ 16 the best S_L latent has held-out AUC ≥ 0.7, and
  CM1 > 0 with the interval excluding 0.
- **S5:** |T1_L| ≤ 0.02 at every layer on new rows: no semantic target
  information beyond the history content, consistent with representation/exp1.
- **S6 (weak prior):** T2_L > 0.02 at some layer ≥ 16. The readout represents
  burst-ness beyond simple SID-overlap features.

## Limits

- **One fit per SAE.** Its features are one decomposition among many; the
  dictionary size and k were fixed, not tuned.
- **Small corpus.** 0.6M training tokens is small for an SAE, so latents may be
  coarser than in large-corpus SAEs. Fidelity F1 and F2 is the check.
- **Single-layer ablation.** An ablation at one layer and one position leaves
  later layers free to re-read the history. Small effects bound what that
  position carries at that layer, not the whole computation.
- **Industrial category labels** are not on this cluster, so latents are named
  by SID codes, match status and time only.

## Disclosure (2026-09-30)

`train.py` and `analyze.py` were smoke-tested on the login node CPU. The SAE
was layer 16 of `next-item_best`, trained for 1 epoch on 8k tokens, and the
output went to a scratch directory. That throwaway SAE's log line (its S_L
AUCs) was seen. No estimate from a declared SAE was seen before its stage ran.
`train.py` gained per-role FVU on the train holdout after the smoke run showed
that within-role FVU at readout(0) can exceed 1. This adds a report; it
changes no estimand.

## Amendment A1 (2026-09-30, post hoc): within-digit selection of match latents

**Why.** The declared Q2 rule ranks latents by AUC for match_d pooled over
d ≥ 1. Match rates fall with d:

| | d = 1 | d = 2 | d = 3 |
|---|---|---|---|
| RQ-KMeans | 0.16 | 0.12 | – |
| RQ-VAE | 0.29 | 0.14 | 0.08 |

A latent that only marks which readout it sits at therefore gets a pooled AUC
above 0.5. RQ-VAE's declared selection log showed eight latents with nearly
identical held-out AUC of about 0.65, the signature of such position latents.

**What was already seen.** Before this amendment was written:

- the declared CM result for RQ-KMeans, which is null
- a post-hoc per-d diagnostic of the RQ-KMeans S latents: within-d AUC
  0.91–0.97, i.e. genuine match detectors

**The amendment.** A1 keeps every declared result as declared. It adds one
secondary analysis: S_L selected by the mean over d of the within-d AUC
(screening half), controls drawn as declared from latents with
|mean within-d AUC − 0.5| < 0.02, then CM run on held-out pairs with that
selection (`amend_select_within_d.py`, then `causal.py --parts CM`). A1 results
are reported as post hoc, beside the declared ones, never in their place.
