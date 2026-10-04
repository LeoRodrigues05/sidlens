# representation/exp2: SAEs on the AR residual stream, completed 2026-09-30

| Stage | RQ-KMeans 3×128 (cell 00) | RQ-VAE 4×128 (cell 01) |
|---|---|---|
| Train-split captures (layers 12, 16, 20, 24) | `ar_capture/287160` | `ar_capture/287161` |
| SAE fit + test encoding | `train-287222` | `train-287231` |
| Observational + selection | `analyze-287243` | `analyze-287271` |
| Causal (declared) | `causal-287352` | `causal-287394` |
| Amendment A1: selection / CM | `amend_select_within_d-287374` / `causal-287380` | `amend_select_within_d-287395` / `causal-287415` |

All paths are under `$SIDLENS_WORK/derived/representation/exp2_ar_sae/`, except
the captures, which are under `derived/`.

- **Declared summary:** `summary-287222-287231/`, with `report.md`,
  `causal_estimates.csv`, `observational_estimates.csv`, `fidelity.csv` and
  `figures/`.
- **A1 summary:** `summary-a1-287380-287415/`.
- **Cell grouping:** `groups-20260930/` holds symlinks joining the per-cell job
  directories.

Protocol: [protocol.md](protocol.md), declared before any SAE was trained. It
records one disclosure (a CPU smoke test on a throwaway SAE) and amendment A1,
which is post hoc and reported beside the declared results.

## Acceptance

- **Inputs.** Every store's shard sha256s were re-verified on load, and each
  store's checkpoint sha equals the frozen manifest.
- **Self-patch controls.** Patching a position with its own clean activation
  is bit-exact: 256 / 256 in each of the four causal runs.
- **Why that matters.** The ablations are error-preserving, x − Σ s·a_f·W_dec[f],
  built from activations re-captured in the same fixed-shape runner. So an
  ablation that removes nothing reproduces clean exactly, and every non-zero
  effect is the named latents' contribution.

## The answer in one paragraph

**The SAEs find what the probes found, and add causal units.**

- **Copy latents are real.** At the first-digit readout, a handful of
  code-specific latents fire on the recent item's first digit. Ablating them
  lowers the model's log-prob of copying that digit by 0.10 nats at layer 20,
  and by 0.21 (RQ-KMeans) and 0.36 (RQ-VAE) at layer 24. A matched control
  does nothing. But they carry little of the readout's activation (≤ 3%
  RQ-KMeans, ≤ 12% RQ-VAE), and they matter only from layer 20 on. This
  matches the late copy onset in exp3 and representation/exp1.
- **Match detectors exist but are not used where they sit.** At the later-digit
  readouts, RQ-KMeans has sharp "the recent item matches the prefix so far"
  latents, with within-digit held-out AUC up to 0.94–0.97. Ablating them there
  changes nothing (0.00 ± 0.04 nats), under both selection rules. With exp4
  (the copy is an attention read from the readout to the matching item's
  token, in layers 14–27), this points to the match being computed in attention
  (query–key), not as a readout feature. RQ-VAE's match latents are weaker (AUC
  about 0.66), with a small specific effect (+0.02–0.03 nats).
- **No target information beyond the history.** Neither the residual nor the
  SAE latents at the readout predict the target item's content embedding
  better than a linear map of the history's embeddings.
- **But the readout knows about bursts.** It predicts whether the most recent
  item was reviewed on the target's day, beyond simple SID-overlap features:
  AUC +0.08–0.09 at every layer. Retrospective exp6 showed RQ-KMeans' copying
  ignores this: its copy rate is the same at any age.

## F: fidelity

**F1, test tokens** (the SAEs never saw them):

| Model | Layer | FVU all | FVU hist | FVU target | FVU readout(0), within-role | Dead on test |
|---|---|---|---|---|---|---|
| RQ-KMeans | 12 | 0.009 | 0.024 | 0.020 | 0.354 | 54% |
| | 16 | 0.014 | 0.032 | 0.026 | 0.385 | 47% |
| | 20 | 0.025 | 0.039 | 0.029 | 0.316 | 30% |
| | 24 | 0.055 | 0.065 | 0.057 | 0.100 | 1% |
| RQ-VAE | 12 | 0.004 | 0.007 | 0.005 | 0.304 | 19% |
| | 16 | 0.007 | 0.011 | 0.007 | 0.325 | 10% |
| | 20 | 0.013 | 0.019 | 0.014 | 0.228 | 5% |
| | 24 | 0.042 | 0.056 | 0.050 | 0.129 | 0.2% |

- **Train-holdout FVUs match test** to within 0.01, so the SAEs are not
  overfitted.
- **Why readout(0) looks worse.** Every readout(0) token is the same ":\n",
  so its variance around its own mean is small. The within-role FVU (0.10–0.39)
  measures how much of the row-to-row variation there, which carries the copy
  code, the SAE reconstructs.

**F2, splice recovery.** At one layer, the residual at every captured position
of all 3,681 rows is replaced by the SAE reconstruction. That keeps
**≥ 99.0%** (RQ-KMeans) and **≥ 96.9%** (RQ-VAE) of what mean-ablation of the
same positions destroys, at every digit and layer. The spliced model loses
≤ 0.03 nats per digit.

## Q1: copy latents at readout(0)

A copy latent has selectivity ≥ 0.5 for one first-digit code of the recent
item, and support ≥ 20 on screening users. The rows below are on held-out
users. CC1 is Δ copy score, ablating the on-copy latents, minus the matched
control, in nats.

| | Layer 12 | 16 | 20 | 24 |
|---|---|---|---|---|
| RQ-KMeans: copy latents | 4 | 7 | 17 | 87 |
| RQ-KMeans: share of readout(0) activation on on-copy latents | 0.0% | 0.1% | 0.5% | 2.9% |
| RQ-KMeans: rows with any copy latent active | 8% | 15% | 43% | 75% |
| RQ-KMeans: **CC1** | 0.013 | 0.032 | **0.102** [0.092, 0.112] | **0.208** [0.189, 0.227] |
| RQ-KMeans: copy rate, ablation − control (CC2) | 0.0 | −0.9 pp | −4.2 pp | −6.5 pp |
| RQ-KMeans: Δ golden digit 0, copy latents | 0.003 | 0.006 | 0.011 | 0.039 |
| RQ-VAE: copy latents | 12 | 19 | 30 | 88 |
| RQ-VAE: share on on-copy latents | 0.6% | 1.0% | 2.4% | 12.0% |
| RQ-VAE: **CC1** | −0.021 | 0.005 | **0.102** [0.085, 0.117] | **0.360** [0.305, 0.417] |
| RQ-VAE: CC2 | −0.2 pp | −0.4 pp | 0.0 | −7.7 pp |
| RQ-VAE: Δ golden digit 0, copy latents | 0.000 | −0.001 | 0.042 | 0.123 |

- **Few latents are removed.** An ablation takes out 1 latent per row
  (RQ-KMeans, layers 12–20), rising to 2.7 at layer 24; for RQ-VAE, 1.3 to 4.8.
- **The copy code is readable, not dedicated.** When a copy latent is active,
  its preferred code is right 55–78% of the time. The recent item's first
  digit is linearly decodable at the readout from layer 0 (representation/exp1:
  84–93%), but this SAE does not isolate it as dedicated features before
  layer 20. It is spread across many latents' small contributions.

## Q2: prefix-match latents at readout(d ≥ 1)

The declared rule pools the AUC over d; A1 averages the within-d AUC. The
causal rows are on held-out users, pairs (row, d ≥ 1).

| | Layer 12 | 16 | 20 | 24 |
|---|---|---|---|---|
| RQ-KMeans, declared: best held-out AUC in S | 0.82 | 0.87 | 0.74 | 0.60 |
| … its within-d AUC (post-hoc diagnostic) | 0.94 | 0.97 | 0.92 | 0.78 |
| RQ-KMeans, declared: **CM1** Δ(match) − Δ(non-match) | 0.011 [−0.027, 0.046] | −0.006 | −0.007 | −0.007 |
| RQ-KMeans, A1: best held-out within-d AUC | 0.83 | 0.88 | 0.71 | 0.64 |
| RQ-KMeans, A1: CM1 | 0.011 [−0.027, 0.046] | 0.017 [−0.022, 0.053] | −0.001 | 0.001 |
| RQ-VAE, declared: best held-out AUC | 0.66 | 0.66 | 0.65 | 0.66 |
| RQ-VAE, declared: **CM1** | **0.018** [0.007, 0.030] | **0.031** [0.017, 0.046] | **0.023** | **0.018** |
| RQ-VAE, A1: best held-out within-d AUC | 0.69 | 0.66 | 0.61 | 0.59 |
| RQ-VAE, A1: CM1 | 0.020 [0.005, 0.033] | 0.012 | 0.022 | 0.031 [0.011, 0.051] |

- **Controls are clean.** The matched control sets change the golden digit by
  ≤ 0.003 nats. CM2 (S minus controls, matching pairs) equals CM1 to within
  0.005.
- **The scale of comparison.** exp4's single-edge knockout (readout(d) → the
  matching item's digit-d token) costs about 0.2 nats on the test split. These
  readout-latent ablations reach at most 0.03.
- **A match detector can go unread.** RQ-KMeans' S latents fire on the
  matching pairs (100% have one active at layers 12–16), and removing them does
  not move the output. What the output reads is the attention edge exp4
  knocked out, not this feature.

## Q3: target semantics beyond the history (observational)

The label is the target's Qwen3 content embedding (top 64 PCs). The model is
ridge regression with user-disjoint 5-fold CV.

- **The history baseline alone** (the recent and mean history embeddings)
  gives R² 0.058, and −0.018 / −0.030 on **new** targets (not in history,
  different first digit).
- **Adding the readout(0) residual never helps out of fold.** T1_R is −0.02
  to −0.05 at every layer. The 1,536-d ridge overfits with α ≤ 1000, so read
  this as "no detectable gain".
- **Adding SAE latents instead** gives T1_Z = −0.003 to +0.001 at layers 12–16,
  and lower later.

**S5 held.** Nothing about the target's content is linearly present at the
readout beyond a linear map of the history. This matches exp1's digit probes,
with a continuous label and far more power.

## Q4: same-day bursts (observational)

This is the AUC for "the most recent item is on the target's day", from the
readout(0) residual plus five SID-overlap features, against those features
alone (AUC 0.600 RQ-KMeans, 0.620 RQ-VAE):

| | Layer 12 | 16 | 20 | 24 |
|---|---|---|---|---|
| RQ-KMeans, T2 | **+0.082** [0.055, 0.108] | +0.082 | +0.084 | +0.090 |
| RQ-VAE, T2 | **+0.082** [0.059, 0.106] | +0.082 | +0.092 | +0.077 |

- **The model's state encodes burstiness beyond simple SID overlap**, from
  layer 12 on. Its readout AUC is about 0.69–0.71.
- **No single latent carries it.** The ten most same-day-predictive latents
  have held-out AUC 0.60–0.65 and are not copy latents.
- **This is content, not time.** The model cannot see time. What it has learnt
  is that some histories look like bursts, for instance several similar items
  in a row.
- **Present but not used (RQ-KMeans).** Retrospective exp6: the copy rate is
  67% at any age, ρ ≈ 0. RQ-VAE, whose copy rate does fall with age (ρ = 0.40),
  may be the one using it. That is untested.

## Expectations

| | Statement | RQ-KMeans | RQ-VAE |
|---|---|---|---|
| S1 | FVU < 0.25 at every site; F2 ≥ 70% | **held** for overall FVU (≤ 0.055) and F2 (≥ 99%). Within-role readout(0) FVU is above 0.25 at layers 12–20 | **held** (≤ 0.042; F2 ≥ 97%); readout(0) 0.13–0.33 |
| S2 | copy latents at every layer, ≥ 30% of readout(0) activation at layers 20/24 (RQ-KMeans) | **failed**: they exist, but carry 0.5% / 2.9% | (not declared) 2.4% / 12% |
| S3 | CC1 > 0 at layers 20 and 24 | **held** (0.10, 0.21) | **held** (0.10, 0.36) |
| S4 | best S held-out AUC ≥ 0.7 at layers ≥ 16 and CM1 > 0 | **half**: AUC held at 16 and 20; CM1 **failed** (≈ 0) | **half**: AUC failed (0.65–0.66); CM1 held (0.02–0.03) |
| S5 | \|T1\| ≤ 0.02 on new rows | held for Z at 12–16. R and later-layer Z are more negative (overfitting), never positive | same |
| S6 | T2 > 0.02 at some layer ≥ 16 | **held** (+0.08–0.09) | **held** (+0.08–0.09) |

## How this fits the intervention suite

- **Where the copy lives.** The copy is decided late (layers 20–24), and at
  the readout it runs partly through a few code-specific latents. That matches
  exp3's patching window (19–24) and exp1's logit-lens onset. Before layer 20,
  the information is decodable but not in dedicated features, and ablating
  what features exist does nothing.
- **Open question 2 of the report** (where the prefix match is computed) is
  narrowed. For RQ-KMeans it is not a feature the readout holds and the output
  reads. The remaining candidate is the attention query–key match itself: the
  readout's query attending to history tokens whose earlier digits match. Next
  step: SAEs or feature attribution on the attention inputs (queries at
  readout(d), keys at history digit tokens), or a QK-circuit analysis in layers
  14–27.
- **Targets.** No evidence of target knowledge beyond the history, in digits
  (exp1) or in content (Q3).
- **Time.** The model encodes a burst signal (Q4) that RQ-KMeans' copying
  ignores (exp6, exp9). A natural next test is causal: steer that direction
  and see whether copying responds.

## Limits

- **One SAE per cell and layer**, 12,288 latents and k = 32, fitted on 0.5–0.6M
  tokens; nothing was tuned. The dead fraction is high in early RQ-KMeans
  layers (47–54%).
- **Single layer, single position.** Ablations at one layer and one position
  leave later layers free to re-read the history. Null effects bound that
  position's role at that layer, not the whole computation.
- **Probe power.** Q3 and Q4 are linear, one split, 3,681 rows. The Q3 dense
  ridge overfits.
- **The time-related Q4 label** is from `sidlens.data.timestamps`. The model
  never sees it.
- **Seeds.** One training seed per recommender, and one SAE fit each; the
  intervals exclude both.
