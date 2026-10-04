# Layer-wise decoding of SID digits from AR activations (logit lens and probes)

Declared 2026-09-27, after the captures had been written but before any
decoding or probe had been computed from them. This is observational:
decodability shows information is present, not that the model uses it. exp3
and exp4 test use.

## Inputs

The test-split captures made on the CIAI cluster:

- `derived/ar_capture/281053`: `next-item_best`
- `derived/ar_capture/281054`: `oneoff_rqvae4cb128`

Both use the eval wording, teacher forcing and bf16. They store 29 sites
(`model.layers.0…27` and `model.norm`) at roles `hist_sid`,
`response_header` and `target_sid`. Before use, the store manifest's
checkpoint sha must equal the frozen manifest, and every shard hash must
verify. `two-item_best` is excluded: its slot-1 digit-0 readout (a separator
token) was not captured.

**Positions.** readout(d) is the stored row at `predict_pos(0, d)`:

- d = 0: the last `response_header` token
- d ≥ 1: the `target_sid` token of digit d−1

k\* is the most recent history item, and last(k\*) is its digit-(n−1) token.

## Q1: logit lens (no training)

For each site L and digit d, compute logits over digit d's code tokens at
readout(d) as E_d · RMSNorm_final(h_L), with fp32 math. E_d holds the tied
embedding rows of those tokens; for the `model.norm` site, h is already
normed. Report per layer:

- the golden code's top-1 rate and mean log-prob
- at d = 0, the top-1 rate of **k\*'s digit-0 code** (does the readout lean
  toward the most recent item?)

**Acceptance.** At the `model.norm` site, ranks must agree with the capture's
`digit_scores.csv` for ≥ 99% of (row, digit) pairs. The residual difference
is bf16 storage.

## Q2: linear probes

Multinomial logistic regression on standardized features (fp32, full-batch
L-BFGS, 200 iterations). The L2 strength is chosen from {1e-4, 1e-3, 1e-2}
by inner validation on 20% of each training fold's users. Evaluation is
**user-disjoint 5-fold** cross-validation, with folds from
`sha256("20260927|<user_id>")` mod 5. Classes absent from a training fold are
never predicted.

| Probe | Site | Label |
|---|---|---|
| P1 | readout(0) | target digit-0 code |
| P2 | readout(0) | k\*'s digit-0 code |
| P3 | last(k\*) | k\*'s digit-0 code (item assembly) |
| P4 (secondary) | readout(1) | target digit-1 code |

**Metric.** Out-of-fold top-1 accuracy, pooled over folds, with a
user-bootstrap 95% interval (2,000 draws, seed 20260927).

**Baselines, reported beside every probe:**

- majority class of the training fold
- labels shuffled within the training fold, at layers {0, 7, 14, 21, 27}
- for P1, the no-training copy baseline "predict k\*'s digit-0 code"

## Expectations recorded before outcomes

- **E1:** The logit-lens golden top-1 at readout(0) stays near chance until
  about layer 18, then rises. This is exp3's late transfer.
- **E2:** P3 is high from early layers: an item's last token carries its
  first digit early.
- **E3:** P2 becomes decodable at readout(0) no later than P1.

## Limits

- Probes find linearly decodable information, not information the model uses.
- There is one split (test), 3,681 rows, and up to 128 classes, so rare
  classes are hard for any probe.
- Industrial category labels are not on this cluster, so category probes are
  deferred.

## Amendment (2026-09-27, before any intermediate-layer output was examined)

The Q1 acceptance check as first written failed on a CPU smoke run: at the
`model.norm` site, fp32 lens ranks agreed with the capture's
`digit_scores.csv` for only 87.5% of (row, digit) pairs. A diagnostic on 300
rows located the cause in the reference, not the lens. The capture ranks
**bf16** logits, the model's own output dtype. At logit magnitudes of 16–32
bf16 values are spaced 0.125 apart, so near-tied codes share a value and a
"strictly greater" rank changes.

| Lens logits | Rank agreement | Top-1 agreement | Max \|Δ log p\| |
|---|---|---|---|
| fp32 | 70.7% | 99.7% | 0.039 |
| rounded to bf16 | 99.3% | 100% | 0.003 |

The acceptance check therefore ranks the lens logits after rounding them to
bf16, with the same ≥ 99% threshold. The reported layer-wise metrics stay in
fp32, which is the more precise and less tie-prone option. Both
agreements are recorded in `validation.json`.

## Disclosure (2026-09-27)

`summarize.py` was tested on the output of a CPU smoke run, which let me
see some partial outcomes before the declared run. The smoke run used the
real `281053` store, layers 0 and 28 only, and 10 L-BFGS iterations instead
of 200, so its probes were undertrained. The numbers seen were:

- P1: 0.13 at layer 0 and 0.16 at layer 28
- P2 at layer 0: 0.57
- the copy baseline: 0.16

Nothing in the design, the estimands or the thresholds changed after that.
