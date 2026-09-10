# Experiment 1: equal-beam rerun (2026-09-09)

Confidence-guided decoding is not uniformly better. Its advantage depends
strongly on the beam cap: it loses to fixed order at 64 but wins at 256.

The complete rerun used the same 18 frozen checkpoints and 6,297 test users
per checkpoint. Both methods used the same beam cap, checkpoint, history,
and target in each comparison. No models were retrained or reselected.
GPU inference ran through `sbatch` on `ws-l3-020` and took approximately
65 minutes, excluding implementation and pilot checks.

## Direct answer: retain the original confidence decoder, match the beams

These are equally weighted averages over all 18 configurations. HR@10 means
the target **SID** appears among the top ten distinct legal predicted SIDs;
it does not establish which item is correct when several items share a SID.
Differences and intervals are in percentage points, not relative percentages.

| Confidence / fixed beam | Confidence HR@10 | Fixed HR@10 | Difference | Paired-user 95% interval |
| --- | ---: | ---: | ---: | --- |
| 256 / 64 (original unequal comparison) | 14.562% | 14.218% | +0.344 | — |
| 64 / 64 | 13.255% | 14.218% | −0.963 | [−1.077, −0.846] |
| 256 / 256 | 14.562% | 14.300% | +0.261 | [+0.153, +0.370] |

Thus, the old gain is **not entirely explained by unequal beam widths**:
confidence still wins when both have beam 256. But saying that confidence
guidance simply performs better is too broad: matching both at 64 reverses
the result. Increasing the beam from 64 to 256 improves original confidence
by 1.307 points and fixed order by only 0.083 points.

NDCG@10 agrees in direction: original confidence minus fixed is −0.126 at
beam 64 and +0.326 at beam 256, on the 0–100 NDCG scale.

## Additional control: standardize final-digit expansion too

The frozen implementations differ beyond their beam caps: confidence greedily
fills its last digit, while fixed order considers all final-digit alternatives.
The shared-search comparison gives both policies full final-digit expansion,
identical score accumulation, deterministic tie handling, final catalogue
filtering, and maximum-score SID deduplication. Only the permitted next
position changes. It produces the same directional result:

| Shared beam cap | Confidence minus fixed HR@10 | Paired-user 95% interval |
| --- | ---: | --- |
| 64 | −0.949 points | [−1.066, −0.834] |
| 256 | +0.224 points | [+0.117, +0.338] |

The shared-search beam-64 comparison was specified as the primary analysis
before the full results. The original-decoder comparisons are separately
reported diagnostics answering the literal beam-only rerun.

## Why can a smaller beam hurt confidence decoding more?

Different reveal orders can create multiple paths to the same SID, consuming
beam slots before final deduplication. At beam 64, original confidence returns
ten distinct legal SIDs in only 49.39% of configuration-user cases, versus
94.66% for fixed order; mean list lengths are 7.545 versus 9.796. At beam 256,
those coverage rates rise to 91.87% and 99.17%, respectively.

This supports candidate diversity as a plausible explanation, not a proven
causal mechanism: invalid paths and final filtering also affect coverage.
An intermediate-deduplication intervention would be a separate experiment.
Equal beam caps are not equal numbers of unique candidates or equal compute.

## Checks and remaining limitations

- Both historical HR@10 baselines reproduced exactly in all 18 configurations.
  Maximum NDCG reconstruction error was below 8.51e-10.
- All 18 runs used identical source manifests and arguments. All checkpoint
  hashes and 72 historical metric baselines match the frozen registry.
- Prediction, target-identity, metric, and output-hash checks passed. Nothing
  was excluded. Independent recomputation reproduced the headline estimates
  and confidence intervals within 1e-12.
- Intervals use 2,000 paired bootstrap draws of users, keeping all configurations
  and conditions together. Configurations are not independent training trials.
- Checkpoints were originally selected using confidence-guided validation.
  Results remain conditional on those checkpoints and the original seed-42
  fixed order; they do not average over training seeds or arbitrary fixed orders.
- The code test suite passed: 163 tests, with 3 skipped.

Before future logit-based experiments, audit the existing
`sidlens.models.diffusion.digit_logits` helper: its `use_cache=False` vendor
fallback differs from the projected cross-attention inference path. This
rerun explicitly preserves projected cross-attention and validates against
the frozen decoder; the unrelated helper was not changed.

## Artifacts

- GPU array: `181944`; statistical-summary job: `181946`.
- [Complete statistical report](/l/users/leo.rodrigues/sidlens/derived/controlled/exp1_matched_beam/summary-181944-181946/report.md).
- [Numerical results and uncertainty](/l/users/leo.rodrigues/sidlens/derived/controlled/exp1_matched_beam/summary-181944-181946/result.json).
- [Per-configuration shared-search results](/l/users/leo.rodrigues/sidlens/derived/controlled/exp1_matched_beam/summary-181944-181946/configuration.csv).
- Per-user predictions, ranks, metrics, input hashes, and source archives:
  `/l/users/leo.rodrigues/sidlens/derived/controlled/exp1_matched_beam/181944/cell-00`
  through `cell-17`.
- Common source-manifest SHA-256:
  `ab6a9278a6f3b8d43018d857e36402206f1c40c4c680e86b23d516683341874c`.
