# Does prefix-matched copying hold across SID maps? Core history estimates on 12 retrained Qwen-AR models

Declared 2026-10-06, before any intervened forward on a retrained model. The
only outcomes seen for these models are their acceptance scores
(`derived/substrate/exp1_ar_next_item_retrain/check-r20261004-20261006`).

## Question

The copy findings (exp3 recent-item replacement, exp4 prefix-matched copying
and the single-edge knockout, exp6 copy knockout inside decoding) come from two
checkpoints: RQ-KMeans 3×128 and RQ-VAE 4×128. Do they hold on the other maps
the paper reports: other depths, width 512, and MQ? Does a second training run
of the same map (the retrained RQ-KMeans 3×128) give the same numbers as the
original checkpoint?

## Cohort and models

- **Models:** the 12 retrained next-item models that `check.py` accepted
  (protocol A2 of substrate/exp1), registered in
  `manifests/registry.ar_retrain.json` and loaded as `retrain/r20261004/<variant>`.
  Each run re-hashes `model.safetensors` against the registry.
- **Not included:** the 6 models that failed acceptance (MQ 3×512, MQ 4×128,
  MQ 5×128, MQ 5×512, RQ-KMeans 3×512, RQ-KMeans 4×512) and the ZCR models
  (arm B has no acceptance rule; native-vs-ZCR needs its own protocol). A
  later amendment may add the failed six as a separately labelled set; they
  never enter the declared estimates below.

| cell | variant | cell | variant |
|---|---|---|---|
| 00 | rqkmeans_3codebook_128 | 06 | rqvae_4codebook_128 |
| 01 | rqkmeans_4codebook_128 | 07 | rqvae_4codebook_512 |
| 02 | rqkmeans_5codebook_128 | 08 | rqvae_5codebook_128 |
| 03 | rqkmeans_5codebook_512 | 09 | rqvae_5codebook_512 |
| 04 | rqvae_3codebook_128 | 10 | MQ_3codebook_128 |
| 05 | rqvae_3codebook_512 | 11 | MQ_4codebook_512 |

- **Rows:** the AR next-item test split (3,681 events, 1,606 users) and the
  valid split (3,680 events). Every variant's CSV has the same events in the
  same order, so users are shared across cells (checked before this protocol:
  identical (user, row) lists for all 12 test splits).

## Conditions: the existing experiments, unchanged

Each stage calls the existing `run.py` with `--ckpt retrain/r20261004/<variant>`.
The code is the same as in the original runs apart from checkpoint
resolution (`sidlens.models.ar.checkpoint_dir`/`checkpoint_sha`) and, in
exp6, the reference list for validation V1 (below).

| Stage | Script | Split | Parts / conditions |
|---|---|---|---|
| `exp4_valid` | exp4 `run.py` | valid | R, K (no head screen) |
| `exp3_A` | exp3 `run.py` | test | A (no residual patching) |
| `exp4_test` | exp4 `run.py` | test | R, K |
| `exp6` | exp6 `run.py` | test | B, C_later, C_all, N_later, B_plain, C_all_plain |

The designs, seeds, control-item draws, knockout layers (exp6: 14–27, as
declared for the 28-layer model) and acceptance controls are those of the
three protocols. Head screens (exp4 H) and residual patching (exp3 B) stay on
the two original checkpoints.

## Validation (before a cell's numbers count)

- **Teacher-forced stages:** the exp4 no-op knockouts K0a/K0b must be
  bit-exact on every row, as in exp4. A cell that fails is reported as failed,
  not estimated.
- **exp6 V1:** the hooked baseline `B` must reproduce the retrained model's
  own archived-style evaluation (`eval/predictions.json`: the same
  `evaluate.py`, 4 shards, batches of 8, seed 42 per shard). A retrained model
  never had the archive's lists, so those are not the reference. Acceptance:
  at least 99% of rows with an identical 50-item list. Below that, the cell's
  exp6 estimates are reported as not reproduced and excluded from G1.

## Estimands

Per cell, as declared in the source protocols (paired user bootstrap):

- **A1** (exp3): Δ_0(r = 1, m = 0), the cost in nats to the golden code at
  digit 1 of replacing the most recent history item. **A3:** its retained
  fraction at m = 1.
- **H1** (exp4, valid primary, test secondary): pooled over d ≥ 1, mean Δ_d
  over prefix-matching rows minus over non-matching rows.
- **H2a, H2b, H2c** (exp4): the single-edge knockout readout(d) → tok(k*, d) on
  matching rows; its paired difference from the control edge K3; match minus
  non-match.
- **Late share** (exp4, descriptive): Δ(K5 late) / Δ(K1 all) on matching rows,
  d ≥ 1.
- **P1, P2, P3** (exp6): ΔHR@10(`C_all` − `B`) on repeat rows, on new rows,
  and their difference. Secondary: `N_later`, `C_later`, the plain decoder,
  HR@{1, 5, 20, 50}, NDCG@10 and the copy-rate manipulation check.

Grid estimands (new):

- **G1 (primary):** for each of A1, H1 (valid), H2b (valid) and P1, the number
  of the 12 cells whose 95% interval excludes 0 in the direction found on the
  original checkpoints (A1 > 0, H1 > 0, H2b > 0, P1 < 0).
- **G2:** the retrained RQ-KMeans 3×128 against the original `next-item_best`
  on the same rows: paired differences in A1, H1, H2a and P1. This is a
  between-training-run comparison of one map and one recipe, the closest
  available measure of training noise. The original numbers are re-read from
  their result directories (exp3 280961, exp4 281261 / 281394, exp6 283114).
- **G3 (descriptive):** each estimand by quantizer, depth and width; the
  Spearman correlation across the 12 cells of H2a and P1 with the cell's
  baseline HR@10.

## Statistics

- Paired user bootstrap, 2,000 draws, seed 20260927, percentile 95%
  intervals, as in exp3/exp4/exp6. The same draws are used in every cell, so
  cells and the G2 contrast are paired by user.
- Intervals exclude training-seed variance. G2 is the only estimate of it,
  from one pair of runs.
- 12 cells × several estimands: per-cell intervals are descriptive. G1 counts
  them; it is not a test with a family-wise error rate.
- Different quantizers, depths and widths change the task (number of digits,
  codes per digit, collisions), so cross-cell contrasts are descriptive, not
  controlled effects of the map.

## Expectations recorded before outcomes

- **E1:** A1 > 0 in all 12 cells.
- **E2:** H1 > 0 and H2b > 0 in all RQ-KMeans and RQ-VAE cells. No declared
  expectation for MQ, whose digits are overlapping partitions rather than a
  hierarchy, so "sharing the first d digits" means something different there.
- **E3:** The late share is above 0.5 in every cell.
- **E4:** P1 < 0 in every cell whose V1 passes; |P2| < |P1|.
- **E5 (G2):** the retrained and original RQ-KMeans 3×128 agree within the
  bootstrap intervals on A1, H1 and H2a; P1 may differ more because the beam
  sampler's draws differ.

## Interpretation limits

- Retrained models reproduce the deleted sweep models' scores within the
  acceptance limits, not their weights. Statements are about models trained
  with the recorded recipe on these maps.
- SID-level metrics only. Item-level credit (retro exp7) is not re-scored
  here.
- The 6 failed models and the ZCR models are outside this protocol, so the
  grid is unbalanced: only 2 MQ cells and 1 width-512 RQ-KMeans cell.

## Outputs

`$SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/<array-id>/cell-XX/<stage>/`
(the stage's own `run.py` outputs), with source archive, hashes, environment
and `status.txt` per cell. Summary: `summary-<array-id>/` from
`summarize.py` (estimates CSV, report, figures).
