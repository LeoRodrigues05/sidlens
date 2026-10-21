# Core copy estimates on 12 retrained Qwen-AR models: job 301495, summarized 2026-10-15

Run on the CIAI cluster, job 301495. The per-cell outputs are on `/l`, which was
unreachable on 2026-10-15. The summary (`summary-301495/`, without parquet and
npz files) was copied to `/nfs-stor`. Protocol: [protocol.md](protocol.md),
declared before any intervened forward on a retrained model.

**Provisional.** These are the 12 accepted retrained models of substrate exp1
(arm A). The paper's final numbers come from the final training run's
checkpoints.

## What held

All four copy findings from the two original checkpoints hold on every
retrained model. G1 counts the models whose 95% interval excludes 0 in the
original direction:

| Estimand | What it measures | Models | Range over the 12 models |
|---|---|---|---|
| A1 | Cost to the golden digit 1 of replacing the most recent item (nats) | **12 / 12** | 0.37–0.78 |
| H1 (valid) | Changing a history item's digit d: matching minus non-matching item (nats) | **12 / 12** | 0.89–2.18 |
| H2b (valid) | Knocking out the edge readout(d) → matching item's digit-d token, minus a control edge (nats) | **12 / 12** | 0.20–0.60 |
| P1 | Copy block in beam search, ΔHR@10 on repeat targets (points) | **12 / 12** | −37.0 to −7.5 |

The expectations recorded before the run:

- **E1 held.** A1 > 0 in all 12 models.
- **E2 held**, and also on the two MQ models, for which no expectation was
  declared. H1 and H2b exceed 0 in every model.
- **E3 held.** Layers 14–27 carry 0.98–1.00 of the edge knockout in every
  model.
- **E4 held.** P1 < 0 in every model. |P2| < |P1| everywhere: the copy block
  costs new targets 0.1–4.3 points, against 7.5–37.0 on repeat targets.
  Blocking only later-position items (`N_later`) changes nothing (−0.11 to
  +0.11 points).
- **E5 held in part.** The retrained RQ-KMeans 3×128 and the original
  `next-item_best`, scored on the same rows with the same draws:

  | Estimand | Retrained | Original | Difference |
  |---|---|---|---|
  | A1 | 0.541 | 0.570 | −0.029 [−0.047, −0.010] |
  | H1 | 1.307 | 1.344 | −0.037 [−0.126, +0.051] |
  | H2a | 0.368 | 0.398 | −0.030 [−0.067, +0.006] |
  | P1 | −37.0 | −27.3 | −9.8 [−15.1, −5.0] |

  H1 and H2a agree. A1 differs by 5%, with an interval that excludes 0. P1
  differs by 10 points, as expected, because the beam sampler's draws differ
  between the two models. This pair is the only measure of training-run
  variation we have.

## Validation

- Every exp4 no-op knockout is bit-exact in every model.
- V1: each model's hooked beam baseline reproduces its own evaluation lists in
  3,680 of 3,681 rows (identical 50-item list), above the 99% acceptance
  level.

## Patterns across models (descriptive)

- **The largest A1** is on MQ 4×512 and RQ-KMeans 4×128 (0.77–0.78 nats);
  the smallest on RQ-VAE 5×128 and 5×512 (0.37–0.40).
- **Keeping the replaced item's first digit removes more of A1 on RQ-KMeans
  than on RQ-VAE.** The share left (A3) is 0.14–0.19 on the width-128
  RQ-KMeans models (0.35 at 5×512) and 0.21–0.48 on RQ-VAE, as on the two
  original checkpoints (0.18 and 0.36).
- **Better models lose more from the copy block.** Across the 12 models,
  Spearman ρ between P1 and baseline HR@10 is −0.63. The edge knockout (H2a)
  hardly tracks HR@10 (ρ = 0.14).
- **Test split.** H1 and H2b are smaller on test than on valid, as on the
  original checkpoints (H2b test 0.06–0.40), and every interval excludes 0.

## Limits

- One training run per map. Intervals exclude training-seed variance; E5 is a
  single pair.
- The grid is unbalanced. It has 2 MQ models and 1 width-512 RQ-KMeans model,
  because 6 retrained models failed acceptance.
- SID-level metrics only. Item-level credit (retrospective exp7) has not been
  re-scored on these models.

## Files

`$SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/summary-301495/`:

- `estimates.csv`: every estimand per model
- `g1_counts.csv` and `g2_retrained_vs_original.csv`
- `report.md`
- `figures/`: `fig1_core_estimates` (in the paper as `fig_ar_grid.pdf`),
  `fig2_edge_layers_prefix`, `fig3_decoding`, `fig4_mechanism_vs_accuracy`
  and `fig5_retrained_vs_original`
