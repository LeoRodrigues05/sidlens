# Representation exp3: results

Runs: GPU job 294711 (self-greedy captures, SAEs at every second layer, lenses
and SID decoders) and CPU census job 294712, both on 2026-10-04. Summary job
300844 (2026-10-06):
`$SIDLENS_WORK/derived/representation/exp3_ar_information_map/summary-294711-294712/`
(`estimates.csv`, `census_summary.csv`, `report.md`, `figures/`).

Cells: `next-item_best` (RQ-KMeans 3×128) and `oneoff_rqvae4cb128` (RQ-VAE 4×128),
test split, 3,681 rows, 1,606 users. Intervals: paired user bootstrap, 2,000
draws, seed 20260927, 95% percentile; they exclude training-seed variance.

## Validation

- Captures: SID-table and archive checks pass on all 3,681 rows.
- Amendment A1 alignment check (greedy code within bf16 resolution of the best
  legal fp32 logit): 0.9998 / 1.000 of readouts, above the 0.99 limit. Strict
  fp32 agreement, the ceiling for D1/D2, is 0.985 in both cells.
- SAE test FVU: 0.002–0.083 (RQ-KMeans), 0.001–0.067 (RQ-VAE); it rises with
  depth.
- The greedy answer equals the golden target in 10.9% / 6.3% of rows, and
  equals the plain-beam top-1 in 69.5% / 41.9%. Everything below describes the
  greedy answer.

## Verdicts on the declared expectations

| | Expectation | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|---|
| E1 | At H_d, format, own token and own item dominate at layers ≤ 10; previous-item labels peak mid-depth; at R_0 the copy-source + own-answer share rises after layer 18 | **Partly.** Own token dominates H_d (86–98% at layer 0); format is < 0.2% everywhere. Previous-item labels rise to the end (H_0: 0.3% → 6.7% at layer 22), not mid-depth. At R_0 the share rises from 40% (L18) to 63% (L24). | **Partly.** Same H_d pattern. At R_0 the share stays flat at 20–23% from L18 to L24. |
| E2 | Answer-beyond-copy latents at R_0 at every layer ≥ 20; ≤ 1% of live latents before layer 16 | **First half holds:** 7, 23, 53, 57, 17 latents at L20–27. **Second half fails:** 1.6–6.9% of live latents at L4–14 (1–4 latents out of 32–69). | Same: 7–31 latents at L20–27; 1.5–5.6% at L6–14. |
| E3 | Tuned-lens whole-SID agreement D2 < 0.2 until L16, ≥ 0.8 by L26; tuned ≥ 0.1 above logit at L8–20 | **Fails on the early part.** D2 is 0.43 [0.41, 0.45] at L10 and 0.54 at L16; it passes 0.2 at layer 3. It reaches 0.88 [0.87, 0.89] at L26. Tuned minus logit is ≥ 0.40 at L8–20. | **Fails on the early part.** 0.28 at L10, 0.36 at L16. 0.83 [0.81, 0.84] at L26. Tuned minus logit ≥ 0.25. |
| E4 | `answer_at_R0` at L26 beats the copy rule; if it also does on non-copy answers, look-ahead | **Holds as stated:** 0.726 [0.709, 0.742] vs copy rule 0.425; non-copy answers 0.667. **But the look-ahead reading does not follow** (see below). | **Holds as stated:** 0.668 vs 0.162; non-copy 0.660. Same caveat. |
| E5 | `golden_at_R0` never beats the copy rule by more than 2 pp | **At the limit:** best layer (L12) 0.116 [0.102, 0.130] vs copy rule 0.095, +2.1 pp. | **Fails:** best layer (L24) 0.078 [0.067, 0.090] vs 0.045, +3.3 pp. |
| E6 | `item_at_last` ≥ 0.9 by L4; `prev_item_at_last` above prior mid-depth | **Holds:** 0.999 from L2. The previous item peaks at L14: 0.792 [0.778, 0.806] (prior 0.003). | **Holds:** 0.995 at L2. The previous item peaks at L12: 0.627 (prior 0.025). |

## What the results say

1. **Copy answers form early, other answers late.** The median first layer from
   which the tuned lens holds the whole answer at every later layer is 17 for
   copy answers and 23 for non-copy answers (RQ-KMeans), and 19 vs 23 (RQ-VAE).
   For repeat targets it is 14 / 6. The answer that copies the most recent item
   is readable well before the answer that does not.
2. **The readout position loses the copy source as the answer forms.** A
   decoder reads the most recent item's full SID from R_0 with 0.96 accuracy at
   layer 0. By layer 27 that falls to 0.87 (RQ-KMeans) and 0.52 (RQ-VAE). RQ-VAE,
   which copies less, overwrites the copy source more.
3. **Items are assembled by layer 2, and the previous item is read mid-depth.**
   The last token of every history item holds that item's full SID from
   layer 2. It also holds the previous item's SID, best at layers 12–14, and this
   fades again in late layers.
4. **No readable target beyond simple rules.** The golden target is decoded
   from R_0 at most 2–3 pp above "output the most recent item". The answer is
   not the target: greedy = golden in 11% / 6% of rows.
5. **Answer latents are a late, growing population.** SAE latents labelled with
   the model's own answer digit at R_0 grow from about 5 per layer before
   layer 20 to 53–92 at layers 24–26, and about 60% of them also track
   non-copy answers.

## Limits that change how to read E4

- **Look-ahead is not shown.** The `answer_at_R0` decoder already reaches
  0.62 / 0.49 at layer 0 (0.49 / 0.46 on non-copy answers). At that depth the
  model has done one attention read of the history. So the GRU decoder can
  compute much of the model's answer from history information itself. A high
  layer-26 score therefore does not show that the model has planned later
  digits at R_0. Separating the two needs a decoder with less capacity
  (linear, per digit) or a causal test.
- **E5's "beyond copy" is not "beyond the history".** A decoder can beat
  copying r1 by, for example, outputting r2. exp2's ridge analysis already
  found no target content beyond the history.
- **R_1 labels tie by construction.** Under self-forcing, the token at R_1 is
  the answer's digit 0, so `own_code` and `answer_d0` are the same variable
  there. The token / own-answer alternation at R_1 in figure 1 is a tie, not a
  finding.
- One SAE seed; labels depend on the dictionary. Brand is the only content
  label on this cluster.

## Figures

`figures/fig1_information_map_*.png` (census label shares per layer and token
group), `fig2_sid_lens.png` (tuned vs logit lens, whole answer and per digit),
`fig3_sid_decoders.png` (whole-SID decoders by layer, with prior and rule
baselines).
