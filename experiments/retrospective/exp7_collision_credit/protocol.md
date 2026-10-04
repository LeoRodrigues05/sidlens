# Retrospective exp7: how much of copying's benefit is collision credit?

Declared 2026-10-03, after structure exp3 (why items share a SID) and before
any exp6 prediction was re-scored. CPU only: this re-reads the exp6
copy-knockout outputs, which already exist.

## Why

exp6 found that prefix-matched copying helps AR recommendations. Blocking it
in the archived beam decoder (`C_all`) lowers exact-SID HR@10 by 5.7 pp
(RQ-KMeans 3×128) and 3.3 pp (RQ-VAE 4×128). Most of the RQ-KMeans loss
falls on "repeat" targets, whose SID is already in the history.

A SID hit is not an item hit. Structure exp3 showed that in RQ-KMeans 3×128,
44.8 % of items share their SID, often with a product variant of the same
brand. If the user's history holds item X and the target is a different item
Y with the same SID, copying X's SID scores an exact SID hit. At item level
that hit is worth 1/m for a bucket of m items. RQ-VAE's dedup loop gave most
such variants different SIDs, so there copying a variant's SID misses.

## Seen before this protocol

- exp6 RESULTS (published): `C_all − B` HR@10 = −27.3 pp on repeat-SID
  targets and −23.4 pp on repeat-item targets (RQ-KMeans). RQ-VAE: −10.1 pp
  on repeat-SID targets, −2.1 and −3.0 pp on new targets; its `C_later`
  costs new targets 2.8 pp.
- Structure exp3 (published today): collision causes, and the near-duplicate
  pair list (`293555/result/partA_near_duplicate_pairs.csv`).
- **Data-only class shares** of the 3,681 AR test rows (no model output):

  | Class | RQ-KMeans 3×128 | RQ-VAE 4×128 |
  |---|---|---|
  | same_item | 2.9 % | 2.9 % |
  | sid_partner | 11.3 % | 1.9 % |
  | nd_variant | 3.2 % | 9.9 % |
  | new | 82.6 % | 85.3 % |

  56.4 % of RQ-KMeans test targets and 6.4 % of RQ-VAE targets have a
  collided SID. The item ids in the AR rows match the SID tables on every
  target and history position.

## Inputs

- AR test rows: `sidlens.data.ar_prompts.load_examples(variant, "next-item",
  "test")` for `rqkmeans_3codebook_128` and `rqvae_4codebook_128`.
- exp6 predictions, under `derived/controlled/exp6_ar_copy_in_decoding/`:
  - archived decoder (primary), job 283114:
    `cell-0{0,1}/predictions_{B,C_all,C_later,N_later}.parquet`
  - plain beam search (secondary): `283586/cell-00` and `283621/cell-01`,
    `predictions_{B_plain,C_all_plain}.parquet`
- SID tables (`SidTable.cb2items`), with item ids via `item2id`.
- Near-duplicate pairs:
  `derived/structure/exp3_collision_causes/293555/result/partA_near_duplicate_pairs.csv`.
- The sha256 of every file read goes into `inputs.json`.

## Definitions

- **Row class**, assigned in this order (mutually exclusive):
  1. `same_item`: the target item id is in the history.
  2. `sid_partner`: the target's SID is in the history but its item is not.
     A different item with the same SID was bought, so a copied SID hit here
     is collision credit.
  3. `nd_variant`: the target is a near-duplicate (structure exp3 definition)
     of a history item, and its SID is not in the history.
  4. `new`: everything else.
- **SID HR@10**: the target SID is among the first 10 entries of the
  prediction list, exactly as exp6 scored it.
- **Item HR@10** (collision-corrected, CCE "uniform"): unique legal predicted
  SIDs are expanded into their item buckets in rank order
  (`sidlens.analysis.collisions.item_rank_bounds`). The target's expected hit
  at item cutoff 10 then assumes a uniform tie-break inside its bucket.
  Malformed and out-of-catalogue entries are skipped. The worst-case bound
  ("lower") is reported beside it.
- **Copy effect** of a knockout: knockout − baseline, per row, for each
  metric.
- **Variant copy** (nd_variant rows only): the first legal predicted SID is
  the SID of a history item that is a near-duplicate of the target.
- **Statistics:** paired user bootstrap, 2,000 draws, seed 20260927 (as in
  exp6), over the 1,606 users. All conditions and both metrics of a user are
  resampled together. Intervals exclude training-seed variance. Cells are
  never pooled.

## Estimands

Primary (archived decoder, `C_all − B`, each cell):

- **E1.** The share of the all-rows SID HR@10 loss that comes from
  `sid_partner` rows: sum of ΔSID over sid_partner rows / sum over all rows.
- **E2.** The item-level copy effect relative to the SID-level one:
  sum(Δitem) / sum(ΔSID) over all rows.
- **E3.** For RQ-VAE, `C_later − B`: the share of the new-or-variant rows'
  SID loss that comes from `nd_variant` rows, against those rows' share of
  the new-or-variant rows.

Secondary:

- ΔSID and Δitem HR@10 per class, for every knockout and both decoders.
- Baseline SID and item HR@10 per class.
- The share of baseline SID hits on `sid_partner` rows.
- The variant-copy rate under B and under `C_all`.

## Expectations, recorded before outcomes

- **X1.** RQ-KMeans: E1 ≥ 0.5. RQ-VAE: E1 < 0.15.
- **X2.** RQ-KMeans: E2 ≤ 0.67, so at least a third of the SID-level copy
  benefit disappears at item level. RQ-VAE: E2 within [0.9, 1.1].
- **X3.** RQ-VAE: nd_variant rows supply at least twice their row share of
  the `C_later` loss on new-or-variant rows (E3 ≥ 2 × row share).
- **X4.** In nd_variant rows the baseline's top SID is a variant copy in at
  least 20 % of rows, and `C_all` lowers that rate.

## Amendment A1 (post hoc, after job 293596)

At item cutoff 10, a small collided bucket copied to rank 1 still fits in the
10 item slots. The item metric then credits the target fully, so E2 at
k = 10 hardly tests collision credit. A1 adds HR@1 at SID level and at item
level (uniform: 1/m when the target's bucket of m items is ranked first),
the per-class deltas, and E2 at k = 1. It was added after seeing the k = 10
results. Separately, a 200-row smoke of the RQ-KMeans baseline (first rows
only) printed SID and item HR@10 before the full run.

## What this cannot establish

- It re-scores one decoding run per condition; it cannot say what a
  retrained, collision-free model would do.
- The item-level metric assumes a uniform tie-break. The model has no way to
  rank items inside a bucket, so that tie-break is the right neutral
  baseline. It is still not a measurement.
- Classes are observational strata of a randomised manipulation. The
  knockout is applied to every row; the classes only say where its effect
  falls.
