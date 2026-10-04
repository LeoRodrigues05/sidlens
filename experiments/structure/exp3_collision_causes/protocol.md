# Structure exp3: why do items share a Semantic ID?

Declared 2026-10-03, before any estimand below was computed. CPU only: the
experiment reads the frozen SID tables, the frozen item embeddings and item
metadata, and refits quantizers with faiss. No recommender is run.

## Why

Collision rates differ by two orders of magnitude across the 27 Industrial
tables: 29.3 % of items share a full SID at RQ-KMeans 3×128, 40.1 % at MQ
3×128, 1.2 % at RQ-VAE 3×128, and about 0.35 % (the 11 identical-text items)
at most deep RQ-VAE fits. Collisions matter downstream: SID-level HR@10
overstates item-level HR by 4.5 pp, and collided targets are hit 42.5 pp more
often (retrospective exp3). Before treating "collision rate" as a property of
a quantizer family, we need to know what produces each collision: the
catalogue (near-identical items), the quantizer's geometry, or a processing
step applied after quantization.

## Seen before this protocol

Code and provenance (read, not computed):

- **RQ-VAE.** `OneDiffRec/rq/generate_indices.py` encodes every item with
  nearest-code assignment. Then, for up to 20 rounds, it re-encodes every
  group of items that share a full SID with a Sinkhorn assignment
  (ε = 0.003) on the **last level only**; earlier levels stay nearest-code.
  The RQ-VAE weights were not saved.
- **RQ-KMeans.** `OneDiffRec/rq/rqkmeans_faiss.py` trains
  `faiss.ResidualQuantizer` (`Train_default`, beam 1). An optional `--uniform`
  flag then rebalances every level with Sinkhorn to capacities ⌊N/K⌋ or
  ⌈N/K⌉, filling codes greedily in a random item order.
- **MQ.** DiffGRM's tokenizer default is `OPQ{D},IVF1,PQ{D}x{bits}` (inner
  product). The command that built the frozen MQ tables is not recorded, and
  2560 is not divisible by 3, so the 3-digit tables cannot be plain PQ on the
  frozen 2560-d embedding. MQ is analysed through its tables only. Its refits
  below are counterfactuals, not reconstructions.
- **Reconstruction checks** (pipeline validity, on the login node):
  - `faiss.ResidualQuantizer` refit at 96 OpenMP threads reproduces
    `rqkmeans_3codebook_128` exactly (3,105 / 3,105 rows).
  - A plain loop of `faiss.Kmeans(niter=10, seed=1234)` per level on the
    residuals, with nearest-centroid assignment, also reproduces it exactly.
    So faiss's RQ training is per-level k-means that **reuses one seed at
    every level**.
  - The same refit at 5×512 matches the frozen 5×512 table at 0.03 % of
    first-digit labels, so the frozen 5×512 table is not this procedure.
    The upstream audit (`OneDiffRec/docs/data-transfer.md`, 2026-10-02) calls
    it "cluster-balanced (Sinkhorn-style)" and reports that the 96-thread
    refit reproduces the other 8 RQ-KMeans tables.
  - Rebuilding centroids as the mean residual of each native code reproduces
    97–99 % of rows, not all; it is not used.
- **Earlier results** (semantic-mapping takeaways, 2026-09-22): the rates
  above; RQ-VAE's last digit leaves 99 % of items alone; anomalous RQ-VAE
  buckets of 26 items (3×128) and 19 items (5×256); 11 identical-text items
  that every table except RQ-KMeans 5×512 collides; each MQ digit alone
  explains 36–38 % of embedding variance.
- **Glimpsed, not counted:** in the first ~60 rows of the RQ-KMeans 3×128 info
  file, several SIDs repeat one code value across digits (`<a_40><b_40><c_40>`,
  `<a_55><b_55>`, `<a_57><b_57>`, `<a_64><b_64>`), and many MQ 3×128 SIDs end
  in code 46.

## Hypotheses

**H0, capacity (reference only).** Collisions come from too few addresses.
A uniform random code for every digit gives a collision rate of about
N / K^D, which is at most 0.15 % here (3×128). H0 is not expected to hold;
it fixes the floor the other nulls are compared against.

**H1, a processing step sets the RQ-VAE and RQ-KMeans 5×512 rates.**
(a) RQ-VAE's low rates come from the last-digit Sinkhorn loop, not from the
learned quantizer. The loop cannot split items whose inputs are identical, so
the remaining collisions should be identical inputs, plus groups for which the
loop failed. (b) RQ-KMeans 5×512 was capacity-balanced at every level; the
greedy capacity fill is what separates its identical-text pairs.

**H2, near-duplicate items.** Collided items are product variants (same
brand, near-identical title, nearest neighbours in the embedding) that the
embedding barely separates. They form a floor that more digits or wider
codebooks remove last.

**H3, shared residual codebooks (RQ-KMeans).** Each level ≥ 2 has one
codebook for all parents. (a) A parent's items fall into fewer child codes
than the level's global code frequencies imply. (b) Because faiss reuses the
k-means seed at every level, the same items seed the initial centroids at
every level, so level-l code j starts inside the parent that owned code j at
level l − 1 and absorbs its items (an "own-code sink"). (c) Parents whose
residuals are small compared with the codebook's spacing land in one or two
cells (scale mismatch).

**H4, anisotropy of the embedding.** The unnormalised Qwen3 embedding puts
most variance in a few directions. k-means spends its codes on those
directions, and items that differ only in low-variance directions collide.

**H5, MQ digits are redundant.** MQ's parallel digits partition the same
semantic structure, so the digits are strongly dependent and the number of
occupied digit combinations is far below K^D.

## Inputs

- Embeddings: `frozen/data/embeddings/Industrial_and_Scientific.emb-qwen-td.npy`
  (float16, upcast; row = item id). All refits use float32, as upstream did.
- SID tables: all 27 `frozen/sids/sem_ids/diffgrm/*.sem_ids`, re-indexed by
  item id.
- Metadata: `frozen/data/item_meta/Industrial_and_Scientific.item.json` (title,
  brand). Category labels are not on this cluster and are not used.
- sha256 of every input file in `inputs.json`.

## Definitions

- **Collided item:** an item whose full SID is shared with at least one other
  item. **Collision rate** (primary) = share of collided items. The repo's
  older rate, 1 − unique SIDs / N, is reported beside it.
- **Identical inputs:** two items whose stored float16 embeddings are equal.
- **Near-duplicate pair** (defined from the catalogue alone, never from a SID):
  one item is among the other's 5 Euclidean nearest neighbours, both have the
  same non-empty brand (case- and space-folded), and title-token Jaccard
  ≥ 0.5 (lower-case alphanumeric tokens). Identical-input pairs count as
  near-duplicates.
- **Nulls**, each with 200 permutations (seed 20261003):
  - N0 uniform: every digit is drawn uniformly from K codes (computed
    analytically).
  - N1 independent digits: shuffle each digit column on its own. This keeps
    every digit's code frequencies and removes the dependence between digits.
  - N2 sequential (per level l ≥ 1): keep the true prefix up to digit l − 1
    and shuffle digit l across all items. This keeps the level's global code
    frequencies and removes any tendency of a parent's items to share a code.
  - **Merge ratio** at level l: the number of items that stay merged after
    adding digit l (N − distinct prefixes of length l + 1, among items whose
    length-l prefix is shared), divided by the same number under N2. Above 1,
    a parent's items share codes more than the global frequencies imply.
- **Refit procedure "native"**: per-level `faiss.Kmeans(K, niter=10)` on the
  float32 residuals with one seed s reused at every level, and nearest-centroid
  assignment, at 96 OpenMP threads. With s = 1234 it must reproduce the
  archived nested RQ-KMeans tables: all depths at widths 128 and 256, and
  depths 3 and 4 at 512. If it does not, the run stops. Each width is fitted
  once at 5 levels and truncated to depths 3 and 4. The faiss RQ is greedy,
  so truncation equals a shallower fit.
- **Seeds:** s ∈ {1234, 1, 2, 3, 4}. A refit condition **changes** the
  collision rate only if its 5-seed range does not overlap the matched
  baseline's 5-seed range. The paired per-seed difference is also reported.

## Analyses and primary estimands

### Part A: the catalogue

- The embedding spectrum: participation ratio, variance share of the top
  10 / 64 / 256 PCs, and the spread of norms.
- Identical-input groups; near-duplicate pairs and the items in them.
- The distribution of all pairwise distances and of 1-NN distances.

### Part B: the 27 archived tables

For every table:

- collision rate, both definitions, and bucket sizes
- N0, and the N1 collision rate (5th / 50th / 95th percentile)
- per-level merge ratio under N2
- per-level code-count range and normalised entropy
- per-level repeat rate P(digit l = digit l − 1), against 1/K
- the share of collided items that are in near-duplicate pairs with a
  bucket-mate, and the share of near-duplicate pairs that collide
- Spearman correlation, across parents at depth D − 1 with ≥ 2 items,
  between the parent's RMS radius (raw space) and its share of collided items

The primary estimands, by hypothesis:

- **H1a.** For each RQ-VAE table, the last-level merge ratio. Also the share
  of its collided items that are identical-input items, and the composition
  (titles, distinct inputs, distance percentile) of every RQ-VAE bucket with
  ≥ 3 items.
- **H1b.** For each RQ-KMeans table, whether every level's code counts lie in
  {⌊N/K⌋, ⌈N/K⌉}, and whether any identical-input pair has distinct SIDs.
- **H2.** The item-level near-duplicate share of collided items, by table. For
  the nested RQ-KMeans fits, its trend with depth at each width.
- **H3a.** The merge ratio at levels ≥ 1 for RQ-KMeans.
- **H3b.** The RQ-KMeans repeat rate, and the share of collided items whose
  last two digits are equal, against all items.
- **H5.** For each MQ table, observed collision rate / median N1 rate, and the
  mean pairwise NMI between digits. The same numbers are reported for RQ
  tables for contrast.

### Part C: RQ-KMeans refits (interventions on the procedure)

Every condition is run at widths 128, 256 and 512, at depth 5 (truncated to
3 and 4), with all 5 seeds:

| Condition | What changes from "native" |
|---|---|
| native | nothing (s reused at every level) |
| per-level seed | level l uses seed s + 7919·l |
| scale-norm | before fitting level l ≥ 1, divide each parent's residuals by the parent's RMS residual norm; encode in that scale; the next residual is r − scale·centroid |
| scale-norm + per-level seed | both |
| L2 | rows L2-normalised before fitting |
| PCA-256 | centred, projected on the top 256 PCs (not whitened) |
| PCA-256-white | top 256 PCs, each scaled to unit variance |
| PCA-64-white | top 64 PCs, whitened |

Primary estimands:

- **H3b.** per-level seed − native: collision rate, and the repeat rate.
- **H3c.** scale-norm − native.
- **H4.** PCA-256-white − PCA-256, and L2 − native.

For every condition, the cost is also reported: the R² of the raw embedding
explained by the depth-1 and depth-2 codes, and the 10-NN agreement (the
share of an item's 10 raw-space nearest neighbours that share its first
digit). "Fewer collisions" bought by scattering neighbours is not a fix.

### Part D: post-processing simulations (H1)

- **D1 (H1a).** Apply `generate_indices.py`'s loop to the 8 reproducible
  RQ-KMeans tables, with the refit codebooks: centred distances, ε = 0.003,
  50 Sinkhorn iterations, argmax, ≤ 20 rounds, groups re-detected each round,
  last level only, float64. Report the collision rate after the loop; the
  remaining buckets, split into identical-input buckets and others; the
  rounds used; the share of items whose last digit changed; and the mean
  increase in last-level squared residual. If H1a holds, every table ends at
  or near the identical-input floor whatever its starting rate.
- **D2 (H1b).** Apply the `--uniform` Sinkhorn balancing of
  `rqkmeans_faiss.py` to the native seed-1234 fits at 3×128 and 5×512. It is
  reimplemented in numpy: POT is not installed, and τ, the 30-candidate
  greedy fill and the seed 42 + level are as upstream. Report the collision
  rate, the code-count range per level, and whether identical-input pairs
  are split. This is not expected to reproduce the frozen 5×512 table, whose
  base fit is unknown.

### Part E: parallel-digit refits (exploratory, H5)

The embedding is centred, projected on the top 2,400 PCs (divisible by 3, 4
and 5), and split into D blocks in three ways:

- random orthogonal rotation, then contiguous blocks
- PCs dealt round-robin (balanced variance)
- contiguous PCs (block 1 holds the top PCs)

Each block gets its own k-means (`faiss.Kmeans`, niter = 25, as faiss PQ),
for all 9 (D, K) and 5 seeds. Report the collision rate, the mean pairwise
NMI between digits, and the N1 ratio. This is exploratory because the MQ
build recipe is unknown.

## What would count as support

- **H1a** if every RQ-VAE table has a last-level merge ratio < 0.2 (RQ-KMeans
  and MQ ≥ 0.5 expected), at least 80 % of RQ-VAE collided items outside
  buckets of ≥ 3 items have identical inputs, and D1 brings every RQ-KMeans
  table to within 0.5 pp of its identical-input floor.
- **H1b** if every level of RQ-KMeans 5×512 is capacity-balanced and no other
  RQ-KMeans table is.
- **H2** if collided items are enriched for near-duplicates (their
  near-duplicate share exceeds the share among all items in multi-item
  depth-(D − 1) groups), and within each width the share rises with depth.
  "Near-duplicates are the floor" additionally needs more than 50 % at the
  deepest reproducible fit (5×256).
- **H3a** if the RQ-KMeans merge ratio is > 1 at most levels ≥ 1 in most
  tables.
- **H3b** if the RQ-KMeans repeat rate is ≥ 3/K (RQ-VAE and MQ ≈ 1/K), and
  the per-level seed lowers the collision rate under the decision rule above.
- **H3c** if scale-norm lowers the collision rate under the decision rule,
  without lowering 10-NN agreement by more than 0.05.
- **H4** if PCA-256-white lowers the rate relative to PCA-256 under the
  decision rule. Report the cost beside it.
- **H5** if every MQ table's observed / N1 ratio is ≥ 2.

A hypothesis can be supported for some tables and not others; the report
says which.

## Amendments after the Part A/B smoke run (post hoc, 2026-10-03)

A smoke run of Parts A and B (10 permutations, login node) was inspected
before the full run. Parts C–E had not been run. Two declared rules turned
out to be poorly matched to what they were meant to measure. Both are still
reported exactly as declared. The amendments below are labelled post hoc
wherever they appear.

- **A1 (H1a).** The last-level merge ratio counts the 11 identical-input
  pairs. No assignment can split those, and in deep RQ-VAE tables they are
  nearly the only merges, so the declared ratio exceeds 1 for reasons
  unrelated to the loop. A1 recomputes the ratio, and the near-duplicate
  share, after removing the 22 identical-input items. If the loop operates,
  the A1 ratio should be < 0.2 for RQ-VAE and ≥ 0.5 for RQ-KMeans and MQ.
- **A2 (H1b, H3c).** RQ-KMeans 5×512 is not capacity-balanced: its counts
  range from 2 to 81 per code. Its repeat rate is ≈ 1/K, unlike every other
  RQ-KMeans table. A2 describes rather than tests:
  - per-level count range and entropy, against the D2 balanced mapping
  - for the native refits, each level's most-used ("hub") code: its share of
    items, its centroid norm against the median centroid norm, and the share
    of collided items whose last digit is that level's hub code

  The smoke run showed one level-1 code holding 13.5 % of items in
  RQ-KMeans 4×512.
- **A3 (H1b, added after the full run, job 293555).** The faithful
  reimplementation of `--uniform` (D2) is not capacity-balanced either. When
  none of an item's 32 best codes has room, upstream assigns it to
  `argmin(remaining)`, the code already most over capacity. So the declared
  H1b rule tested a property the procedure does not have. A3
  (`a3_balance_signature.py`) compares fingerprints between the frozen 5×512
  table, the native 5×512 refit, and `--uniform` applied to that refit:
  - codes exactly at capacity per level
  - the largest code
  - identical pairs split
  - collision rate
  - 10-NN first-digit agreement
  - repeat rate
  - first-digit ARI
- The smoke run also showed that the "29.3 %" quoted for RQ-KMeans 3×128 in
  the semantic-mapping takeaways is 1 − unique / N. The share of items in a
  shared SID, this protocol's primary rate, is 44.8 %.

## What this cannot establish

- RQ-VAE's latents and codebooks are gone, so H1a is tested through its
  signature (table structure) and a simulation on RQ-KMeans geometry, not by
  re-running RQ-VAE.
- MQ's build is unknown; its refits are counterfactuals.
- One catalogue. Seed ranges cover k-means initialisation only.
- Nothing here says whether a recommender is hurt by a given collision type.
  That needs the model-side follow-up.

## Outputs

`$SIDLENS_WORK/derived/structure/exp3_collision_causes/<job-id>/`: the
source archive and hashes, `inputs.json`, `validation.json` (the native
reproduction check), one CSV per part, `result.json`, `report.md`,
`status.txt` and `output.sha256`.
