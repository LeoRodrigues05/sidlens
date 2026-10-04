# Why items share a Semantic ID: completed 2026-10-03

Each quantizer family collides for a different reason, and two of the three
families' rates are set by a step that runs after quantization:

1. **RQ-VAE's near-zero rates come from post-processing.** Upstream's
   `generate_indices.py` rewrites the last digit of colliding items with a
   Sinkhorn assignment. Applied to the RQ-KMeans tables, that loop alone takes
   every one of them from 10–45 % to exactly the identical-input floor (22
   items, 0.71 %). It changes the last digit of up to 44 % of items. Each item
   it moves gains 1.0–2.8 times the table's mean last-level squared error.
2. **RQ-KMeans 5×512 is a balanced table.** Its low rate (0.64 %) comes from
   upstream's `--uniform` Sinkhorn balancing. That balancing halves the share
   of an item's 10 nearest neighbours that share its first digit (0.46 → 0.22).
3. **Near-duplicate products are the floor.** In the nested RQ-KMeans fits,
   the share of collided items with a same-brand, near-identical-title
   neighbour in their bucket rises with depth: 46 → 63 → 73 % at width 128,
   60 → 73 → 81 % at 256, and 75 → 85 % at 512.
4. **Shallow RQ-KMeans collisions are structural.** At every level after the
   first, the most-used code is the centroid nearest the origin. It holds
   9–17 % of items, all with small residuals, and parents with a small
   radius collide more.
5. **MQ collides because its digits repeat each other.** Its digits are
   strongly dependent: mean pairwise NMI 0.50–0.70. Shuffling each digit
   independently gives a collision rate of 7.4 % at 3×128, against 54.5 %
   observed, and about 0 % at every other size.

Capacity, embedding norm, whitening and faiss's shared k-means seed do not
explain collisions. The shared seed leaves a clear signature in the tables but
**lowers** the collision rate.

The protocol ([protocol.md](protocol.md)) was declared before any estimand was
computed. A smoke run of Parts A–B (job 293551) was inspected before the full
run, and amendments A1–A2 were added after it. A3 was added after the full
run. All three are post hoc and are labelled as such below. No recommender is
run, and there are no user-level intervals. Ranges are over 5 k-means seeds
and cover initialisation only. There is one catalogue.

| Run | Job | Output under `$SIDLENS_WORK/derived/structure/exp3_collision_causes/` |
|---|---|---|
| Full run (Parts A–E, A1, A2) | 293555 | `293555/result/` (`report.md`, CSVs, `result.json`, `validation.json`) |
| Smoke run (width 128, 2 seeds, 20 permutations) | 293551 | `293551/result/` |
| A3, balancing fingerprints of 5×512 | 293573 | `a3-20261003/` |

## Numerical acceptance

- **Native reproduction.** Per-level `faiss.Kmeans(niter=10)` with seed 1234
  reused at every level, at 96 threads, reproduces all 8 nested archived
  RQ-KMeans tables row for row (3,105 / 3,105). The refit codebooks are
  therefore the archived tables' own centroids.
- **5×512.** The refit matches the frozen 5×512 table on no row; the
  first-digit ARI is 0.21. The frozen table is a separate fit (see A3).
- **Collision definition.** The primary rate is the share of items whose full
  SID is shared. The semantic-mapping takeaways quoted 29.3 % for RQ-KMeans
  3×128 as "items sharing a full SID". That number is 1 − unique / N; the
  share of items is 44.8 %. Both are in `partB_tables.csv`.

## Catalogue (Part A)

- 11 pairs of items have identical stored embeddings (22 items, 0.71 %).
- 1,291 near-duplicate pairs cover 900 items (29 %). A pair counts if:
  - one item is among the other's 5 nearest neighbours
  - both have the same brand
  - their title-token Jaccard is ≥ 0.5

  Typical pairs are size, wattage or length variants of one product, e.g.
  Wilton 3-inch and 4-inch drill-press vises, or Gorilla clear duct tape
  9 yd and 5 yd.
- The spectrum is moderately concentrated:
  - participation ratio 49
  - the top 10 / 64 / 256 PCs carry 38 / 70 / 89 % of the variance
- Norms vary by only 4 % (coefficient of variation). The mean vector has norm
  80, while the RMS distance to the mean is 46.

## Results by hypothesis

### H0, capacity: rejected

Uniform random codes would collide 0.15 % of items at 3×128 and about 0 %
elsewhere. Even the independent-digit null (N1), which keeps every digit's
code frequencies, gives at most 7.7 % (RQ-KMeans 3×128). Every table exceeds
both.

### H1a, the RQ-VAE last-digit loop: supported in substance; declared rule failed

| | Declared rule | Outcome |
|---|---|---|
| Last-level merge ratio < 0.2 in every RQ-VAE table | failed (0.93–9.2) | The 11 identical-input pairs are counted, and in deep RQ-VAE tables they are almost the only merges |
| ≥ 80 % of collided items in 2-item buckets have identical inputs | failed in 3×256 (46 %) | holds in the other 8 |
| D1 brings every RQ-KMeans table to within 0.5 pp of the floor | **held**: every table ends exactly at 0.71 % | |

**A1 (post hoc)** removes the 22 identical-input items and recomputes the
last-level merge ratio:

- RQ-VAE: 0.00 in 5 tables; 0.12 and 0.13 in 4×128 and 3×512; 0.66–0.68 in
  3×128 and 3×256; 6.2 in 5×256
- RQ-KMeans and MQ: ≥ 3.1 in every table

**D1, the loop applied to RQ-KMeans geometry.** This uses the archived
centroids and the loop's own settings: ε = 0.003, 50 Sinkhorn iterations,
≤ 20 rounds.

| Table | Before | After | Last digit changed | Last-level sq. error of changed items |
|---|---|---|---|---|
| 3×128 | 44.8 % | 0.71 % | 44.5 % | +777 on a mean of 748 |
| 4×128 | 29.3 % | 0.71 % | 29.0 % | +662 on 658 |
| 5×256 | 14.6 % | 0.71 % | 14.6 % | +624 on 414 |
| 4×512 | 9.7 % | 0.71 % | 9.7 % | +717 on 256 |

All 8 tables end with only the 11 identical-input buckets. The loop runs all
20 rounds because those pairs can never be split.

**What is left in RQ-VAE.** Six RQ-VAE tables sit at the identical-input
floor or one pair above it. Three keep buckets of distinct items that the loop
did not split:

- 3×128: one bucket `(27)(89)(7)` of 26 items from 22 brands (a vacuum, a
  ladder lift, a microwave, micrometers, a water test kit, ...)
- 3×256: one 10-item bucket (dropper bottles, pH testers, calibration
  solutions) and smaller ones
- 5×256: 19 HATCHBOX ABS filaments (one brand; colour variants)

The loop splits any group whose rows of last-level distances differ.
Leaving distinct items together means the RQ-VAE encoder put them at
(nearly) the same latent point. This cannot be checked further, because the
RQ-VAE weights were not saved.

So RQ-VAE's last digit is an identity digit (semantic-mapping takeaways)
because the loop rewrites it, and RQ-VAE's collision rate measures the loop
plus the identical inputs, not the learned quantizer.

### H1b, RQ-KMeans 5×512 was balanced: declared rule failed; supported by A3

The declared rule expected exact capacities of 6 or 7 items per code. That
was a misreading of upstream's code: when none of an item's 32 best codes has
room, `sinkhorn_balance_level` assigns it to `argmin(remaining)`, the code
already most over capacity. The procedure therefore leaves one overflow code
per level. Neither the frozen table nor a faithful reimplementation is
exactly balanced.

**A3 (post hoc, job 293573)** compares fingerprints of three tables: the
frozen 5×512, the native refit, and upstream's `--uniform` mapping applied to
the native refit (D2):

| | Frozen 5×512 | `--uniform` of the native fit | Native fit |
|---|---|---|---|
| Codes exactly at capacity, per level | 467–502 of 512 | 478–495 | 8–73 |
| Largest code, per level | 21–81 items | 35–110 | 29–465 |
| Identical-input pairs split | 11 / 11 | 11 / 11 | 0 / 11 |
| Collision rate | 0.64 % | 0.39 % | 5.99 % |
| 10-NN first-digit agreement | 0.215 | 0.216 | 0.455 |
| Repeat rate at level 1 | 0.003 | 0.100 | 0.137 |

The frozen table matches the balanced mapping on every fingerprint except the
repeat rate. Its base fit therefore did not reuse one seed across levels, and
it differs from the native fit (first-digit ARI 0.21). Balancing alone moves
the native 5×512 fit from 6.0 % to 0.39 % collisions, and the native 3×128 fit
from 44.8 % to 12.6 %. In both cases it halves first-digit neighbour agreement
(0.72 → 0.43 and 0.46 → 0.22).

Consequence: the RQ-KMeans width-512 depth series mixes "one more digit" with
"balanced instead of nearest-centroid". The 5×512 point in the takeaways'
Fig 8 should not be read as the effect of a fifth digit.

### H2, near-duplicates: supported as the floor; the declared "all tables" enrichment failed in one table

| Nested RQ-KMeans, near-duplicate share of collided items | depth 3 | depth 4 | depth 5 |
|---|---|---|---|
| width 128 | 0.46 | 0.63 | 0.73 |
| width 256 | 0.60 | 0.73 | **0.81** |
| width 512 | 0.75 | 0.85 | (balanced table) |

- **Enrichment.** Collided items have a near-duplicate bucket-mate more often
  than items in multi-item depth-(D − 1) groups have one in their group. This
  holds in 26 of 27 tables. The exception is the balanced RQ-KMeans 5×512
  (0.30 vs 0.33), whose balancing scatters near-duplicates.
- **Closeness.** A collided item's closest bucket-mate is within the closest
  0.1 % of all 4.8 M catalogue pairs for 69–100 % of collided items in
  RQ-KMeans, and 44–87 % in MQ.
- **Share of near-duplicate pairs that collide.** 65 % at RQ-KMeans 3×128,
  33 % at 5×256, and 0.9 % in deep RQ-VAE (only the identical pairs).
- **Deep MQ** fits reach 0.59–0.68.

At shallow fits about half the collided items have no near-duplicate partner
(54 % at RQ-KMeans 3×128). These are mostly same-function items from
different brands, e.g. three brands' anti-static wrist straps, or an
oscilloscope, a waveform generator and a USB scope. H3 is about these.

### H3, shared residual codebooks (RQ-KMeans): (a) supported, (b) falsified, (c) partly supported

**(a) Concentration within a parent: supported.** At every level ≥ 1 of every
RQ-KMeans table, a parent's items share child codes far more than the level's
global code frequencies imply. The merge ratio is 1.3–16.5 (median 8.0)
in the nested tables. It is also 1.8–35 in MQ, so it mostly restates that
codes follow geometry. A2 shows what the geometry is:

| A2, native refits | level 1 | level 2 | level 3 | level 4 |
|---|---|---|---|---|
| Hub code's share of items (128 / 256 / 512) | 16 / 17 / 13.5 % | 12 / 15 / 14 % | 14 / 13 / 12 % | 9 / 11 / 15 % |
| Hub centroid's norm rank (0 = smallest) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 1 | 0 / 0 / 0 |

- At every level after the first, the most-used code is the centroid nearest
  the origin. Its items have small residuals, e.g. 1.8 against 21.8 at 512,
  level 2.
- Items whose residual is already small get this code at each later level
  and are not refined further.
- Parents with a smaller RMS radius have more collided items: Spearman
  −0.41 to −0.68 across the 8 nested tables.

**(b) The shared seed causes collisions: falsified.**

- The signature is real: the repeat rate P(digit l = digit l − 1) is
  0.032–0.075 in the 8 nested RQ-KMeans tables, 4–38 times chance. It is
  ≈ 1/K in RQ-VAE, MQ and the balanced 5×512.
- Giving each level its own seed removes the signature (repeat rate ≈ 1/K),
  but it **raises** the collision rate wherever the change exceeds the seed
  range:
  - +13.0 pp at 3×128, +4.2 at 4×128
  - +4.5 at 3×256, +3.2 at 4×256
  - +2.2 at 3×512
  - no change at depth 5
- Per-level seeds also lower the variance explained by the first two digits
  (R² 0.778 → 0.761 at width 128).

A likely reason, not tested here: with the shared seed, each parent's own
seed item initialises one centroid at the next level, so every parent starts
with a nearby code.

**(c) Scale mismatch: partly supported.** Dividing each parent's residuals by
the parent's RMS norm before fitting the next level:

- **Depth 5:** lowers the collision rate at every width, with seed ranges
  disjoint from native: −3.0 / −2.9 / −1.0 pp at widths 128 / 256 / 512.
- **3×512:** lowers it by 1.6 pp, ranges disjoint.
- **3×128:** **raises** it by 5.7 pp, ranges disjoint.
- **The other cells:** lower it within the seed range.
- **With per-level seeds:** the combination reaches 3.5 % at 5×512 (native
  6.2 %), and is 3.7 and 4.3 pp below native at 4×128 and 5×128.
- **Cost:** R² of the first two digits falls from 0.78 to 0.74 at width 128.

The declared rule (lower in every cell) fails. The 10-NN cost criterion is
uninformative here, because scale normalisation leaves the first digit
unchanged.

### H4, anisotropy: not supported

| Δ collision rate vs baseline, depth 3 / 5 | width 128 | width 256 | width 512 |
|---|---|---|---|
| L2-normalised − native | −0.8 / −0.4 pp | +0.5 / −0.5 | −0.2 / −0.3 |
| PCA-256 − native | −4.4 / −1.6 | −1.6 / −0.9 | +0.3 / +0.2 |
| PCA-256-white − PCA-256 (declared) | **+14.4 / +2.6** | +2.5 / −0.0 | −1.2 / −0.7 |
| PCA-64-white − PCA-256 (exploratory) | −9.3 / −5.4 | −6.1 / −2.7 | −2.9 / −0.9 |

- **L2.** Normalising changes nothing, since norms vary by only 4 %.
- **PCA-256-white.** Whitening the top 256 PCs raises collisions at width 128
  and lowers neighbour agreement (0.72 → 0.62).
- **PCA-64-white.** Whitening only the top 64 lowers collisions at every
  width. Its neighbour agreement is 0.70 at 128, against 0.72 for PCA-256.

No single direction holds, so anisotropy is not a consistent cause.

### H5, MQ redundancy: supported

| MQ | 3×128 | 3×256 | 3×512 | 4×128 | 4×256 | 4×512 | 5×* |
|---|---|---|---|---|---|---|---|
| Collision rate | 54.5 % | 42.8 % | 30.7 % | 37.0 % | 29.1 % | 22.7 % | 18–26 % |
| Independent-digit null (N1) | 7.4 % | 1.0 % | 0.13 % | 0.53 % | 0.06 % | 0.00 % | 0.00 % |
| Mean pairwise NMI between digits | 0.50 | 0.60 | 0.68 | 0.50 | 0.59 | 0.70 | 0.51–0.70 |

Every MQ table is ≥ 7 times its N1 rate (declared rule: ≥ 2).

Part E (exploratory) splits the top 2,400 PCs into D blocks in three ways and
gives each block its own k-means. Collision rate follows the dependence
between digits:

| Allocation | Mean NMI between digits | Collision rate (9 sizes) |
|---|---|---|
| Random rotation | 0.86–0.93 | 72–92 % |
| PCs dealt round-robin | 0.59–0.84 | 45–76 % |
| Contiguous PCs | 0.18–0.44 | 4–88 %, unstable across seeds |

Random subspaces of this embedding all carry the same cluster structure, so
parallel codebooks on them make nearly the same partition. The archived MQ
tables (NMI 0.50–0.70) sit between the round-robin and contiguous
allocations. How they were built is unknown.

## What this means for other results

- **Comparing collision rates across families compares post-processing.**
  RQ-VAE's rate measures the dedup loop, and RQ-KMeans 5×512's measures
  balancing. Any analysis that relates a family's collision rate to
  recommendation quality inherits this. For example:
  - CCE's ranking reversal (RQ-KMeans 3×128 → RQ-VAE 5×512)
  - the Snapchat "uniqueness plateau" comparison in the takeaways
- **RQ-VAE's last digit and RQ-KMeans 5×512's digits are partly arbitrary.**
  RQ-VAE's last digit was rewritten for 0–44 % of items by an assignment
  that ignores semantics. 5×512's first digit agrees with neighbours half as
  often as a nearest-centroid fit.
- **Near-duplicate collisions are the ones a recommender can exploit.** Users
  buy variants together (retrospective exp6: "recent" means same day). When
  two variants share a SID, copying the history item's SID scores an exact
  SID hit on the other variant. The intervention follow-up measures how much
  of exp6's copy benefit is this.

## What this does not establish

- RQ-VAE was not re-run; its latent collapse for the three unsplit buckets is
  inferred from the loop's behaviour.
- MQ's construction is unknown; Part E is a counterfactual family.
- Nothing here measures what a collision type costs a recommender.
