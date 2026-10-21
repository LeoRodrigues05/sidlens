# The final (v3) SID maps: completed 2026-10-20

The map analyses of the paper (RQ1; final plan M1, M2, M5) were rerun on the
SID tables of the final training run: 27 maps per catalog for Industrial and
Office, and their 39 ZCR twins. Several analyses were added that the new
quantizer weights make possible. CPU only; no recommender is run.

Main results:

1. **Most of the draft's RQ1 numbers replicate on v3 and on Office.**
   - RQ-KMeans digit 1 alone explains 53 % of the embedding variance in both
     catalogs (4×128). Each MQ digit explains about the same share as the
     others: the largest is 1.01–1.11 times the smallest.
   - In the 4×128 maps, three shared digits put 95–99 % of grouped items in
     one category in the residual maps (Industrial level-1 categories, Office
     level-2), against 79–80 % in MQ.
   - Near-duplicate products make up most collisions in the deepest
     RQ-KMeans maps.
2. **In RQ-VAE, digit 2 carries the most meaning, in the old maps and in v3.**
   - Its first digit is coarse: it uses 3–45 % of the first codebook on
     Industrial.
   - Given digit 1, digit 2 explains the most variance: 43–59 % on
     Industrial, more than digit 1 in all 9 maps (Office: 5 of 9). The old
     maps give the same range, 43–59 %, in 9 of 9; the draft reports digits
     alone only, which hides it.
   - Digits 4–5 add at most 1.5 % (Industrial) and 2.2 % (Office).
   - The draft's "92.5 %" for RQ-VAE 4×128 is the share of items whose full
     SID any three of their four digits determine (the paper plan calls it a
     "unique 3-digit prefix", which is a different statistic). It is 89.98 %
     on v3 Industrial and 87.03 % on Office; the archived 92.54 % is
     reproduced exactly on the old map.
3. **RQ-VAE's low collision rate comes from post-processing; this is now
   shown directly.**
   - The checkpoints' own nearest codes collide 5.5–19.8 % of items on
     Industrial and 7.3–29.9 % on Office. The published tables collide
     0.71–2.54 % and 0.66–1.37 %.
   - Re-running upstream's last-digit reassignment from the nearest codes
     reproduces all 9 Industrial tables item for item. On Office it
     reproduces 6 of 9 exactly; the other 3 differ on one item each.
   - What the loop leaves is explained by the encoder (post hoc). The 57
     distinct Industrial items that RQ-VAE 5×512 keeps in one SID (a vacuum, a
     ladder lift, tool steel, …) all map to the same latent point: every unit
     of the encoder's first hidden layer is zero for them. Exp3 could only
     infer this kind of collapse for the old 26-item bucket. Office's
     unsplit buckets are 26 small groups (3–5 items) with close but distinct
     latents.
4. **MQ's digits depend on each other because its encoders never train.**
   - In all 18 MQ models, every encoder weight is still at its random
     initialisation (`rq/mq.py` runs them under `no_grad`). Each digit
     therefore quantizes a different random projection of the same item.
   - Mean pairwise AMI between digits is 0.29–0.35 for MQ against
     0.04–0.15 for the residual maps.
   - The plug-in NMI the draft quotes (MQ 0.50–0.70) overlaps with
     RQ-KMeans (0.28–0.59 on Industrial), so it should not carry the claim.
5. **Some old collision results do not carry over.**
   - RQ-KMeans 5×512 is no longer a balanced table (6.31 % collided on
     Industrial). No v3 map gives two identical items different SIDs.
   - The reassignment loop does not bring every RQ-KMeans and MQ table to the
     floor on Office: RQ-KMeans 3×128 stays at 3.21 % and MQ 3×128 at 9.67 %.
   - The shared k-means seed lowers collisions on Industrial at depth 3,
     width 128 and 256, and on Office only at width 512.
6. **Exposure (new, exploratory).**
   - Under the RQ-KMeans maps, 4.6–10.8 % of Industrial next-item test
     targets share their SID with a different item in the user's history
     (Office 1.9–5.3 %); repeats are 5.0 % (5.6 %). ZCR twins have none.
   - Collided items are bought more often than their share of the catalogue:
     RQ-VAE collides 0.7–2.5 % of Industrial items but 4.3–6.0 % of
     interactions.

The protocol ([protocol.md](protocol.md)) was declared before any estimand
was computed. One amendment, A1, was made after an Industrial smoke run and is
labelled post hoc wherever it applies: the v3 RQ-KMeans tables cannot be
reproduced by refitting on this cluster.

| Run | Job | Output under `$SIDLENS_WORK/derived/` |
|---|---|---|
| Full run, both catalogs | 327442 (cn-03, 128 CPUs, 20 min) | `structure/exp4_final_maps/327442/{industrial,office,summary}/` |
| Explorer, Industrial v3 | login node | `semantic_mapping/v3-industrial-20261020/` |
| Explorer, Office v3 | 327460 | `semantic_mapping/327460/` |

Two reports, both built from the run's CSVs:

- [`sidlens_final_maps_paper.html`](../../../docs/results/final_maps/sidlens_final_maps_paper.html):
  the paper-facing page. Every map result the draft displays, its value on the
  old maps and on v3, what to change, the figures and the collision-cause
  table, and an old-against-v3 comparison.
- [`sidlens_final_maps.html`](../../../docs/results/final_maps/sidlens_final_maps.html):
  the full analysis, with every part and expectation.

Every number in this file is in `summary/facts.md` or
`summary/compare/claims.csv`, next to the table it comes from. The old values
in the comparison are recomputed from the files the draft's numbers came from
(`compare_old.py`): the paper data package of 2026-09-28, checked against its
manifest, and the structure exp3 and old ZCR outputs, copied from `/l` with
their sha256.

## Inputs and validation (Part 0)

- **Download.** `WFRaain/onediffrec-sid-codebooks` at revision `9733eda7`
  (790 files, 7.7 GB, all sizes checked) is in
  `external/sid_codebooks/WFRaain-onediffrec-sid-codebooks-9733eda/`.
  Both embedding files have the sha256 recorded in the ZCR manifests, so the
  maps were fitted to the frozen embeddings.
- **Office labels (new).** These are built from the Amazon-2018 release
  `meta_Office_Products.json.gz` and join all 17,696 items. The release
  HTML-escapes some breadcrumbs (`Office &amp; School Supplies`), which split
  categories in two, so `build_labels.py` now unescapes them; Industrial has
  no escaped values. Office's level-1 breadcrumb has 3 classes, so its
  primary field is level 2 (42 classes) and its secondary field level 3
  (215).
- **Hashes and ZCR twins.**
  - Every native table matches the sha256 in its `done.json`.
  - All 39 ZCR twins keep the native prefixes and have no shared SID. Each
    changes exactly the recorded minimum number of last digits.
- **Nesting.** The RQ-KMeans 3- and 4-digit tables are the prefixes of the
  5-digit table for every item, at every width, in both catalogs. This
  settles the draft's `\pending{check on the v3 tables}`. RQ-VAE and MQ
  depths are separate fits.
- **RQ-VAE checkpoints.** Each checkpoint is loaded with an allowlist
  unpickler. Its encoder and codebooks give digits 1 to D−1 of every table
  for every item, and the exported last-level residuals to within 1.5e-5.
  The checkpoint's recorded `best_collision_rate` equals our pre-reassignment
  rate (old definition), so upstream picked each RQ-VAE checkpoint by its
  collision rate before reassignment.
- **MQ models.** The eval-mode encoders reproduce every digit of all 18
  tables, with one exception: one Office item in MQ 5×512, whose two nearest
  codewords differ in distance by a relative 1.8e-7 (a float32 tie).
- **RQ-KMeans refits (amendment A1).**
  - On this cluster (AMD EPYC 7402), neither per-level `faiss.Kmeans` nor
    faiss's own `ResidualQuantizer` with the builder's settings reproduces
    the v3 tables. On Industrial 3×128 both get digits 1–2 exactly and
    digit 3 for about a third of items, at 1, 8 and 96 threads. At 96 threads
    the refit equals the old frozen table.
  - So the v3 tables were fitted on hardware whose floating-point results
    differ from this cluster's from level 3 on.
  - The published codebooks (`codebooks.npy`) reproduce all 18 tables by
    nearest centroid. Part D and the hub-code analysis use them, and Part C
    is a counterfactual on this cluster's refits.

## Expectations

| | Industrial | Office | Where it fails |
|---|---|---|---|
| E1 residual digit 1 > any later digit alone; MQ shares within 1.5× | failed | held | RQ-VAE 3×512, 4×512, 5×512 (Industrial): digit 2 alone explains more. The rule was miscalibrated: the old maps fail it in 5 of 9 RQ-VAE maps (`compare/claims.csv`, e1_rule) |
| E2 Industrial cat_l1 purity ≥ 0.95 at length 3 (residual, D ≥ 4); MQ < RQ-KMeans | failed | not declared | RQ-VAE 5×512: 0.907, with 26 % of items still grouped |
| E3 RQ-VAE uses ≤ 50 % of its first codebook in ≥ 6/9 maps | held (9/9) | held (7/9) | Office RQ-VAE 3×128 uses 100 %, 4×128 58 % |
| E4 ≥ 95 % of items alone after RQ-VAE's last digit | held | held | |
| E5 RQ-VAE before > published; published within 0.5 pp of the floor in ≥ 7/9 | held (8/9) | held (8/9) | Industrial 5×512 (2.54 %), Office 3×128 (1.37 %) |
| E6 the loop brings every RQ-KMeans and MQ map within 0.5 pp of the floor | failed | failed | Industrial MQ 4×128 (1.35 %). Office: RQ-KMeans 3×128 (3.21 %); MQ 3×128 (9.67 %), 3×256 (1.55 %), 4×128 (3.41 %), 5×128 (2.14 %) |
| E7 nested RQ-KMeans: near-duplicate share rises with depth, > 0.5 at 5×W | held | held | |
| E8 MQ collision rate ≥ 2× its independent-digit null | held | held | |
| E9 hub code has norm rank ≤ 1 at every level ≥ 2 | held | failed | Office W = 128, level 3: rank 7 |
| E10 per-level seed raises the depth-3 rate at every width | failed | failed | Industrial W = 512; Office W = 128, 256 (seed ranges overlap) |
| E11 MQ encoders and `pos` at initialisation | held | held | |

## What each digit encodes (M1)

Variance of the item embedding explained by each digit, 4×128, digits 1–4:

| | Alone, Industrial | Alone, Office | Given earlier digits, Industrial | Given earlier digits, Office |
|---|---|---|---|---|
| RQ-KMeans | 53, 16, 15, 12 % | 53, 19, 13, 11 % | 52.8, 25.7, 17.3, 2.8 % | 52.7, 21.2, 18.2, 5.6 % |
| RQ-VAE | 29, 20, 12, 9 % | 42, 7, 3, 3 % | 29.2, 48.0, 21.9, 0.9 % | 41.5, 38.2, 18.3, 1.9 % |
| MQ | 38, 38, 38, 38 % | 36, 38, 35, 36 % | 37.7, 31.2, 17.0, 7.7 % | 35.6, 23.4, 19.1, 10.6 % |

- **Across all nine maps of each family:**
  - RQ-KMeans digit 1 alone explains 53–69 % (Industrial) and 53–63 %
    (Office); later digits 9–26 % and 9–21 %.
  - Each MQ digit explains 36–57 % and 34–47 %.
  - RQ-VAE digit 1 explains 25–37 % and 34–46 %. Given digit 1, digit 2
    explains 43–59 % and 38–46 % (old maps: 43–59 %).
- **Codes in use.** RQ-KMeans uses every code, and MQ 85–99 % (Industrial)
  and 98–100 % (Office). RQ-VAE's first digit uses 3–45 % on Industrial and
  4–100 % on Office (7 of 9 at most half).
- **Category purity at prefix length 3** (4×128), among items that still
  share a group, with the shuffled-label null in brackets:

  | | Industrial (level-1) | Office (level-2) |
  |---|---|---|
  | RQ-KMeans | 97.0 % (44.8 %) | 95.3 % (40.8 %) |
  | RQ-VAE | 98.6 % (49.9 %) | 96.7 % (46.4 %) |
  | MQ | 79.2 % (37.9 %) | 79.8 % (35.2 %) |

  Over all residual maps with D ≥ 4 the range is 90.7–99.3 % on Industrial
  and 95.3–97.2 % on Office; over MQ maps, 79.0–94.6 % and 78.4–90.8 %.
- **Category information each digit adds** (conditional AMI, 4×128, primary
  field; † marks too few items per parent):

  | | Industrial | Office |
  |---|---|---|
  | RQ-KMeans | 0.52, 0.14, 0.24, 0.38† | 0.56, 0.13, 0.13, 0.07 |
  | RQ-VAE | 0.50, 0.19, 0.25†, n/a | 0.50, 0.16, 0.15, 0.00† |
  | MQ | 0.32, 0.19, 0.14, 0.10 | 0.36, 0.20, 0.13, 0.15 |

- **Refinement index at full length**, 5×128: 0.96 (RQ-KMeans), 1.00
  (RQ-VAE) and 0.82 (MQ) on Industrial; 0.93, 1.00 and 0.75 on Office.
- **Items alone after the last digit:**
  - RQ-VAE: 97.5–99.3 % (Industrial) and 98.6–99.3 % (Office).
  - RQ-KMeans: 55.8–93.7 % and 42.3–87.4 %.
  - MQ: 44.1–81.0 % and 30.4–84.4 %.
- **RQ-VAE 4×128, two identity measures.** Items whose full SID any three of
  their four digits determine (the draft's 92.5 % on the old map, reproduced
  as 92.54 %): 89.98 % on Industrial and 87.03 % on Office. Items alone at
  prefix length 3: 77.1 % and 65.8 % (old map 79.6 %).

## Why items share a SID (M2)

- **Collision rates** (share of items in a shared SID):

  | | Industrial | Office |
  |---|---|---|
  | RQ-KMeans | 6.3–44.2 % | 12.6–57.7 % |
  | RQ-VAE | 0.7–2.5 % | 0.7–1.4 % |
  | MQ | 19.0–55.9 % | 15.6–69.6 % |

  At 3×128: RQ-KMeans 44.2 % / RQ-VAE 0.7 % / MQ 55.9 % on Industrial, and
  57.7 % / 1.4 % / 69.6 % on Office.
- **Floor.** 22 Industrial items (11 groups, 0.71 %) and 117 Office items (57
  groups, 0.66 %) have identical embeddings. All 27 maps of each catalog give
  every identical group one SID.
- **Post-processing, measured directly on RQ-VAE.** The loop rewrote the last
  digit of 6–20 % (Industrial) and 7–30 % (Office) of items. Per map, before
  reassignment → published:

  | | 3×128 | 3×256 | 3×512 | 4×128 | 4×256 | 4×512 | 5×128 | 5×256 | 5×512 |
  |---|---|---|---|---|---|---|---|---|---|
  | Industrial | 19.8 → 0.71 | 17.1 → 0.77 | 15.2 → 0.71 | 11.9 → 0.84 | 9.5 → 0.71 | 11.0 → 0.71 | 10.2 → 0.71 | 5.5 → 0.71 | 6.9 → 2.54 |
  | Office | 29.9 → 1.37 | 27.3 → 0.98 | 25.6 → 0.90 | 16.9 → 0.72 | 15.8 → 0.70 | 13.2 → 0.66 | 9.8 → 0.68 | 9.3 → 0.69 | 7.3 → 0.66 |

- **The same loop on the other families** (exp3's reimplementation, from each
  table's own last-level geometry):
  - RQ-KMeans goes from 6.3–44.2 % to 0.71–0.84 % on Industrial and from
    12.6–57.7 % to 0.66–3.21 % on Office.
  - MQ goes from 19.0–55.9 % to 0.71–1.35 % and from 15.6–69.6 % to
    0.62–9.67 %.
  - In Office's MQ maps at W = 128, one prefix group holds 249–530 items, more
    than the 128 codes of the last digit can separate. ZCR is infeasible on
    the same three maps. The other maps left above the floor are feasible;
    the loop stops at its 20-round limit there.
  - Five Office MQ maps end slightly below the floor (as low as 0.62 %). MQ's
    exported vectors for identical inputs differ by up to 2.7e-6 (GPU
    arithmetic), and the loop's sharp Sinkhorn step splits a few of those
    groups. RQ-VAE's exported vectors are exactly equal within identical
    groups.
  - Each item the loop moves gains 1.0–3.0 (Industrial) and 0.7–1.7 (Office)
    times the table's mean last-level squared distance (RQ-KMeans).
- **Near-duplicates.**
  - Share of collided items with a near-duplicate bucket-mate in the nested
    RQ-KMeans maps, depths 3 → 4 → 5:
    - Industrial: 46 → 62 → 71 % (W = 128), 58 → 74 → 79 % (256),
      75 → 90 → 94 % (512).
    - Office: 39 → 54 → 68 %, 49 → 65 → 76 %, 59 → 77 → 84 %.
  - A collided item's closest bucket-mate is among the closest 0.1 % of
    catalogue pairs for 70–100 % (Industrial) and 75–100 % (Office) of
    collided items.
  - Near-duplicate pairs: 1,291 covering 900 Industrial items (29.0 %), and
    8,874 covering 7,057 Office items (39.9 %).
- **Hub codes (RQ-KMeans).**
  - At levels 2–5 the most-used code is the centroid nearest the origin at
    12/12 levels (Industrial) and 10/12 (Office). It holds 9–21 % and
    3–10 % of items.
  - Parents with a smaller radius collide more: Spearman −0.66 to −0.40
    (Industrial) and −0.60 to −0.16 (Office).
  - The reused k-means seed shows: digit l equals digit l−1 for 0.032–0.076
    and 0.022–0.046 of items, against 1/K = 0.002–0.008.
- **MQ.**
  - Per-map mean pairwise AMI is 0.29–0.33 (Industrial) and 0.29–0.35
    (Office), against 0.06–0.11 / 0.08–0.15 for RQ-KMeans and
    0.06–0.13 / 0.04–0.09 for RQ-VAE.
  - The observed collision rate is at least 8× (Industrial) and 5× (Office)
    the independent-digit null. At 3×128: 55.9 % observed against 6.8 %
    with independent digits (Industrial), 69.6 % against 14.2 % (Office).
  - All 18 MQ models' encoder weights and biases lie within PyTorch's
    default ±1/√fan-in range, with the spread of a uniform draw (std ratio
    0.88–1.09). BatchNorm scales are exactly 1 and shifts exactly 0, and
    `pos` is within its ±1/K initial range. The encoders never train.
- **Refits (counterfactual on this cluster, A1)**, condition minus baseline
  in points, depth 3 / depth 5, over 5 seeds. "(o)" marks overlapping seed
  ranges, which do not count.

  | Condition | Industrial W = 128 / 256 / 512 | Office W = 128 / 256 / 512 |
  |---|---|---|
  | Per-level seed | +12.9 / +1.5 (o); +4.8 / +0.6 (o); +2.4 (o) / −0.7 (o) | +0.9 (o) / +1.0 (o); +3.1 (o) / −0.0 (o); +3.2 / −1.0 (o) |
  | Scale-normalised residuals | +5.4 / −2.0 (o); −0.5 (o) / −2.9; −1.4 (o) / −1.6 | −3.4 (o) / +0.8 (o); −0.2 (o) / −1.5; −0.2 (o) / −1.4 |
  | L2-normalised | all within seed range | all within seed range |
  | Top-256 PCs whitened | +13.3 / +3.6; +3.1 / +0.1 (o); −1.1 (o) / −0.7 (o) | +14.9 / +5.0; +7.4 / +2.6; +3.4 / +1.7 |
  | Top-64 PCs whitened | −9.4 / −5.1; −6.2 / −3.0; −2.7 / −1.0 | −14.5 / −5.2; −10.0 / −3.3; −6.4 / −2.5 |

  The native refit is close to the published rates. For example, Industrial
  3×128 refits at 46.3 % against 44.2 % published, and Office 3×128 at
  57.8 % against 57.7 %.

## ZCR twins

- There are 21 Industrial and 18 Office twins.
  - The 6 RQ-VAE maps per catalog without a twin are below the 5 % threshold.
  - Office MQ 3×128, 4×128 and 5×128 are infeasible.
- **Last digits changed:** RQ-KMeans 4–29 % (Industrial) and 7–41 %
  (Office); MQ 12–40 % and 10–41 %; RQ-VAE 0.4 % and 0.4–0.5 %.
- **Last-level squared distance:** RQ-KMeans +0.7–5.1 % and +0.7–5.2 %; MQ
  +4.8–13.2 % and +2.2–11.8 %. The RQ-VAE twins lower it by 0.2–0.4 %: they
  move the few items the loop left colliding to nearer free codes.
- RQ-KMeans 3×128 changes 29 % of last digits on Industrial (896 items; the
  draft says 910 for the old map) and 41 % on Office.

## Naming and co-purchase (M5)

- **Naming.** Median held-out AUC of a digit-1 cell's own title terms against
  its siblings, ranges over 9 maps: RQ-KMeans 0.96–0.99 in both catalogs;
  RQ-VAE 0.76–0.83 (Industrial) and 0.85–0.91 (Office); MQ 0.79–0.88 and
  0.84–0.91.
- **Co-purchase.** Lift of items that share digit 1 (`also_buy`): RQ-KMeans
  23–52× and 30–76×; RQ-VAE 7–17× and 8–26×; MQ 5–16× and 5–15×.
- The offline explorer was rebuilt on the v3 tables for both catalogs.
  Industrial has 83,835 assignments (`explorer.html` 2.2 MB); Office has
  477,792 (12.2 MB).
  - `semantic_mapping build` gained `--source v3 --catalog`. The default
    frozen build is unchanged byte for byte, apart from its timestamp.
  - Office's coherence uses level-2 and level-3 categories.

## Exploratory parts (X, R)

- **Collision exposure.** Next-item test target = each user's last
  interaction, with up to 50 earlier items as history (frozen sequences).
  - Repeats are 5.0 % (Industrial) and 5.6 % (Office) of targets.
  - SID partners, targets whose SID a different history item carries:
    RQ-KMeans 4.6–10.8 % and 1.9–5.3 %; MQ 7.7–10.0 % and 1.9–6.0 %; RQ-VAE
    3.6–3.9 % and 0.6 %.
  - Under RQ-KMeans 3×128 they are 10.8 % (Industrial) and 5.3 % (Office).
    The ZCR twin has none, and its variant class rises from 2.1 % to 10.6 %
    (Industrial).
  - Interactions on collided items: RQ-VAE 4.3–6.0 % (Industrial) and
    1.4–2.0 % (Office); RQ-KMeans 11.1–51.6 % and 14.8–64.4 %; MQ
    26.6–61.3 % and 19.7–74.8 %.
- **RQ-VAE in its own latent.**
  - 71–496 (Industrial) and 0–490 (Office) first-level codewords are
    nobody's nearest code.
  - Unused codewords have median norm 5.2–5.6, against 13.6–36.6 for used
    ones: they sit near the latent origin, away from the data. The training
    history is not saved, so "initialised and never chosen again" is a
    reading, not a test.
  - Digit 1 explains 71–84 % (Industrial) and 68–76 % (Office) of the
    latent's variance, against 25–37 % and 34–46 % of the input embedding's.
    RQ-VAE's first digit is a coarse partition of its own latent, which is not
    the input space where every other measure is taken.
- **v3 against the old maps (Industrial).**
  - RQ-KMeans digit 1 is unchanged for every item in 8 of 9 maps (the old
    5×512 was a separate balanced fit). Digit 2 is unchanged for 100 %, 89 %
    and 78 % of items at widths 128, 256 and 512.
  - The RQ-KMeans 3×128 full SID is unchanged for 33 % of items.
  - RQ-VAE and MQ are new fits: first-digit ARI against the old map is 0.40
    for RQ-VAE 4×128 and 0.07–0.12 for MQ.
  - Mechanistic results on the old RQ-KMeans 3×128 used the same first two
    digits. Results on the old RQ-VAE 4×128 and MQ maps do not transfer item
    by item.

## What changes in the paper's RQ1 section

Recomputed claim by claim in `summary/claims.csv`; the HTML report shows
the table.

| Draft (old maps, Industrial) | v3 | Action |
|---|---|---|
| Digit 1 alone: 53 % RQ-KMeans, 34 % RQ-VAE; later digits 8–18 % (4×128) | RQ-KMeans 52.8 % (Office 52.7 %); RQ-VAE 29.2 % (41.5 %); later residual digits 9–20 % (3–19 %) | Keep for RQ-KMeans. For RQ-VAE add that digit 2, given digit 1, explains the most (43–59 %, Industrial; the old maps give the same range) |
| Every MQ digit explains 36–38 % (4×128) | 38 % each (Office 35–38 %) | Keep |
| Refinement index after five digits: MQ 0.80, RQ-KMeans 0.95 (5×128) | 0.82 / 0.96 (Office 0.75 / 0.93) | Update the numbers |
| Purity after three digits: 97–99 % residual, 81 % MQ; null 39–50 % | 97.0 / 98.6 / 79.2 %; null 38–50 % (Office, level-2: 95.3 / 96.7 / 79.8 %; null 35–46 %) | Keep; add Office |
| RQ-VAE uses 2–31 % of its first codebook in 8 of 9 maps | 3–45 % in 9 of 9 (Office 4–100 %, 7 of 9 at ≤ 50 %) | Update |
| After four digits 99 % of RQ-VAE items have their own SID | 99.2 % (Office 99.3 %) | Keep |
| 44.8 % (RQ-KMeans) and 54.5 % (MQ) share a SID at 3×128 | 44.2 % / 55.9 % (Office 57.7 % / 69.6 %) | Update |
| For 92.5 % of RQ-VAE 4×128 items any three digits determine the full SID (the plan words it as a unique 3-digit prefix) | 89.98 % (Office 87.03 %); archived 92.54 % reproduced on the old map | Update; fix the plan's wording |
| 11 identical pairs (0.71 %); every map but one collides them | 11 groups, 22 items (0.71 %); all 27 maps (Office: 57 groups, 117 items, 0.66 %, all 27) | Drop "but one" |
| 73–85 % of collided items in the deepest RQ-KMeans fits are near-duplicates | 71 / 79 / 94 % at 5×128 / 256 / 512 (Office 68 / 76 / 84 %) | Update; 5×512 now counts |
| The reassignment takes RQ-KMeans from 10–45 % to the 0.71 % floor, changing up to 44 % of last digits | Industrial 6.3–44.2 % → 0.71–0.84 %, up to 44 %; Office 12.6–57.7 % → 0.66–3.21 % | Qualify for Office |
| RQ-VAE's rate comes from the reassignment (inferred) | Shown directly: 5.5–19.8 % → 0.71–2.54 %; the loop reproduces the published tables | Strengthen; cite the direct test |
| Near-origin hub code holds 9–17 % of items; Spearman −0.41 to −0.68 | 9–21 %, −0.66 to −0.40 (Office 3–10 %, −0.60 to −0.16) | Update |
| MQ digits dependent, NMI 0.50–0.70 | AMI 0.29–0.35 vs 0.04–0.15 for residual maps; encoders untrained | Replace NMI by AMI; add the cause |
| MQ 3×128: 54.5 % observed vs 7.4 % with independent digits | 55.9 % vs 6.8 % (Office 69.6 % vs 14.2 %) | Update |
| RQ-KMeans 5×512 is Sinkhorn-balanced (0.64 %) | Not balanced, 6.31 % (Office 12.60 %) | Drop the 5×512 row of the collision table |
| Uniform codes would give at most 0.15 % at 3×128 | 0.15 % (Office 0.84 %) | Keep; add Office |
| The reused k-means seed lowers collisions (+13.0 points at 3×128 without it) | +12.9 points on Industrial (refit on this cluster); not significant on Office at 128 or 256 | Qualify: catalog-dependent |
| ZCR changes 910 last digits (29 %) at RQ-KMeans 3×128 | 896 (29 %), +5.1 % squared distance (Office 41 %, +5.2 %) | Update |

Figures for the paper are `summary/figures/fig_v3_map_digits` and
`fig_v3_map_identity`, the old two map figures redrawn with one row per
catalog. Two new figures are `fig_v3_post_processing`, the RQ-VAE before/after
test, and `fig_v3_exposure`. Each comes as PDF and PNG with a CSV twin.

## What this does not establish

- No model is run. What a collision costs a trained recommender needs the
  final run's predictions (M3, M4).
- Each map is one fit. Part C's ranges cover k-means initialisation only, on
  this cluster's refits, which match the published v3 RQ-KMeans tables only
  on digits 1–2.
- Category labels and the embedding both come from item text; `also_buy` is
  the one behavioural check.
- The exposure classes use the frozen interaction sequences as a stand-in
  for the v3 split, which holds out the same last interaction per user.
