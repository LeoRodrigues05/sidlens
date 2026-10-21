# Structure exp4: the final (v3) SID maps of both catalogs

Declared 2026-10-20, before any estimand below was computed. CPU only. The
experiment reads the published v3 SID tables, their ZCR twins, the exported
quantizer geometry and weights, the frozen item embeddings and metadata, and
the Amazon-2018 labels. No recommender is run.

## Why

The paper's map section (RQ1; paper plan of 2026-10-19, Sections 5.1–5.2;
final plan M1, M2, M5) rests on the 27 frozen Industrial maps. The final run
trains on new maps: 27 per catalog, Industrial and Office, published as
`WFRaain/onediffrec-sid-codebooks` (revision `9733eda7`, 2026-10-19). Every
map number must be recomputed on them. Three things changed that bear on the
old findings:

- MQ is a new implementation (`rq/mq.py`, TokenRec's code, commit 7ae3512).
- RQ-KMeans tables are unbalanced FAISS fits at 8 threads. The old balanced
  RQ-KMeans 5×512 table no longer exists.
- Quantizer weights and geometry now exist: the RQ-VAE checkpoints, the MQ
  models, the last-level residuals and centroids of every table, and all
  RQ-KMeans codebooks inside the ZCR bundles. The old analysis could not test
  RQ-VAE's post-processing directly or measure RQ-VAE in its own latent space.

## Seen before this protocol

- The dataset README's table of "items sharing a SID" for all 54 maps, the
  `done.json` of three Industrial tables, `tables.json` settings, one MQ
  `stats.json` (MQ 3×128 Industrial: 123/123/122 codes used), and one ZCR
  manifest and `stats.json` (RQ-KMeans 3×128 Industrial: 896 tails changed).
- A check of item alignment on five Industrial tables against the frozen
  tables: v3 RQ-KMeans 3×128 equals the frozen table on digits 1–2 for every
  item and on digit 3 for 33 % of items; v3 RQ-VAE 3×128 and 4×128, MQ 3×128
  and RQ-KMeans 5×512 agree with their frozen namesakes on 0–3 % of digits.
- The RQ-VAE 4×128 Industrial checkpoint's metadata: epoch 8749,
  `best_collision_rate` 0.0744.
- Office label coverage after the 2018 metadata join (100 % join; cat_l1 3
  classes, cat_l2 42, cat_l3 215 after HTML unescaping).
- Code (read, not run): `rq/rqkmeans_faiss.py`, `rq/rqvae.py`,
  `rq/generate_indices.py`, `rq/export_rqvae_geometry.py`, `rq/mq.py`,
  `rq/zcr.py` on OneDiffRec `leo-dev-changes` at 7ae3512. Facts from it that
  this protocol relies on:
  - RQ-VAE: nearest-code indices, then up to 20 rounds that re-encode every
    group of items sharing a full SID with a Sinkhorn assignment (ε = 0.003,
    50 iterations) of the last digit only.
  - MQ: each digit is the nearest codeword to `encoder_m(x + pos)`. The
    encoders and `pos` run under `torch.no_grad()`, so only the codebooks
    (k-means initialised at epochs 0, 10, 20) and the decoder are trained.
  - ZCR: within each group of items that share their first D−1 digits, an
    exact assignment that first minimises the number of changed last digits,
    then the squared distance to the new centroid. Prefixes are kept.

## Inputs

- Tables: `$SIDLENS_WORK/external/sid_codebooks/WFRaain-onediffrec-sid-codebooks-9733eda/`,
  all 790 files size-checked after download. Each table is checked against
  the sha256 in its `done.json` (native) or `manifest.json` (ZCR).
- Embeddings: `frozen/data/embeddings/{Industrial_and_Scientific,Office}.emb-qwen-td.npy`.
  Their sha256 equals the one in the ZCR manifests.
- Item ids: the tables are keyed by the frozen `item2id` index. Checked by
  the RQ-KMeans refit (below), by the frozen-table comparison and by
  `item_ids.json`.
- Metadata: frozen titles and brands (near-duplicates, naming); Amazon-2018
  labels in `external/labels/` (categories, `also_buy`).

## Definitions

Unchanged from structure exp3 and the semantic-mapping figures unless stated.

- **Collision rate** (primary): share of items whose full SID is shared.
  `1 − unique/N` is reported beside it.
- **Identical-input floor:** share of items whose stored float16 embedding
  equals another item's.
- **Near-duplicate pair:** one item among the other's 5 nearest neighbours,
  same folded non-empty brand, title-token Jaccard ≥ 0.5; identical inputs
  count.
- **Category fields.** Industrial: cat_l1 (25 classes) primary, cat_l2
  secondary. Office: cat_l1 has 3 classes, so the primary field is cat_l2
  (42 classes) and the secondary cat_l3 (215). Rule: the primary field is the
  coarsest breadcrumb level with at least 20 classes. Purity is reported for
  all three levels in both catalogs.
- **Purity** (fig4 of the semantic-mapping figures): share of labelled items
  in multi-item prefix groups whose label is their group's majority, against
  200 label permutations that keep the groups.
- **Variance explained by a digit alone:** 1 − R²_within of the partition by
  that digit only, in the raw 2,560-d embedding. **Given its prefix:**
  R²_within(prefix d−1) − R²_within(prefix d).
- **Refinement index:** 1 − R²_within / ((n − k)/(n − 1)).
- **Conditional AMI:** within-parent AMI pooled by size, against 200
  within-parent label permutations (structure exp1, `attribute.alignment`).
- **Pre-reassignment RQ-VAE table:** the nearest-code indices at every level,
  computed from the checkpoint (encoder and codebooks, float32). The exported
  geometry gives the last level independently.

## Parts and estimands

All parts run on both catalogs. Seeds: permutations 20261020 (M1 purity,
Part B nulls); k-means seeds {1234, 1, 2, 3, 4} as in exp3, where 1234 is
FAISS's default and the native seed.

### Part 0: validation (must pass before anything is interpreted)

1. Every table's sha256 matches its record; item ids are 0..N−1.
2. ZCR twins: prefixes equal the native table, no collisions, changed tails
   equal the recorded minimum.
3. Nesting: for each width, the RQ-KMeans 3- and 4-digit tables are the
   prefixes of the 5-digit table for every item. RQ-VAE and MQ are measured
   too. (The paper draft marks this "check on the v3 tables".)
4. RQ-KMeans refit: per-level `faiss.Kmeans(K, niter=10, seed=1234)` at 8
   OpenMP threads must reproduce every RQ-KMeans table item for item, and the
   ZCR-bundle codebooks must give the same nearest-centroid codes. If not,
   Parts C and D stop for that catalog.
5. RQ-VAE checkpoint: its nearest-code digits 1..D−1 must equal the table for
   every item, and its encoder output minus the chosen earlier centroids must
   equal the exported `last_stage_vectors` (max abs difference reported).
6. MQ model: eval-mode `encode` must reproduce every digit of the table.

### Part M1: what each digit encodes (54 native tables and every ZCR table)

Per table and digit: codes used, normalised entropy, variance explained alone
and given the prefix, marginal and conditional AMI (primary and secondary
field), digit-pair AMI and NMI. Per prefix length: groups, singleton share,
largest group, median RMS radius, R²_within, refinement index, 10-NN lift,
cumulative bits resolved, purity and its null (cat_l1, cat_l2, cat_l3), and
the refinement lost when each digit is deleted.

Expectations, taken from the current draft (Industrial, old maps):

- E1. In every RQ-KMeans and RQ-VAE table, digit 1 alone explains more
  variance than any later digit alone. In every MQ table, the largest
  single-digit share is less than 1.5 times the smallest.
- E2. Industrial cat_l1 purity at prefix length 3 is ≥ 0.95 in every
  residual table with D ≥ 4 and lower in MQ than in RQ-KMeans at the same
  (D, W). Office (cat_l2) is reported without an expectation.
- E3. RQ-VAE uses at most half of its first codebook in at least 6 of its 9
  tables per catalog.
- E4. After its last digit, at least 95 % of items are alone in every RQ-VAE
  table.

### Part M2: why items share a SID (54 native tables)

- **A. Catalogue:** identical inputs, near-duplicate pairs, spectrum, norms
  (exp3 Part A).
- **B. Tables:** exp3 Part B unchanged (N0, N1 with 200 permutations, merge
  ratios, repeat rate, near-duplicate shares, closest-mate percentiles,
  radius Spearman, mean pairwise NMI, balance check, A1 ratios).
- **D. Post-processing:**
  - D0 (new, direct test of H1a): each RQ-VAE table's collision rate before
    the reassignment (pre-reassignment table), the share of items whose last
    digit the loop changed, and the last-level squared-distance increase.
  - D1: generate_indices.py's loop (exp3's numpy reimplementation, unchanged)
    applied to every table from its own last-level geometry: RQ-KMeans
    (residuals from the ZCR-bundle codebooks), MQ (last encoder output) and
    RQ-VAE (starting from the pre-reassignment table). On RQ-VAE it must end
    at the published rate; the item-level agreement with the published table
    is reported.
- **Hub codes:** for every RQ-KMeans 5-level fit and level, the most-used
  code's share, its centroid's norm rank, and its items' residual norms
  (exp3 A2).
- **C. Refits:** exp3 Part C conditions (native, per-level seed, scale-norm,
  both, L2, PCA-256, PCA-256-white, PCA-64-white) × widths × 5 seeds, at 8
  threads, both catalogs. The decision rule is exp3's: a condition changes
  the rate only if its 5-seed range does not overlap the baseline's.
- **MQ model audit (new):** whether the encoder weights and `pos` are still at
  PyTorch's default initialisation (every Linear weight and bias within
  ±1/√fan_in, BatchNorm affine weights 1 and biases 0); per-digit codes used;
  agreement between digits.

Expectations:

- E5. Every RQ-VAE table's pre-reassignment rate exceeds its published rate,
  and the published rate is within 0.5 pp of the identical-input floor in at
  least 7 of 9 RQ-VAE tables per catalog.
- E6. D1 brings every RQ-KMeans and MQ table to within 0.5 pp of the floor.
- E7. In the nested RQ-KMeans fits, the near-duplicate share of collided
  items rises with depth at each width and exceeds 0.5 at 5×W.
- E8. Every MQ table's observed/N1 ratio is ≥ 2.
- E9. At every level ≥ 2 of every 5-level RQ-KMeans fit, the hub code's
  centroid has norm rank ≤ 1 (0 = smallest).
- E10. A per-level seed raises the collision rate at depth 3 for every width
  (exp3's direction), under the decision rule.
- E11. The MQ encoders and `pos` are at initialisation in every MQ model.

### Part Z: ZCR twins

Per twin: changed last digits (share of items), the relative increase in
last-level squared distance (recomputed from the geometry), prefix and
collision checks, and the M1 measures. Reported without expectations.

### Part M5 (supplementary): naming and co-purchase

exp1's naming AUC per digit (`naming.describe_cells`, held-out half) and
co-purchase lift per prefix length (`attribute.copurchase_preservation`), on
all 54 native tables.

### Part R (Industrial): how the final maps relate to the old ones

Per table name: share of items with the same full SID, per-digit agreement,
ARI of each prefix partition, and both collision rates. This tells which
provisional mechanistic results (all on old maps) used a map that still
exists.

### Part X (exploratory, labelled as such)

- Prefix sparsity per prefix length: occupied prefixes against W^d, mean
  children per parent, share of items with a unique prefix.
- RQ-VAE in its own latent: per level, codes used under nearest assignment,
  codewords never chosen, and the variance explained per digit in the latent.
- Collision exposure of the data: the share of all interactions, and of
  next-item test targets (each user's last interaction, history = up to 50
  earlier items), whose item shares its SID. Test targets are split into
  repeat (item in the history), SID partner (another history item has its
  SID), variant (a near-duplicate of a history item) and new. This uses the
  frozen interaction sequences as a stand-in for the v3 split, which holds out
  each user's last interaction in the same way.

## What would count

Each expectation is reported as held or failed, per catalog, with the
tables where it fails. No user-level intervals: no model is run. Ranges in
Part C cover k-means initialisation only. Every map is one fit.

## What this cannot establish

- Nothing here measures what a collision costs a trained model (M3/M4 need the
  final run's predictions).
- Part X's test classes use the frozen sequences, not the v3 split files.
- Text-derived labels and a text embedding: category alignment may reflect
  the encoder. `also_buy` (M5) is the behavioural check.

## Amendment after the Industrial smoke run (post hoc, 2026-10-20)

A smoke run (`--quick`: 5 permutations, 4 tables in M1, no Part C) was
inspected before the full run. Part 0 item 4 failed, and the declared stop
rule does not fit what the failure shows.

- **A1 (Part 0.4, Parts C and D).** On this cluster (AMD EPYC 7402), the
  per-level `faiss.Kmeans` refit and faiss's own `ResidualQuantizer`
  (`Train_default`, beam 1, the v3 builder's settings) reproduce the v3
  RQ-KMeans tables on digits 1–2 for every item and on digit 3 for about a
  third of items, at 1, 8 and 96 threads. Thread count changes the refit (at
  96 threads it equals the frozen tables exactly, as in exp3), so the v3
  tables were fitted on hardware whose floating-point results differ from
  this cluster's from level 3 on. The tables are therefore not reproducible
  here by refitting. Their published codebooks (`codebooks.npy` in the ZCR
  bundles) do reproduce every table by nearest centroid (Part 0 checks this
  for all 18), so:
  - Part D and the hub-code analysis use the published codebooks. The stop
    rule was meant to keep them off refit codebooks that do not match the
    tables; they no longer need refits.
  - Part C runs, but as a counterfactual on this cluster. Its native
    baseline is this cluster's refit at 8 threads, not the published table.
    Only within-run differences (condition minus native, exp3's decision
    rule) are interpreted, and the native refit's collision rates are
    reported next to the published ones.

## Outputs

`$SIDLENS_WORK/derived/structure/exp4_final_maps/<job-id>/<catalog>/`: one
CSV per part, `validation.json`, `inputs.json`, `result.json`; then a summary
directory with `report.md`, the paper tables and figures (CSV twins),
`status.txt` and `output.sha256`.
