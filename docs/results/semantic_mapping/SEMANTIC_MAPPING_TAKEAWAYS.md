# SID structure across quantizers: takeaways and figures

Generated 2026-09-22 from the frozen SID tables, the frozen item embeddings,
the external Amazon-2018 labels, and the completed derived tables. CPU only;
no model was run. Every figure has a CSV twin with the exact numbers.

**Where the files are.** All semantic-mapping outputs live on the bulk
filesystem, not in the repository:

| What | Path |
|---|---|
| Today's figures (PDF + PNG + CSV) | `/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/figures-20260922/` |
| Copies of the PNG/PDF for the repo | [`docs/results/semantic_mapping/figures/`](figures/) |
| The 15 Sept mapping export (explorer, SQLite, CSVs, coherence) | `/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/` |
| The RQ-VAE collision-bucket audit | `/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/anomaly-20260915/` |
| Generator | [`experiments/structure/semantic_mapping/figures.py`](../../../experiments/structure/semantic_mapping/figures.py); style in [`src/sidlens/viz/style.py`](../../../src/sidlens/viz/style.py) |

Regenerate (about 30 s plus one t-SNE, which is cached afterwards):

```bash
export SIDLENS_WORK=/l/users/leo.rodrigues/sidlens
$SIDLENS_WORK/venv/bin/python experiments/structure/semantic_mapping/figures.py --only 12345678
```

The "primary grid" used by the digit-by-digit figures is the three 4-digit ×
128-code variants (`rqkmeans_4codebook_128`, `rqvae_4codebook_128`,
`MQ_4codebook_128`); the RQ-VAE one is also the AR/diffusion matched cell.
Summary figures use all 27 Industrial variants.

## Takeaways

### 1. The three quantizers spend the same address budget on different things

| | RQ-KMeans | RQ-VAE | MQ |
|---|---|---|---|
| Codes used at digit 1 (of 128 / 256 / 512, 4-digit fits) | 128 / 256 / 512 | **39 / 10 / 40** | 123 / 238 / 450 |
| Items alone after 4 digits (width 128) | 71 % | **99 %** | 63 % |
| Items sharing a full SID, 3 digits × 128 | 29.3 % | 1.2 % | **40.1 %** |
| Items sharing a full SID, 5 digits × 512 | 0.35 %* | 0.35 % | 11.9 % |
| Tightness beyond a random partition, digit 1 → 5, 5 digits × 128 | 0.51 → 0.95 | 0.21 → 1.00 | 0.36 → 0.80 |

\* the RQ-KMeans 5×512 table is a separate post-repair fit, not the nested
extension of 3×512 / 4×512. Its 11 collisions are **not** the 11
identical-text pairs: it gives every one of those pairs two distinct SIDs,
which no deterministic quantizer of these embeddings can do (the other 26
tables collide all 11). How that table was produced is an open provenance
question.

RQ-VAE **under-uses its first codebook** in 8 of its 9 fits (2–31 % of codes;
10 of 256 at 4×256; the exception is 3×128 at 81 %), and in its 4- and 5-digit
fits makes its **last digit an identity digit**: after it, 99 % of items are
alone, and the shared SIDs are essentially the 11 identical-text pairs (one
extra pair at 4×128; at 5×256 two extra pairs and one 19-item bucket, a second
anomaly like the 26-item 3×128 bucket). RQ-KMeans uses 100 % of every codebook
and refines genuinely, but collides heavily. MQ collides most and refines
least. (Figs 1, 5, 8.)

### 2. Residual digits carry semantics only relative to their prefix; parallel digits each carry it alone

Variance of the item embedding explained by **one digit on its own**, and the
category AMI of that digit alone (4×128 fits):

| digit | RQ-KMeans var. / AMI | RQ-VAE var. / AMI | MQ var. / AMI |
|---|---|---|---|
| 1 | 53 % / 0.52 | 34 % / 0.50 | 38 % / 0.34 |
| 2 | 16 % / 0.14 | 18 % / 0.17 | 37 % / 0.33 |
| 3 | 14 % / 0.13 | 10 % / 0.08 | 36 % / 0.32 |
| 4 | 12 % / 0.10 | 8 % / 0.05 | 38 % / 0.34 |

For the residual quantizers, digit 1 is a real partition of the catalogue and
digits 2–4 are corrections that only mean something given digit 1 (Fig 7, top
row: the same code value lands all over the map). For MQ every digit is a
full, similar-quality partition (Fig 7, bottom row), and the digits overlap:
MQ 4×128's digit 2 has category AMI 0.33 on its own but adds only 0.19 once
digit 1 is known (conditional AMI; 0.17–0.23 across the nine MQ fits, and
0.01–0.17 at digits 3–4 where the estimate is supported). For RQ-KMeans the
two are about equal (digit 2: 0.14 alone, 0.14 given digit 1): its later
digits add information rather than repeat it (Fig 5, bottom two rows).

How this relates to the decoding results is still a hypothesis, and the
evidence is mixed. On the six depth-3 diffusion checkpoints, the best fixed
reveal order differs by quantizer. For RQ-VAE, orders that reveal the last
(identity-like) digit first are the worst at both widths. For MQ-128, those
same orders are the best, and confidence decoding gains most there (+0.9 pp
HR@10 over the fixed-order mean) — but MQ-512 gains least (+0.16 pp) and
prefers a different order. For RQ-KMeans, the best fixed order reveals digit 2
first, not digit 1. Reveal-order traces (plan, Tier 1) are the test.

### 3. Category structure is exhausted by digit 2–3; the remaining digits separate identity

Majority-category purity among items that still share a prefix group, against
a shuffled-label null with the same groups (level-1 categories, 4×128 fits):

| depth | RQ-KMeans (null) | RQ-VAE (null) | MQ (null) |
|---|---|---|---|
| 1 | 0.79 (0.20) | 0.64 (0.17) | 0.58 (0.20) |
| 2 | 0.89 (0.36) | 0.91 (0.41) | 0.73 (0.32) |
| 3 | 0.97 (0.45) | 0.99 (0.50) | 0.81 (0.39) |
| 4 | 0.99 (0.47) | 1.00 (0.56)† | 0.87 (0.43) |

† only 1 % of items still share a group.

Level-2 categories (181 classes) follow the same pattern (Fig 4, bottom row).
After three digits an RQ group is almost always one category, so a fourth or
fifth digit separates items *within* a category. That is a statement about
the mapping, not about what the model uses — which is exactly what the
model-side experiments must test (identity-digit probe, plan §4).

The earlier pair-weighted "same full SID" number for RQ-VAE 3×128 (20.6 %)
was an artefact of one 26-item bucket contributing 96 % of the pairs; on the
item-weighted, multi-item-restricted measure that fit is 0.68 against a 0.42
null at depth 3, with only 2 % of items still grouped.

### 4. How the groups shrink (Figs 1, 2)

At width 128 (4-digit fits), the median *item* sits in a prefix group of
33 → 4 → 1 → 1 items under RQ-KMeans, 97 → 3 → 1 → 1 under RQ-VAE (only 39
first-digit groups, then a fast collapse to singletons), and 32 → 5 → 2 → 1
under MQ. The largest groups tell the rest: RQ-VAE 258 → 55 → 22 → 2, MQ
411 → 150 → 28 → 21, RQ-KMeans 86 → 27 → 22 → 22. RQ-KMeans and MQ keep a few
20-item groups at full depth (these are the collision buckets); RQ-VAE does
not.

### 5. Refinement geometry (Fig 3)

The median RMS radius of multi-item groups falls from 30 to 6.8 over five
digits for RQ-KMeans 5×128 (30 → 22 → 13 → 8.6 → 6.8 in the shared 2,560-d
embedding). The chance-corrected index, 1 − R²_within / E[R² of a random
partition into the same number of groups], rises 0.51 → 0.95 (0 = no tighter
than a random partition). MQ shrinks more slowly (35 → 14; index 0.36 → 0.80):
intersecting parallel partitions tightens groups less than residual
quantization does. Wider codebooks give tighter groups for RQ-KMeans and MQ;
not for RQ-VAE, whose first digit collapses further at larger widths.

### 6. What this does and does not establish

- Everything above is a property of the item-to-SID **map**. Whether a trained
  recommender relies on digit 1's semantics, or treats RQ-VAE's last digit as a
  lookup, is the model-side question (plan §4, Tier 2).
- Geometry is measured in the shared input embedding, not in RQ-VAE's own
  latent, whose weights are not in the frozen substrate.
- MQ prefix groups are intersections of parallel partitions; the icicle plot
  draws the prefix tree, which is a valid grouping but not a hierarchy.
- RQ-VAE and MQ refit per depth, so 3- vs 4- vs 5-digit differences confound
  "adding a digit" with a new fit. Only the RQ-KMeans depths are nested.
- One catalogue and one seed per fit; no interval covers fit-to-fit variance.

## Figure captions (paste-ready)

**Fig 1. Prefix trees of one 4-digit × 128-code fit per quantizer.** Each row
splits the row above; width is the share of the 3,105 items in the prefix
group; colour is the group's majority level-1 category (seven largest
categories; grey = other/unlabelled). Annotations give the number of groups
and the share of items that are already alone.

**Fig 2. Item-weighted distribution of prefix-group size at each depth**
(same fits as Fig 1). The dot at size 1 is the share of items alone.

**Fig 3. Group geometry across all 27 variants.** Top: median RMS radius of
multi-item groups in the shared 2,560-d item embedding. Bottom:
chance-corrected tightness, 1 − R²_within / E[R²_random], where E[R²_random]
= (n − k)/(n − 1) is the expected within-group variance share of a random
partition of the n items into the same number k of groups. Solid lines are the
5-digit fits; fainter lines the 3- and 4-digit fits (identical for RQ-KMeans,
whose depths are one nested fit).

**Fig 4. Category coherence above a shuffled-label null.** Purity is the share
of items in multi-item prefix groups whose category equals the group's
majority; the null (200 permutations) keeps the groups and shuffles labels.
Hollow points: fewer than 10 % of items still share a group.

**Fig 5. What each digit position knows.** Per digit: share of the codebook
used; variance of the item embedding explained by that digit alone;
level-1-category AMI of that digit alone; category AMI the digit adds given
the earlier digits (within-parent, hollow where the support gate fails).

**Fig 6. Zooming into one RQ-KMeans branch.** t-SNE of all items coloured by
digit 1; then local PCA of one first-digit group coloured by digit 2, and of
one two-digit group coloured by digit 3; the last panel lists that group's
items with their third and fourth digits. In each map only the seven most
populated codes get a colour; grey marks items with any other code (121 codes,
85 % of items, in the first map). Map axes have no units.

**Fig 7. One digit at a time.** The same t-SNE map coloured by the seven most
populated codes of a single digit position, ignoring all other digits, for
RQ-KMeans (top) and MQ (bottom), with the variance that digit alone explains.

**Fig 8. Full-SID collision rate across depth and width.** Share of items
whose complete SID is shared with another item; the grey line is the floor set
by the 11 identical-text items. The RQ-KMeans 5×512 point is a separate
post-repair fit.

## Literature the takeaways sit against

- [Chen et al., 2026, "What Makes a Good Semantic ID for Generative Recommendation? A Reproducibility Study"](https://arxiv.org/html/2609.24430):
  no SID design wins everywhere; "L1 normalized entropy has near-zero
  correlation with NDCG@10 (Pearson r = −0.02)", i.e. first-level codebook
  balance is "diagnostic but insufficient". Consistent with Takeaway 1: how
  evenly the first codebook is used says little on its own; where the address
  budget goes (semantics in digit 1, identity in the last digit) is the
  question.
- [SIDInspector, CIKM 2026](https://arxiv.org/pdf/2606.10375): mapping-level
  probes (utilisation, aliasing, neighbourhood alignment, popularity
  allocation, structural cost). Our explorer and Figs 1–5 cover the same
  ground with permutation and random-subdivision nulls and add the
  standalone-vs-conditional digit analysis; the model-side layer is what
  neither tool has.
- [Kuai et al., 2024, "hourglass phenomenon"](https://arxiv.org/pdf/2407.21488):
  intermediate RQ layers concentrate on few codes (path sparsity and a
  long-tailed token distribution). Here the under-use is in the *first*
  RQ-VAE layer, not an intermediate one, so it is not the pattern they
  describe; its cause cannot be diagnosed without the RQ-VAE weights, which
  are not in the frozen substrate. RQ-KMeans shows no under-use at any layer.
- [Ju et al., 2026, Semantic IDs at Snapchat](https://arxiv.org/html/2604.03949v1):
  the correlation between SID uniqueness and performance "plateaus" once
  uniqueness passes about 70 % in their experiment, and "uniqueness should not
  be evaluated as a gold standard, but rather as a foundational sanity check
  against collapse"; 1024³ and 256³ codebooks gave the same Recall@10 (6.1).
  Relevant to whether RQ-KMeans's collisions cost retrieval: our
  collision-aware evaluation already shows SID-level hit rates overstate item
  retrieval by 4.5 pp.
