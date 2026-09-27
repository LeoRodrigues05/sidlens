# Semantic mapping: results and next experiments

Executed September 15, 2026, CPU job **194268**. This is a new lossless mapping
export and category-coherence experiment using the frozen assignments.

> **Where the outputs are.** Every link below is an absolute path on the bulk
> filesystem (`/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/`), not
> a file in this repository, so the IDE cannot open them as workspace links.
> The 2026-09-22 follow-up — eight publication figures and the takeaways drawn
> from them — is in [SEMANTIC_MAPPING_TAKEAWAYS.md](SEMANTIC_MAPPING_TAKEAWAYS.md),
> with figure copies under [`docs/results/semantic_mapping/figures/`](figures/).

## Explore the complete catalogue

- [Interactive explorer](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/explorer.html): choose a variant, follow digit prefixes, filter any position, search titles/ASINs, and download every matching item.
- [Complete worked examples](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/examples.md): every item under three illustrative first-digit prefixes, with all two-digit branches.
- [All item mappings, CSV](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/mappings.csv).
- [All prefix groups, CSV](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/prefix_groups.csv).
- [Indexed SQLite database](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/mappings.sqlite).
- [Coherence results](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/coherence.csv), [validation](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/validation.json), and [provenance](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/manifest.json).

The export contains **3,105 items × 27 variants = 83,835 assignments** and
**189,191 observed prefix groups**. All titles and metadata identities joined.
All collision members are preserved. SQLite integrity and 324 prefix lookups
against the original loader passed. Categories are present for 3,003 items at
level 1; a joined label record can still lack a category value.

## Concrete mappings

These examples were selected after inspection to make relationships and
exceptions visible. They are not a random sample. Titles below are shortened;
the exports retain their full text.

### RQ-KMeans, three digits, width 128

`(14)` contains **20 eSUN printer-filament products**. Its child `(14)(16)`
contains **14 items**. One complete SID remains ambiguous:

| Full SID | ASIN | Item |
|---|---|---|
| (14)(16)(14) | B00LMPP5Y8 | eSUN Black PLA filament |
| (14)(16)(14) | B00MVIPG80 | eSUN Natural PLA filament |
| (14)(16)(14) | B00MV7T0KQ | eSUN White PLA filament |
| (14)(16)(14) | B00MVIPK68 | eSUN Pink PLA filament |

The sibling `(14)(16)(123)` contains three PLA PRO products. Another sibling,
`(14)(16)(47)`, is mostly ABS but also contains glass-red PLA. The grouping
suggests product-family and material associations, with explicit exceptions.
It does not establish an exact material rule or a universal color digit.

### RQ-VAE, three digits, width 128: compare the same items

`(10)` contains **58 filament/accessory items**. `(10)(12)` contains exactly:

| Full SID | ASIN | Item |
|---|---|---|
| (10)(12)(0) | B00MVIPK68 | eSUN Pink PLA filament |
| (10)(12)(19) | B00MV7T0KQ | eSUN White PLA filament |
| (10)(12)(53) | B0186FW2OM | eSUN Black PLA filament |
| (10)(12)(64) | B00LMPP5Y8 | eSUN Black PLA filament |

The final digit separates these four catalogue identities, including two black
products with the same title. This supports a hypothesis about local identity
resolution; it is insufficient to label that entire codebook “color.”

### MQ, three digits, width 128

`(122)` contains **8 hydrometer/test-jar items**. `(122)(99)` contains exactly:

| Full SID | ASIN | Item |
|---|---|---|
| (122)(99)(67) | B00TUQIBG0 | Brew Tapper Triple Scale Hydrometer |
| (122)(99)(67) | B0064O94I0 | Hydrometer, Triplescale |
| (122)(99)(121) | B000E60U6Y | Triple Scale Hydrometer |
| (122)(99)(121) | B0064O952K | Plastic hydrometer test jar |
| (122)(99)(121) | B00KN2AP6Y | Glass hydrometer test jar |

Instruments and complementary containers group together, but some full SIDs
mix both. MQ prefixes are intersections of parallel partitions; prefix browsing
does not establish an intrinsic coarse-to-fine order.

## Measured relationships across the catalogue

For each variant and prefix depth, we measured the fraction of distinct,
category-labelled item pairs in the same prefix group that share their level-1
category. We compared this with **200 shuffled-label assignments**, preserving
group sizes and missing-label locations. All 27 variants and both category
levels 1 and 2 are in the CSV; these are the three-digit, width-128 results:

| Quantizer | Same first digit | Same first two | Same full SID | Items with a unique full SID |
|---|---:|---:|---:|---:|
| RQ-KMeans | 68.93% | 85.98% | 96.96% | 55.20% |
| RQ-VAE | 59.00% | 81.29% | 20.58% | 98.45% |
| MQ | 22.80% | 36.82% | 49.80% | 45.48% |

Random catalogue pairs share the level-1 category **8.34%** of the time.
First-digit permutation means were 8.34%, 8.36%, and 8.32%, respectively.
All three quantizers therefore group categories more than random assignment,
with different identity/collision behavior.

**The RQ-VAE full-SID column needs care.** Most items have unique SIDs and
contribute no pairs. A single anomalous SID, `(27)(89)(7)`, contains **26 diverse
products**, including a microwave, vacuum, calipers, and cleaner. Its 25 labelled
members contribute **300 of 311 eligible pairs (96.46%)**. The full-SID statistic
mostly describes that bucket, rather than the typical RQ-VAE item. This is a
useful anomaly to investigate next.

The [completed follow-up audit](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/anomaly-20260915/report.md)
found **26 distinct, finite, nonzero input embeddings** in that bucket, with
median pair distance 36.93 and RMS radius 25.48. The other 11 colliding buckets
in this fit each contain two identical input vectors. Every other available
RQ-VAE fit assigns these same 26 items 26 distinct SIDs. The audit preserves
[all members and their assignments across fits](/l/users/leo.rodrigues/sidlens/derived/semantic_mapping/anomaly-20260915/members.csv).
These are distances in the shared input space, not the unavailable learned
quantizer latent space; they do not locate the cause of the anomaly.

The measurements are descriptive associations with external category labels.
They do not isolate information added conditionally by each digit, establish
model use of semantics, or include training-seed uncertainty. Pair weighting
emphasizes large groups; the exports report pair counts and singleton share.
Null ranges are permutation ranges, not confidence intervals.

## Scale generation and query it

```bash
# Generate all variants into a fresh job-specific output directory.
sbatch scripts/structure/semantic_mapping.sbatch

# Retrieve ALL members of any prefix, including complete-SID collisions.
/l/users/leo.rodrigues/sidlens/venv/bin/python \
  -m sidlens.analysis.semantic_mapping query \
  --db /l/users/leo.rodrigues/sidlens/derived/semantic_mapping/194268/mappings.sqlite \
  --variant rqkmeans_3codebook_128 --prefix 14,16 --format csv
```

Use `--prefix 14` for the first digit and `--prefix 14,16,14` for the full SID.
Add `--contains PLA` to filter titles. The browser also supports independent
position filters and exports all matches, not just the displayed page.

The generator indexes observed assignments and prefixes without enumerating
the mostly empty code space. SQLite stores each item's metadata once and has
a composite index on variant plus all digit values. The 2.2 MB standalone
explorer needs no server or LLM calls. For much larger catalogues, use the
database or partition browser payloads per variant. Permutations stream one
assignment at a time. Existing atlas `naming.csv` files provide a route to
automated sibling-discriminating labels with held-out naming evaluation.

## Completed decoding experiment

The [full fixed-order report](/l/users/leo.rodrigues/sidlens/derived/controlled/exp2_fixed_orders/summary-194295-194297/report.md)
compares all six reveal orders with confidence on **six depth-three diffusion
configurations × 6,297 users**, at beam 64. Confidence SID hit@10 was **16.680%**,
versus **16.278%** averaged over the six fixed orders: **+0.402 percentage points**
with a paired-user 95% interval **[+0.287, +0.523]**. This compares outcomes of
separate fixed-order lists; it is not an ensemble or a claim about every
individual model/order comparison. Full-cohort baseline reconstruction,
predictions, identities, hashes, and an independent primary-metric/bootstrap
audit passed. The complete run used the archive's batch/chunk settings of
32/1,024; the initial attempt with other settings remains recorded separately.
Beam 256 was validated in a 64-user pilot; its full sweep remains a follow-up.

## Runnable follow-ups

1. **Beam sensitivity:** run the existing `exp2_fixed_orders` runner at beam
   256 with batch/chunk settings 32/1,024, extending the completed beam-64 result.
2. **Collision anomalies:** the input-embedding audit above is complete;
   recover quantizer provenance/weights and quantify effect on item retrieval.
3. **Attribute refinement:** validate color/material/product-family labels,
   then test digit information within each parent with item-held-out checks.
   Existing conditional-AMI and naming infrastructure supports this extension.
4. **Behavioral relationships:** compare shared prefixes with co-purchase
   neighbors, adding degree-aware and within-parent controls to existing results.
5. **Partial-state deduplication:** a new decoder intervention and numerical
   validation are needed before its GPU run.
6. **Logit traces, probes, history interventions:** require the documented
   diffusion-logit path validation and new capture/intervention code.

New model training, SAEs, and a matched joint-block next-two comparison require
additional artifacts or implementation and are not runnable as existing jobs.
