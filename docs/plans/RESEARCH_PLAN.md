# SidLens research plan

Written 2026-09-22 from the repository, the derived results under
`$SIDLENS_WORK/derived`, the project brief
(`docs/plans/LeoRodrigues_ProjectProposal_GenRec_MechInterpAnalysis.pdf`), and the
September 2026 literature. Companion documents:
[semantic-mapping takeaways](../results/semantic_mapping/SEMANTIC_MAPPING_TAKEAWAYS.md) (today's figures),
[the three-hour ladder](THREE_HOUR_EXPERIMENT_PLAN.md) (experiment levels and
interpretation constraints), and `CLAUDE.md` (repository orientation).

## 1. What the project is for

Developers of Semantic-ID recommenders choose a quantizer (RQ-VAE, RQ-KMeans,
parallel MQ), a depth and width, a digit order, a decoding paradigm
(autoregressive or masked diffusion) and a search policy. Leaderboards say
which combination scored higher on one dataset; they do not say **why**, and
the September 2026 reproducibility study of SID design
([Chen et al., "What Makes a Good Semantic ID for Generative Recommendation?"](https://arxiv.org/html/2609.24430))
finds that the effects are non-monotonic, that no design wins everywhere,
and that first-level codebook balance (normalised entropy) has near-zero
correlation with NDCG@10 (Pearson r = −0.02). Mapping-level diagnostic tools now exist
([SIDInspector, CIKM 2026](https://arxiv.org/pdf/2606.10375): utilisation,
aliasing, neighbourhood alignment, popularity allocation, structural cost),
but they stop at the item-to-code table.

SidLens's claim to novelty is the chain from **structure** to **mechanism**:

1. **Structure.** What each design choice does to the item-to-SID map:
   which digit carries which information, how prefixes refine, where the
   identity budget is spent, how many items collide. (Static; done for all 27
   Industrial variants.)
2. **Search.** How the decoder interacts with that structure: beam waste on
   duplicate reveal paths, invalid paths, order sensitivity, collision-aware
   item retrieval. (Done for diffusion at depth 3–5; AR retrospectively.)
3. **Computation.** What the trained network represents and causally uses:
   per-digit logit trajectories, layer/step probes, history and digit
   interventions, feature families shared or not shared across paradigms.
   (Hooks exist; nothing else yet.)

A design recommendation is only defensible when all three layers agree. That
is the paper's structure, and it is what distinguishes the work from both the
reproducibility study (layer 0: scores) and SIDInspector (layer 1 only).

## 2. What is established, with the evidence behind it

### Structure (static, 27 variants, `derived/exp1_atlas`, `exp2_*`, `semantic_mapping`)

| Finding | Evidence | Figure |
|---|---|---|
| Residual quantizers put almost all standalone semantics in digit 1; later digits are relative refinements (informative only given the prefix). Parallel MQ digits are each a full partition of similar quality, and therefore mutually redundant. | Variance explained by one digit alone: RQ-KMeans 4×128 = 53 / 16 / 14 / 12 %; MQ 4×128 = 38 / 37 / 36 / 38 %. Category AMI of digit alone: RQ 0.52 → 0.14; MQ 0.34 flat. | Figs 5, 7 |
| RQ-VAE under-uses its first codebook in 8 of 9 fits (2–31 % of codes at digit 1; 10 of 256 for 4×256; 81 % only at 3×128) and, in its 4- and 5-digit fits, uses a last digit that is almost purely an identity digit (99 % of items alone after it). | `utilization.csv`, `fig1_icicle.csv` | Figs 1, 5 |
| RQ-KMeans uses 100 % of every codebook but collides heavily (29 % of items share a full SID at 3×128; 9.6 % at 3×512); MQ collides most (40 % at 3×128). Six of the nine RQ-VAE fits sit at or one pair above the 11-pair identical-text floor (0.35 %). | `collisions.csv` | Fig 8 |
| RQ-KMeans 5×512 (the post-repair refit) has 11 collisions, but they are not the identical-text pairs: it gives all 11 pairs distinct SIDs, which a deterministic quantizer of these embeddings cannot do; the other 26 tables collide all 11. Its construction procedure needs provenance before it is compared with the rest. | SID tables + embeddings | — |
| Category coherence of grouped items is far above a shuffled-label null at depths 1–2 for all quantizers and saturates by depth 3; additional digits mainly separate identity. | `fig4_coherence.csv` | Fig 4 |
| Groups are much tighter than chance: RQ-KMeans's index against a random partition with the same number of groups rises 0.51 → 0.95 over 5 digits; MQ is the least tight (0.36 → 0.80). | `exp2_refinement` | Fig 3 |
| RQ-KMeans depth variants are one nested fit (except 5×512, a post-repair refit); RQ-VAE and MQ refit per depth, so "add a digit" is not a controlled manipulation for them. | `sids.py::check_nesting` | — |
| One anomalous RQ-VAE 3×128 bucket puts 26 unrelated products under one SID despite distinct embeddings; every other fit separates them. | `semantic_mapping/anomaly-20260915` | — |

### Search (`derived/controlled`, `derived/retrospective`)

| Finding | Evidence |
|---|---|
| Confidence-guided diffusion decoding beats the mean of all six fixed orders at depth 3 (+0.40 pp SID HR@10, 95 % CI [0.29, 0.52]), but the best fixed order is quantizer-specific (last-digit-first is best for MQ-128, worst for RQ-VAE). | exp2_fixed_orders |
| At equal beam 64 confidence loses to the seed-42 fixed order by 0.95 pp overall, and the gap grows with depth (+0.4 → −0.7 → −2.6 pp); at beam 256 it wins. | exp1_matched_beam |
| Mechanism: the confidence beam keeps paths, not states. Duplicate paths to the same partial SID consume 57 / 79 / 91 % of the final beam at depths 3 / 4 / 5 (beam 64); at depth 5 only 5 % of users receive ten predictions. The seed-42 fixed order instead loses 30 / 47 / 50 % of its final beam to invalid SIDs. | matched-beam diagnostics |
| SID-level hit rates overstate item retrieval by 4.5 pp on average (14 pp for collided targets). | retro exp3 |
| AR errors are front-loaded: the first wrong digit is digit 1 in 82 % of misses. | retro exp2 |
| Next-two: P(item 2 correct | item 1 correct) = 4.0 % vs 0.3 %, associational only. | retro exp4 |

Three training-time facts (verified in the vendored next-item DiffGRM code)
constrain what the diffusion results can mean. At each training step, each
example is trained along **one** reveal order, ranked by the model's own
confidence on the fully masked input, so fixed orders mostly visit partial
states that training rarely covers. Depth-5 models never see the fully masked
SID during training (`guided_steps` is capped at 4), so every depth-5 decode
starts out of distribution. And the ranking pass runs with `use_cache=False`,
the unprojected cross-attention fallback, so the order is chosen by a slightly
different network from the one being trained.

## 3. Research questions → experiments → status

| RQ (brief) | Experiment | Layer | Status | Needs |
|---|---|---|---|---|
| RQ2 SID organisation | Exp 1 atlas, Exp 2 geometry, semantic mapping | structure | **done** (27 variants) | — |
| RQ2 | Per-digit probes: can category / broad prefix / item identity be read from the decoder state, and at which layer/step? | computation | not started | matched cells, capture code |
| RQ2 | Digit-swap and later-digit patching: does replacing an identity digit change the output item but not its category? | computation | not started | intervention code |
| RQ3 order | Fixed orders vs confidence (depth 3) | search | **done** | beam-256 sweep, depth-4 (24 orders) |
| RQ3 order | Reveal-order traces: which digit does confidence reveal first, does it vary by user, does it match the training-time order and the standalone informativeness of digits? | search→computation | not started | trace runner (validated logit path exists in `matched_decode`) |
| RQ3 scale | Why depth 5 fails: partial-state dedup, legality-aware search, exact oracle | search | not started | new decoder + validation |
| RQ3 scale | Identity-digit hypothesis: models treat RQ-VAE's last digit as a lookup, not semantics (logit entropy, probe readability) | computation | not started | capture code |
| RQ3 collisions | Collision-aware evaluation; ZCR at item level | search | partially done | ZCR outputs from upstream |
| RQ1 paradigm | Matched AR/diffusion probes, history patching, feature alignment | computation | not started | 2 matched cells exist |
| RQ3 AR order | Coarse-to-fine vs fine-to-coarse vs random retraining | training | not started | upstream fine-tunes with checkpoint retention |
| RQ4 next-two | Joint-block diffusion vs AR; clamp item 1 | training+computation | blocked | joint-block checkpoints |

## 4. Roadmap

### Tier 1 — this week, existing checkpoints, CPU + minutes of GPU

1. **Partial-state deduplication and legality-aware search** for diffusion:
   merge candidates that reach the same (mask, values) state before pruning;
   optionally prune states no catalogue SID can complete. Run all 18 cells at
   beams 64 and 256. Add an **exact-search oracle** (score every legal SID
   under each order) for a user sample, so each miss is attributed to search
   error or model error. *Claim:* the depth penalty is a search artefact, and
   what remains is order inconsistency plus the out-of-distribution first step.
   *Figure:* depth × beam heatmap before/after; unique-state ratio per step.
2. **Reveal-order traces.** Record the order confidence chooses per user and
   cell. Test three things: (a) agreement with the training-time guided order;
   (b) whether the first revealed digit is the most standalone-informative one
   (Fig 5 row 2), which would predict RQ→digit 1 first (AR-like) and MQ→any
   digit — the fixed-order results already cut against this for RQ-KMeans,
   whose best fixed order reveals digit 2 first; (c) user dependence.
   *Figure:* order flow diagrams per quantizer.
3. **Paper figure set and write-up** of the structure layer (done today; see
   takeaways). Add the same tables for Office (SIDs and embeddings are frozen;
   labels need one download).

### Tier 2 — next two weeks, GPU, new capture code

4. **Activation capture format** that the upstream training-time logs (about
   100 GB per dataset × paradigm) can be loaded into: example, checkpoint,
   layer, digit position, step, mask state, seed. Do this before the logs
   arrive.
5. **Per-digit logit trajectories** on the two matched cells (AR layers ×
   diffusion steps): when does the correct code become top-1, and does the
   target's category become decodable before its identity? Use the validated
   projected-cache path, not `diffusion.digit_logits`.
6. **Probes** (user-disjoint splits, shuffled-label baselines): target
   category, first digit, full identity, at each AR layer and diffusion step.
7. **Interventions with provenance:** history-item replacement with
   frequency/category-matched controls; digit swaps; head-output ablation.
   Each needs the registered-patch mechanism that `spec.py` names but the
   repository does not yet implement.

### Tier 3 — needs upstream artefacts

8. AR order retraining (Exp 4) with checkpoint retention per the run-export
   contract in the README.
9. Joint-block next-two checkpoints (Exp 6); the retained runs use two-pass
   inference and cannot answer the joint-plan question.
10. ZCR / CCE evaluation here, at item level, once exported and hashed.
11. Office replication of the structure layer and of the fixed-order result.

## 5. Design-choice verdicts the evidence supports today

| Choice | What can be said now | What is still needed |
|---|---|---|
| Quantizer | RQ-KMeans: full code use, strongest hierarchy, many collisions. RQ-VAE: under-used first digit, near-zero collisions, last digit is an identity digit. MQ: every digit informative alone but overlapping, least tight groups, most collisions; the largest single gain from flexible reveal order (+0.9 pp on MQ-128), though MQ-512 gains least (+0.16 pp). | Whether the models *use* the hierarchy (probes, patching). |
| Depth | Beyond digit 3, extra digits add identity, not category information (Fig 4). Depth 5 hurts confidence decoding through beam duplication, and depth-5 diffusion models never train on the fully masked start state. | Dedup experiment; identity-digit probe. |
| Width | Width lowers collisions for RQ-KMeans and MQ; RQ-VAE uses only 2–8 % of a 512-code first codebook. | Whether under-use costs retrieval (the reproducibility study finds first-level balance alone does not predict NDCG). |
| Digit order (diffusion) | Order matters per quantizer; confidence is a good default at depth 3 and beam ≥ 256. | Beam-256 sweep, depth-4 orders, traces. |
| Beam / search | Equal caps are not equal compute; confidence needs state-level dedup to scale. | Tier 1 item 1. |
| Paradigm | Only two matched next-item cells; no claim yet. | Tier 2. |

## 6. Threats to validity to keep in the paper

- One seed per cell; user-bootstrap intervals exclude training-seed variance.
- One catalogue (3,105 items). Office (17,696) is the replication target.
- Geometry is measured in the shared input embedding, not RQ-VAE's latent.
- MQ prefixes are intersections of parallel partitions; prefix-tree figures
  for MQ must say so.
- SID-level metrics hide collisions; report item-level bounds alongside.
- Diffusion training saw one reveal order per item; fixed-order results are
  conditional on that training recipe.
