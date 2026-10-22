# The paper set: what runs on the grid, and where

Declared 2026-10-22, before any grid result. Code: `src/sidlens/grid/pipeline.py`
(`Spec.paper`, `PAPER_CORE_MAPS`, `CASE_MAPS`); scripts:
`scripts/grid/run_quick.sh`, `scripts/grid/run_paper.sh`,
`scripts/grid/run_case_study.sh`. It replaces `run_best_state.sh` as what the
coauthor runs, because every experiment on every map (the best set) costs
about 4,400 A100 GPU-hours with the grid's AR decoder as measured.

## Why these experiments go where they go

The paper's thesis (`docs/paper/rewrite-20261005/NARRATIVE_20261022.md`) is
that SID recommenders mainly search the user's history, with the SID prefix as
the search key. A claim the paper makes for every configuration needs a test
on every map. A claim about how the search works, or about one model's
behavior, needs a few maps that span the quantizers. So each experiment runs on
one of three groups of maps, all at each run's best weight state:

- **every map:** 54 maps per catalog (27 native, 27 ZCR twins), both catalogs,
  both paradigms, both tasks where the experiment applies;
- **core maps:** `rqkmeans_4codebook_256`, `rqvae_4codebook_256` and
  `MQ_4codebook_256` (the three quantizers at one size), plus
  `rqkmeans_3codebook_128` and `rqvae_4codebook_128` (the two maps of our
  archived matched cells, so grid numbers can sit beside the draft's), each
  native and ZCR, both catalogs: 20 per paradigm and task;
- **native core maps:** the 10 native tables of the core maps.

| Group | Experiments | What they support |
|---|---|---|
| Every map | Saved-list analyses with target classes (retrospective exp2, exp3, exp4, exp6) and N4, on all 432 runs | T3: where hits come from; the copy rule's agreement with each model; T4: copying by the recent item's age |
| Every map | AR exp3 Part A; exp4 Parts R and K on the validation split (exp4's declared primary split); exp10; N1 | T1: the prefix condition, the late-layer edge, the last digit, native against ZCR on every map |
| Every map | DiffGRM exp7, exp8, exp12 (CPU), exp13, exp14, representation exp5, N1 | T1 and T2 in DiffGRM; T3's copy block during decoding |
| Core maps | AR exp3 Part B; exp4 in full (R, K and H on both splits); exp5 (next-two runs); exp6 and retrospective exp7; exp9; N2 | T1: heads and how the match is computed; T3: the copy block during decoding, with ZCR twins showing collision credit; T4: order against time |
| Core maps | DiffGRM exp1, exp2 (depth 3), representation exp4 | The search (reveal order, equal beams, visited states) |
| Native core maps | AR representation chain (representation exp1–exp3); N3 (both paradigms) | The layer story; dose-response |
| Once | N2 (f), the base model's induction scores | Whether the search reuses induction heads |

Outside the core, exp3 runs Part A only and exp4 runs R and K on the
validation split only. Those cells record `scope: broad`; a later best-set run
reruns them in full rather than counting them as complete.

## Cost

The grid's AR decoder (50 beams, then raw-vocabulary rescoring of every
returned SID) takes 1.30 s per row on an A100 when it rescores one SID per
forward pass, as the runs' evaluator did, or 1.63 s with copy reads blocked
(measured 2026-10-22). Rescoring a row's SIDs in one forward pass takes 0.37
and 0.51 s. exp6, exp9 and N3 therefore rescore in one batch for their
intervention decodes (protocol amendments of 2026-10-23); their reproduction
gates still decode at the evaluator's setting. DiffGRM minutes per cell are
medians of our 20-checkpoint runs of 2026-10-14.

H200 hours divide A100 hours by 2-2.5 for batched forward passes, by 1.2-1.5
for beam decoding and DiffGRM, and by 1.3 for the representation chain. These
ratios are assumptions; the job's progress file measures the real ones.

| Set | GPU cells | A100 GPU-hours | H200 GPU-hours | 16 H200 (+25 %) | 8 H200 (+25 %) |
|---|---|---|---|---|---|
| Quick set | 553 | about 210 | 115-143 | 9.0-11.2 h | 18-22 h |
| Paper set | 1,019 | about 480 | 274-330 | 21-26 h | 43-52 h |

CPU cells (about 70 CPU-hours at 16 threads) run beside the GPU cells on
`BATCH_CPU_WORKERS` workers per node.

## The quick set

`run_quick.sh` runs the paper set cut to one test per claim, sized to 8-14
hours on 16 H200s (two 8-GPU nodes, `BATCH_NODES=2`). It is a subset of the
paper set: `run_paper.sh` afterwards skips every finished cell, except that
the core maps' exp3 and exp4 cells, which the quick set runs with fewer parts,
are rerun with all of them.

| Group | Experiments |
|---|---|
| Every map | Saved-list analyses with target classes and N4 (all 432 runs); AR exp3 Part A, exp4 R and K on the validation split, exp10; DiffGRM exp7, exp8, exp12 (CPU) and exp14 |
| Core maps | AR exp6 (and retrospective exp7), N1 (both paradigms); DiffGRM exp13, representation exp5 |
| Native core maps | AR exp9, exp5 (next-two runs) |
| Once | N2 (f) |

Left for the paper set: exp3 Part B, exp4 Part H and the test split, N2, N3,
the AR representation chain, DiffGRM exp1, exp2 and representation exp4.

## The case study (optional)

`run_case_study.sh` runs every experiment, in full, on the native
`rqkmeans`, `rqvae` and `MQ` maps at 3x256 and 4x256 in both catalogs, plus
the cheap set (AR exp3 Part A and exp4 Part R; DiffGRM exp7) at their
intermediate weight states. It covers what the paper set leaves out: the
expensive experiments on two maps per quantizer, and when copying forms during
training.

## Not claimed

The core maps are a choice, not a sample: results there describe those maps.
One seed per run; intervals exclude training-seed variance.
