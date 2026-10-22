# The paper set: what runs on the grid, and where

Declared 2026-10-22, before any grid result. Code: `src/sidlens/grid/pipeline.py`
(`Spec.paper`, `PAPER_CORE_MAPS`, `CASE_MAPS`); scripts:
`scripts/grid/run_paper.sh`, `scripts/grid/run_case_study.sh`. It replaces
`run_best_state.sh` as what the coauthor runs, because every experiment on
every map (the best set) costs about 1,650 A100 GPU-hours at plan rates and
about 4,400 with the grid's AR decoder as measured.

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

## Cost (A100 GPU-hours)

Measured on 2026-10-22: the grid's AR decoder (50 beams, every beam rescored
one at a time, as the run's evaluator did) takes 1.30 s per row, or 1.63 s
with copy reads blocked. Batching each row's rescoring takes 0.37 and 0.51 s.
DiffGRM minutes per cell are medians of our 20-checkpoint runs of 2026-10-14.

| Set | Cells | GPU-hours, evaluator's rescoring | GPU-hours, batched rescoring for interventions |
|---|---|---|---|
| Paper set | 3,307 | about 840 | about 490 |
| Case study, after the paper set | 150 | about 290 | about 150 |
| Best set (every experiment, every map) | 4,249 | about 4,400 | — |

CPU work (about 70 CPU-hours) runs inside the same job on its CPU workers.
Batched rescoring is not implemented; it would change the intervention decodes
of exp6, exp9 and N3, and V2a would keep the evaluator's settings.

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
