# N4: no-training baselines (protocol)

Declared 2026-10-21, before any baseline or comparison was computed on a grid
run. Plan: `docs/plans/COPYING_GRID_PLAN.md` §4.3.

## Question

How much of what each grid model does could a rule with no training do? A
model that copies prefix-matched history items should agree with a copy rule
on the first code and often on the whole SID; on new targets it should beat
popularity, or copying is all it learned.

## Cohort and runs

Every grid run at its best state (both catalogs, both paradigms, next-item and
next-two), on its full prepared test cohort (no user sample: no model is run).
Baselines are built per run from that run's own SID table and training rows.

## Baselines (top-10 lists of catalog SIDs; next-two: slot 1 only)

- **copy**: rank catalog SIDs by their history LCP (longest prefix shared with
  any history item, so a history SID itself comes first), then by the recency
  of the most recent history item giving that LCP, then by training
  popularity, then by code order.
- **pop**: training popularity of the target SID (count of training-row
  targets), ties by code order.
- **pop_r1**: SIDs whose first code equals the most recent history item's,
  by popularity, then every other SID by popularity.

Training popularity counts the targets of the run's training cohort
(`train-cohort.json`).

## Estimands (paired user bootstrap, 2,000 draws, seed 20261021)

Per run, at SID level and at item level (uniform tie-break inside a SID's
bucket; lower and upper bounds also kept):

1. HR@10 and NDCG@10 of each baseline, all rows and by retrospective exp7's
   target class (same_item, sid_partner, nd_variant, new).
2. The model's saved list against the same rows: HR@10 by class, and
   model minus each baseline on `new` targets (primary: model - pop, SID
   HR@10, new targets).
3. Agreement of the model's top-1 with the copy rule's top-1: same first code,
   same SID (primary: same first code).

## Not claimed

Agreement is not mechanism: a model can agree with the copy rule for other
reasons. Intervals exclude training-seed variance (one seed per run).
SID-level and item-level numbers are reported separately.
