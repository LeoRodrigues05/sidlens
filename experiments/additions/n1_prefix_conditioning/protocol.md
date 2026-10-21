# N1: prefix-conditioning test (protocol)

Declared 2026-10-21, before any condition was scored on a grid model. Plan:
`docs/plans/COPYING_GRID_PLAN.md` §4.3.

## Question

exp4 (AR) and exp7 (DiffGRM) found that a history item h supports its own
digit-d code at the target's digit d when h's first d digits equal the
target's (prefix-matched copying). Does the copy need the whole prefix to
match, or only the digit token just before it, which is what a plain
induction head would match on?

## Runs and rows

Every next-item grid run at its best state, both catalogs, both paradigms.
Industrial: all 6,297 test users; Office: the fixed 6,297-user sample.

A (row, d) is eligible for d in 2 .. n-1 (0-based digit index; depth-3 maps
contribute their last digit only) when some history item matches the target
on digits 0..d-1. h is the most recent such item. The row is used only if no
other history item supports h's digit-d code under either reading (same
digit-d code and either the same first d digits as the target or the same
digit d-1).

## Conditions (h replaced by h', golden prefix given, digit d scored)

| condition | h' | supports h's digit d under |
|---|---|---|
| full | h | both readings |
| suffix | digits d-1 and d equal h's; some earlier digit differs from the target's | last-token matching only |
| none | digit d-1 differs from the target's; digit d differs from h's | neither (baseline) |
| prefix | first d digits equal the target's; digit d differs from h's | prefix matching, at h''s code |

h' is a catalog item (not in the row's history or target; drawn with a
recorded per-row seed) where one exists; otherwise a code combination outside
the catalog, built from codes the map uses at each digit, and flagged
`synthetic`. Catalog and synthetic rows are reported separately.

## Estimands (paired user bootstrap, 2,000 draws, seed 20261021)

support(c) = log p(h's digit-d code | condition c) - log p(same | none).

- Primary: rho = mean support(suffix) / mean support(full), catalog rows.
  Prefix conditioning predicts rho near 0; last-token copying predicts rho
  near 1.
- support(full), support(suffix) separately.
- Prefix move: log p(h''s digit-d code | prefix) - log p(same | none), and
  the drop of h's own code under prefix.

MQ maps (parallel partitions, not a hierarchy) are reported separately.

## Not claimed

These are teacher-forced log-probabilities at the golden prefix, not beam
search outcomes. Intervals exclude training-seed variance.
