# N3: copy dose-response inside beam search (protocol)

Declared 2026-10-21, before any scaled decode was run on a grid model. Plan:
`docs/plans/COPYING_GRID_PLAN.md` §4.3.

## Question

exp6 (AR) and exp14 (DiffGRM) removed the copy reads and measured Hit@10.
Removal is one point. If copying helps repeat targets and costs new ones, the
trade should move smoothly as the reads are turned down and up: is there a
dose at which repeat targets keep most of their hits while new targets gain?

## Runs and rows

Every next-item grid run at its best state, both catalogs, both paradigms.
Industrial: all 6,297 test users; Office: the fixed 6,297-user sample. Each
run is decoded with its own evaluator's search (`grid.decode_ar`,
`grid.decode_diff`) at its own beam, list length and batch size.

## Conditions

alpha in {0, 0.25, 0.5, 1, 1.5, 2} multiplies the attention weights of:

- AR: exp6's `C_all` edges (each beam's reads of digit d of history items whose
  first d digits equal its prefix, and every first-digit read, in layers
  14-27 of 28), in generation and in the rescoring pass, as log(alpha) added
  to the attention logits on the float-mask route.
- DiffGRM: exp14's `block_r1` reads (every other real slot reading the most
  recent item in the encoder, and every digit's cross-attention to that slot
  in every decoder block), reweighted after the softmax and renormalised.
  Users with one history item are decoded but excluded (no other slot).

alpha = 0 is exp6's / exp14's block; alpha = 1 changes nothing.

## Gates

- V2a: the plain decoder reproduces the run's saved lists (AR: first 512 rows
  in CSV order; DiffGRM: `diff_setup.check`, the evaluator's first 64 users).
- V3: alpha = 1 equals the unscaled route bit for bit (AR: the float-mask
  route with an empty plan, on 64 rows; DiffGRM: the plain decode, all users).
- V2b (reported): AR alpha = 1 against the saved plain lists (the explicit-mask
  kernel differs from the plain one by float noise).

## Estimands (paired user bootstrap, 2,000 draws, seed 20261021)

Per run and alpha, at SID level and item level (uniform tie-break): Hit@10 on
repeat targets (retrospective exp7's `same_item` and `sid_partner`) and on new
targets (`new`), the change from alpha = 1, and the share of history SIDs in
the top 10. Primary: the change in new-target SID Hit@10 at alpha = 0.5.

## Not claimed

Scaling attention weights is not a model the run was trained as; values of
alpha other than 1 are interventions, not settings. Intervals exclude
training-seed variance.
