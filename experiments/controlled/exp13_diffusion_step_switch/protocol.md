# controlled/exp13: at which reveal step does DiffGRM use the history?

Declared 2026-10-14, before any forward pass of this experiment. Plan:
`docs/plans/DIFFUSION_MECHANISMS_PLAN.md` (X3), which is D1 Part A of
`DIFFUSION_STEP_PATCHING_PLAN.md`.

## Question

Between reveal steps, only the partial SID and the beam score pass on, so a
step's prediction depends on the history only through that step's forward.
Give one step another history: the donor's prediction on the recipient's own
state. How much of the final recommendation does that one step decide? Is a
copy committed at the first step, or can later steps still take it back?

## Design

- **Cells**: the 20 trained next-item checkpoints, in representation/exp4's
  order. **Cohort**: the 6,297-user diffusion test cohort.
- **Donor history H′**: r1 (slot L−1) is replaced by exp7's control at
  shared-prefix level m = 0. The control comes from `ControlPool.candidates`
  with exp7's exclusions, drawn by `condition_rng(20260928, user, L−1, 0)`.
  Users without a control are counted and left out.
- **Schedules** (a set T of steps that use H′): `none`; `all`; `only_t` for
  each step t = 0…n−1; `but_t`, which switches every step except t.
- **Decoding**: `matched_decode.decode` with the confidence policy at beam 1
  (greedy) and beam 64, top 10, catalogue filter, batch 32, decoder chunk
  1,024. These are the settings that reproduce controlled exp1's archive.
  The history is switched by `interventions.diffusion.history_switch_scorer`.
- **Recorded**: top-10 SIDs and scores for every schedule. For beam 1, the
  revealed position and code at every step, from `BeamTrace`.

## Estimands

**Discordant users** are those whose clean top-1 SID (`none`) differs from
the donor's (`all`).

- **Primary**:
  - F(t) = the share of discordant users whose `only_t` top-1 equals the
    donor's top-1. This is a profile over t, at beam 64.
  - The primary contrast is F(0) − F(n−1), paired over users.
- **Secondary**:
  - G(t) = the share of discordant users whose `but_t` top-1 equals the
    clean top-1, which asks whether step t alone can hold the clean answer;
  - the same profiles at beam 1;
  - SID HR@10 of every schedule;
  - at beam 1 and for `only_t`, the split of a changed top-1 into "step t
    revealed another position" and "same position, another code".

## Numerical acceptance

- **V1**: exp1's pilot checks on 8 users.
- **V2**: `none` equals the plain decode (no `score_fn`) bit for bit: codes,
  scores and counts.
- **V3**: `all` equals an independent plain decode of H′ bit for bit.
- **V4**: the plain beam-64 decode matches controlled exp1's archive
  (`181944/cell-XX`, cells 0–17). Codes and counts must be identical, and
  finite scores must agree within atol = rtol = 1e-4. Up to 0.1 % of users
  may instead differ only by ties: their sorted score lists agree within
  1e-5. The A100 runs of 2026-10-13 showed such ties. This tolerance is
  declared here, before any run. Cells 18 and 19 have no archive.
- **V5**: the checkpoint sha256 equals the registry.

## Statistics

Paired user bootstrap, 2,000 draws, seed 20261014. Intervals exclude
training-seed uncertainty.

## Expectations recorded before outcomes

- **E1**: F(0) > F(n−1) at beam 64, with the interval excluding 0, in at
  least 15 of 20 cells. The first step decides most.
- **E2**: F(0) < 0.8 in every cell. Later steps still use the history.
- **E3**: at depth 5, F(t) is flatter than at depth 3: F(0) − F(n−1) is
  smaller at depth 5 in at least 4 of the 6 quantizer × width pairs. Depth-5
  models never trained on the fully masked state, so their first step should
  be less decisive.

## Interpretation limits

- One donor type: the most recent item replaced. A whole-history swap would
  move other evidence.
- At beam > 1, a step's switch also reorders which paths survive. F(t) mixes
  "changed the commit" with "changed the pruning".
- One checkpoint per cell.
