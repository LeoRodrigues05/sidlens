# controlled/exp12: paired-run activation patching in DiffGRM

Declared 2026-10-14, before any forward pass of this experiment. Plan:
`docs/plans/DIFFUSION_MECHANISMS_PLAN.md` (X2); it carries out D2 of
`DIFFUSION_STEP_PATCHING_PLAN.md` on all 20 checkpoints.

## Question

Two runs of one checkpoint differ in a single input. Either one history item
is replaced (Part H), or one revealed digit of the partial SID holds a
sibling's code (Part S). Where between the input and the scored digit does the
difference live?

- **Part H.** Which encoder slots carry the replaced item, through which
  decoder block's cross-attention it enters, and at which digit position.
- **Part S.** How a revealed digit's information reaches the digit being
  predicted: at which block, and how much through the direct self-attention
  edge.

exp7 and exp8 answered the first question with knockouts on two cells. Here it
is asked by sufficiency (denoising) and necessity (noising), on every cell, so
that the answer can be compared across SID digit and codebook width.

## Cells, cohort, scoring

- **Cells**: the 20 trained next-item checkpoints in representation/exp4's
  order.
- **Cohort**: the 6,297-user diffusion test cohort; Part H uses users with
  L ≥ 2. Never joined to the AR cohort.
- **Scoring**: `interventions.diffusion.score_states` (projected cross cache,
  the path exp1 validated). fp32, TF32 off, deterministic algorithms, GPU;
  fixed 1,024-row forwards.

## Pairs

- **H0**: the fully masked state, scoring digit 0. The corrupted history
  replaces r1 (the most recent item, slot L−1) with exp7's control at
  shared-prefix level m = 0. That control comes from `ControlPool.candidates`
  with exp7's exclusions, drawn by `condition_rng(20260928, user, L−1, 0)`.
- **Hd, d = 1…n−1**: the state `S_pref(d)` (golden digits 0…d−1 revealed, the
  rest masked), scoring digit d. The corrupted history replaces r1 with exp7's
  control at level m = d. Rows are split by **match_d** (r1 equals the target
  on digits 0…d−1). Matching rows are the copy rows.
- **Sd, d = 1…n−1**: the history is the user's own. The clean state is
  `S_pref(d)` and is scored at digit d. In the corrupted state, digit d−1
  holds c′, the digit-(d−1) code of a catalogue item that shares the target's
  digits 0…d−2 and differs at d−1. That item comes from
  `ControlPool.candidates(target, d−1, {target id}, {target SID})`, drawn by
  `condition_rng(20261014, user, "s", d)`. Some catalogue SID completes the
  corrupted state.

A missing control is counted, never dropped silently.

## Sites

Denoising runs the corrupted input with a clean piece patched in. Noising runs
the clean input with a corrupted piece patched in. Both directions are run at
every site.

| Site | What is taken from the other run | Parts |
|---|---|---|
| `E:r1` | encoder output at r1's slot | H |
| `E:others` | encoder output at every other real slot | H |
| `E:r2` | encoder output at the second most recent slot | H |
| `E:all` | encoder output at every slot (control, see below) | H |
| `D:L{l}:p{p}` | decoder residual after block l = 0…3, at digit position p | H, S |
| `X:L{l}:p{p}` | block l's cross-attention output at position p (`patch_cross_attention`) | H, S |

**Part S knockouts** cut the self-attention edge from query d to key d−1 in
both runs, and Δ is taken between the knocked-out runs:

- `K:all`: the edge is cut in all four blocks.
- `K:L{l}`: the edge is cut in block l only.
- `K:h{h}`: the edge is cut for head h in all blocks (descriptive).

## Measures

Every measure is taken at the scored digit's distribution, with p_x the
distribution under condition x.

- **KL** = KL(p_clean ‖ p_x).
- **Denoising restored share**: R = 1 − Σ KL(patched) / Σ KL(corrupted),
  pooled over rows.
- **Noising damage share**: N = Σ KL(noised) / Σ KL(corrupted).
- **Golden versions** (secondary), with g = log p(golden code):
  (g_patched − g_corrupted) / (g_clean − g_corrupted) for denoising, and
  (g_clean − g_noised) / (g_clean − g_corrupted) for noising.
- **Copy score** (Part H, secondary): log p of r1's digit-d code.
- **Edge share** (Part S): 1 − Δg(knocked out) / Δg, with
  Δg = g_clean − g_corrupted.

## Estimands

- **PH1** (H0, all rows): R(`E:others`) and R(`E:r1`), and their paired
  difference. This asks which encoder slots carry the replaced item. It
  re-tests exp8 with sufficiency instead of knockout.
- **PH2** (Hd, matching rows, pooled over d): the same.
- **PH3** (H0 and Hd matching rows): the block profile of R(`X:L{l}:p{query}`)
  for l = 0…3. It says which block's cross-attention brings the history
  difference to the predicted digit. R(`X:L{l}:p{other}`) is pooled over the
  other positions, for contrast.
- **PS1** (Sd, pooled over d): the block profile of R(`D:L{l}:p{d−1}`) and
  R(`D:L{l}:p{d}`). The changed position's share falls with l and the query
  position's share rises; the crossing block is where the revealed digit's
  information has moved to the predicted digit.
- **PS2** (Sd, pooled over d): the edge share of `K:all`.

Trends: every estimand per cell and per d. Widths are compared within the nine
quantizer × depth pairs. Descriptive.

## Numerical acceptance (all must pass before an estimand counts)

- **V1**: exp1's pilot checks on 8 users.
- **V2**: denoising `E:all` reproduces the clean scores bit for bit, on every
  Part H row.
- **V3**: a self-patch (clean values into the clean run) at every `D` and `X`
  site of block 1 is bit-exact on the first chunk.
- **V4**: patching every position after block 3 with the clean run's values
  reproduces the clean scores bit for bit, on the first chunk of every part.
- **V5**: the checkpoint sha256 equals the registry, and the cohort hashes
  equal the loader's.
- **Reference, not a gate**: for the two matched cells, H0's mean Δg is
  reported beside exp7's A1 (0.556 / 0.476). It is the same draw, but exp7 ran
  on CPU.

## Statistics

Paired user bootstrap (`UserBootstrap`), 2,000 draws, seed 20261014. Every row
and site of a user stays together. Ratios are ratios of pooled sums,
resampled as such. Intervals exclude training-seed uncertainty: there is one
checkpoint per cell.

## Expectations recorded before outcomes

- **E1** (PH1): R(`E:others`) > R(`E:r1`) in both matched cells. That is
  exp8's spread, read as sufficiency.
- **E2** (PH2): the same for matching rows in at least 15 of 20 cells.
- **E3** (PH3): no single block carries more than 0.6 of the query position's
  restoration in either matched cell. The read is distributed over blocks, in
  line with exp7's per-block knockouts, each of which was near 0.
- **E4** (PS1): the query position restores at least 0.5 by block 1 (the
  second block) in at least 15 of 20 cells: digit-to-digit transfer is early.
- **E5** (PS2): the edge share of `K:all` is below 0.5 in at least 15 of 20
  cells. Other positions relay the information too. No MQ vs RQ direction is
  predicted (see representation/exp4 S6).

## Interpretation limits

- The states are fixed reveal states, not the decoding path (exp13 covers the
  path).
- A replaced item changes the history's content and its encoder interactions
  at once. Restoration is a share of KL, not a share of HR.
- Sublayer patches (`X`) and residual patches (`D`) answer different
  questions: what one block read, versus everything accumulated so far.
- One checkpoint per cell; widths also differ by SID table.
