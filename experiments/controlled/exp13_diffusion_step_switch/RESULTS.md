# controlled/exp13: at which reveal step does DiffGRM use the history? (2026-10-14, all 20 cells)

The most recent item r1 is replaced with exp7's m = 0 control, but only at
chosen reveal steps of the decode. Every other step uses the real history.
The donor history H′ is used at no step (`none`), at every step (`all`), at
one step (`only_t`), or at every step but one (`but_t`). Greedy decoding and
beam 64, confidence policy.

**Findings.**

1. **The first reveal step decides most, but no single step decides alone.**
   - *Users.* Analysis is restricted to discordant users, whose clean and
     donor top-1 SIDs differ: 56–88 % of users.
   - *First step.* Using H′ only at the first step gives the donor's top-1
     for **16–68 %** of them (beam 64). At the last step it does so for 1–10 %.
   - *Contrast.* F(first) − F(last) is positive, with the interval excluding
     0, in all 20 cells. The F(t) summed over steps is only 0.36–0.80:
     replacing the history at every step moves the answer further than any
     single step does.
2. **Wider codebooks commit at the first step.**
   - *Width trend.* F(first) is larger at width 512 than at 128 in all 9
     quantizer × depth pairs: 0.25–0.68 vs 0.16–0.39.
   - *Quantizer and depth.* The most front-loaded cells are MQ at width 512
     (and 5×256) and RQ-KMeans 5×512: F(first) 0.55–0.68. RQ-VAE is the least: 0.16–0.38.
     Depth matters less than width.
3. **Later steps still use the history.** Switching only one step changes the
   final top-1 for 69–93 % of discordant users at the first step, and still
   for 4–34 % at the last. The history is read at every step. Late steps
   rarely carry the answer all the way to the donor's.
4. **What a switch changes (greedy).** When one step's switch changes the
   greedy SID, it most often changes which digit is revealed at that step,
   not only the code: 38–87 % of changes at the first step. At the last step
   only one position is left.
5. **Hit rate.**
   - *Replacing r1 at every step* lowers SID HR@10 from 5.3–20.5 % to
     2.9–14.7 %.
   - *Replacing r1 at the first step only* recovers part of the loss: for
     example, RQ-KMeans 3×128 goes from 20.3 % (clean) to 17.8 %, against
     13.1 % with every step replaced.

The protocol ([protocol.md](protocol.md)) was declared before any forward pass.
Cohort: 6,297 users. Every user had an m = 0 control. Intervals: paired user
bootstrap, 2,000 draws, seed 20261014. They exclude training-seed
uncertainty, since each cell has one checkpoint.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp13_diffusion_step_switch/` |
|---|---|---|
| Pilot, 256 users, cells 6 and 14 (CPU) | 312766 | `pilot-312766/` |
| **Primary**, 20 cells, A100 | 312774 | `312774/cell-00…19` |
| CPU copy, cancelled before it started (the GPU job had started) | 312805 | none |
| **Summary** | login node | `summary-312774/` (`estimates.csv`, `report.md`) |

## Numerical acceptance

- **V1**: exp1's pilot checks passed in all 20 cells.
- **V2**: the `none` schedule equals the plain decode bit for bit (codes,
  scores, counts) in every batch of every cell. The scorer path changes
  nothing when nothing is switched.
- **V3**: the `all` schedule equals an independent plain decode of H′, bit for
  bit, everywhere.
- **V4**: the plain beam-64 decode matches controlled exp1's archive in all 18
  grid cells. In RQ-KMeans 3×512, 2 users (0.03 %) differ only by tied scores,
  which the protocol's tolerance allows. exp4 found the same two-user tie in
  that cell. Cells 18 and 19 have no archive.
- **V5**: checkpoint hashes matched.

## Declared estimands (beam 64, discordant users)

| Cell | Discordant | F(t), t = 1…n | F(1) − F(n) [95 % CI] |
|---|---|---|---|
| MQ 3×128 | 0.63 | 0.39 / 0.15 / 0.10 | 0.30 [0.28, 0.31] |
| MQ 3×512 | 0.77 | 0.58 / 0.05 / 0.02 | 0.57 [0.55, 0.58] |
| MQ 4×128 | 0.75 | 0.35 / 0.09 / 0.05 / 0.03 | 0.32 [0.31, 0.33] |
| MQ 4×512 | 0.80 | 0.55 / 0.03 / 0.02 / 0.01 | 0.54 [0.52, 0.55] |
| MQ 5×128 | 0.67 | 0.36 / 0.13 / 0.05 / 0.03 / 0.03 | 0.33 [0.32, 0.35] |
| MQ 5×512 | 0.75 | 0.68 / 0.03 / 0.01 / 0.01 / 0.01 | 0.67 [0.65, 0.68] |
| RQ-KMeans 3×128 | 0.71 | 0.31 / 0.19 / 0.05 | 0.26 [0.24, 0.27] |
| RQ-KMeans 3×512 | 0.85 | 0.41 / 0.09 / 0.03 | 0.38 [0.37, 0.40] |
| RQ-KMeans 4×128 | 0.77 | 0.26 / 0.15 / 0.05 / 0.02 | 0.24 [0.23, 0.26] |
| RQ-KMeans 4×512 | 0.87 | 0.37 / 0.08 / 0.04 / 0.02 | 0.35 [0.34, 0.36] |
| RQ-KMeans 5×128 | 0.75 | 0.28 / 0.12 / 0.04 / 0.02 / 0.01 | 0.27 [0.26, 0.28] |
| RQ-KMeans 5×512 | 0.56 | 0.60 / 0.12 / 0.06 / 0.01 / 0.01 | 0.59 [0.58, 0.61] |
| RQ-VAE 3×128 | 0.87 | 0.28 / 0.09 / 0.04 | 0.24 [0.23, 0.25] |
| RQ-VAE 3×512 | 0.88 | 0.33 / 0.10 / 0.03 | 0.30 [0.28, 0.31] |
| RQ-VAE 4×128 | 0.86 | 0.16 / 0.15 / 0.04 / 0.02 | 0.14 [0.13, 0.15] |
| RQ-VAE 4×512 | 0.88 | 0.38 / 0.08 / 0.02 / 0.01 | 0.37 [0.35, 0.38] |
| RQ-VAE 5×128 | 0.86 | 0.22 / 0.10 / 0.05 / 0.03 / 0.02 | 0.20 [0.19, 0.21] |
| RQ-VAE 5×512 | 0.88 | 0.25 / 0.08 / 0.03 / 0.02 / 0.01 | 0.23 [0.22, 0.24] |
| MQ 5×256 | 0.66 | 0.60 / 0.07 / 0.02 / 0.02 / 0.01 | 0.58 [0.57, 0.60] |
| RQ-VAE 3×256 | 0.86 | 0.29 / 0.09 / 0.04 | 0.25 [0.24, 0.26] |

G(t), the share of discordant users whose `but_t` top-1 is still the clean
one, has the same shape. It is 0.20–0.69 at the first step and 0.01–0.10 at
the last. The greedy profiles have the same shape and are smaller. Both are in
`estimates.csv` and drawn in the report.

## Expectations recorded before outcomes

- **E1** (F(first) > F(last), interval excluding 0, in ≥ 15 of 20):
  **held**, 20 of 20.
- **E2** (F(first) < 0.8 in every cell): **held**. The maximum is 0.68.
- **E3** (F(first) − F(last) smaller at depth 5 than at depth 3, in ≥ 4 of 6
  quantizer × width pairs): **failed**, 2 of 6 (only RQ-VAE). The first step
  is no less decisive at depth 5, although depth-5 models never trained on the
  fully masked state. Width, not depth, sets how front-loaded the decision is.

## Limits

- One donor type: the most recent item replaced at shared-prefix level 0.
- At beam 64, a step's switch also changes which paths survive pruning. F
  mixes "changed the commit" with "changed the pruning". Greedy has only the
  first.
- One checkpoint per cell.
