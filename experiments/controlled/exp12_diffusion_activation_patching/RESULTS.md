# controlled/exp12: paired-run activation patching in DiffGRM (2026-10-14, all 20 cells)

Two runs of one checkpoint differ in one input. In Part H the most recent item
r1 is replaced (exp7's controls). In Part S the revealed parent digit holds a
sibling's code. Patches go both ways between the runs.

**Findings.**

1. **The replaced item reaches the decoder mostly through the other history
   slots.** Restoring the clean encoder states of every other slot recovers
   **0.83–0.90** of the first-digit effect (KL) in all 20 cells, and
   **0.83–0.96** in copy rows. Restoring r1's own slot recovers only
   **0.32–0.53** and **0.35–0.57**. The second most recent slot alone
   recovers as much as r1's own slot (0.36–0.52). Noising gives the same
   picture: the replaced history in the other slots alone reproduces
   **0.47–0.68** of the damage, and in r1's own slot only **0.10–0.17**.
   exp8 found the encoder spread with knockouts on two cells; here it is found
   by sufficiency and necessity, on every quantizer, depth and width.
2. **Each digit position reads the history through its own cross-attention,
   spread over the four blocks.** Restoring one block's cross-attention output
   at the predicted digit recovers up to 0.34–0.77 of the effect (the best
   block, depending on the model). Some blocks recover almost nothing, and the
   four together recover 0.70–1.49, so the blocks overlap. No block carries
   more than 0.6 of it in the matched cells. The same patch at any other digit
   position recovers at most 0.23 in block 1 and at most 0.10 in later blocks. At the fully masked state, block 1 reads little (≤ 0.13) in five
   cells (MQ 5×256 and 5×512, RQ-KMeans 5×512, RQ-VAE 3×256 and 4×512) and a
   later block takes over. In RQ-VAE 5×512 only block 1 reads (0.57; the
   others ≤ 0.07).
3. **A revealed digit's information moves to the predicted digit in the first
   decoder block, mostly over the direct self-attention edge.** After block 1
   the predicted position's residual restores **0.65–0.96** of the effect of
   changing the parent digit, and after block 2 **0.91–0.99**. Cutting the
   edge from the predicted digit to the parent digit removes **0.37–0.89**
   of the effect, almost all of it in block 1 (block 1 alone: 0.36–0.88;
   blocks 2–4 ≈ 0). Single heads carry 0.05–0.62 each, with overlapping
   shares.
4. **Trends.**
   - *By depth.* The direct-edge share falls with depth (depth 3:
     0.68–0.89; depth 5: 0.37–0.66). At depth 5 it is lowest for the first
     predicted digit, whose parent is digit 1 (0.21–0.55; later digits
     0.63–0.83). r1's own slot matters less at depth 5 (0.32–0.46 vs
     0.46–0.52 at depth 3).
   - *By quantizer.* RQ-KMeans keeps more of the changed digit at its own
     position after block 1 (0.17–0.47 vs MQ 0.09–0.20).
   - *By width.* The encoder results barely change with width.
5. **Cross-check with exp7.** The first-digit cost of the replacement here is
   0.56 / 0.48 nats on the matched cells; exp7's A1 was 0.556 / 0.476, from
   the same control draws.

The protocol ([protocol.md](protocol.md)) was declared before any forward pass.
Cohort: 6,297 users. Intervals: paired user bootstrap, 2,000 draws, seed
20261014. They exclude training-seed uncertainty, since each cell has one
checkpoint.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp12_diffusion_activation_patching/` |
|---|---|---|
| Pilot, 300 users, cells 6 and 14 | 312753 | `pilot-312753/` |
| Primary, cells 0–4 / 5–9 / 10–14 | 312754 / 312755 / 312756 | `312754/`, `312755/`, `312756/` |
| Primary, cells 15–19 (5 cells in parallel) | 312806 | `312806/` |
| A job cancelled before it started (cells 15–19, resubmitted as 312806) | 312757 | none |
| **Summary** | login node | `summary-312754-312806/` (`estimates.csv`, `profiles.csv`, `report.md`) |

## Deviation from the protocol

The protocol says GPU. All runs used CPU (`cscc-cpu-p`, 32 threads per cell,
fp32), because the GPU queue was full on 2026-10-14. No estimand depends on
the device. Every bit-exact control passed on CPU (below).

## Numerical acceptance (all 20 cells)

- **V1**: exp1's pilot checks passed.
- **V2**: denoising every encoder slot reproduced the clean scores bit for bit
  on every Part H row.
- **V3**: self-patches of block 1 (residual and cross-attention, every
  position) were bit-exact.
- **V4**: patching every position after block 4 with clean values reproduced
  the clean run bit for bit, in both parts.
- **V5**: checkpoint and cohort hashes matched.
- **Missing controls** are in `missing.csv` per cell. They are mostly deep
  prefix levels: no catalogue item shares r1's first d digits and differs at d.

## Declared estimands (denoising, share of KL restored)

| Cell | PH1 R(other slots) | PH1 R(r1's slot) | PH2 R(other slots), copy rows | PH2 R(r1's slot), copy rows | PS2 edge share |
|---|---|---|---|---|---|
| MQ 3×128 | 0.85 | 0.51 | 0.93 | 0.57 | 0.89 |
| MQ 3×512 | 0.86 | 0.52 | 0.93 | 0.51 | 0.86 |
| MQ 4×128 | 0.84 | 0.47 | 0.83 | 0.47 | 0.76 |
| MQ 4×512 | 0.83 | 0.48 | 0.83 | 0.35 | 0.85 |
| MQ 5×128 | 0.87 | 0.38 | 0.93 | 0.55 | 0.52 |
| MQ 5×512 | 0.90 | 0.32 | 0.96 | 0.48 | 0.37 |
| RQ-KMeans 3×128 | 0.83 | 0.50 | 0.94 | 0.51 | 0.68 |
| RQ-KMeans 3×512 | 0.83 | 0.48 | 0.93 | 0.53 | 0.72 |
| RQ-KMeans 4×128 | 0.87 | 0.48 | 0.93 | 0.49 | 0.66 |
| RQ-KMeans 4×512 | 0.88 | 0.45 | 0.94 | 0.53 | 0.66 |
| RQ-KMeans 5×128 | 0.89 | 0.38 | 0.94 | 0.48 | 0.62 |
| RQ-KMeans 5×512 | 0.85 | 0.46 | 0.91 | 0.53 | 0.48 |
| RQ-VAE 3×128 | 0.84 | 0.50 | 0.92 | 0.48 | 0.82 |
| RQ-VAE 3×512 | 0.83 | 0.52 | 0.92 | 0.56 | 0.80 |
| RQ-VAE 4×128 | 0.84 | 0.53 | 0.90 | 0.55 | 0.75 |
| RQ-VAE 4×512 | 0.86 | 0.48 | 0.92 | 0.53 | 0.78 |
| RQ-VAE 5×128 | 0.88 | 0.38 | 0.88 | 0.45 | 0.66 |
| RQ-VAE 5×512 | 0.84 | 0.37 | 0.88 | 0.47 | 0.55 |
| MQ 5×256 | 0.89 | 0.41 | 0.91 | 0.55 | 0.55 |
| RQ-VAE 3×256 | 0.84 | 0.46 | 0.91 | 0.49 | 0.82 |

The 95 % intervals are in `estimates.csv` and `report.md`. Their half-width
is at most 0.10 for the slot restorations (the widest are copy rows of
small cells) and 0.01 for the edge share. The PH1 and PH2 differences (other
slots − r1's slot) exclude 0 in every cell; the smallest lower bound is 0.29. Block profiles (PH3, PS1), per-digit breakdowns, noising,
golden-log-prob versions and per-head knockouts are in `estimates.csv` and
`profiles.csv`, and are drawn in the report
(`docs/results/diffusion_mechanisms/index.html`).

## Expectations recorded before outcomes

- **E1** (R(other slots) > R(r1's slot) on the first digit, both matched
  cells): **held**. 0.83 vs 0.50 and 0.84 vs 0.53.
- **E2** (the same in copy rows, in ≥ 15 of 20 cells): **held**, 20 of 20.
- **E3** (no single block carries > 0.6 of the first digit's restoration, in
  the matched cells): **held**. Maximum single block 0.50 / 0.55.
- **E4** (the predicted position restores ≥ 0.5 by block 2, in ≥ 15 of 20):
  **held**, 20 of 20 (0.91–0.99). It already holds at block 1 (0.65–0.96).
- **E5** (edge share < 0.5 in ≥ 15 of 20): **failed**. The share is below 0.5
  in only 2 of 20 cells (MQ 5×512, RQ-KMeans 5×512). The direct edge in
  block 1 carries most of a revealed digit's effect. No MQ vs RQ direction
  was predicted. MQ has the highest edge shares at depth 3–4 and the lowest at
  depth 5.

## Limits

- Fixed reveal states (fully masked; golden prefix), not the decoding path
  (see exp13).
- Restoration is a share of KL at one digit, not of HR.
- Restoration shares of single sites can sum to more than 1 (redundant routes),
  so they are not a decomposition.
- One checkpoint per cell. Widths also differ by SID table.
