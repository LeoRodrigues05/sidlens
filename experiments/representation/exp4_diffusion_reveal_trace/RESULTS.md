# DiffGRM reveal trace: provisional results, 16 of 20 cells (2026-10-13)

**Provisional.** Four cells stopped at the archive check (V6) and are not in
these numbers; the three expectations cannot be judged until all 20 are in.
Figures and the per-user viewer are not built yet.

DiffGRM does not settle a recommendation in one shot. Scoring the catalogue
from the fully masked state alone gives SID HR@10 that is **2.8 points lower,
on average, than the beam-256 list** (lower in all 16 cells; 16.3 % vs 20.3 %
on RQ-KMeans 3×128). The single greedy path is far worse than the beam (3.5–9.8 %
vs 8.7–20.6 %). Which digit is revealed first depends on the quantizer, most
decode states at depth 5 were never visited in training, and by the last step
92 % of a depth-5 beam is duplicate paths.

Protocol: [protocol.md](protocol.md), declared before any forward pass.
Observational; nothing here shows what the model uses.

| Run | Job | Output under `$SIDLENS_WORK/derived/representation/exp4_diffusion_reveal_trace/` |
|---|---|---|
| Pilot, 64 users, cells 7 and 16 | 310755 | `pilot-310755/` |
| Cancelled seconds after start, no results | 310758 | `310758.cancelled/` |
| **Primary**, 20 cells in 5 blocks | 310761, 310769, 310775, 310782, 310789 | `310761/cell-00…19` (16 complete, 4 failed at V6) |
| V6 diagnosis of the 4 failed cells | 310805 | `310761.v6-diagnosis/` |
| Summary of the 16 complete cells | login node | `summary-310761-16cells/` |

Cohort: 6,297 users, the diffusion next-item cohort. GPU: A100. Intervals are
user-bootstrap 95 % intervals (2,000 draws, seed 20261013) and exclude
training-seed variance: one checkpoint per cell.

## Numerical acceptance

- **V1–V5** passed in all 20 cells: exp1's pilot checks; the observer changes
  nothing; re-scored path states reproduce the search's path scores (within
  5e-6 on the 16 saved cells); the last block's readout is the model's distribution; the
  target's beam score never exceeds its lattice best-order score.
- **V6** (exact match to controlled exp1's archive at beam 64 and 256): passed
  in 14 of the 18 archived cells; not applicable to cells 18 and 19.
- **V6 failed in four cells, all width 512**: MQ 5×512, RQ-KMeans 3×512,
  RQ-VAE 4×512, RQ-VAE 5×512. `archive_diff.py` re-decoded them:

  | Cell | Users with a different list (beam 64 / 256), of 6,297 | Kind | Score spread of the entries involved | Target rank changed | SID HR@10 vs archive |
  |---|---|---|---|---|---|
  | MQ 5×512 | 0 / 1 | another SID at the list boundary | 0 | 0 users | identical |
  | RQ-KMeans 3×512 | 2 / 2 | same SIDs, other order | 1.7e-6 | 0 users | identical |
  | RQ-VAE 4×512 | 0 / 1 | another SID at the list boundary | 4.8e-7 | 0 users | identical |
  | RQ-VAE 5×512 | 0 / 1 | same SIDs, other order | 0 | 0 users | identical |

  The archive was decoded on an RTX 5000 Ada; this run used A100s. Scores at
  identical entries differ by at most 1.1e-5. The differences are ties broken
  differently on other hardware. Under the declared V6 these cells still
  fail, and their traces were not saved. Accepting them needs an amendment to
  V6, which has not been made.

## Declared statistics (16 cells)

**S1, reveal order.** Share of users whose greedy path reveals digit 1 first:
RQ-VAE 45–81 %, RQ-KMeans 29–40 %, MQ 17–29 %. Chance is 33 / 25 / 20 % at
depth 3 / 4 / 5, so RQ-KMeans at depth 3–4 and MQ are close to no preference.
RQ-KMeans 3×128 reveals digit 2 first most often (48 %). The modal full order
covers 20–58 % of users at depth 3 and 4–7 % at depth 5: the order is
user-specific.

Share of greedy-path states inside the family training visited: 60 % at depth
3, 44 % at depth 4, **11 % at depth 5** (where the fully masked start is never
in it).

**S2, commitment.** The one-shot guess at the fully masked state already
equals the greedy path's final SID for 29 % of users at depth 3, 21 % at depth
4 and 7 % at depth 5.

**S3, catalogue ranking.** State 0 against the final list is the comparison
above. *The later states are not comparable to a final list*: a path's later
state has committed digits, and the target is unreachable from it for most
users (on RQ-KMeans 3×128, for 20 % after one commit and 12 % after two). I had
declared S3 for every state; only state 0 answers "what does iterating buy".
Item-level bounds at state 0 on RQ-KMeans 3×128: 10.1–12.6 % against 16.3 %
at SID level; RQ-VAE cells have no gap.

**S4, beam make-up at beam 256.** Duplicate share of kept paths by step:
depth 3: 0 / 28 / 58 %; depth 4: 0 / 26 / 58 / 79 %; depth 5: 0 / 21 / 54 /
81 / 92 %. Paths no catalogue SID completes: 24 / 25 / 11 % at the last step.

**S5, decoder blocks.** A masked digit's top code after block 1 / 2 / 3 / 4
equals the final one in 36–71 / 56–81 / 73–87 / 100 % of cases. MQ settles
earliest, RQ-VAE latest.

**S6, target lattice.** Revealing one target digit raises the target code's
log-probability at another digit in every cell and digit pair (0.13 to 4.3
nats). The mean over digit pairs is 0.65–1.96 nats in the width-128 cells and
2.16–3.85 in the width-512 cells; a wider codebook starts from lower
probabilities, so widths should not be compared with each other. At width 128
and depth 3 / 4 / 5 the mean is 1.15 / 1.36 / 1.96 for MQ, 0.83 / 0.65 / 1.05
for RQ-KMeans and 1.50 / 0.78 / 1.30 for RQ-VAE. This is observational and the
states are teacher-forced, but MQ is not weak, which is what I had pencilled
as the prediction for the activation-patching experiment (strong
digit-to-digit flow for RQ, weak for MQ); that prediction must not go into
exp13's protocol as written. The target's best reveal order differs from its
greedy order for 66–95 % of users.

**S7, regret.** 0–14 % of first-ranked SIDs have a digit that is no longer the
top code when re-masked.

**S8, the fully masked start.** Mean max probability minus top-1 accuracy:
0.05–0.19. By depth on the complete triples: MQ ×128 0.045 / 0.096 / 0.111,
RQ-KMeans ×128 0.072 / 0.103 / 0.126, RQ-VAE ×128 0.144 / 0.084 / 0.190.

## Expectations

Not judged: they are declared over all 20 cells.

- **E1** (digit 1 first more often on RQ than on the matching MQ cell): true
  in all 8 comparisons available so far.
- **E2** (state-0 HR@10 below the final list in at least 15 cells): already
  true in 16.
- **E3** (S8 largest at depth 5 in at least 4 of 6 quantizer × width pairs):
  true in the 3 complete ×128 triples; already false for RQ-KMeans ×512
  (0.142 at depth 4, 0.092 at depth 5); two pairs incomplete.

## Limits

- Sixteen cells; the four missing ones are all width 512.
- Two paths per user, not the whole beam.
- The block-wise readout and the lattice are descriptions, not interventions.
- One seed per checkpoint.
