# controlled/exp14: perturbations held through DiffGRM decoding (2026-10-15, 19 of 20 cells)

Three perturbations are held for the whole beam search: a copy block, an order
swap, and a position shift. Lists are read at beam 256 (primary) and beam 64.
MQ 5×512 failed the archive gate and is excluded.

**Findings.**

1. **Copying the most recent item helps DiffGRM's lists, and only for items the
   user already had.** exp8's `both` block on r1 lowers SID HR@10 by
   **1.0–3.7 points** more than the same block on a control item. The interval
   is below 0 in all 19 cells. The control block alone moves HR@10 by −0.4 to
   +0.5 points. At item level the drop is 1.0–2.9 points (both bounds). By
   target relation (beam 256, SID level):

   | Target relation | Share of users | ΔHR@10 from the r1 block |
   |---|---|---|
   | r1's own item (repeat purchase) | 4.0 % | **−11.5 to −38.9 points** (19/19 below 0) |
   | another item with r1's SID | 0–7.2 % | 0 to −14.9 (18/19 below 0) |
   | shares r1's first code only | 3.5–27.5 % | −3.9 to −20.4 (19/19 below 0) |
   | **new** | 65.7–88.8 % | **−0.7 to +0.4** (1/19 below 0) |

   This is the AR models' pattern (exp6: −5.7 / −3.3 points; retrospective
   exp7: the AR copy benefit on new targets vanishes at item level). Copying in
   DiffGRM serves re-purchases and near-duplicates; it neither helps nor hurts
   new targets.
2. **DiffGRM follows the item, not the slot.** When r1 and r2 swap places, the
   top-1 SID changes for 27–66 % of users and HR@10 falls by 0–2.4 points. Take
   the users whose clean top-1 first code copied r1:
   - after the swap, **39–86 %** still output r1's code;
   - only **0–18 %** switch to the code of the item now in the most recent slot.

   Users who copied r2 keep r2's code in 69–96 % of cases. The AR RQ-KMeans
   model (exp9) did the opposite and followed the position. These are post hoc
   measures; see the deviation below.
3. **Absolute positions matter a great deal.** Moving the whole history right,
   with the order and the padding count kept, changes the top-1 SID for:
   - **47–82 %** of users when moved five slots;
   - 34–69 % when moved one slot.

   The five-slot shift costs 0.5–3.5 points of HR@10, and moves the fully
   masked first-digit distribution by 0.28–1.30 nats of KL. The encoder reads
   absolute slot positions, not only order. Training always filled slot 1
   first, so part of this effect is the novelty of leading padding (stated as
   a limit in the protocol).
4. **Trends.**
   - *By width.* The copy-block cost is similar at widths 128 and 512 (PB
     −1.0 to −3.7). The shift is more disruptive at width 512 for MQ 3
     and RQ-KMeans 3–4.
   - *By depth.* RQ-KMeans 5×512, the Sinkhorn-balanced table, barely copies
     (PB −1.0, repeat −11.5, follow-position 0.00): its first codes carry less
     neighbour agreement (structure exp3).

The protocol ([protocol.md](protocol.md)) was declared before any forward pass.
Cohort: 6,297 users; 6,297 eligible for the r1 block, 5,717–6,232 for the
control (depends on the SID table), 6,294 for the five-slot shift. Intervals: paired user
bootstrap, 2,000 draws, seed 20261014. They exclude training-seed uncertainty,
since each cell has one checkpoint.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp14_diffusion_decode_perturbations/` |
|---|---|---|
| Pilot, 256 users, cells 6 and 14 (CPU) | 312767 | `pilot-312767/` |
| CPU copy, cancelled when the GPU run started (no cell finished) | 312804 | `312804.cancelled-cpu-hedge/` |
| **Primary**, cells 0–12, A100 (cancelled at the start of cell 13 to avoid a duplicate) | 312775 | `312775/` |
| Reverse-order run that hung in `pip freeze` on gpu-02 (no cell started) | 312973 | `312973.cancelled-pip-freeze-hang/` |
| **Primary**, cells 13–16, A100 | 313047 | `313047/` |
| Cells 17–19 on gpu-52, no CUDA device (nothing computed) | 313048 | `313048.failed-gpu52-no-cuda/` |
| **Primary**, cells 17–19, A100 | 313104 | `313104/` |
| **Summary** | login node | `summary-312775-313104/` (`estimates.csv`, `report.md`) |

## Numerical acceptance

- **V1**: exp1's pilot checks passed in every cell.
- **V2**: the masked-score decode path with nothing blocked, plus the
  padding-only encoder mask, equals the plain decode bit for bit (codes,
  scores, counts) at both beams, in every cell.
- **V3**: the clean decode matches controlled exp1's archive at beams 64 and
  256 in 16 of 17 archived cells:
  - RQ-KMeans 3×512 has 2 users and RQ-VAE 4×512 and 5×512 1 user each that
    differ only by tied scores, within the declared tolerance. exp4 found the
    same ties.
  - **MQ 5×512 failed V3.** At beam 256, user 5355's archived list has 7
    legal SIDs; this run's has 6. The missing SID (score −4.136) lost its path
    at an intermediate beam cutoff, and the declared tolerance covers only ties
    in the final list. Beam 64 matches exactly. The cell is excluded. Its
    decodes are kept for diagnosis and were not read.
  - Cells 18 and 19 have no archive.
- **V4**: checkpoint hashes matched.
- **Guards**: no padded slot was blocked; every shifted history kept its items,
  order and padding count.

## Deviation from the protocol: PO1 is mis-scaled

The protocol read PO1, [P(first code = r2's) − P(= r1's)] under the swap minus
the same under clean, as +1 for a model that follows the position. That holds
only if the model's top-1 always copies r1. Here its first code is r1's for
only 4–23 % of users and r2's for 5–22 %. A model that followed the position perfectly
would then score near 0, just as a model that ignored the order. PO1 is
reported as declared: +0.01 to +0.19, below 0.5 in all 19 cells. E3 is scored
"failed" mechanically, but **PO1 cannot answer the question**.

The post hoc measures in finding 2 condition on the users whose clean top-1
copied r1 (or r2). They are labelled `posthoc_*` in `estimates.csv`. They were
added after PO1's values were seen, and are not part of the declared design.

## Declared estimands (beam 256)

| Cell | PB SID [95 % CI] | PB item, lower bound | ΔHR@10 repeat | ΔHR@10 new | PO1 | top-1 changed, shift 5 |
|---|---|---|---|---|---|---|
| MQ 3×128 | −2.6 [−3.3, −1.9] | −1.3 | −31.7 | −0.7 | +0.13 | 0.49 |
| MQ 3×512 | −3.2 [−3.9, −2.4] | −2.2 | −38.9 | −0.5 | +0.12 | 0.70 |
| MQ 4×128 | −2.7 [−3.4, −1.9] | −1.7 | −27.8 | −0.1 | +0.12 | 0.67 |
| MQ 4×512 | −2.4 [−3.1, −1.7] | −2.1 | −29.0 | −0.4 | +0.08 | 0.68 |
| MQ 5×128 | −2.4 [−3.0, −1.7] | −1.5 | −23.0 | −0.5 | +0.05 | 0.60 |
| RQ-KMeans 3×128 | −3.7 [−4.5, −3.0] | −2.0 | −34.1 | −0.4 | +0.19 | 0.61 |
| RQ-KMeans 3×512 | −2.8 [−3.6, −2.1] | −2.6 | −32.9 | 0.0 | +0.09 | 0.74 |
| RQ-KMeans 4×128 | −2.6 [−3.3, −1.9] | −2.1 | −29.0 | −0.1 | +0.15 | 0.65 |
| RQ-KMeans 4×512 | −2.5 [−3.3, −1.8] | −2.6 | −34.1 | 0.0 | +0.08 | 0.76 |
| RQ-KMeans 5×128 | −2.0 [−2.7, −1.3] | −2.0 | −29.0 | +0.1 | +0.11 | 0.67 |
| RQ-KMeans 5×512 | −1.0 [−1.5, −0.6] | −1.0 | −11.5 | −0.2 | +0.01 | 0.47 |
| RQ-VAE 3×128 | −3.0 [−3.7, −2.3] | −2.5 | −32.1 | −0.2 | +0.11 | 0.78 |
| RQ-VAE 3×512 | −3.0 [−3.7, −2.2] | −2.9 | −38.9 | 0.0 | +0.16 | 0.78 |
| RQ-VAE 4×128 | −2.9 [−3.7, −2.2] | −2.9 | −37.3 | +0.2 | +0.19 | 0.72 |
| RQ-VAE 4×512 | −2.3 [−3.0, −1.5] | −2.2 | −34.5 | +0.4 | +0.12 | 0.77 |
| RQ-VAE 5×128 | −2.7 [−3.4, −2.0] | −2.7 | −21.0 | −0.4 | +0.09 | 0.82 |
| RQ-VAE 5×512 | −2.7 [−3.5, −2.1] | −2.7 | −26.6 | −0.3 | +0.14 | 0.81 |
| MQ 5×256 | −1.9 [−2.5, −1.3] | −1.3 | −18.7 | −0.3 | +0.04 | 0.60 |
| RQ-VAE 3×256 | −2.9 [−3.6, −2.2] | −2.8 | −32.9 | −0.2 | +0.13 | 0.79 |

HR@10 is in points. Item-level lower bounds for every cell, the upper bounds,
beam 64, the `same_sid` and `same_d0` groups, the swap and the one-slot shift
are in `estimates.csv` and the report (`docs/results/diffusion_mechanisms/`).
For RQ-VAE the item and SID levels nearly coincide: its tables have almost no
collisions.

## Expectations recorded before outcomes

- **E1** (copy block lowers SID HR@10 vs control, interval below 0, both
  matched cells): **held**. −3.7 and −2.9 points.
- **E2** (new targets: ΔHR@10 includes 0 or is positive, in ≥ 15 of 20):
  **held**, 18 of 19.
- **E3** (PO1 > 0.5 in ≥ 15 of 20): **failed** as scored, 0 of 19, but see the
  deviation above. PO1 was mis-scaled, and post hoc DiffGRM follows the item
  (39–86 % keep the copied item's code).
- **E4** (a five-slot shift changes the top-1 for < 20 % of users, in ≥ 15 of
  20): **failed**, 0 of 19 (47–82 %).

## Limits

- The r1 block leaves r1's encoder state in place, unread. It is close to
  removing the item, but not identical.
- Shifted histories start with padding, which training never did.
- 19 of 20 cells: MQ 5×512 is excluded by the declared gate.
- One checkpoint per cell. Widths also differ by SID table.
