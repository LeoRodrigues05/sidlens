# AR history interventions and residual patching: completed 2026-09-27

The two frozen next-item AR recommenders lean heavily on the **most recent**
history item, and mostly through its **first SID digit**:

- **Effect size.** Replacing that item with a catalogue item that has a
  different first digit lowers the golden first digit's teacher-forced
  log-prob by **0.57 nats** [0.53, 0.62] for RQ-KMeans 3×128 and **0.51 nats**
  [0.47, 0.55] for RQ-VAE 4×128.
- **Top-1 flips.** The replacement changes the model's top-1 first digit in
  75% and 50% of rows respectively.
- **Recency.** The second most recent item has 28% (RQ-KMeans) and 43%
  (RQ-VAE) of that effect (0.16 and 0.22 nats); items 6–10 back have ≤0.03.
- **The first digit carries most of it.** When the control keeps the item's
  first digit (m = 1), 82% (RQ-KMeans) and 64% (RQ-VAE) of the digit-1 effect
  disappears.

Activation patching places the transfer late in the network. For digit 1,
the replaced item's own token positions hold essentially all of the effect
(recovery ≥ 0.95) through layer 20 (RQ-KMeans) or 16 (RQ-VAE). It then moves
into the response header, which reaches ≥ 0.5 recovery at layer 21 or 20.
For later digits the item signal fades more gradually from mid-depth, and the
bulk arrives in the teacher-forced target tokens at layers 19–21. The tokens
between the item and the header carry none of it at any layer.

Cohort: all 3,681 historical AR next-item test rows (1,606 users), eval
wording, teacher forcing, bf16. Protocol declared before any intervened
forward: [protocol.md](protocol.md).

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp3_ar_history_patching/` |
|---|---|---|
| Pilot, 64 rows (validation and timing only) | 280955 | `pilot-280955/cell-0{0,1}` |
| **Primary**: eval wording, Parts A + B | 280961 | `280961/cell-0{0,1}`, `summary-280961/` (figures redrawn in `figures-r2/`) |
| Declared secondary: sft wording, Part A | 280962 | `280962/cell-0{0,1}`, `summary-280962-r2/` |
| Failed attempts, kept as records | 280939, 280945 | `pilot-280939`, `pilot-280945` (bad nodes `gpu-54`, `gpu-05`), `summary-280962.failed` (figure bug) |

## Numerical acceptance (all passed, full cohort)

| Check | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|
| Inputs: checkpoint sha256 = manifest; rows = archive (3,681/3,681); SIDs = `.sem_ids` | pass | pass |
| V1 no-op patch (positions before the item), 4 layers × 3,681 | 14,724 / 14,724 bit-exact | 14,724 / 14,724 |
| V2 full-layer restore = clean run | 14,724 / 14,724 bit-exact | 14,724 / 14,724 |
| V3 self-patch = corrupted run | 14,724 / 14,724 bit-exact | 14,724 / 14,724 |
| V4 batch invariance (clean and corrupted rows, Part A vs Part B batches) | max \|Δ\| = 0 | max \|Δ\| = 0 |
| V6 bf16 vs fp32, 16 rows (max \|Δ log p\|; top-1 agreement) | 0.10 nats; 100% | 0.07 nats; 98.4% |
| Independent path: exp3 clean scores vs `ar_capture` 281053/281054 | median \|Δ\| 0, rank agreement 99.94% | median 0, 99.97% |

## Part A: replace one history item

Δ is the golden code's log-prob in the clean run minus that in the
intervened run (nats, higher = more harm). Intervals are paired user-bootstrap
95% percentile intervals (2,000 draws, seed 20260927).

**A1: the most recent item, different first digit (m = 0)**

| | RQ-KMeans digit 1 | 2 | 3 | RQ-VAE digit 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| Δ log p | **+0.570** [0.526, 0.615] | +0.113 [0.091, 0.135] | +0.040 [0.029, 0.053] | **+0.511** [0.472, 0.552] | +0.246 [0.218, 0.275] | +0.049 [0.033, 0.067] | +0.042 [0.013, 0.079] |
| top-1 flip rate | 0.75 | 0.21 | 0.08 | 0.51 | 0.39 | 0.10 | 0.04 |

Whole-SID Δ (sum over digits): RQ-KMeans +0.72 [0.67, 0.78], RQ-VAE +0.85
[0.78, 0.93]. Later-digit effects are small partly because teacher forcing
conditions them on the correct prefix.

**A2: recency** (digit 1, m = 0; figure `fig1_recency`)

| Recency | 1 | 2 | 3 | 4 | 5 | 6–10 |
|---|---|---|---|---|---|---|
| RQ-KMeans | 0.570 | 0.161 | 0.099 | 0.048 | 0.032 | 0.009–0.031 |
| RQ-VAE | 0.511 | 0.222 | 0.101 | 0.052 | 0.028 | 0.009–0.026 |

**A3: shared prefix** (recency 1; figure `fig2_prefix_sharing`). The digit-1
effect retained when the control shares m leading digits with the replaced
item:

| m | RQ-KMeans | RQ-VAE |
|---|---|---|
| 1 | 0.175 [0.147, 0.204] | 0.356 [0.313, 0.402] |
| 2 | 0.040 [0.024, 0.057] | 0.181 [0.141, 0.223] |
| 3 | — | 0.058 [0.026, 0.096] |

On rows that have a control at every m, the m = 1 retention is 0.108
(RQ-KMeans) and 0.189 (RQ-VAE). Deep controls are often unavailable, and
every missing control is counted:

- RQ-KMeans, m = 2: 5,431 of 17,949 item slots
- RQ-VAE, m = 2: 4,987
- RQ-VAE, m = 3: 13,368, because its last digit is nearly an identity digit

**Unexpected, and declared as such:** changing only digits ≥ m+1 of the most
recent item hurts output digit m+1 *more* than replacing the whole item.
Retained fractions are:

- RQ-KMeans: 1.63 for digit 2 at m = 1, and 1.88 for digit 3 at m = 2
- RQ-VAE: 1.03, 1.26 and 1.97

**Secondary, sft (training) wording:** the same pattern with slightly larger
effects. Digit-1 A1 is +0.600 [0.551, 0.651] for RQ-KMeans and +0.640
[0.591, 0.691] for RQ-VAE. The m = 1 retentions are 0.21 and 0.35.

## Part B: residual-stream patching

- **Corrupted input:** the most recent item replaced (the A1 condition).
- **Patch:** the clean run's residual stream at one layer (0–27) and one
  position group.
- **Recovery:** Σ(patched − corrupted) / Σ(clean − corrupted), bootstrapped by
  user.
- **Figure:** `summary-280961/figures-r2/fig3_patching_recovery`.

| Where the evidence is | RQ-KMeans | RQ-VAE |
|---|---|---|
| Digit 1: item tokens hold it (recovery ≥ 0.95) through | layer 20 | layer 16 |
| Digit 1: response header recovers ≥ 0.5 from | layer 21 (0.62; 0.90 by L24) | layer 20 (0.59; 0.86 at L21) |
| Digits ≥ 2: target tokens reach ≥ 0.5 at | L21 for digits 2 and 3 (digit 2: 0.56 → 0.96 by L24) | L21 / L20 / L19 for digits 2 / 3 / 4 (digit 2: 0.55 → 0.89 by L24) |
| `between` tokens (separators, "\n\n") | ≈ 0 at every layer and digit | ≈ 0 |

For digits ≥ 2, the item tokens start losing recovery earlier. They stay
≥ 0.95 only through layer 10 (RQ-KMeans digit 2), layer 5 (RQ-KMeans digit 3)
and layer 13–16 (RQ-VAE); RQ-KMeans digit 3 is down to 0.78 at layer 9. A small, early share appears in the target
tokens (0.1–0.2 in layers 6–18), and the bulk arrives in layers 19–24, as for
digit 1. Intervals for digits 3–4 are wide because the corruption moves them
only 0.04–0.05 nats.

**Expectations recorded before outcomes**

- **E1 (recency decay):** held.
- **E2 (the first digit carries most of the digit-1 effect):** held, more
  strongly for RQ-KMeans than for RQ-VAE.
- **E3 (the item-to-readout crossing is "in the middle layers"):** held in
  shape but not in location. The crossing is late: layers 20–22 of 28.

## Post-hoc exploration: not in the protocol

The A3 surprise was stratified by a pre-treatment variable: whether the most
recent item already shares the target's first m digits. Δ on output digit
m+1:

| Stratum | RQ-KMeans m=1 → digit 2 | m=2 → digit 3 | RQ-VAE m=1 → digit 2 | m=2 → digit 3 | m=3 → digit 4 |
|---|---|---|---|---|---|
| Item prefix = target prefix | **+0.99** (n=602) | **+0.71** (247) | **+0.73** (1,068) | **+0.43** (352) | **+0.68** (121) |
| Item prefix ≠ target prefix | +0.02 (3,051) | +0.01 (2,285) | +0.06 (2,613) | +0.01 (2,237) | +0.00 (888) |

Almost the entire later-digit effect sits in rows where the history item
matches the target on the prefix already generated. Read as a hypothesis,
this points to a **prefix-matched copy mechanism**: to choose digit m+1, the
model reads digit m+1 of history items whose first m digits match the prefix
it has produced.

Also, in 348 of 3,681 RQ-KMeans rows (9.5%) the most recent item's full SID
equals the target's (repeats plus collisions).

These numbers carry no intervals and were not declared. They motivate a
declared follow-up (below), not a claim.

## Limits

- **Scores:** teacher-forced conditional digit scores, not beam-search HR@10.
- **Intervals:** exclude training-seed variance (one seed per checkpoint) and
  control-draw variance.
- **Controls:** uniform over the catalogue, not frequency- or
  category-matched. A replacement effect mixes removing the item's evidence
  with adding the control's.
- **Quantizer contrast:** the two cells differ in quantizer *and* depth (3
  vs 4 digits), so any difference between them is descriptive.
- **Patching** localizes the information by layer and position group. It
  does not identify the heads or MLPs that move it.

## Next experiments this suggests

1. **Declared test of prefix-matched copying.** Path-patch or ablate
   attention from target digit positions to history digit positions,
   stratified by prefix match. Include heads in layers 18–24.
2. **Head-level localization** of the layer 19–24 transfer (item → header and
   item → target).
3. **Category-matched controls** once `external/labels` is on this cluster.
4. **Next-two version** on `two-item_best` (MQ 4×256): clamp item 1 and
   measure item 2.
