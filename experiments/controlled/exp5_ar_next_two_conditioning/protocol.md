# Next-two AR: is item 2 computed from item 1?

Declared 2026-09-27, before any forward pass of `two-item_best` in this
repository. No training, no checkpoint selection. This is brief RQ4 and
Experiment 6, adapted to the only surviving next-two AR checkpoint.

## Question

Under teacher forcing on "sid1 ||| sid2\n", which is the format the model was
trained on:

- How much does the second item's score depend on the first item's SID,
  compared with the most recent history item?
- At which layers, and through which positions, does item 1's information
  reach the positions that score item 2?

## Fixed design

**Checkpoint.** `two-item_best` (MQ 4×256, model.safetensors sha256
`15f96b80…a8f`). MQ digits are parallel partitions, so a shared "prefix" means
sharing the codes of the first m partitions. It does not mean a coarser
cluster.

**Cohort.** All 3,452 two-item test rows (1,606 users). Rows must equal the
archive (`gt1`/`gt2`) and the frozen `.sem_ids`. There are no exclusions.
Eval wording, `encode(..., with_target=True)`, bf16. The fixed-shape runner
passes an explicit mask on every forward (as in exp4).

**Score.** s_{t,d} is the golden code's log-softmax over digit d's code
tokens at `predict_pos(t, d)`, for slot t ∈ {0, 1}. Effects are
Δ = s(clean) − s(intervened).

### Part A: input interventions

| Condition | Change | Scored |
|---|---|---|
| `item1_m` | target slot 0 (item 1) → a catalogue control sharing exactly m leading digits, m = 0…3 | slot-1 digits |
| `hist1_m0` | most recent history item → control with m = 0 | slot-0 and slot-1 digits |

- **Controls** exclude every history item, both targets, and their SIDs.
- **Seed:** `sha256("20260927|<example_id>|<slot-or-k>|<m>")`.
- **Input check:** each intervened input is rebuilt by `encode`, and must
  differ from the clean one only at the edited SID's digits ≥ m.

**Estimands:**

- **A1 (primary):** mean Δ_{1,0} for `item1_m0`.
- **A2 (primary contrast):** paired mean Δ_{1,0}(`item1_m0`) −
  Δ_{1,0}(`hist1_m0`), on the same rows.
- **A3:** Δ_{1,d} for `item1_m`, m = 0…3, every d.
- **Secondary (copy check):**
  - For d ≥ 1, Δ_{1,d}(`item1_m=d`), split by whether item 1 already shares
    item 2's first d digits (match vs non-match). This is exp4's H1 with item
    1 as the source.
  - Δ_{0,d}(`hist1_m0`) as a reference for how strongly item 1 depends on
    history.

### Part B: residual patching

- **Corrupted input:** `item1_m0`.
- **Clean input:** the original row.
- **Patch:** for each layer L ∈ {0…27} (the output of `model.layers.L`) and
  each position group G, the clean activation at (L, G) is substituted into
  the corrupted run. Groups:
  - `slot0`: item 1's 4 SID tokens
  - `sep`: the separator tokens between item 1 and item 2. The last one is
    `predict_pos(1, 0)`.
  - `slot1`: item 2's SID tokens (the predict positions for digits ≥ 1)

**Recovery** is R_{1,d}(L, G) = Σ[s(patched) − s(corr)] / Σ[s(clean) −
s(corr)], bootstrapped by user.

**Exact controls** at L ∈ {0, 9, 18, 27}:

- a no-op patch of the positions before item 1
- a full-layer restore
- a self-patch

All three must be bit-exact, together with batch invariance.

## Statistics

- Paired user bootstrap, 2,000 draws, seed 20260927, percentile 95%
  intervals.
- Intervals exclude training-seed variance.
- Per-digit and per-layer numbers are descriptive.

## Expectations recorded before outcomes

- **E1:** Item 2 depends on item 1: A1 is > 0 by at least 0.1 nats.
- **No directional expectation** for A2.

## Interpretation limits

- This is teacher-forced dependence on a given item 1. It is not the
  archived two-pass decoding, and not HR.
- A replacement effect mixes removing item 1's evidence with adding the
  control's.
- This is one AR checkpoint. There is no matched diffusion or joint-block
  model, so no paradigm comparison is possible.
- MQ prefixes are intersections of parallel partitions.
