# Plan: how DiffGRM arrives at a recommendation (encoder features, paired-run patching, perturbations)

Written 2026-10-14 at the user's request. It extends
`DIFFUSION_STEP_PATCHING_PLAN.md` (D0, the reveal trace, has run; D1 and D2 had
not been declared). Every experiment below runs on **all 20 next-item
checkpoints**, so each result can be read along two axes the user asked for:
**SID digit** (within a cell) and **codebook width** (the nine quantizer ×
depth pairs at width 128 and 512; the two width-256 cells are extra points).

## What is already known (do not re-derive)

| Source | Finding that shapes this plan |
|---|---|
| exp7 | Replacing the most recent item r1 costs the golden first digit 0.56 / 0.48 nats; prefix-matched copying 0.35 / 0.96 nats; a direct cross-attention read of r1 carries nothing. |
| exp8 | The copy runs through the encoder: other slots read r1 (`enc` 0.17 / 0.30), and the routes back each other up. Only the two matched cells were run. |
| repr. exp4 (D0) | One-shot scoring at the fully masked state is 2.8 points of HR@10 below the beam; reveal order is quantizer-specific; 11 % of depth-5 decode states were visited in training; a revealed target digit raises other target digits by 0.65–3.85 nats (MQ too). |
| controlled exp1, exp2 | Reveal order matters, and depth 5 loses most at beam 64. |

Open: what the encoder states contain (only knockouts so far), where in the
decoder the history difference enters, how one revealed digit reaches another,
at which reveal step a copy is committed, and whether copying helps DiffGRM's
lists (exp6 answered that for AR only). All but the first are known only for
two cells.

## Architecture facts the designs depend on (`vendor/diffgrm/.../DIFF_GRM/model.py`)

- **Encoder.** One token per history item: `item_mlp(concat digit embeddings) + pos_emb_enc(slot)`.
  Slot 0 is the oldest item kept and histories are right-padded, so r1 sits at
  slot L−1. The absolute position embedding does not say which item is most
  recent: the encoder must work it out from which slots are present. One block
  (`ln_1` → self-attention → `ln_2` → MLP), then the **shared** `ln_f`, then
  padding is zeroed. The sites read here: **E0** = block input, **E1** = after
  self-attention, **E2** = encoder output, the only thing the decoder reads.
- **Decoder.** One position per digit, no positional code, four blocks
  (self-attention over digits → cross-attention to the 50 slots → MLP).
  Padded slots have zero K and V and are attended.
- Only the partial SID passes between reveal steps.

## Experiments

| Id | Folder | Answers | Kind | Device |
|---|---|---|---|---|
| X1 | `experiments/representation/exp5_diffusion_encoder_features/` | User item 1: what the encoder holds; probes, SAEs, a description autoencoder (NLA-style) | observational + SAE ablations | GPU |
| X2 | `experiments/controlled/exp12_diffusion_activation_patching/` | User item 2: paired runs, a history item changed (Part H) or a revealed digit changed (Part S); where the difference lives | causal | GPU |
| X3 | `experiments/controlled/exp13_diffusion_step_switch/` | User item 2, across steps: at which reveal step the history is used | causal, decoding | GPU |
| X4 | `experiments/controlled/exp14_diffusion_decode_perturbations/` | User item 3: copy block inside decoding (HR), order swap, position shift | causal, decoding | GPU |

Common to all: the 6,297-user diffusion cohort (never joined to the AR
cohort); fp32, TF32 off; the validated decoder path (`score_states` /
`matched_decode`); exp1's pilot checks (V1) before any condition; no-op
controls bit-exact; paired user bootstrap with 2,000 draws and seed 20261014;
intervals exclude training-seed uncertainty (one checkpoint per cell). SID and
item metrics are kept apart. The replaced-item donor is always exp7's draw
(`condition_rng(20260928, user, k, m)`), so the corrupted runs are exp7's.
Because the CIAI GPU queue holds at most four jobs per user, one job runs the
20 cells in sequence and writes `<job>/cell-XX/` per cell.

### X1. Encoder features (representation/exp5)

**Data.** *Fit split*: every training-target history, i.e. each user's sequence
cut before item t for 2 ≤ t ≤ n−3 (history length ≥ 2, target before the
validation item); about 24k histories and 90k slot rows. *Analysis split*: the
test cohort (36,783 slot rows). Probes, SAEs and the reconstructor are fitted on
the fit split only.

**Part P, linear probes** (multinomial logistic regression; L2 chosen on 10 %
of fit users; test accuracy against the majority class and normalized
information gain 1 − CE/CE_prior):

| Probe | Rows | Sites | What it shows |
|---|---|---|---|
| P1 own SID digit d (every d) | all slots | E0, E1, E2 | whether a slot keeps its item's identity, digit by digit |
| P2 own category (`cat_l2`, `main_cat`) | all slots | E0, E2 | semantic content |
| P3 recency bin (1, 2, 3, 4, 5, 6–10, 11+) | all slots | E0, E2 | whether the encoder computes recency, which E0 does not contain |
| P4 r1's first digit | non-r1 slots whose own first digit differs from r1's | E0, E1, E2 | the spread exp8 found, read directly; E0 is the no-spread baseline |
| P5 target first digit, target `cat_l2` | users | [E·(r1), mean E·] at E0 and E2 | target information beyond the history content, on all rows and on new-target rows |

**Part S, TopK SAEs on E2** (`sidlens.sae.topk`; 2,048 latents, k = 16,
AuxK 128, 60 epochs, batch 1,024, lr 5e-4; 5 % of fit users held out):

- *Fidelity.* Test FVU and dead fraction. Splice: E2 → reconstruction at every
  real slot; KL of the fully masked state's digit distributions against
  mean-ablation; recovered share = 1 − KL_SAE / KL_mean.
- *Census.* Each latent is assigned the label family (own digit d, own
  category, recency, r1's first digit in non-r1 slots, target first digit) with
  the largest chance-corrected selectivity κ = (s − p)/(1 − p), if κ ≥ 0.5 and
  support ≥ 20 on the fit split.
- *Causal (error-preserving ablation).* **CS1**: in every non-r1 slot, ablate
  the active spread latents whose preferred code is r1's first digit. Measure
  the copy score (log p of r1's first digit at the fully masked state) and the
  golden first digit. Control: as many other active latents, nearest in
  activation. **CS2**: the same in r1's own slot with own-first-digit latents.

**Part N, a description autoencoder in the style of NLAs.** Anthropic's NLAs
(Transformer Circuits, 2026) train an LLM verbalizer to write an activation as
text and an LLM reconstructor to map the text back, with RL on reconstruction.
For a 256-wide encoder with no language prior, the version that can be run and
checked here uses a fixed, grounded vocabulary:

- *Verbalizer*: the Part P probes on E2, one word per field: recency,
  `cat_l1`, `cat_l2`, `main_cat`, price decile, own first code, and two
  context fields (r1's `cat_l2` and first code). Rendered as a sentence per slot.
- *Reconstructor*: ridge from the one-hot description to E2, trained on the
  verbalizer's own outputs on the fit split (as the NLA reconstructor reads
  the verbalizer's text). An oracle reconstructor reads the true labels.
- *Measures*: test FVU; a behavioural splice (E2 → reconstruction from the
  description) scored like the SAE splice; field drop (one field set to its
  fit mode): the KL increase says which interpretable word the decoder needs.

Limits stated in the protocol: a grounded vocabulary can find only what we
named; the verbalizer is supervised, not RL-trained. A full LLM NLA is future
work.

### X2. Paired-run activation patching (controlled/exp12)

Two runs of the same checkpoint, identical except for one input.

**Part H, history contrast.** Clean H vs H′:

- **H0**, at the fully masked state, scoring digit 0: H′ replaces r1 with exp7's m = 0 control.
- **Hd**, at the golden-prefix state `S_pref(d)`, scoring digit d: H′ replaces r1 with exp7's m = d control.
  Rows are split by whether r1 matches the target's first d digits.

Patches are taken from the other run, in both directions:

- *Denoising*: the corrupted run with clean pieces. This tests sufficiency.
- *Noising*: the clean run with corrupted pieces. This tests necessity.

| Site | Pieces |
|---|---|
| encoder output E2 | r1's slot; all other real slots; r2's slot; all slots (control: exact) |
| decoder residual after block L = 0…3 | at each digit position |
| decoder cross-attention output of block L | at each digit position (new hook; this is where history enters a digit position) |

**Primary estimand** (per site): restored share
R = 1 − Σ KL(clean ‖ patched) / Σ KL(clean ‖ corrupted) over rows, on the
scored digit's distribution. Secondary: the same with golden log-prob, and
with the copy score.

**Part S, state contrast** (same history; the intermediate representation is
changed). At `S_pref(d)`, d ≥ 1, the revealed parent digit d−1 holds the
golden code in one run and a sibling's code in the other. The sibling is a
catalogue item sharing digits 0…d−2 with the target, drawn with a seeded RNG.
The scored digit is d.

- Patch the decoder residual at the changed position vs at the query position, block by block.
  The layer where the query position starts to restore marks where the revealed digit's information has moved.
- Knock out the self-attention edge from query d to key d−1, in all blocks and in each block, in both runs.
  The edge share is 1 − Δ(knocked out)/Δ.

No direction is predicted for MQ vs RQ: D0's lattice showed strong
digit-to-digit dependence for MQ as well.

**Controls**:

- every self-patch is bit-exact;
- denoising all encoder slots reproduces the clean run exactly;
- patching every position after the last block reproduces the donor.

### X3. Step-level history switch (controlled/exp13; D1 Part A of the earlier plan)

Decode with H′ (exp7's m = 0 donor) at the steps in a schedule and with H
elsewhere: none, all, each single step, and all-but-one. Greedy (beam 1) and
beam 64, confidence policy, batch 32, chunk 1,024.

- **Primary**: among users whose clean and donor top-1 SIDs differ, the share
  whose top-1 equals the donor's when only step t is switched, as a profile
  over t; the contrast is step 0 vs the last step.
- **Secondary**: the share with all-but-t that still equals the clean top-1;
  SID HR@10 of every schedule.
- **Controls**: `none` equals the plain decode and `all` equals an independent
  donor decode, bit for bit.

### X4. Perturbations inside decoding (controlled/exp14)

**Part B, the copy block in the beam** (the DiffGRM version of AR exp6).
exp8's `both` block is held for the whole decode: other encoder slots stop
reading r1, and no decoder position reads r1's slot. The control is the same
block on the most recent item whose first digit differs from r1's. Beam 256
is primary and beam 64 secondary, with the confidence policy.

- *Measure*: SID HR@10 and item-level HR@10 bounds.
- *Splits*: the target is r1's item; the target is another item with r1's SID;
  the target shares r1's first digit; no relation.
- *Primary*: ΔHR@10(block r1) − ΔHR@10(block control), over users with L ≥ 2.

**Part O, order and position** (the analogue of AR exp9):

- *swap*: r1 and r2 change places.
- *shift*: the whole history moves s = 1 and s = 5 slots right, with the left
  slots padded. The order is kept and every absolute position changes. If the
  model reads recency from the end of the history, a shift changes little.

Measures: share of changed top-1 SIDs, KL at the fully masked state, and
ΔHR@10. The baseline decode must match controlled exp1's archive. Up to 0.1 %
of users may differ by ties within 1e-5, which is what the 2026-10-13 A100 runs
showed; this is declared here, before any run.

Deferred: the implicit-step probe and patch, dose–response attention scaling,
negative guidance, and next-two models.

## Order of work

1. X1: protocol, pilot (64 users, 2 cells), full run. While it runs:
2. X2: protocol, pilot, full run.
3. X3 and X4: protocol, pilot, full run.
4. Summaries; `RESULTS.md` per experiment; the experiment index and the
   status table in `CLAUDE.md`.
5. One self-contained HTML report (`docs/results/diffusion_mechanisms/`)
   combining these results with exp1, exp2, exp7, exp8, the reveal trace,
   and the AR comparison points.

## Verification

- New library code is tested on the random-weight `random_diffgrm` fixture:
  the cross-attention output patch (self-patch exact, fire-once, refuses a bad
  site), and encoder internal capture (E1 + MLP = block output; ln_f of block
  output with padding zeroed = `encode`).
- `pytest -q` passes before any primary run.
- Each run writes `inputs.json`, `validation.json`, `output.sha256` and the
  source archive, and its summary writes `report.md` with CSV twins of every
  figure.
