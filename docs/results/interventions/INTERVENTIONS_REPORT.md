# SidLens intervention experiments: collective report

Status as of 2026-09-30 (§9 time and §10 SAEs added; everything else as of 2026-09-28). The experiments ran on the CIAI cluster, which holds
the frozen snapshot `base-20260826`. This report collects every causal
intervention run so far on the AR (Qwen) and diffusion (DiffGRM) Semantic-ID
recommenders, plus one observational study that reads their activations.
Each experiment's `RESULTS.md` has the full tables. This report gives the
questions, the answers, how far they can be trusted, and what they add up to.

## The answer in one paragraph

Both paradigms recommend largely by **copying**:

- **Recency.** Their first-digit choice leans on the user's most recent item.
  Replacing it costs the correct first digit about 0.5 nats in both the AR and
  the diffusion models.
- **Prefix-matched copying.** Each later digit is taken from a history item
  whose earlier digits match the prefix generated so far. We call this
  **prefix-matched copying**. The models implement it differently.
- **AR.** The AR models read the matching item's next-digit token directly,
  through attention links in layers 14–27. Those links are spread over many
  heads.
- **DiffGRM.** DiffGRM spreads the item's information across its history
  slots in the encoder, and reads it from there. A direct read of the item
  only matters once that spread is blocked.
- **Next-two.** In the next-two AR model, the second item is built from the
  first in the same way.
- **Effect on recommendations.** Copying helps. Blocking the AR copy reads
  inside the archived decoder lowers HR@10 by 5.7 pp (RQ-KMeans) and 3.3 pp
  (RQ-VAE), mostly on targets the user already had (−27 and −10 pp). Blocking
  never raises new-target HR@10, and plain beam search gives the same picture
  (§4).

Along the way we found that the archived AR evaluation was not plain beam
search (§8).

## Models, cohorts and methods

| | AR | Diffusion (DiffGRM) |
|---|---|---|
| Models | Qwen2.5-1.5B SFT, 28 layers × 1536; `next-item_best` (RQ-KMeans 3×128), `oneoff_rqvae4cb128` (RQ-VAE 4×128), `two-item_best` (MQ 4×256) | 1 encoder + 4 decoder layers × 256; `diff-next1-rqkmeans-3cb-128`, `diff-next1-rqvae-4cb-128` (same SID tables as the two AR next-item models) |
| Cohort | 3,681 next-item test rows (1,606 users), histories ≤ 10 items; valid split 3,680 rows; next-two test 3,452 rows | 6,297 users, leave-last-out, histories ≤ 50 items |
| Score | golden code's log-prob among the digit's codes at the position that predicts it (teacher forcing) | the same, at a fixed decoder state: all digits masked, or golden prefix revealed |

**The two cohorts differ, so AR and DiffGRM numbers are compared side by
side, never joined.**

Every experiment follows the same rules:

- **Protocol first.** The question, conditions, estimands and expectations
  were written down before any outcome existed. Amendments are dated inside
  each protocol.
- **Bit-exact controls.** Every intervention has controls that must reproduce
  the unintervened model exactly: no-op patches or knockouts, full restores,
  self-patches and batch invariance. The code refuses silent no-ops, meaning
  hooks that never fire or knockouts of already-masked keys.
- **Statistics.** Intervals are paired user-bootstrap 95% intervals (2,000
  draws, recorded seed), and they **exclude training-seed variance**, because
  each checkpoint was trained once.
- **Units.** Effects are Δ = score(clean) − score(intervened), in nats. A
  positive Δ means the intervention hurt the correct answer.

## AR recommenders

### 1. exp3: swap one past item; patch layers

- **Split and model.** Test split, next-item.
- **What we did.** We replaced one history item at a time with a catalogue
  item sharing its first m digits. Then we copied the clean run's residual
  stream back in, layer by layer and position group by position group.
- **Most recent item.** Replacing the most recent item costs the correct first
  digit **0.57 nats** (RQ-KMeans) and **0.51** (RQ-VAE). It flips the model's
  top-1 first digit in 75% and 50% of rows.
- **Recency.** The second most recent item matters about a third to a half as
  much, and items 6–10 back barely matter.
- **First digit.** Keeping the replacement's first digit removes 82% and 64%
  of the effect.
- **Where the information travels.** It stays on the item's own tokens until
  layer 16–20, then moves to the answer positions in layers 19–24. The
  separator tokens never carry it.
- **Post-hoc lead.** Changing only an item's later digits hurt the matching
  answer digit *more* than replacing the whole item. That seeded exp4.

### 2. exp4: the copy mechanism, confirmed on held-out data

This was run on the valid split, which played no part in forming the
hypothesis.

| | RQ-KMeans | RQ-VAE |
|---|---|---|
| Change digit d of the recent item when it matches the target's first d digits, vs when it does not (H1) | **+1.34** [1.19, 1.51] | **+1.04** [0.92, 1.15] |
| Block the single attention link from the digit-d prediction position to that item's digit-d token (H2a) | **0.40** [0.32, 0.48] | **0.36** [0.31, 0.42] |
| Same link aimed at a non-matching item's token (paired difference, H2b) | +0.43 | +0.37 |
| Block it in layers 0–13 / layers 14–27 | **0.000** / 0.395 | 0.001 / 0.355 |
| Top-5 heads (chosen on half the users) vs random heads, on the other half (H3) | +0.025 | +0.024 |
| Share of the all-heads effect carried by those 5 heads | **9%** | **8%** |

The link is causal and late, and it is spread over many heads: no single head
exceeds 0.007 nats. On the test split every effect points the same way at
about half the size.

### 3. Representation: logit lens and probes (observational)

This study reads the 22 GB of captured activations.

- **The first digit is mostly a copy.** The model's final first-digit
  prediction equals the most recent item's first digit in **64%**
  (RQ-KMeans) and **53%** (RQ-VAE) of rows. The correct first digit is top-1
  in 18% and 34%.
- **Present early, used late.** That copied digit is linearly readable at the
  prediction position from layer 0 (84–93%). It becomes the model's output
  only after layer 20 (RQ-KMeans) or layer 11 (RQ-VAE). Direct reads before
  layer 14 are causally inert (exp4).
- **No extra target information.** At no layer does a linear probe recover the
  correct first digit better than the rule "copy the recent item's first
  digit".

### 4. exp6: does copying help or hurt real recommendations?

- **What we did.** We ran the archived decoder (trie-constrained beam
  search, 50 beams, reproduced batch for batch) on all 3,681 test rows. We
  blocked the copy links in layers 14–27, from the step choosing digit d to
  digit d of every history item that matches that beam's own prefix. The
  outcome is exact-SID HR@10. **Repeat** targets have their SID already in
  the history; **new** targets do not.
- **Reproduction.** The unblocked baseline reproduces the archived
  RQ-KMeans predictions on 3,680 of 3,681 rows, and HR@1–50 exactly.

| HR@10 change (pp) | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|
| Baseline HR@10: all / repeat / new | 24.0% / 91.0% / 13.0% | 14.4% / 91.6% / 10.5% |
| Block copy reads at every digit: repeat targets | **−27.3** [−32.3, −22.1] | **−10.1** [−15.0, −5.8] |
| … new targets | **−2.1** [−3.1, −1.2] | **−3.0** [−3.8, −2.2] |
| … all rows | −5.7 [−6.8, −4.6] | −3.3 [−4.1, −2.5] |
| Block digits 2+ only: repeat / new | −11.3 / +0.1 | −5.6 / −2.8 |
| Control, as many non-matching reads blocked | −0.03 [−0.19, 0.13] | −0.11 [−0.31, 0.08] |

- **Copying helps.** It is how both models hit 91% of repeat targets.
  Repeat rows are 14% (RQ-KMeans) and 5% (RQ-VAE) of the cohort, but supply
  54% and 31% of all HR@10 hits.
- **It does not crowd out new items.** Blocking never raises new-target
  HR@10. For RQ-VAE, blocking the later-digit reads alone costs new targets
  2.8 pp: copying a matching item's next digit also finds new items that share
  a prefix with the history.
- **The manipulation reaches the output.** The share of top-1 SIDs already in
  the history falls from 58% to 18% (RQ-KMeans) and from 25% to 10%
  (RQ-VAE).
- **All four declared expectations held.**

**Under plain beam search** (a secondary decoder with no sampling and no
repetition penalty, re-run after amendment 2), the picture is the same:

| | RQ-KMeans | RQ-VAE |
|---|---|---|
| HR@10, plain vs archived decoder (same rows) | 24.7% vs 24.0% | 15.3% vs 14.4% |
| Blocking copying at every digit: all / repeat / new | −4.0 / −19.4 / −1.5 pp | −3.2 / −8.4 / −3.0 pp |

- **Every interval excludes zero.** The full table is in exp6's RESULTS.
- **The repetition penalty was holding copying back.** Without it, the top-1
  SID is a history SID more often: 64% vs 58% (RQ-KMeans), 34% vs 25%
  (RQ-VAE).
- **One exception.** For RQ-KMeans, blocking raises new-target HR@1 by
  0.6 pp [0.06, 1.15]: with copying off, the top slot sometimes goes to a
  correct new item. Within the top-10 the loss outweighs it.

### 5. exp5: next-two recommendation (`two-item_best`, MQ 4×256)

- **Item 1 matters.** Replacing the first predicted item costs the second
  item's first digit **0.60 nats**.
- **More than history does.** That is **0.40 more** than replacing the most
  recent history item costs.
- **Same copying.** Prefix-matched copying runs from item 1 to item 2: 0.43–0.73
  nats when they share a prefix, about 0 otherwise.
- **Same late window.** Item 1's information reaches item 2 in layers 20–26.

## DiffGRM

### 6. exp7: what DiffGRM uses from history

- **The same behaviour as AR.**
  - Replacing the most recent item costs the correct first digit **0.56** and
    **0.48 nats** (AR: 0.57 and 0.51).
  - The effect decays just as steeply with recency.
  - Prefix-matched copying is strong: **0.35** and **0.96 nats** in matching
    rows, 0.01 and 0.07 otherwise.
- **Not the same mechanism.** Blocking the direct cross-attention link does
  nothing (−0.007 and −0.004). That holds from digit d, from all digits, and
  in any single decoder block, all aimed at the matching item's slot.

### 7. exp8: how DiffGRM routes the copied item

The route turns out to be the encoder:

- **Encoder route alone** (other history slots can't read the recent item):
  **0.17** and **0.30 nats**.
- **Decoder route alone** (digits can't read its slot): about **0**.
- **Both together:** **0.24** and **0.60 nats**, about two thirds of the full
  replacement effect.
- **Control.** A non-matching item gives −0.02 and −0.05.
- **Backup behaviour.** The two routes are super-additive (+0.08 and +0.30).
  The direct read is a backup that matters only once the spread is blocked.

## AR vs DiffGRM, side by side

These are different cohorts, so the comparison is descriptive.

| | AR (Qwen) | DiffGRM |
|---|---|---|
| First-digit reliance on the most recent item | 0.57 / 0.51 nats | 0.56 / 0.48 nats |
| Recency decay (second item ÷ most recent, digit 1) | 0.28 / 0.43 | 0.26 / 0.24 |
| Prefix-matched copying (match vs non-match) | 1.35 / 1.06 vs ≈ 0 | 0.35 / 0.96 vs ≈ 0 |
| How the history item is represented | one token per SID digit | one token per item |
| Route of the copy | direct attention from the prediction position to the item's digit token, layers 14–27, many heads | the encoder spreads the item into other slots; the decoder reads it there; the direct read is a backup |
| Where the evidence reaches the answer | layers 19–24 of 28 | not layer-resolved; any single decoder block is dispensable |

## 8. Findings about the evaluation itself

These came out of the experiments, and each matters for the paper:

- **The archived AR evaluation used beam sampling, not beam search.**
  `evaluate.py` passes a `GenerationConfig` without sampling settings, and
  transformers 4.57.1 merges the checkpoint's Qwen-Instruct defaults into it:
  `do_sample=True`, temperature 0.7, top-k 20, top-p 0.8, repetition penalty
  1.1. The sweep ran on exactly that version. The consequences:
  - Each beam is cut to a small nucleus.
  - After the first step about 80% of the 50 beams are filler.
  - 18.9% (RQ-KMeans) and 15.0% (RQ-VAE) of archived list entries are not
    catalogue SIDs, and 7.6% and 0.7% of rows have one inside the top-10.
  - The repetition penalty works against copying history tokens.

  The reproduction under the same settings is exact: 3,680 of 3,681 rows
  have identical 50-SID lists, and HR@1–50 match the recorded values.
- **Turning the sampling off takes more than setting it.** In transformers
  ≥ 4.50, `generate()` refills every field left at its *library* default
  from the checkpoint's `generation_config.json`. `do_sample=False` and
  `repetition_penalty=1.0` are those defaults, so passing them explicitly
  changes nothing. Only `use_model_defaults=False` gives plain beam search.
  exp6's first plain-decoder run fell into this trap (amendment 2).
- **The frozen `oneoff_rqvae4cb128` is not the model behind the archived
  RQ-VAE predictions.** It reproduces 0 of 3,681 of them, and scores HR@10 14.5% against the recorded 13.1% under the archive's own rule. It came from an earlier
  training run of the same configuration. RQ-VAE AR results describe the
  one-off, not the sweep's model.
- **The prompt wording changed upstream.** OneDiffRec commit `a165f65`
  (2026-09-13) stripped trailing spaces from the prompt templates, which
  changes token ids. The frozen checkpoints predate it, and SidLens rebuilds
  their exact original prompts. Any checkpoint trained after that date needs
  the new wording.

## 9. Time: what "recent" means (added 2026-09-30)

No model is given a timestamp. The frozen reviews keep a day for every event,
and `sidlens.data.timestamps` puts it back: it is exact for all 43,102 events,
with ambiguous duplicate days resolved by the global split order. Two new
experiments use it:

- **retrospective exp6:** the data, plus exp3/exp4/exp6/exp7 re-read by time
- **controlled exp9:** swap the two most recent items, with rows split by
  their days

### What the data say

- **Same-day bursts are the norm.** 41% (test) to 50% (train) of targets fall
  on the day of the most recent item.
- **Same-day order is ASIN order.** 88% of same-day pairs are ASIN-ascending,
  the raw-file order that upstream's stable sort keeps on ties.
- **Age, not position, carries the recency signal.** Once an item's age is
  known, its position adds ≤ 0.002 McFadden R² about whether it shares the
  target's first digit. Age adds 0.10–0.13 beyond position.
- **Duplicate records are common.** 2,653 of 43,102 events duplicate a (user,
  item, day) review. Duplicate targets are 8% of train, 5% of valid and 3% of
  test rows.

### What the models do with it

| | RQ-KMeans AR | RQ-VAE AR | DiffGRM RQ-KMeans / RQ-VAE |
|---|---|---|---|
| Copies r1's first digit: same day vs > 1 year (data: 0.33 / 0.50 vs 0.03 / 0.09) | 0.67 vs 0.67 | 0.65 vs 0.48 | 0.41 vs 0.30 / 0.49 vs 0.40 |
| ρ: the share of the data's decay the copy tracks | **0.01** [−0.15, 0.16] | 0.40 | 0.33 / 0.26 |
| exp9 position effect A: r1 later / r1, r2 tied with the target later | 0.38 / **0.36** | 0.10 / 0.07 | – |
| exp9 first-digit flips under a timestamp-preserving swap | **50%** | 18% | – |
| exp9 tie averaging, Δ log p(SID): tied rows / real order | **+0.027** / −0.019 | −0.008 / −0.002 | – |

- **RQ-KMeans AR's recency is purely positional.** It copies the last item of
  the prompt, whatever its age and whatever the timestamps say about the order.
  Half its first-digit decisions on tied rows depend on ASIN order.
- **Averaging over the orders the timestamps cannot distinguish helps only
  there.** It is a free, timestamp-aware inference rule that improves
  likelihood (+0.027) but not HR@10.
- **RQ-VAE AR is weakly positional.** Its preference follows the truly later
  item about as strongly as the position (B ≈ 0.07).

### What this changes

- **exp6's "copying helps"** is a same-day effect. Rows whose target falls on
  the day of the most recent item carry 87% (RQ-KMeans) and 88% (RQ-VAE) of the
  HR@10 lost when copying is blocked. Copying does nothing for new items on
  later days (+0.1 / −0.7 pp).
- **Duplicate records inflate HR@10.** They are 2.7% of test rows but supply
  **11% and 17% of all HR@10 hits** (98% / 91% HR@10 on them).
- **Open question 4 (valid effects about 2× test):** about half comes from
  period composition. Valid has twice test's share of duplicate-record matches.
  Post-stratifying on duplicate and same-day rows moves the exp4 H1 ratio from
  1.51 to 1.25 (RQ-KMeans) and from 1.65 to 1.32 (RQ-VAE).
- **The DiffGRM cohort** (leave-last-out) is not a time split. Its targets span
  2013–2018.

Details: `experiments/retrospective/exp6_time_structure/RESULTS.md` and
`experiments/controlled/exp9_ar_order_vs_time/RESULTS.md`.

## 10. SAE features at the readouts (added 2026-09-30)

TopK SAEs (12,288 latents, k = 32) were fitted per model at layers 12, 16, 20
and 24, on train-split captures. They were then read on the test captures and
ablated, error-preserving, through the fixed-shape runner. All 1,024 self-patch
controls are bit-exact. On test tokens the SAEs keep ≥ 97% of what
mean-ablation destroys when spliced in.

| | RQ-KMeans AR | RQ-VAE AR |
|---|---|---|
| Copy latents at the first-digit readout, ablated (copy score, nats vs matched control), L20 / L24 | 0.10 / **0.21** | 0.10 / **0.36** |
| … the same at L12 / L16 | 0.01 / 0.03 | −0.02 / 0.01 |
| Their share of readout(0) latent activation, L24 | 2.9% | 12% |
| Prefix-match latents at the digit-d readout: best within-digit held-out AUC | 0.94–0.97 (L12–20) | about 0.66 |
| Ablating them: Δ(match) − Δ(non-match), nats | **≈ 0** (±0.04; also under amendment A1) | +0.02–0.03 |
| Target content beyond the history (ridge, Qwen3 embedding) | none | none |
| Same-day burst decodable beyond SID overlap (AUC gain) | +0.08–0.09 | +0.08–0.09 |

- **The copy is a late, feature-level decision.** A few code-specific latents
  carry part of it from layer 20 on. Earlier, the recent item's digit is
  decodable (exp1) but not held in features whose removal matters.
- **The prefix match is detected at the readout, but that detection is not
  what the output reads (RQ-KMeans).** With exp4's edge knockout, the match
  looks like it lives in attention's query–key comparison. That narrows open
  question 2.
- **No target knowledge beyond the history** shows up in digit probes (exp1) or
  content probes (here).
- **The readout encodes whether the history looks like a same-day burst.**
  RQ-KMeans' copying ignores this (§9).

Details: `experiments/representation/exp2_ar_sae/RESULTS.md`.

## How much to trust these numbers

| Check | Result |
|---|---|
| Frozen checkpoints = upstream weights | sha256 matches for all three AR checkpoints |
| Inputs = what the models saw | prompt ids equal upstream's dataset classes row for row; rows equal the archive (3,681 / 3,681) and the SID tables |
| AR patch controls (exp3, exp5) | 44,172 + 41,424 controls bit-exact (no-op, full restore, self-patch) |
| AR knockout no-ops (exp4) | 14,722 bit-exact (valid + test) |
| Knockout inside `generate()` (exp6) | an empty plan is bit-identical to unhooked `generate()` (sequences and scores, 8 GPU batches per cell); the baseline reproduces 3,680 / 3,681 archived lists |
| DiffGRM (exp7, exp8) | the all-ones cross mask and the padding-only encoder mask are bit-exact; 119,643 clean-row batch-invariance comparisons exact; exp1's decoder-path checks pass |
| Probe pipeline | reproduces the model's own output ranks at bf16 precision (99.95% / 99.99%) |
| Tests | 276 passing CPU tests. The 11 failures all need `external/labels`, which is not on this cluster. |

## Limits that apply throughout

- **Scores.** Teacher-forced (AR) and fixed-state (DiffGRM) scores are
  conditional log-probs, not HR. exp6 is the only experiment that measures
  recommendation quality directly.
- **Seeds.** There is one training seed per checkpoint, so the intervals are
  conditional on these weights.
- **Controls.** They are uniform catalogue items, not category-matched.
  Industrial category labels are not on this cluster.
- **Cell differences.** The two cells differ in quantizer and depth; neither
  comparison isolates the quantizer.
- **Knockouts.** A knockout removes direct reads, and indirect routes remain.
  They are bounded in exp4 (K2, K6) and exp8, not eliminated.
- **Hypotheses.** One hypothesis (prefix-matched copying) was found post hoc,
  then confirmed on a separate split (AR) and on another model family
  (DiffGRM). The DiffGRM route experiment (exp8) was designed after seeing
  exp7.

## Where everything is

| Experiment | Code | Results under `$SIDLENS_WORK/derived/` |
|---|---|---|
| exp3 | `experiments/controlled/exp3_ar_history_patching/` | `controlled/exp3_ar_history_patching/summary-280961/` |
| exp4 | `experiments/controlled/exp4_ar_copy_circuit/` | `controlled/exp4_ar_copy_circuit/summary-281261/` (valid), `summary-281394/` (test) |
| exp5 | `experiments/controlled/exp5_ar_next_two_conditioning/` | `controlled/exp5_ar_next_two_conditioning/summary-281263/` |
| exp6 | `experiments/controlled/exp6_ar_copy_in_decoding/` | `controlled/exp6_ar_copy_in_decoding/summary-283114/` (primary), `summary-283586-283621/` (plain decoder) |
| exp7 | `experiments/controlled/exp7_diffusion_history_use/` | `controlled/exp7_diffusion_history_use/summary-283124-r2/` |
| exp8 | `experiments/controlled/exp8_diffusion_copy_route/` | `controlled/exp8_diffusion_copy_route/summary-283159/` |
| Representation | `experiments/representation/exp1_ar_digit_decoding/` | `representation/exp1_ar_digit_decoding/summary-281264-281411/` |
| Activation captures | `scripts/activations/ar_capture.*` | `ar_capture/281053`, `281054`, `281055` (22 GB) |
| Retro exp6 (time) | `experiments/retrospective/exp6_time_structure/` | `retrospective/exp6_time_structure/287189/result/` |
| exp9 (order vs time) | `experiments/controlled/exp9_ar_order_vs_time/` | `controlled/exp9_ar_order_vs_time/summary-287182/` (test), `summary-287183/` (valid) |
| Representation exp2 (SAEs) | `experiments/representation/exp2_ar_sae/` | `representation/exp2_ar_sae/summary-287222-287231/`, `summary-a1-287380-287415/`; train captures `ar_capture/287160`, `287161` (12.6 GB) |

The shared intervention code is in `src/sidlens/interventions/`:

- `residual`: residual patching
- `attention`: teacher-forced edge knockout
- `generation`: knockout inside `generate()`
- `diffusion`: DiffGRM states, cross-attention and encoder knockouts
- `history`: prefix-matched controls
- `runner`: fixed-shape runner
- `scoring`

The bootstrap is `sidlens.analysis.bootstrap`.

## Open questions and the next experiments

1. **DiffGRM in decoding.** Does copying help or hurt DiffGRM's actual
   recommendations? This would be the exp6 analogue, with the encoder and
   decoder routes blocked inside `matched_decode`.
2. **Where the prefix match is computed.** *(Narrowed 2026-09-30, §10: not
   a readout feature the output reads, for RQ-KMeans. The next step is QK
   features on attention inputs in layers 14–27.)* Something in AR layers 14–27, or in
   DiffGRM's encoder, has to decide that an item matches. Candidates are MLP
   or attention features. This needs feature-level work: SAEs on the existing
   captures.
3. **Category-matched controls and category probes.** These are waiting for
   `external/labels` on this cluster, which needs `hf auth login`.
4. **Valid vs test effect sizes (AR).** *(About half explained 2026-09-30,
   §9: valid has twice test's share of duplicate-record and same-day
   matches.)* The effects on the valid split are
   about twice the test-split ones. Why is unexplained.
5. **A fair AR re-evaluation.** Plain beam search, with no sampling defaults,
   for the paper's AR numbers. exp6's `B_plain` gives it for the two surviving
   next-item checkpoints: +0.65 and +0.95 pp HR@10 over the archived decoder.
   The other 26 AR runs have no weights left to re-decode.
