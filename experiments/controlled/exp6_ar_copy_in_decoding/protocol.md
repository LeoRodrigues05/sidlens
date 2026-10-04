# Does prefix-matched copying help or hurt real recommendations? Copy knockout in beam search

Declared 2026-09-28, before any decoding run in this repository. No training
and no checkpoint selection.

## Question

exp4 found that, under teacher forcing, the AR recommenders choose digit d by
reading digit d of history items whose first d digits match the prefix
generated so far. They do that read in layers 14–27. The representation
experiment found that the final first-digit prediction equals the most recent
item's first digit in 53–64% of rows.

Here the question is whether that read raises or lowers recommendation quality
in the archived decoder (trie-constrained beam search, 50 beams). It is asked
separately for targets the user has already interacted with (**repeat**) and
for targets that are **new** to the user.

## Fixed design

**Cells.**

- **Cell 00: `next-item_best`** (RQ-KMeans 3×128). Its weights are the ones
  that produced the archived predictions (hashes matched 2026-09-27).
- **Cell 01: `oneoff_rqvae4cb128`** (RQ-VAE 4×128). The archived RQ-VAE
  predictions came from the sweep's own run of the same configuration, whose
  weights were deleted. Agreement with that archive is therefore reported as a
  behavioural test of whether the one-off is the same model. It is not
  required.

**Cohort.** All 3,681 AR next-item test rows (1,606 users), which match the
archive and the SID tables. There are no exclusions.

**Decoder.** Exactly the archived evaluator (`vendor/onediffrec/evaluate.py`
as run by `scripts/eval_variant.sh`):

- HF `generate()` in bf16, with 50 beams, `length_penalty` 0,
  `num_return_sequences` 50 and `max_new_tokens` 256
- the vendored `ConstrainedLogitsProcessor`, with a trie built from the frozen
  info file exactly as `evaluate.py` builds it
- eval wording and left padding
- the archive's batch layout: 4 contiguous shards (`split.py`), batches of 8
  consecutive rows within a shard

Completions are decoded as `evaluate.py` does. The output is a list of 50 SIDs
per row.

**Knockout** (`sidlens.interventions.generation`). Edges are removed in
**layers 14–27, all heads**, and only at the steps that choose SID digits. At
the step choosing digit d, the query is that step's position: the last prompt
token for d = 0, and the previously generated digit for d ≥ 1. Each beam uses
its **own** generated prefix. The conditions:

| Condition | d = 0 | d ≥ 1 |
|---|---|---|
| `B` baseline | no edges removed (same hooked code path) | no edges removed |
| `C_later` | nothing | block digit d of every history item whose first d digits equal the beam's generated digits |
| `C_all` | block digit 0 of every history item | as `C_later` |
| `N_later` (control) | nothing | block digit d of the same number of history items whose first d digits do *not* match (most recent first; all available if fewer) |

All four conditions run through the same hooked path, and the conditions
differ only in the removed edges.

## Validation (before any condition counts)

- **V1, reproduction (cell 00).** The `B` lists must equal the archived
  `predict` lists. The fraction of rows with identical 50-SID lists and with
  identical top-10 is reported. Acceptance: at least 99% of rows identical,
  and archive-equivalent HR@{1,5,10} (`calc.py`'s rule) equal to the recorded
  values within 0.1 pp. Every differing row is listed.
- **V2, hook neutrality.** On a pilot of 64 rows, `B` equals plain unhooked
  `generate()` token for token and score for score. The CPU unit test
  `tests/interventions/test_generation_knockout.py` proves the same on a toy
  model.
- **V3, manipulation.** Removed edges are counted per step and condition.
  `C_later`, `C_all` and `N_later` must each remove at least one edge on a
  majority of rows at d ≥ 1 (`C_all` also at d = 0). The guards (a live edge,
  the query keeps a key, fire-once) must never trip.
- **V4, cell 01 identity.** The same agreement numbers for the one-off vs the
  archive. These are reported, not required.

## Outcomes and strata

**Metrics.** Exact-SID rank of the target in the 50-list (first occurrence),
HR@{1, 5, 10, 20, 50} and NDCG@10. The `calc.py`-equivalent HR is used for V1
only. SID-level: a collided target counts as a hit when its SID is listed.

**Strata** (from the CSV, fixed before decoding):

- **repeat:** the target SID equals the SID of some history item.
  - `repeat_item` (reported within it): the target item id itself is in the
    history.
- **new:** all other rows.

**Primary estimands** (per cell; HR@10; paired user bootstrap):

- **P1:** ΔHR@10(`C_all` − `B`) on repeat rows.
- **P2:** ΔHR@10(`C_all` − `B`) on new rows.
- **P3:** the interaction P1 − P2.

**Secondary:**

- the same three quantities for `C_later`
- ΔHR@10(`N_later` − `B`), and the paired difference (`C_later` −
  `N_later`) on all rows
- HR@{1, 5, 20, 50} and NDCG@10 for all of the above

**Manipulation check.** The copy rate, meaning the share of rows whose top-1
SID equals some history SID, and the share whose top-1 first digit equals the
most recent item's first digit, under each condition.

## Statistics

- Paired user bootstrap: 1,606 users, 2,000 draws, seed 20260927, percentile
  95% intervals. All conditions and both cells of a sampled user are kept
  together.
- Intervals exclude training-seed variance.
- Strata are defined before treatment. Per-K and per-stratum secondary results
  are descriptive.

## Expectations recorded before outcomes

- **E1:** P1 < 0. Blocking copying lowers HR@10 on repeat rows, clearly for
  `C_all`.
- **E2:** |P2| < |P1|. The effect on new rows is smaller in magnitude, with no
  declared sign.
- **E3:** `C_later` moves repeat-row HR@10 less than `C_all` does.
- **E4:** `N_later` changes HR@10 by much less than `C_later`.

## Interpretation limits

- A knockout removes direct reads in layers 14–27. Indirect routes and layers
  0–13 remain (see exp4 K2/K6).
- Results hold at beam 50, for these two checkpoints, with one seed each.
- SID-level HR hides collisions; the repeat stratum includes collision repeats.
- The cells differ in quantizer and in depth.
- The AR cohort is not the diffusion cohort.

## Amendment (2026-09-28, after the 64-row pilot baseline, before any knockout outcome existed)

**What the pilot showed.** transformers 4.57.1 warned that `generate()`
merges the checkpoint's `generation_config.json` into the GenerationConfig
that `evaluate.py` passes, because that config never sets these fields:

- `do_sample=True`
- `temperature=0.7`
- `top_k=20`
- `top_p=0.8`
- `repetition_penalty=1.1`

The sweep ran the same transformers 4.57.1 (`results/job_manifests`) with
the same checkpoint files. So the **archived decoder was beam sampling with a
repetition penalty, not plain beam search.**

**The consequences:**

- **Each beam is restricted to a small nucleus.** Temperature 0.7 plus top-k
  20 / top-p 0.8 leave few candidates per beam. The sampling therefore usually
  selects all of them, and ranking is nearly deterministic.
- **Short pools are filled with garbage beams.** When fewer finite candidates
  exist than beams, the remaining slots are filled with invalid beams. In the
  archive, 18.9% (RQ-KMeans) and 15.0% (RQ-VAE) of list entries are not
  catalogue SIDs. For 7.6% and 0.7% of rows, one sits inside the top-10.
- **Copying is penalized.** The repetition penalty lowers the scores of tokens
  already present in the input, including the history SID tokens.

The pilot baseline (`B`, unseeded) reproduced the archive on 63 / 64 rows
(identical 50-lists) and on 64 / 64 top-10s and top-1s. A knockout condition
crashed on a garbage beam whose "prefix" is not a digit sequence.

**Changes:**

1. Every condition calls `evaluate.py`'s `set_seed(42)` at the start of each
   of the 4 shards, as each shard process did.
2. A beam whose generated prefix is not a valid digit sequence gets no
   knockout. It is counted, not raised.
3. **V5, determinism:** on the pilot, `B` is re-run with seed 1234, and top-10
   agreement is reported.
4. **Declared secondary decoder:** `B_plain` and `C_all_plain` use plain
   constrained beam search. Every other setting is identical, except
   `do_sample=False` and `repetition_penalty=1.0`, both set explicitly. That
   version asks what copying does when the decoder does not itself penalize
   it.

The primary estimands (P1–P3) stay on the archived decoder, since that
decoder produced the reported numbers. V1 acceptance is unchanged.

## Amendment 2 (2026-09-28, after the primary results of job 283114)

**What went wrong.** In job 283114 the plain-decoder conditions (`B_plain`,
`C_all_plain`) came out bit-identical to `B` and `C_all`. transformers 4.57
`generate()` runs with `use_model_defaults=None`. For checkpoints saved with
transformers ≥ 4.50, it then refills **every field left at its library
default** from the checkpoint's `generation_config.json`. The explicit
`do_sample=False` and `repetition_penalty=1.0` are those library defaults, so
they were replaced by the Qwen sampling defaults again. The declared secondary
decoder therefore never ran.

**The fix.** The `*_plain` conditions now pass `use_model_defaults=False`. They
are re-run as a separate job, and nothing else changes. The job-283114
`*_plain` outputs are kept, but superseded.

**Disclosure.** This re-run happens after the primary estimands (P1–P3 on
`C_all`/`C_later`/`N_later`) were seen. The plain decoder is a secondary,
declared before any run, and its design is unchanged.
