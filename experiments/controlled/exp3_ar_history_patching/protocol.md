# AR history interventions and residual-stream patching: frozen next-item checkpoints

Declared 2026-09-27 on the upstream (CIAI) cluster before any intervened or
patched forward pass had been run. No training, no checkpoint selection. The
only data inspected beforehand were input identities: the history-length
distribution, the token-role map of one prompt, and checkpoint hashes.

## Questions

Brief RQ1/RQ2, Experiment 1 ("history patching will replace one past
interaction at a time to show which interactions affect each digit") and
Experiment 3 ("matched-history activation patching will show how each history
position affects each output digit").

- **Q1 (Part A, input intervention).** Which past interactions causally change
  the score of each target SID digit? How does the effect depend on the item's
  recency, and on which of that item's digits are changed?
- **Q2 (Part B, activation patching).** At which layer does the evidence
  carried by the most recent history item leave that item's token positions,
  and when does it reach the positions that score the target digits?

## Fixed design

**Checkpoints.** The two surviving next-item AR checkpoints, which are also the
two cells matched to diffusion (`sidlens.models.ar.MATCHED_CELLS`):

| cell | checkpoint | SID variant | model.safetensors sha256 |
|---|---|---|---|
| 00 | `next-item_best` | `rqkmeans_3codebook_128` | `4028d511…5f82` |
| 01 | `oneoff_rqvae4cb128` | `rqvae_4codebook_128` | `2cb56795…090e` |

`two-item_best` (MQ 4×256) is excluded. Next-two is a separate experiment
(clamp item 1, RQ4).

**Cohort.** All 3,681 historical AR next-item test rows (1,606 users). Before
the model is loaded, the rows must equal the archived predictions
(`check_against_archive`) and the frozen `.sem_ids` (`check_against_table`).
There are no row exclusions. Histories hold 1–10 items. This is not the diffusion cohort.

**Input.** `sidlens.data.ar_prompts.encode(..., template="eval",
with_target=True)`. That is the wording behind every archived prediction, with
prompt + golden target + EOS (teacher forcing). The sft wording is a declared
secondary run of Part A only (`--template sft`, `--parts A`). Intervened inputs
are built by editing the example's history and re-running `encode`. Never
edit token ids by hand.

**Score.** Target digit d is scored at `Encoded.predict_pos(0, d)`:

- Primary: s_d = log-softmax over that digit's code tokens
  (`vocab.ids(d)`), evaluated at the golden code.
- Secondary: rank among the codes, top-1 code, log-softmax over the codes that
  extend the golden prefix to a catalogue SID ("legal"), and full-vocabulary
  log-prob.
- Whole-SID: S = Σ_d s_d.

These are conditional scores at a fixed teacher-forced state. They are not
beam-search HR.

### Part A: replace one history item (total effect)

For every row, every history item k and every shared-prefix level
m ∈ {0, …, n−1} (n = SID depth), replace item k with a control item c:

- c is drawn uniformly from catalogue items whose SID equals item k's on
  digits 0…m−1 and differs at digit m.
- Candidates whose item id or full SID appears in the row's history or target
  are excluded.
- The seed is `sha256("20260927|<example_id>|<k>|<m>")`.

When no candidate exists, the condition is recorded as `no_control` and
counted; it is never silently dropped. Recency r = len(history) − k, so r = 1
is the most recent item.

The effect is Δ_d(k, m) = s_d(clean) − s_d(intervened). Positive means the
replacement lowered the golden code's score.

- **A1 (Part A primary):** mean over rows of Δ_d(r = 1, m = 0), for each digit
  and checkpoint.
- **A2:** the recency profile Δ_d(r, m = 0), r = 1…10, over rows with
  len(history) ≥ r.
- **A3:** the prefix-sharing profile Δ_d(r = 1, m) and the retained fraction
  mean Δ_d(r=1, m) / mean Δ_d(r=1, 0).
- **Secondary:** top-1 flip rate per digit; ΔS.

### Part B: residual-stream patching

- **Corrupted input:** the Part A condition r = 1, m = 0 (same control item).
- **Clean input:** the original row.
- **Sites:** for each layer L ∈ {0…27}, the output of `model.layers.L`
  (residual stream after block L), crossed with position groups G:
  - `item`: the n replaced SID tokens
  - `between`: tokens after the item's last SID token and before the response
    header
  - `header`: the three response-header tokens (the last is digit 0's
    predict_pos)
  - `target`: the target SID tokens (the predict positions for d ≥ 1)
- **Patch:** run the corrupted input with the clean activation substituted at
  (L, G). The patch source is the clean run of the same row.

Recovery is R_d(L, G) = Σ_rows [s_d(patched) − s_d(corr)] / Σ_rows
[s_d(clean) − s_d(corr)], a ratio of sums bootstrapped by user. The
unnormalized mean of s_d(patched) − s_d(corr) is also reported.

- **Primary (Part B):** the curves R_d(L, item) and R_d(L, header) +
  R_d(L, target) over L.
- **Secondary:** the median per-row normalized recovery over rows with
  |s_d(clean) − s_d(corr)| ≥ 0.5 nats.

### Controls and numerical acceptance

Every forward uses one fixed shape: B rows × T_pad tokens. T_pad is the
cohort's maximum teacher-forced length rounded up to a multiple of 8. A short
last batch is filled with copies of a real row, whose outputs are discarded.
Padding is on the right with explicit `position_ids` (`collate`). With shapes
fixed, the kernels do not change between forwards, so a row's output depends
only on its own tokens and patches. The following must then hold exactly (bit
equality of the digit logits):

- **V1 no-op site:** patching the positions before the item (`pre`) with clean
  values at L ∈ {0, 9, 18, 27} equals the corrupted run. Causal masking makes
  those activations identical in both runs.
- **V2 full restore:** patching all positions at L ∈ {0, 9, 18, 27} equals
  the clean run.
- **V3 self-patch:** patching `item` with the corrupted run's own values at
  L ∈ {0, 9, 18, 27} equals the corrupted run.
- **V4 batch invariance:** the first rows re-run in a different batch
  composition give identical clean scores.
- **V5 input identity:** the checkpoint sha256 equals the manifest. Rows equal
  the archive and the SID tables. Every intervened `encode` has the clean
  length and differs only at item k's digit positions ≥ m, including digit m.
- **V6 sanity:** clean teacher-forced top-1 rate per digit, and agreement of
  bf16 with fp32 on 16 rows. This is not a metric reproduction.

The patch hook must fire exactly once per patched layer per forward; any
other count raises. If V1–V4 are not exact on the pilot, the run stops. If
they are not exact on the full run, the maximum deviation is reported as the
numerical floor, and effects below ten times that floor are treated as zero.

**Precision.** bf16, the dtype `evaluate.py` used, with sdpa attention, eval
mode, `torch.no_grad` and deterministic algorithms. The bf16/fp32 agreement
from V6 is recorded.

## Statistics

- Paired user bootstrap: 1,606 users. For every sampled user, all rows,
  conditions and both cells are kept together.
- 2,000 draws, seed 20260927, percentile 95% intervals. Intervals are in nats
  (Part A) and in recovery fractions (Part B).
- Intervals exclude training-seed variance (one seed per checkpoint). They
  also exclude control-draw variance beyond what resampling rows captures (one
  draw per row, k and m).
- Per-digit, per-recency and per-layer estimates are descriptive. Do not read
  many unadjusted intervals as independent confirmations.
- The two cells differ in quantizer and in depth (3 vs 4 digits), so their
  contrast is descriptive, not a controlled quantizer effect.

## Expectations recorded before outcomes

These are not hypotheses to be confirmed. They are written down so that a
surprise is visible.

- **E1:** The most recent item has the largest effect, and effects decay with
  recency.
- **E2:** In both RQ checkpoints, keeping the replaced item's first digit
  (m = 1) removes a large part of the effect on output digit 0.
- **E3:** `item` recovery is near 1 in early layers and near 0 by layer 27;
  `header`/`target` recovery rises correspondingly, crossing in the middle
  layers.

## Interpretation limits

- A replacement effect is a total effect of an input change. It mixes
  removing the item's evidence with adding the control's.
- Uniform catalogue controls are not frequency- or category-matched.
  Category-matched controls need `external/labels`, which is not on this
  cluster. They are a follow-up.
- Patching localizes where information sits at grouped positions per layer.
  It does not say which heads or MLPs move it. Attention output alone is not
  causal evidence.
- A teacher-forced digit score is not HR@10.
- This is one seed per checkpoint.

## Outputs

`$SIDLENS_WORK/derived/controlled/exp3_ar_history_patching/<array-id>/cell-XX/`
holds:

- `protocol.md`, `source.tar.gz`, `source.sha256`, `arguments.json`
- `inputs.json`: checkpoint, CSV and SID-table hashes, T_pad, template
- `validation.json`
- `clean.parquet`, `part_a.parquet`, `part_b.parquet`, `controls.parquet`:
  one row per (row, condition, digit)
- `status.txt`, `output.sha256`

`summarize.py` writes `summary-<array>-<job>/`, which holds `result.json`,
`report.md` and `figures/`. The summary recomputes every estimate from the
per-row tables.
