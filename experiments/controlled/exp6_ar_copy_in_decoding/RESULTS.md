# Copy knockout in the archived AR decoder: completed 2026-09-28

Prefix-matched copying **helps** real recommendations. Blocking it inside the
archived beam-search decoder lowers HR@10 for both models. It never raises it.

- **Repeat targets lose most.** These are targets whose SID is already in the
  user's history. Blocking the copy reads at every digit (`C_all`) lowers
  their HR@10 by **27.3 pp** (RQ-KMeans 3×128) and **10.1 pp** (RQ-VAE 4×128).
  With copying intact both models hit 91% of them.
- **New targets lose too, less.** HR@10 drops by **2.1 pp** and **3.0 pp**.
  For RQ-VAE, blocking only the later-digit reads (`C_later`) already costs
  new targets 2.8 pp. There, copying a matching item's next digit helps find
  *new* items that share a prefix with something the user had. For RQ-KMeans
  the same block leaves new targets unchanged (+0.06 pp).
- **Where the loss comes from.** Repeat rows are 14.2% (RQ-KMeans) and 4.8%
  (RQ-VAE) of the cohort. They supply 54% and 31% of the baseline's HR@10
  hits. About two thirds of RQ-KMeans's `C_all` loss comes from repeat rows;
  for RQ-VAE, 85% of the loss comes from new rows.
- **Control.** Blocking as many *non-matching* reads (`N_later`) changes HR@10
  by −0.03 and −0.11 pp. Neither interval excludes zero.
- **The knockout does what it says.** The share of rows whose top-1 SID is
  already in the history falls from 58.4% to 18.4% (RQ-KMeans) and from 25.1%
  to 9.5% (RQ-VAE) under `C_all`. Under `N_later` it stays at 59.4% and
  25.9%.
- **The same holds under plain beam search** (secondary). Blocking at every
  digit lowers HR@10 by 4.0 and 3.2 pp there, and repeat targets by 19.4 and
  8.4 pp. The one gain anywhere is RQ-KMeans new-target HR@1, which rises by
  0.6 pp.

The protocol ([protocol.md](protocol.md)) was declared before any decoding run.
It carries two dated amendments: the archived decoder turned out to be beam
sampling (amendment 1), and the plain-decoder conditions of the primary job
had silently run the sampling decoder (amendment 2). Cohort: all 3,681 AR
next-item test rows (1,606 users). Intervals are paired user-bootstrap 95%
intervals (2,000 draws, seed 20260927), in percentage points, and exclude
training-seed variance. HR is exact-SID unless stated.

| Run | Job | Output under `$SIDLENS_WORK/derived/controlled/exp6_ar_copy_in_decoding/` |
|---|---|---|
| Primary (`B`, `C_later`, `C_all`, `N_later`) | 283114 | `283114/cell-0{0,1}`, `summary-283114/` |
| Plain decoder, corrected (`B_plain`, `C_all_plain`) | 283586 (cell 00), 283621 (cell 01) | `283586/cell-00`, `283621/cell-01`, `summary-283586-283621/` |
| Pilot, 64 rows (V2, V5) | 283110 | `pilot-283110/cell-0{0,1}` |

The job-283114 `*_plain` outputs are kept but superseded: they are bit-identical
to `B` and `C_all` (amendment 2). The corrected re-run is the same code and
arguments in every submission. Two submissions died at start-up, writing
nothing, on nodes that cannot see the Python environment: job 283296 (both
cells) and task 1 of 283586. Cell 01 was then resubmitted alone as 283621.
Pilot 283100 failed on the garbage-beam crash that led to amendment 1.

## Numerical acceptance

- **V1, reproduction (cell 00).** The hooked baseline reproduces the archive
  on 3,680 of 3,681 rows (identical 50-SID lists; the one exception is the
  cohort's last row). `calc.py`-rule HR@{1, 5, 10, 20, 50} equal the recorded
  values exactly (11.22 / 20.10 / 24.07 / 27.06 / 30.32%). Acceptance (≥ 99%
  identical, HR within 0.1 pp): **passed**.
- **V2, hook neutrality.** In the 64-row GPU pilot, the hooked baseline is
  bit-identical to unhooked `generate()` from the same RNG state, in sequences
  and scores, for all 8 batches of both cells. The CPU test
  `tests/interventions/test_generation_knockout.py` shows the same on a toy
  model.
- **V5, determinism.** Re-running the pilot baseline with seed 1234 instead of
  42 leaves the top-10 unchanged in 64 of 64 rows for both cells. The full
  50-lists are unchanged in 64 of 64 rows (RQ-KMeans) and 53 of 64 (RQ-VAE):
  beam sampling moves only the tail of the list.
- **V3, manipulation.** No guard tripped. `C_all` removed edges at the first
  digit on all 184,050 beam rows (3,681 × 50). At d ≥ 1 the counts are per
  *beam*, not per cohort row. Only beams whose own prefix matches a history
  item can lose an edge, and after step 1 about 80% of beams are filler from
  beam sampling (next table). So the protocol's "majority of rows" criterion
  was not evaluated as written. The copy rate above is the direct check that
  the manipulation reached the output.

  | Cell 00, beam rows at step 1 | `C_later` | `N_later` | `C_all` |
  |---|---|---|---|
  | invalid prefix (filler beams) | 149,280 | 149,280 | 142,045 |
  | valid prefix, ≥ 1 edge removed | 7,570 | 7,164 | 5,799 |
  | edges removed, all steps | 17,468 | 16,817 | 911,165 |

- **V4, cell 01 identity (reported, not required).** `oneoff_rqvae4cb128`
  reproduces 0 of 3,681 archived RQ-VAE lists. Its `calc.py`-rule HR@10 is
  14.53% against the archive's 13.09%, so it is a different, somewhat better
  model. RQ-VAE results describe the one-off.

## Declared estimands (HR@10, pp)

| | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|
| **P1** `C_all` − `B`, repeat rows | **−27.26** [−32.31, −22.14] | **−10.11** [−14.95, −5.82] |
| **P2** `C_all` − `B`, new rows | **−2.12** [−3.12, −1.15] | **−2.97** [−3.78, −2.15] |
| **P3** P1 − P2 | **−25.14** [−30.28, −19.93] | **−7.14** [−12.06, −2.82] |
| `C_all` − `B`, all rows | −5.68 [−6.84, −4.57] | −3.31 [−4.14, −2.49] |
| `C_later` − `B`, repeat / new | −11.32 [−14.55, −8.07] / +0.06 [−0.29, +0.44] | −5.62 [−9.45, −2.37] / −2.80 [−3.60, −2.04] |
| `N_later` − `B`, all rows | −0.03 [−0.19, +0.13] | −0.11 [−0.31, +0.08] |
| `C_later` − `N_later`, all rows (paired) | −1.52 [−2.09, −0.92] | −2.83 [−3.64, −2.08] |

**Levels** (% of rows):

| | RQ-KMeans: HR@10 (repeat / new) | top-1 in history | RQ-VAE: HR@10 (repeat / new) | top-1 in history |
|---|---|---|---|---|
| `B` | 24.02 (90.98 / 12.97) | 58.4 | 14.37 (91.57 / 10.45) | 25.1 |
| `C_later` | 22.47 (79.65 / 13.04) | 30.6 | 11.44 (85.96 / 7.65) | 10.6 |
| `C_all` | 18.34 (63.72 / 10.85) | 18.4 | 11.06 (81.46 / 7.48) | 9.5 |
| `N_later` | 23.99 (91.17 / 12.91) | 59.4 | 14.26 (91.57 / 10.33) | 25.9 |

HR@1, HR@50 and NDCG@10 move the same way (`summary-283114/report.md`). One
exception: for RQ-KMeans, `C_all` leaves new-row HR@1 unchanged
(+0.22 [−0.31, +0.76]).

**Expectations recorded before outcomes**

- **E1** (P1 < 0): **held** (−27.3, −10.1).
- **E2** (|P2| < |P1|): **held** (2.1 vs 27.3; 3.0 vs 10.1).
- **E3** (`C_later` moves repeat rows less than `C_all`): **held** (−11.3 vs
  −27.3; −5.6 vs −10.1).
- **E4** (`N_later` ≪ `C_later`): **held** (−0.03 vs −1.55; −0.11 vs −2.93,
  all rows).

## Plain beam search (secondary, amendment 2)

With `use_model_defaults=False`, the plain decoder really is plain beam
search. For cell 00, its lists differ from `B` in all 3,681 rows. It has no
non-SID entries, against 18.9% under `B`. This secondary run was re-run after
the primary estimands were seen (amendment 2); its design is unchanged.

| | RQ-KMeans 3×128 | RQ-VAE 4×128 |
|---|---|---|
| HR@10, `B_plain` (repeat / new) | 24.67 (93.28 / 13.35) | 15.32 (93.82 / 11.33) |
| HR@10, `B`, archived decoder, same rows | 24.02 | 14.37 |
| Top-1 already in the history: `B_plain` / `B` | 64.5 / 58.4 | 34.0 / 25.1 |
| `C_all_plain` − `B_plain`, all rows | **−3.99** [−4.95, −3.07] | **−3.23** [−4.03, −2.45] |
| … repeat rows | **−19.39** [−23.73, −15.58] | **−8.43** [−12.78, −4.47] |
| … new rows | **−1.46** [−2.29, −0.59] | **−2.97** [−3.74, −2.18] |
| … repeat − new | −17.93 [−22.24, −13.91] | −5.46 [−9.85, −1.47] |

- **Plain search scores a little higher.** On the same rows, HR@10 is higher
  by 0.65 pp (RQ-KMeans) and 0.95 pp (RQ-VAE). No paired interval was computed
  for this difference, because `B` and `B_plain` sit in different summaries.
- **The repetition penalty was suppressing copying.** Without it, the top-1
  SID is a history SID more often: 64.5% vs 58.4%, and 34.0% vs 25.1%.
- **Copying still helps.** Blocking it lowers HR@10 under plain search too,
  and the direction of every P1–P3 analogue is unchanged. The repeat-row loss
  is smaller than under the archived decoder (−19.4 vs −27.3; −8.4 vs −10.1).
- **One exception, at top-1.** For RQ-KMeans, blocking raises new-row HR@1 by
  +0.60 pp [+0.06, +1.15]. When copying is off, the top slot sometimes goes to
  a correct new item instead of a history SID. That gain is outweighed within
  the top-10 (new-row HR@10 −1.46). Under the archived decoder the same HR@1
  contrast was +0.22 [−0.31, +0.76]. For RQ-VAE, new-row HR@1 falls
  (−1.03 [−1.80, −0.35]).

`summary-283586-283621/report.md` is headed "archived AR decoder" and shows
empty V1 lines. That is the summarizer's fixed heading: there is no `B`
condition in this run, so no V1 applies.

## Limits

- **Reads, not all routes.** The knockout removes direct reads in layers
  14–27. Indirect routes and layers 0–13 remain (exp4 K2/K6). The effects are
  lower bounds on what copying contributes.
- **Scope.** Beam 50, two checkpoints, one training seed each. The cells differ
  in quantizer and depth.
- **SID-level HR.** Collided targets count as hits when their SID is listed.
  The repeat stratum includes collision repeats; `repeat_item` rows (the
  target item itself in the history) are reported in the summary.
- **Cohort.** The AR cohort is not the diffusion cohort.
