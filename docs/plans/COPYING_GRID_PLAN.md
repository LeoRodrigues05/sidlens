# Plan: running our analyses on the coauthor's grid

Written 2026-10-21 and revised twice the same day after your reviews. Nothing
in it has run, and nothing is committed or pushed.

## Changes in revision 3

- **Two run scripts.** `scripts/grid/run_best_state.sh` runs every experiment
  on every configuration at the best state (about 1,270 GPU-hours).
  `scripts/grid/run_full.sh` runs the same plus the cheap set at every
  intermediate checkpoint (about 1,290 GPU-hours). Finished cells are always
  skipped, so running the best-state script first and the full one later
  repeats nothing.
- **Only the cheap set runs at the intermediate checkpoints** (your decision):
  AR exp3 Part A and exp4 Part R; DiffGRM exp7's replacement and copying parts.
  exp4 Part K, exp7's knockouts, exp8 and exp12 Part H run at the best state
  only. This supersedes "every weight state" in §4.1, §4.2 and §5.1.

## Changes in revision 2

- **Every experiment runs on every configuration.** The focal set is gone.
- **The core copy estimates run at every weight state of every
  configuration**, in both paradigms.
- **Defaults confirmed:** N2(f) is kept; results go to
  `$SIDLENS_WORK/results_remote/`; Office experiments that run a model use
  6,297 sampled users; no cuts.
- **Cost recomputed** from the per-part run times in our earlier job logs
  (§5.1): about 1,400 GPU-hours and 220 CPU-hours.
- **The fixed-shape runner pads by length bucket** (§3.2), since a few
  histories reach 50 items.

## Changes in revision 1

- **Scope.** Every analysis we have run before, ported to the grid, plus four
  additions: the prefix-conditioning test (N1), how the AR model matches the
  prefix (N2), dose-response (N3) and no-training baselines (N4). The other new
  experiments of the first version are dropped (§4.4).
- **Option 1, as is.** The coauthor clones sidlens and runs one script.
  OneDiffRec is not changed.
- **No commits or pushes** until you have reviewed the code.
- **The grid is complete.** The coauthor has finished all four blocks; what
  remains is to send the code.
- **Both catalogs treated the same.** Every scope applies equally to Industrial
  and Office.
- **One results folder, `results_remote/`** (§2.3).
- **History length checked on real files** (§1.4).

## 0. Summary

The coauthor's four blocks hold 372 runs. Each run's checkpoints, test lists
and cohort files sit in a fixed layout (§1). sidlens cannot read them yet: the
cohorts, the decoders, the DiffGRM code version and the SID tables differ from
the frozen substrate, and sidlens must learn to rebuild the AR prompt that
OneDiffRec now uses (§1.5). The work is an adapter in sidlens, ports of our
existing experiments to it, code for N1–N4, and one superscript the coauthor
runs (§3). Every result lands in `results_remote/`, which comes back to us
whole. The run costs about 1,400 GPU-hours and 220 CPU-hours: about 4.5 days
on 16 GPUs or 2.3 days on 32 (§5.1). The superscript's pilot step replaces
these estimates with measured numbers before the full run.

## 1. The coauthor's runs

### 1.1 What exists

The full grid of the OneDiffRec handoff: run name `grid-v1`, the v3 SID tables
(the ones the coauthor uploaded as `WFRaain/onediffrec-sid-codebooks`, which we
analysed in structure exp4), AR (Qwen2.5-1.5B SFT) and DiffGRM, seed 42.

| Block | Native maps | ZCR twins | Runs (AR + DiffGRM) | Test events |
|---|---|---|---|---|
| Industrial next-item | 27 | 21 (9 RQ-KMeans, 9 MQ, RQ-VAE 3/4/5×256) | 96 | 6,297 |
| Industrial next-two | 27 | 21 | 96 | 6,297 |
| Office next-item | 27 | 18 (9 RQ-KMeans, MQ at widths 256/512, RQ-VAE 3/4/5×256) | 90 | 61,564 |
| Office next-two | 27 | 18 | 90 | 61,564 |

Office MQ at width 128 has no ZCR twin: its prefix groups are larger than the
codebook. Each AR run keeps three weight states (epochs 0.5 and 1.5, and the
best model by validation loss). Each DiffGRM run keeps its best model (by
validation NDCG@10) and a snapshot every 20 epochs until early stopping; our
old DiffGRM runs stopped between epochs 37 and 57, so expect one to three
snapshots per run.

### 1.2 Directory layout

`WORK_ROOT` is `onediffrec-handoff-v1/work` unless the coauthor set it.

```text
WORK_ROOT/
  shared/envs/onediffrec/                 Python env: torch 2.6 cu124, transformers 4.57.1, numpy 1.26
  shared/models/Qwen2.5-1.5B-Instruct/    AR base model
  {industrial,office}-v3/                 the grid
    data/onediffrec/Amazon18/<Category>/  inter.json, item.json, item2id, user2id, review.json,
                                          emb-qwen-td.npy, the 27 v3 index tables
    data/diffgrm-cache/.../processed/     <variant>.sem_ids, all_item_seqs.json, id_mapping.json
    data/assignment-caches/<ds>/<label>/  DiffGRM caches of the ZCR maps
    tokenizers/<ds>/<variant>/, zcr/<ds>/<variant>/
    prepared/disjoint_horizon_v3/analysis-v3/<ds>/<predtype>/<label>/
                                          train/val/test.csv, {train,val,test}-cohort.json, data-manifest.json
    sources/<sha256>/                     the code snapshot every run executed
    experiments/grid-v1/full/<ds>/<predtype>/<ar|diffgrm>/<label>/seed-42/
      cell.json  train.log  eval.log  provenance/{train,eval}/  resources/
      training/   AR:      final_checkpoint/ (best), snapshots/epoch-0.5/, snapshots/epoch-1.5/,
                           snapshots/index.json, sft-run-stats.json
                  DiffGRM: pytorch_model.bin (best), resolved-config.json,
                           intermediate/epoch-NNNN.bin (every 20 epochs), history.jsonl
      evaluation/ predictions.jsonl (top 10 of 50 beams per event), metrics.json, protocol.json
```

`<label>` is the variant (`rqkmeans_3codebook_128`) for a native run and
`<variant>__zcr-<16 hex>` for a ZCR run.

### 1.3 What sidlens reads

| Need | File |
|---|---|
| Run identity, settings, SID-table hash | `cell.json` |
| Test and validation events | `prepared/.../test.csv` (AR), `test-cohort.json` (DiffGRM); both hashed in `data-manifest.json` |
| SID table | `data-manifest.json` → `input_paths.sid_index` (the ZCR table for ZCR runs) |
| AR weights and tokenizer | `training/final_checkpoint/`, `training/snapshots/<state>/` |
| DiffGRM weights and architecture | `training/pytorch_model.bin`, `training/intermediate/*.bin`, `training/resolved-config.json` |
| Saved test lists | `evaluation/predictions.jsonl` |
| Review days, titles | the data root (`review.json`, `item.json`) |
| Recorded hashes | `provenance/{train,eval}/result.json` |

AR rows name items by integer index and DiffGRM rows by ASIN; both carry the
same `example_id`, so every event is paired across paradigms, maps, native and
ZCR, and weight states.

### 1.4 History length

Every run, in both paradigms, both catalogs and both tasks, gives the model
the user's earlier events up to a cap of 50 (the same splitter,
`max_hist_len=50`, prepares every cohort; DiffGRM's `max_history_len` is 50).
The cap is rarely reached:

| Test events | Count | Mean | Median | 90th percentile | At the cap of 50 |
|---|---|---|---|---|---|
| Industrial next-item | 6,297 | 5.84 | 5 | 9 | 2 |
| Industrial next-two | 6,297 | 4.84 | 4 | 8 | 2 |
| Office next-item | 61,564 | 6.62 | 5 | 11 | 97 (0.16 %) |
| Office next-two | 61,564 | 5.62 | 4 | 10 | 90 (0.15 %) |

Checked on 2026-10-21:

- On the handoff's prepared Industrial cohorts, AR and DiffGRM get the same
  events and the same history items in the same order for every row, and the
  events and histories are identical across maps.
- The Office numbers come from our frozen `Office.inter.json`, which is
  byte-identical to the handoff's (same sha256).
- The longest possible AR training input (depth 5, 50 history items, next-two
  target) is 438 tokens, under the SFT cutoff of 512, so training never cut a
  history. Evaluation has no cutoff.

The frozen AR cohort used at most 10 history items while DiffGRM used up to 50;
in the grid both paradigms see the same history.

### 1.5 What differs from the frozen substrate

1. **Cohorts.** Both paradigms use leave-last-out per user, on the same events.
   The historical cohort loaders do not read these files.
2. **AR prompt.** Nothing about the prompt changes in our analysis: every model
   gets exactly the prompt it was trained and tested with. sidlens, however,
   currently rebuilds only the frozen prompt. OneDiffRec's prompt now uses the
   training wording for testing too, and commit `a165f65` removed two spaces
   (after "completes the request." and after "### User Input:"). sidlens needs
   a third template that rebuilds this prompt; every token is checked against
   OneDiffRec's own code (V1, §3.6).
3. **AR decoding.** Plain beam search, 50 beams, a catalog trie at every step;
   the archived lists were beam sampling.
4. **DiffGRM.** Trained with `DiffGRM_new` (its model file is byte-identical
   to sidlens's `vendor/diffgrm_new`); decoded left to right with a catalog
   check at every step (batch 8, beam 50). `resolved-config.json` records
   every architecture field.
5. **SID tables and catalogs.** Per-run v3 tables, native or ZCR, two catalogs.
   sidlens's table and timestamp loaders read only frozen Industrial files.

## 2. How the coauthor runs our code (option 1)

### 2.1 What we send, after your review

- The sidlens code, by push or as a tarball (your choice once you have
  reviewed it).
- `external/labels/` (36 MB, both catalogs, hash-checked), which some
  experiments read for categories and brands.

The coauthor needs nothing else from us: the checkpoints, cohorts, tables, base
model and environment are already on their cluster.

### 2.2 What the coauthor runs

```bash
bash scripts/grid/sidlens-grid.sh setup --onediffrec-work /path/to/onediffrec-handoff-v1/work
bash scripts/grid/sidlens-grid.sh register   # runs and weight states found; anything missing or failed
bash scripts/grid/sidlens-grid.sh pilot      # one short cell per experiment; prints measured cost
bash scripts/grid/run_best_state.sh          # every experiment, every configuration, best state
bash scripts/grid/run_full.sh                # the same plus the cheap set at every intermediate checkpoint
bash scripts/grid/sidlens-grid.sh status
bash scripts/grid/sidlens-grid.sh export     # packs results_remote/ with a hash manifest
```

`setup` builds sidlens's environment on top of the OneDiffRec environment (the
same torch and transformers that produced the saved lists), writes the site
file and runs the CPU tests. Slurm settings use the names the coauthor already
uses for OneDiffRec (`SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_QOS`,
`SLURM_EXCLUDE`), plus `MAX_GPUS`, the number of GPU jobs to run at once. `run`
can be repeated: finished cells are skipped and failed ones are rerun into new
folders.

### 2.3 The results folder

Everything the pipeline writes goes to `$SIDLENS_WORK/results_remote/` on the
coauthor's cluster. It comes back whole and goes to the same path here.

```text
results_remote/
  README.md                       sidlens commit, OneDiffRec work root, dates, commands
  MANIFEST.sha256                 every file's hash
  registry/                       runs and weight states, input hashes, coverage, event samples
  checks/<ds>/<task>/<paradigm>/<map>/
                                  input checks (V1) and reproduction of the saved lists (V2), per run
  <group>/<experiment>/           group = controlled | representation | retrospective | additions
    <ds>/<task>/<paradigm>/<map>/<state>/
                                  one cell: arguments.json, inputs.json, validation.json,
                                  results (*.parquet, *.csv), run.log, status.txt, output.sha256
    summary/                      estimates.csv, report.md, figures/
  report/                         one combined report
  provenance/                     source archive and hashes of the code that ran, environment, Slurm logs
```

`<map>` is the variant, with `__zcr` for a ZCR twin; `<state>` is `best`,
`epoch-0.5`, `epoch-1.5` or `epoch-NNNN`. Experiment folders keep their sidlens
names (for example `controlled/exp4_ar_copy_circuit`), and the four additions
go under `additions/`. A failed cell is renamed `<state>.failed-<date>` and
rerun; nothing is overwritten. Activation captures for the representation
experiments are working files in `$SIDLENS_WORK/scratch/`, deleted for each
model once its experiments finish; the SAE weights and the analyses go in
`results_remote/`. Expect `results_remote/` to reach 100–200 GB and scratch to
peak near 1 TB.

### 2.4 How results come back

`export` writes `results_remote/` with its manifest as a tarball. The coauthor
sends it, or pushes it with `sidlens bundle push` to a private Hugging Face
dataset repository. Here it is unpacked to `$SIDLENS_WORK/results_remote/` and
checked against the manifest. Analyses we later run here on these outputs
write into the same folders.

## 3. Changes to sidlens

### 3.1 The grid adapter (new package `src/sidlens/grid/`)

| Module | Job |
|---|---|
| `layout.py` | Read a OneDiffRec work root: runs, weight states, prepared cohorts, data root, saved lists. Never writes into it. |
| `registry.py` | One entry per (run, weight state), with the sha256 of everything an experiment reads, checked against the run's provenance and `cell.json`. |
| `inputs.py` | Build the objects the experiments already use (`ArExample`, `SidTable`, `SidVocab`, `DiffusionEvalCohort`, the DiffGRM model) from grid files. |
| `decode_ar.py`, `decode_diff.py` | The two v3 decoders, with sidlens's intervention hooks; with no intervention they must reproduce the saved lists (V2). |
| `events.py` | What our existing experiments split by: target classes (repeat, SID partner, near-duplicate variant, new), review-day gaps and duplicate records, for both catalogs. |
| `samples.py` | The fixed Office sample of 6,297 users (and matching training-row samples for the SAEs), drawn once with a recorded seed. |
| `pipeline.py` | The superscript's planner: stages, cells, Slurm arrays with dependencies, skip-if-finished, cost projection. |

### 3.2 Changes to existing code

- `vendor/onediffrec_v3/`: byte copies of the OneDiffRec files the grid depends
  on (the two evaluators, `joint_decode.py`, `scripts/ar/data.py`, the cohort
  builder and splitter, the metrics), with their own manifest that `verify`
  checks. The base snapshot manifest is not touched. At run time every
  experiment also checks that the run's own `sources/<sha>/` copy has the same
  hashes.
- `provenance/profiles.py`: a `grid` profile, so the job gate works without the
  frozen snapshot.
- `data/ar_prompts.py`: a template that rebuilds OneDiffRec's current prompt,
  and a loader for prepared CSVs. The existing templates are not touched.
- `interventions/runner.py` (the fixed-shape runner): pad each row to the
  longest prompt in its length bucket, not in the whole split. A few histories
  reach 50 items, and padding every row to them would make each forward about
  three times longer. A row's clean and intervened inputs always share a
  bucket, so the bit-exact controls still hold.
- `models/ar.py`, `models/diffusion.py`, `data/diffusion_eval.py`,
  `data/timestamps.py`, `scripts/activations/ar_capture.py`: accept grid
  checkpoints, cohorts and review days for both catalogs.
- `interventions/`: patching of an attention head's query, key or value input
  (N2), and scaling of chosen attention logits (N3). Each comes with a no-op
  test that must be bit-exact.

### 3.3 Porting the existing experiments

Each `run.py` gets one input step: frozen checkpoint names take today's path
unchanged; grid names take the adapter. The analysis below that step does not
change. The archive checks are replaced by the reproduction check (V2).

### 3.4 Code for the four additions

Each gets a folder with `protocol.md` (written before any outcome), `run.py`,
`summarize.py`, `README.md` and `RESULTS.md`, as our other experiments do.

### 3.5 The superscript

`scripts/grid/sidlens-grid.sh` wraps `python -m sidlens.grid.pipeline` (the
commands of §2.2). Stages, in dependency order: `register`; `checks` (V1, and
V2 at the best state); `cpu` (the retrospective analyses and N4); `causal`
(teacher-forced experiments); `decode` (beam-search experiments); `features`
(activation captures, probes, SAEs); `summarize`; `export`. The stages a cell
needs must finish first (Slurm `afterok`). Only runs that pass V1 and V2 enter
the experiments that run a model. All per-cluster settings come from the site
file.

### 3.6 Validation gates

- **V0, frozen path unchanged:** the 64-row pilots of exp4, exp6 and exp7 on the
  frozen checkpoints give identical outputs with the current and the modified
  code, run in the same job on the same node.
- **V1, inputs:** hashes match the run's provenance and `cell.json`; prompt
  token ids equal those of OneDiffRec's own `SidSFTDataset` on every row of a
  200-row sample per run; the vocabulary covers the table; AR and DiffGRM test
  sets have the same `example_id`s.
- **V2, reproduction (best state):** DiffGRM gives the saved top 10 exactly
  (ties within 1e-4). AR gives identical top-10 lists for at least 99 % of
  events and joint log-probs within 1e-3. Batched AR decoding is used only if it
  passes; otherwise batch 1, as the evaluator ran.
- **V3:** each protocol's no-op controls, bit-exact.

### 3.7 Tests

New `tests/grid/`: layout parsing on a synthetic work root (with AR snapshots
and ZCR labels), prompt ids against the vendored `data.py`, registry hashing,
the v3 DiffGRM decoder on a random model against the vendored `joint_decode`,
length buckets against whole-split padding, the N4 rule on hand-built
histories, and stage planning. The handoff smoke runs on `/l` (AR and DiffGRM,
native and ZCR, both tasks) are the integration test: the adapter, V1, V2 and
one cell of every stage run on them before anything is sent.

## 4. Experiments

### 4.1 Scopes

- **Every configuration:** every next-item run of both catalogs (93 per
  paradigm: 27 native and 21 ZCR on Industrial, 27 and 18 on Office); every
  next-two run (93 per paradigm) for the next-two experiment; every run of all
  four blocks for analyses of saved lists.
- **Every weight state:** the core copy estimates also run at AR epochs 0.5 and
  1.5 and at every DiffGRM snapshot, for every configuration. Decoding
  experiments run at the best state only, because only the best state has saved
  test lists to check our decoders against.
- **Events:** all 6,297 Industrial test events; a fixed sample of 6,297 Office
  test users for every experiment that runs a model; full cohorts for the
  analyses of saved lists.

### 4.2 Everything we have run before, on the grid

"Every configuration" below means every next-item run of the paradigm named,
both catalogs, at the best state.

| Experiment | What it measures | Runs on |
|---|---|---|
| controlled exp3, Part A | Replacing a history item at each recency; how much a shared prefix keeps | Every configuration (AR); also every weight state |
| controlled exp3, Part B | Residual-stream patching | Every configuration (AR) |
| controlled exp4, Parts R and K | Prefix-matched copying; the single attention-edge knockout; early vs late layers | Every configuration (AR), test and validation splits; test split also at every weight state |
| controlled exp4, Part H | Head screen, tested on held-out users | Every configuration (AR) |
| controlled exp5 | Next-two: item 2 from item 1 | Every next-two configuration (AR) |
| controlled exp6 | Copy reads blocked inside beam search; Hit@10 by target class | Every configuration (AR) |
| controlled exp9 | List position against review time (swap of the two most recent items) | Every configuration (AR) |
| controlled exp10 | Last digit: copy or avoid; native against ZCR | Every configuration (AR) |
| controlled exp11 | Its every-map design is now the default | — |
| controlled exp1, exp2 | DiffGRM reveal policies at equal beams; six fixed orders at depth 3 | Every configuration (DiffGRM; exp2 on the 31 depth-3 configurations) |
| controlled exp7, exp8 | DiffGRM history use; the copy route through the encoder | Every configuration (DiffGRM); also every weight state |
| controlled exp12 | DiffGRM paired-run patching (Parts H and S) | Every configuration (DiffGRM); Part H also at every weight state |
| controlled exp13 | DiffGRM step switch | Every configuration (DiffGRM) |
| controlled exp14 | DiffGRM copy block in beam search; swap; shift | Every configuration (DiffGRM) |
| representation exp1 | AR logit lens and probes on activation captures | Every configuration (AR) |
| representation exp2 | AR SAEs, copy latents and their ablation | Every configuration (AR) |
| representation exp3 | AR information map: SAEs every second layer, census, tuned lens, SID decoders | Every configuration (AR) |
| representation exp4 | DiffGRM reveal trace | Every configuration (DiffGRM) |
| representation exp5 | DiffGRM encoder probes, SAEs, description autoencoder | Every configuration (DiffGRM) |
| retrospective exp2 | First wrong digit of the top-1 | All runs, both paradigms (saved lists) |
| retrospective exp3 | SID-level against item-level credit; collisions | All runs (saved lists) |
| retrospective exp4 | Next-two association | All next-two runs (saved lists) |
| retrospective exp6 | Review-day structure of the data and of the outputs | All runs (saved lists and data) |
| retrospective exp7 | exp6's knockout re-scored at item level by target relation | Every configuration (AR, on exp6's lists) |
| structure exp1–exp4, semantic mapping | Map statistics | Not rerun: structure exp4 already covers these exact tables in both catalogs; `register` checks the hashes |
| retrospective exp1 | Archived reveal orders at unequal beams | Not applicable (no archived runs); controlled exp1/exp2 cover its question |
| substrate exp1 | AR retraining | Not applicable; the grid replaces it |

### 4.3 The four additions

Terms: for a history H and a SID s, the **copy set** at digit d is the set of
digit-d codes of history items whose first d digits equal s's. The **history
LCP** of s is the longest prefix s shares with any history item.

**N1. Prefix-conditioning test** (both paradigms; every configuration; best
state). Does the copy need the whole prefix to match, or only the digit token
just before it (the pattern of a plain induction head)? For digit d ≥ 2 and a
history item h whose first d digits equal the target's (golden prefix given),
replace h by:

| Condition | Replacement h′ | Supports h's digit d under |
|---|---|---|
| full | h itself | both readings |
| suffix-only | same digits d−1 and d as h, an earlier digit different from the target's | last-token matching only |
| none | first d digits differ from the target's, digit d differs from h's | neither (baseline) |
| prefix-only | first d digits equal the target's, digit d differs from h's | prefix matching moves the copy to h′'s digit d |

Only rows where no other history item supports h's digit d are used. Copy
support is the log-prob of h's digit-d code under a condition minus under
`none`; the estimand is ρ = support(suffix-only) / support(full). Prefix
conditioning predicts ρ ≈ 0; ρ near 1 would mean last-token copying, and the
paper would have to say so. Catalog items are used where they exist, otherwise
code combinations outside the catalog, reported separately. MQ, whose digits
are parallel partitions, is reported separately. Depth-3 maps contribute digit
3 only.

**N2. How the AR model matches the prefix** (AR; every configuration; best
state). These experiments change internal attention inputs in their own test
conditions only; the prompt and the model are never changed, and every
comparison is against the unmodified run.

- (a) Attention from the readout position to h's digit-d token, per head, in
  N1's full, suffix-only and none rows.
- (b) In layers 14–27, replace only the key inputs (or only the value inputs) at
  h's digit-d token with those of the suffix-only run. If the key replacement
  removes the copy and the value replacement does not, the match is decided at
  the key position.
- (c) Block, in layers 0–13, the attention from h's digit-d token to h's earlier
  digit tokens. If the keys learn h's prefix there, prefix conditioning weakens
  (ρ rises) or the copy drops.
- (d) Replace the query input at the readout with one from a run whose golden
  prefix matches a different history item; the attention and the copied code
  should move to that item.
- (e) The copy heads' direct contribution to the copied code's logit.
- (f) Whether the copy heads are Qwen's pretrained induction heads: induction
  scores of the 336 heads of the base Qwen2.5-1.5B-Instruct on repeated random
  token sequences (computed once), against each head's copy effect from exp4's
  head screen.

**N3. Dose-response** (both paradigms; every configuration; best state). Scale
the copy reads by α ∈ {0, 0.25, 0.5, 1, 1.5, 2} inside beam search: add log α
to the attention logits of the edges exp6 blocks (AR), and to the attention
that exp14's block cuts (DiffGRM). α = 0 is exp6's and exp14's block; α = 1
must equal the plain decode exactly. Output: Hit@10 on repeat targets and on
new targets against α, at SID and item level, and the share of history SIDs in
the top 10.

**N4. No-training baselines** (CPU; every configuration; full cohorts).

- A copy rule: rank catalog SIDs by history LCP, then by the recency of the
  history item that gives it, then by training popularity.
- Training popularity, and popularity among items sharing the most recent
  item's first code.
- Hit@10 and NDCG@10 at SID and item level, by target class.
- For each model: how often its top-1 agrees with the copy rule's (first digit,
  whole SID), and how its new-target hits compare with the popularity
  baselines.

### 4.4 Dropped from the first version

Copy census, copy mass and calibration, the predictors of copying, copying
over training as its own analysis (the core estimates still run at every
weight state), DiffGRM prefix steering, copy-source agreement between
paradigms, the `AR_GRM` backbone, the new error breakdown (retrospective exp2
stays), the paradigm gap by target class, gated decoding, retraining without
repeat targets, and extra seeds.

### 4.5 The paper's research questions

"RQ wording" in the first version meant the phrasing of the paper's three
research questions. It is text in the paper only; no model, prompt or code
depends on it. How the experiments serve your three questions:

- **RQ1, when do they copy:** exp3 A (recency), exp4 R (prefix-matched copying
  on every map), exp7, exp9 and retrospective exp6 (position and time), N4 (how
  rule-like each model is), the core estimates at every weight state.
- **RQ2, how copying works inside each backbone:** exp3 B, exp4 K and H,
  representation exp1–exp3 (AR); exp8, exp12, exp13, representation exp4–exp5
  (DiffGRM); N1, N2.
- **RQ3, when copying helps or fails:** exp6 and exp14 (by target class),
  retrospective exp2, exp3 and exp7 (errors and collision credit), exp10 (native
  against ZCR), exp5 (next-two), N3, N4.

### 4.6 Statistics

As before: paired user bootstrap, 2,000 draws, fixed recorded seed; events
paired across paradigms, maps, native/ZCR and weight states by `example_id`;
intervals exclude training-seed variance; SID-level and item-level results kept
apart; grid claims as declared counts of runs, as in exp11; the four additions
declared before their first run on a grid model.

## 5. Cost and order of work

### 5.1 Cost

Unit times are the per-part times in the logs of our earlier jobs (one A100;
AR on 3,681 events with at most 10 history items; DiffGRM on 6,297 users),
scaled to 6,297 events per catalog and averaged over SID depths. Deeper maps
take longer: exp4's knockouts took 15 minutes at depth 3 and 21 at depth 4.

| Work | Runs | Minutes per run | Hours |
|---|---|---|---|
| exp3 Part A | 93 AR | 7 | 11 GPU |
| exp3 Part B | 93 AR | 41 | 64 GPU |
| exp4 R and K, test and validation | 93 AR | 36 + 36 | 112 GPU |
| exp4 Part H | 93 AR | 37 | 57 GPU |
| exp5 | 93 AR next-two | 35 | 54 GPU |
| exp6 (four decodes) | 93 AR | 70 | 109 GPU |
| exp9 (two decodes) | 93 AR | 34 | 53 GPU |
| exp10 | 93 AR | 2 | 3 GPU |
| representation exp1 | 93 AR | 145 | 225 GPU |
| representation exp2 | 93 AR | 30 GPU + 70 CPU | 47 GPU, 109 CPU |
| representation exp3 | 93 AR | 120 GPU + 5 CPU | 186 GPU, 8 CPU |
| N1 | 93 AR | 15 | 23 GPU |
| N2 | 93 AR | 50 | 78 GPU |
| N3 (six decodes) | 93 AR | 105 | 163 GPU |
| Weight states: exp3 A, exp4 R and K (test) | 93 AR × 2 states | 43 | 133 GPU |
| DiffGRM: exp1, exp13, exp14, representation exp4 and exp5, N1, N3 | 93 DiffGRM | 51 | 79 GPU |
| DiffGRM: exp2 | 31 depth-3 DiffGRM | 7 | 4 GPU |
| DiffGRM: exp7, exp8, exp12 | 93 DiffGRM | 25 | 38 CPU |
| DiffGRM weight states: exp7, exp8, exp12 Part H | about 190 snapshots | 18 | 55 CPU |
| Registry, checks, retrospective analyses, N4 | all 372 runs | — | about 15 CPU |
| **Total** | | | **about 1,400 GPU (1,320 AR, 80 DiffGRM) and 220 CPU** |

Wall-clock time, with about 25 % added for queueing, stage dependencies and
reruns (the CPU work runs alongside):

| GPUs at once | Days |
|---|---|
| 16 | about 4.5 |
| 32 | about 2.3 |
| 64 | about 1.2 |

How the estimate could be wrong:

- **AR decoding batch size.** exp6, exp9 and N3 (about 325 GPU-hours) assume
  batched decoding, as our earlier exp6 used. If our batched decoder does not
  reproduce the saved lists (V2) and must decode one event at a time, as the
  coauthor's evaluator did, these could take two to four times as long (300 to
  1,000 GPU-hours more). The pilot checks this first.
- **The representation experiments** (about 460 GPU-hours) are scaled from two
  runs each; they could be off by half.
- **GPU type.** The units are A100 times.

The pilot takes a few GPU-hours and half a day including queueing.

### 5.2 Order of work

Here, with no commits:

1. The adapter, vendored files, `grid` profile, prompt template, length buckets
   and tests; integration tests on the handoff smoke runs.
2. Port the existing experiments; pass V0.
3. Protocols for N1–N4, written before any outcome.
4. Code for N1–N4, tested with 64-row pilots on the frozen checkpoints (code
   tests, not results).
5. The superscript end to end on the smoke runs (`plan`, `pilot`, a local
   run), and a short runbook for the coauthor.
6. Hand everything to you for review.

After your go-ahead: send the code; the coauthor runs §2.2; `results_remote/`
comes back.

## 6. Risks

| Risk | Handling |
|---|---|
| Some grid runs failed or are incomplete | `register` lists them; every estimate states how many runs it covers. |
| Our decoders do not reproduce the saved lists | V2; a run that fails is excluded from decoding experiments and listed. Batch-1 decoding is the fallback (§5.1). |
| The coauthor's env differs from ours (numpy 1.26 there, 2.4 here; Python version unknown) | `setup` runs the full CPU test suite in their env before any job; fixes go in before anything runs. |
| Porting changes the frozen-path results | V0. |
| Scratch space for activation captures | Captures are deleted per model once its experiments finish; `pilot` reports the peak. |
| Code combinations outside the catalog in N1 | Reported separately from catalog items. |
| One seed per configuration | Stated with every interval. |

## 7. Decisions

Taken: option 1 as is; no commits or pushes before your review; every
experiment on every configuration; the core estimates at every weight state of
every configuration; N2(f) kept; `results_remote/` under `$SIDLENS_WORK`; an
Office sample of 6,297 users; no cuts.

Still open, after your review: how the code reaches the coauthor (push or
tarball), and how `results_remote/` comes back (tarball or a private Hugging
Face dataset).

## 8. Implementation notes (2026-10-21)

What the code does where the plan left a choice, or where building it changed
something. Everything is uncommitted, for review.

- **Results layout.** As §2.3. The AR representation experiments run as one
  chain per run (`representation/ar_chain`: captures in scratch, rep exp1, the
  SAE stages of rep exp2 including amendment A1, rep exp3, then the captures
  are deleted); their outputs land in the three experiment folders.
- **AR decoding experiments use the runs' own decoder** (`grid/decode_ar.py`,
  OneDiffRec's batch-1 plain beam search plus raw-vocabulary rescoring; copy
  edits applied in both phases). exp6: V2a re-decodes the first 512 rows in
  CSV order and must reproduce the saved lists; `B_plain` is then the saved
  list itself, so the four conditions are the four decodes the cost table
  assumed. N3 and exp9's decode arm use the same harness.
- **DiffGRM.** `grid/diff_setup.py` builds every DiffGRM experiment's inputs
  from a grid state and replaces the exp1-archive check with two checks: the
  run's own decoder reproduces its saved lists on the evaluator's first 64
  users (full cohort order, so batches match), and sidlens's state scorer
  equals the evaluator's per-digit forward. exp13 and exp14 decode with the
  run's own left-to-right catalog search (step t is digit t there); exp1 and
  exp2 keep `matched_decode` (reveal policies are their question) and add the
  run's own search as a condition (`grid_l2r`; `grid_fixed_<order>`). On the
  smoke run, exp14's block, N3's alpha = 0 and the mask block give identical
  lists.
- **Saved-list analyses.** Retrospective exp2, exp3, exp4 and exp6 are
  recomputed from the grid's one prediction schema by
  `grid/saved_lists.py` (their headline estimands, every run, both
  paradigms); retrospective exp7 re-scores each AR run's exp6 cell.
  Retrospective exp1 has no grid counterpart (no archived reveal orders).
- **Time for both catalogs.** `timestamps.load_event_times` now takes any
  catalog or data root. Office needed two additions: identical repeated
  10-item windows are matched to positions in occurrence order, and one
  position (user 61413, four copies of one item over two adjacent days) has no
  unique day; with `on_ambiguous="flag"` it takes the earliest fitting day and
  every row touching it is marked `time_ambiguous` and left out of time
  splits. Industrial is unchanged (strict mode).
- **Near-duplicate pairs** for target classes and exp10 come from structure
  exp4 (both catalogs) and travel with the code in the inputs tarball
  (`derived/grid_inputs/`), checked against the run's `item2id` hash.
- **Additions.** Folders `experiments/additions/n1_prefix_conditioning`,
  `n2_ar_prefix_match`, `n3_copy_dose`, `n4_no_training_baselines`, each with
  its protocol. N1 and N3 cover next-item runs of both paradigms; N4 every run.
- **Validation.** CPU suite 405 passed. V0 for the DiffGRM runners (CPU job
  329437): the original exp7, exp8, exp12, exp13, exp14 and representation
  exp4/exp5 runners and the ported ones give identical result files on the
  frozen checkpoints (8 users; rep exp5 64). The GPU pilot on the handoff smoke runs
  (Industrial AR next-item and next-two, DiffGRM next-item; Office DiffGRM
  next-two) and the V0 check of the ported AR runners are recorded in
  `$SIDLENS_WORK/scratch/grid-validation/`.
