# Plan: step-level patching and intermediate solutions in DiffGRM

Approved 2026-10-13. **Numbering as run (2026-10-14):** D1 became `controlled/exp13_diffusion_step_switch` (Part A only) and D2 became `controlled/exp12_diffusion_activation_patching`, the reverse of the draft numbering below; both ran on all 20 checkpoints under `DIFFUSION_MECHANISMS_PLAN.md`. Status at the end of 2026-10-13: the shared code (section 5) and the D0 protocol and runner exist and are tested on a random-weight model; the work root was rebuilt on `/nfs-stor` from the Hub bundle and the reveal trace (D0) ran on all 20 checkpoints, 16 accepted. See `experiments/representation/exp4_diffusion_reveal_trace/README.md`.

## Context

The AR side of SidLens now has a causal account of how a recommendation is made
(copying from a prefix-matching history item, read by late attention). The
diffusion side has two results on *what history is used* (exp7, exp8) and two on
*search* (controlled exp1, exp2), but nothing on the part that is specific to
diffusion: how the answer is built across reveal steps and across digit
positions. You proposed patching the "predicted noise" at step t with the
prediction from another run. This plan checks that idea against the DiffGRM
code, restates it in a form the model supports, and lays out the experiments,
the code changes and the feasibility limits. It was written before any code
change; the status line above says what has happened since.

## 1. Where the project stands

| Layer | Established | Open for diffusion |
|---|---|---|
| Map (structure exp1–3, semantic mapping) | RQ digits mean something only given the prefix; MQ digits are parallel partitions; each quantizer collides for a different reason | Whether the decoder *uses* the hierarchy between digits |
| AR computation (controlled exp3–6, 9, 10; representation exp1–3) | Prefix-matched copying, read through attention in layers 14–27, causal SAE latents from layer 20, follows list order not time | — |
| Diffusion history use (exp7, exp8) | Same behaviour as AR (0.56 / 0.48 nats for the recent item). The copy runs through the encoder; the direct decoder read carries nothing alone | Only at fixed states (all masked, golden prefix). Never on the decoding path, never between digits |
| Diffusion search (controlled exp1, exp2) | Confidence vs fixed order depends on beam cap and depth; 57–91 % of the final beam is duplicate paths | No per-step record exists: which digit is revealed first, when the answer is settled, what the beam holds at each step |
| Evaluation (retro exp3, 6, 7) | SID hits overstate item hits; copy benefit sits in same-day, repeat and same-SID rows | Not re-scored for DiffGRM |

Gaps this plan fills: (a) no trace of intermediate states exists
(`matched_decode.decode` returns only the final list); (b) digit-to-digit
computation in the decoder is untested; (c) `RUN_PLAN.md` item 3 ("DiffGRM
layer and reveal-step readout, 20 checkpoints") is still "to write".

## 2. How DiffGRM makes a recommendation (read from the code)

Source: `OneDiffRec/DiffGRM/genrec/models/DIFF_GRM/{model,beam}.py`. I compared
9 files there and 3 in `DiffGRM_new` against `sidlens/vendor/`; all are
byte-identical, so `vendor/` is the same code.

1. **Encoder, once per user.** Each history item becomes one token:
   `item_mlp(concat of its digit embeddings) + pos_emb_enc(slot)`. One encoder
   layer, then `ln_f`, then padded slots are zeroed (`model.py:544-603`).
2. **State.** A partial SID: one decoder position per digit, holding either the
   code's embedding or that digit's mask embedding. No decoder position code and
   **no timestep input** (`model.py:938-956`).
3. **One step.** Four decoder blocks (self-attention over the digit positions,
   cross-attention to the history slots, MLP), then `ln_f` and a dot product
   with the digit's slice of the embedding table. Output: a distribution over
   K codes at every position (`model.py:958-1009`). Training puts the loss on
   masked positions only (`model.py:862-871`), so outputs at revealed positions
   are untrained.
4. **Commit.** Beam search over (position, code) pairs on accumulated
   log-probability; the chosen code is written into the state
   (`beam.py:249-258`, `351-409`).
5. **Repeat** for `n_digit` steps. Legality is checked only at the end.

Two facts decide the experiment design:

- **There is no noise prediction.** DiffGRM is masked (absorbing) discrete
  diffusion. A grep of OneDiffRec finds no Gaussian noise, schedule or timestep
  code. The network predicts the clean code at each masked position (an
  x0-prediction).
- **Nothing but the partial SID passes between steps.** Self-attention is
  recomputed from scratch each step (`past` is `None`), and the cross-attention
  K/V depend only on the history. So the process is Markov in the discrete
  state (plus the beam score).

## 3. What "patch the predicted noise at step t" becomes

| Continuous diffusion | DiffGRM |
|---|---|
| Noisy latent x_t | Partial SID (revealed codes + masks) |
| Step / noise level | Number of masked digits, implicit |
| Predicted noise ε(x_t, t, c) | Per-position code distributions at step t |
| Denoising update | Committing one (position, code) |
| Conditioning c | History, through encoder states and cross-attention |

Because only the discrete state crosses steps, your idea has three well-defined
versions, and the plan runs all three:

- **P1, same state, other run's conditioning.** Score step t with the donor
  history, on the recipient's current state. This is the exact analogue (the
  donor's prediction at the same step and same x_t) and it works under beam
  search.
- **P2, other run's own state.** The donor ran freely, so its states differ.
  Transplanting its step-t prediction reduces to transplanting its commit, that
  is, forcing a digit and letting the recipient continue.
- **P3, inside one step.** Classic activation patching of encoder slots and
  decoder residuals (layer × digit position) between two runs, read out on
  that step's distributions.

One advantage over continuous diffusion: every patched trajectory stays a valid
partial SID, so there are no off-manifold latents. The off-distribution risk is
a (state, history) pairing the model never trained on.

## 4. Experiments

Cells. Trace: all 20 next-item checkpoints. Interventions: the two matched
cells (`diff-next1-rqkmeans-3cb-128`, `diff-next1-rqvae-4cb-128`) plus three
that vary what matters here: `rqkmeans-5cb-128`, `rqvae-5cb-128` (five steps;
never trained on the all-masked state) and `MQ-4cb-128` (parallel digits).
Cohort: the 6,297-user diffusion cohort only.

### D0. Reveal trace: what intermediate solutions look like (observational)

`experiments/representation/exp4_diffusion_reveal_trace/`. This is `RUN_PLAN`
item 3 and the input to everything below.

Recorded per user and step, for the greedy path, the path that ends as the
final top-1, and the golden path forced in the model's own confidence order:

- revealed position and code, path score;
- the full per-position distributions (fp16), entropy, golden log-prob and rank;
- the current guess (argmax SID over all positions), whether it is a catalogue
  SID, and its item(s) via `SidTable.cb2items`;
- a catalogue ranking at every step: score each of the 3,105 items by the sum
  of its digits' log-probs given the revealed digits. This gives golden-item
  rank and HR@10 "if decoding stopped at step t", i.e. what each further step
  buys;
- the same readout at every decoder layer (`ln_f` + output embedding applied
  to each block's output);
- decoder residuals of the top-1 path, written to the activation store
  (`hooks/store.py`; about 0.3–0.5 GB per cell in fp16).

Per beam and step: parent, position, code, score, number of distinct partial
states, duplicate share.

Also: the training-style reveal order for each test user (all-masked MSP
ranking with `use_cache=False`, as the trainer computes it), to measure how
often decode-path states fall inside the family of states training visited.

Outputs: figures (reveal order by quantizer and depth; golden rank by step and
position; entropy by step; commitment-step histogram; legal share of the
current guess; layer × step readout), an offline HTML viewer of per-user
"denoising strips", and the regret statistic (re-mask each final digit and
re-score it) for free.

### D1. Step-level switch of the history: P1 and P2

`experiments/controlled/exp12_diffusion_step_switch/`.

- **Donor history H′.** Primary: the most recent item replaced by exp7's
  different-first-digit control (`ControlPool`). Secondary: another user's
  history (seeded).
- **Part A (P1).** Decode with H′ at the steps in a set T and H elsewhere. All
  2ⁿ sets at depth 3–4; single-step, prefix and suffix sets at depth 5. Greedy,
  beam 64 and beam 256.
- **Part B (P2).** A 2 × 2 at each step of the greedy path: state from the
  recipient's or the donor's path × history H or H′. This splits each step's
  prediction into the part set by the history and the part set by the revealed
  digits.
- **Primary estimand.** Among users whose clean and donor top-1 SIDs differ:
  the share whose final top-1 equals the donor's when only step 0 is switched,
  minus the share when only the last step is switched. Secondary: top-10
  overlap with each pure run, SID and item HR@10, and the split of each effect
  into "changed which position was revealed" and "changed the code" (replay the
  clean order).
- **Controls.** H′ = H encoded as a separate row must be bit-exact with the
  baseline; the all-steps-switched run must equal an independent decode of H′.
- **Reading.** Tells at which step the copy is committed and whether a later
  step can undo an early commit made under other evidence.

### D2. Activation patching inside one step: P3

`experiments/controlled/exp13_diffusion_activation_patching/`.

- **Part H, history contrast** (same state, H vs H′). Sites: final encoder
  states by slot group (the item's own slot; all other real slots; single
  slots), and decoder residuals at layer 0–3 × digit position. Both directions.
  States: all masked, golden prefix (as exp7), and the D0 path states. This
  re-tests exp8 with sufficiency instead of knockout, and removes its limit
  that a blocked slot stays in the softmax.
- **Part S, state contrast** (same history; one revealed digit holds the golden
  code vs a prefix-matched legal sibling). Sites: decoder residuals layer ×
  position, plus decoder self-attention edge knockouts (per layer, per head)
  from the masked position to the revealed one.
- **Primary estimand.** Restored share of Δ log p(golden code) when patching
  the queried masked position vs the revealed position, by layer: the layer at
  which a revealed digit's information reaches the digit being predicted.
- **Prediction to state in the protocol.** Strong digit-to-digit flow for RQ
  maps, weak for MQ.
- **Controls.** Self-patch bit-exact; patching every position at the last
  layer reproduces the clean output exactly.

### D3. The implicit step (optional, after D0)

Does the decoder encode how many digits are revealed, and is the all-masked
start off-distribution at depth 5? First the free check from D0 (step-0
calibration at depth 5 vs depth 3–4), then a probe and a cross-step patch
(`DIFFUSION_INTERVENTIONS.md` item 2).

### Complementary, enabled by the same code

- Copy knockout inside decoding (the DiffGRM analogue of exp6;
  `DIFFUSION_INTERVENTIONS.md` item 1), using D1's per-step scorer.
- Guidance-style mixing of the two histories' log-probs (dose-response, and a
  negative weight as an "explore" knob; item 6).
- SAEs on decoder states, masked and revealed positions apart (item 4), from
  D0's activation store.
- One path per partial SID (`RUN_PLAN` item 8), from D0's per-step beam record.
- Next-two models: item-1 to item-2 conditioning across 6–8 steps. Deferred;
  the retained next-two evaluation is two-pass and needs its own validation.

## 5. Code changes

| File | Change |
|---|---|
| `src/sidlens/analysis/matched_decode.py` | Factor the per-chunk forward into one function. Add two optional arguments to `decode`: `observer` (called once per step with states, log-probs, parents, positions, codes, scores) and `score_fn` (replaces the per-step scorer). Defaults leave behaviour bit-identical. |
| `src/sidlens/interventions/residual.py` | `patch_residual` cannot patch `decoder_blocks.N` today: `DecoderBlock` returns a dict and the write-back path rebuilds only tuples and lists, so the next line of the vendor forward would raise. Add dict handling. No DiffGRM test covers it yet. |
| `src/sidlens/interventions/diffusion.py` | Add `decoder_self_attention_masks` (same pre-hook pattern as `cross_attention_masks`; the vendor passes `attention_mask=None` there, and a 4-D mask gives per-head control). Add a `patches=` argument to `score_states`, a slot-wise encoder-state mixer, and the per-step history-switch scorer. |
| `src/sidlens/analysis/` (new `diffusion_lens.py`) | Per-layer readout and the per-step catalogue ranking. |
| `experiments/…/exp4`, `exp12`, `exp13` | `protocol.md` first, then `run.py`, `summarize.py`, `README.md`, `RESULTS.md`; matching `scripts/{representation,controlled}/*.sbatch` copied from `diffusion_copy_route.sbatch`. |
| `tests/` | See Verification. |

Reused as is: `diffusion.load` / `encode`, `load_eval_cohort`,
`project_cross_cache`, `score_states`, `knockout_mask`, `encoder_block_mask`,
`ControlPool` and `condition_rng`, `UserBootstrap`, `collisions.item_rank_bounds`,
`hooks.capture` and the activation store, exp1's `verify_model_and_decoder`,
`viz/style.py`.

## 6. Feasibility

- **Access.** `onediffrec/OneDiffRec` is readable, including `DiffGRM` and
  `DiffGRM_new`.
- **Compute.** Not a constraint. exp8 took 67 s and exp7 447 s on CPU for the
  full cohort; exp2 used 5 GPU-minutes for 42 decodes. D0 on 20 cells is under
  one GPU-hour; D1 and D2 are minutes per cell.
- **Data root unreachable (found 2026-10-13).** `/l` is the cluster's Lustre
  filesystem (`ciai-head-1-ib@o2ib:/l`). On `ciai-login-2` its mount unit has
  been failed since 2026-10-10 06:51; on CPU nodes `cn-02`, `cn-10` and `cn-17`
  `/l/users` returns "No such device". The last SidLens jobs that reached it
  ran on 2026-10-07. GPU nodes were not probed: the GPU partition rejects
  jobs that only run monitoring commands. Until `/l` is back, or the substrate
  is restored elsewhere, nothing can run on a trained checkpoint and the tests
  that load one cannot run.
- **Exactness gate.** I could not confirm that exp1's archive
  (`derived/controlled/exp1_matched_beam/181944`) is on this cluster. If it is
  not, the gate is exp1's pilot checks plus reproduction of the registry's
  recorded metrics, as exp7 used.
- **Labels are not on this cluster** (`site.env`). The trace stores SIDs and
  item ids; titles and categories for the viewer need `external/labels` pulled
  with `sidlens bundle`, or the join done on the analysis cluster.
- **Limits on what results mean.** One seed per checkpoint. Training visits one
  reveal order per example, ranked by a different network (`use_cache=False`).
  Depth-5 models never train on the all-masked state. Logits at revealed
  positions are untrained and are never read. SID and item metrics stay
  separate; the AR cohort is never joined.

## 7. Order of work

1. Pilot sbatch: data root, `verify --strict --quick`, the current test suite.
2. Shared code (section 5) with tests; confirm `decode` is unchanged by default.
3. D0: protocol, 64-user pilot, full run on 20 cells, summary and viewer.
4. D1: protocol (written after D0's descriptive results, before any switched
   decode), pilot, five cells.
5. D2: protocol, pilot, five cells.
6. D3 and the complementary items, by priority after reading D0–D2.
7. Update `CLAUDE.md`'s status table, `experiments/README.md`,
   `docs/plans/DIFFUSION_INTERVENTIONS.md`, and copy this plan to
   `docs/plans/DIFFUSION_STEP_PATCHING_PLAN.md`.

## 8. Verification

- `decode` with default arguments returns the same codes and scores as before
  the change (existing `tests/analysis/test_matched_decode.py`, plus a new
  equality test with a no-op observer and the default scorer passed explicitly).
- Full-cohort baseline: archived predictions and ranks reproduced exactly at
  batch 32 and chunk 1024 (`atol = rtol = 1e-4` on scores) before any new
  condition counts.
- New CPU tests on the real `rqkmeans 3x128` checkpoint, following
  `tests/interventions/test_diffusion_knockout.py`: decoder residual self-patch
  is bit-exact; a patch from a different history moves the queried digit; a
  patch that does not fire raises; an all-ones decoder self-attention mask is
  bit-exact; the per-layer readout at the last layer equals the model's logits;
  the trace's final step equals `decode`'s output.
- Each run: no-op controls bit-exact, batch-invariance check on duplicated
  clean rows, `validation.json`, `output.sha256`.
- Statistics: paired-user bootstrap, 2,000 draws, recorded seed, intervals
  stated to exclude training-seed uncertainty.
- `pytest -q` and `sidlens.cli verify --strict --quick` under sbatch.

## 9. Decisions I made that you may want to change

- D1's primary donor is exp7's recent-item replacement, not a full history swap.
- Five intervention cells (two matched, two depth-5, one MQ), next-item only.
- D0 is filed under `representation/` as exp4; D1 and D2 are controlled exp12
  and exp13.
- Next-two and the pretrained-diffusion comparison are out of scope here.
