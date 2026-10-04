# SidLens: orientation for Claude

SidLens is the mechanistic-interpretability companion to OneDiffRec. OneDiffRec
trains generative recommenders that emit multi-digit Semantic IDs (SIDs). SidLens
asks what those models compute, when, and whether it matters causally. The
analysis code lives here. Model training happens upstream, on another cluster.
`README.md` holds the full data and provenance contract; the project brief is
`docs/plans/LeoRodrigues_ProjectProposal_GenRec_MechInterpAnalysis.pdf` (RQ1–RQ4 and
Experiments 1–6).

## Two roots and one Python

| What | Where |
|---|---|
| Code (this repo) | `/home/leo.rodrigues/GenRecSys/sidlens/sidlens` |
| Bulk data, `$SIDLENS_WORK` | `/l/users/leo.rodrigues/sidlens`, the default in `src/sidlens/paths.py` |
| Python 3.11 (torch 2.6+cu124) | `/l/users/leo.rodrigues/sidlens/venv/bin/python`, not on `PATH`. Call it explicitly. `sidlens` is installed editable. |

Under `$SIDLENS_WORK`:

| Directory | Contents |
|---|---|
| `frozen/` | Read-only, hash-pinned substrate (about 13 GB): SIDs, data, checkpoints, archived results. Pinned by `manifests/provenance.base-20260826.json`. |
| `external/` | Evidence acquired after training (labels, raw metadata, the extracted tarball) |
| `derived/` | Every result that SidLens produces |
| `runs/` | Empty. Intended for exported upstream runs. |
| `bundles/` | Transfer manifests from `sidlens bundle create/pull` (see `docs/clusters/CROSS_CLUSTER.md`) |
| `env/` | The environment; `venv` is a symlink to it |

Resolve every path through `sidlens.paths`, and never hardcode `/l/users/...`
in new code. **Never write to `frozen/` or `vendor/`.** `vendor/` holds
byte-frozen upstream code (`diffgrm`, `diffgrm_new`, `onediffrec`), and
`verify` fails if anything in it changes.

## Where the data came from

- **`frozen/`** was copied on 2026-08-26 by `sidlens freeze` from the upstream
  OneDiffRec trees on the old cluster. It is the only substrate that analysis
  code reads.
- **`data/OneDiffRec_data.tar.gz`** (gitignored, 1.67 GB) was extracted to
  `$SIDLENS_WORK/external/onediffrec_data/data/{Amazon18,two-item}`. **No SidLens
  code reads it.** It is mostly redundant with `frozen/`, with two exceptions:
  - It is the only local copy of the **Office SFT CSVs**: next-item
    `Amazon18/{train,valid,test}/Office_*.csv` and next-two
    `two-item/{train,valid,test}/Office_*.csv`.
  - Five Industrial RQ-KMeans index tables in it predate the 2026-08-20
    bit-unpacking repair and differ from `frozen/`. Never prefer them, and
    never extract the tarball over `frozen/`.
- **`external/amazon2018/` and `external/labels/`** hold post-training
  interpretability labels built by `scripts/data/build_labels.py`: category levels,
  store, brand, price and rank, and `also_buy`/`also_view`. These exist for
  **Industrial only**; Office metadata has not been downloaded.

## Substrate facts that cause silent bugs

- **Catalogues.** `Industrial_and_Scientific` has 3,105 items and is the primary
  catalogue for all analysis. `Office` has 17,696 items; its SIDs, embeddings
  and interactions are frozen, but it has no checkpoints, no `.sem_ids` and no
  labels. The legacy `OneDiffRec/data/Amazon` is a different item set; never
  join it.
- **SID variants.** Each catalogue has 27: `{rqvae, rqkmeans, MQ}` × depth
  `{3,4,5}` × width `{128,256,512}`, named like `rqkmeans_3codebook_128`. The
  ASIN-keyed `.sem_ids` file is ground truth; load it with
  `sidlens.data.sids.SidTable.load(name)`.
- **Collisions.** `tokens2item_*.pkl` is lossy (last writer wins). Decode
  through `SidTable.cb2items`, which maps one SID to many items. SID-level HR
  hides item ambiguity, so report item-level bounds
  (`sidlens.analysis.collisions`).
- **Nesting.** Within one width, the RQ-KMeans depths are truncations of a
  single fit. The exception is `rqkmeans_5codebook_512`, a separate
  post-repair fit that was also Sinkhorn-balanced (`rqkmeans_faiss.py
  --uniform`; structure exp3 A3). Every other RQ-KMeans table is
  `faiss.Kmeans(niter=10)` per level with seed 1234 reused at every level,
  and reproduces exactly only at 96 OpenMP threads. RQ-VAE's last digit was
  rewritten for colliding items by `generate_indices.py`'s Sinkhorn loop, so
  RQ-VAE collision rates measure that loop, not the quantizer. RQ-VAE and MQ depths are independent fits. MQ digits are
  parallel partitions, so an MQ prefix is an intersection of partitions, not a
  step down a hierarchy.
- **AR token IDs.** Read them from `added_tokens.json` via
  `sidlens.models.ar.load_vocab`. IDs are string-sorted (`<a_100>` comes before
  `<a_10>`), and unused codes are omitted; for example, digit 0 of
  `rqvae_4cb_128` has only 39 codes. Never compute `base + code`.
- **AR prompts.** Rebuild them with `sidlens.data.ar_prompts`; its ids equal
  upstream's dataset classes row for row. Test evaluation (`EvalSidDataset`,
  which produced every archived prediction) and training (`SidSFTDataset`)
  use **different wording**, so `template="eval"|"sft"` is required. Digit d is
  scored at `Encoded.predict_pos(slot, d)`: digit 0 at the ":\n" of
  "### Response:\n", digit d > 0 at the previous target digit under teacher
  forcing. Left padding needs explicit `position_ids` (`collate`). The test-CSV
  row index is the only join key to the archived AR predictions (verified
  for all 28 files).
- **Time.** No model sees a timestamp: the AR prompt says "chronological" and
  DiffGRM has a recency position. Days live only in `frozen/data/reviews/*.review.json`
  keys `(user, item, unixReviewTime)`. `sidlens.data.timestamps.load_event_times()`
  aligns them to every `inter.json` position. Ambiguous duplicate days are
  resolved by the global split order, and it raises rather than guesses.
  `ar_row_times` and `diffusion_row_times` join AR windows and the DiffGRM
  cohort. Facts that bite:
  - The next-item splits are **consecutive periods**: train until 2017-06-11,
    valid 2017-06-13 to 2017-12-09, test 2017-12-11 to 2018-09-28.
  - The DiffGRM leave-last-out cohort is **not** a time split.
  - Days are whole UTC days, and **same-day order is ASIN order**: 88% of
    same-day pairs are in ascending ASIN order, the raw-file order kept by the
    stable sort.
  - 41–50% of targets fall on the day of the most recent item.
  - **2,653 of 43,102 events duplicate a (user, item, day) review.** Duplicate
    targets are 8% of train, 5% of valid and 3% of test rows. They supply 11% /
    17% of AR test HR@10 hits.
  - Use `timestamps.GAP_BINS` for any age split.
- **DiffGRM token IDs.** `id = 3 + digit*K + code`, in code order. Histories and
  labels use raw codes, with `PAD = -1`.
- **DiffGRM runtime:**
  - `DIFF_GRM.forward(return_loss=False)` returns after the encoder. The decoder
    runs only through `forward_decoder_only`, so a hook on a decoder block fed by
    `forward()` captures nothing.
  - `models/diffusion.digit_logits` uses `use_cache=False`, which drops the
    learned cross-attention K/V projections, so it is **unvalidated**. For
    logits or decoding, use `analysis/matched_decode.py`, which applies the
    projected cross-attention cache and is validated against the archive.
  - Registry-built models cannot `generate()` because the decode config is
    missing.
- **Diffusion checkpoints** are bare state dicts. Their config comes from
  `manifests/registry.diffusion.json`, whose paths are rebased onto
  `$SIDLENS_WORK` at runtime; never edit those paths. `n_head=4` cannot be
  checked from tensor shapes; only behavioural reproduction of the recorded
  metrics confirms it.
- **Exact reproduction** of archived decoding requires batch size 32 and decoder
  chunk size 1024. Other settings have failed exact reconstruction on full
  cohorts.
- **DiffGRM training facts that bound what decoding results mean** (verified
  in `vendor/diffgrm/.../model.py`): with `masking_strategy=guided`, at each
  training step each example is trained along ONE reveal order (the model's
  own confidence ranking on the fully masked input, least-confident masked
  first, `refresh=False`), so fixed reveal orders mostly visit partial states
  training rarely covers;
  `guided_steps = min(d, 4)`, so **depth-5 models never see the fully masked
  SID during training** although every decode starts there (next-item code
  only: the next-two `diffgrm_new/.../model.py` caps at `n_target_digits`
  instead, but the joint-block `model_block.py` keeps the cap of 4); and the ranking
  pass uses `use_cache=False`, i.e. the unprojected cross-attention fallback,
  a different network from the one trained. The vendored confidence decoder
  also fills the last digit greedily (no branching) and pads short lists by
  repeating the last legal SID; `matched_decode` does neither.
- **Beam search keeps paths, not states.** Both the vendor decoder and
  `matched_decode` let several reveal orders reach the same partial SID and
  keep all of them; duplicates are 57 / 79 / 91 % of the final beam at depths
  3 / 4 / 5. Legality is checked only after the last step (no trie for
  diffusion; the AR side uses a trie at every step).

## Models and cohorts

- **AR.** Qwen2.5-1.5B SFT: 28 layers × 1536, `model.layers.N`. Three weight
  sets survive:

  | Checkpoint | SID variant |
  |---|---|
  | `next-item_best` | `rqkmeans_3codebook_128` |
  | `oneoff_rqvae4cb128` | `rqvae_4codebook_128` |
  | `two-item_best` | `MQ_4codebook_256` |

  Upstream deleted the weights for the other 26 AR runs of the frozen sweep;
  only their metrics and predictions remain. Outside `frozen/`, the CIAI
  cluster also holds 17 next-two checkpoints from the 2026-09-14 recovery
  runs (every quantizer × depth at widths 128 and 512 except MQ 3×512; full
  3.1 GB weights) in
  `/l/users/leo.rodrigues/onediffrec/sweep/industrial-next-two-recovery-20260914-a/checkpoints/`.
  They are not hashed or registered in SidLens yet. Register them before
  any result uses them.
- **Diffusion (DiffGRM).** One encoder layer and four decoder layers × 256,
  four heads, history length 50, `encoder_blocks.N` and `decoder_blocks.N`
  (each with `self_attn` and `cross_attn`).
  - next1: 20 trained checkpoints covering the full 3 × 3 × {128, 512} grid,
    plus MQ-5cb-256 and rqvae-3cb-256.
  - next2: rqvae 3cb-256 and 4cb-256 are trained; 5cb-256 is incomplete.
    Retained next2 evaluation runs in two passes. It is not a joint block.
- **Matched AR↔diffusion cells** exist only for next-item:
  `rqkmeans_3codebook_128` and `rqvae_4codebook_128`
  (`sidlens.models.ar.MATCHED_CELLS`). No next-two pair is matched.
- **Test cohorts.**
  - Diffusion next-item: 6,297 users, leave-last-out, loaded with
    `sidlens.data.diffusion_eval.load_eval_cohort`.
  - Historical AR next-item: 3,681 events from 1,606 users.
  - The AR and diffusion cohorts differ. **Never join them by row number.**

## Code map

| Location | Contents |
|---|---|
| `src/sidlens/data/` | `sids` (SID tables, nesting), `diffusion_eval` (test cohort), `ar_prompts` (exact AR input ids, per-position role/item/digit map, archive and SID-table checks, `collate`), `labels`/`meta` (Amazon labels), `embeddings` (Qwen3-Embedding-4B, `(3105, 2560)` FP16, not normalized), `timestamps` (review day of every event, AR/DiffGRM row time tables, `GAP_BINS`) |
| `src/sidlens/models/` | `ar` (vocab and loader), `diffusion` (registry → model, shape checks) |
| `src/sidlens/hooks/` | Observation-only forward hooks: `residual_points`, `attention_points`, `capture`, `run_capture`. Attention hooks see module outputs, not per-head weights. Hooks must never change an activation; interventions belong in `interventions/`. `hooks/store.py`: the activation store (safetensors shards + parquet rows + manifest written last; no pickle), the one format for captures from any cluster. |
| `src/sidlens/analysis/` | `atlas`, `attribute` (conditional AMI with within-parent null), `naming`, `geometry`, `refinement`, `collisions`, `matched_decode` (the validated diffusion beam search: `policy="confidence"` or `"fixed"` with an `order`), `semantic_mapping` (CLI: `build` / `query`) |
| `src/sidlens/viz/semantic_explorer.py` | Standalone offline HTML explorer |
| `src/sidlens/viz/style.py` | The one figure style: quantizer colours (RQ-KMeans blue, RQ-VAE orange, MQ aqua, fixed everywhere), one-hue ramps for width/depth, ink tokens, `save()` writes PDF+PNG. Palettes were validated for CVD/contrast; keep using them. |
| `experiments/structure/semantic_mapping/figures.py` | The eight structure-layer figures (icicle, group sizes, geometry, purity-vs-null, per-digit information, t-SNE zoom, one-digit maps, collisions); each writes a CSV twin. t-SNE is cached in the output dir. |
| `src/sidlens/interventions/` | `residual` (residual-stream patching; a patch must fire exactly once or the call raises), `attention` (attention-edge knockout via explicit per-layer (B,H,T,T) masks; refuses no-op, self and replaced-mask cases), `history` (prefix-matched controls for history items and target slots, re-encoded and position-checked), `scoring` (per-digit codes/legal/vocab log-prob, rank, top-1), `runner` (fixed-shape runner: patches + knockouts + multi-slot scoring; no-op controls are bit-exact on GPU, not on CPU where MKL is row-position dependent), `generation` (attention knockout inside KV-cached `generate()` beam search, per-beam plans from each beam's own prefix), `diffusion` (DiffGRM: digit scores at chosen decoder states via the projected cross cache; cross-attention and encoder self-attention knockouts) |
| `src/sidlens/hooks/attention_probs.py` | Observation-only per-head attention probabilities recomputed from a layer's inputs |
| `src/sidlens/analysis/bootstrap.py` | `UserBootstrap`: one paired cluster bootstrap over users for every estimand (means, ratios, paired differences) |
| `src/sidlens/sae/topk.py` | TopK SAE (Gao et al.): fixed input scale, unit-norm decoder, AuxK for dead latents, `ablate` (error-preserving removal of named latents; an empty set returns x exactly), safetensors io |
| `src/sidlens/probes/` | **Empty scaffold** (probes so far live in `experiments/representation/`) |
| `src/sidlens/provenance/`, `registry/` | freeze, verify (`--profile core\|ar\|full` checks a partial substrate and says so) and manifests; `profiles` + `bundle` (hash-certified transfer through private HF dataset repos: `sidlens bundle plan/create/check/push/pull`); the diffusion registry and log parsing |
| `experiments/` | Entry points, by task. `representation/` (observational reads of activation captures: `exp1_ar_digit_decoding`; `exp2_ar_sae`, TopK SAEs on train-split captures with train / analyze / causal stages; `exp3_ar_information_map`, self-forced captures + SAEs at every second layer + latent census + tuned lens + SID decoders). `structure/` (static SID-map analyses: `exp1_atlas`, `exp2_geometry`, `semantic_mapping` incl. `anomaly_audit.py`, `exp3_collision_causes`). `retrospective/exp{1..4}` (CPU audits of archived outputs); `retrospective/exp6_time_structure` (review days in the data and in the exp3/4/6/7 outputs); `retrospective/exp7_collision_credit` (exp6 re-scored at item level by target–history relation). `controlled/exp1_matched_beam`, `controlled/exp2_fixed_orders` (GPU decoding); `controlled/exp3_ar_history_patching`, `exp4_ar_copy_circuit`, `exp5_ar_next_two_conditioning`, `exp6_ar_copy_in_decoding`, `exp9_ar_order_vs_time` (AR causal interventions, run on the CIAI cluster); `exp7`/`exp8` (DiffGRM); `exp10_ar_last_digit_sibling` (AR last-digit edge knockout: copy vs exclusion of a prefix-matching item's code). `paper_www27/export_paper_data.py` packages the numbers cited in the WWW'27 draft (`docs/paper/archive/Leo_WebConf27_GenRecSys_sidlens.zip`) with source hashes. Index: `experiments/README.md`. |
| `scripts/` | By task, mirroring `experiments/`: `common/` (`_preamble.sh`, sourced by every wrapper: `~/.config/sidlens/site.env`, verify gate honouring `SIDLENS_VERIFY_PROFILE`, offline HF, provenance stamp; `site.env.example`), `data/` (`build_labels.py`), `provenance/` (`audit_checkpoints.py`), `structure/`, `retrospective/`, `controlled/` (sbatch wrappers for the matching `experiments/` group), `activations/` (`ar_capture.{py,sbatch}`, site-neutral; `hooks_smoke.{py,sbatch}`). Submit from the repo root. Index: `scripts/README.md`. |
| `requirements/` | `lock-20260925-cu124.txt`, pip freeze of this venv (a record, not an install recipe) |
| `docs/` | `plans/` (project brief PDF, research plan, three-hour ladder), `results/semantic_mapping/` (results, takeaways, figures, offline explorer, worked prefix-14 example), `clusters/` (cross-cluster runbook, prompts for the upstream-cluster agents), `paper/` (WWW'27 paper: current draft `rewrite-20261004/` with `RQ_PLAN.md`; writing guides `WRITING.md` and `REWRITING.md`; `archive/` holds the earlier zips, PDFs and rewrite folders). Index: `docs/README.md`. |
| `tests/` | pytest, CPU only, about 1 minute, grouped like the code: `data/`, `models/`, `hooks/`, `analysis/`, `provenance/`, `experiments/`. Basenames must stay unique (no `__init__.py`). 253 passed and 3 skipped on 2026-09-27. |

## Running things

```bash
export SIDLENS_WORK=/l/users/leo.rodrigues/sidlens
PY=$SIDLENS_WORK/venv/bin/python
$PY -m sidlens.cli verify --strict --quick     # substrate gate (size check + vendor hashes)
$PY -m sidlens.cli bundle plan --profile ar    # what a transfer would carry; writes nothing
$PY -m pytest -q                               # full tests, CPU
$PY scripts/provenance/audit_checkpoints.py --missing-only   # exits 1 by design while cells are missing
$PY -m sidlens.analysis.semantic_mapping query \
  --db $SIDLENS_WORK/derived/semantic_mapping/194268/mappings.sqlite \
  --variant rqkmeans_3codebook_128 --prefix 14,16 --format csv
```

- The login node (`lo-02`) has **no GPU**. Never load models there; the AR
  model is 3 GB. Submit with `sbatch` to partition `ws-ia`. GPU wrappers pin
  `--nodelist=ws-l3-020 --gres=gpu:1`. Compute nodes have no outbound network.
- Each wrapper sources `scripts/common/_preamble.sh`, calls `sidlens_gate`, archives
  the executed source, and writes to a new job-specific directory. Example
  commands are in `experiments/controlled/exp2_fixed_orders/README.md`.
- `scripts/retrospective/retrospective_experiments.sbatch` writes into fixed
  `derived/retrospective/*` directories and would overwrite reports. Do not
  rerun it casually.

## Conventions for a new experiment

The two `controlled/` experiments are the templates.

- Create `experiments/controlled/<name>/` containing:
  - `protocol.md`, written **before** outcomes are seen: question, cohort,
    conditions, primary estimand
  - `run.py` and `summarize.py`
  - `README.md` and `RESULTS.md`

  Add a matching `scripts/controlled/<name>.sbatch`.
- Write outputs to `$SIDLENS_WORK/derived/controlled/<name>/<array-id>/cell-XX/`
  and summaries to `summary-<array>-<job>/`. Existing result directories are
  never overwritten; `mkdir` fails on a collision, by design. Keep:
  - `source.tar.gz` and source hashes
  - arguments, `inputs`/`validation.json`
  - predictions and the per-user table
  - `report.md`, `figures/`
  - `status.txt`, `output.sha256`
- Statistics: a paired-user bootstrap, 2,000 draws, fixed recorded seed, with
  all cells and conditions of a sampled user kept together. Always state that
  intervals exclude training-seed uncertainty. Keep SID-level and item-level
  metrics separate, and keep AR and diffusion cohorts separate.
- New decoders must reproduce the archived baseline exactly (predictions and
  ranks, `atol = rtol = 1e-4` on scores) before any new condition counts.
- Code style: dense "why" docstrings that name the trap each guard prevents;
  fail loudly rather than fall back. Match this style.

## Experiment status and result locations (2026-09-30)

| Result | Headline | Where |
|---|---|---|
| Exp 1 static atlas (27 variants) | Per-digit conditional AMI, naming AUC, co-purchase lift | `derived/exp1_atlas/<variant>/` |
| Exp 2 geometry and refinement (27) | Radius, R², null-corrected refinement, digit deletion | `derived/exp2_geometry/`, `derived/exp2_refinement/` |
| Retro 1: reveal order (archived) | Confidence +0.34 pp HR@10, but at unequal beams | `derived/retrospective/exp1_reveal_order/` |
| Retro 2: AR prefix errors | First mismatch at digit 1 in 82% of rows | `derived/retrospective/exp2_prefix/` |
| Retro 3: collisions | SID HR@10 exceeds item-uniform HR by 4.5 pp; collided targets +42.5 pp | `derived/retrospective/exp3_collisions/` |
| Retro 4: next-two association | P(p2 correct \| p1 correct) 4.0% vs 0.3%; associational only | `derived/retrospective/exp4_next_two/` |
| Controlled 1: matched beam (18 cells, 2026-09-09) | Confidence − fixed(seed 42): −0.95 pp at beam 64, +0.22 pp at beam 256. At beam 64 the gap moves from +0.4 to −0.7 to −2.6 pp over depths 3, 4 and 5; at beam 256 depth 5 is only −0.2 pp. | `derived/controlled/exp1_matched_beam/summary-181944-181946/` |
| Controlled 2: all six fixed orders at depth 3 (2026-09-15) | Confidence − mean fixed: +0.40 pp [0.29, 0.52]. The best order differs by quantizer. About 55% of confidence's final paths are duplicates. | `derived/controlled/exp2_fixed_orders/summary-194295-194297/` |
| Semantic mapping and explorer (2026-09-15) | 83,835 assignments; category coherence against a shuffled null. RQ-VAE 3cb-128 has an anomalous 26-item bucket `(27)(89)(7)`. | `derived/semantic_mapping/194268/` (`explorer.html`, `mappings.sqlite`), `anomaly-20260915/` |
| Structure-layer figures and takeaways (2026-09-22) | RQ digits carry semantics only given the prefix (digit 1 alone: 53 % of variance; digits 2–4: 12–16 %); MQ digits are four similar, overlapping partitions (36–38 % each); category purity saturates by digit 3; RQ-VAE under-uses digit 1 in 8 of 9 fits (2–31 % of codes) and uses its last digit as an identity digit. RQ-KMeans 5×512 gives the 11 identical-embedding pairs distinct SIDs (every other table collides them), so it was not made by the same deterministic procedure. | `derived/semantic_mapping/figures-20260922/`, `docs/results/semantic_mapping/SEMANTIC_MAPPING_TAKEAWAYS.md`, `docs/plans/RESEARCH_PLAN.md` |
| Hooks smoke test | Both paradigms capture correctly | `derived/hooks_smoke/` |
| Controlled 3: AR history interventions + residual patching (2026-09-27, CIAI cluster) | Replacing the most recent history item costs the golden digit 1 0.57 / 0.51 nats (RQ-KMeans 3×128 / RQ-VAE 4×128), flipping top-1 in 75 / 50 % of rows; effect decays steeply with recency; keeping the item's first digit removes 82 / 64 % of it. Patching: the item's evidence stays at its tokens to ~L16–20 and moves to header/target positions in L19–24. Post hoc: later-digit effects sit almost entirely in rows whose history item matches the target prefix (prefix-matched copying hypothesis). All 44,172 exactness controls per cell bit-exact. | `derived/controlled/exp3_ar_history_patching/summary-280961/` (on the CIAI cluster; see `experiments/controlled/exp3_ar_history_patching/RESULTS.md`) |
| Controlled 4: AR prefix-matched copying (2026-09-27, CIAI, valid split) | Confirms exp3's lead on data not used to form it: changing a history item's digit d when it already matches the target's first d digits costs 1.35 / 1.06 nats (RQ-KMeans / RQ-VAE) vs ~0 otherwise. Knocking out the single attention edge readout(d)→that item's digit-d token costs 0.40 / 0.36 (control edge ~0); layers 0–13 contribute 0.000, layers 14–27 all of it. Distributed over many heads: top-5 heads (screened on half the users) carry 8–9% on held-out users. Test split: same direction, about half the size. | `derived/controlled/exp4_ar_copy_circuit/summary-281261/` (valid), `summary-281394/` (test) |
| Controlled 5: next-two conditioning on two-item_best (2026-09-27, CIAI) | Clamping item 1 costs item 2's first digit 0.60 nats, 0.40 more than replacing the last history item; prefix-matched copying also runs item 1 → item 2 (0.43–0.73 vs ~0); item-1 information moves to the separator / item-2 tokens in layers 20–26. | `derived/controlled/exp5_ar_next_two_conditioning/summary-281263/` |
| Representation 1: AR logit lens + user-disjoint probes on the test captures (2026-09-28, CIAI) | The final first-digit prediction equals the most recent item's first digit in 64% / 53% of rows (golden top-1 18% / 34%); that copy preference appears at layers 20–23 (RQ-KMeans) / 11 and 20 (RQ-VAE), though the recent item's digit is linearly present at the readout from block 0 (84–93%). The target's first digit is never probed above the copy baseline. Items are assembled by layers 1–4. | `derived/representation/exp1_ar_digit_decoding/summary-281264-281411/` |
| Controlled 6: AR copy knockout inside the archived beam decoder (2026-09-28, CIAI) | Copying helps: blocking copy reads (layers 14–27, each beam's own prefix) at every digit lowers exact-SID HR@10 by 5.7 / 3.3 pp, 27.3 / 10.1 pp on repeat targets (baseline HR@10 91% on them) and 2.1 / 3.0 pp on new ones; no block raises new-target HR@10. Matched control −0.03 / −0.11 pp. Baseline reproduces 3,680/3,681 archived lists. Plain beam search (`use_model_defaults=False`, secondary): HR@10 24.7 / 15.3% (archived 24.0 / 14.4%), copy block −4.0 / −3.2 pp (repeat −19.4 / −8.4); only RQ-KMeans new-target HR@1 rises (+0.6 pp). | `derived/controlled/exp6_ar_copy_in_decoding/summary-283114/`, `summary-283586-283621/` (plain) |
| Controlled 7: DiffGRM history use (2026-09-28, CIAI, CPU) | Same behaviour as AR on the matched cells: replacing the most recent item costs the golden first digit 0.56 / 0.48 nats (AR 0.57 / 0.51); prefix-matched copying 0.35 / 0.96 vs ~0. But the direct cross-attention read of the matching item carries nothing (−0.007 / −0.004). | `derived/controlled/exp7_diffusion_history_use/summary-283124-r2/` |
| Controlled 8: DiffGRM copy route (2026-09-28, CIAI, CPU) | The copy runs through the encoder: blocking other slots from reading the recent item costs 0.17 / 0.30; blocking both the encoder spread and the direct decoder read costs 0.24 / 0.60 (~2/3 of the replacement effect), super-additive (+0.08 / +0.30); control ≈ −0.02 / −0.05. | `derived/controlled/exp8_diffusion_copy_route/summary-283159/` |
| Structure 3: why items share a SID (2026-10-03, CIAI, CPU) | Each family collides for a different reason. RQ-VAE's near-zero rates come from upstream's last-digit Sinkhorn loop (`generate_indices.py`): applied to RQ-KMeans it takes every table from 10–45 % to the 22-item identical-input floor; RQ-VAE keeps 3 buckets the loop could not split (26 / 10 / 19 items). RQ-KMeans 5×512 is `--uniform` Sinkhorn-balanced (fingerprints match; halves first-digit neighbour agreement). Near-duplicate variants are the floor (73–85 % of collided items at the deepest nested RQ-KMeans fits). Shallow RQ-KMeans collisions come from a near-origin hub code at every level ≥ 1 (9–17 % of items). MQ digits are redundant (NMI 0.50–0.70; ≥ 7× the independent-digit null). Not causes: capacity, norms, whitening; faiss reuses one k-means seed per level (repeat rate 4–38× chance) but that *lowers* collisions. Faiss refit at 96 threads reproduces all 8 nested RQ-KMeans tables exactly. Primary rate = share of items in a shared SID (44.8 % at RQ-KMeans 3×128; the 29.3 % in the takeaways is 1 − unique/N). | `derived/structure/exp3_collision_causes/293555/result/`, `a3-20261003/`, `experiments/structure/exp3_collision_causes/RESULTS.md` |
| Retro 7: collision credit in copying (2026-10-03, CIAI, CPU) | Re-scores exp6 at item level (CCE) by target–history relation. RQ-KMeans 3×128: targets that share a history item's SID but are a different item (11.3 % of rows) supply 56 % of the `C_all` HR@10 loss and 42 % of baseline SID hits; only 28 % [22, 36] of the SID-level HR@1 copy benefit survives at item level (83 % at HR@10). RQ-VAE 4×128: near-duplicate variants with a different SID (9.9 % of rows, split by the dedup loop) supply 80 % of the loss and 94 % of exp6's 'copying helps new targets'. Truly new targets (83–85 %): item HR@10 change −0.2 / −0.1 pp. Same with plain beam search. | `derived/retrospective/exp7_collision_credit/293597/result/`, `experiments/retrospective/exp7_collision_credit/RESULTS.md` |
| Retro 6: time structure (2026-09-30, CIAI, CPU) | In the data, "recent" means same day. Once an item's age is known its position adds ≤ 0.002 McFadden R² (age adds 0.10–0.13 beyond position). RQ-KMeans AR copies r1's first digit in 67% of same-day rows and 67% of > 1-year rows (data: 33% vs 3%; ρ = 0.01). Other cells track 26–40% of the decay. Same-day rows carry 87 / 88% of exp6's copy benefit; copying does nothing for new later-day targets; duplicate records are 2.7% of test rows but 11 / 17% of HR@10 hits. The exp4 valid/test ratio drops from 1.5 / 1.65 to 1.25 / 1.32 after post-stratifying on duplicate and same-day rows. | `derived/retrospective/exp6_time_structure/287189/result/`, `experiments/retrospective/exp6_time_structure/RESULTS.md` |
| Controlled 9: order vs time, swap r1 and r2 (2026-09-30, CIAI) | RQ-KMeans AR follows position, not item or time: position effect A = 0.38 (r1 later) vs 0.36 (r1, r2 same day, target later; data asymmetry 0.007), P2 = −0.02. A timestamp-preserving swap flips 50% of first-digit decisions and 42% of top-1 SIDs, with HR@10 unchanged. Averaging over the two tied orders gains +0.027 nats where licensed and −0.019 where the order is real. RQ-VAE is weakly positional (A ≈ 0.07–0.10, item effect ≈ 0.07). Valid replicates. All identical-SID swaps bit-exact; clean decoder reproduces exp6 3,680 / 3,681. | `derived/controlled/exp9_ar_order_vs_time/summary-287182/` (test), `summary-287183/` (valid) |
| Controlled 10: last-digit read of a prefix-matching item (2026-10-03, CIAI) | Edge knockout readout(D−1) → last-digit token of the most recent history item h matching the target on all earlier digits. RQ-KMeans 3×128 copies h's last digit (−0.17 / −0.34 nats test / valid vs control), which costs sibling targets (+0.15 when blocked). RQ-VAE 4×128 does the opposite in sibling rows: blocking raises top-1 = h by 11.8 / 11.6 pp and lowers top-1 = sibling target by 10.7 / 9.3 pp (intervals exclude 0); its declared pooled primary (+0.09 / +0.06 nats) includes 0. At clean, RQ-VAE puts h's exact code first in 18 % of sibling rows (RQ-KMeans 53 %), 92 % vs 22 % for re-bought vs other items. All no-op controls bit-exact; clean matches exp4 within 1e-6. | `derived/controlled/exp10_ar_last_digit_sibling/summary-293619-293620/`, `experiments/controlled/exp10_ar_last_digit_sibling/RESULTS.md` |
| Representation 2: TopK SAEs on AR residuals, layers 12/16/20/24 (2026-09-30, CIAI) | Fitted on train-split captures; test FVU ≤ 0.055; the splice keeps ≥ 97% of what mean-ablation destroys. Copy latents at the first-digit readout are causal from layer 20: ablating them costs the copy score 0.10 (L20) and 0.21 / 0.36 (L24) nats vs a matched control ~0, though they carry ≤ 3% / 12% of readout activation. RQ-KMeans has sharp prefix-match latents (within-digit AUC up to 0.97) that are causally inert at the readout (0.00 ± 0.04; also under post-hoc amendment A1); RQ-VAE's are weaker with +0.02–0.03 nats. No target-content information beyond the history (ridge on the Qwen3 embedding). The readout encodes same-day bursts beyond SID overlap (AUC +0.08–0.09) that RQ-KMeans' copying ignores. | `derived/representation/exp2_ar_sae/summary-287222-287231/` (declared), `summary-a1-287380-287415/` (A1) |
| Archived AR decoder audit (2026-09-28) | `evaluate.py` + transformers 4.57.1 merge the checkpoint's Qwen `generation_config.json`: archived AR beams used do_sample=True, T=0.7, top-k 20, top-p 0.8, repetition penalty 1.1. 18.9% / 15.0% of archived list entries are not catalogue SIDs (7.6% / 0.7% of rows inside top-10). Reproduced exactly with `set_seed(42)` per shard. `oneoff_rqvae4cb128` reproduces 0/3,681 archived RQ-VAE lists (calc.py HR@10 14.5% vs recorded 13.1%): it is not the sweep's model. Plain beam search needs `use_model_defaults=False`: explicit `do_sample=False` / `repetition_penalty=1.0` equal the library defaults and are refilled from the checkpoint. | `experiments/controlled/exp6_ar_copy_in_decoding/protocol.md` (amendment), `docs/results/interventions/INTERVENTIONS_REPORT.md` |
| AR test-split captures (2026-09-27, CIAI cluster) | eval wording, all 28 layers + norm, roles hist_sid/response_header/target_sid, bf16; 6.0 / 8.0 / 7.9 GB for next-item_best / oneoff_rqvae4cb128 / two-item_best; archive, SID-table and sha checks pass. Observational data, not a result. | `derived/ar_capture/281053`, `281054`, `281055` (CIAI cluster) |
| AR capture pilot (2026-09-25) | `scripts/activations/ar_capture.sbatch`, 8 rows, CPU fp32, layers 0/13/27/norm: checkpoint sha, archive and SID-table checks pass; batch-invariance 5e-4 on activation norm ~300. Pipeline check only, not a result. | `derived/ar_capture/208116/` |

Queued 2026-10-04 (CIAI jobs 294711 gpu → 294712 census; pilot 294030 found a bf16-tie check bug, fixed as amendment A1): representation/exp3 information map.
Queued 2026-10-04: substrate/exp1 AR next-item retrain, pilot job 294718 (1 GPU, smoke). Arm A recreates the 18 Qwen-AR next-item models of the width-128/512 grid with the recorded recipe (vendor code byte-identical to upstream `eae9ecc`; inputs hashed against the freeze manifest). Arm B trains ZCR versions of five RQ-KMeans maps; inputs are in `derived/substrate/zcr_inputs/20261004`. Weights go to `runs/ar_next_item_retrain/<run>/<variant>/weights/final/`. No result may use a retrained model before `check.py` accepts it. Final RQs and the run list: `docs/paper/rewrite-20261004/RUN_PLAN.md`. DiffGRM intervention ideas from the literature: `docs/plans/DIFFUSION_INTERVENTIONS.md`.

Not yet started: crosscoders, SAEs on attention inputs (QK features for the prefix match), activation patching,
partial-state deduplication, full sweep at beam 256, reveal-order traces,
AR order retraining, joint-block next-two. See
`docs/plans/THREE_HOUR_EXPERIMENT_PLAN.md` for the leveled experiment ladder and its
interpretation constraints.

## Outside this repo

The upstream OneDiffRec training code is on another cluster and is **not on this
machine**. That includes branch `leo-dev-changes`, the next-2-item sweep, ZCR
for RQ-KMeans, CCE, and training-time activation logging (about 100 GB per
dataset × paradigm). Before anything from there counts as evidence, export it,
hash it and register it under `$SIDLENS_WORK/runs/`; the "Run export contract"
section of the README gives the steps. `docs/clusters/CROSS_CLUSTER.md` is the runbook
for running SidLens there: bundles through private HF dataset repos, setup, and
measured activation sizes. `docs/clusters/AR_CLUSTER_PROMPT.md` is the prompt for the
agent doing AR captures and interventions there. `docs/clusters/MAIN_PROJECT_DATA_PROMPT.md`
is the extraction prompt for that repo. `docs/results/semantic_mapping/rqkmeans_prefix14.xlsx` is the worked SID
explorer example (eSUN filaments under RQ-KMeans prefix 14).
