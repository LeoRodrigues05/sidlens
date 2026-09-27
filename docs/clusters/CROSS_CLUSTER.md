# Running SidLens on another cluster, and moving data through the Hub

Written 2026-09-25. This runbook covers AR activation capture and interventions
on the cluster that holds the upstream OneDiffRec code. It says which parts of
this cluster's data that work needs, how they move, and how results come back.
It supersedes the "Moving to another cluster" section of the README, which
predates the GitHub remote.

## The short version

| What | How it moves | Size |
|---|---|---|
| Code | `git clone https://github.com/LeoRodrigues05/sidlens` (the `origin` of this checkout) | < 10 MB |
| Frozen substrate + labels + derived results | `sidlens bundle push` → private HF **dataset** repo → `sidlens bundle pull` | 3.0 / 12.3 / 15.3 GB (profile `core` / `ar` / `full`) |
| Activations and intervention results coming back | Same tool, profile `results`, a second private dataset repo | Data-dependent. See [storage](#how-much-bulk-storage) |
| Python environment | Rebuilt on the target. `requirements/lock-20260925-cu124.txt` is the record | — |

Acceptance on the receiving side is always a hash check against a manifest,
never the transport. For frozen files that manifest is
`manifests/provenance.base-20260826.json`, which is in git. For `external/` and
`derived/` it is the bundle manifest.

## What the AR work needs from this cluster

Everything below is in profile `ar`. Profile `core` is the same minus the
three 3.1 GB AR weight files; profile `full` adds the Qwen2.5-1.5B-Instruct
base weights, which only matter for retraining (Exp 4).

| Needed for | Files (under `$SIDLENS_WORK`) | Why it can't be rebuilt there |
|---|---|---|
| Running the three frozen AR models | `frozen/ckpt/ar/{next-item_best,oneoff_rqvae4cb128,two-item_best}/` (weights + tokenizer, 3.1 GB each) | Upstream deleted the other 26 AR weight sets. Even surviving upstream copies may have been overwritten by later sweeps; these are the hash-pinned ones |
| Exact prompts and cohorts | `frozen/data/splits/{next-item,two-item}/{train,valid,test}/` | The row order of the test CSVs is the only join key to the archived predictions |
| Decoding SIDs to items | `frozen/sids/{sem_ids,index_json,info,mappings,mispacked}/`, `frozen/data/id_maps/` | RQ-VAE and MQ assignments cannot be regenerated; the RQ-KMeans refit is non-deterministic |
| Validation targets | `frozen/results/sweep_metrics/` (ranked beams for 28 AR runs) | These are the only record of what the models predicted |
| Probe labels | `external/labels/`, `external/amazon2018/` | Industrial only; built after training |
| Probe/geometry targets | `frozen/data/embeddings/`, `derived/exp1_atlas/`, `derived/semantic_mapping/` | Weeks of computed results |
| Picking intervention examples | `derived/retrospective/exp2_prefix/` (first wrong digit per row), `exp3_collisions/` | Same |

Not needed there: `env/` (rebuild it), `cache/`, and
`external/onediffrec_data/`, which no SidLens code reads and whose five
Industrial RQ-KMeans tables are pre-repair. That tree and the 1.67 GB
`data/OneDiffRec_data.tar.gz` it was extracted from (sha256 `c3ab0bed…`, see
the README) are the only local copies of the **Office** SFT CSVs. Upload the
tarball separately if you want it backed up; the bundles leave it out on
purpose.

## Step 1 — on this cluster, before leaving

1. **Commit and push the code.** A clone gets only what is committed. On
   2026-09-25, 59 paths were untracked. They include `CLAUDE.md`, `docs/`,
   `experiments/controlled/exp2_fixed_orders/`, `experiments/structure/semantic_mapping/`,
   `src/sidlens/viz/style.py`, `src/sidlens/analysis/semantic_mapping.py`, and
   everything added for this runbook:
   `src/sidlens/provenance/{bundle,profiles}.py`,
   `src/sidlens/data/ar_prompts.py`, `src/sidlens/hooks/store.py`,
   `scripts/activations/ar_capture.{py,sbatch}`, `scripts/common/site.env.example`,
   `requirements/`, `tests/data/test_ar_prompts.py`, `tests/hooks/test_activation_store.py`,
   and `tests/provenance/test_transfer.py`. The 2026-09-27 reorganization into
   task subfolders also shows every moved file as deleted plus untracked until
   `git add -A`, which records them as renames.
   Check `git status --short` before you commit.

2. **Create two private dataset repos** on the Hub (the tool never creates or
   re-scopes a repo, and it refuses a public one):

   ```bash
   hf repo create LeoRodrigues05/sidlens-substrate   --repo-type dataset --private
   hf repo create LeoRodrigues05/sidlens-activations --repo-type dataset --private
   ```

   Keep them separate. The substrate is small, immutable and tagged per
   bundle. Activations are large and churn, and you may want to prune them
   later, which on the Hub means squashing history (`HfApi.super_squash_history`).
   You should not have to do that to the repo that holds the frozen bytes.

3. **Bundle and push**, from the login node. Compute nodes here have no
   outbound network.

   ```bash
   export SIDLENS_WORK=/l/users/leo.rodrigues/sidlens
   PY=$SIDLENS_WORK/venv/bin/python
   $PY -m pip install -e ".[hub]"        # huggingface_hub is already in this venv
   $PY -m sidlens.cli bundle plan   --profile full           # sizes; writes nothing
   $PY -m sidlens.cli bundle create --profile full --bundle-id full-20260925
   $PY -m sidlens.cli bundle push   full-20260925 --repo LeoRodrigues05/sidlens-substrate
   ```

   `create` re-hashes every selected frozen file against the provenance
   manifest and refuses on any mismatch, so a bundle only certifies bytes
   that `verify` would pass. `push` is resumable: re-run the same command
   after an interruption. It uploads a symlink mirror under
   `$SIDLENS_WORK/cache/bundle-stage/<id>/`, then tags the commit
   `full-20260925`. Push `full` once, even if the other cluster only pulls
   `ar`. A pull can fetch any subset of a bundle's files, and the Hub then
   holds a complete off-cluster backup.

   *Profiles are nested.* A `core` or `ar` pull from the `full` bundle is not
   built in (pull takes a bundle id). To keep pulls minimal, also create and
   push an `ar` bundle. It costs no extra Hub storage, because identical file
   contents are deduplicated.

## Step 2 — on the other cluster

```bash
git clone https://github.com/LeoRodrigues05/sidlens.git <CODE>/sidlens
mkdir -p ~/.config/sidlens
cp <CODE>/sidlens/scripts/common/site.env.example ~/.config/sidlens/site.env
$EDITOR ~/.config/sidlens/site.env      # SIDLENS_REPO, SIDLENS_WORK, SIDLENS_VERIFY_PROFILE=ar, ...
source ~/.config/sidlens/site.env        # also add this line to ~/.bashrc

# Environment: install that cluster's torch/CUDA build first, then:
python3.11 -m venv "$SIDLENS_WORK/venv"
"$SIDLENS_WORK/venv/bin/pip" install torch   # the build matching that cluster's CUDA
"$SIDLENS_WORK/venv/bin/pip" install -e "$SIDLENS_REPO[hub,test]"
diff <("$SIDLENS_WORK/venv/bin/pip" freeze) "$SIDLENS_REPO/requirements/lock-20260925-cu124.txt"

# Data (on a node with Hub access; `hf auth login` once):
PY=$SIDLENS_WORK/venv/bin/python
$PY -m sidlens.cli bundle pull ar-20260925 --repo LeoRodrigues05/sidlens-substrate --dry-run
$PY -m sidlens.cli bundle pull ar-20260925 --repo LeoRodrigues05/sidlens-substrate

# Acceptance:
$PY -m sidlens.cli verify --strict --profile ar       # full sha256 of the ar profile
$PY -m pytest -q "$SIDLENS_REPO/tests"
```

Do not re-run `scripts/data/build_labels.py` there. The pulled label table is
hash-verified, and the absolute `source.path` in its manifest is provenance
only: loaders resolve through `paths.EXTERNAL`. A rebuild would make the local
files differ from the bundle, and a later pull of that bundle would then
refuse.

`pull` checks every file already on disk against the manifest before it
downloads anything. It skips identical files and **refuses** if any file
differs, so it never overwrites. It downloads at the bundle's tag, marks the
frozen files read-only, and finishes with a full hash check.

With `SIDLENS_VERIFY_PROFILE=ar` in `site.env`, the gate at the start of every
SLURM job (`scripts/common/_preamble.sh`) checks exactly that profile. The log shows
`VERIFY OK (profile ar)` and the provenance stamp records `verify_profile=ar`,
so a partial copy can never pass as the full snapshot. `site.env` is also
where that cluster's `SIDLENS_PYTHON` (for conda), `SBATCH_PARTITION` /
`SBATCH_ACCOUNT` and module loads belong. The repo holds no per-cluster
settings.

The older SLURM wrappers pin this cluster's partition and node (`ws-ia`,
`ws-l3-020`). Override them at submission (`sbatch -p ...`); `--nodelist`
cannot be unset from the command line, so copy the wrapper if it gets in the way.
`scripts/activations/ar_capture.sbatch` is site-neutral from the start.

## Step 3 — capture activations

The reference path is `scripts/activations/ar_capture.py`, run through its wrapper:

```bash
cd $SIDLENS_REPO
sbatch -p <partition> --gres=gpu:1 scripts/activations/ar_capture.sbatch \
    --ckpt next-item_best --template eval --layers all
```

The wrapper writes to `$SIDLENS_WORK/derived/ar_capture/<job-id>/`:

- the activation store (`capture/store/`)
- per-digit teacher-forced ranks and log-probs (`capture/digit_scores.csv`)
- `validation.json`
- the standard provenance record: source archive, hashes, git state, environment

It was validated end to end here as job 208116 (8 rows, CPU, fp32):

- The checkpoint sha256 matched the snapshot.
- CSV rows matched the archive (3,681/3,681) and the SID tables (21,630 SIDs).
- Batch-invariance error was 5e-4 on activations of norm ~300, and 6e-6 on
  digit log-probs.

### Traps that apply wherever the capture code comes from

These are enforced in `src/sidlens/data/ar_prompts.py`. Its docstring has the
detail, and `tests/data/test_ar_prompts.py` proves the ids equal upstream's own
dataset classes row for row.

1. **Evaluation and training used different prompt wording.** Archived
   predictions come from `EvalSidDataset` ("Can you predict the next possible
   item the user may expect, given the following chronological interaction
   history: …"). Training and validation loss used `SidSFTDataset` ("The user
   has interacted with items … in chronological order. …"). A capture must
   record which one it used (`--template eval|sft`).
2. **Instruction and prompt are tokenized separately** and concatenated.
3. **Digit d is scored at `predict_pos(slot, d)`.** For digit 0 that is the
   ":\n" of "### Response:\n"; for d > 0 it is the previous target digit,
   under teacher forcing. Do not compute `prompt_len + d`.
4. **Left padding shifts RoPE positions** unless `position_ids` are passed.
   `collate()` returns them.
5. **Token ids come from `added_tokens.json`** (`models.ar.load_vocab`), never
   from `base + code`. RQ-VAE 4×128 has only 39 codes at digit 0.
6. **The AR test cohort (3,681 rows, 1,606 users) is not the diffusion cohort
   (6,297 users).** Never join them by row number.

If upstream's own capture or intervention code is used instead, feed it the
ids from `ar_prompts.encode(...)` or assert equality with them. Write its
outputs with `hooks.store.StoreWriter`, whose rows carry `example_id`, `pos`,
`role`, `item`, `digit` and `code`. That keeps the result readable here with
`StoreReader`, independent of which code produced it. Interventions change
numerics, so they belong in `src/sidlens/interventions/` (still a scaffold),
not in `hooks/`.

### New AR checkpoints trained there

`models.ar` loads only the three frozen checkpoints, and it does so on
purpose. A checkpoint trained on the other cluster becomes SidLens evidence
only after it is exported under `$SIDLENS_WORK/runs/<run-id>/`, hashed and
registered, following the "Run export contract" in the README. The loader
for `runs/` is not written yet. It is the next piece of code needed if the
interventions target new checkpoints rather than the three frozen ones.

## Step 4 — send results back

```bash
# on the other cluster, login node
$PY -m sidlens.cli bundle create --profile results \
    --extra derived/ar_capture/<job-id> --bundle-id arcap-<job-id>
$PY -m sidlens.cli bundle push arcap-<job-id> --repo LeoRodrigues05/sidlens-activations

# back here (or anywhere)
$PY -m sidlens.cli bundle pull arcap-<job-id> --repo LeoRodrigues05/sidlens-activations
```

Use one bundle per run directory. `--extra` refuses paths outside
`$SIDLENS_WORK` and anything under `frozen/`. A pull refuses to overwrite a
local directory whose contents differ.

## How much bulk storage

Measured from the actual prompts: residual stream, 28 layers × 1,536, bf16.
fp32 doubles every figure. "Selected" means the history SID tokens, the
three response-header tokens and the target SID tokens.

| Checkpoint | Split | Rows | Users | All positions | Selected positions |
|---|---|---:|---:|---:|---:|
| `next-item_best` | test | 3,681 | 1,606 | 31.8 GB | 6.5 GB |
| | valid | 3,680 | 1,608 | 31.4 GB | 6.3 GB |
| | train | 29,444 | 5,757 | 239.7 GB | 43.6 GB |
| `oneoff_rqvae4cb128` | test | 3,681 | 1,606 | 33.6 GB | 8.4 GB |
| | train | 29,444 | 5,757 | 251.7 GB | 55.5 GB |
| `two-item_best` | test | 3,452 | 1,606 | 32.3 GB | 8.2 GB |
| | train | 23,687 | 5,510 | 212.7 GB | 50.1 GB |

A sensible first capture is test and valid, selected positions, all layers,
for all three checkpoints: about 45 GB. That is enough for user-disjoint
probes and for choosing patching sites. Full-position train captures are 200+
GB per checkpoint. Capture them only for a specific layer set once probes have
said which layers matter. Check the private-storage quota of your Hub plan
before pushing tens of GB: private storage on the Hub is metered, and the free
allowance is far below a full training-set capture. The store keeps each shard
at or below 1 GiB (`--shard-gb`), far under the Hub's per-file limit, and the
file count stays in the hundreds.
