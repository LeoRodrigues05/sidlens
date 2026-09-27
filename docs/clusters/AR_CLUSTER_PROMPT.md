# Prompt for the agent on the upstream (AR) cluster

Copy everything below the rule into the coding agent working on the cluster
that holds the upstream OneDiffRec code. The setup steps are in
[CROSS_CLUSTER.md](CROSS_CLUSTER.md); this prompt adds the contracts that the
agent's work must respect.

---

You are working on the cluster that holds the upstream OneDiffRec code: AR
SFT training and evaluation, branch `leo-dev-changes`, the next-2-item sweep,
ZCR/CCE, and training-time activation logging. The goal is to capture
intermediate activations from the autoregressive Qwen2.5-1.5B SID
recommenders and run interventions on them. The results must be readable by
SidLens, the analysis repository (`https://github.com/LeoRodrigues05/sidlens`).
Read its `CLAUDE.md` and `docs/clusters/CROSS_CLUSTER.md` before doing anything else.

## 1. Set up SidLens here

Follow `docs/clusters/CROSS_CLUSTER.md` step 2: clone, write `~/.config/sidlens/site.env`,
build the environment, `sidlens bundle pull` the `ar` bundle from the
private dataset repo, then `sidlens verify --strict --profile ar` and the test
suite. Report any failure verbatim. Do not work around a failed verify:
never edit `frozen/`, `vendor/` or `manifests/`.

## 2. Inventory before writing code

Report, with paths:

- the upstream capture/intervention code: entry points, hooked modules, and
  what it records (layers, positions, dtype, examples);
- its on-disk activation format and the size of any existing logs;
- which AR checkpoints exist here, their SID variant, and whether their
  weights still exist. Hash-compare them against the three frozen SidLens
  checkpoints (the sha256 values are in `manifests/provenance.base-20260826.json`,
  sections `ckpt/ar/*`);
- whether upstream prompt construction differs from
  `vendor/onediffrec/data.py` (`EvalSidDataset` / `SidSFTDataset`) in the
  SidLens repo. The vendored copy is upstream commit `eae9ecc` plus 72 dirty
  paths; `leo-dev-changes` may have moved on.

## 3. Contracts every capture or intervention must meet

1. **Inputs.** Use `sidlens.data.ar_prompts.encode(...)` for input ids, or
   assert equality with it for every row you use. Record the template:
   `eval` (the wording behind every archived prediction) or `sft` (the
   training wording). They differ, so this is not optional.
2. **Positions.** Read digit d of target slot s at
   `Encoded.predict_pos(s, d)`, never `prompt_len + d`. If you left-pad, pass
   the `position_ids` from `ar_prompts.collate`.
3. **Token ids** come from `sidlens.models.ar.load_vocab` (`added_tokens.json`),
   never from `base + code`.
4. **Identity.** Every stored vector is a row carrying `example_id`, `pos`,
   `role`, `item`, `digit`, `code` (`Encoded.records()`). Example ids look
   like `next-item/test/<variant>/<row>`. The test CSV row index is the join
   key to the archived predictions. Never join the AR cohort to the diffusion
   cohort by row number.
5. **Format.** Write activations with `sidlens.hooks.store.StoreWriter`:
   safetensors shards, parquet rows, manifest written last. Its `meta` must
   name the model and the checkpoint sha256, the template, the dtype and the
   capture config. Convert upstream's existing logs into this format rather
   than inventing another one. Never use pickle.
6. **Interventions** change numerics. Record the clean input, the patched
   input, the site (layer, position/role), the patch source, and the output
   change: target-digit log-prob and rank at `predict_pos`, per digit. Keep
   the clean and patched runs on the same examples and batch composition.
   Add intervention code under `src/sidlens/interventions/`, never in `hooks/`
   (observation only) and never in `vendor/`.
7. **Provenance.** Run on compute nodes through a SLURM wrapper that sources
   `scripts/common/_preamble.sh` and calls `sidlens_gate`. `scripts/activations/ar_capture.sbatch`
   is the template. Write to a new
   `$SIDLENS_WORK/derived/<experiment>/<job-id>/` and never overwrite one.
   For a hypothesis-testing experiment, write `protocol.md` before seeing
   outcomes (see the controlled-experiment conventions in `CLAUDE.md`).
8. **New checkpoints** trained here become evidence only after export
   under `$SIDLENS_WORK/runs/<run-id>/` per the README's run-export contract:
   weights, tokenizer, resolved config, seeds, split and SID hashes, logs,
   predictions. SidLens has no `runs/` loader yet; writing one, with a hash
   check equivalent to `scripts/activations/ar_capture.py::checkpoint_sha`, is in scope
   if the interventions need these checkpoints.

## 4. Sending results back

For each finished run directory, from a node with Hub access:

```bash
sidlens bundle create --profile results --extra derived/<experiment>/<job-id> \
    --bundle-id <short-name>-<job-id>
sidlens bundle push <short-name>-<job-id> --repo <user>/sidlens-activations
```

Before pushing tens of GB, report the size and ask. `docs/clusters/CROSS_CLUSTER.md`
("How much bulk storage") has measured sizes: about 45 GB for test+valid,
selected positions, 28 layers, all three checkpoints; 200+ GB per checkpoint
for every position of the train split.

## 5. Report

Return:

- what ran, with job ids and output directories;
- each validation result (prompt-id equality, archive and table checks,
  checkpoint hashes, batch invariance);
- bundle ids pushed;
- anything that failed or was skipped, and why.

Keep observational results (probe accuracy, attention) separate from causal
ones (interventions). Report intervals over users, and state that they
exclude training-seed variation.
