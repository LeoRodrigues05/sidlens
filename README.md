# sidlens

Mechanistic interpretability of Semantic-ID generative recommenders — the
analysis arm of [OneDiffRec](../OneDiffRec).

Upstream trains and scores AR (Qwen2.5-1.5B SFT) and masked-diffusion (DiffGRM)
recommenders across three SID quantizers. This repo asks what those models
compute internally.

## Why this is a separate repo

Three properties of the upstream setup make in-place work unsafe:

1. **No code isolation.** Every diffusion run directory symlinks its code back
   into the live tree — all 20 runs point at the same mutable `DiffGRM/genrec`.
   Editing a model file retroactively changes what every checkpoint "was trained
   with."
2. **The baseline is a dirty tree, not a commit.** 72 modified/untracked paths at
   `eae9ecc`. Every checkpoint on disk was produced by code in no commit, so a
   git submodule cannot express the state we need to pin.
3. **The benchmark pipeline destroys what interp needs.** `sweep_runner.sbatch`
   deletes AR weights after scoring; 26 of 28 AR runs kept metrics but no weights.

## Structure

    vendor/          byte-frozen upstream code. NEVER EDITED.
    src/sidlens/     the interp codebase
    experiments/     one directory per experiment in the project brief
    manifests/       provenance JSON, checked in

    $SIDLENS_WORK (/l/users/leo.rodrigues/sidlens)
      frozen/        hash-verified read-only substrate (14.75 GB)
      derived/       activations, SAEs, probe caches
      runs/          new training output

## Discipline

**`vendor/` is byte-frozen. Every deviation is a registered monkeypatch** in
`sidlens.models.patches`, carrying an id, target, source hash, rationale, and a
`changes_numerics` flag. Every result records the patch set that was active, so
an analysis cannot silently run different model code than it claims.

**`frozen/` is written once** by `sidlens freeze`, then `chmod a-w`. Nothing in
`src/` may write there.

## Commands

    sidlens freeze --snapshot-id <id>   # build the substrate (once)
    sidlens verify --strict             # re-hash everything; head of every job
    sidlens show --sections             # summarize a snapshot

`verify` runs at the head of every SLURM job. If it is red, nothing downstream
is trustworthy.
