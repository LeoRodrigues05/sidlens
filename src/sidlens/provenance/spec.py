"""Declarative inventory of the frozen substrate.

Every entry names a destination under FROZEN and one or more source globs.
Nothing outside this file decides what gets copied, so "what is our substrate?"
has exactly one answer that can be read, reviewed, and diffed.

`kind` drives verification strictness:
    sid     - SID tables. The consistency crux. Reuse-only; several cannot be
              regenerated at all (rqvae has no checkpoints, MQ has no code).
    data    - interactions, embeddings, metadata, splits.
    ckpt    - model weights + the logs that are their only config record.
    result  - archived metrics/predictions; the validation targets for retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The live lineage only. `OneDiffRec/data/Amazon/` holds a legacy 3686-item
# Industrial set and a 3459-item Office_Products set -- a DIFFERENT item set,
# deliberately excluded so it can never be joined against Amazon18 SIDs.
CATEGORIES = ("Industrial_and_Scientific", "Office")
PRIMARY_CATEGORY = "Industrial_and_Scientific"


@dataclass(frozen=True)
class Entry:
    dest: str            # path under FROZEN
    root: str            # "repo" (upstream code tree) or "work" (upstream bulk)
    globs: tuple[str, ...]
    kind: str
    note: str = ""
    required: bool = True
    min_files: int = 1


def _e(dest, root, globs, kind, note="", required=True, min_files=1) -> Entry:
    return Entry(dest, root, tuple(globs), kind, note, required, min_files)


SPEC: tuple[Entry, ...] = (
    # ---------------------------------------------------------------- SIDs --
    _e("sids/index_json", "work",
       [f"data/Amazon18/{c}/{c}.index*.json" for c in CATEGORIES]
       + [f"data/Amazon18/{c}/{c}.rqkmeans.index_*.json" for c in CATEGORIES],
       "sid",
       "27 index JSONs per category: {rqvae,rqkmeans,MQ} x {3,4,5}cb x {128,256,512}. "
       "rqvae and MQ can never be regenerated; rqkmeans only non-deterministically.",
       min_files=40),

    _e("sids/mispacked", "work",
       [f"data/Amazon18/{c}/*.mispacked" for c in CATEGORIES],
       "sid",
       "Pre-repair rqkmeans originals, kept as evidence for the 2026-08-20 "
       "bit-unpacking repair. AR rqkmeans 5cb x 512 was trained on these.",
       min_files=5),

    _e("sids/sem_ids/diffgrm", "repo",
       [f"DiffGRM/cache/AmazonReviews2014/{PRIMARY_CATEGORY}/processed/*.sem_ids"],
       "sid", "ASIN-keyed raw codes consumed by the diffusion tokenizer.",
       min_files=27),

    _e("sids/sem_ids/diffgrm_new", "repo",
       [f"DiffGRM_new/cache/AmazonReviews2014/{PRIMARY_CATEGORY}/processed/*.sem_ids"],
       "sid", "Two-item tree copy. Should agree byte-for-byte with diffgrm/.",
       min_files=27),

    _e("sids/mappings", "repo",
       [f"DiffGRM/cache/AmazonReviews2014/{PRIMARY_CATEGORY}/processed/item_id2tokens_*.npy",
        f"DiffGRM/cache/AmazonReviews2014/{PRIMARY_CATEGORY}/processed/tokens2item_*.pkl"],
       "sid",
       "Forward/inverse maps. tokens2item is LOSSY (last writer wins on collision) "
       "-- use cb2items for any item-level decode.",
       min_files=2),

    _e("sids/info/next-item", "work",
       [f"data/Amazon18/info/{PRIMARY_CATEGORY}*.txt"],
       "sid", "SID <tab> title <tab> item_id. Sole input to the constrained-decoding trie.",
       min_files=18),

    _e("sids/info/two-item", "work",
       [f"data/two-item/info/{PRIMARY_CATEGORY}*.txt"],
       "sid", "", min_files=18),

    # ---------------------------------------------------------------- data --
    _e("data/embeddings", "work",
       [f"data/Amazon18/{c}/{c}.emb-qwen-td.npy" for c in CATEGORIES],
       "data",
       "Pre-quantization item embeddings. Qwen3-Embedding-4B, mean-pooled over "
       "[title, description], NO PCA, NO L2 norm. (3105, 2560) / (17696, 2560) fp16. "
       "The 'sentence-t5-base_pca256' in DiffGRM filenames is an artifact -- external "
       "mode short-circuits before any sentence encoding runs.",
       min_files=2),

    _e("data/item_meta", "work",
       [f"data/Amazon18/{c}/{c}.item.json" for c in CATEGORIES],
       "data", "title/brand/description present; `categories` empty for every item.",
       min_files=2),

    _e("data/id_maps", "work",
       [f"data/Amazon18/{c}/{c}.item2id" for c in CATEGORIES]
       + [f"data/Amazon18/{c}/{c}.user2id" for c in CATEGORIES],
       "data", "ASIN <-> OneDiffRec int id. DiffGRM id == OneDiffRec id + 1.",
       min_files=4),

    _e("data/interactions", "work",
       [f"data/Amazon18/{c}/{c}.inter.json" for c in CATEGORIES]
       + [f"data/Amazon18/{c}/{c}.*.inter" for c in CATEGORIES],
       "data", "", min_files=2),

    _e("data/reviews", "work",
       [f"data/Amazon18/{c}/{c}.review.json" for c in CATEGORIES],
       "data", "Keyed '(uid, iid, unix_ts)' -> {review, summary}.", min_files=2),

    # One entry per split: the three source dirs hold same-named files, so a
    # shared destination would silently flatten train over valid over test.
    *[
        _e(f"data/splits/next-item/{s}", "work",
           [f"data/Amazon18/{s}/{PRIMARY_CATEGORY}*.csv"],
           "data", "Per-variant SFT CSVs, Industrial only.", min_files=18)
        for s in ("train", "valid", "test")
    ],

    *[
        _e(f"data/splits/two-item/{s}", "work",
           [f"data/two-item/{s}/{PRIMARY_CATEGORY}*.csv"],
           "data", "", min_files=18)
        for s in ("train", "valid", "test")
    ],

    _e("data/sequences", "repo",
       [f"DiffGRM/cache/AmazonReviews2014/{PRIMARY_CATEGORY}/processed/all_item_seqs.json",
        f"DiffGRM/cache/AmazonReviews2014/{PRIMARY_CATEGORY}/processed/id_mapping.json"],
       "data", "DiffGRM's own sequence + id view.", min_files=2),

    # --------------------------------------------------------- checkpoints --
    _e("ckpt/diffusion/next1", "work",
       ["diffgrm/run/*/saved/*/pytorch_model.bin"],
       "ckpt", "20 next-item diffusion checkpoints, bare state_dicts.", min_files=20),

    _e("ckpt/diffusion/next1_logs", "work",
       ["diffgrm/run/*/logs/AmazonReviews2014/DIFF_GRM/*.log"],
       "ckpt",
       "The ONLY per-checkpoint config record. Recovers every shape-visible field; "
       "n_head is NOT recoverable from these and comes from the vendored sbatch.",
       min_files=20),

    _e("ckpt/diffusion/next2", "work",
       ["diffgrm_2item/run/*/saved/*/pytorch_model.bin"],
       "ckpt", "Two-item diffusion. One run may still be training.", min_files=2),

    _e("ckpt/diffusion/next2_logs", "work",
       ["diffgrm_2item/run/*/logs/AmazonReviews2014/DIFF_GRM/*.log"],
       "ckpt", "", min_files=2),

    _e("ckpt/ar/next-item_best", "work",
       [f"sweep/next-item/best/{PRIMARY_CATEGORY}/**"],
       "ckpt", "= rqkmeans 3cb x 128. Post-repair. The retraining anchor.", min_files=3),

    _e("ckpt/ar/two-item_best", "work",
       [f"sweep/two-item/best/{PRIMARY_CATEGORY}/**"],
       "ckpt", "= MQ 4cb x 256.", min_files=3),

    _e("ckpt/ar/oneoff_rqvae4cb128", "work",
       ["outputs/sft-full-160793/final_checkpoint/**"],
       "ckpt",
       "= rqvae 4cb x 128 (423 SID tokens), from run_sft.sbatch not the sweep, so it "
       "is NOT the model behind the published row. Second validation anchor; also the "
       "source of the measured run-to-run noise floor.",
       min_files=3),

    _e("ckpt/ar/best_variant_labels", "work",
       ["sweep/*/best/*.variant"], "ckpt", "", min_files=2),

    _e("base_models/qwen2.5-1.5b-instruct", "work",
       ["models/Qwen2.5-1.5B-Instruct/**"],
       "ckpt", "AR base model, needed for retraining.", min_files=5),

    # -------------------------------------------------------------- results --
    # One section: upstream writes these four families side by side, and
    # globbing them separately double-copies (`*.json` also matches
    # `*.assets.json` and `*.predictions.json`).
    _e("results/sweep_metrics", "work",
       ["sweep/*/metrics/*"],
       "result",
       "28 runs x {metrics.json, assets.json, predictions.json, eval.log}. "
       "metrics.json carries eval_loss_curve / best_model_checkpoint / "
       "stopped_at_step -- the AR retraining validation targets. "
       "predictions.json is 3681 samples x 50 ranked beams.",
       min_files=112),

    _e("results/sweep_state", "work",
       ["sweep/*/state/*", "sweep/*/*.tsv"],
       "result", "Per-variant status + train_seconds; the run-timeline evidence.",
       min_files=28),

    # next1 and next2 use the same variant filenames, so they need separate roots.
    _e("results/diffusion_transcripts/next1", "work",
       [f"diffgrm/{PRIMARY_CATEGORY}/*.txt"],
       "result", "Full train+eval transcripts; tail carries the Test Results OrderedDict.",
       min_files=20),

    _e("results/diffusion_transcripts/next2", "work",
       [f"diffgrm_2item/{PRIMARY_CATEGORY}/*.txt"],
       "result", "", min_files=2),

    _e("results/wandb", "work", ["wandb/**"], "result",
       "Offline runs: FLOPs, GPU energy, wall time.", required=False, min_files=1),

    _e("results/job_manifests", "work", ["manifests/**"], "result",
       "Upstream's own per-job provenance (git head, source.patch, pip-freeze).",
       min_files=3),

    # Two independent analysis lineages that reuse filenames (collisions.csv,
    # metrics.csv). `new/` is a later re-analysis, not a revision of the first.
    _e("results/diagram_data/v1", "repo",
       ["diagrams/data/*.csv", "diagrams/README.md"],
       "result",
       "diagrams/README.md is the clearest existing statement of the SID-level "
       "vs item-level metric problem.",
       min_files=3),

    _e("results/diagram_data/v2", "repo",
       ["diagrams/new/data/*.csv", "diagrams/new/analysis_report.md"],
       "result", "18 tidy CSVs incl. the 54-run core grid and W&B resource joins.",
       min_files=10),

    _e("results/collision_workbook", "repo",
       ["data/CollisionRates_GenRec_Paradigms.xlsx"],
       "result", "Hand-maintained source for the published collision tables.",
       min_files=1),

    _e("results/slurm_logs", "repo", ["slurm_logs/*.out"], "result",
       "", required=False, min_files=1),
)


# Upstream code vendored byte-frozen into vendor/. Never edited; every deviation
# is a registered monkeypatch in sidlens.models.patches.
VENDOR_SPEC = {
    "onediffrec": [
        "sft.py", "data.py", "evaluate.py", "evaluate_two_item.py",
        "LogitProcessor.py", "calc.py", "calc_two_item.py", "split.py",
        "merge.py", "utility.py", "rl.py", "minionerec_trainer.py",
        "sasrec.py", "SASRecModules_ori.py",
        "convert_dataset.py", "convert_index_to_diffgrm.py",
        "convert_inter_to_diffgrm.py", "create_office_mapping.py",
        "rq/**", "config/**",
        "requirements.txt", "requirements-sft.txt",
        "requirements-eval.txt", "requirements-diffgrm.txt",
        # The sbatch files are load-bearing CONFIG, not scripts: they are the
        # only record of hyperparameters that CLI-override the YAMLs, including
        # n_head=4 which no checkpoint or log can reveal.
        "scripts/**",
        "data/amazon18_data_process.py", "data/process.py",
        "data/convert_to_two_item.py",
    ],
    "diffgrm": ["DiffGRM/main.py", "DiffGRM/genrec/**"],
    "diffgrm_new": ["DiffGRM_new/main.py", "DiffGRM_new/genrec/**"],
}
