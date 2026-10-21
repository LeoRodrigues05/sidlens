"""Inputs of a grid run, built as the objects sidlens's experiments already use.

An experiment written for the frozen checkpoints reads an `ArExample` list, a
`SidTable`, a `SidVocab`, a `DiffusionEvalCohort` and a DIFF_GRM model. This
module builds the same objects from a grid run's own files, so the analysis
code below an experiment's input step does not change.

Traps, each with the guard that closes it
-----------------------------------------
1. **The ZCR twin's table.** A ZCR run trained on its ZCR table, not on the
   native one with the same variant name. Every table is read from the path the
   run's prepared manifest names, and its sha256 must equal the registry's.
2. **Item ids.** AR rows name items by integer index (`item2id` of the run's
   data root), DiffGRM rows by ASIN; both are mapped through the data root's
   own `item2id`, never the frozen Industrial map.
3. **The prompt.** `check_prompts` builds every row with sidlens's `v3`
   template and with OneDiffRec's own `SidSFTDataset` (vendored) and requires
   identical token ids, with and without the target.
4. **The DiffGRM model.** Built exactly as OneDiffRec's evaluator builds it
   (`evaluate_diff_joint.load_model`): the run's full resolved config, a
   tokenizer stub with only the four fields the model reads, a strict state
   dict load, and the trained target width checked.
5. **Sampled Office users.** Office has about 61,600 test users. Experiments
   that run a model use a fixed sample of `SAMPLE_USERS` users per catalog,
   drawn by hash of (seed, user) from the cohort's users, so both paradigms,
   every map and every weight state see the same users.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sidlens.data import ar_prompts as P
from sidlens.data.diffusion_eval import DiffusionEvalCohort
from sidlens.data.sids import RE_TOKEN, SidTable, SidVariant
from sidlens.grid import registry, v3code
from sidlens.provenance.hashing import HashCache, sha256_file

SAMPLE_SEED = 20261021
SAMPLE_USERS = 6297          # Industrial's whole test cohort; Office is sampled to this size
SPLIT_FILES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
DIFF_CACHE_CATEGORY = {"industrial": "Industrial_and_Scientific", "office": "Office_Products"}


# --------------------------------------------------------------- catalog --
def entry(name: str) -> dict:
    return registry.load().entry(name)


def category_dir(e: dict) -> Path:
    return Path(e["data_root"]) / "Amazon18" / e["category"]


def _id_map(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw, idx = line.split("\t")
        if raw in out:
            raise ValueError(f"{path}:{n}: duplicate {raw!r}")
        out[raw] = int(idx)
    if sorted(out.values()) != list(range(len(out))):
        raise ValueError(f"{path}: ids must be contiguous and zero-based")
    return out


def item2id(e: dict) -> dict[str, int]:
    """ASIN -> item index of the run's data root (= AR item id, = embedding row)."""
    return _id_map(category_dir(e) / f"{e['category']}.item2id")


def user2id(e: dict) -> dict[str, int]:
    return _id_map(category_dir(e) / f"{e['category']}.user2id")


def sid_table(e: dict) -> SidTable:
    """The SID table the run trained on (native or ZCR), keyed by ASIN in item order."""
    path = Path(e["sid_index"]["path"])
    if sha256_file(path) != e["sid_index"]["sha256"]:
        raise RuntimeError(f"{e['key']}: SID table {path} changed since registration")
    raw = json.loads(path.read_text())
    ids = item2id(e)
    id2asin = {i: a for a, i in ids.items()}
    if set(raw) != {str(i) for i in id2asin}:
        raise ValueError(f"{path}: table keys differ from the data root's item ids")
    codes = {}
    for i in range(len(id2asin)):
        tokens = raw[str(i)]
        codes[id2asin[i]] = tuple(int(RE_TOKEN.fullmatch(t).group(2)) for t in tokens)
    return SidTable(SidVariant.parse(e["variant"]), codes, path, ids)


# -------------------------------------------------------------- samples --
def sample_users(users, dataset: str, n: int = SAMPLE_USERS) -> set[str]:
    """The fixed user sample of one catalog: the `n` users with the smallest
    sha256(seed|user). All users when the cohort has at most `n`."""
    users = sorted(set(users))
    if len(users) <= n:
        return set(users)
    key = lambda u: hashlib.sha256(f"{SAMPLE_SEED}|{dataset}|{u}".encode()).hexdigest()  # noqa: E731
    return set(sorted(users, key=key)[:n])


# ------------------------------------------------------------------- AR --
def csv_path(e: dict, split: str) -> Path:
    return Path(e["prepared_dir"]) / f"{SPLIT_FILES[split]}.csv"


def ar_examples(e: dict, split: str = "test", *, sample: bool = True) -> list[P.ArExample]:
    """The run's prepared rows as `ArExample`s, ids = OneDiffRec `example_id`.

    `sample=True` keeps the catalog's fixed user sample (all of Industrial)."""
    path = csv_path(e, split)
    _check_prepared(e, path)
    split_name = "valid" if split in ("valid", "val") else split
    ex = P.load_examples(e["variant"], e["task"], split_name, path=path, id_column="example_id")
    if sample:
        keep = sample_users([x.user_id for x in ex], e["dataset"])
        ex = [x for x in ex if x.user_id in keep]
    return ex


def _check_prepared(e: dict, path: Path) -> None:
    manifest = json.loads((Path(e["prepared_dir"]) / "data-manifest.json").read_text())
    want = manifest["output_hashes"].get(path.name)
    if want is None or sha256_file(path) != want:
        raise RuntimeError(f"{path} differs from its prepared manifest")


def check_prompts(examples: list[P.ArExample], tokenizer, vocab, csv: Path,
                  limit: int | None = 200) -> dict:
    """Token ids of the `v3` template against OneDiffRec's own `SidSFTDataset`.

    OneDiffRec's class reads the whole CSV, so rows are matched by their
    position in the file (`ArExample.row`). Both the test-time prompt
    (test=True) and the teacher-forced input with the target and EOS
    (test=False) must match on every compared row.
    """
    data = v3code.ar_data()
    rows = examples if limit is None else examples[:limit]
    out = {"n": len(rows), "n_bad_prompt": 0, "n_bad_target": 0, "n_bad_eval": 0, "bad_rows": []}
    # The test-time prompt as OneDiffRec's evaluator builds it (evaluate_ar_joint.encode_prompt).
    ev = v3code.ar_eval()
    for ex in rows:
        ours = P.encode(ex, tokenizer, vocab, template="v3").input_ids
        if list(ours) != list(ev.encode_prompt(tokenizer, list(ex.history_sids))):
            out["n_bad_eval"] += 1
            if len(out["bad_rows"]) < 10:
                out["bad_rows"].append(ex.row)
    import os
    import warnings
    os.environ.setdefault("TQDM_DISABLE", "1")          # OneDiffRec's class shows a progress bar per CSV
    for test, field in ((True, "n_bad_prompt"), (False, "n_bad_target")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")               # pandas chained-assignment warning inside it
            ds = data.SidSFTDataset(train_file=str(csv), tokenizer=tokenizer, test=test)
        for ex in rows:
            ours = P.encode(ex, tokenizer, vocab, template="v3", with_target=not test).input_ids
            theirs = ds[ex.row]["input_ids"]
            if list(ours) != list(theirs):
                out[field] += 1
                if len(out["bad_rows"]) < 10:
                    out["bad_rows"].append(ex.row)
    out["ok"] = out["n_bad_prompt"] == 0 and out["n_bad_target"] == 0 and out["n_bad_eval"] == 0
    return out


class ArSetup:
    """Everything an AR experiment's input step needs for one grid weight state."""

    def __init__(self, ckpt: str, split: str, tokenizer, vocab, *, task: str | None = None,
                 sample: bool = True, prompt_rows: int | None = 200):
        from sidlens.models import ar
        self.ckpt = ckpt
        self.entry = registry.load().entry(ckpt)
        e = self.entry
        if task is not None and e["task"] != task:
            raise ValueError(f"{ckpt} is a {e['task']} run, this experiment needs {task}")
        self.template = "v3"
        self.csv = csv_path(e, split)
        self.examples = ar_examples(e, split, sample=sample)
        self.table = sid_table(e)
        self.item2id = item2id(e)
        self.validation = {
            "table": P.check_against_table(self.examples, self.table, self.item2id),
            "vocab_vs_table": ar.validate_against(vocab, self.table),
            "prompts_vs_onediffrec": check_prompts(self.examples, tokenizer, vocab, self.csv, limit=prompt_rows),
            "run_registered_usable": {"ok": bool(e["usable"]), "problems": e["problems"]},
        }
        self.info = {"grid_key": e["key"], "grid_state": ckpt.split("@")[-1], "dataset": e["dataset"],
                     "sid_assignment": e["sid_assignment"], "csv": str(self.csv),
                     "csv_sha256": sha256_file(self.csv), "template": self.template,
                     "user_sample": f"{SAMPLE_USERS} users by sha256({SAMPLE_SEED}|dataset|user)"
                     if sample else "all users"}


def sort_by_length(examples: list, encoded: list) -> tuple[list, list]:
    """Rows ordered by encoded length (stable), so batches rarely span buckets.
    Row identity is the example id, which every record carries."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    return [examples[i] for i in order], [encoded[i] for i in order]


# -------------------------------------------------------------- DiffGRM --
def sem_ids_path(e: dict) -> Path:
    return (Path(e["cache_root"]) / "AmazonReviews2014" / DIFF_CACHE_CATEGORY[e["dataset"]]
            / "processed" / f"{e['variant']}.sem_ids")


def diffgrm_cohort(e: dict, split: str = "test", *, sample: bool = True) -> DiffusionEvalCohort:
    """The DiffGRM test cohort of a next-item grid run, as `DiffusionEvalCohort`.

    Rows follow the prepared cohort file; histories keep the most recent
    `max_history_len` items, right-padded with -1 (OneDiffRec's evaluator).
    """
    if e["paradigm"] != "diffgrm" or e["predtype"] != "next-item":
        raise ValueError(f"{e['key']}: the DiffGRM cohort is built for next-item DiffGRM runs")
    prepared = Path(e["prepared_dir"])
    cohort_file = prepared / f"{SPLIT_FILES[split]}-cohort.json"
    _check_prepared(e, cohort_file)
    examples = json.loads(cohort_file.read_text())["examples"]
    table = sid_table(e)
    sem = json.loads(sem_ids_path(e).read_text())
    if {k: tuple(v) for k, v in sem.items()} != table.asin2codes:
        raise RuntimeError(f"{e['key']}: the DiffGRM .sem_ids differ from the run's SID table")
    items, users = item2id(e), user2id(e)
    if sample:
        keep = sample_users([x["user_id"] for x in examples], e["dataset"])
        examples = [x for x in examples if x["user_id"] in keep]
    H = int((e.get("config") or {}).get("max_history_len", 50))
    d = table.variant.n_codebook
    N = len(examples)
    histories = np.full((N, H, d), -1, dtype=np.int64)
    mask = np.zeros((N, H), dtype=np.bool_)
    lengths = np.empty(N, dtype=np.int64)
    targets = np.empty((N, d), dtype=np.int64)
    target_ids = np.empty(N, dtype=np.int64)
    ident = hashlib.sha256()
    for r, x in enumerate(examples):
        hist = [str(a) for a in x["history_item_ids"]][-H:]
        tgt = [str(a) for a in x["target_item_ids"]]
        if len(tgt) != 1 or not hist:
            raise ValueError(f"{cohort_file} row {r}: expected one target and a history")
        histories[r, :len(hist)] = [table.asin2codes[a] for a in hist]
        mask[r, :len(hist)] = True
        lengths[r] = len(hist)
        targets[r] = table.asin2codes[tgt[0]]
        target_ids[r] = items[tgt[0]]
        ident.update(json.dumps([x["example_id"], hist, tgt], separators=(",", ":")).encode())
    buckets = {sid: tuple(sorted(items[a] for a in asins)) for sid, asins in table.cb2items.items()}
    return DiffusionEvalCohort(
        users=tuple(str(x["user_id"]) for x in examples),
        user_ids=np.array([users[str(x["user_id"])] for x in examples], dtype=np.int64),
        histories=histories, history_mask=mask, history_lengths=lengths,
        sequence_lengths=lengths + 1, target_sids=targets, target_item_ids=target_ids,
        target_asins=tuple(str(x["target_item_ids"][0]) for x in examples),
        sid_buckets=buckets, item_to_sid={items[a]: c for a, c in table.asin2codes.items()},
        input_sha256={str(cohort_file): sha256_file(cohort_file), str(table.source): sha256_file(table.source)},
        cohort_sha256=ident.hexdigest(),
        sid_lineage={"sem_ids_name": e["variant"], "sid_assignment": e["sid_assignment"],
                     "table": str(table.source), "grid_key": e["key"],
                     "n_catalogue_items": table.n_items, "n_unique_sids": len(buckets)},
        example_ids=tuple(str(x["example_id"]) for x in examples))


def diffgrm_model(name: str, device: str = "cpu"):
    """A grid DiffGRM weight state, built as OneDiffRec's evaluator builds it.

    Returns (model, report). The weights' sha256 is checked against the registry.
    """
    import torch
    from sidlens import vendorpath

    reg = registry.load()
    e, st = reg.state(name)
    if e["paradigm"] != "diffgrm":
        raise KeyError(f"{name} is not a DiffGRM model")
    sha = registry.verify_state(name, HashCache())
    config = dict(e["config"] or {})
    required = ("n_digit", "codebook_size", "n_embd", "n_head", "n_inner", "encoder_n_layer",
                "decoder_n_layer", "dropout", "attn_pdrop", "resid_pdrop")
    if missing := [k for k in required if k not in config]:
        raise ValueError(f"{name}: resolved config lacks {missing}")
    vendorpath.activate("diffgrm_new")
    import importlib
    model_mod = importlib.import_module("genrec.models.DIFF_GRM.model")
    vendorpath.assert_frozen(model_mod)
    _check_model_source(e)
    n_digit, K = int(config["n_digit"]), int(config["codebook_size"])
    tok = SimpleNamespace(sid_offset=3, n_digit=n_digit, codebook_size=K, vocab_size=3 + n_digit * K)
    model = model_mod.DIFF_GRM(config, None, tok)
    state = torch.load(st["path"], map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    n_targets = 2 if e["predtype"] == "next2item" else 1
    if int(model.mask_emb_table.num_embeddings) != n_targets * n_digit:
        raise ValueError(f"{name}: checkpoint does not have the trained target slots")
    model.to(device).eval()
    return model, {"ok": True, "sha256": sha, "config": {k: config[k] for k in required},
                   "n_params": sum(p.numel() for p in model.parameters())}


def _check_model_source(e: dict) -> None:
    """The run's own DIFF_GRM model file must equal the vendored one we build from."""
    from sidlens import paths
    ours = paths.VENDOR / "diffgrm_new" / "genrec" / "models" / "DIFF_GRM" / "model.py"
    theirs = Path(e["source_snapshot"]) / "DiffGRM_new" / "genrec" / "models" / "DIFF_GRM" / "model.py"
    if not theirs.is_file() or sha256_file(ours) != sha256_file(theirs):
        raise RuntimeError(f"{e['key']}: the run's DIFF_GRM model.py differs from vendor/diffgrm_new")
