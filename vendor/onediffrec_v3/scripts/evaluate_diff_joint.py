#!/usr/bin/env python3
"""Evaluate existing Diff checkpoints using all trained target slots.

This creates a new output directory and never rewrites historical evaluation.
Reevaluation does not remove overlap in a checkpoint's training/validation data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cohort(path, split, n_targets, max_examples):
    path = Path(path)
    if path.suffix == ".jsonl":
        payload = {"examples": [json.loads(line) for line in path.read_text().splitlines() if line.strip()]}
    else:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            payload = {"examples": payload}
    if payload.get("split", split) != split:
        raise ValueError("requested split differs from cohort split")
    examples = payload.get("examples", payload.get("records"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("cohort must contain a non-empty examples list")
    if max_examples:
        examples = examples[:max_examples]
    result, seen = [], set()
    for index, source in enumerate(examples):
        row = dict(source)
        row["history_item_ids"] = [str(item) for item in row["history_item_ids"]]
        row["target_item_ids"] = [str(item) for item in row["target_item_ids"]]
        if len(row["target_item_ids"]) != n_targets or not row["history_item_ids"]:
            raise ValueError(f"cohort row {index} has wrong target width or empty history")
        row.setdefault("example_index", index)
        row.setdefault("example_id", hashlib.sha256(json.dumps(
            [index, row.get("user_id"), row["history_item_ids"], row["target_item_ids"]],
            separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest())
        if row["example_id"] in seen:
            raise ValueError("cohort contains duplicate example_id")
        seen.add(row["example_id"])
        result.append(row)
    return result, {key: value for key, value in payload.items() if key not in ("examples", "records")}


def load_model(config_path, checkpoint, source_root, predtype, device):
    import torch
    import yaml
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise ValueError("config must contain the resolved model configuration")
    required = ("n_digit", "codebook_size", "n_embd", "n_head", "n_inner",
                "encoder_n_layer", "decoder_n_layer", "dropout", "attn_pdrop", "resid_pdrop")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config lacks model architecture fields: {missing}")
    n_targets = 2 if predtype == "next2item" else 1
    if int(config.get("n_target_items", 1)) != n_targets:
        raise ValueError("predtype disagrees with saved n_target_items; never resize a trained target head")
    source_root = Path(source_root).resolve()
    sys.path.insert(0, str(source_root))
    model_module = importlib.import_module("genrec.models.DIFF_GRM.model")
    if not Path(model_module.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError("another genrec model was already imported; run this CLI in a fresh process")
    tokenizer = SimpleNamespace(sid_offset=3, n_digit=int(config["n_digit"]),
        codebook_size=int(config["codebook_size"]),
        vocab_size=3 + int(config["n_digit"]) * int(config["codebook_size"]))
    model = model_module.DIFF_GRM(config, None, tokenizer)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if int(model.mask_emb_table.num_embeddings) != n_targets * int(config["n_digit"]):
        raise ValueError("checkpoint does not have the required trained target slots")
    return model.to(device).eval(), config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Resolved JSON/YAML from this training run")
    parser.add_argument("--sem-ids", type=Path, required=True, help="Raw item ID -> integer code list JSON")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, nonexistent directory")
    parser.add_argument("--predtype", choices=("next-item", "next2item"), default="next2item")
    parser.add_argument("--source-root", type=Path, default=REPO / "DiffGRM_new")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--beam-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reveal-order", choices=("left-to-right", "random"), default="left-to-right")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.beam_size, args.top_k) < 1 or args.max_examples < 0 or args.beam_size < args.top_k:
        parser.error("batch/beam/top-k must be positive, beam >= top-k, and max-examples >= 0")
    if args.output_dir.exists():
        parser.error("output directory already exists; use a new evaluation directory")
    import numpy as np
    import torch
    from metrics.ordered_targets import ordered_target_metrics
    from metrics.current_cce import current_cce_metrics
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Load the helper by its file path so an older model source may be selected.
    helper_path = REPO / "DiffGRM_new/genrec/models/DIFF_GRM/joint_decode.py"
    spec = importlib.util.spec_from_file_location("onediffrec_joint_decode", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    model, config = load_model(args.config, args.checkpoint, args.source_root, args.predtype, args.device)
    item_codes = {str(item): tuple(int(code) for code in codes)
                  for item, codes in json.loads(args.sem_ids.read_text()).items()}
    n_targets = 2 if args.predtype == "next2item" else 1
    examples, cohort_metadata = read_cohort(args.cohort, args.split, n_targets, args.max_examples)
    max_history = int(config.get("max_history_len", 50))
    if max_history <= 0:
        raise ValueError("max_history_len must be positive")
    for row in examples:
        absent = [item for item in row["history_item_ids"][-max_history:] + row["target_item_ids"] if item not in item_codes]
        if absent:
            raise ValueError(f"cohort has items absent from the exact SID catalog: {absent[:5]}")
        supplied_targets = row.get("target_item_sids", row.get("target_sids"))
        if supplied_targets is not None and supplied_targets != [helper.sid_text(item_codes[item]) for item in row["target_item_ids"]]:
            raise ValueError("cohort target SIDs do not match the supplied catalog")
    records = []
    for start in range(0, len(examples), args.batch_size):
        selected = examples[start:start + args.batch_size]
        histories, masks, labels = [], [], []
        for row in selected:
            items = row["history_item_ids"][-max_history:]
            padding = max_history - len(items)
            histories.append([list(item_codes[item]) for item in items] + [[-1] * model.n_digit] * padding)
            masks.append([True] * len(items) + [False] * padding)
            labels.append([code for item in row["target_item_ids"] for code in item_codes[item]])
            row["model_history_item_ids"] = items
        batch = {"history_sid": torch.tensor(histories, device=args.device),
                 "history_mask": torch.tensor(masks, device=args.device, dtype=torch.bool)}
        decoded = helper.decode_joint_catalog(model, batch, item_codes.values(), beam_size=args.beam_size,
            top_k=args.top_k, reveal_order=args.reveal_order, seed=args.seed)
        records.extend(helper.prediction_records(decoded, labels, item_codes, selected))
        print(f"Evaluated {len(records)}/{len(examples)} examples", flush=True)
    metrics = ordered_target_metrics(records, ks=sorted({
        k for k in (1, 5, 10, args.top_k) if k <= args.top_k
    }))
    if n_targets == 1:
        metrics["cce"] = current_cce_metrics(records, item_codes,
            metrics["ks"], retained_top_k=args.top_k)
    protocol = {
        "schema": "onediffrec.diffgrm-joint-evaluation/v1",
        "evaluation_protocol": helper.PROTOCOL, "predtype": args.predtype,
        "cce_protocol": "paper-cce-v1" if n_targets == 1 else None,
        "cce_status": "single-target CCE included" if n_targets == 1 else "not defined for ordered pairs",
        "split": args.split, "cohort_metadata": cohort_metadata,
        "examples": len(records), "full_cohort": args.max_examples == 0,
        "score_definition": helper.SCORE_DEFINITION,
        "reveal_order": helper.make_reveal_order(n_targets * model.n_digit, args.reveal_order, args.seed),
        "beam_size": args.beam_size, "top_k": args.top_k, "seed": args.seed,
        "source_root": str(args.source_root.resolve()), "device": args.device,
        "runtime": {"python": sys.version, "torch": str(torch.__version__),
            "numpy": str(np.__version__), "parameter_dtype": str(next(model.parameters()).dtype),
            "cuda": torch.version.cuda, "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "checkpoint_training_split_protocol": config.get("split_protocol", "unknown"),
        "checkpoint_selection_metric": config.get("val_metric", "unknown"),
        "source_sha256": {str(path.relative_to(args.source_root)): sha256(path)
            for path in sorted(args.source_root.rglob("*.py"))},
        "decoder_helper_sha256": sha256(helper_path),
        "metrics_source_sha256": sha256(REPO / "metrics/ordered_targets.py"),
        "cce_source_sha256": {path: sha256(REPO / path)
            for path in ("metrics/current_cce.py", "metrics/collision.py")},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (("checkpoint", args.checkpoint), ("config", args.config),
                               ("sem_ids", args.sem_ids), ("cohort", args.cohort))},
        "limitations": [
            "Reevaluation cannot remove training/validation target overlap from the supplied checkpoint.",
            "Existing decoder has slot-specific masks but revealed tokens have no explicit slot position embedding.",
            "Catalog groups identify SID equivalence classes; collisions do not resolve item identity.",
            "Beam search is approximate; decoding order and beam budget must be held fixed across compared runs.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
