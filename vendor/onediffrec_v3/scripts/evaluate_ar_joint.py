#!/usr/bin/env python3
"""Separate AR evaluation with SFT-aligned prompts and ordered-target decoding.

For next2item, one constrained beam generates SID1 + ' ||| ' + SID2 + '\\n'
+ EOS. Both slots and the separator contribute to the raw-vocabulary joint
log probability. This is approximate beam search, not exhaustive pair ranking.
Historical evaluations and split definitions are unchanged.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.ordered_targets import ordered_target_metrics
from metrics.current_cce import current_cce_metrics
from scripts.run_provenance import fingerprint_path


INSTRUCTION = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nCan you predict the next possible item that the user may expect?\n\n"
)


def encode_training_piece(tokenizer, text, bos=False, eos=False):
    """Match data.Tokenizer.encode used by SidSFTDataset without importing torch."""
    tokens = list(tokenizer.encode(text))
    while tokens and tokens[0] == tokenizer.bos_token_id:
        tokens.pop(0)
    while tokens and tokens[-1] == tokenizer.eos_token_id:
        tokens.pop()
    if bos and tokenizer.bos_token_id is not None:
        tokens.insert(0, tokenizer.bos_token_id)
    if eos:
        if tokenizer.eos_token_id is None:
            raise ValueError("Checkpoint tokenizer has no EOS token")
        tokens.append(tokenizer.eos_token_id)
    return tokens


def encode_prompt(tokenizer, history_sids):
    history = ", ".join(history_sids)
    prompt = (
        "### User Input:\nThe user has interacted with items " + history
        + " in chronological order. Can you predict the next possible item that the user may expect?"
        + "\n\n### Response:\n"
    )
    return encode_training_piece(tokenizer, INSTRUCTION, bos=True) + encode_training_piece(tokenizer, prompt)


class CatalogGrammar:
    """Two composable tries: first SID/separator, then final SID/newline/EOS.

    Tries contain O(catalog size * SID length) edges, rather than materializing
    every catalog pair. Tokenizer boundary compatibility is checked explicitly.
    """

    def __init__(self, tokenizer, sids, horizon=2):
        if horizon not in (1, 2):
            raise ValueError("Only horizons one and two are supported")
        self.horizon = horizon
        self.eos_id = tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("An EOS token is required")
        self.first, self.final = {}, {}
        self.first_max = self.final_max = 0
        self.tokenizer = tokenizer
        separator = encode_training_piece(tokenizer, " ||| ")
        ending = encode_training_piece(tokenizer, "\n", eos=True)
        self.catalog_sids = set(sids)
        for sid in sorted(self.catalog_sids):
            bare = encode_training_piece(tokenizer, sid)
            final = encode_training_piece(tokenizer, sid + "\n", eos=True)
            if not bare or final != bare + ending:
                raise ValueError(f"SID/newline tokenizer boundary is not compositional: {sid}")
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in final:
                raise ValueError(f"Catalog SID has unknown tokens in checkpoint tokenizer: {sid}")
            self._insert(self.final, final, sid)
            self.final_max = max(self.final_max, len(final))
            if horizon == 2:
                first = encode_training_piece(tokenizer, sid + " ||| ")
                following = encode_training_piece(tokenizer, " ||| " + sid + "\n", eos=True)
                if first != bare + separator or following != separator + final:
                    raise ValueError(f"SID/separator tokenizer boundary is not compositional: {sid}")
                self._insert(self.first, first, sid)
                self.first_max = max(self.first_max, len(first))
        if not self.catalog_sids:
            raise ValueError("Empty SID catalog")
        self.max_new_tokens = self.first_max + self.final_max

    @staticmethod
    def _insert(root, tokens, sid):
        node = root
        for token in tokens:
            if "sid" in node:
                raise ValueError("Catalog token sequences have ambiguous prefix boundaries")
            node = node.setdefault(int(token), {})
        if node and node.get("sid") != sid:
            raise ValueError("Catalog token sequences are non-unique or prefix-ambiguous")
        node["sid"] = sid

    def _walk(self, prefix):
        node = self.first if self.horizon == 2 else self.final
        found = []
        consumed = []
        for token in prefix:
            if len(found) == self.horizon:
                if int(token) != self.eos_id:
                    raise ValueError("Non-EOS token after complete ordered target")
                continue
            if int(token) not in node:
                raise ValueError(f"Generated prefix left the legal catalog grammar: {prefix}")
            consumed.append(int(token))
            node = node[int(token)]
            if "sid" in node:
                found.append(node["sid"])
                node = self.final if len(found) < self.horizon else {}
        return node, found, consumed

    def allowed(self, prefix):
        node, found, _ = self._walk(prefix)
        return [self.eos_id] if len(found) == self.horizon else sorted(node)

    def parse(self, prefix):
        _, found, consumed = self._walk(prefix)
        if len(found) != self.horizon:
            raise ValueError("Beam search returned an incomplete ordered target")
        expected = encode_training_piece(self.tokenizer, " ||| ".join(found) + "\n", eos=True)
        if consumed != expected:
            raise ValueError("Generated target tokenization differs from SFT target tokenization")
        return found, consumed


def load_catalog(path):
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict):
        raise ValueError("Index must map item IDs to SID token lists")
    groups = {}
    for item, value in mapping.items():
        sid = "".join(value) if isinstance(value, list) else str(value).strip()
        if not sid or "|||" in sid or "\n" in sid:
            raise ValueError(f"Invalid catalog SID for item {item}")
        groups.setdefault(sid, []).append(str(item))
    return {sid: sorted(items) for sid, items in groups.items()}


def load_examples(path, groups, horizon, max_examples=0):
    examples = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"user_id", "history_item_id", "history_item_sid", "item_id", "item_sid"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Test CSV is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for position, row in enumerate(reader):
            history_ids = ast.literal_eval(row["history_item_id"])
            history_sids = ast.literal_eval(row["history_item_sid"])
            if not isinstance(history_ids, (list, tuple)) or not isinstance(history_sids, (list, tuple)):
                raise ValueError(f"History is not a list at row {position}")
            history_ids, history_sids = list(map(str, history_ids)), list(map(str, history_sids))
            targets = [x.strip() for x in row["item_sid"].split("|||")]
            target_ids = [x.strip() for x in row["item_id"].split("|||")]
            if len(targets) != horizon or len(target_ids) != horizon:
                raise ValueError(f"Row {position} target horizon differs from --predtype")
            if len(history_ids) != len(history_sids):
                raise ValueError(f"Row {position} history ID/SID lengths differ")
            for item, sid in zip(history_ids + target_ids, history_sids + targets):
                if sid not in groups or item not in groups[sid]:
                    raise ValueError(f"Row {position}: item {item} disagrees with the supplied SID index")
            examples.append({
                "example_id": str(row.get("example_id") or row.get("Unnamed: 0") or position),
                "source_row": position,
                "user_id": str(row["user_id"]),
                "history_item_ids": history_ids,
                "history_sids": history_sids,
                "target_item_ids": target_ids,
                "target_sids": targets,
                "target_candidate_item_ids": [groups[sid] for sid in targets],
            })
            for field in ("split_protocol", "history_event_positions", "target_event_positions",
                          "raw_history_item_ids", "raw_target_item_ids"):
                if row.get(field):
                    examples[-1][field] = row[field] if field == "split_protocol" else ast.literal_eval(row[field])
            if max_examples and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError("No examples in evaluation CSV")
    if len({row["example_id"] for row in examples}) != len(examples):
        raise ValueError("Evaluation CSV example IDs are not unique")
    return examples


def decode_batch(model, tokenizer, grammar, prompts, num_beams=50, top_k=20, score_batch_size=1):
    import torch
    from transformers import GenerationConfig

    if num_beams < 2 or top_k < 1 or top_k > num_beams or score_batch_size < 1:
        raise ValueError("Require num_beams >= 2, 1 <= top_k <= num_beams and positive score_batch_size")
    width = max(map(len, prompts))
    pad = tokenizer.eos_token_id
    device = next(model.parameters()).device
    input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
    attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    context = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", None))
    if context and width + grammar.max_new_tokens > context:
        raise ValueError(f"Prompt plus target exceeds model context ({width} + {grammar.max_new_tokens} > {context}); no silent truncation")

    config = GenerationConfig(
        do_sample=False, num_beams=num_beams, num_return_sequences=num_beams,
        max_new_tokens=grammar.max_new_tokens, length_penalty=0.0,
        early_stopping=False, renormalize_logits=False, repetition_penalty=1.0,
        eos_token_id=pad, pad_token_id=pad, bos_token_id=tokenizer.bos_token_id,
        forced_bos_token_id=None, forced_eos_token_id=None,
    )
    # SFT checkpoints inherit Qwen2.5-Instruct's generation_config.json (do_sample,
    # T=0.7, top-k 20, top-p 0.8, repetition penalty 1.1). transformers >= 4.50
    # refills every value above that equals a library default (do_sample=False,
    # repetition_penalty=1.0) from it unless use_model_defaults=False; the search
    # then samples, penalizes history SIDs, and can draw tokens outside the grammar.
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=config, use_model_defaults=False,
            prefix_allowed_tokens_fn=lambda batch_id, ids: grammar.allowed(ids[width:].tolist()),
        )
    candidates = []
    by_example = [[] for _ in prompts]
    for example, prompt in enumerate(prompts):
        seen = set()
        for beam in output[example * num_beams:(example + 1) * num_beams]:
            sids, target_tokens = grammar.parse(beam[width:].tolist())
            key = tuple(sids)
            if key in seen:
                continue
            seen.add(key)
            candidate = {"sids": sids, "generated_token_ids": target_tokens}
            candidates.append((prompt, candidate))
            by_example[example].append(candidate)
    # Teacher-forced score every returned candidate under the raw vocabulary.
    # This also counts mandatory separator/newline/EOS likelihood and avoids
    # conflating legal-set renormalization with the model's joint probability.
    for start in range(0, len(candidates), score_batch_size):
        part = candidates[start:start + score_batch_size]
        sequences = [prompt + candidate["generated_token_ids"] for prompt, candidate in part]
        size = max(map(len, sequences))
        ids = torch.tensor([seq + [pad] * (size - len(seq)) for seq in sequences], device=device)
        mask = torch.tensor([[1] * len(seq) + [0] * (size - len(seq)) for seq in sequences], device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
        for index, (prompt, candidate) in enumerate(part):
            count = len(candidate["generated_token_ids"])
            positions = logits[index, len(prompt) - 1:len(prompt) - 1 + count].float()
            labels = torch.tensor(candidate["generated_token_ids"], device=device)
            values = positions.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1).cpu().tolist()
            candidate["raw_vocabulary_token_logprobs"] = values
            candidate["joint_logprob"] = float(sum(values))
            if not math.isfinite(candidate["joint_logprob"]):
                raise ValueError("Non-finite model score during joint evaluation")
        del logits
    return [sorted(candidates, key=lambda c: (-c["joint_logprob"], c["sids"]))[:top_k]
            for candidates in by_example]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--index-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--predtype", choices=("next-item", "next2item"), required=True)
    parser.add_argument("--num-beams", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0, help="0 evaluates every CSV row")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_examples < 0 or min(args.batch_size, args.score_batch_size, args.top_k, *args.ks) < 1:
        raise ValueError("Example cap must be nonnegative; batch sizes, top-k and ks must be positive")
    if args.num_beams < 2 or args.top_k > args.num_beams or max(args.ks) > args.top_k:
        raise ValueError("Require 2 <= num_beams, max(ks) <= top_k <= num_beams")
    if not args.checkpoint.is_dir():
        raise ValueError("--checkpoint must be a local model directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty; historical outputs are never overwritten")
    if args.output_dir.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError("Output directory must be outside the checkpoint directory")
    groups = load_catalog(args.index_file)
    examples = load_examples(args.test_file, groups, 2 if args.predtype == "next2item" else 1, args.max_examples)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    dtype = (torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32) if args.dtype == "auto" else getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    grammar = CatalogGrammar(tokenizer, groups, 2 if args.predtype == "next2item" else 1)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype, local_files_only=True).to(device).eval()
    protocol = {
        "schema_version": "onediffrec.corrected_ar/v1",
        "decoder": "joint_ordered_sid_constrained_beam",
        "generation": "plain beam search (do_sample=False, repetition_penalty=1.0, length_penalty=0.0); "
                      "checkpoint generation_config.json ignored via use_model_defaults=False",
        "prompt": "SidSFTDataset exact instruction/history prompt",
        "target_grammar": "SID1 ||| SID2\\n + EOS" if grammar.horizon == 2 else "SID\\n + EOS",
        "score": "raw full-vocabulary token logprob sum including separator, newline and EOS; no length normalization",
        "search": "approximate beam search over catalog-valid ordered tuples; not exhaustive top-K",
        "metric_schema": "ordered_targets_v1",
        "cce_protocol": "paper-cce-v1" if grammar.horizon == 1 else None,
        "cce_status": "single-target CCE included" if grammar.horizon == 1 else "not defined for ordered pairs",
        "split_status": "Uses supplied CSV unchanged; corrected decoding does not repair training/validation leakage",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "resolved_device": device, "resolved_dtype": str(dtype),
        "inputs": {"checkpoint": fingerprint_path(args.checkpoint), "test_file": fingerprint_path(args.test_file),
                   "index_file": fingerprint_path(args.index_file)},
        "source": {path.name: fingerprint_path(path) for path in (
            Path(__file__), ROOT / "metrics/ordered_targets.py", ROOT / "metrics/current_cce.py",
            ROOT / "metrics/collision.py", ROOT / "scripts/ar/data.py")},
    }
    # Hash reproducible content, not machine-specific paths or output destination.
    identity = {key: value for key, value in protocol.items() if key not in ("inputs", "source", "arguments")}
    identity["inputs"] = {key: value["sha256"] for key, value in protocol["inputs"].items()}
    identity["source"] = {key: value["sha256"] for key, value in protocol["source"].items()}
    identity["arguments"] = {key: value for key, value in protocol["arguments"].items()
                             if key not in ("checkpoint", "test_file", "index_file", "output_dir")}
    protocol_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    protocol["protocol_hash"] = protocol_hash
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with (args.output_dir / "predictions.jsonl").open("x") as handle:
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            predictions = decode_batch(model, tokenizer, grammar,
                [encode_prompt(tokenizer, row["history_sids"]) for row in batch],
                args.num_beams, args.top_k, args.score_batch_size)
            for row, ranked in zip(batch, predictions):
                row["predictions"] = [dict(prediction, candidate_item_ids=[groups[sid] for sid in prediction["sids"]])
                                      for prediction in ranked]
                row["protocol_hash"] = protocol_hash
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"Evaluated {min(start + args.batch_size, len(examples))}/{len(examples)} examples", flush=True)
    metrics = ordered_target_metrics(examples, args.ks)
    if grammar.horizon == 1:
        metrics["cce"] = current_cce_metrics(examples,
            {item: sid for sid, items in groups.items() for item in items},
            args.ks, retained_top_k=args.top_k)
    metrics["protocol_hash"] = protocol_hash
    metrics["prediction_fingerprint"] = fingerprint_path(args.output_dir / "predictions.jsonl")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
