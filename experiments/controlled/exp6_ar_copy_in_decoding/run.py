#!/usr/bin/env python
"""Copy-link knockout inside the archived AR decoder (trie-constrained beam search, 50 beams).

Protocol: `protocol.md` beside this file (declared before any decoding run).
One invocation = one cell; conditions run one after another on identical
batches, all through the same hooked `generate()` path
(`sidlens.interventions.generation`).

Why everything here copies `evaluate.py` instead of re-implementing it
--------------------------------------------------------------------
The baseline must reproduce the archived predictions before any knockout
counts, and bf16 beam search is sensitive to batch composition, padding and
the generation code path. So the trie is built with `evaluate.py`'s own
algorithm from the frozen info file, the vendored `ConstrainedLogitsProcessor`
is imported from `vendor/` unmodified, `generate()` gets the same
GenerationConfig, and rows are batched exactly as the archive was
(`split.py`'s 4 contiguous shards, batches of 8). The model is the frozen
checkpoint in bf16.

    python run.py --cell 0 --out <new dir> [--limit 64 --check-plain]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.data.sids import SidTable                                   # noqa: E402
from sidlens.interventions.generation import generation_knockout         # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
CONDITIONS = ("B", "C_later", "C_all", "N_later", "B_plain", "C_all_plain")
KO_LAYERS = tuple(range(14, 28))
NUM_BEAMS, MAX_NEW, LENGTH_PENALTY, EVAL_BATCH, N_SHARDS = 50, 256, 0.0, 8, 4
PREFIX_INDEX = 3
SEED = 42                        # evaluate.py's default; set at the start of every shard


def checkpoint_sha(ckpt: str) -> str:
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def vendored_processor():
    """`vendor/onediffrec/LogitProcessor.py`, imported by path, unmodified."""
    spec = importlib.util.spec_from_file_location("_sidlens_vendor_logitprocessor",
                                                  paths.VENDOR / "onediffrec" / "LogitProcessor.py")
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod.ConstrainedLogitsProcessor


def build_trie(tokenizer, info_file: Path) -> dict:
    """`evaluate.py`'s hash_dict, line for line (Qwen: prefix_index 3, no BOS strip)."""
    get_hash = lambda x: "-".join(str(_) for _ in x)                      # noqa: E731
    lines = info_file.read_text().splitlines(keepends=True)
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in lines]
    info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]
    prefix_ids = [tokenizer(_).input_ids for _ in info_semantic]
    hash_dict: dict = {}
    for ID in prefix_ids:
        ID.append(tokenizer.eos_token_id)
        for i in range(PREFIX_INDEX, len(ID)):
            h = get_hash(ID[:i]) if i == PREFIX_INDEX else get_hash(ID[PREFIX_INDEX:i])
            hash_dict.setdefault(h, set()).add(ID[i])
    return {k: list(v) for k, v in hash_dict.items()}


def batches(n_rows: int) -> list[tuple[int, list[int]]]:
    """(shard, rows): split.py's contiguous shards, then evaluate.py's batches of 8."""
    out = []
    for i in range(N_SHARDS):
        s, e = i * n_rows // N_SHARDS, (i + 1) * n_rows // N_SHARDS
        out += [(i, list(range(b, min(e, b + EVAL_BATCH)))) for b in range(s, e, EVAL_BATCH)]
    return out


def set_seed(seed: int) -> None:
    """evaluate.py's set_seed, which each shard process ran once at start-up."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Plan:
    """Edges to remove at each generation step, from each beam's own prefix."""

    def __init__(self, condition, encs, vocab, max_len, n_digits):
        # *_plain conditions knock out the same edges under the plain decoder
        self.condition = condition.removesuffix("_plain")
        self.vocab, self.max_len, self.n = vocab, max_len, n_digits
        self.hist = []
        for e in encs:
            pad = max_len - len(e)
            k = len(e.example.history_sids)
            codes = [P.parse_sid(s, n_digits) for s in e.example.history_sids]
            tok = {(j, d): pad + e.positions("hist_sid", item=j, digit=d)[0]
                   for j in range(k) for d in range(n_digits)}
            self.hist.append((codes, tok))
        self.rows_with_edges: dict[int, int] = {}
        self.rows_seen: dict[int, int] = {}
        self.rows_invalid_prefix: dict[int, int] = {}

    def __call__(self, seqs, step, q, kv):
        if self.condition == "B" or step >= self.n:
            return []
        gen = seqs[:, self.max_len:self.max_len + step].tolist()
        edges = []
        for row in range(seqs.shape[0]):
            codes, tok = self.hist[row // NUM_BEAMS]
            items = range(len(codes))
            if step == 0:
                if q != self.max_len:
                    raise RuntimeError(f"step 0 must be the prefill ({q} != {self.max_len})")
                qi = self.max_len - 1
                keys = [tok[(j, 0)] for j in items] if self.condition == "C_all" else []
            else:
                if q != 1:
                    raise RuntimeError(f"step {step} expected one query, got {q}")
                qi, d = 0, step
                dc = [self.vocab.digit_code_by_id.get(t) for t in gen[row]]
                if any(x is None for x in dc) or [x[0] for x in dc] != list(range(d)):
                    # A garbage beam from beam sampling (protocol amendment): it
                    # has no SID prefix, so there is nothing to match. Counted.
                    self.rows_invalid_prefix[step] = self.rows_invalid_prefix.get(step, 0) + 1
                    self.rows_seen[step] = self.rows_seen.get(step, 0) + 1
                    continue
                prefix = tuple(x[1] for x in dc)
                match = [j for j in items if tuple(codes[j][:d]) == prefix]
                if self.condition in ("C_later", "C_all"):
                    keys = [tok[(j, d)] for j in match]
                else:                                                   # N_later
                    non = [j for j in reversed(items) if j not in match]
                    keys = [tok[(j, d)] for j in non[:len(match)]]
            self.rows_seen[step] = self.rows_seen.get(step, 0) + 1
            if keys:
                self.rows_with_edges[step] = self.rows_with_edges.get(step, 0) + 1
                edges.append((row, qi, keys))
        return edges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)))
    ap.add_argument("--ckpt", choices=CELLS)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None, help="first N rows of shard 0 (pilot)")
    ap.add_argument("--check-plain", action="store_true", help="V2: also run unhooked generate()")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    ckpt = args.ckpt or CELLS[args.cell]
    conditions = args.conditions.split(",")
    if not set(conditions) <= set(CONDITIONS):
        ap.error(f"--conditions is a subset of {CONDITIONS}")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    logf = open(args.out / "run.log", "a")
    t_start = time.time()

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    ckpt_dir = paths.FROZEN_CKPT / "ar" / ckpt
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    examples = P.load_examples(vocab.variant, "next-item", "test")
    table = SidTable.load(vocab.variant)
    validation = {"table": P.check_against_table(examples, table), "archive_rows": P.check_against_archive(examples)}
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:2000]}")
    encs = [P.encode(ex, tok, vocab, template="eval", with_target=False) for ex in examples]
    info_file = paths.FROZEN_SIDS / "info" / "next-item" / f"{P.PRIMARY_CATEGORY}_{vocab.variant.name}.txt"
    trie = build_trie(tok, info_file)
    Constrained = vendored_processor()
    plan_batches = batches(len(examples))
    if args.limit:
        plan_batches = [(sh, b) for sh, b in plan_batches if b[-1] < args.limit]
    sha = checkpoint_sha(ckpt)
    arch = P.archive_path(vocab.variant, "next-item")
    archived = json.loads(arch.read_text())
    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": sha, "variant": vocab.variant.name,
              "n_digits": n, "split": "test", "template": "eval", "n_rows": len(examples),
              "limited": args.limit, "n_batches": len(plan_batches), "conditions": conditions,
              "ko_layers": list(KO_LAYERS), "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW,
              "length_penalty": LENGTH_PENALTY, "eval_batch": EVAL_BATCH, "n_shards": N_SHARDS,
              "info_file": str(info_file.relative_to(paths.WORK)), "info_sha256": hashing.sha256_file(info_file),
              "archive": str(arch.relative_to(paths.WORK)), "archive_sha256": hashing.sha256_file(arch),
              "csv_sha256": hashing.sha256_file(P.csv_path(vocab.variant, "next-item", "test")),
              "vendored_processor_sha256": hashing.sha256_file(paths.VENDOR / "onediffrec" / "LogitProcessor.py"),
              "trie_keys": len(trie), "seed_per_shard": SEED,
              "checkpoint_generation_config": json.loads((ckpt_dir / "generation_config.json").read_text()),
              "note": "generate() merges the checkpoint generation_config defaults (do_sample, temperature, "
                      "top_k, top_p, repetition_penalty) into evaluate.py's GenerationConfig; *_plain "
                      "conditions override do_sample=False, repetition_penalty=1.0"}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"inputs ok: {ckpt} rows={len(examples)} batches={len(plan_batches)} trie={len(trie)}")

    # evaluate.py: bf16 weights, pad = eos, left padding, config ids set on the model
    model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.bfloat16,
                                                 local_files_only=True).to("cuda").eval()
    tok.pad_token, tok.pad_token_id, tok.padding_side = tok.eos_token, tok.eos_token_id, "left"
    model.config.pad_token_id = model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = tok.bos_token_id
    inputs.update({"device": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "transformers": __import__("transformers").__version__,
                   "attn_implementation": model.config._attn_implementation})
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))

    def allowed(batch_id, input_ids):
        return trie.get("-".join(str(_) for _ in input_ids), [])

    def decode(rows, condition, hooked=True):
        ids = [encs[i].input_ids for i in rows]
        max_len = max(len(x) for x in ids)
        inp = torch.tensor([[tok.pad_token_id] * (max_len - len(x)) + x for x in ids], device="cuda")
        am = torch.tensor([[0] * (max_len - len(x)) + [1] * len(x) for x in ids], device="cuda")
        extra = {"do_sample": False, "repetition_penalty": 1.0} if condition.endswith("_plain") else {}
        gc = GenerationConfig(num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY, num_return_sequences=NUM_BEAMS,
                              pad_token_id=model.config.pad_token_id, eos_token_id=model.config.eos_token_id,
                              max_new_tokens=MAX_NEW, **extra)
        clp = Constrained(prefix_allowed_tokens_fn=allowed, num_beams=NUM_BEAMS, base_model=str(ckpt_dir))
        kw = dict(attention_mask=am, generation_config=gc, return_dict_in_generate=True, output_scores=True,
                  logits_processor=LogitsProcessorList([clp]))
        if condition.endswith("_plain"):
            # transformers >= 4.50 refills every field left at its LIBRARY default from the
            # checkpoint's generation_config (use_model_defaults=None). do_sample=False and
            # repetition_penalty=1.0 ARE the library defaults, so without this they were
            # silently replaced by the Qwen sampling defaults (protocol amendment 2).
            kw["use_model_defaults"] = False
        plan = Plan(condition, [encs[i] for i in rows], vocab, max_len, n)
        stats = None
        with torch.no_grad():
            if hooked:
                with generation_knockout(model, KO_LAYERS, plan, prompt_len=max_len) as stats:
                    out = model.generate(inp, **kw)
            else:
                out = model.generate(inp, **kw)
        comp = tok.batch_decode(out.sequences[:, max_len:], skip_special_tokens=True)
        comp = [c.split("Response:\n")[-1].strip() for c in comp]
        return ([comp[k * NUM_BEAMS:(k + 1) * NUM_BEAMS] for k in range(len(rows))], out.sequences.cpu(),
                out.sequences_scores.float().cpu(), plan, stats)

    recs, edge_recs, v2, v5 = [], [], {}, {}
    for cond in conditions:
        t0 = time.time()
        shard = None
        for bi, (sh, rows) in enumerate(plan_batches):
            if sh != shard:
                set_seed(SEED)
                shard = sh
            if args.check_plain and cond == "B":
                # V2: hooked vs unhooked from the same RNG state; V5: another seed
                state = (torch.get_rng_state(), torch.cuda.get_rng_state())
                p2, s2, sc2, _, _ = decode(rows, cond, hooked=False)
                torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])
                preds, seqs, scores, plan, stats = decode(rows, cond)
                v2["batches"] = v2.get("batches", 0) + 1
                v2["identical_sequences"] = v2.get("identical_sequences", True) and torch.equal(seqs, s2)
                v2["identical_scores"] = v2.get("identical_scores", True) and torch.equal(scores, sc2)
                state = (torch.get_rng_state(), torch.cuda.get_rng_state())
                torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
                p3, _, _, _, _ = decode(rows, cond)
                torch.set_rng_state(state[0]); torch.cuda.set_rng_state(state[1])
                v5["rows"] = v5.get("rows", 0) + len(rows)
                v5["identical_top10"] = v5.get("identical_top10", 0) + sum(a[:10] == b[:10] for a, b in zip(preds, p3))
                v5["identical_50"] = v5.get("identical_50", 0) + sum(a == b for a, b in zip(preds, p3))
            else:
                preds, seqs, scores, plan, stats = decode(rows, cond)
            for k, i in enumerate(rows):
                ex = examples[i]
                recs.append({"example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id,
                             "condition": cond, "batch": bi, "predict": preds[k],
                             "scores": scores[k * NUM_BEAMS:(k + 1) * NUM_BEAMS].tolist()})
            edge_recs.append({"condition": cond, "batch": bi, "shard": sh, "rows": len(rows),
                              **{f"beam_rows_invalid_prefix_step{s}": v for s, v in plan.rows_invalid_prefix.items()},
                              "forwards": stats["forwards"], "materialized": stats["materialized"],
                              "edges": stats["edges"], **{f"edges_step{s}": v for s, v in stats["edges_by_step"].items()},
                              **{f"beam_rows_with_edges_step{s}": v for s, v in plan.rows_with_edges.items()},
                              **{f"beam_rows_step{s}": v for s, v in plan.rows_seen.items()}})
            if (bi + 1) % 50 == 0:
                log(f"[{cond}] {bi + 1}/{len(plan_batches)} batches, {time.time() - t0:.0f}s")
        log(f"[{cond}] done, {time.time() - t0:.0f}s")
        pd.DataFrame([r for r in recs if r["condition"] == cond]).to_parquet(
            args.out / f"predictions_{cond}.parquet", index=False)

    # ---- V1: the hooked baseline against the archive --------------------------
    base = {r["row"]: r["predict"] for r in recs if r["condition"] == "B"}
    if base:
        rows_done = sorted(base)
        arch_l = {i: [s.strip("\"\n").strip() for s in archived[i]["predict"]] for i in rows_done}
        same50 = [base[i] == arch_l[i] for i in rows_done]
        same10 = [base[i][:10] == arch_l[i][:10] for i in rows_done]
        validation["V1_reproduction"] = {"rows": len(rows_done), "identical_50": float(np.mean(same50)),
                                         "identical_top10": float(np.mean(same10)),
                                         "differing_rows": [i for i, s in zip(rows_done, same50) if not s][:50]}
    if v2:
        validation["V2_hook_neutrality"] = v2
    if v5:
        validation["V5_determinism_seed1234"] = v5
    ed = pd.DataFrame(edge_recs)
    ed.to_csv(args.out / "knockout_edges.csv", index=False)
    validation["V3_edges"] = {c: {k: int(v) for k, v in g.drop(columns=["condition"]).sum(numeric_only=True).items()}
                              for c, g in ed.groupby("condition")}
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: V1 {validation.get('V1_reproduction', {}).get('identical_50')} identical lists; "
        f"V2 {v2}; V5 {v5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
