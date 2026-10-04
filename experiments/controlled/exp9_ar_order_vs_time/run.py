#!/usr/bin/env python
"""Order vs time: swap the two most recent history items; stratify by their days.

Protocol: `protocol.md` beside this file, declared before any forward. One
invocation = one cell (checkpoint) x one split.

The strata (U, T_later, T_same) come from `sidlens.data.timestamps` and are
written to `rows.parquet` before the model is loaded. The teacher-forced part
runs through `FixedShapeRunner`. There, rows whose two most recent items have
the same SID are swapped anyway: the input is token-identical, so every code
logit must be bit-equal to clean. That is the batch-slot invariance that lets a
clean-SWAP difference be the swap's alone. Part D (test only) is exp6's plain
beam search on clean and swapped prompts, in exp6's batches. Its clean arm must
reproduce exp6's `predictions_B_plain` lists.

    python run.py --cell 0 --split test --out <new dir> [--limit 64] [--parts TF,D]
"""

from __future__ import annotations

import argparse
import dataclasses
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
from sidlens.data import timestamps as T                                 # noqa: E402
from sidlens.data.sids import SidTable                                   # noqa: E402
from sidlens.interventions.runner import FixedShapeRunner, Job, max_diff, records  # noqa: E402
from sidlens.interventions.scoring import DigitScorer                    # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
EXP6_PLAIN = {"next-item_best": "283586/cell-00", "oneoff_rqvae4cb128": "283621/cell-01"}
SEED = 20260930
# Part D: exp6's plain decoder, constants copied so this run's archived source is complete.
NUM_BEAMS, MAX_NEW, LENGTH_PENALTY, EVAL_BATCH, N_SHARDS, PREFIX_INDEX, DECODE_SEED = 50, 256, 0.0, 8, 4, 3, 42


def checkpoint_sha(ckpt: str) -> str:
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def swap_last_two(ex: P.ArExample) -> P.ArExample:
    """`ex` with history items L-1 and L-2 exchanged; the example id names the clean row."""
    if len(ex.history_item_ids) < 2:
        raise ValueError(f"{ex.example_id}: fewer than two history items")
    ids, sids = list(ex.history_item_ids), list(ex.history_sids)
    ids[-1], ids[-2] = ids[-2], ids[-1]
    sids[-1], sids[-2] = sids[-2], sids[-1]
    return dataclasses.replace(ex, history_item_ids=tuple(ids), history_sids=tuple(sids))


def check_swap(clean: P.Encoded, new: P.Encoded) -> list[int]:
    """Prove `new` differs from `clean` only inside the last two history items."""
    if len(clean) != len(new) or clean.prompt_len != new.prompt_len:
        raise ValueError(f"{clean.example.example_id}: swapped length differs")
    if (clean.role, clean.item, clean.digit) != (new.role, new.item, new.digit):
        raise ValueError(f"{clean.example.example_id}: swapped position map differs")
    L = len(clean.example.history_sids)
    allowed = set(clean.positions("hist_sid", item=L - 1)) | set(clean.positions("hist_sid", item=L - 2))
    changed = [i for i, (a, b) in enumerate(zip(clean.input_ids, new.input_ids)) if a != b]
    if stray := sorted(set(changed) - allowed):
        raise ValueError(f"{clean.example.example_id}: positions {stray[:5]} changed outside the pair")
    return changed


def strata(rows: pd.DataFrame, hist: pd.DataFrame, examples, n: int) -> pd.DataFrame:
    """Per-row stratum, eligibility and the digit-0 codes of r1, r2 and the target."""
    t = hist.set_index(["example_id", "recency"]).t
    out = []
    for ex, r in zip(examples, rows.itertuples()):
        L = len(ex.history_sids)
        rec = {"example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id, "hist_len": L,
               "gap1_days": r.gap1_days, "gap12_days": r.gap12_days, "dup_target": r.dup_target,
               "stratum": None, "eligible": False, "control": False, "c1": -1, "c2": -1,
               "ct": P.parse_sid(ex.target_sids[0], n)[0]}
        if L >= 2:
            t1, t2 = t[(ex.example_id, 1)], t[(ex.example_id, 2)]
            rec["stratum"] = "U" if t1 > t2 else ("T_same" if t1 == r.t_target else "T_later")
            if t1 < t2:
                raise ValueError(f"{ex.example_id}: most recent item earlier than the one before")
            same = ex.history_sids[-1] == ex.history_sids[-2]
            rec["eligible"], rec["control"] = not same, same
            rec["c1"] = P.parse_sid(ex.history_sids[-1], n)[0]
            rec["c2"] = P.parse_sid(ex.history_sids[-2], n)[0]
        out.append(rec)
    return pd.DataFrame(out)


# ---- Part D helpers: exp6's, line for line ---------------------------------
def vendored_processor():
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
    get_hash = lambda x: "-".join(str(_) for _ in x)                      # noqa: E731
    lines = info_file.read_text().splitlines(keepends=True)
    semantic_ids = [line.split("\t")[0].strip() + "\n" for line in lines]
    prefix_ids = [tokenizer(f"### Response:\n{_}").input_ids for _ in semantic_ids]
    hash_dict: dict = {}
    for ID in prefix_ids:
        ID.append(tokenizer.eos_token_id)
        for i in range(PREFIX_INDEX, len(ID)):
            h = get_hash(ID[:i]) if i == PREFIX_INDEX else get_hash(ID[PREFIX_INDEX:i])
            hash_dict.setdefault(h, set()).add(ID[i])
    return {k: list(v) for k, v in hash_dict.items()}


def batches(n_rows: int) -> list[tuple[int, list[int]]]:
    out = []
    for i in range(N_SHARDS):
        s, e = i * n_rows // N_SHARDS, (i + 1) * n_rows // N_SHARDS
        out += [(i, list(range(b, min(e, b + EVAL_BATCH)))) for b in range(s, e, EVAL_BATCH)]
    return out


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)))
    ap.add_argument("--ckpt", choices=CELLS)
    ap.add_argument("--split", default="test", choices=("valid", "test"))
    ap.add_argument("--parts", default="TF,D")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    ckpt = args.ckpt or CELLS[args.cell]
    parts = set(args.parts.split(","))
    if not parts or not parts <= {"TF", "D"}:
        ap.error("--parts is a subset of TF,D")
    if "D" in parts and args.split != "test":
        ap.error("Part D is declared for the test split only")
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

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ---- inputs and strata, fixed before the model is loaded ----------------
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    from transformers import AutoTokenizer
    ckpt_dir = paths.FROZEN_CKPT / "ar" / ckpt
    tok = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    examples = P.load_examples(vocab.variant, "next-item", args.split)
    table = SidTable.load(vocab.variant)
    validation = {"table": P.check_against_table(examples, table),
                  "vocab_vs_table": ar.validate_against(vocab, table)}
    if args.split == "test":
        validation["archive"] = P.check_against_archive(examples)
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:2000]}")
    ev = T.load_event_times()
    trows, thist = T.ar_row_times(examples, ev)
    st = strata(trows, thist, examples, n)
    n_full = len(examples)
    if args.limit:
        examples, st = examples[:args.limit], st.iloc[:args.limit].reset_index(drop=True)
    st.to_parquet(args.out / "rows.parquet", index=False)
    clean = [P.encode(ex, tok, vocab, template="eval", with_target=True) for ex in examples]
    t_pad = -(-max(len(e) for e in clean) // 8) * 8
    sha = checkpoint_sha(ckpt)
    csv = P.csv_path(vocab.variant, "next-item", args.split)
    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": sha, "variant": vocab.variant.name,
              "n_digits": n, "split": args.split, "template": "eval",
              "csv": str(csv.relative_to(paths.WORK)), "csv_sha256": hashing.sha256_file(csv),
              "sem_ids_sha256": hashing.sha256_file(table.source), "timestamps": ev.report,
              "n_rows": len(examples), "n_users": len({e.user_id for e in examples}),
              "limited": args.limit, "t_pad": t_pad, "batch_rows": args.rows, "seed": SEED,
              "parts": sorted(parts),
              "strata": {str(k): int(v) for k, v in st[st.eligible].stratum.value_counts().items()},
              "strata_distinct_d0": {str(k): int(v) for k, v in
                                     st[st.eligible & (st.c1 != st.c2)].stratum.value_counts().items()},
              "n_controls": int(st.control.sum())}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"inputs ok: {ckpt} {args.split} rows={len(examples)} strata={inputs['strata']} "
        f"controls={inputs['n_controls']}")

    # ---- Part TF: teacher-forced 2x2 ----------------------------------------
    if "TF" in parts:
        model = ar.load_model(ckpt, device="cuda", dtype="bfloat16")
        inputs.update({"attn_implementation": model.config._attn_implementation,
                       "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "transformers": __import__("transformers").__version__})
        (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
        runner = FixedShapeRunner(model, DigitScorer(vocab, table), tok.pad_token_id, t_pad, args.rows, 1)
        t0 = time.time()
        jobs, recs, logits = [], [], {}

        def flush():
            if not jobs:
                return
            s, cl, _, _ = runner.forward(jobs)
            recs.extend(records(runner, jobs, s, runner.n_forwards))
            for r, job in enumerate(jobs):
                if job.meta["i"] in need_logits:
                    logits[(job.meta["i"], job.meta["condition"])] = [c[r] for c in cl]
            jobs.clear()

        need_logits = set(np.flatnonzero(st.control.to_numpy()).tolist())
        n_changed = []
        # clean rows first, then all swaps: a control's two copies land in different batch slots
        for cond in ("clean", "swap"):
            for i, ex in enumerate(examples):
                if cond == "swap" and not (st.eligible[i] or st.control[i]):
                    continue
                if cond == "clean" and not (st.eligible[i] or st.control[i]):
                    continue
                if cond == "clean":
                    e = clean[i]
                else:
                    e = P.encode(swap_last_two(ex), tok, vocab, template="eval", with_target=True)
                    ch = check_swap(clean[i], e)
                    if st.control[i] and ch:
                        raise AssertionError(f"{ex.example_id}: identical-SID swap changed tokens")
                    if st.eligible[i] and not ch:
                        raise AssertionError(f"{ex.example_id}: eligible swap changed nothing")
                    n_changed.append(len(ch))
                jobs.append(Job(e, meta={"i": i, "condition": cond, "stratum": st.stratum[i],
                                         "control": bool(st.control[i])}))
                if len(jobs) >= runner.B:
                    flush()
            flush()
        tf = pd.DataFrame(recs)
        tf.to_parquet(args.out / "part_tf.parquet", index=False)
        ctrl = [{"example_id": examples[i].example_id,
                 "max_abs_diff_code_logits": max_diff(logits[(i, "swap")], logits[(i, "clean")])}
                for i in sorted(need_logits)]
        c = pd.DataFrame(ctrl, columns=["example_id", "max_abs_diff_code_logits"])
        c["exact"] = c.max_abs_diff_code_logits == 0.0
        c.to_parquet(args.out / "controls.parquet", index=False)
        validation["controls_identical_sid_swap"] = {
            "n": int(len(c)), "n_exact": int(c.exact.sum()),
            "max_abs_diff_code_logits": float(c.max_abs_diff_code_logits.max()) if len(c) else None}
        validation["swap_changed_positions"] = {"min": int(min(n_changed)), "max": int(max(n_changed))}
        validation["n_forwards"] = runner.n_forwards
        log(f"[TF] {len(tf)} records, {runner.n_forwards} forwards, {time.time() - t0:.0f}s; "
            f"controls {validation['controls_identical_sid_swap']}")
        del model
        torch.cuda.empty_cache()

    # ---- Part D: plain beam search on clean and swapped prompts --------------
    if "D" in parts:
        from transformers import AutoModelForCausalLM, GenerationConfig, LogitsProcessorList
        info_file = paths.FROZEN_SIDS / "info" / "next-item" / f"{P.PRIMARY_CATEGORY}_{vocab.variant.name}.txt"
        trie = build_trie(tok, info_file)
        Constrained = vendored_processor()
        model = AutoModelForCausalLM.from_pretrained(str(ckpt_dir), torch_dtype=torch.bfloat16,
                                                     local_files_only=True).to("cuda").eval()
        tok.pad_token, tok.pad_token_id, tok.padding_side = tok.eos_token, tok.eos_token_id, "left"
        model.config.pad_token_id = model.config.eos_token_id = tok.eos_token_id
        model.config.bos_token_id = tok.bos_token_id
        # exp6 ran generate() with torch's default numerics; match them so V-D0 can be exact.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.allow_tf32 = True
        enc_clean = [P.encode(ex, tok, vocab, template="eval", with_target=False) for ex in examples]
        enc_swap = [P.encode(swap_last_two(ex), tok, vocab, template="eval", with_target=False)
                    if st.eligible[i] else enc_clean[i] for i, ex in enumerate(examples)]
        plan = batches(n_full)             # exp6's batches over the whole test split
        if args.limit:
            plan = [(sh, b) for sh, b in plan if b[-1] < args.limit]

        def allowed(batch_id, input_ids):
            return trie.get("-".join(str(_) for _ in input_ids), [])

        def decode(encs):
            ids = [e.input_ids for e in encs]
            max_len = max(len(x) for x in ids)
            inp = torch.tensor([[tok.pad_token_id] * (max_len - len(x)) + x for x in ids], device="cuda")
            am = torch.tensor([[0] * (max_len - len(x)) + [1] * len(x) for x in ids], device="cuda")
            gc = GenerationConfig(num_beams=NUM_BEAMS, length_penalty=LENGTH_PENALTY,
                                  num_return_sequences=NUM_BEAMS, pad_token_id=model.config.pad_token_id,
                                  eos_token_id=model.config.eos_token_id, max_new_tokens=MAX_NEW,
                                  do_sample=False, repetition_penalty=1.0)
            clp = Constrained(prefix_allowed_tokens_fn=allowed, num_beams=NUM_BEAMS, base_model=str(ckpt_dir))
            with torch.no_grad():
                out = model.generate(inp, attention_mask=am, generation_config=gc, return_dict_in_generate=True,
                                     output_scores=True, logits_processor=LogitsProcessorList([clp]),
                                     use_model_defaults=False)
            comp = tok.batch_decode(out.sequences[:, max_len:], skip_special_tokens=True)
            comp = [c_.split("Response:\n")[-1].strip() for c_ in comp]
            sc = out.sequences_scores.float().cpu()
            return ([comp[k * NUM_BEAMS:(k + 1) * NUM_BEAMS] for k in range(len(encs))],
                    [sc[k * NUM_BEAMS:(k + 1) * NUM_BEAMS].tolist() for k in range(len(encs))])

        drecs = []
        for cond, encs in (("clean_plain", enc_clean), ("swap_plain", enc_swap)):
            t0 = time.time()
            shard = None
            for bi, (sh, rows_) in enumerate(plan):
                if sh != shard:
                    set_seed(DECODE_SEED)
                    shard = sh
                if cond == "swap_plain" and not any(st.eligible[i] for i in rows_):
                    continue                    # identical batch: reuse the clean lists below
                preds, scores = decode([encs[i] for i in rows_])
                for k, i in enumerate(rows_):
                    drecs.append({"example_id": examples[i].example_id, "row": examples[i].row,
                                  "user_id": examples[i].user_id, "condition": cond, "batch": bi,
                                  "swapped": bool(st.eligible[i]) if cond == "swap_plain" else False,
                                  "predict": preds[k], "scores": scores[k]})
                if (bi + 1) % 50 == 0:
                    log(f"[D {cond}] {bi + 1}/{len(plan)} batches, {time.time() - t0:.0f}s")
            log(f"[D {cond}] done, {time.time() - t0:.0f}s")
        d = pd.DataFrame(drecs)
        # batches with no eligible row are token-identical: their swap lists are the clean lists
        done = set(d[d.condition == "swap_plain"].row)
        fill = d[(d.condition == "clean_plain") & ~d.row.isin(done)].copy()
        fill["condition"], fill["swapped"] = "swap_plain", False
        d = pd.concat([d, fill], ignore_index=True).sort_values(["condition", "row"])
        d.to_parquet(args.out / "part_d.parquet", index=False)
        ref = pd.read_parquet(paths.DERIVED / "controlled" / "exp6_ar_copy_in_decoding" / EXP6_PLAIN[ckpt]
                              / "predictions_B_plain.parquet").set_index("row").predict
        mine = d[d.condition == "clean_plain"].set_index("row").predict
        same = [list(mine[r]) == list(ref[r]) for r in mine.index]
        same10 = [list(mine[r])[:10] == list(ref[r])[:10] for r in mine.index]
        validation["V_D0_exp6_plain_reproduction"] = {
            "rows": len(same), "identical_50": float(np.mean(same)), "identical_top10": float(np.mean(same10)),
            "ok": bool(np.mean(same) >= 0.99),
            "differing_rows": [int(r) for r, s in zip(mine.index, same) if not s][:50]}
        log(f"[D] V-D0 {validation['V_D0_exp6_plain_reproduction']}")

    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {validation['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
