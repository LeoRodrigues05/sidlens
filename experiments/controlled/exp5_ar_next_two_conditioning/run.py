#!/usr/bin/env python
"""Next-two AR (two-item_best): clamp item 1 (Part A) and patch its residual stream (Part B).

Protocol: `protocol.md` beside this file, declared before any forward of this
checkpoint here. Teacher forcing on "sid1 ||| sid2\\n"; every target digit of
both slots is scored at `predict_pos(slot, digit)` through the shared
fixed-shape runner (`sidlens.interventions.runner`), whose no-op, self-patch
and full-restore controls must be bit-exact.

    python experiments/controlled/exp5_ar_next_two_conditioning/run.py --out <new dir>
"""

from __future__ import annotations

import argparse
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
from sidlens.data.sids import SidTable, load_item2id                     # noqa: E402
from sidlens.interventions import history as Hi                          # noqa: E402
from sidlens.interventions.runner import FixedShapeRunner, Job, max_diff, records  # noqa: E402
from sidlens.interventions.scoring import DigitScorer                    # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402

CKPT = "two-item_best"
SEED = 20260927
GROUPS = ("slot0", "sep", "slot1")
CONTROL_LAYERS = (0, 9, 18, 27)
CONTROLS = {"noop_pre": ("pre", "clean", "corrupted"),       # kind: (group, source, reference)
            "full_restore": ("all", "clean", "clean"),
            "self_patch": ("slot0", "corrupted", "corrupted")}


def checkpoint_sha(ckpt: str) -> str:
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def groups(e: P.Encoded, n: int) -> dict[str, list[int]]:
    g = {"slot0": e.positions("target_sid", item=0), "sep": e.positions("target_sep"),
         "slot1": e.positions("target_sid", item=1)}
    g["pre"] = list(range(g["slot0"][0]))
    g["all"] = list(range(len(e)))
    if len(g["slot0"]) != n or len(g["slot1"]) != n or not g["sep"]:
        raise ValueError(f"{e.example.example_id}: unexpected two-slot target layout")
    if g["sep"][-1] != e.predict_pos(1, 0) or g["sep"][0] != g["slot0"][-1] + 1 \
            or g["slot1"][0] != g["sep"][-1] + 1:
        raise ValueError(f"{e.example.example_id}: separator is not between the two SIDs")
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", default="A,B")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cell", type=int, default=0, help="accepted for the array wrapper; one cell")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    parts = set(args.parts.split(","))
    if not parts or not parts <= {"A", "B"}:
        ap.error("--parts is a subset of A,B")
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

    vocab = ar.load_vocab(CKPT)
    n = vocab.variant.n_codebook
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / CKPT), local_files_only=True)
    examples = P.load_examples(vocab.variant, "two-item", "test")
    table = SidTable.load(vocab.variant)
    validation = {"table": P.check_against_table(examples, table),
                  "archive": P.check_against_archive(examples),
                  "vocab_vs_table": ar.validate_against(vocab, table)}
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:2000]}")
    clean = [P.encode(ex, tok, vocab, template="eval", with_target=True) for ex in examples]
    t_pad = -(-max(len(e) for e in clean) // 8) * 8
    if args.limit:
        examples, clean = examples[:args.limit], clean[:args.limit]
    pool = Hi.ControlPool(table, load_item2id(P.PRIMARY_CATEGORY))
    csv = P.csv_path(vocab.variant, "two-item", "test")
    arch = P.archive_path(vocab.variant, "two-item")
    sha = checkpoint_sha(CKPT)
    inputs = {"checkpoint": f"ar/{CKPT}", "checkpoint_sha256": sha, "variant": vocab.variant.name,
              "n_digits": n, "split": "test", "task": "two-item", "template": "eval",
              "csv": str(csv.relative_to(paths.WORK)), "csv_sha256": hashing.sha256_file(csv),
              "archive_sha256": hashing.sha256_file(arch), "sem_ids_sha256": hashing.sha256_file(table.source),
              "n_rows": len(examples), "n_users": len({e.user_id for e in examples}), "limited": args.limit,
              "t_pad": t_pad, "batch_rows": args.rows, "seed": SEED, "parts": sorted(parts)}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"inputs ok: {CKPT} rows={len(examples)} T_pad={t_pad} sha={sha[:12]}")

    model = ar.load_model(CKPT, device=args.device, dtype=args.dtype)
    inputs.update({"dtype": args.dtype, "attn_implementation": model.config._attn_implementation,
                   "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                   "torch": torch.__version__, "transformers": __import__("transformers").__version__})
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    runner = FixedShapeRunner(model, DigitScorer(vocab, table), tok.pad_token_id, t_pad, args.rows, 2)
    n_layers = model.config.num_hidden_layers

    # ---- Part A ---------------------------------------------------------------
    corr, missing = [None] * len(examples), []
    recs, clean_logits, buf, batch = [], {}, [], 0

    def flush():
        nonlocal buf, batch
        s, cl, _, _ = runner.forward(buf)
        recs.extend(records(runner, buf, s, batch))
        for r, job in enumerate(buf):
            if job.meta["condition"] == "clean":
                clean_logits[job.meta["i"]] = [c[r] for c in cl]
        batch += 1
        buf = []

    def push(job):
        buf.append(job)
        if len(buf) >= runner.B:
            flush()

    t0 = time.time()
    for i, ex in enumerate(examples):
        push(Job(clean[i], meta={"i": i, "condition": "clean", "m": -1, "control_item": -1}))
        for m in range(n):
            c = pool.draw_target(ex, 0, m, SEED)
            if c is None:
                missing.append({"example_id": ex.example_id, "condition": "item1", "m": m})
                continue
            e2 = P.encode(Hi.replace_target_slot(ex, 0, c), tok, vocab, template="eval", with_target=True)
            Hi.check_replacement(clean[i], e2, 0, m, role="target_sid")
            if m == 0:
                corr[i] = e2
            if "A" in parts:
                push(Job(e2, meta={"i": i, "condition": "item1", "m": m, "control_item": c.item_id}))
        k = len(ex.history_sids) - 1
        c = pool.draw(ex, k, 0, SEED)
        if c is None:
            missing.append({"example_id": ex.example_id, "condition": "hist1", "m": 0})
        elif "A" in parts:
            e2 = P.encode(Hi.replace_history_item(ex, k, c), tok, vocab, template="eval", with_target=True)
            Hi.check_replacement(clean[i], e2, k, 0)
            push(Job(e2, meta={"i": i, "condition": "hist1", "m": 0, "control_item": c.item_id}))
    if buf:
        flush()
    df = pd.DataFrame(recs)
    df[df.condition == "clean"].to_parquet(args.out / "clean.parquet", index=False)
    if "A" in parts:
        df[df.condition != "clean"].to_parquet(args.out / "part_a.parquet", index=False)
    pd.DataFrame(missing, columns=["example_id", "condition", "m"]).to_csv(
        args.out / "part_a_no_control.csv", index=False)
    validation["part_a"] = {"records": len(df), "no_control": len(missing)}
    log(f"[A] {len(df)} records, {len(missing)} missing controls, {time.time() - t0:.0f}s")

    # ---- Part B ---------------------------------------------------------------
    if "B" in parts:
        if any(c is None for c in corr):
            raise RuntimeError("an item-1 m=0 control is missing; Part B needs every row")
        sites = [f"model.layers.{L}" for L in range(n_layers)]
        ctrl_sites = [f"model.layers.{L}" for L in CONTROL_LAYERS]
        brecs, crecs, inv, batch = [], [], 0.0, 0
        t0 = time.time()
        for start in range(0, len(examples), runner.B):
            idx = list(range(start, min(len(examples), start + runner.B)))
            cl_jobs = [Job(clean[i], j, meta={"i": i, "condition": "clean", "layer": -1, "group": ""})
                       for j, i in enumerate(idx)]
            co_jobs = [Job(corr[i], j, meta={"i": i, "condition": "corrupted", "layer": -1, "group": ""})
                       for j, i in enumerate(idx)]
            s1, lg1, cap1, _ = runner.forward(cl_jobs, capture=sites)
            s2, lg2, cap2, _ = runner.forward(co_jobs, capture=ctrl_sites)
            caches = {"clean": {s: cap1[s] for s in sites}, "corrupted": {s: cap2[s] for s in ctrl_sites}}
            brecs += records(runner, cl_jobs, s1, batch) + records(runner, co_jobs, s2, batch + 1)
            batch += 2
            ref = {"clean": {i: [c[j] for c in lg1] for j, i in enumerate(idx)},
                   "corrupted": {i: [c[j] for c in lg2] for j, i in enumerate(idx)}}
            for i in idx:
                inv = max(inv, max_diff(ref["clean"][i], clean_logits[i]))
            jobs = []
            for j, i in enumerate(idx):
                g = groups(clean[i], n)
                for L, site in enumerate(sites):
                    for grp in GROUPS:
                        jobs.append(Job(corr[i], j, [(site, g[grp], "clean")],
                                        meta={"i": i, "condition": "patch", "layer": L, "group": grp}))
                for kind, (grp, src, _) in CONTROLS.items():
                    for L in CONTROL_LAYERS:
                        jobs.append(Job(corr[i], j, [(f"model.layers.{L}", g[grp], src)],
                                        meta={"i": i, "condition": kind, "layer": L, "group": grp}))
            for b in range(0, len(jobs), runner.B):
                chunk = jobs[b:b + runner.B]
                s, lg, _, _ = runner.forward(chunk, caches=caches)
                keep = np.array([jb.meta["condition"] == "patch" for jb in chunk])
                brecs += records(runner, [jb for jb in chunk if jb.meta["condition"] == "patch"],
                                 {k: v[keep] for k, v in s.items()}, batch)
                for r, jb in enumerate(chunk):
                    kind = jb.meta["condition"]
                    if kind == "patch":
                        continue
                    d_ = max_diff([c[r] for c in lg], ref[CONTROLS[kind][2]][jb.meta["i"]])
                    crecs.append({"example_id": jb.enc.example.example_id, "control": kind,
                                  "layer": jb.meta["layer"], "max_abs_diff_code_logits": d_, "exact": d_ == 0.0})
                batch += 1
            log(f"[B] {idx[-1] + 1}/{len(examples)} rows, {runner.n_forwards} forwards, {time.time() - t0:.0f}s")
        pd.DataFrame(brecs).to_parquet(args.out / "part_b.parquet", index=False)
        c = pd.DataFrame(crecs)
        c.to_parquet(args.out / "controls.parquet", index=False)
        validation["controls"] = {k: {"n": int(len(g)), "n_exact": int(g.exact.sum()),
                                      "max_abs_diff_code_logits": float(g.max_abs_diff_code_logits.max())}
                                  for k, g in c.groupby("control")}
        validation["batch_invariance_clean_vs_part_a"] = inv

    ref = pd.read_parquet(args.out / "clean.parquet")
    validation["clean_top1_by_slot_digit"] = {f"{s}.{d}": float((g["rank"] == 0).mean())
                                              for (s, d), g in ref.groupby(["slot", "digit"])}
    validation["n_forwards"] = runner.n_forwards
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: {json.dumps(validation.get('controls'), default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
