#!/usr/bin/env python
"""Capture AR residual-stream activations at named positions, into an activation store.

This is the reference path from a frozen AR checkpoint to analysis-ready
activations, on any cluster that holds the `ar` profile. It chains the pieces
that are each validated elsewhere:

    data.ar_prompts   exact upstream ids + position map (tests/data/test_ar_prompts.py)
    hooks.capture     observation-only forward hooks (tests/hooks/test_hooks.py)
    hooks.store       pickle-free sharded store (tests/hooks/test_activation_store.py)

For every example it runs ONE teacher-forced forward over prompt + golden
target. It stores the chosen layers at the chosen roles, and scores every
target digit at its `predict_pos` against that digit's codes (the model's
actual decision set, `vocab.n_codes`, not `codebook_size`).

What it checks before anything is written
-----------------------------------------
* The checkpoint bytes are the frozen ones (sha256 vs the provenance manifest).
* The CSV SIDs equal the frozen `.sem_ids`, and on the test split the rows
  equal the archived predictions, so the rows are the archived cohort.
* Batch invariance: the first example is re-run alone, and the max |diff| of
  its captured vectors and digit log-probs is recorded. bf16 on GPU is not
  bit-invariant to batch composition; the number says by how much.

What it does not claim
----------------------
Teacher-forced digit ranks are not beam-search HR. The archived evaluator
decodes with a trie over legal SIDs and 50 beams, so `summary.digit_top1` is a
sanity signal (a wrong prompt collapses it), not a reproduction of recorded
metrics.

    python scripts/activations/ar_capture.py --ckpt next-item_best --template eval \\
        --layers 0,7,14,21,27 --limit 64 --out $SIDLENS_WORK/derived/ar_capture/pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch                                                  # noqa: E402

from sidlens import hooks as H, paths                         # noqa: E402
from sidlens.data import ar_prompts as P                      # noqa: E402
from sidlens.hooks.store import StoreWriter                   # noqa: E402
from sidlens.models import ar                                 # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod  # noqa: E402

TASK_OF = {"next-item_best": "next-item", "oneoff_rqvae4cb128": "next-item",
           "two-item_best": "two-item"}
DEFAULT_ROLES = ("hist_sid", "response_header", "target_sid")


def checkpoint_sha(ckpt: str) -> str:
    """Hash the weights and compare with the snapshot; refuse on mismatch."""
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def targets(e: P.Encoded, vocab) -> list[tuple[int, int, int]]:
    """(slot, digit, code) for every target digit of one example."""
    n = vocab.variant.n_codebook
    return [(s, d, c) for s, sid in enumerate(e.example.target_sids)
            for d, c in enumerate(P.parse_sid(sid, n))]


def forward(model, batch, device, names, pad_id, vocab):
    """One no-grad forward. Returns the capture and, per example, the full-vocab
    logits at each target digit's predict position (only those rows leave the
    device: the whole (B, T, 152k) tensor is ~1 GB per batch in fp32)."""
    c = P.collate(batch, pad_id=pad_id, side="right")
    kw = {k: c[k].to(device) for k in ("input_ids", "attention_mask", "position_ids")}
    out, cap = H.run_capture(model, lambda: model(**kw), names=names)
    per_ex = []
    for i, e in enumerate(batch):
        o = int(c["offset"][i])
        pos = torch.tensor([o + e.predict_pos(s, d) for s, d, _ in targets(e, vocab)])
        per_ex.append(out.logits[i, pos.to(out.logits.device)].float().cpu())
    return per_ex, cap, c["offset"]


def digit_scores(lg_rows, e: P.Encoded, vocab) -> list[dict]:
    """Rank + log-prob of each target digit among that digit's codes."""
    rows = []
    for lg, (slot, d, c) in zip(lg_rows, targets(e, vocab)):
        codes = vocab.codes(d)
        sub = ar.digit_logits(lg, vocab, d)
        logp = torch.log_softmax(sub, -1)
        j = codes.index(c)
        rows.append({"example_id": e.example.example_id, "slot": slot, "digit": d,
                     "target_code": c, "n_codes": len(codes),
                     "rank": int((sub > sub[j]).sum()),
                     "logp_codes": float(logp[j]),
                     "logp_vocab": float(torch.log_softmax(lg, -1)[vocab.id_of(d, c)])})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, choices=sorted(ar.AR_CHECKPOINTS))
    ap.add_argument("--template", required=True, choices=P.TEMPLATES,
                    help="eval = the archived-prediction wording; sft = training wording")
    ap.add_argument("--split", default="test", choices=P.SPLITS)
    ap.add_argument("--layers", default="all",
                    help="'all' or comma list of decoder layer indices; 'norm' adds model.norm")
    ap.add_argument("--roles", default=",".join(DEFAULT_ROLES))
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilots)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None,
                    help="default bfloat16 on cuda (as evaluate.py), float32 on cpu")
    ap.add_argument("--shard-gb", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True, help="must not exist")
    args = ap.parse_args(argv)
    t0 = time.time()
    dtype = args.dtype or ("bfloat16" if args.device.startswith("cuda") else "float32")
    roles = tuple(r for r in args.roles.split(",") if r)

    # ---- inputs, validated before the model is touched ---------------------
    vocab = ar.load_vocab(args.ckpt)
    task = TASK_OF[args.ckpt]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / args.ckpt),
                                        local_files_only=True)
    examples = P.load_examples(vocab.variant, task, args.split)
    validation = {"table": P.check_against_table(examples)}
    if args.split == "test":
        validation["archive"] = P.check_against_archive(examples)
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation)[:2000]}")
    examples = examples[:args.limit] if args.limit else examples
    encoded = [P.encode(ex, tok, vocab, template=args.template, with_target=True)
               for ex in examples]

    sha = checkpoint_sha(args.ckpt)
    model = ar.load_model(args.ckpt, device=args.device, dtype=dtype)
    n_layers = model.config.num_hidden_layers
    want = args.layers.split(",")
    idx = range(n_layers) if "all" in want else [int(x) for x in want if x != "norm"]
    names = [f"model.layers.{i}" for i in idx] + (["model.norm"] if "norm" in want else [])
    known = set(H.residual_points(model))
    if unknown := [n for n in names if n not in known]:
        raise KeyError(f"not residual read points of this model: {unknown}")

    meta = {"model": f"ar/{args.ckpt}", "checkpoint_sha256": sha,
            "variant": vocab.variant.name, "task": task, "split": args.split,
            "template": args.template, "dtype": dtype, "device": args.device,
            "sites_are": "decoder-layer outputs (residual stream after the block)",
            "roles": list(roles), "batch_size": args.batch_size, "padding": "right",
            "teacher_forced": True, "n_examples": len(encoded),
            "torch": torch.__version__}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "arguments.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=1))

    scores: list[dict] = []
    first_batched = None
    with StoreWriter(args.out / "store", meta, shard_bytes=int(args.shard_gb * (1 << 30))) as w:
        for b in range(0, len(encoded), args.batch_size):
            batch = encoded[b:b + args.batch_size]
            logits, cap, offset = forward(model, batch, args.device, names,
                                          tok.pad_token_id, vocab)
            rows, take = [], []
            for i, e in enumerate(batch):
                scores += digit_scores(logits[i], e, vocab)
                for rec in e.records():
                    if rec["role"] in roles:
                        rows.append(rec)
                        take.append((i, int(offset[i]) + rec["pos"]))
            bi = torch.tensor([t[0] for t in take])
            pi = torch.tensor([t[1] for t in take])
            tensors = {n: cap[n][bi, pi] for n in names}
            if first_batched is None:
                k = sum(1 for t in take if t[0] == 0)
                first_batched = ({n: tensors[n][:k].float() for n in names},
                                 digit_scores(logits[0], batch[0], vocab))
            w.add(rows, tensors)
            print(f"[capture] {min(b + args.batch_size, len(encoded))}/{len(encoded)}", flush=True)

    # ---- batch invariance on the first example -----------------------------
    logits1, cap1, _ = forward(model, encoded[:1], args.device, names,
                               tok.pad_token_id, vocab)
    pos = [r["pos"] for r in encoded[0].records() if r["role"] in roles]
    single = {n: cap1[n][0, pos].float() for n in names}
    s1 = digit_scores(logits1[0], encoded[0], vocab)
    validation["batch_invariance"] = {
        "max_abs_diff_activation": max(float((single[n] - first_batched[0][n]).abs().max())
                                       for n in names),
        "max_abs_diff_logp_codes": max(abs(a["logp_codes"] - b["logp_codes"])
                                       for a, b in zip(s1, first_batched[1])),
        "note": "first example alone vs inside its batch; bf16 is not batch-invariant"}

    with open(args.out / "digit_scores.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(scores[0]))
        wr.writeheader()
        wr.writerows(scores)
    by_digit: dict = {}
    for r in scores:
        by_digit.setdefault(f"{r['slot']}.{r['digit']}", []).append(r["rank"] == 0)
    summary = {"n_examples": len(encoded), "n_layers_stored": len(names),
               "digit_top1": {k: sum(v) / len(v) for k, v in by_digit.items()},
               "seconds": round(time.time() - t0, 1)}
    (args.out / "validation.json").write_text(json.dumps(
        {**validation, "summary": summary}, indent=1, default=str))
    print(f"[capture] done: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
