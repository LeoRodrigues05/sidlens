#!/usr/bin/env python
"""SAE fidelity splice (F2) and latent ablations (CC copy latents, CM prefix-match latents).

Protocol: `protocol.md` beside this file. One invocation = one cell, test split.
Latent sets come from `analyze.py`'s `selection.json` (screening half). Every
ablation here runs on held-out rows only (F2 uses all rows).

Why the clean activation is re-captured here
--------------------------------------------
The test store was captured in batches of 16. bf16 is not batch-invariant, so
its vectors differ from what `FixedShapeRunner` computes by up to ~1 unit. An
error-preserving ablation x - sum s*a_f*W_dec[f] is only exact if x is the
very value the patched forward would have produced. Each chunk of rows is
therefore first run clean through the runner with a capture, and every patch
value is built from that capture. The self-patch control (patch x itself)
must then be bit-exact, and it is checked.

    python causal.py --cell 0 --sae-run <train cell dir> --selection <analysis cell dir> --out <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
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
from sidlens.hooks.store import StoreReader                              # noqa: E402
from sidlens.interventions.runner import FixedShapeRunner, Job           # noqa: E402
from sidlens.interventions.scoring import DigitScorer                    # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402
from sidlens.sae.topk import TopKSAE                                     # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
TRAIN_STORE = {"next-item_best": "287160", "oneoff_rqvae4cb128": "287161"}
LAYERS = (12, 16, 20, 24)
SEED = 20260930
ROLES = ("hist_sid", "response_header", "target_sid")
N_SELF_CONTROLS = 64


def half(user: str) -> int:
    return int(hashlib.sha256(f"{SEED}|{user}".encode()).hexdigest(), 16) % 2


def checkpoint_sha(ckpt: str) -> str:
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--sae-run", type=Path, required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--parts", default="F2,CC,CM")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ckpt = CELLS[args.cell]
    if (args.sae_run / f"cell-{args.cell:02d}").is_dir():          # a run group: take this cell
        args.sae_run = args.sae_run / f"cell-{args.cell:02d}"
    if (args.selection / f"cell-{args.cell:02d}").is_dir():          # a run group: take this cell
        args.selection = args.selection / f"cell-{args.cell:02d}"
    parts = set(args.parts.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "validation.json").exists():
        raise FileExistsError(f"{args.out} already holds results")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()
    logf = open(args.out / "run.log", "a")

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    sel = json.loads((args.selection / "selection.json").read_text())
    if sel["cell"] != ckpt:
        raise ValueError(f"selection is for {sel['cell']}, not {ckpt}")
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / ckpt), local_files_only=True)
    examples = P.load_examples(vocab.variant, "next-item", "test")
    table = SidTable.load(vocab.variant)
    validation = {"table": P.check_against_table(examples, table), "archive": P.check_against_archive(examples)}
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:1000]}")
    clean = [P.encode(ex, tok, vocab, template="eval", with_target=True) for ex in examples]
    t_pad = -(-max(len(e) for e in clean) // 8) * 8
    if args.limit:
        examples, clean = examples[:args.limit], clean[:args.limit]
    codes = [(P.parse_sid(e.history_sids[-1], n), P.parse_sid(e.target_sids[0], n)) for e in examples]
    halves = [half(e.user_id) for e in examples]
    saes = {L: TopKSAE.load(args.sae_run / "sae" / f"L{L}", device="cuda") for L in LAYERS}
    tr = StoreReader(paths.DERIVED / "ar_capture" / TRAIN_STORE[ckpt] / "capture" / "store")
    means = {}
    for L in LAYERS:
        X = tr.load(f"model.layers.{L}")
        means[L] = X.double().mean(0).float().cuda()
        del X
    sha = checkpoint_sha(ckpt)
    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": sha, "variant": vocab.variant.name,
              "sae_run": str(args.sae_run), "selection": str(args.selection), "parts": sorted(parts),
              "n_rows": len(examples), "limited": args.limit, "t_pad": t_pad, "batch_rows": args.rows,
              "layers": LAYERS, "train_mean_store": TRAIN_STORE[ckpt]}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    model = ar.load_model(ckpt, device="cuda", dtype="bfloat16")
    runner = FixedShapeRunner(model, DigitScorer(vocab, table), tok.pad_token_id, t_pad, args.rows, 1)
    sites = [f"model.layers.{L}" for L in LAYERS]
    d_model = model.config.hidden_size
    c0_codes = list(vocab.codes(0))

    recs, ctrl = [], []
    buf: list = []

    def flush():
        if not buf:
            return
        B = len(buf)
        caches = {"mod": {s: torch.zeros(B, t_pad, d_model, dtype=torch.bfloat16, device="cuda") for s in
                          {b[1] for b in buf}}}
        jobs = []
        for r, (job, site, posl, vals) in enumerate(buf):
            caches["mod"][site][r, torch.as_tensor(posl, device="cuda")] = vals.to(torch.bfloat16)
            job.cache_row, job.patches = r, [(site, posl, "mod")]
            jobs.append(job)
        s, cl, _, _ = runner.forward(jobs, caches=caches)
        for r, job in enumerate(jobs):
            m = job.meta
            i = m["i"]
            if m["part"] == "SELF":
                ctrl.append({"i": i, "layer": m["layer"], "max_abs_diff": float(
                    max((a[r] - b).abs().max() for a, b in zip(cl, clean_logits[i])))})
                continue
            for d in m["digits"]:
                rec = {"example_id": examples[i].example_id, "row": i, "user_id": examples[i].user_id,
                       "half": halves[i], "part": m["part"], "layer": m["layer"], "condition": m["condition"],
                       "d": m.get("d", -1), "digit": d, "logp_codes": float(s["logp_codes"][r, d]),
                       "top1": int(s["top1"][r, d]), "n_active_removed": m.get("n_active", -1)}
                if d == 0:
                    lp = torch.log_softmax(cl[0][r], -1)
                    rec["copy_score"] = float(lp[c0_codes.index(codes[i][0][0])])
                recs.append(rec)
        buf.clear()

    def add(job, site, posl, vals):
        buf.append((job, site, posl, vals))
        if len(buf) >= runner.B:
            flush()

    clean_logits: dict[int, list] = {}
    for b0 in range(0, len(examples), runner.B):
        idxs = list(range(b0, min(len(examples), b0 + runner.B)))
        jobs = [Job(clean[i], meta={"i": i}) for i in idxs]
        s, cl, cap, _ = runner.forward(jobs, capture=sites)
        for r, i in enumerate(idxs):
            clean_logits[i] = [c[r] for c in cl]
            for d in range(n):
                rec = {"example_id": examples[i].example_id, "row": i, "user_id": examples[i].user_id,
                       "half": halves[i], "part": "clean", "layer": -1, "condition": "clean", "d": -1, "digit": d,
                       "logp_codes": float(s["logp_codes"][r, d]), "top1": int(s["top1"][r, d]),
                       "n_active_removed": 0}
                if d == 0:
                    rec["copy_score"] = float(torch.log_softmax(cl[0][r], -1)[c0_codes.index(codes[i][0][0])])
                recs.append(rec)
        for r, i in enumerate(idxs):
            e = clean[i]
            p0 = e.predict_pos(0, 0)
            for L, site in zip(LAYERS, sites):
                x = cap[site][r]                                          # (T_pad, d) bf16
                sae = saes[L]
                if i < N_SELF_CONTROLS:
                    add(Job(e, meta={"i": i, "part": "SELF", "layer": L}), site, [p0], x[[p0]])
                if "F2" in parts:
                    posl = [p for p, ro in enumerate(e.role) if ro in ROLES]
                    xs = x[posl]
                    with torch.no_grad():
                        add(Job(e, meta={"i": i, "part": "F2", "layer": L, "condition": "sae",
                                         "digits": list(range(n))}), site, posl, sae.reconstruct(xs))
                    add(Job(e, meta={"i": i, "part": "F2", "layer": L, "condition": "mean",
                                     "digits": list(range(n))}), site, posl, means[L].expand(len(posl), -1))
                if halves[i] != 1:
                    continue
                if "CC" in parts:
                    plan = sel["layers"][str(L)]["copy_plan"].get(examples[i].example_id)
                    if plan:
                        for cond in ("copy", "control"):
                            with torch.no_grad():
                                v, keep = sae.ablate(x[[p0]], [plan[cond]])
                            add(Job(e, meta={"i": i, "part": "CC", "layer": L, "condition": cond, "digits": [0],
                                             "n_active": int(keep.sum())}), site, [p0], v)
                if "CM" in parts:
                    lay = sel["layers"][str(L)]
                    sets = [("S", lay["S"])] + [(f"C{j}", cs) for j, cs in enumerate(lay["control_sets"])]
                    for d in range(1, n):
                        pd_ = e.predict_pos(0, d)
                        for cond, fs in sets:
                            with torch.no_grad():
                                v, keep = sae.ablate(x[[pd_]], [fs])
                            add(Job(e, meta={"i": i, "part": "CM", "layer": L, "condition": cond, "d": d,
                                             "digits": [d], "n_active": int(keep.sum()),
                                             "match": codes[i][0][:d] == codes[i][1][:d]}), site, [pd_], v)
        flush()
        log(f"[chunk] rows {idxs[-1] + 1}/{len(examples)}, {runner.n_forwards} forwards, {time.time() - t0:.0f}s")
    flush()
    df = pd.DataFrame(recs)
    df["match"] = [codes[i][0][:d] == codes[i][1][:d] if d >= 1 else codes[i][0][0] == codes[i][1][0]
                   for i, d in zip(df.row, df.digit)]
    df.to_parquet(args.out / "records.parquet", index=False)
    c = pd.DataFrame(ctrl)
    c.to_parquet(args.out / "controls.parquet", index=False)
    validation["self_patch"] = {"n": int(len(c)), "n_exact": int((c.max_abs_diff == 0).sum()),
                                "max_abs_diff": float(c.max_abs_diff.max()) if len(c) else None}
    if len(c) and validation["self_patch"]["n_exact"] != len(c):
        (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
        raise RuntimeError(f"self-patch controls not bit-exact: {validation['self_patch']}")
    validation["n_forwards"] = runner.n_forwards
    validation["seconds"] = round(time.time() - t0, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: {json.dumps(validation['self_patch'])} in {validation['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
