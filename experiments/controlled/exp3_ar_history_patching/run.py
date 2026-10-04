#!/usr/bin/env python
"""History-item replacement (Part A) and residual patching (Part B) on one frozen AR checkpoint.

Protocol: `protocol.md` beside this file, declared before any intervened
forward was run. One invocation = one cell (checkpoint x template).

Why every forward has the same shape
------------------------------------
The acceptance controls are bit-equality checks: a no-op patch must change
nothing, a full-layer patch must reproduce the clean run, and a clean row must
score the same whichever rows share its batch. bf16 GPU kernels are not
invariant to the batch SHAPE (tiling and algorithm choice change with it), but
for a fixed shape a row's output depends only on that row. So every forward is
B rows x T_pad tokens: T_pad from the whole cohort (a pilot and the full run
share it), short batches filled with copies of a real row whose outputs are
discarded, right padding with explicit `position_ids`. If the controls are
still not exact, the deviation is recorded as the numerical floor (protocol).

What it does not claim
----------------------
Teacher-forced digit scores are conditional scores at a fixed state, not
beam-search HR; the archived evaluator's trie over 50 beams is not run here.

    python experiments/controlled/exp3_ar_history_patching/run.py \\
        --ckpt next-item_best --template eval --parts A,B --out <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import hooks as H, paths                                   # noqa: E402
from sidlens.data import ar_prompts as P                                # noqa: E402
from sidlens.data.sids import SidTable, load_item2id                    # noqa: E402
from sidlens.interventions import history as Hi                         # noqa: E402
from sidlens.interventions.residual import Patch, patch_residual        # noqa: E402
from sidlens.interventions.scoring import MEASURES, DigitScorer         # noqa: E402
from sidlens.models import ar                                           # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod        # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")         # cell-00, cell-01
SEED = 20260927
GROUPS = ("item", "between", "header", "target")
CONTROL_LAYERS = (0, 9, 18, 27)
CONTROLS = {"noop_pre": ("pre", "clean", "corrupted"),      # kind: (group, source, reference)
            "full_restore": ("all", "clean", "clean"),
            "self_patch": ("item", "corrupted", "corrupted")}


# ------------------------------------------------------------------ inputs --
def checkpoint_sha(ckpt: str) -> str:
    """Hash the weights and compare with the snapshot; refuse on mismatch.
    Same check as scripts/activations/ar_capture.py::checkpoint_sha."""
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def ids_sha(e: P.Encoded) -> str:
    return hashlib.sha256(np.asarray(e.input_ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def position_groups(e: P.Encoded, k: int, n: int) -> dict[str, list[int]]:
    """The protocol's position groups for history item k, checked, not assumed."""
    item = e.positions("hist_sid", item=k)
    header = e.positions("response_header")
    target = e.positions("target_sid", item=0)
    g = {"item": item, "between": list(range(item[-1] + 1, header[0])), "header": header,
         "target": target, "pre": list(range(item[0])), "all": list(range(len(e)))}
    if len(item) != n or item != list(range(item[0], item[0] + n)):
        raise ValueError(f"{e.example.example_id}: item {k} is not {n} contiguous SID tokens")
    if len(header) != 3 or header[-1] != e.predict_pos(0, 0) or len(target) != n:
        raise ValueError(f"{e.example.example_id}: unexpected header/target layout")
    if not g["between"] or not g["pre"]:
        raise ValueError(f"{e.example.example_id}: empty 'between' or 'pre' group")
    return g


# ------------------------------------------------------------------ runner --
@dataclass
class Row:
    enc: P.Encoded
    slot: int = 0                          # row of this example in the chunk caches
    patches: list = field(default_factory=list)   # [(site, positions, source)]
    meta: dict = field(default_factory=dict)


class Runner:
    """Fixed-shape forwards returning per-digit scores and code logits."""

    def __init__(self, model, scorer: DigitScorer, pad_id: int, t_pad: int, rows: int):
        self.model, self.scorer, self.pad_id = model, scorer, pad_id
        self.t_pad, self.B = t_pad, rows
        self.dev = next(model.parameters()).device
        self.n = scorer.n_digits
        self.n_forwards = 0

    def _tensors(self, encs: list[P.Encoded]):
        c = P.collate(encs, pad_id=self.pad_id, side="right")
        if int(c["offset"].abs().max()) != 0:
            raise AssertionError("right padding must leave every offset at 0")
        extra = self.t_pad - c["input_ids"].shape[1]
        if extra < 0:
            raise ValueError(f"sequence longer than T_pad={self.t_pad}")
        pad = torch.nn.functional.pad
        # Same convention as collate(): pad id, mask 0, position 1.
        return (pad(c["input_ids"], (0, extra), value=self.pad_id).to(self.dev),
                pad(c["attention_mask"], (0, extra), value=0).to(self.dev),
                pad(c["position_ids"], (0, extra), value=1).to(self.dev))

    def _patches(self, rows: list[Row], caches: dict | None) -> list[Patch]:
        acc: dict[str, list] = {}
        for r, row in enumerate(rows):
            for site, positions, source in row.patches:
                p = torch.as_tensor(positions, dtype=torch.long)
                vals = caches[source][site][row.slot, p.to(self.dev)]
                a = acc.setdefault(site, [[], [], []])
                a[0].append(torch.full((len(p),), r, dtype=torch.long))
                a[1].append(p)
                a[2].append(vals)
        return [Patch(s, torch.cat(a[0]), torch.cat(a[1]), torch.cat(a[2])) for s, a in acc.items()]

    @torch.no_grad()
    def forward(self, rows: list[Row], caches: dict | None = None,
                capture: list[str] | None = None):
        real = len(rows)
        if not 0 < real <= self.B:
            raise ValueError(f"batch of {real} rows, capacity {self.B}")
        if capture and any(r.patches for r in rows):
            raise ValueError("capture and patch in one forward would record patched values")
        encs = [r.enc for r in rows] + [rows[0].enc] * (self.B - real)
        ids, mask, pos = self._tensors(encs)
        pred = torch.tensor([[e.predict_pos(0, d) for d in range(self.n)] for e in encs],
                            device=self.dev)
        patches = self._patches(rows, caches)
        cap_ctx = H.capture(self.model, names=capture, to_cpu=False) if capture else nullcontext()
        with patch_residual(self.model, patches), cap_ctx as cap:
            h = self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                 use_cache=False).last_hidden_state
            logits = self.model.lm_head(h[torch.arange(self.B, device=self.dev)[:, None], pred])
        self.n_forwards += 1
        logits = logits[:real]
        targets = np.array([P.parse_sid(r.enc.example.target_sids[0], self.n) for r in rows])
        scores = self.scorer.score(logits, targets)
        code_logits = [self.scorer.code_logits(logits, d).float().cpu() for d in range(self.n)]
        return scores, code_logits, cap


def records(rows: list[Row], scores: dict, batch: int, n: int) -> list[dict]:
    out = []
    for r, row in enumerate(rows):
        ex = row.enc.example
        tgt = P.parse_sid(ex.target_sids[0], n)
        for d in range(n):
            rec = {"example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id,
                   "hist_len": len(ex.history_item_ids), **row.meta, "digit": d,
                   "target_code": tgt[d]}
            rec.update({k: scores[k][r, d].item() for k in MEASURES})
            rec["batch"] = batch
            out.append(rec)
    return out


def max_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


# ------------------------------------------------------------------ part A --
def part_a(runner: Runner, examples, clean, tok, vocab, pool, template, n, log):
    """Every row x history item x shared-prefix level, plus the clean row."""
    recs, missing, clean_logits = [], [], {}
    buf: list[Row] = []
    batch = 0

    def flush():
        nonlocal batch, buf
        s, cl, _ = runner.forward(buf)
        recs.extend(records(buf, s, batch, n))
        for r, row in enumerate(buf):
            if row.meta["condition"] == "clean":
                clean_logits[row.meta["i"]] = [c[r] for c in cl]
        batch += 1
        buf = []

    def push(row: Row):
        # Flush on every append: a check after only some appends let a clean
        # row land on the boundary and the buffer grow past B unnoticed.
        buf.append(row)
        if len(buf) >= runner.B:
            flush()

    t0 = time.time()
    for i, ex in enumerate(examples):
        push(Row(clean[i], meta={"i": i, "condition": "clean", "k": -1, "recency": 0,
                                 "m": -1, "control_item": -1, "control_sid": "",
                                 "n_candidates": 0, "input_sha": ids_sha(clean[i])}))
        L = len(ex.history_item_ids)
        for k in range(L):
            for m in range(n):
                c = pool.draw(ex, k, m, SEED)
                if c is None:
                    missing.append({"example_id": ex.example_id, "row": ex.row, "k": k,
                                    "recency": L - k, "m": m})
                    continue
                e2 = P.encode(Hi.replace_history_item(ex, k, c), tok, vocab,
                              template=template, with_target=True)
                Hi.check_replacement(clean[i], e2, k, m)
                push(Row(e2, meta={"i": i, "condition": "replace", "k": k, "recency": L - k,
                                   "m": m, "control_item": c.item_id, "control_sid": c.sid,
                                   "n_candidates": c.n_candidates, "input_sha": ids_sha(e2)}))
        if (i + 1) % 500 == 0:
            log(f"[A] {i + 1}/{len(examples)} rows, {batch} forwards, {time.time() - t0:.0f}s")
    if buf:
        flush()
    log(f"[A] done: {len(recs)} records, {len(missing)} missing controls, {time.time() - t0:.0f}s")
    return recs, missing, clean_logits


# ------------------------------------------------------------------ part B --
def part_b(runner: Runner, examples, clean, corr, controls, n_layers, n, log,
           clean_logits_a: dict | None):
    """Clean -> corrupted residual patching at every layer x position group."""
    sites = [f"model.layers.{L}" for L in range(n_layers)]
    ctrl_sites = [f"model.layers.{L}" for L in CONTROL_LAYERS]
    recs, ctrl_recs = [], []
    inv = {"clean_vs_part_a": 0.0, "n_compared": 0}
    batch, t0 = 0, time.time()
    for start in range(0, len(examples), runner.B):
        idx = list(range(start, min(len(examples), start + runner.B)))
        cl_rows = [Row(clean[i], slot=j, meta={"i": i, "condition": "clean", "layer": -1,
                                                "group": "", "source": ""}) for j, i in enumerate(idx)]
        co_rows = [Row(corr[i], slot=j, meta={"i": i, "condition": "corrupted", "layer": -1,
                                               "group": "", "source": ""}) for j, i in enumerate(idx)]
        s_cl, lg_cl, cap_cl = runner.forward(cl_rows, capture=sites)
        s_co, lg_co, cap_co = runner.forward(co_rows, capture=ctrl_sites)
        caches = {"clean": {s: cap_cl[s] for s in sites},
                  "corrupted": {s: cap_co[s] for s in ctrl_sites}}
        recs += records(cl_rows, s_cl, batch, n) + records(co_rows, s_co, batch + 1, n)
        batch += 2
        ref = {"clean": {i: [c[j] for c in lg_cl] for j, i in enumerate(idx)},
               "corrupted": {i: [c[j] for c in lg_co] for j, i in enumerate(idx)}}
        if clean_logits_a is not None:
            for i in idx:
                inv["clean_vs_part_a"] = max(inv["clean_vs_part_a"],
                                             max_diff(ref["clean"][i], clean_logits_a[i]))
                inv["n_compared"] += 1

        jobs: list[Row] = []
        for j, i in enumerate(idx):
            k = len(examples[i].history_item_ids) - 1
            g = position_groups(clean[i], k, n)
            for L, site in enumerate(sites):
                for grp in GROUPS:
                    jobs.append(Row(corr[i], j, [(site, g[grp], "clean")],
                                    {"i": i, "condition": "patch", "layer": L, "group": grp,
                                     "source": "clean"}))
            for kind, (grp, src, _) in CONTROLS.items():
                for L in CONTROL_LAYERS:
                    jobs.append(Row(corr[i], j, [(f"model.layers.{L}", g[grp], src)],
                                    {"i": i, "condition": kind, "layer": L, "group": grp,
                                     "source": src}))
        for b in range(0, len(jobs), runner.B):
            chunk = jobs[b:b + runner.B]
            s, lg, _ = runner.forward(chunk, caches=caches)
            is_patch = np.array([r.meta["condition"] == "patch" for r in chunk])
            patch_rows = [r for r in chunk if r.meta["condition"] == "patch"]
            recs += records(patch_rows, {k: v[is_patch] for k, v in s.items()}, batch, n)
            for r, row in enumerate(chunk):
                kind = row.meta["condition"]
                if kind == "patch":
                    continue
                refname = CONTROLS[kind][2]
                d = max_diff([c[r] for c in lg], ref[refname][row.meta["i"]])
                ctrl_recs.append({"example_id": row.enc.example.example_id, "row": row.enc.example.row,
                                  "control": kind, "layer": row.meta["layer"],
                                  "group": row.meta["group"], "source": row.meta["source"],
                                  "reference": refname, "max_abs_diff_code_logits": d,
                                  "exact": d == 0.0, "batch": batch})
            batch += 1
        log(f"[B] {idx[-1] + 1}/{len(examples)} rows, {runner.n_forwards} forwards, "
            f"{time.time() - t0:.0f}s")
    return recs, ctrl_recs, inv


# -------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)),
                    help="0 = next-item_best, 1 = oneoff_rqvae4cb128 (sbatch array index)")
    ap.add_argument("--ckpt", choices=CELLS)
    ap.add_argument("--template", default="eval", choices=P.TEMPLATES)
    ap.add_argument("--parts", default="A,B")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128, help="fixed batch rows B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--fp32-check-rows", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    ckpt = args.ckpt or CELLS[args.cell]
    parts = set(args.parts.split(","))
    if not parts <= {"A", "B"} or not parts:
        ap.error("--parts is a subset of A,B")
    t_start = time.time()
    # The sbatch wrapper creates --out (mkdir fails on a collision); a direct
    # call may too, but must never land on a directory holding results.
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json", "part_a.parquet",
                                              "part_b.parquet")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    logf = open(args.out / "run.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ---- inputs, validated before the model is touched ---------------------
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / ckpt), local_files_only=True)
    examples = P.load_examples(vocab.variant, "next-item", "test")
    table = SidTable.load(vocab.variant)
    validation = {"table": P.check_against_table(examples, table),
                  "archive": P.check_against_archive(examples),
                  "vocab_vs_table": ar.validate_against(vocab, table)}
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:2000]}")
    clean_all = [P.encode(ex, tok, vocab, template=args.template, with_target=True) for ex in examples]
    t_pad = -(-max(len(e) for e in clean_all) // 8) * 8
    if args.limit:
        examples, clean_all = examples[:args.limit], clean_all[:args.limit]
    pool = Hi.ControlPool(table, load_item2id(P.PRIMARY_CATEGORY))
    scorer = DigitScorer(vocab, table)

    csv = P.csv_path(vocab.variant, "next-item", "test")
    arch = P.archive_path(vocab.variant, "next-item")
    sha = checkpoint_sha(ckpt)
    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": sha,
              "added_tokens_sha256": hashing.sha256_file(paths.FROZEN_CKPT / "ar" / ckpt / "added_tokens.json"),
              "variant": vocab.variant.name, "n_digits": n,
              "codes_per_digit": [vocab.n_codes(d) for d in range(n)],
              "csv": str(csv.relative_to(paths.WORK)), "csv_sha256": hashing.sha256_file(csv),
              "sem_ids": str(table.source.relative_to(paths.WORK)),
              "sem_ids_sha256": hashing.sha256_file(table.source),
              "archive": str(arch.relative_to(paths.WORK)), "archive_sha256": hashing.sha256_file(arch),
              "template": args.template, "split": "test", "task": "next-item",
              "n_rows": len(examples), "n_users": len({e.user_id for e in examples}),
              "limited": args.limit, "t_pad": t_pad, "batch_rows": args.rows, "seed": SEED,
              "parts": sorted(parts), "groups": list(GROUPS), "control_layers": list(CONTROL_LAYERS),
              "teacher_forced": True, "padding": "right, fixed shape"}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()},
                                                        indent=1))
    log(f"inputs ok: {ckpt} {vocab.variant.name} rows={len(examples)} T_pad={t_pad} sha={sha[:12]}")

    model = ar.load_model(ckpt, device=args.device, dtype=args.dtype)
    inputs.update({"dtype": args.dtype, "device": torch.cuda.get_device_name() if
                   args.device.startswith("cuda") else "cpu",
                   "attn_implementation": model.config._attn_implementation,
                   "torch": torch.__version__,
                   "transformers": __import__("transformers").__version__})
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    runner = Runner(model, scorer, tok.pad_token_id, t_pad, args.rows)
    n_layers = model.config.num_hidden_layers

    clean_logits_a = None
    if "A" in parts:
        recs, missing, clean_logits_a = part_a(runner, examples, clean_all, tok, vocab, pool,
                                               args.template, n, log)
        df = pd.DataFrame(recs)
        df[df.condition == "clean"].to_parquet(args.out / "clean.parquet", index=False)
        df[df.condition != "clean"].to_parquet(args.out / "part_a.parquet", index=False)
        pd.DataFrame(missing, columns=["example_id", "row", "k", "recency", "m"]).to_csv(
            args.out / "part_a_no_control.csv", index=False)
        validation["part_a"] = {"records": len(df), "no_control": len(missing),
                                                "no_control_by_m": {int(m): int(c) for m, c in pd.Series(
                                    [x["m"] for x in missing], dtype=int).value_counts().sort_index().items()}}
        del recs, df

    if "B" in parts:
        corr = []
        for i, ex in enumerate(examples):
            k = len(ex.history_item_ids) - 1
            c = pool.draw(ex, k, 0, SEED)
            e2 = P.encode(Hi.replace_history_item(ex, k, c), tok, vocab,
                          template=args.template, with_target=True)
            Hi.check_replacement(clean_all[i], e2, k, 0)
            corr.append(e2)
        recs, ctrl, inv = part_b(runner, examples, clean_all, corr, CONTROLS, n_layers, n, log,
                                 clean_logits_a)
        pd.DataFrame(recs).to_parquet(args.out / "part_b.parquet", index=False)
        cdf = pd.DataFrame(ctrl)
        cdf.to_parquet(args.out / "controls.parquet", index=False)
        validation["controls"] = {
            k: {"n": int(len(g)), "n_exact": int(g.exact.sum()),
                "max_abs_diff_code_logits": float(g.max_abs_diff_code_logits.max())}
            for k, g in cdf.groupby("control")}
        validation["batch_invariance"] = inv
        if "A" in parts:
            a = pd.read_parquet(args.out / "part_a.parquet")
            b = pd.DataFrame(recs)
            a1 = a[(a.recency == 1) & (a.m == 0)].set_index(["example_id", "digit"])["logp_codes"]
            b1 = b[b.condition == "corrupted"].set_index(["example_id", "digit"])["logp_codes"]
            validation["batch_invariance"]["corrupted_logp_vs_part_a"] = float(
                (a1 - b1.reindex(a1.index)).abs().max())
        del recs

    # ---- V6: clean top-1 sanity and bf16 vs fp32 ---------------------------
    ref = pd.read_parquet(args.out / "clean.parquet") if (args.out / "clean.parquet").exists() else \
        pd.read_parquet(args.out / "part_b.parquet").query("condition == 'clean'")
    validation["clean_top1_by_digit"] = ref.groupby("digit")["rank"].apply(
        lambda r: float((r == 0).mean())).to_dict()
    validation["clean_mean_logp_codes_by_digit"] = ref.groupby("digit").logp_codes.mean().to_dict()
    if args.fp32_check_rows and args.dtype != "float32":
        k = min(args.fp32_check_rows, len(examples))
        model.to(torch.float32)
        s32, _, _ = runner.forward([Row(clean_all[i]) for i in range(k)])
        want = ref[ref.example_id.isin([e.example.example_id for e in clean_all[:k]])] \
            .sort_values(["row", "digit"])
        got = s32["logp_codes"].reshape(-1)
        validation["fp32_vs_" + args.dtype] = {
            "rows": k, "max_abs_diff_logp_codes": float(np.abs(got - want.logp_codes.to_numpy()).max()),
            "top1_agreement": float((s32["top1"].reshape(-1) == want.top1.to_numpy()).mean())}
    validation["n_forwards"] = runner.n_forwards
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: {json.dumps({k: validation[k] for k in ('controls', 'batch_invariance') if k in validation}, default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
