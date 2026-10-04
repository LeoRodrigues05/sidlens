#!/usr/bin/env python
"""Retrain next-item Qwen-AR models with the recorded sweep recipe, keep every
checkpoint the analyses need, and accept a model only if it reproduces its
archived evaluation within tolerance (protocol.md).

Why this exists: the 2026-08 next-item sweep ran with a keep-best retention
policy and deleted every other final checkpoint right after evaluation
(`rm -rf final_checkpoint`). Interventions need weights, so the deleted models
are rebuilt here from frozen inputs and byte-identical code.

Traps, each with its guard
--------------------------
1. **Wrong code.** The sweep ran OneDiffRec at eae9ecc; today's OneDiffRec
   `scripts/ar/sft.py` has diverged. Only `vendor/onediffrec` runs, and each
   vendor file it uses must hash to the provenance manifest's value.
2. **Wrong SID table.** Five RQ-KMeans tables have pre-repair `.mispacked`
   twins. Every archived next-item run generated SIDs in the repaired code
   space (checked 2026-10-04), so inputs come from `frozen/`. Every split CSV
   must pass `ar_prompts.check_against_table`, and the test CSV must pass
   `check_against_archive`.
3. **Silent input drift.** Every input file is hashed against the manifest,
   and the manifest's `source` path must equal the path the archived run
   recorded in its `assets.json`.
4. **Losing checkpoints again.** Nothing under `weights/` or `checkpoints/` is
   ever deleted. Trainer `checkpoint-*` directories (optimizer state) are
   removed only after `weights/final` is hashed, load-tested, its SID tokens
   match the index, and the manifest is on disk (README run export contract).
5. **Writing into vendor/.** Vendor code runs with PYTHONDONTWRITEBYTECODE=1,
   a pycache prefix inside the run, and a working directory inside the run.
6. **An evaluation that cannot be compared.** The archived pipeline is
   replayed exactly: `split.py` into shards 0-3, `evaluate.py` per shard (it
   seeds 42 and pins itself to device 0), `merge.py`, `calc.py`. Shard
   composition fixes the sampling RNG streams, so four shards are used on any
   number of GPUs.
7. **Snapshots rotated away.** The trainer keeps only two checkpoints. A
   watcher hard-links `model.safetensors` of selected saves once
   `trainer_state.json` exists (it is written after the weights), well before
   rotation (two saves later) can delete them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from sidlens import paths
from sidlens.data import ar_prompts as P
from sidlens.data.sids import SidVariant

EXP = Path(__file__).resolve().parent
VENDOR = paths.VENDOR / "onediffrec"
CONDA = Path(os.environ.get("ONEDIFFREC_ENV", "/home/leo.rodrigues/onediffrec/OneDiffRec/.conda"))
CATEGORY = "Industrial_and_Scientific"
PROVENANCE = paths.MANIFESTS / "provenance.base-20260826.json"
ARCHIVE = paths.FROZEN_RESULTS / "sweep_metrics" / "next-item" / "metrics"
BASE_MODEL = paths.FROZEN_BASE_MODELS / "qwen2.5-1.5b-instruct"
ORIGINAL_CONTROL = paths.FROZEN_CKPT / "ar" / "next-item_best"   # rqkmeans_3codebook_128
VENDOR_FILES = ("sft.py", "data.py", "evaluate.py", "calc.py", "split.py", "merge.py",
                "LogitProcessor.py")

# The sweep's arguments (vendor/onediffrec/scripts/sweep_runner.sbatch, full mode).
RECIPE = {"sample": -1, "batch_size": 1024, "micro_batch_size": 8, "num_epochs": 10,
          "learning_rate": "3e-4", "cutoff_len": 512, "seed": 42,
          "train_from_scratch": "False", "freeze_LLM": "False", "save_total_limit": 2}
# The sweep's own smoke settings, used by the pilot only.
SMOKE = {**RECIPE, "sample": 64, "num_epochs": 1, "batch_size": 32}
EVAL = {"batch_size": 8, "num_beams": 50, "max_new_tokens": 256, "length_penalty": 0.0}
SHARDS = (0, 1, 2, 3)
SAVE_FRACTION = 0.05                     # sft.py: eval_steps = save_steps = 0.05
KEEP_SAVES = (1, 2, 4, 8, 12, 16, 20)    # 5, 10, 20, 40, 60, 80, 100 % of planned steps
TOL = {"HR@10": 1.5, "NDCG@10": 1.0}     # points; protocol.md "Acceptance"
SID_TOKEN = re.compile(r"^<[a-e]_\d+>$")
DONE = ("accepted", "flagged")


def log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


# ----------------------------------------------------------------- inputs --
def load_configs(task: int | None, names: list[str] | None) -> list[dict]:
    rows = []
    lines = (EXP / "configs.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    for ln in lines[1:]:
        if ln.strip():
            rows.append(dict(zip(head, ln.split("\t"))))
    if names:
        unknown = set(names) - {r["sid_name"] for r in rows}
        if unknown:
            raise ValueError(f"not in configs.tsv: {sorted(unknown)}")
        return [r for r in rows if r["sid_name"] in names]
    return [r for r in rows if int(r["task"]) == task]


def variant_id(sid_name: str) -> str:
    v = SidVariant.parse(sid_name)
    return f"nextitem__{CATEGORY}__{v.quantizer}__{v.n_codebook}cb__{v.codebook_size}"


def manifest_entry(prov: dict, group: str, basename: str) -> dict:
    files = prov["data"][group]["files"]
    if basename not in files:
        raise KeyError(f"{basename} is not in manifest group {group}")
    return files[basename]


def check_vendor(prov: dict) -> dict:
    """Each vendor file used must hash to the manifest's value (trap 1)."""
    tree = prov["vendor"]["trees"]["onediffrec"]["files"]
    out = {}
    for name in VENDOR_FILES:
        rec = tree.get(name)
        got = sha256(VENDOR / name)
        want = rec["sha256"] if isinstance(rec, dict) else rec
        if got != want:
            raise RuntimeError(f"vendor/onediffrec/{name} hash {got} != manifest {want}")
        out[name] = got
    return out


def check_base_model(prov: dict) -> dict:
    files = prov["data"]["base_models/qwen2.5-1.5b-instruct"]["files"]
    out = {}
    for name, rec in sorted(files.items()):
        got = sha256(BASE_MODEL / name)
        if got != rec["sha256"]:
            raise RuntimeError(f"base model {name}: {got} != manifest {rec['sha256']}")
        out[name] = got
    return out


def resolve_inputs(prov: dict, sid_name: str) -> dict:
    """Map the archived run's recorded input paths to frozen copies and prove
    they are the same files (traps 2 and 3)."""
    vid = variant_id(sid_name)
    assets = json.loads((ARCHIVE / f"{vid}.assets.json").read_text())
    if assets.get("status") != "ok":
        raise RuntimeError(f"{vid}: archived assets status {assets.get('status')!r}")
    groups = {"train": "data/splits/next-item/train", "valid": "data/splits/next-item/valid",
              "test": "data/splits/next-item/test", "index": "sids/index_json",
              "info": "sids/info/next-item", "item": "data/item_meta"}
    out = {"variant_id": vid, "sid_name": sid_name, "files": {},
           "archived": {k: assets[k] for k in ("unique_full_sids", "collisions",
                                                "component_tokens", "codebook_utilization")}}
    for key, group in groups.items():
        recorded = assets["paths"][key]
        base = Path(recorded).name
        rec = manifest_entry(prov, group, base)
        frozen = paths.FROZEN / group / base
        if not frozen.is_file():
            raise FileNotFoundError(frozen)
        if rec.get("source") != recorded:
            raise RuntimeError(f"{vid} {key}: manifest source {rec.get('source')} != "
                               f"archived run's path {recorded}")
        got = sha256(frozen)
        if got != rec["sha256"]:
            raise RuntimeError(f"{vid} {key}: {frozen} hash {got} != manifest {rec['sha256']}")
        out["files"][key] = {"path": str(frozen), "sha256": got, "recorded_path": recorded}
    checks = {}
    for split in ("train", "valid", "test"):
        ex = P.load_examples(sid_name, "next-item", split)
        if Path(P.csv_path(sid_name, "next-item", split)) != Path(out["files"][split]["path"]):
            raise RuntimeError(f"{vid}: ar_prompts reads a different {split} CSV")
        res = P.check_against_table(ex)
        if not res["ok"]:
            raise RuntimeError(f"{vid} {split}: check_against_table failed: {res}")
        checks[f"table_{split}"] = {"n_rows": len(ex), **res}
        if split == "test":
            arc = P.check_against_archive(ex)
            if not arc["ok"]:
                raise RuntimeError(f"{vid}: check_against_archive failed: {arc}")
            checks["archive_test"] = arc
    out["checks"] = checks
    return out


# ------------------------------------------------------------ subprocess --
def vendor_env(run_dir: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(VENDOR),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(run_dir / ".pycache"),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(run_dir / "logs"),
        "WANDB_SILENT": "true",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_DATASETS_CACHE": str(run_dir / ".hf_datasets"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def run_logged(cmd: list[str], log_path: Path, env: dict, cwd: Path) -> None:
    log(f"run: {' '.join(map(str, cmd[:4]))} ... > {log_path.name}")
    with open(log_path, "ab") as f:
        f.write(f"\n### {dt.datetime.now().isoformat()} {' '.join(map(str, cmd))}\n".encode())
        f.flush()
        proc = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                              env=env, cwd=cwd)
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-30:]
        raise RuntimeError(f"command failed rc={proc.returncode}: {cmd[:3]}\n" + "\n".join(tail))


# -------------------------------------------------------------- snapshots --
class Snapshotter(threading.Thread):
    """Hard-link selected trainer saves before rotation deletes them (trap 7)."""

    def __init__(self, train_dir: Path, dest_root: Path, keep: tuple[int, ...]):
        super().__init__(daemon=True)
        self.train_dir, self.dest_root, self.keep = train_dir, dest_root, set(keep)
        self.seen: dict[str, dict] = {}
        self.stop = threading.Event()
        self.errors: list[str] = []

    def poll(self) -> None:
        if not self.train_dir.is_dir():
            return
        for ck in sorted(self.train_dir.glob("checkpoint-*")):
            if ck.name in self.seen or not re.fullmatch(r"checkpoint-\d+", ck.name):
                continue
            state_path = ck / "trainer_state.json"
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text())
            except json.JSONDecodeError:
                continue          # still being written; next poll
            step, max_steps = int(state["global_step"]), int(state["max_steps"])
            save_every = math.ceil(SAVE_FRACTION * max_steps)
            index = round(step / save_every)
            rec = {"checkpoint": ck.name, "global_step": step, "max_steps": max_steps,
                   "save_index": index, "kept": index in self.keep}
            if rec["kept"]:
                try:
                    dest = self.dest_root / f"step-{step:05d}"
                    dest.mkdir(parents=True)
                    weights = ck / "model.safetensors"
                    os.link(weights, dest / "model.safetensors")
                    for small in ("config.json", "generation_config.json", "trainer_state.json"):
                        if (ck / small).is_file():
                            shutil.copy2(ck / small, dest / small)
                    rec["weights_sha256"] = sha256(dest / "model.safetensors")
                    write_json(dest / "snapshot.json", rec)
                except Exception as e:      # recorded, and the export step fails on it
                    self.errors.append(f"{ck.name}: {e!r}")
            self.seen[ck.name] = rec

    def run(self) -> None:
        while not self.stop.is_set():
            self.poll()
            self.stop.wait(10)
        self.poll()


# ------------------------------------------------------------------ train --
def train(inp: dict, run_dir: Path, recipe: dict, ngpu: int, keep: tuple[int, ...]) -> dict:
    f = inp["files"]
    train_dir = run_dir / "train"
    cmd = [CONDA / "bin" / "torchrun", "--standalone", f"--nproc_per_node={ngpu}",
           VENDOR / "sft.py",
           "--base_model", BASE_MODEL, "--train_file", f["train"]["path"],
           "--eval_file", f["valid"]["path"], "--output_dir", train_dir,
           "--sample", recipe["sample"], "--batch_size", recipe["batch_size"],
           "--micro_batch_size", recipe["micro_batch_size"], "--num_epochs", recipe["num_epochs"],
           "--learning_rate", recipe["learning_rate"], "--cutoff_len", recipe["cutoff_len"],
           "--category", CATEGORY, "--seed", recipe["seed"],
           "--sid_index_path", f["index"]["path"], "--item_meta_path", f["item"]["path"],
           "--train_from_scratch", recipe["train_from_scratch"],
           "--freeze_LLM", recipe["freeze_LLM"], "--save_total_limit", recipe["save_total_limit"],
           "--wandb_project", "SidLens_ar_retrain", "--wandb_run_name", inp["variant_id"]]
    snap = Snapshotter(train_dir, run_dir / "checkpoints", keep)
    snap.start()
    t0 = time.time()
    try:
        run_logged(cmd, run_dir / "logs" / "train.log", vendor_env(run_dir), run_dir / "work")
    finally:
        snap.stop.set()
        snap.join()
    seconds = time.time() - t0
    state_files = sorted(train_dir.glob("checkpoint-*/trainer_state.json"),
                         key=lambda p: int(p.parent.name.split("-")[1]))
    if not state_files:
        raise RuntimeError("training left no trainer_state.json")
    state = json.loads(state_files[-1].read_text())
    shutil.copy2(state_files[-1], run_dir / "logs" / "trainer_state.json")
    evals = [{"step": h["step"], "epoch": h.get("epoch"), "eval_loss": h["eval_loss"]}
             for h in state["log_history"] if "eval_loss" in h]
    best = state.get("best_model_checkpoint") or ""
    return {"train_seconds": round(seconds, 1), "ngpu": ngpu, "recipe": recipe,
            "max_steps": state["max_steps"], "stopped_at_step": state["global_step"],
            "stopped_at_epoch": state.get("epoch"), "best_eval_loss": state.get("best_metric"),
            "best_step": int(best.rsplit("-", 1)[1]) if best else None,
            "eval_loss_curve": evals, "snapshots": list(snap.seen.values()),
            "snapshot_errors": snap.errors}


# ----------------------------------------------------------------- export --
LOAD_TEST = r"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ck = sys.argv[1]
tok = AutoTokenizer.from_pretrained(ck)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained(ck, torch_dtype=torch.bfloat16).to(dev).eval()
ids = tok("### Response:\n", return_tensors="pt").to(dev)
with torch.no_grad():
    logits = m(**ids).logits
print("LOADTEST " + json.dumps({"n_tokenizer": len(tok),
    "n_embedding_rows": int(m.get_input_embeddings().weight.shape[0]),
    "finite": bool(torch.isfinite(logits).all()), "dtype": str(m.dtype), "device": dev}))
"""


def index_tokens(index_path: str) -> set[str]:
    idx = json.loads(Path(index_path).read_text())
    return {t for codes in idx.values() for t in codes}


def export(inp: dict, run_dir: Path, snapshot_errors: list[str]) -> dict:
    src = run_dir / "train" / "final_checkpoint"
    dest = run_dir / "weights" / "final"
    if not (src / "config.json").is_file():
        raise RuntimeError(f"final checkpoint missing: {src}")
    if snapshot_errors:
        raise RuntimeError(f"snapshots failed: {snapshot_errors}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)
    files = {p.name: {"sha256": sha256(p), "size": p.stat().st_size}
             for p in sorted(dest.iterdir()) if p.is_file()}
    if "model.safetensors" not in files:
        raise RuntimeError(f"no model.safetensors in {dest}: {sorted(files)}")
    added = json.loads((dest / "added_tokens.json").read_text())
    sid_tokens = {t for t in added if SID_TOKEN.match(t)}
    want = index_tokens(inp["files"]["index"]["path"])
    if sid_tokens != want:
        raise RuntimeError(f"SID tokens differ from the index: {len(sid_tokens)} vs {len(want)}")
    if len(want) != inp["archived"]["component_tokens"]:
        raise RuntimeError(f"index has {len(want)} tokens, archived run had "
                           f"{inp['archived']['component_tokens']}")
    out = subprocess.run([str(CONDA / "bin" / "python"), "-c", LOAD_TEST, str(dest)],
                         capture_output=True, text=True, env=vendor_env(run_dir))
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("LOADTEST ")]
    if out.returncode != 0 or not line:
        raise RuntimeError(f"load test failed rc={out.returncode}: {out.stderr[-2000:]}")
    lt = json.loads(line[0][len("LOADTEST "):])
    if not lt["finite"] or lt["n_embedding_rows"] != lt["n_tokenizer"]:
        raise RuntimeError(f"load test: {lt}")
    return {"path": str(dest), "files": files, "n_sid_tokens": len(sid_tokens), "load_test": lt}


def prune_trainer_state(run_dir: Path) -> list[str]:
    """Only after a successful export (trap 4)."""
    removed = []
    train_dir = run_dir / "train"
    for ck in sorted(train_dir.glob("checkpoint-*")):
        shutil.rmtree(ck)
        removed.append(ck.name)
    dup = train_dir / "model.safetensors"
    if dup.is_file():
        dup.unlink()
        removed.append("model.safetensors (top-level duplicate)")
    return removed


# --------------------------------------------------------------- evaluate --
CALC_ARRAY = re.compile(r"^(NDCG|HR):?\s*\[([^\]]*)\]", re.MULTILINE)


def parse_calc(text: str) -> dict:
    """calc.py prints the cutoffs as a list, then numpy arrays that numpy may
    wrap over several lines; [^\\]]* spans the line breaks."""
    topk = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("[") and s.endswith("]") and "," in s:
            try:
                cand = json.loads(s)
            except json.JSONDecodeError:
                continue
            if cand and all(isinstance(x, int) for x in cand):
                topk = cand
                break
    vals = {m.group(1): [float(x) for x in m.group(2).split()] for m in CALC_ARRAY.finditer(text)}
    if topk is None or set(vals) != {"NDCG", "HR"} or any(len(v) != len(topk) for v in vals.values()):
        raise RuntimeError(f"cannot parse calc.py output:\n{text[-2000:]}")
    out = {}
    for name in ("HR", "NDCG"):
        for k, v in zip(topk, vals[name]):
            out[f"{name}@{k}"] = round(100 * v, 4)
    return out


def calc(pred_json: Path, info: str, run_dir: Path) -> dict:
    out = subprocess.run([str(CONDA / "bin" / "python"), str(VENDOR / "calc.py"),
                          "--path", str(pred_json), "--item_path", info],
                         capture_output=True, text=True, env=vendor_env(run_dir), cwd=run_dir / "work")
    if out.returncode != 0:
        raise RuntimeError(f"calc.py failed: {out.stderr[-2000:]}")
    return parse_calc(out.stdout)


def evaluate(ckpt: Path, inp: dict, run_dir: Path, limit: int | None) -> dict:
    ev = run_dir / "eval"
    shards = ev / "shards"
    shards.mkdir(parents=True)
    test = Path(inp["files"]["test"]["path"])
    if limit:
        import pandas as pd
        sub = ev / "test_subset.csv"
        pd.read_csv(test, dtype=str, keep_default_na=False).head(limit).to_csv(sub, index=False)
        test = sub
    env = vendor_env(run_dir)
    py = CONDA / "bin" / "python"
    t0 = time.time()
    run_logged([py, VENDOR / "split.py", "--input_path", test, "--output_path", shards,
                "--cuda_list", ",".join(map(str, SHARDS))], ev / "split.log", env, run_dir / "work")
    procs = []
    for i in SHARDS:
        cmd = [py, "-u", VENDOR / "evaluate.py", "--base_model", ckpt,
               "--info_file", inp["files"]["info"]["path"], "--category", CATEGORY,
               "--test_data_path", shards / f"{i}.csv", "--result_json_data", shards / f"{i}.json",
               "--batch_size", EVAL["batch_size"], "--num_beams", EVAL["num_beams"],
               "--max_new_tokens", EVAL["max_new_tokens"], "--length_penalty", EVAL["length_penalty"]]
        fh = open(shards / f"{i}.log", "wb")
        procs.append((i, subprocess.Popen([str(c) for c in cmd], stdout=fh,
                                          stderr=subprocess.STDOUT, env=env, cwd=run_dir / "work"), fh))
    failed = []
    for i, p, fh in procs:
        if p.wait() != 0:
            failed.append(i)
        fh.close()
    if failed:
        raise RuntimeError(f"eval shards failed: {failed}; see {shards}")
    pred = ev / "predictions.json"
    run_logged([py, VENDOR / "merge.py", "--input_path", shards, "--output_path", pred,
                "--cuda_list", ",".join(map(str, SHARDS))], ev / "merge.log", env, run_dir / "work")
    metrics = calc(pred, inp["files"]["info"]["path"], run_dir)
    return {"predictions": str(pred), "predictions_sha256": sha256(pred), "metrics": metrics,
            "n_rows": len(json.loads(pred.read_text())), "limit": limit,
            "eval_seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------- accept --
def lists(pred_json: Path) -> list[list[str]]:
    return [[s.strip("\"\n").strip() for s in r["predict"]] for r in json.loads(pred_json.read_text())]


def targets(pred_json: Path) -> list[str]:
    out = []
    for r in json.loads(pred_json.read_text()):
        o = r["output"][0] if isinstance(r["output"], list) else r["output"]
        out.append(o.strip(" \n\""))
    return out


def exact_hr(ls: list[list[str]], tg: list[str], k: int = 10) -> float:
    return round(100 * sum(t in l[:k] for l, t in zip(ls, tg)) / len(tg), 4)


def agreement(a: list[list[str]], b: list[list[str]]) -> dict:
    n = min(len(a), len(b))
    same_list = sum(a[i] == b[i] for i in range(n))
    top1 = sum(bool(a[i]) and bool(b[i]) and a[i][0] == b[i][0] for i in range(n))
    overlap = sum(len(set(a[i][:10]) & set(b[i][:10])) / 10 for i in range(n))
    return {"n": n, "identical_lists": same_list, "top1_agree": round(top1 / n, 4),
            "mean_top10_overlap": round(overlap / n, 4)}


def accept(inp: dict, ev: dict, run_dir: Path, train_info: dict | None, role: str) -> dict:
    vid = inp["variant_id"]
    archived_pred = ARCHIVE / f"{vid}.predictions.json"
    record = json.loads((ARCHIVE / f"{vid}.json").read_text())
    pred = Path(ev["predictions"])
    new_ls, arc_ls = lists(pred), lists(archived_pred)
    tg = targets(pred)
    if ev["limit"]:
        arc_ls = arc_ls[:ev["limit"]]
    arc_calc = calc(archived_pred, inp["files"]["info"]["path"], run_dir) if not ev["limit"] else None
    out = {"archived_record_metrics": record.get("metrics"),
           "archived_calc_metrics": arc_calc, "new_calc_metrics": ev["metrics"],
           "exact_sid_hr10": {"new": exact_hr(new_ls, tg),
                              "archived": exact_hr(arc_ls, targets(archived_pred)[:len(arc_ls)])},
           "agreement_with_archive": agreement(new_ls, arc_ls)}
    if train_info:
        out["training_vs_record"] = {
            "best_eval_loss": [train_info["best_eval_loss"], record.get("best_eval_loss")],
            "stopped_at_step": [train_info["stopped_at_step"], record.get("stopped_at_step")],
            "max_steps": [train_info["max_steps"], record.get("max_steps")]}
    if ev["limit"] is None:
        d = {k: round(ev["metrics"][k] - arc_calc[k], 4) for k in TOL}
        out["delta"] = d
        out["verdict"] = "accepted" if all(abs(d[k]) <= TOL[k] for k in TOL) else "flagged"
    else:
        out["verdict"] = "not_judged_subset"
    if role == "control" and ev["limit"] is None:
        orig = run_dir.parent / "_control_original" / "eval" / "predictions.json"
        if orig.is_file():
            out["agreement_with_original_weights"] = agreement(new_ls, lists(orig))
    return out


# ---------------------------------------------------------------- drivers --
def run_variant(row: dict, root: Path, args, prov: dict, code: dict) -> str:
    sid_name, role = row["sid_name"], row["role"]
    vid = variant_id(sid_name)
    run_dir = root / vid
    if run_dir.exists():
        man = run_dir / "manifest.json"
        status = json.loads(man.read_text()).get("status") if man.is_file() else None
        if args.resume and status in DONE:
            log(f"{vid}: already {status}; skipped (--resume)")
            return status
        raise RuntimeError(f"{run_dir} exists with status {status!r}; inspect it before rerunning")
    run_dir.mkdir()
    for sub in ("logs", "work", "checkpoints"):
        (run_dir / sub).mkdir()
    manifest = {"variant_id": vid, "sid_name": sid_name, "role": role, "mode": args.mode,
                "status": "running", "started": dt.datetime.now().isoformat(),
                "slurm_job": os.environ.get("SLURM_JOB_ID"), "node": os.uname().nodename,
                "initial_weights": {"path": str(BASE_MODEL), "sha256": code["base_model"]},
                "vendor_sha256": code["vendor"], "env_python": str(CONDA / "bin" / "python")}
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "status.txt").write_text(f"running started={manifest['started']}\n")
    try:
        log(f"{vid}: inputs")
        inp = resolve_inputs(prov, sid_name)
        write_json(run_dir / "inputs.json", inp)
        recipe = SMOKE if args.mode == "pilot" else RECIPE
        log(f"{vid}: train ({'smoke' if args.mode == 'pilot' else 'full'} recipe, {args.ngpu} GPU)")
        tinfo = train(inp, run_dir, recipe, args.ngpu, KEEP_SAVES)
        manifest["training"] = tinfo
        log(f"{vid}: export")
        manifest["export"] = export(inp, run_dir, tinfo["snapshot_errors"])
        manifest["status"] = "exported"
        write_json(run_dir / "manifest.json", manifest)
        manifest["pruned"] = prune_trainer_state(run_dir)
        log(f"{vid}: evaluate")
        limit = 64 if args.mode == "pilot" else None
        ev = evaluate(run_dir / "weights" / "final", inp, run_dir, limit)
        manifest["evaluation"] = ev
        manifest["acceptance"] = accept(inp, ev, run_dir, tinfo, role)
        manifest["status"] = manifest["acceptance"]["verdict"] if args.mode == "full" else "pilot_ok"
    except Exception as e:
        manifest["status"] = "failed"
        manifest["error"] = repr(e)
        manifest["traceback"] = traceback.format_exc()
        log(f"{vid}: FAILED {e!r}")
    manifest["finished"] = dt.datetime.now().isoformat()
    write_json(run_dir / "manifest.json", manifest)
    (run_dir / "status.txt").write_text(f"{manifest['status']} finished={manifest['finished']}\n")
    return manifest["status"]


def pilot_eval_reproduction(root: Path, prov: dict) -> dict:
    """Pilot (A): the original weights through this pipeline must reproduce
    the archived lists (protocol: >= 3,680 of 3,681 identical)."""
    run_dir = root / "_control_original"
    run_dir.mkdir()
    (run_dir / "work").mkdir()
    (run_dir / "logs").mkdir()
    inp = resolve_inputs(prov, "rqkmeans_3codebook_128")
    ev = evaluate(ORIGINAL_CONTROL, inp, run_dir, None)
    arc = ARCHIVE / f"{inp['variant_id']}.predictions.json"
    agr = agreement(lists(Path(ev["predictions"])), lists(arc))
    res = {"evaluation": ev, "agreement_with_archive": agr,
           "archived_calc_metrics": calc(arc, inp["files"]["info"]["path"], run_dir),
           "pass": agr["identical_lists"] >= 3680}
    write_json(run_dir / "result.json", res)
    log(f"pilot A: identical lists {agr['identical_lists']}/{agr['n']}; pass={res['pass']}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", type=int, help="array task: run this task's configs")
    ap.add_argument("--variants", help="comma-separated sid names instead of --task")
    ap.add_argument("--mode", choices=("pilot", "full"), required=True)
    ap.add_argument("--root", type=Path, required=True, help="run-group directory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="resolve and check inputs only")
    args = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    args.ngpu = len([d for d in vis.split(",") if d.strip()]) if vis else 0
    prov = json.loads(PROVENANCE.read_text())
    if args.dry_run:
        if args.task is None and not args.variants:
            all_names = [ln.split("\t")[1] for ln in (EXP / "configs.tsv").read_text().splitlines()[1:] if ln]
            rows = load_configs(None, all_names)
        else:
            rows = load_configs(args.task, args.variants.split(",") if args.variants else None)
        code = {"vendor": check_vendor(prov)}
        for r in rows:
            inp = resolve_inputs(prov, r["sid_name"])
            log(f"{inp['variant_id']}: inputs ok; rows "
                + ", ".join(f"{k}={v['n_rows']}" for k, v in inp["checks"].items() if "n_rows" in v))
        log(f"dry run ok: {len(rows)} configs; vendor files {sorted(code['vendor'])}")
        return 0
    if args.ngpu < 1:
        raise SystemExit("no GPU visible (CUDA_VISIBLE_DEVICES empty)")
    args.root.mkdir(parents=True, exist_ok=True)
    code = {"vendor": check_vendor(prov), "base_model": check_base_model(prov)}
    if args.mode == "pilot":
        res_a = pilot_eval_reproduction(args.root, prov)
        status = run_variant({"sid_name": "rqkmeans_3codebook_128", "role": "pilot_smoke"},
                             args.root, args, prov, code)
        ok = res_a["pass"] and status == "pilot_ok"
        log(f"pilot: A pass={res_a['pass']}, B status={status} -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    rows = load_configs(args.task, args.variants.split(",") if args.variants else None)
    if not rows:
        raise SystemExit(f"no configs for task {args.task}")
    if any(r["role"] == "control" for r in rows) and not (args.root / "_control_original").exists():
        # The control is compared with the surviving original weights; evaluate them here.
        pilot_eval_reproduction(args.root, prov)
    results = {}
    for row in rows:
        results[row["sid_name"]] = run_variant(row, args.root, args, prov, code)
        log(f"progress: {results}")
    return 0 if all(s in DONE for s in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
