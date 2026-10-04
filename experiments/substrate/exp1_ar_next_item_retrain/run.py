#!/usr/bin/env python
"""Retrain Qwen-AR next-item models with the recorded August-2026 sweep recipe.

See protocol.md. One call processes a list of variants in order. Per variant:
preflight (hash every input against the freeze manifest) -> train (vendor
sft.py under torchrun) -> retain (move and hash the final weights before
anything else) -> evaluate (the archived split / evaluate / merge / calc
procedure) -> manifest. `status.json` is written after every step, so a
resubmitted job resumes at the first incomplete step.

Traps this guards against
-------------------------
* **Losing weights again.** Upstream's keep-best retention deleted 25 of 27
  next-item models. Here nothing deletes a model file except two duplicates,
  and each only after its sha256 matches the retained copy (`_retain`).
* **Training on the wrong SID table.** Five RQ-KMeans tables were repaired on
  2026-08-20 and pre-repair copies exist. Inputs come from `frozen/` only, are
  hashed against `provenance.base-20260826.json`, and their basenames must
  equal the paths recorded in each archived run's `assets.json`.
* **Writing into vendor/.** The recipe runs from `vendor/onediffrec`, which
  `verify` hashes. `PYTHONDONTWRITEBYTECODE=1` and a cwd outside vendor keep
  the tree byte-identical.
* **A GPU-count-dependent evaluation.** The archived lists came from 4 shards,
  each seeded 42 inside `evaluate.py`. The 4-shard split is kept for any GPU
  count; with fewer GPUs the shards run in sequence.
* **A silent partial evaluation.** A missing or failed shard stops the
  variant; merge never runs on fewer than 4 shard outputs.
* **Running out of wall time mid-variant.** A variant starts only if the
  remaining budget covers its expected time. Training gets a timeout, and an
  interrupted attempt resumes from its newest trainer checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from sidlens import paths
from sidlens.provenance.hashing import sha256_file

CATEGORY = "Industrial_and_Scientific"
VENDOR = paths.VENDOR / "onediffrec"
ORDER = [  # priority order; the sbatch wrapper deals these round-robin to tasks
    "rqkmeans_3codebook_128",  # calibration cell: original weights survive
    "rqvae_4codebook_128",     # surviving one-off is not the sweep model
    "MQ_3codebook_128", "rqkmeans_4codebook_128", "rqvae_3codebook_128",
    "MQ_4codebook_128", "rqkmeans_5codebook_128", "rqvae_5codebook_128",
    "MQ_5codebook_128",
    "rqkmeans_3codebook_512", "rqvae_3codebook_512", "MQ_3codebook_512",
    "rqkmeans_4codebook_512", "rqvae_4codebook_512", "MQ_4codebook_512",
    "rqkmeans_5codebook_512", "rqvae_5codebook_512", "MQ_5codebook_512",
]
ZCR_SUFFIX = "__zcr"   # arm B: e.g. rqkmeans_3codebook_128__zcr, inputs from build_zcr_inputs.py
FULL = dict(sample=-1, epochs=10, batch=1024, micro=8)
SMOKE = dict(sample=64, epochs=1, batch=32, micro=8)       # the runner's smoke mode
EVAL = dict(num_beams=50, batch_size=8, max_new_tokens=256, length_penalty=0.0)
N_SHARDS = 4
TRAIN_SECONDS_4GPU = 4500   # slowest recorded sweep training (rqvae 5x128: 4,469 s)
EVAL_SECONDS = 1800


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def parse_variant(name: str) -> tuple[str, int, int]:
    method, rest = name.split("_", 1)
    cb, sz = rest.split("codebook_")
    return method, int(cb), int(sz)


def frozen_inputs(name: str) -> dict[str, tuple[Path, str, str]]:
    """Map a variant to its frozen inputs: role -> (path, manifest dir, file name)."""
    method, cb, sz = parse_variant(name)
    stem = f"{CATEGORY}_{method}_{cb}codebook_{sz}"
    index = {"MQ": f"{CATEGORY}.index.MQ.{cb}codebook_{sz}.json",
             "rqvae": f"{CATEGORY}.index_{cb}codebook_{sz}.json",
             "rqkmeans": f"{CATEGORY}.rqkmeans.index_{cb}codebook_{sz}.json"}[method]
    roles = {
        "train": ("data/splits/next-item/train", f"{stem}.csv"),
        "valid": ("data/splits/next-item/valid", f"{stem}.csv"),
        "test": ("data/splits/next-item/test", f"{stem}.csv"),
        "index": ("sids/index_json", index),
        "info": ("sids/info/next-item", f"{stem}.txt"),
        "item": ("data/item_meta", f"{CATEGORY}.item.json"),
    }
    return {r: (paths.FROZEN / d / f, d, f) for r, (d, f) in roles.items()}


def zcr_inputs(name: str, zcr_root: Path) -> dict[str, tuple[Path, str]]:
    """Arm B: role -> (path, expected sha256) from build_zcr_inputs.py's manifest."""
    base = name.removesuffix(ZCR_SUFFIX)
    man = json.loads((zcr_root / base / "manifest.json").read_text())
    if man["variant"] != base:
        raise RuntimeError(f"{zcr_root / base}: manifest is for {man['variant']}")
    out = {r: (Path(man["files"][r]["path"]), man["files"][r]["sha256"])
           for r in ("train", "valid", "test", "index", "info")}
    p, d, f = frozen_inputs(base)["item"]
    out["item"] = (p, None)  # checked against the freeze manifest below
    return out


def preflight(name: str, base_model: Path, zcr_root: Path | None = None) -> dict:
    """Hash every input against its manifest; check native names against assets.json."""
    manifest = json.loads((paths.MANIFESTS / "provenance.base-20260826.json").read_text())["data"]
    out: dict = {"inputs": {}, "base_model": {}}
    if name.endswith(ZCR_SUFFIX):
        if zcr_root is None:
            raise RuntimeError(f"{name}: pass --zcr-inputs")
        frozen_item = frozen_inputs(name.removesuffix(ZCR_SUFFIX))["item"]
        for role, (p, expected) in zcr_inputs(name, zcr_root).items():
            if expected is None:
                expected = manifest[frozen_item[1]]["files"][frozen_item[2]]["sha256"]
            got = sha256_file(p)
            if got != expected:
                raise RuntimeError(f"{name}: {role} {p} sha256 {got} != {expected}")
            out["inputs"][role] = {"path": str(p), "sha256": got}
        out["zcr_inputs_manifest"] = str(zcr_root / name.removesuffix(ZCR_SUFFIX) / "manifest.json")
        _check_base_model(manifest, base_model, out)
        return out
    for role, (p, d, f) in frozen_inputs(name).items():
        expected = manifest[d]["files"][f]["sha256"]
        got = sha256_file(p)
        if got != expected:
            raise RuntimeError(f"{name}: {role} {p} sha256 {got} != frozen manifest {expected}")
        out["inputs"][role] = {"path": str(p), "sha256": got}
    method, cb, sz = parse_variant(name)
    assets = paths.FROZEN_RESULTS / "sweep_metrics/next-item/metrics" / \
        f"nextitem__{CATEGORY}__{method}__{cb}cb__{sz}.assets.json"
    recorded = json.loads(assets.read_text())["paths"]
    for role in ("train", "valid", "test", "index", "info", "item"):
        if Path(recorded[role]).name != Path(out["inputs"][role]["path"]).name:
            raise RuntimeError(f"{name}: {role} basename differs from the archived run's "
                               f"assets.json ({recorded[role]})")
    _check_base_model(manifest, base_model, out)
    return out


def _check_base_model(manifest: dict, base_model: Path, out: dict) -> None:
    for f, meta in manifest["base_models/qwen2.5-1.5b-instruct"]["files"].items():
        got = sha256_file(base_model / f)
        if got != meta["sha256"]:
            raise RuntimeError(f"base model {f}: sha256 {got} != frozen manifest {meta['sha256']}")
        out["base_model"][f] = got


def read_status(vdir: Path) -> dict:
    p = vdir / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"step": "new"}


def write_status(vdir: Path, step: str, **extra) -> None:
    st = read_status(vdir)
    st.update(step=step, updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
    tmp = vdir / "status.json.tmp"
    tmp.write_text(json.dumps(st, indent=2))
    tmp.replace(vdir / "status.json")


def hash_dir(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.rglob("*")) if p.is_file()}


def latest_trainer_checkpoint(vdir: Path) -> Path | None:
    """Newest resumable checkpoint across earlier attempts of this variant."""
    best, best_step = None, -1
    for ck in vdir.glob("attempts/*/train/checkpoint-*"):
        if (ck / "trainer_state.json").exists() and (ck / "optimizer.pt").exists():
            step = int(ck.name.split("-")[-1])
            if step > best_step:
                best, best_step = ck, step
    return best


def train_env(attempt: Path) -> dict:
    env = dict(os.environ)
    env.update({
        "NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "WARN", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "offline", "WANDB_DIR": str(attempt / "wandb"),
        "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
    })
    (attempt / "wandb").mkdir(exist_ok=True)
    return env


def train(name: str, vdir: Path, attempt: Path, inputs: dict, base_model: Path,
          gpus: int, hp: dict, timeout_s: int) -> None:
    out = attempt / "train"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           f"--nproc_per_node={gpus}", str(VENDOR / "sft.py"),
           "--base_model", str(base_model),
           "--train_file", inputs["train"]["path"], "--eval_file", inputs["valid"]["path"],
           "--output_dir", str(out), "--sample", str(hp["sample"]),
           "--batch_size", str(hp["batch"]), "--micro_batch_size", str(hp["micro"]),
           "--num_epochs", str(hp["epochs"]), "--learning_rate", "3e-4", "--cutoff_len", "512",
           "--category", CATEGORY, "--seed", "42",
           "--sid_index_path", inputs["index"]["path"], "--item_meta_path", inputs["item"]["path"],
           "--train_from_scratch", "False", "--freeze_LLM", "False", "--save_total_limit", "2",
           "--wandb_project", "SidLens_ar_next_item_retrain", "--wandb_run_name", name]
    resume = latest_trainer_checkpoint(vdir)
    if resume is not None:
        cmd += ["--resume_from_checkpoint", str(resume)]
        log(f"  resuming from {resume}")
    (attempt / "train_command.json").write_text(json.dumps(cmd, indent=1))
    log(f"  training on {gpus} GPU(s), grad accumulation {hp['batch'] // hp['micro'] // gpus}, "
        f"timeout {timeout_s}s")
    t0 = time.time()
    with open(attempt / "train.log", "w") as fh:
        # Own process group: on timeout the torchrun launcher AND its workers must
        # die, or orphaned ranks keep the GPUs busy into the next variant.
        proc = subprocess.Popen(cmd, cwd=attempt, env=train_env(attempt), stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            (attempt / "train_seconds").write_text(f"{int(time.time() - t0)}\n")
            raise
    seconds = int(time.time() - t0)
    (attempt / "train_seconds").write_text(f"{seconds}\n")
    if rc != 0:
        raise RuntimeError(f"{name}: training exited {rc} after {seconds}s; see {attempt/'train.log'}")
    if not (out / "final_checkpoint" / "config.json").exists():
        raise RuntimeError(f"{name}: training returned 0 but final_checkpoint is missing")
    write_status(vdir, "trained_unretained", train_seconds=seconds, train_attempt=str(attempt))


def _retain(name: str, vdir: Path, attempt: Path) -> dict:
    """Move and hash the final weights first; delete only verified duplicates."""
    out = attempt / "train"
    final_src = out / "final_checkpoint"
    final_dst = vdir / "weights" / "final"
    if final_dst.exists():
        raise RuntimeError(f"{name}: {final_dst} already exists; refusing to overwrite weights")
    final_dst.parent.mkdir(parents=True, exist_ok=True)
    final_src.rename(final_dst)
    weights_sha = hash_dir(final_dst)
    (vdir / "weights" / "final.sha256.json").write_text(json.dumps(weights_sha, indent=1))
    log(f"  retained final weights: {final_dst} ({len(weights_sha)} files hashed)")

    dup = out / "model.safetensors"
    if dup.exists():
        if sha256_file(dup) == weights_sha.get("model.safetensors"):
            dup.unlink()
        else:
            (vdir / "weights" / "top_level_copy_differs.txt").write_text(
                f"{dup} differs from weights/final/model.safetensors; kept\n")
    ck_root = vdir / "checkpoints"
    ck_root.mkdir(exist_ok=True)
    kept = []
    for ck in sorted(out.glob("checkpoint-*")):
        for f in ("optimizer.pt", "scheduler.pt"):
            (ck / f).unlink(missing_ok=True)
        for f in ck.glob("rng_state*.pth"):
            f.unlink()
        dst = ck_root / ck.name
        if dst.exists():
            dst = ck_root / f"{ck.name}-{attempt.name}"
        ck.rename(dst)
        kept.append(dst.name)
    (vdir / "checkpoints" / "checkpoints.sha256.json").write_text(
        json.dumps({k: hash_dir(ck_root / k) for k in kept}, indent=1))
    write_status(vdir, "trained", weights=str(final_dst), trainer_checkpoints=kept)
    return weights_sha


def evaluate(name: str, vdir: Path, inputs: dict, gpus: int, eval_limit: int | None) -> dict:
    work = vdir / "eval" / "work"
    if work.exists():
        shutil.rmtree(work)          # scratch from an interrupted evaluation only
    shards = work / "shards"
    shards.mkdir(parents=True)
    test = Path(inputs["test"]["path"])
    if eval_limit:
        import pandas as pd
        lim = work / "test_limited.csv"
        pd.read_csv(test).head(eval_limit).to_csv(lim, index=False)
        test = lim
    env = train_env(work)
    py = sys.executable
    subprocess.run([py, str(VENDOR / "split.py"), "--input_path", str(test),
                    "--output_path", str(shards), "--cuda_list", "0,1,2,3"],
                   cwd=work, env=env, check=True)
    weights = vdir / "weights" / "final"
    t0 = time.time()
    # The job's own device list (Slurm may expose physical ids such as "2,3").
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d] \
        or [str(i) for i in range(gpus)]
    free = visible[:gpus]
    pending = list(range(N_SHARDS))
    running: list[tuple[int, subprocess.Popen, object, str]] = []
    while pending or running:
        while pending and free:
            i = pending.pop(0)
            dev = free.pop(0)
            fh = open(shards / f"{i}.log", "w")
            e = dict(env, CUDA_VISIBLE_DEVICES=dev)
            p = subprocess.Popen(
                [py, "-u", str(VENDOR / "evaluate.py"), "--base_model", str(weights),
                 "--info_file", inputs["info"]["path"], "--category", CATEGORY,
                 "--test_data_path", str(shards / f"{i}.csv"),
                 "--result_json_data", str(shards / f"{i}.json"),
                 "--batch_size", str(EVAL["batch_size"]), "--num_beams", str(EVAL["num_beams"]),
                 "--max_new_tokens", str(EVAL["max_new_tokens"]),
                 "--length_penalty", str(EVAL["length_penalty"])],
                cwd=work, env=e, stdout=fh, stderr=subprocess.STDOUT)
            running.append((i, p, fh, dev))
        time.sleep(5)
        still = []
        for i, p, fh, dev in running:
            rc = p.poll()
            if rc is None:
                still.append((i, p, fh, dev))
                continue
            fh.close()
            if rc != 0:
                for _, q, _, _ in still + [r for r in running if r[0] != i]:
                    if q.poll() is None:
                        q.kill()
                raise RuntimeError(f"{name}: eval shard {i} exited {rc}; see {shards / f'{i}.log'}")
            free.append(dev)
        running = still
    missing = [i for i in range(N_SHARDS) if not (shards / f"{i}.json").exists()]
    if missing:
        raise RuntimeError(f"{name}: eval shards {missing} produced no output")
    eval_dir = vdir / "eval"
    pred = eval_dir / "predictions.json"
    subprocess.run([py, str(VENDOR / "merge.py"), "--input_path", str(shards),
                    "--output_path", str(pred), "--cuda_list", "0,1,2,3"],
                   cwd=work, env=env, check=True)
    with open(eval_dir / "calc.log", "w") as fh:
        subprocess.run([py, str(VENDOR / "calc.py"), "--path", str(pred),
                        "--item_path", inputs["info"]["path"]],
                       cwd=work, env=env, check=True, stdout=fh, stderr=subprocess.STDOUT)
    for i in range(N_SHARDS):            # keep the shard logs, drop the shard copies
        shutil.copy2(shards / f"{i}.log", eval_dir / f"shard{i}.log")
    shutil.rmtree(work)
    seconds = int(time.time() - t0)
    write_status(vdir, "evaluated", eval_seconds=seconds, eval_limit=eval_limit,
                 predictions_sha256=sha256_file(pred), last_error=None)
    return {"eval_seconds": seconds, "predictions_sha256": sha256_file(pred)}


def gpu_info() -> list[str]:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="comma-separated variant names")
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--job", required=True, help="job id used for the attempt directory")
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--budget-seconds", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="64 samples, 1 epoch, batch 32")
    ap.add_argument("--eval-limit", type=int, default=None, help="score only the first N test rows")
    ap.add_argument("--zcr-inputs", type=Path, default=None,
                    help="root written by build_zcr_inputs.py (needed for *__zcr variants)")
    args = ap.parse_args()
    t_start = time.time()
    hp = SMOKE if args.smoke else FULL
    base_model = paths.FROZEN_BASE_MODELS / "qwen2.5-1.5b-instruct"
    names = [v for v in args.variants.split(",") if v]
    for v in names:
        base = v.removesuffix(ZCR_SUFFIX)
        if base not in ORDER or (v != base and not base.startswith("rqkmeans")):
            raise SystemExit(f"unknown variant {v}")
    per_variant = (TRAIN_SECONDS_4GPU * 4 // args.gpus if not args.smoke else 900) + EVAL_SECONDS
    failures = []
    for name in names:
        vdir = args.run_root / name
        vdir.mkdir(parents=True, exist_ok=True)
        st = read_status(vdir)
        if st["step"] == "evaluated":
            log(f"{name}: already evaluated; skipping")
            continue
        left = args.budget_seconds - int(time.time() - t_start)
        needs_training = st["step"] not in ("trained", "trained_unretained")
        if needs_training and left < per_variant:
            log(f"{name}: {left}s left < {per_variant}s needed; stopping for resubmission")
            break
        log(f"{name}: step={st['step']}")
        try:
            pre = preflight(name, base_model, args.zcr_inputs)
            man_path = vdir / "manifest.json"
            man = json.loads(man_path.read_text()) if man_path.exists() else {}
            man.update(variant=name, protocol="experiments/substrate/exp1_ar_next_item_retrain",
                       recipe="vendor/onediffrec at eae9ecc (sft/data/evaluate/calc byte-identical)",
                       hyperparameters=hp, eval=EVAL, smoke=args.smoke, **pre)
            if needs_training:
                attempt = vdir / "attempts" / args.job
                attempt.mkdir(parents=True, exist_ok=False)
                timeout = max(600, left - EVAL_SECONDS)
                train(name, vdir, attempt, pre["inputs"], base_model, args.gpus, hp, timeout)
                st = read_status(vdir)
            if st["step"] == "trained_unretained":
                man["weights_sha256"] = _retain(name, vdir, Path(st["train_attempt"]))
                man.update(train_gpus=args.gpus, gpu_info=gpu_info(),
                           train_seconds=st.get("train_seconds"))
            man_path.write_text(json.dumps(man, indent=1))
            man.update(evaluate(name, vdir, pre["inputs"], args.gpus, args.eval_limit))
            man_path.write_text(json.dumps(man, indent=1))
            log(f"{name}: done")
        except subprocess.TimeoutExpired:
            log(f"{name}: training hit the wall-time budget; resubmit to resume")
            failures.append(name)
            break
        except Exception as e:  # noqa: BLE001 -- record and continue with the next variant
            log(f"{name}: FAILED: {e}")
            write_status(vdir, read_status(vdir)["step"], last_error=str(e))
            failures.append(name)
    if failures:
        log(f"variants not finished: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
