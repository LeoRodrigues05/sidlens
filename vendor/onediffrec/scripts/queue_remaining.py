#!/usr/bin/env python3
"""Work out which Industrial cells the research notes still need, and queue them.

Scope
-----
Pages 8-14 of "Research Notes - OneDiffRec (3)" mark the cells we own in blue:
codebook sizes 128 and 512, all three SID methods, 3/4/5 codebooks, for both
model families. That is 2 x 3 x 3 x 2 = 36 cells. The 256 tables belong to the
co-author and are left alone.

    Qwen2.5-1.5B rows  -> sft.py, driven by scripts/sweep_runner.sbatch
                          (one job walks a manifest, 4 GPUs, self-resubmits)
    Mask Diffusion rows -> scripts/run_diffgrm.sbatch
                          (one job per variant, 1 GPU)

A cell counts as done when its metrics file (Qwen) or per-variant log
(Mask Diffusion) already holds test numbers, so re-running this is safe and
only ever queues what is missing.

Nothing is submitted unless --submit is given.

Usage
-----
    python scripts/queue_remaining.py                 # show the plan
    python scripts/queue_remaining.py --submit        # queue it
    python scripts/queue_remaining.py --submit --lanes 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants as V

REPO = "/home/leo.rodrigues/onediffrec/OneDiffRec"
WORK = "/l/users/leo.rodrigues/onediffrec"
DATA_ROOT = f"{WORK}/data"
CATEGORY = "Industrial_and_Scientific"

SIZES = (128, 512)
METHODS = ("rqvae", "rqkmeans", "MQ")
CODEBOOKS = (3, 4, 5)

QWEN_METRICS = f"{WORK}/sweep/next-item/metrics"
QWEN_STATE = f"{WORK}/sweep/next-item/state"
DIFF_LOGS = f"{WORK}/diffgrm/{CATEGORY}"
MANIFEST = f"{WORK}/sweep/next-item/industrial_round3.tsv"


def qwen_id(method: str, cb: int, size: int) -> str:
    return f"nextitem__{CATEGORY}__{method}__{cb}cb__{size}"


def diff_variant(method: str, cb: int, size: int) -> str:
    return f"{method}_{cb}codebook_{size}"


def qwen_done(method: str, cb: int, size: int) -> bool:
    path = os.path.join(QWEN_METRICS, qwen_id(method, cb, size) + ".json")
    if not os.path.isfile(path):
        return False
    try:
        return bool(json.load(open(path)).get("metrics", {}).get("HR@10"))
    except Exception:
        return False


def diff_done(method: str, cb: int, size: int) -> bool:
    path = os.path.join(DIFF_LOGS, diff_variant(method, cb, size) + ".txt")
    if not os.path.isfile(path):
        return False
    with open(path, errors="ignore") as f:
        return "Test Results: OrderedDict" in f.read()


def qwen_ready(method: str, cb: int, size: int) -> tuple[bool, str]:
    """Run the same asset validation the sweep runner runs before training."""
    r = subprocess.run(
        [f"{REPO}/.conda/bin/python", f"{REPO}/scripts/validate_sft_assets.py",
         "--data-root", DATA_ROOT, "--tree", "next-item", "--category", CATEGORY,
         "--method", method, "--codebooks", str(cb),
         "--codebook-size", str(size), "--skip-rows"],
        capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr or r.stdout).strip().splitlines()[-1][:70]
    d = json.loads(r.stdout)
    return True, f"{d['items']} items, {d['unique_full_sids']} SIDs"


def diff_ready(method: str, cb: int, size: int) -> tuple[bool, str]:
    """The runner needs sem_ids whose width and range match the variant name."""
    path = os.path.join(REPO, "DiffGRM/cache/AmazonReviews2014", CATEGORY,
                        "processed", diff_variant(method, cb, size) + ".sem_ids")
    if not os.path.isfile(path):
        return False, "sem_ids missing"
    sem = json.load(open(path))
    rows = list(sem.values())
    digits = len(rows[0])
    top = max(max(r) for r in rows)
    if digits != cb:
        return False, f"{digits} digits, want {cb}"
    if top >= size:
        return False, f"max code {top} >= {size}"
    if any(len(r) != cb for r in rows):
        return False, "ragged sem_ids"
    return True, f"{len(sem)} items, {digits}d, max {top}"


def stale_state(method: str, cb: int, size: int) -> str | None:
    """A leftover 'failed' marker would make the sweep runner skip the variant."""
    p = os.path.join(QWEN_STATE, qwen_id(method, cb, size))
    if os.path.isfile(p):
        s = open(p).read().strip()
        if s in ("failed", "done"):
            return s
    return None


def survey():
    qwen, diff = [], []
    for size in SIZES:
        for method in METHODS:
            for cb in CODEBOOKS:
                qwen.append((method, cb, size, qwen_done(method, cb, size)))
                diff.append((method, cb, size, diff_done(method, cb, size)))
    return qwen, diff


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submit", action="store_true", help="actually queue the jobs")
    ap.add_argument("--lanes", type=int, default=4,
                    help="how many Mask Diffusion jobs may run at once (default 4)")
    a = ap.parse_args()

    qwen, diff = survey()

    print("=" * 78)
    print(f"  Industrial blue scope: sizes {SIZES}, methods {METHODS}, codebooks {CODEBOOKS}")
    print("=" * 78)
    for label, rows in (("Qwen2.5-1.5B (sft.py)", qwen),
                        ("Mask Diffusion (DiffGRM)", diff)):
        done = sum(1 for *_, d in rows if d)
        print(f"\n  {label}: {done}/{len(rows)} done")
        print(f"    {'':>6}" + "".join(f"{f'{cb}cb':>8}" for size in SIZES for cb in CODEBOOKS))
        print(f"    {'':>6}" + "".join(f"{size:>8}" for size in SIZES for cb in CODEBOOKS))
        for method in METHODS:
            cells = []
            for size in SIZES:
                for cb in CODEBOOKS:
                    d = next(x[3] for x in rows if x[:3] == (method, cb, size))
                    cells.append("done" if d else "TO RUN")
            print(f"    {method:<6}" + "".join(f"{c:>8}" for c in cells))

    pend_qwen = [(m, c, s) for m, c, s, d in qwen if not d]
    pend_diff = [(m, c, s) for m, c, s, d in diff if not d]

    print("\n" + "=" * 78)
    print("  Pre-flight on everything still to run")
    print("=" * 78 + "\n")
    ok_qwen, ok_diff, blocked = [], [], []
    for m, c, s in pend_qwen:
        ready, note = qwen_ready(m, c, s)
        st = stale_state(m, c, s)
        if ready and st == "failed":
            ready, note = False, "stale 'failed' state - clear it first"
        (ok_qwen if ready else blocked).append((m, c, s))
        print(f"  qwen  {m:<9}{c}cb/{s:<5} {'OK ' if ready else 'BLOCKED'}  {note}")
    print()
    for m, c, s in pend_diff:
        ready, note = diff_ready(m, c, s)
        (ok_diff if ready else blocked).append((m, c, s))
        print(f"  diff  {m:<9}{c}cb/{s:<5} {'OK ' if ready else 'BLOCKED'}  {note}")

    print("\n" + "=" * 78)
    print(f"  {len(ok_qwen)} Qwen + {len(ok_diff)} Mask Diffusion ready, {len(blocked)} blocked")
    print("=" * 78)
    if blocked:
        print("  blocked:", ", ".join(f"{m} {c}cb/{s}" for m, c, s in blocked))

    # Qwen manifest: cheap-first, matching the sweep runner's own ordering.
    if ok_qwen:
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w") as f:
            f.write("variant_id\ttree\tcategory\tmethod\tcodebooks\tsize\n")
            for m, c, s in sorted(ok_qwen, key=lambda x: (x[0], x[2], x[1])):
                f.write(f"{qwen_id(m, c, s)}\tnext-item\t{CATEGORY}\t{m}\t{c}\t{s}\n")
        print(f"\n  manifest -> {MANIFEST} ({len(ok_qwen)} variants)")

    sweep_cmd = ["sbatch", f"--export=ALL,PHASE=next-item,MANIFEST={MANIFEST}",
                 "scripts/sweep_runner.sbatch"]
    diff_cmds = [["sbatch", f"--export=ALL,VARIANT={diff_variant(m, c, s)},N_DIGIT={c}",
                  "scripts/run_diffgrm.sbatch"]
                 for m, c, s in sorted(ok_diff, key=lambda x: (x[2], x[0], x[1]))]

    print("\n" + "=" * 78)
    print(f"  Plan ({'SUBMITTING' if a.submit else 'dry run'})")
    print("=" * 78 + "\n")
    if ok_qwen:
        print("  " + " ".join(sweep_cmd))
    print(f"\n  {len(diff_cmds)} Mask Diffusion jobs across {a.lanes} lane(s);"
          f" each lane runs one at a time")
    for i, cmd in enumerate(diff_cmds):
        print(f"    lane {i % a.lanes}  " + " ".join(cmd[1:2]))

    if not a.submit:
        print("\n  nothing submitted; re-run with --submit")
        return 0

    if ok_qwen:
        r = subprocess.run(sweep_cmd, cwd=REPO, capture_output=True, text=True)
        print("\n  sweep:", (r.stdout or r.stderr).strip())

    tails: dict[int, str] = {}
    for i, cmd in enumerate(diff_cmds):
        lane = i % a.lanes
        full = list(cmd)
        if lane in tails:
            full.insert(1, f"--dependency=afterany:{tails[lane]}")
        r = subprocess.run(full, cwd=REPO, capture_output=True, text=True)
        out = (r.stdout or r.stderr).strip()
        job = re.search(r"(\d+)", out)
        if job:
            tails[lane] = job.group(1)
        print(f"  lane {lane}  {cmd[1].split('VARIANT=')[1].split(',')[0]:<24} {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
