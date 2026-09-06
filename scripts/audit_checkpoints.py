#!/usr/bin/env python
"""Account for every checkpoint the sweeps were supposed to produce.

The question this answers is not "what is in the registry" -- that is one line
of json. It is "for each cell of the planned grid that has no weights, are those
weights gone, or are they sitting somewhere nobody looked?"

Three ledgers get reconciled:

  planned    the sweep's own manifests (results/sweep_state/*/manifest.tsv) for
             AR, and the 27 frozen SID variants for diffusion. This is what the
             experiment intended to exist, independent of what happened.
  trained    results/sweep_state/*/state/<variant> == "done" for AR, and the
             registry for diffusion. Training finished; weights existed once.
  on disk    weights findable right now, anywhere we can search.

A cell that is trained-but-not-on-disk is the interesting case: upstream's
sweep_runner.sbatch deleted AR weights after scoring, so "trained" and "exists"
came apart for 26 of 28 AR runs. This script says which of those are recoverable
and which are genuinely gone.

Search order, cheapest first. Every root is optional -- a root that does not
exist is reported as such rather than skipped silently, because "we did not look
there" and "it is not there" are different answers:

  1. frozen/ckpt/**             the hash-verified substrate
  2. $SIDLENS_WORK/runs/**      anything trained since the freeze
  3. the transfer zip           read from the central directory, no extraction
  4. --search <dir>             extra roots (old scratch, another cluster path)

Orphan weights -- a checkpoint on disk that no registry entry claims -- are
identified from the weights themselves. For DiffGRM, embedding.weight has shape
(3 + n_digit*codebook_size, n_embd), which pins (n_digit, codebook_size) without
any log. The quantizer is not in the weights; it comes from the run's log if one
sits beside the checkpoint, and is left unknown otherwise.

Usage
-----
    python scripts/audit_checkpoints.py                    # full audit
    python scripts/audit_checkpoints.py --missing-only     # just the gaps
    python scripts/audit_checkpoints.py --search /l/users/someone/old_runs
    python scripts/audit_checkpoints.py --json report.json

Exit status is 1 if any planned cell has no weights on disk, so this can gate a
job that is about to assume a checkpoint is there.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidlens import paths                                    # noqa: E402
from sidlens.data import sids                                # noqa: E402
from sidlens.registry import diffusion as diffreg            # noqa: E402

WEIGHT_NAMES = ("pytorch_model.bin", "model.safetensors")
TRANSFER_ZIP = Path("/home/leo.rodrigues/GenRecSys/sidlens/sidlens-transfer-20260827.zip")


# ----------------------------------------------------------------- ledger --

@dataclass
class Cell:
    """One planned (paradigm, task, quantizer, depth, width) combination."""
    paradigm: str            # ar | diffusion
    task: str                # next-item/next1 | two-item/next2
    category: str
    quantizer: str
    n_codebook: int
    codebook_size: int

    planned_by: str = ""     # which manifest asked for it
    trained: bool = False    # training finished at least once
    ckpt_id: str | None = None
    metrics: bool = False    # eval numbers survive even if weights do not
    found: list[str] = field(default_factory=list)   # where weights are now

    @property
    def key(self) -> str:
        return (f"{self.paradigm}/{self.task}/{self.category}/"
                f"{self.quantizer}_{self.n_codebook}cb_{self.codebook_size}")

    @property
    def status(self) -> str:
        if self.found:
            return "OK"
        if self.trained and self.metrics:
            return "WEIGHTS LOST"     # trained, scored, weights deleted
        if self.trained:
            return "WEIGHTS LOST"
        if self.metrics:
            return "METRICS ONLY"
        return "NEVER TRAINED"


def plan_ar() -> dict[str, Cell]:
    """AR cells, from the sweep's own manifests plus its done-flags."""
    cells: dict[str, Cell] = {}
    root = paths.FROZEN_RESULTS / "sweep_state"
    if not root.is_dir():
        return cells

    for tree_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        tree = tree_dir.name                       # next-item | two-item
        for manifest in sorted(tree_dir.glob("*.tsv")):
            for row in _read_tsv(manifest):
                c = Cell(paradigm="ar", task=tree, category=row["category"],
                         quantizer=row["method"], n_codebook=int(row["codebooks"]),
                         codebook_size=int(row["size"]))
                cells.setdefault(c.key, c).planned_by = manifest.name

        state = tree_dir / "state"
        if state.is_dir():
            for flag in state.iterdir():
                if flag.suffix == ".train_seconds" or not flag.is_file():
                    continue
                if flag.read_text().strip() != "done":
                    continue
                for c in cells.values():
                    if c.task == tree and _ar_variant_id(c) == flag.name:
                        c.trained = True

        metrics_dir = paths.FROZEN_RESULTS / "sweep_metrics" / tree / "metrics"
        if metrics_dir.is_dir():
            scored = {p.name.split(".")[0] for p in metrics_dir.glob("*.json")}
            for c in cells.values():
                if c.task == tree and _ar_variant_id(c) in scored:
                    c.metrics = True
    return cells


def _ar_variant_id(c: Cell) -> str:
    stem = c.task.replace("-", "")               # next-item -> nextitem
    return f"{stem}__{c.category}__{c.quantizer}__{c.n_codebook}cb__{c.codebook_size}"


def _read_tsv(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def plan_diffusion(category: str = "Industrial_and_Scientific") -> dict[str, Cell]:
    """Diffusion cells: every frozen SID variant is a cell the grid could fill.

    next1 was meant to span the full 27. next2 was only ever run at cb=256, so
    its other cells are marked planned-but-never-attempted rather than lost --
    the distinction the status column exists to make.
    """
    cells: dict[str, Cell] = {}
    for v in sids.available("diffgrm"):
        for task in ("next1", "next2"):
            c = Cell(paradigm="diffusion", task=task, category=category,
                     quantizer=v.quantizer, n_codebook=v.n_codebook,
                     codebook_size=v.codebook_size, planned_by="frozen SID grid")
            cells[c.key] = c

    try:
        registry = diffreg.load_runtime()
    except FileNotFoundError:
        return cells

    for ckpt_id, e in registry.items():
        c = Cell(paradigm="diffusion", task=e["task"], category=category,
                 quantizer=e["quantizer"], n_codebook=e["n_codebook"],
                 codebook_size=e["codebook_size"])
        cell = cells.setdefault(c.key, c)
        cell.ckpt_id = ckpt_id
        cell.trained = e.get("status") == "trained"
        cell.metrics = bool(e.get("recorded_metrics"))
        p = Path(e["ckpt_path"])
        if p.exists():
            cell.found.append(str(p))
    return cells


# ----------------------------------------------------------------- search --

def scan_dir(root: Path) -> list[Path]:
    """Every weight file under a root. Skips the frozen base model."""
    if not root.is_dir():
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "base_models" in Path(dirpath).parts:
            dirnames[:] = []
            continue
        out.extend(Path(dirpath) / f for f in filenames if f in WEIGHT_NAMES)
    return sorted(out)


def scan_zip(path: Path) -> list[str]:
    """Weight entries in the transfer archive, from its central directory.

    Nothing is extracted and nothing is decompressed -- this reads the index at
    the end of the archive, so a 13 GB zip costs milliseconds.
    """
    if not path.is_file():
        return []
    try:
        with zipfile.ZipFile(path) as z:
            return sorted(n for n in z.namelist()
                          if Path(n).name in WEIGHT_NAMES
                          and "base_models" not in n)
    except zipfile.BadZipFile:
        return []


def identify(weights: Path) -> dict:
    """Recover (n_digit, codebook_size) from a DiffGRM checkpoint's own shapes.

    embedding.weight is (3 + n_digit*codebook_size, n_embd). That product has
    one factorization within the sweep's grid, so the pair is determined. The
    quantizer is not in the weights; it is read from a sibling log if present.
    """
    info = {"path": str(weights), "kind": None, "n_digit": None,
            "codebook_size": None, "quantizer": None, "n_embd": None}

    if weights.name == "model.safetensors":
        info["kind"] = "ar (qwen sft)"
        cfg = weights.parent / "config.json"
        if cfg.is_file():
            info["n_embd"] = json.loads(cfg.read_text()).get("hidden_size")
        return info

    try:
        import torch
        sd = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:                       # unreadable is a finding
        info["kind"] = f"unreadable: {type(exc).__name__}"
        return info

    info["kind"] = "diffusion (diffgrm)"
    emb = sd.get("embedding.weight")
    if emb is not None:
        rows, info["n_embd"] = int(emb.shape[0]), int(emb.shape[1])
        span = rows - 3                            # sid_offset
        for n_digit in (3, 4, 5):
            if span % n_digit == 0 and (span // n_digit) in (128, 256, 512):
                info["n_digit"], info["codebook_size"] = n_digit, span // n_digit
                break
    info["quantizer"] = _quantizer_from_logs(weights)
    return info


def _quantizer_from_logs(weights: Path) -> str | None:
    """Find the sem_ids file a run loaded, by looking for its log.

    Upstream writes weights to <task>/<jobid>/saved/<stamp>/ but the log to the
    parallel <task>_logs/<jobid>/ tree, so the sibling directory has to be
    searched too -- looking only under the checkpoint finds nothing.
    """
    from sidlens.registry import parse_logs

    bases = [weights.parent]
    if len(weights.parents) > 2:
        job_dir = weights.parents[2]                     # .../<task>/<jobid>
        bases.append(job_dir)
        bases.append(job_dir.parent.with_name(job_dir.parent.name + "_logs")
                     / job_dir.name)

    for base in bases:
        if not base.is_dir():
            continue
        for log in list(base.rglob("*.log"))[:8]:
            try:
                p = parse_logs.parse_text(log.read_text(errors="replace"))
            except Exception:
                continue
            if p.sem_ids_path:
                return Path(p.sem_ids_path).stem.split("_")[0]
    return None


# ----------------------------------------------------------------- report --

def audit(extra_roots: list[Path], use_zip: bool = True) -> dict:
    cells = plan_ar()
    cells.update(plan_diffusion())

    roots = [("frozen", paths.FROZEN_CKPT), ("runs", paths.RUNS)]
    roots += [(f"search:{r}", r) for r in extra_roots]

    searched, on_disk = [], []
    for label, root in roots:
        present = root.is_dir()
        hits = scan_dir(root) if present else []
        searched.append({"root": str(root), "label": label,
                         "exists": present, "n_weights": len(hits)})
        on_disk.extend(hits)

    zip_entries = scan_zip(TRANSFER_ZIP) if use_zip else []
    searched.append({"root": str(TRANSFER_ZIP), "label": "transfer zip",
                     "exists": TRANSFER_ZIP.is_file(), "n_weights": len(zip_entries)})

    # A weight file the registry already points at is accounted for. Anything
    # else is an orphan: real weights that no planned cell claims, which is
    # exactly the "checkpoint nobody looked for" case this script exists to find.
    claimed = {f for c in cells.values() for f in c.found}
    orphans = sorted(str(w) for w in on_disk if str(w) not in claimed)

    # AR weights live in a flat directory named for the task, not for a variant,
    # so they are matched through the sweep's own best-variant label files.
    for label_file in (paths.FROZEN_CKPT / "ar" / "best_variant_labels").rglob("*.variant"):
        variant_id = label_file.read_text().strip()
        task = label_file.parents[1].name              # next-item | two-item
        ckpt_dir = paths.FROZEN_CKPT / "ar" / f"{task}_best"
        weights = ckpt_dir / "model.safetensors"
        if not weights.is_file():
            continue
        for c in cells.values():
            if c.paradigm == "ar" and _ar_variant_id(c) == variant_id:
                c.found.append(str(weights))
                if str(weights) in orphans:
                    orphans.remove(str(weights))

    return {"cells": cells, "searched": searched,
            "zip_entries": zip_entries, "orphans": orphans}


def format_report(rep: dict, missing_only: bool = False) -> str:
    out = ["SEARCHED", "--------"]
    for s in rep["searched"]:
        mark = "ok " if s["exists"] else "GONE"
        out.append(f"  [{mark}] {s['n_weights']:3d} weight files  {s['root']}")

    cells = rep["cells"]
    rows = sorted(cells.values(), key=lambda c: (c.paradigm, c.task, c.category,
                                                 c.quantizer, c.n_codebook,
                                                 c.codebook_size))
    if missing_only:
        rows = [c for c in rows if not c.found]

    out += ["", "CELLS", "-----",
            f"  {'cell':58s} {'status':14s} metrics"]
    for c in rows:
        out.append(f"  {c.key:58s} {c.status:14s} {'yes' if c.metrics else '-'}")

    tally: dict[str, int] = {}
    for c in cells.values():
        tally[c.status] = tally.get(c.status, 0) + 1
    out += ["", "SUMMARY", "-------"]
    for k in ("OK", "WEIGHTS LOST", "METRICS ONLY", "NEVER TRAINED"):
        if k in tally:
            out.append(f"  {k:14s} {tally[k]:3d}")

    if rep["orphans"]:
        out += ["", "ORPHAN WEIGHTS (on disk, unclaimed by the registry)", "-" * 50]
        for o in rep["orphans"]:
            out.append(f"  {o}")
    else:
        out += ["", "No orphan weights: every checkpoint on disk is registered."]

    # The archive is the last place a lost checkpoint could hide. Comparing it
    # to frozen/ by path suffix says whether it holds anything new, without
    # extracting 13 GB to find out.
    if rep["zip_entries"]:
        live = {Path(f).relative_to(paths.WORK).as_posix()
                for c in rep["cells"].values() for f in c.found
                if str(f).startswith(str(paths.WORK))}
        live |= {Path(o).relative_to(paths.WORK).as_posix() for o in rep["orphans"]
                 if o.startswith(str(paths.WORK))}
        extra = [e for e in rep["zip_entries"] if e not in live]
        out += ["", "TRANSFER ARCHIVE", "-" * 16]
        if extra:
            out.append(f"  {len(extra)} weight file(s) in the zip but NOT unpacked "
                       "on disk -- recoverable:")
            out += [f"    {e}" for e in extra]
        else:
            out.append("  Nothing the frozen tree does not already have.")

    lost = [c for c in cells.values() if c.status == "WEIGHTS LOST"]
    if lost:
        out += ["", f"{len(lost)} cell(s) finished training but have no weights "
                    "anywhere we looked.",
                "Retraining is the only way to recover them."]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--search", action="append", default=[], type=Path,
                    help="extra root to scan for weights (repeatable)")
    ap.add_argument("--missing-only", action="store_true",
                    help="list only cells with no weights on disk")
    ap.add_argument("--no-zip", action="store_true",
                    help="skip the transfer archive index")
    ap.add_argument("--identify", type=Path,
                    help="identify one checkpoint from its weights and exit")
    ap.add_argument("--json", type=Path, help="also write the report as json")
    args = ap.parse_args(argv)

    if args.identify:
        print(json.dumps(identify(args.identify), indent=2))
        return 0

    rep = audit(args.search, use_zip=not args.no_zip)
    print(format_report(rep, missing_only=args.missing_only))

    if args.json:
        args.json.write_text(json.dumps(
            {"cells": [asdict(c) | {"status": c.status} for c in rep["cells"].values()],
             "searched": rep["searched"],
             "orphans": rep["orphans"],
             "zip_entries": rep["zip_entries"]}, indent=1))
        print(f"\njson -> {args.json}")

    return 1 if any(not c.found for c in rep["cells"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
