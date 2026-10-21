"""Find the runs of a OneDiffRec work root, and the weight states each one kept.

Layout (OneDiffRec `scripts/onediffrec.py`, `scripts/run_experiment.py`)::

    WORK_ROOT/<dataset>-<tables>/experiments/<run_name>/<mode>/<dataset>/<predtype>/
        <paradigm>/<label>/seed-<n>/{cell.json, provenance/, training/, evaluation/}

Traps, each with the guard that closes it
-----------------------------------------
1. **Adopted runs are symlinks.** A run finished on the previous tables
   (`<dataset>-v2`) is linked into `<dataset>-v3` instead of retrained
   (`onediffrec.py train --adopt-from`). `Path.rglob` does not enter them, so
   discovery walks the fixed five levels and follows the `seed-*` link; the
   cell's own paths then point into the v2 tree, which is where its files are.
2. **Two table versions in one work root.** `<dataset>-v2` and `<dataset>-v3`
   can both hold `grid-v1`. Only `<dataset>-<tables>` is read (default v3), so a
   run is never counted twice.
3. **The path is not the identity.** `cell.json` is what the worker ran; a
   folder whose name disagrees with its cell (dataset, task, paradigm, label,
   seed) raises instead of being filed under the wrong configuration.
4. **Snapshot names come from the run.** AR keeps the epochs listed in
   `training/snapshots/index.json` (the requested epochs changed between
   handoff versions, and an early-stopped run may lack the later ones); DiffGRM
   keeps `training/intermediate/epoch-NNNN.bin`. Nothing assumes a fixed list.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from sidlens.grid import BEST, DATASETS, PARADIGMS, PREDTYPES

RE_LABEL = re.compile(r"^(?P<variant>(?P<method>rqvae|rqkmeans|MQ)_(?P<digits>\d)codebook_(?P<size>\d+))"
                      r"(?:__(?P<assignment>zcr-[0-9a-f]{16}))?$")
RE_DIFF_SNAPSHOT = re.compile(r"^epoch-(\d{4})\.bin$")


@dataclass(frozen=True)
class State:
    """One kept weight state of a run."""
    name: str                 # best | epoch-0.5 | epoch-0020 ...
    path: Path                # AR: a model directory; DiffGRM: a .bin state dict
    best: bool
    epoch: float | None = None
    global_step: int | None = None
    eval_loss: float | None = None


@dataclass
class GridRun:
    key: str
    found_at: Path            # the seed-<n> path as found (may be a symlink)
    cell: dict = field(repr=False)

    # -- identity, from cell.json ------------------------------------------
    @property
    def run_dir(self) -> Path:
        return self.found_at.resolve()

    @property
    def dataset(self) -> str:
        return self.cell["dataset"]

    @property
    def category(self) -> str:
        return DATASETS[self.dataset]

    @property
    def predtype(self) -> str:
        return self.cell["predtype"]

    @property
    def paradigm(self) -> str:
        return self.cell["paradigm"]

    @property
    def method(self) -> str:
        return self.cell["method"]

    @property
    def digits(self) -> int:
        return int(self.cell["digits"])

    @property
    def size(self) -> int:
        return int(self.cell["size"])

    @property
    def variant(self) -> str:
        return f"{self.method}_{self.digits}codebook_{self.size}"

    @property
    def sid_assignment(self) -> str:
        return self.cell.get("sid_assignment", "native")

    @property
    def assignment_id(self) -> str:
        return self.cell.get("assignment_id", "native")

    @property
    def label(self) -> str:
        return self.variant if self.sid_assignment == "native" else f"{self.variant}__{self.assignment_id}"

    @property
    def seed(self) -> int:
        return int(self.cell["seed"])

    @property
    def prepared_dir(self) -> Path:
        return Path(self.cell["prepared_dir"])

    @property
    def data_root(self) -> Path:
        return Path(self.cell["data_root"])

    @property
    def cache_root(self) -> Path:
        return Path(self.cell["cache_root"])

    @property
    def source_snapshot(self) -> Path:
        return Path(self.cell["repo"])

    # -- phases ---------------------------------------------------------------
    def provenance(self, phase: str) -> dict | None:
        p = self.run_dir / "provenance" / phase / "result.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def status(self, phase: str) -> str:
        rec = self.provenance(phase)
        if rec is not None:
            return rec.get("status", "unknown")
        return "started" if (self.run_dir / f"{phase}.log").exists() else "missing"

    @property
    def done(self) -> bool:
        return self.status("train") == "completed" and self.status("eval") == "completed"

    # -- weight states --------------------------------------------------------
    @cached_property
    def states(self) -> dict[str, State]:
        training = self.run_dir / "training"
        out: dict[str, State] = {}
        if self.paradigm == "ar":
            best = training / "final_checkpoint"
            if best.is_dir():
                out[BEST] = State(BEST, best, True)
            index = training / "snapshots" / "index.json"
            if index.is_file():
                for snap in json.loads(index.read_text()).get("snapshots", []):
                    name = snap["path"]
                    if name == BEST or not (training / "snapshots" / name).is_dir():
                        raise ValueError(f"{self.key}: snapshot {name!r} listed in index.json is missing")
                    out[name] = State(name, training / "snapshots" / name, False,
                                      epoch=snap.get("epoch"), global_step=snap.get("global_step"),
                                      eval_loss=snap.get("eval_loss"))
        else:
            best = training / "pytorch_model.bin"
            if best.is_file():
                out[BEST] = State(BEST, best, True, epoch=self.best_epoch)
            inter = training / "intermediate"
            if inter.is_dir():
                for p in sorted(inter.iterdir()):
                    m = RE_DIFF_SNAPSHOT.match(p.name)
                    if m:
                        out[p.stem] = State(p.stem, p, False, epoch=float(int(m.group(1))))
        return out

    @cached_property
    def best_epoch(self) -> int | None:
        """DiffGRM: the epoch of the selected model, from the per-epoch history."""
        hist = self.run_dir / "training" / "history.jsonl"
        if not hist.is_file():
            return None
        best = None
        for line in hist.read_text().splitlines():
            if line.strip():
                best = json.loads(line).get("best_epoch", best)
        return best

    # -- evaluation outputs ---------------------------------------------------
    @property
    def predictions(self) -> Path:
        return self.run_dir / "evaluation" / "predictions.jsonl"

    @property
    def metrics(self) -> Path:
        return self.run_dir / "evaluation" / "metrics.json"

    @property
    def resolved_config(self) -> Path:
        return self.run_dir / "training" / "resolved-config.json"


def _dirs(p: Path) -> list[Path]:
    """Sub-directories (following links), sorted by name."""
    return sorted((Path(e.path) for e in os.scandir(p) if e.is_dir(follow_symlinks=True)),
                  key=lambda q: q.name)


def table_roots(work_root: Path, tables: str | None = "v3") -> list[Path]:
    """`<work_root>/<dataset>-<tables>` for every dataset present; `<work_root>/<dataset>`
    when `tables` is empty (the first setup's SID tables, as in the handoff smoke runs)."""
    roots = [work_root / (f"{ds}-{tables}" if tables else ds) for ds in DATASETS]
    return [r for r in roots if r.is_dir()]


def discover(work_root: Path, run_name: str = "grid-v1", tables: str = "v3",
             mode: str = "full") -> list[GridRun]:
    """Every run of `run_name` under the `<dataset>-<tables>` roots of `work_root`."""
    runs: list[GridRun] = []
    seen: set[str] = set()
    for root in table_roots(Path(work_root), tables):
        base = root / "experiments" / run_name / mode
        if not base.is_dir():
            continue
        for ds in _dirs(base):
            for predtype in _dirs(ds):
                for paradigm in _dirs(predtype):
                    for label in _dirs(paradigm):
                        for seed in _dirs(label):
                            if not seed.name.startswith("seed-"):
                                continue
                            cell_path = seed / "cell.json"
                            if not cell_path.is_file():
                                # Planned but never started: OneDiffRec creates the folder first.
                                continue
                            cell = json.loads(cell_path.read_text())
                            run = GridRun("/".join([run_name, ds.name, predtype.name, paradigm.name,
                                                    label.name, seed.name]), seed, cell)
                            _check_identity(run, ds.name, predtype.name, paradigm.name, label.name, seed.name)
                            if run.key in seen:
                                raise ValueError(f"duplicate grid run {run.key}")
                            seen.add(run.key)
                            runs.append(run)
    return runs


def _check_identity(run: GridRun, ds: str, predtype: str, paradigm: str, label: str, seed: str) -> None:
    want = (ds, predtype, paradigm, label, seed)
    got = (run.dataset, run.predtype, run.paradigm, run.label, f"seed-{run.seed}")
    if want != got:
        raise ValueError(f"{run.found_at}: folder says {want}, cell.json says {got}")
    if run.dataset not in DATASETS or run.predtype not in PREDTYPES or run.paradigm not in PARADIGMS:
        raise ValueError(f"{run.found_at}: unknown dataset/predtype/paradigm {got}")
    m = RE_LABEL.match(label)
    if not m:
        raise ValueError(f"{run.found_at}: unparseable label {label!r}")
    if (m["assignment"] is None) != (run.sid_assignment == "native"):
        raise ValueError(f"{run.found_at}: label {label!r} vs sid_assignment {run.sid_assignment!r}")
