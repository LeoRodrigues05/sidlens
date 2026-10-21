"""The grid registry: every run and weight state sidlens may analyse, with proof.

`build` walks a OneDiffRec work root (`grid.layout`), and for every run records:

    identity     dataset, task, paradigm, map, ZCR assignment, seed
    inputs       prepared cohort folder, data root, SID table path and sha256
    states       every kept weight state, the sha256 of every file in it
    checks       each state's bytes against the fingerprint the run's own
                 provenance recorded; the run's source snapshot against
                 `vendor/onediffrec_v3`; the saved test list's sha256
    problems     every reason the run cannot be used, never silently dropped

Experiments resolve `grid:` names only through this file (see `grid` and
`models.ar.checkpoint_dir`). A state whose bytes later change fails its hash
check at load time instead of being analysed under the registered name.

The registry lives at `paths.RESULTS/registry/registry.json`. Rebuilding it
moves the previous file aside (never deletes it).
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from sidlens import paths
from sidlens.grid import BEST, PREFIX, TASK_OF, make_name, split_name, v3code
from sidlens.grid.fingerprint import fingerprint
from sidlens.grid.layout import GridRun, discover
from sidlens.provenance.hashing import HashCache, walk_files

SCHEMA = "sidlens.grid-registry/v1"


def default_path() -> Path:
    return paths.RESULTS / "registry" / "registry.json"


def _state_files(state_path: Path) -> list[Path]:
    return [state_path] if state_path.is_file() else list(walk_files(state_path))


def _check_artifact(run: GridRun, phase: str, artifact: str, path: Path, cache: HashCache) -> str:
    rec = run.provenance(phase)
    if not rec or artifact not in rec.get("artifacts", {}):
        return "no-record"
    want = rec["artifacts"][artifact]
    got = fingerprint(path, cache)
    return "ok" if (got["kind"], got["sha256"]) == (want.get("kind"), want.get("sha256")) else "mismatch"


def _entry(run: GridRun, cache: HashCache, hash_weights: bool, check_source: bool) -> dict:
    problems: list[str] = []
    phases = {ph: run.status(ph) for ph in ("train", "eval")}
    for ph, st in phases.items():
        if st != "completed":
            problems.append(f"{ph} phase {st}")
    manifest_path = run.prepared_dir / "data-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if not manifest:
        problems.append(f"no prepared data manifest at {manifest_path}")
    sid_index = Path(manifest.get("input_paths", {}).get("sid_index", "")) if manifest else None
    sid = {"path": str(sid_index) if sid_index else None, "sha256": None,
           "recorded_sha256": run.cell.get("sid_index_sha256")}
    if sid_index and sid_index.is_file():
        sid["sha256"] = cache.get(sid_index)
        if sid["recorded_sha256"] and sid["sha256"] != sid["recorded_sha256"]:
            problems.append("SID table changed since the run was planned")
    else:
        problems.append("SID table missing")

    states = {}
    for name, st in run.states.items():
        rec = {"path": str(st.path), "best": st.best, "epoch": st.epoch,
               "global_step": st.global_step, "eval_loss": st.eval_loss, "files": None}
        if hash_weights:
            root = st.path if st.path.is_dir() else st.path.parent
            rec["files"] = {(p.relative_to(root).as_posix()): cache.get(p) for p in _state_files(st.path)}
        states[name] = rec
    if BEST not in states:
        problems.append("no best weight state")

    checks = {}
    if hash_weights:
        if run.paradigm == "ar":
            if BEST in run.states:
                checks["best"] = _check_artifact(run, "train", "checkpoint", run.states[BEST].path, cache)
            if any(not s.best for s in run.states.values()):
                checks["snapshots"] = _check_artifact(run, "train", "epoch_snapshots",
                                                      run.run_dir / "training" / "snapshots", cache)
        else:
            if BEST in run.states:
                checks["best"] = _check_artifact(run, "train", "checkpoint", run.states[BEST].path, cache)
            if any(not s.best for s in run.states.values()):
                checks["snapshots"] = _check_artifact(run, "train", "intermediate_checkpoints",
                                                      run.run_dir / "training" / "intermediate", cache)
        if run.predictions.is_file():
            checks["predictions"] = _check_artifact(run, "eval", "predictions", run.predictions, cache)
        for what, verdict in checks.items():
            if verdict == "mismatch":
                problems.append(f"{what}: bytes differ from the run's provenance record")
    if not run.predictions.is_file():
        problems.append("no saved test predictions")

    source = None
    if check_source:
        source = v3code.check_source(run.source_snapshot) if run.source_snapshot.is_dir() else \
            {"ok": False, "missing": ["<source snapshot>"], "mismatch": [], "source_root": str(run.source_snapshot)}
        if not source["ok"]:
            problems.append("run's source snapshot differs from vendor/onediffrec_v3")

    config = None
    if run.paradigm == "diffgrm" and run.resolved_config.is_file():
        config = json.loads(run.resolved_config.read_text())

    return {
        "key": run.key, "names": {s: make_name(run.key, s) for s in states},
        "run_name": run.key.split("/")[0], "dataset": run.dataset, "category": run.category,
        "predtype": run.predtype, "task": TASK_OF[run.predtype], "paradigm": run.paradigm,
        "method": run.method, "digits": run.digits, "size": run.size, "variant": run.variant,
        "label": run.label, "sid_assignment": run.sid_assignment, "assignment_id": run.assignment_id,
        "seed": run.seed, "found_at": str(run.found_at), "run_dir": str(run.run_dir),
        "adopted": run.found_at.is_symlink(),
        "prepared_dir": str(run.prepared_dir), "data_root": str(run.data_root),
        "cache_root": str(run.cache_root), "source_snapshot": str(run.source_snapshot),
        "source_origin": run.cell.get("source_origin"), "phases": phases,
        "sid_index": sid, "prepared_rows": manifest.get("rows"),
        "predictions": str(run.predictions), "metrics": str(run.metrics),
        "resolved_config": str(run.resolved_config) if run.paradigm == "diffgrm" else None,
        "config": config, "best_epoch": run.best_epoch if run.paradigm == "diffgrm" else None,
        "states": states, "checks": checks, "source_check": source,
        "usable": not problems, "problems": problems,
    }


def build(work_root: Path, run_name: str = "grid-v1", tables: str = "v3", mode: str = "full",
          hash_weights: bool = True, check_source: bool = True, workers: int = 8,
          log=print) -> dict:
    """Discover and check every run of `run_name` (one name, or several, e.g. the
    two handoff smoke runs). Hashing reads every weight file once (cached)."""
    names = [run_name] if isinstance(run_name, str) else list(run_name)
    runs = [r for n in names for r in discover(work_root, n, tables, mode)]
    if not runs:
        raise FileNotFoundError(f"no {run_name}/{mode} runs under {work_root} (tables {tables})")
    cache = HashCache()
    t0 = time.time()
    entries: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_entry, r, cache, hash_weights, check_source): r for r in runs}
        for i, fut in enumerate(futures, 1):
            run = futures[fut]
            entries[run.key] = fut.result()
            if log and (i % 25 == 0 or i == len(runs)):
                log(f"[register] {i}/{len(runs)} runs, {time.time() - t0:.0f}s")
    return {"schema": SCHEMA, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "work_root": str(Path(work_root).resolve()), "run_name": run_name, "tables": tables,
            "mode": mode, "hashed": hash_weights, "vendor_onediffrec_v3": v3code.manifest()["merkle_root"],
            "entries": dict(sorted(entries.items()))}


def save(doc: dict, path: Path | None = None) -> Path:
    """Write the registry and a coverage table beside it. An existing, different
    registry is moved aside to `registry.json.replaced-<stamp>`, never deleted."""
    path = Path(path or default_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        path.rename(path.with_name(f"{path.name}.replaced-{time.strftime('%Y%m%dT%H%M%S')}"))
    path.write_text(text)
    cols = ["key", "dataset", "predtype", "paradigm", "variant", "sid_assignment", "usable",
            "n_states", "states", "problems"]
    with (path.parent / "coverage.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in doc["entries"].values():
            w.writerow({"key": e["key"], "dataset": e["dataset"], "predtype": e["predtype"],
                        "paradigm": e["paradigm"], "variant": e["variant"],
                        "sid_assignment": e["sid_assignment"], "usable": e["usable"],
                        "n_states": len(e["states"]), "states": " ".join(sorted(e["states"])),
                        "problems": "; ".join(e["problems"])})
    load.cache_clear()
    return path


class Registry:
    def __init__(self, doc: dict, path: Path):
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a grid registry ({doc.get('schema')!r})")
        self.doc, self.path = doc, path
        self.entries: dict[str, dict] = doc["entries"]

    def entry(self, name_or_key: str) -> dict:
        key = split_name(name_or_key)[0] if name_or_key.startswith(PREFIX) else name_or_key
        try:
            return self.entries[key]
        except KeyError:
            raise KeyError(f"{key!r} is not in the grid registry {self.path}") from None

    def state(self, name: str) -> tuple[dict, dict]:
        """(run entry, state record) for `grid:<key>@<state>`; refuses unusable runs."""
        key, state = split_name(name)
        e = self.entry(key)
        if not e["usable"]:
            raise RuntimeError(f"{key} is not usable: {'; '.join(e['problems'])}")
        if state not in e["states"]:
            raise KeyError(f"{key} has no weight state {state!r}; it has {sorted(e['states'])}")
        return e, e["states"][state]

    def names(self, *, paradigm: str | None = None, predtype: str | None = None,
              dataset: str | None = None, states: str = "best", usable: bool = True) -> list[str]:
        """Model names, filtered. `states` is "best" or "all"."""
        out = []
        for e in self.entries.values():
            if (paradigm and e["paradigm"] != paradigm) or (predtype and e["predtype"] != predtype) \
                    or (dataset and e["dataset"] != dataset) or (usable and not e["usable"]):
                continue
            for s in sorted(e["states"]):
                if states == "all" or s == BEST:
                    out.append(make_name(e["key"], s))
        return out


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> Registry:
    p = Path(path)
    return Registry(json.loads(p.read_text()), p)


def load(path: Path | None = None) -> Registry:
    p = Path(path or default_path())
    if not p.is_file():
        raise FileNotFoundError(f"no grid registry at {p}; run `sidlens grid register` first")
    return _load(str(p), p.stat().st_mtime_ns)


load.cache_clear = _load.cache_clear  # type: ignore[attr-defined]


def exists(path: Path | None = None) -> bool:
    return Path(path or default_path()).is_file()


def verify_state(name: str, cache: HashCache | None = None) -> str:
    """Re-hash a state's files against the registry; return the weights' sha256.

    The cache makes this a stat check unless the file changed. A registry built
    without hashing cannot vouch for anything, so it is refused.
    """
    e, st = load().state(name)
    if st["files"] is None:
        raise RuntimeError(f"{name}: the registry was built without hashes; rerun `sidlens grid register`")
    cache = cache or HashCache()
    root = Path(st["path"])
    base = root if root.is_dir() else root.parent
    for rel, want in st["files"].items():
        got = cache.get(base / rel)
        if got != want:
            raise RuntimeError(f"{name}: {rel} sha256 {got[:12]} != registered {want[:12]}")
    # AR state folders also hold training_args.bin; the weights are the safetensors.
    weights = ([rel for rel in st["files"] if rel.endswith(".safetensors")] if e["paradigm"] == "ar"
               else list(st["files"]))
    if len(weights) != 1:
        raise RuntimeError(f"{name}: expected one weight file, found {weights}")
    return st["files"][weights[0]]
