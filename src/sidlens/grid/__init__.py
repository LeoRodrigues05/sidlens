"""The OneDiffRec grid (`grid-v1`): reading the coauthor's runs as sidlens evidence.

The frozen substrate holds the archived sweep; the grid is the final training
run (372 runs: two catalogs, two tasks, AR and DiffGRM, 27 native maps and
their ZCR twins), which lives in a OneDiffRec work root on the coauthor's
cluster. Nothing here copies or rewrites those runs. sidlens reads them where
they are, records what it read in a registry (with hashes tied to the runs'
own provenance), and writes every result under `paths.RESULTS`.

Names
-----
A grid model is named `grid:<key>@<state>`:

    key    <run_name>/<dataset>/<predtype>/<paradigm>/<label>/seed-<n>
           e.g. grid-v1/industrial/next-item/ar/rqkmeans_3codebook_128/seed-42
           (label = variant, or variant__zcr-<16 hex> for a ZCR twin)
    state  best | epoch-0.5 | epoch-1.5 (AR snapshots) | epoch-0020 ... (DiffGRM)

Every name resolves through the registry (`grid.registry`), never by building
a path: the registry is what proves the bytes are the ones the run recorded.
"""

from __future__ import annotations

import os
from pathlib import Path

PREFIX = "grid:"
BEST = "best"
DATASETS = {"industrial": "Industrial_and_Scientific", "office": "Office"}
PREDTYPES = ("next-item", "next2item")
# sidlens's own task names (`ar_prompts.TASKS`).
TASK_OF = {"next-item": "next-item", "next2item": "two-item"}
PARADIGMS = ("ar", "diffgrm")


def is_grid(name: str) -> bool:
    return isinstance(name, str) and name.startswith(PREFIX)


def make_name(key: str, state: str = BEST) -> str:
    if "@" in key or not state:
        raise ValueError(f"bad grid key/state: {key!r} @ {state!r}")
    return f"{PREFIX}{key}@{state}"


def split_name(name: str) -> tuple[str, str]:
    """`grid:<key>@<state>` -> (key, state). A name without a state means `best`."""
    if not is_grid(name):
        raise ValueError(f"not a grid model name: {name!r}")
    body = name[len(PREFIX):]
    key, _, state = body.partition("@")
    parts = key.split("/")
    if len(parts) != 6 or not parts[5].startswith("seed-"):
        raise ValueError(f"grid key must be run/dataset/predtype/paradigm/label/seed-N, got {key!r}")
    return key, state or BEST


def onediffrec_work_root() -> Path:
    """The OneDiffRec handoff work root (`WORK_ROOT`), from `ONEDIFFREC_WORK_ROOT`.

    Read at call time, never at import, so a site file sourced after import
    still applies. Unset is an error: guessing a root is how a pipeline ends up
    analysing someone else's runs.
    """
    value = os.environ.get("ONEDIFFREC_WORK_ROOT")
    if not value:
        raise RuntimeError("ONEDIFFREC_WORK_ROOT is not set (the OneDiffRec handoff work root, "
                           "the folder holding industrial-v3/ and office-v3/)")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"ONEDIFFREC_WORK_ROOT={root} is not a directory")
    return root


def cell_info(name: str) -> dict:
    """What a summary groups a grid model by, parsed from its name alone (no registry):
    quantizer (rqvae | rqkmeans | MQ), depth, width, zcr, dataset, task, paradigm, state."""
    import re
    key, state = split_name(name)
    run, dataset, predtype, paradigm, label, seed = key.split("/")
    m = re.fullmatch(r"(rqvae|rqkmeans|MQ)_(\d)codebook_(\d+)(__zcr-[0-9a-f]{16})?", label)
    if not m:
        raise ValueError(f"unrecognised grid label {label!r}")
    return {"quantizer": m.group(1), "depth": int(m.group(2)), "width": int(m.group(3)),
            "zcr": m.group(4) is not None, "dataset": dataset, "task": TASK_OF[predtype],
            "paradigm": paradigm, "state": state, "map": label, "grid_run": run}


def map_dir(name: str) -> str:
    """The results path of a grid model below an experiment folder:
    `<dataset>/<task>/<paradigm>/<map>/<state>` (map: variant, `__zcr` for a ZCR twin)."""
    i = cell_info(name)
    m = i["map"].split("__zcr-")[0] + ("__zcr" if i["zcr"] else "")
    return f"{i['dataset']}/{i['task']}/{i['paradigm']}/{m}/{i['state']}"
