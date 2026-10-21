"""What an AR analysis needs about a checkpoint's rows, for frozen and grid checkpoints alike.

The representation experiments read captures and then need, per checkpoint,
the rows the capture was made from, the SID table, item ids, review days,
item embeddings and item metadata. For the three frozen checkpoints those are
frozen Industrial files; for a grid weight state (`grid:<key>@<state>`) they
are the run's own (`sidlens.grid`). This module is the one switch, so the
analysis below an experiment's input step stays as it was.

Traps, each with its guard
--------------------------
1. **Rows of another cohort.** A grid capture holds the sampled rows of the
   run's prepared split; `examples` returns exactly those (same sample rule),
   and callers index them by `example_id`, never by position.
2. **Another catalog's side files.** Embeddings, metadata and days are read
   from the run's own data root, whose `item2id` defines the item rows; the
   frozen files are used only for frozen checkpoints.
3. **The wrong prompt template.** `template` is the one the checkpoint was
   tested with: `eval` for the archived frozen predictions, `v3` for the grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sidlens.data import ar_prompts as P
from sidlens.models import ar

TASKS = {"next-item_best": "next-item", "oneoff_rqvae4cb128": "next-item", "two-item_best": "two-item"}


def _entry(ckpt: str) -> dict:
    from sidlens.grid import registry
    return registry.load().entry(ckpt)


def task(ckpt: str) -> str:
    if ar.is_grid(ckpt):
        return _entry(ckpt)["task"]
    if ckpt in TASKS:
        return TASKS[ckpt]
    return "next-item"                                    # registered retrains are next-item


def template(ckpt: str) -> str:
    return "v3" if ar.is_grid(ckpt) else "eval"


def examples(ckpt: str, split: str = "test") -> list[P.ArExample]:
    """The checkpoint's rows of `split` (grid: the run's sampled prepared rows)."""
    if ar.is_grid(ckpt):
        from sidlens.grid import inputs as GI
        return GI.ar_examples(_entry(ckpt), split, sample=True)
    return P.load_examples(ar.load_vocab(ckpt).variant, task(ckpt), split)


def table(ckpt: str):
    from sidlens.data.sids import SidTable
    if ar.is_grid(ckpt):
        from sidlens.grid import inputs as GI
        return GI.sid_table(_entry(ckpt))
    return SidTable.load(ar.load_vocab(ckpt).variant)


def item2id(ckpt: str) -> dict[str, int]:
    if ar.is_grid(ckpt):
        from sidlens.grid import inputs as GI
        return GI.item2id(_entry(ckpt))
    from sidlens.data.sids import load_item2id
    return load_item2id(P.PRIMARY_CATEGORY)


def row_times(ckpt: str, exs: list[P.ArExample], split: str = "test"):
    """`timestamps.ar_row_times`-shaped (rows, hist) tables for `exs`, in their order."""
    from sidlens.data import timestamps as T
    if ar.is_grid(ckpt):
        from sidlens.grid import events as GE
        return GE.row_times(_entry(ckpt), split, [e.example_id for e in exs])
    return T.ar_row_times(exs, T.load_event_times())


def embeddings(ckpt: str, dtype=np.float32) -> np.ndarray:
    """(n_items, 2560) item embeddings, row = item id of the checkpoint's catalog."""
    if ar.is_grid(ckpt):
        from sidlens.grid import inputs as GI
        e = _entry(ckpt)
        return np.load(GI.category_dir(e) / f"{e['category']}.emb-qwen-td.npy").astype(dtype)
    from sidlens.data import embeddings as E
    return E.load(P.PRIMARY_CATEGORY, dtype)


def item_meta(ckpt: str) -> dict:
    """ASIN -> `meta.ItemMeta` for the checkpoint's catalog."""
    from sidlens.data import meta
    if ar.is_grid(ckpt):
        from sidlens.grid import inputs as GI
        e = _entry(ckpt)
        return meta.load(e["category"], root=str(GI.category_dir(e)))
    return meta.load(P.PRIMARY_CATEGORY)


def weights_file(ckpt: str) -> Path:
    """The checkpoint's single safetensors file (for reading the unembedding directly)."""
    f = ar.checkpoint_dir(ckpt) / "model.safetensors"
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing")
    return f


def capture_store(where, ckpt: str, split: str, *, verify: bool = True, sites=()):
    """Open an activation store made by `scripts/activations/ar_capture.py` and check
    it is `ckpt`'s `split` under the checkpoint's template.

    where  a frozen capture job id (`derived/ar_capture/<job>/capture/store`) or a
           capture directory (holding `store/` or `capture/store/`).
    """
    from sidlens import paths
    from sidlens.hooks.store import StoreReader
    p = Path(str(where))
    if not p.is_absolute() and str(where).isdigit():
        p = paths.DERIVED / "ar_capture" / str(where) / "capture" / "store"
    elif (p / "capture" / "store").is_dir():
        p = p / "capture" / "store"
    elif (p / "store").is_dir():
        p = p / "store"
    st = StoreReader(p, verify=verify)
    m = st.meta
    want = (f"ar/{ckpt}", ar.checkpoint_sha(ckpt), split, template(ckpt))
    got = (m["model"], m["checkpoint_sha256"], m["split"], m["template"])
    if got != want:
        raise RuntimeError(f"store {p}: {got[0]} {got[2]} {got[3]} is not {ckpt} {split} {template(ckpt)}")
    missing = [s for s in sites if s not in st.sites]
    if missing:
        raise KeyError(f"store {p} lacks {missing}")
    return st
