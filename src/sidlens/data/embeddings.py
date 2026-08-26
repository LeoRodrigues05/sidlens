"""Pre-quantization item embeddings.

These are the `z_i` of the cluster-geometry analysis: the vectors the quantizers
were fitted to. Qwen3-Embedding-4B, mean-pooled over [title, description] with
NO PCA and NO L2 normalisation, stored float16.

Two naming traps worth knowing:

  * The `-td` suffix means "text + time", but the time block is dead code --
    `amazon_text2emb.py` builds a concatenated array, prints "text+random_time",
    then saves the text-only one. These are text embeddings, 2560-d.
  * `sentence-t5-base_pca256` appears throughout the DiffGRM cache filenames and
    describes none of this. In external-SID mode the tokenizer short-circuits
    before any sentence encoding runs; the tag is stamped in regardless.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from sidlens import paths
from sidlens.data.sids import load_item2id

EMB_DIM = 2560


@lru_cache(maxsize=4)
def _load_cached(category: str, dtype_name: str) -> np.ndarray:
    """Cached raw load. The matrix is identical across SID variants, and a
    27-variant sweep would otherwise re-read 64 MB off Lustre 27 times."""
    arr = _read(category)
    out = arr.astype(np.dtype(dtype_name))
    out.flags.writeable = False
    return out


def _read(category: str) -> np.ndarray:
    path = paths.FROZEN_DATA / "embeddings" / f"{category}.emb-qwen-td.npy"
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != EMB_DIM:
        raise ValueError(f"{path}: expected (n, {EMB_DIM}), got {arr.shape}")
    return arr


def load(category: str, dtype=np.float64) -> np.ndarray:
    """(n_items, 2560) matrix, row i = OneDiffRec item id i.

    Upcast from the stored float16 by default: the geometry sums 2560 squared
    terms over thousands of items, and float16 accumulation loses real precision
    at that scale.
    """
    return _load_cached(category, np.dtype(dtype).name)


def load_aligned(category: str, keys: list[str], dtype=np.float64) -> np.ndarray:
    """Embeddings reordered to match a SidTable's ASIN key order.

    SidTable is ASIN-keyed and the matrix is item-id-indexed, so every join
    between them has to go through item2id. Doing it here keeps that mapping in
    one place instead of at each call site.
    """
    emb = load(category, dtype)
    item2id = load_item2id(category)
    missing = [k for k in keys if k not in item2id]
    if missing:
        raise KeyError(f"{len(missing)} ASINs absent from {category}.item2id, "
                       f"e.g. {missing[:3]}")
    idx = np.array([item2id[k] for k in keys])
    if idx.max() >= emb.shape[0]:
        raise IndexError(
            f"item id {idx.max()} out of range for embedding matrix "
            f"{emb.shape} -- the item2id map and the matrix disagree, which "
            f"means they come from different lineages")
    return emb[idx]
