"""Item metadata, keyed the way the SID tables are keyed.

`<category>.item.json` is keyed by the OneDiffRec integer item id; every SID
table is keyed by ASIN. Everything that wants an item's text has to bridge those
two, and doing it inline is how exp2 ended up with a private `brand_labels`.

`categories` is present on every record and empty on every record (3105/3105 for
Industrial). It is exposed anyway rather than dropped, so that a caller asking
for it gets an honest empty string instead of a KeyError that looks like a bug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sidlens import paths
from sidlens.data.sids import load_item2id


@dataclass(frozen=True)
class ItemMeta:
    asin: str
    item_id: int
    title: str
    brand: str
    description: str
    categories: str

    @property
    def short_title(self) -> str:
        """Title trimmed for table display, on a word boundary where possible."""
        t = " ".join(self.title.split())
        if len(t) <= 68:
            return t
        cut = t[:68].rsplit(" ", 1)[0]
        return (cut if len(cut) >= 40 else t[:68]) + "…"


@lru_cache(maxsize=4)
def load(category: str) -> dict[str, ItemMeta]:
    """ASIN -> ItemMeta for every item in the frozen catalogue."""
    raw = json.loads(
        (paths.FROZEN_DATA / "item_meta" / f"{category}.item.json").read_text())
    item2id = load_item2id(category)
    out: dict[str, ItemMeta] = {}
    for asin, iid in item2id.items():
        rec = raw.get(str(iid))
        if rec is None:
            continue
        out[asin] = ItemMeta(
            asin=asin,
            item_id=iid,
            title=(rec.get("title") or "").strip(),
            brand=(rec.get("brand") or "").strip(),
            description=(rec.get("description") or "").strip(),
            categories=(rec.get("categories") or "").strip()
                       if isinstance(rec.get("categories"), str) else "",
        )
    return out


def field_labels(category: str, keys: list[str], field: str) -> tuple[np.ndarray, list[str]]:
    """Integer label per item in `keys` order, plus the vocabulary.

    -1 marks an item whose value for `field` is missing or empty. Returned as
    codes rather than strings because every consumer (AMI, purity, probes)
    wants codes, and building the vocab twice is how two callers disagree about
    what class 7 is.
    """
    meta = load(category)
    vocab: dict[str, int] = {}
    out = np.full(len(keys), -1, dtype=np.int64)
    for i, k in enumerate(keys):
        rec = meta.get(k)
        if rec is None:
            continue
        v = getattr(rec, field, "").strip()
        if v:
            out[i] = vocab.setdefault(v, len(vocab))
    return out, [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def coverage(category: str) -> dict[str, dict]:
    """Per-field non-empty count and distinct-value count.

    The number that matters for Experiment 1 is `distinct` against `n_items`:
    a field with 1188 values over 3105 items cannot serve as a probe target.
    """
    meta = load(category)
    n = len(meta)
    out = {}
    for field in ("title", "brand", "description", "categories"):
        vals = [getattr(m, field) for m in meta.values()]
        nonempty = [v for v in vals if v]
        out[field] = {
            "n_items": n,
            "non_empty": len(nonempty),
            "distinct": len(set(nonempty)),
            "mean_items_per_value": (len(nonempty) / len(set(nonempty))) if nonempty else 0.0,
        }
    return out
