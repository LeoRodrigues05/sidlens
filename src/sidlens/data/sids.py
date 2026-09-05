"""Semantic-ID tables, loaded from the frozen substrate.

The `.sem_ids` JSON is the ground truth. The cached `item_id2tokens_*.npy` and
`tokens2item_*.pkl` in the DiffGRM caches are write-only side effects upstream --
the tokenizer's load tag and save tag use different formats and never match, so
those files are never read back. They are frozen as evidence, not as inputs.

The pickle is also LOSSY: `tokens2item[tuple(tokens)] = item_id` lets colliding
items overwrite each other, so it holds 2195 entries for 3105 items at
rqkmeans 3cb x 128. Decoding an SID through it silently loses ~29% of the
catalogue. `SidTable.cb2items` is the one-to-many map that does not.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from sidlens import paths

RE_SEM_NAME = re.compile(r"^(?P<q>rqvae|rqkmeans|MQ)_(?P<cb>\d+)codebook_(?P<size>\d+)$")
RE_TOKEN = re.compile(r"<([a-z])_(\d+)>")
SID_OFFSET = 3  # PAD=0, BOS=1, EOS=2

# Items that are genuine duplicates in the catalogue -- identical text, so any
# quantizer must collide them. This is the floor below which a collision rate
# cannot go, and it is not a property of the quantizer.
DUPLICATE_FLOOR = 11


@dataclass(frozen=True)
class SidVariant:
    quantizer: str
    n_codebook: int
    codebook_size: int

    @property
    def name(self) -> str:
        return f"{self.quantizer}_{self.n_codebook}codebook_{self.codebook_size}"

    @classmethod
    def parse(cls, name: str) -> "SidVariant":
        m = RE_SEM_NAME.match(name.replace(".sem_ids", ""))
        if not m:
            raise ValueError(f"unparseable SID variant name: {name!r}")
        return cls(m["q"], int(m["cb"]), int(m["size"]))


class SidTable:
    """One quantizer's assignment of items to semantic IDs."""

    def __init__(self, variant: SidVariant, asin2codes: dict[str, tuple[int, ...]],
                 source: Path, item2id: dict[str, int] | None = None):
        self.variant = variant
        self.asin2codes = asin2codes
        self.source = source
        self._item2id = item2id
        widths = {len(v) for v in asin2codes.values()}
        if widths != {variant.n_codebook}:
            raise ValueError(
                f"{source.name}: expected every SID to be {variant.n_codebook} "
                f"digits, found widths {sorted(widths)}")
        lo = min(min(v) for v in asin2codes.values())
        hi = max(max(v) for v in asin2codes.values())
        if lo < 0 or hi >= variant.codebook_size:
            raise ValueError(
                f"{source.name}: codes out of range [0,{variant.codebook_size}) "
                f"-- observed [{lo},{hi}]. This is the signature of the faiss "
                f"bit-packing defect; check whether this is a .mispacked file.")

    # ---------------------------------------------------------------- load --
    @classmethod
    def load(cls, variant: SidVariant | str, tree: str = "diffgrm",
             item2id: dict[str, int] | None = None) -> "SidTable":
        if isinstance(variant, str):
            variant = SidVariant.parse(variant)
        path = paths.FROZEN_SIDS / "sem_ids" / tree / f"{variant.name}.sem_ids"
        raw = json.loads(path.read_text())
        return cls(variant, {k: tuple(v) for k, v in raw.items()}, path, item2id)

    @classmethod
    def load_index_json(cls, category: str, variant: SidVariant | str) -> "SidTable":
        """Load from the item-id-keyed `<a_N>` index JSON instead of `.sem_ids`.

        Same content, different key space and encoding. Useful for asserting the
        two representations agree.
        """
        if isinstance(variant, str):
            variant = SidVariant.parse(variant)
        stem = {"rqvae": f"{category}.index_{variant.n_codebook}codebook_{variant.codebook_size}.json",
                "rqkmeans": f"{category}.rqkmeans.index_{variant.n_codebook}codebook_{variant.codebook_size}.json",
                "MQ": f"{category}.index.MQ.{variant.n_codebook}codebook_{variant.codebook_size}.json"}[variant.quantizer]
        path = paths.FROZEN_SIDS / "index_json" / stem
        if not path.exists() and variant.quantizer == "MQ":
            # Industrial MQ 3cb x 256 ships without the usual suffix.
            alt = paths.FROZEN_SIDS / "index_json" / f"{category}.index.MQ.json"
            if alt.exists():
                path = alt
        raw = json.loads(path.read_text())
        codes = {k: tuple(int(RE_TOKEN.match(t).group(2)) for t in v)
                 for k, v in raw.items()}
        return cls(variant, codes, path)

    # -------------------------------------------------------------- views --
    @property
    def n_items(self) -> int:
        return len(self.asin2codes)

    @cached_property
    def keys(self) -> list[str]:
        return list(self.asin2codes)

    @cached_property
    def codes(self) -> np.ndarray:
        """(n_items, n_codebook) int array in `keys` order."""
        return np.array([self.asin2codes[k] for k in self.keys], dtype=np.int64)

    def tokens(self, key: str) -> tuple[int, ...]:
        """Offset token ids, matching the model's embedding layout.

        Digit d occupies ids [3 + d*K, 3 + (d+1)*K).
        """
        K = self.variant.codebook_size
        return tuple(c + SID_OFFSET + d * K for d, c in enumerate(self.asin2codes[key]))

    @cached_property
    def cb2items(self) -> dict[tuple[int, ...], list[str]]:
        """SID -> ALL items carrying it.

        The upstream pickle keeps only the last writer, so it cannot answer this.
        Anything that decodes an SID back to items must go through here.
        """
        out: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for key, code in self.asin2codes.items():
            out[code].append(key)
        return dict(out)

    def prefix(self, d: int) -> dict[str, tuple[int, ...]]:
        """Depth-d prefix of every SID. Depth 0 is the whole catalogue."""
        return {k: v[:d] for k, v in self.asin2codes.items()}

    def clusters(self, d: int) -> dict[tuple[int, ...], list[str]]:
        """Items grouped by depth-d prefix.

        For RQ-VAE and RQ-KMeans this is a residual refinement hierarchy. For MQ
        it is NOT -- MQ quantizes the full embedding independently per digit
        (3cb and 4cb share digit 0 for 17/3105 items), so a "depth-d cluster" is
        an intersection of parallel partitions, not a refinement of a coarser one.
        """
        out: dict[tuple[int, ...], list[str]] = defaultdict(list)
        for key, code in self.asin2codes.items():
            out[code[:d]].append(key)
        return dict(out)

    # --------------------------------------------------------- statistics --
    @cached_property
    def collision_stats(self) -> dict:
        buckets = self.cb2items
        sizes = np.array([len(v) for v in buckets.values()])
        n_collided = int(self.n_items - len(buckets))
        return {
            "n_items": self.n_items,
            "unique_sids": len(buckets),
            "collisions": n_collided,
            "collision_rate": n_collided / self.n_items,
            "max_bucket": int(sizes.max()),
            "n_singletons": int((sizes == 1).sum()),
            "duplicate_floor": DUPLICATE_FLOOR,
            "floor_rate": DUPLICATE_FLOOR / self.n_items,
        }

    @cached_property
    def codebook_utilization(self) -> list[dict]:
        """Per-digit code usage. Reveals how unbalanced a quantizer is."""
        K = self.variant.codebook_size
        out = []
        for d in range(self.variant.n_codebook):
            counts = np.bincount(self.codes[:, d], minlength=K)
            used = counts[counts > 0]
            out.append({
                "digit": d,
                "n_used": int((counts > 0).sum()),
                "utilization": float((counts > 0).sum() / K),
                "min_count": int(used.min()),
                "max_count": int(used.max()),
                "entropy_bits": float(
                    -(used / used.sum() * np.log2(used / used.sum())).sum()),
                "max_entropy_bits": float(np.log2(K)),
            })
        return out

    def __repr__(self) -> str:
        s = self.collision_stats
        return (f"SidTable({self.variant.name}: {s['n_items']} items, "
                f"{s['unique_sids']} unique, {s['collision_rate']:.2%} collision)")


def available(tree: str = "diffgrm") -> list[SidVariant]:
    root = paths.FROZEN_SIDS / "sem_ids" / tree
    return sorted((SidVariant.parse(p.stem) for p in root.glob("*.sem_ids")),
                  key=lambda v: (v.quantizer, v.n_codebook, v.codebook_size))


def load_item2id(category: str) -> dict[str, int]:
    """ASIN -> OneDiffRec integer item id. DiffGRM's own id is this + 1."""
    path = paths.FROZEN_DATA / "id_maps" / f"{category}.item2id"
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        asin, idx = line.split("\t")
        out[asin] = int(idx)
    return out


# --- depth families: which variants are truncations of one fit ---------------
#
# Measured across all 27 tables, not assumed. For RQ-KMeans, the 3/4/5-codebook
# tables at one codebook size are the SAME fit truncated to different depths --
# every item's 3-digit SID is a prefix of its 5-digit SID (3105/3105). For
# RQ-VAE and MQ they are independent fits: agreement is 0/3105 everywhere, so a
# 3cb and a 4cb table share no structure at all.
#
# This decides what a depth comparison can claim. Within a nested family,
# "add a digit" is a clean manipulation and RQ3's redundancy question is
# answerable directly. Across independent fits, 3cb vs 4cb differs by BOTH the
# extra digit and a whole refit, and any difference confounds the two.
#
# One exception, and it is not a rounding error: rqkmeans 5cb x 512 agrees with
# its own 3cb/4cb siblings on 0 of 3105 items while 3cb and 4cb agree on
# 3105/3105. It also has a different code-usage profile (largest first-digit
# cell 81 vs 29) and a 0.35% collision rate against their 9.57%/5.57%. That is
# the signature of a different fit, and it lines up with the 2026-08-20
# bit-unpacking repair -- `sids/mispacked` exists precisely because the AR
# rqkmeans 5cb x 512 run was trained on the pre-repair originals. Treat it as
# its own lineage, never as the 5-digit extension of rqkmeans 3cb x 512.
NESTED_DEPTH_FAMILIES = {"rqkmeans"}
NOT_NESTED_EXCEPTIONS = {"rqkmeans_5codebook_512"}


def is_nested_family(variant: SidVariant | str) -> bool:
    """Is this table a truncation of the same fit as its depth siblings?

    False means a depth comparison against its siblings changes the quantizer
    as well as the number of digits.
    """
    if isinstance(variant, str):
        variant = SidVariant.parse(variant)
    if variant.name in NOT_NESTED_EXCEPTIONS:
        return False
    return variant.quantizer in NESTED_DEPTH_FAMILIES


def check_nesting(shallow: "SidTable", deep: "SidTable") -> dict:
    """Measure, rather than trust, whether `shallow` is a prefix of `deep`.

    The constants above were derived from this; keeping it callable means a new
    or repaired SID table can be checked instead of inheriting an assumption.
    """
    if shallow.variant.codebook_size != deep.variant.codebook_size:
        raise ValueError("nesting is only defined within one codebook size")
    if shallow.variant.n_codebook >= deep.variant.n_codebook:
        raise ValueError("`shallow` must have fewer digits than `deep`")
    keys = [k for k in shallow.keys if k in deep.asin2codes]
    d = shallow.variant.n_codebook
    agree = sum(shallow.asin2codes[k] == deep.asin2codes[k][:d] for k in keys)
    return {
        "shallow": shallow.variant.name,
        "deep": deep.variant.name,
        "n_compared": len(keys),
        "n_prefix_agree": agree,
        "agree_rate": agree / len(keys) if keys else 0.0,
        "nested": bool(keys) and agree == len(keys),
    }
