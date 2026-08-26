#!/usr/bin/env python3
"""Single source of truth for the dataset variants shipped in OneDiffRec_data.tar.gz.

The archive holds two parallel task trees over the same 54 (category, SID method,
codebook count, codebook size) combinations:

    data/Amazon18/   next-item prediction      target = one SID
    data/two-item/   next-2-items prediction   target = "sid_A ||| sid_B"

Only the Amazon18 tree carries the per-category SID assets; the two-item tree
reuses them. Index filenames differ per SID method, and two combinations carry a
date suffix on their CSVs, so every consumer resolves paths through here rather
than rebuilding the naming rules.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, asdict

TREES = {
    "next-item": "Amazon18",
    "two-item": "two-item",
}

# Per-category SID assets always live under the Amazon18 tree.
ASSET_TREE = "Amazon18"

CATEGORIES = ("Industrial_and_Scientific", "Office")
METHODS = ("rqvae", "rqkmeans", "MQ")
CODEBOOKS = (3, 4, 5)
SIZES = (128, 256, 512)

# Catalogue size drives run cost, so "cheap first" is really "small catalogue first".
CATEGORY_ITEMS = {"Industrial_and_Scientific": 3105, "Office": 17696}

INDEX_PATTERNS = {
    "rqvae": "{category}.index_{codebooks}codebook_{size}.json",
    "rqkmeans": "{category}.rqkmeans.index_{codebooks}codebook_{size}.json",
    "MQ": "{category}.index.MQ.{codebooks}codebook_{size}.json",
}

# Industrial MQ 3codebook_256 ships without the usual suffix. Verified by reading
# it out of the archive: 3105 items, 3 tokens each, codes 0..255.
INDEX_OVERRIDES = {
    ("Industrial_and_Scientific", "MQ", 3, 256): "Industrial_and_Scientific.index.MQ.json",
}


@dataclass(frozen=True)
class Variant:
    tree: str
    category: str
    method: str
    codebooks: int
    size: int

    @property
    def variant_id(self) -> str:
        tree = self.tree.replace("-", "")
        return f"{tree}__{self.category}__{self.method}__{self.codebooks}cb__{self.size}"

    @property
    def tag(self) -> str:
        """Filename stem used by the train/valid/test/info CSVs."""
        return f"{self.category}_{self.method}_{self.codebooks}codebook_{self.size}"

    @property
    def items(self) -> int:
        return CATEGORY_ITEMS[self.category]


def _one(pattern: str, what: str) -> str:
    """Resolve a glob to exactly one path.

    Two combinations (Industrial rqvae/MQ 3codebook_256) carry a
    `_2013-10-2018-11` date suffix, so split files are matched by prefix. The
    trailing-character guard keeps `..._3codebook_512` from also matching
    `..._3codebook_512X` style names.
    """
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {what} matching {pattern}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous {what}: {pattern} -> {matches}")
    return matches[0]


def resolve(data_root: str, variant: Variant) -> dict:
    """Map a variant to every file sft.py / evaluate.py needs."""
    tree_dir = os.path.join(data_root, TREES[variant.tree])
    asset_dir = os.path.join(data_root, ASSET_TREE, variant.category)

    key = (variant.category, variant.method, variant.codebooks, variant.size)
    index_name = INDEX_OVERRIDES.get(
        key,
        INDEX_PATTERNS[variant.method].format(
            category=variant.category, codebooks=variant.codebooks, size=variant.size
        ),
    )

    paths = {
        "train": _one(os.path.join(tree_dir, "train", f"{variant.tag}*.csv"), "train csv"),
        "valid": _one(os.path.join(tree_dir, "valid", f"{variant.tag}*.csv"), "valid csv"),
        "test": _one(os.path.join(tree_dir, "test", f"{variant.tag}*.csv"), "test csv"),
        "info": _one(os.path.join(tree_dir, "info", f"{variant.tag}*.txt"), "info txt"),
        "index": os.path.join(asset_dir, index_name),
        "item": os.path.join(asset_dir, f"{variant.category}.item.json"),
    }
    for what, path in (("index", paths["index"]), ("item", paths["item"])):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {what}: {path}")
    return paths


def enumerate_variants(tree: str, order: str = "cheap-first") -> list[Variant]:
    variants = [
        Variant(tree, category, method, codebooks, size)
        for category in CATEGORIES
        for method in METHODS
        for codebooks in CODEBOOKS
        for size in SIZES
    ]
    if order == "cheap-first":
        # Small catalogue first, so a complete Industrial table lands early.
        variants.sort(key=lambda v: (v.items, v.method, v.codebooks, v.size))
    elif order != "declared":
        raise ValueError(f"unknown order {order!r}")
    return variants


def write_manifest(path: str, tree: str, order: str = "cheap-first") -> int:
    variants = enumerate_variants(tree, order)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        handle.write("variant_id\ttree\tcategory\tmethod\tcodebooks\tsize\n")
        for v in variants:
            handle.write(
                f"{v.variant_id}\t{v.tree}\t{v.category}\t{v.method}\t{v.codebooks}\t{v.size}\n"
            )
    return len(variants)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", choices=sorted(TREES), default="next-item")
    parser.add_argument("--order", choices=("cheap-first", "declared"), default="cheap-first")
    parser.add_argument("--data-root", help="verify every variant resolves under this root")
    parser.add_argument("--write-manifest", help="write the manifest TSV here")
    args = parser.parse_args()

    variants = enumerate_variants(args.tree, args.order)
    print(f"{args.tree}: {len(variants)} variants ({args.order})")

    if args.data_root:
        missing = 0
        for v in variants:
            try:
                resolve(args.data_root, v)
            except (FileNotFoundError, ValueError) as exc:
                missing += 1
                print(f"  UNRESOLVED {v.variant_id}: {exc}")
        print(f"resolved {len(variants) - missing}/{len(variants)}")
        if missing:
            raise SystemExit(1)

    if args.write_manifest:
        n = write_manifest(args.write_manifest, args.tree, args.order)
        print(f"wrote {n} rows to {args.write_manifest}")


if __name__ == "__main__":
    main()
