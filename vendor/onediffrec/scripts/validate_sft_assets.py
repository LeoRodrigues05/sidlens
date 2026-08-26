#!/usr/bin/env python3
"""Validate that prepared SFT CSVs, item metadata, and SIDs agree.

Handles all three SID methods (rqvae / rqkmeans / MQ) and both task trees.
In the two-item tree the target is "sid_A ||| sid_B", so both halves are checked.

SID collisions are reported, not fatal: distinct items legitimately share a full
SID in the coarser configurations (Industrial MQ 3codebook_256 is ~29% colliding),
and the collision rate is itself a reported quantity.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import variants as V  # noqa: E402

SID_TOKEN = re.compile(r"^<([a-z])_(\d+)>$")
TWO_ITEM_SEP = "|||"

REQUIRED_COLUMNS = {
    "user_id",
    "history_item_title",
    "item_title",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tree", choices=sorted(V.TREES), default="next-item")
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--method", default="rqvae", choices=V.METHODS)
    parser.add_argument("--codebooks", type=int, default=4, choices=V.CODEBOOKS)
    parser.add_argument("--codebook-size", type=int, default=128, choices=V.SIZES)
    parser.add_argument(
        "--skip-rows",
        action="store_true",
        help="only check the index/item/info agreement, not every CSV row",
    )
    return parser.parse_args()


def load_literal_list(raw: str, field: str, split: str, row_number: int) -> list:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{split} row {row_number}: invalid {field}") from exc
    if not isinstance(value, list):
        raise TypeError(f"{split} row {row_number}: {field} is not a list")
    return value


def check_index(index: dict, codebooks: int, codebook_size: int) -> tuple[dict, set]:
    """Validate token shape and return full SIDs plus the distinct component tokens."""
    expected_prefixes = [chr(ord("a") + i) for i in range(codebooks)]
    full_sids: dict[str, str] = {}
    component_tokens: set[str] = set()

    for item_id, tokens in index.items():
        if len(tokens) != codebooks:
            raise ValueError(f"item {item_id}: expected {codebooks} SID tokens, got {len(tokens)}")
        for position, token in enumerate(tokens):
            match = SID_TOKEN.fullmatch(token)
            if match is None:
                raise ValueError(f"item {item_id}: malformed SID token {token!r}")
            prefix, raw_code = match.groups()
            if prefix != expected_prefixes[position]:
                raise ValueError(f"item {item_id}: token {token!r} is in the wrong SID position")
            if not 0 <= int(raw_code) < codebook_size:
                raise ValueError(f"item {item_id}: token {token!r} exceeds codebook size")
            component_tokens.add(token)
        full_sids[item_id] = "".join(tokens)
    return full_sids, component_tokens


def split_target(raw: str, tree: str) -> list[str]:
    """A next-item target is one SID; a two-item target is 'sid_A ||| sid_B'."""
    if tree == "two-item":
        return [part.strip() for part in raw.split(TWO_ITEM_SEP)]
    return [raw]


def check_rows(paths: dict, full_sids: dict, tree: str) -> tuple[dict, dict]:
    row_counts: dict[str, int] = {}
    history_counts: dict[str, int] = {}

    for split in ("train", "valid", "test"):
        rows = 0
        histories = 0
        with Path(paths[split]).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
                missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
                raise ValueError(f"{split}: missing columns {missing}")
            for row_number, row in enumerate(reader, start=2):
                targets = split_target(row["item_sid"], tree)
                item_ids = split_target(str(row["item_id"]), tree)
                if len(targets) != len(item_ids):
                    raise ValueError(
                        f"{split} row {row_number}: {len(item_ids)} item ids but {len(targets)} sids"
                    )
                for item_id, target in zip(item_ids, targets):
                    expected = full_sids.get(item_id)
                    if expected is None:
                        raise ValueError(f"{split} row {row_number}: unknown target item {item_id}")
                    if target != expected:
                        raise ValueError(
                            f"{split} row {row_number}: target SID does not match index"
                        )

                history_ids = load_literal_list(row["history_item_id"], "history_item_id", split, row_number)
                history_sids = load_literal_list(row["history_item_sid"], "history_item_sid", split, row_number)
                history_titles = load_literal_list(row["history_item_title"], "history_item_title", split, row_number)
                if not (len(history_ids) == len(history_sids) == len(history_titles)):
                    raise ValueError(f"{split} row {row_number}: history fields have different lengths")
                for history_id, history_sid in zip(history_ids, history_sids):
                    expected_history_sid = full_sids.get(str(history_id))
                    if expected_history_sid is None or history_sid != expected_history_sid:
                        raise ValueError(f"{split} row {row_number}: history SID does not match index")
                rows += 1
                histories += len(history_ids)
        row_counts[split] = rows
        history_counts[split] = histories
    return row_counts, history_counts


def check_info(info_path: Path, full_sids: dict, tree: str) -> None:
    catalog_sids = []
    with info_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"info line {line_number}: expected at least three tab-separated fields")
            catalog_sids.append(fields[0])
    if Counter(catalog_sids) != Counter(full_sids.values()):
        raise ValueError("info catalog SIDs do not match the full index")


def validate(args: argparse.Namespace) -> dict:
    variant = V.Variant(args.tree, args.category, args.method, args.codebooks, args.codebook_size)
    paths = V.resolve(str(args.data_root), variant)

    with Path(paths["index"]).open() as handle:
        index = json.load(handle)
    with Path(paths["item"]).open() as handle:
        items = json.load(handle)

    if set(index) != set(items):
        raise ValueError("SID index and item metadata keys do not match")

    full_sids, component_tokens = check_index(index, args.codebooks, args.codebook_size)

    unique_full = len(set(full_sids.values()))
    collisions = len(full_sids) - unique_full
    capacity = args.codebooks * args.codebook_size

    summary = {
        "variant_id": variant.variant_id,
        "tree": args.tree,
        "category": args.category,
        "method": args.method,
        "codebooks": args.codebooks,
        "codebook_size": args.codebook_size,
        "items": len(index),
        "unique_full_sids": unique_full,
        "collisions": collisions,
        "collision_rate": round(collisions / len(index), 6) if index else 0.0,
        "component_tokens": len(component_tokens),
        "codebook_utilization": round(len(component_tokens) / capacity, 6),
        "paths": paths,
        "status": "ok",
    }

    check_info(Path(paths["info"]), full_sids, args.tree)
    if not args.skip_rows:
        row_counts, history_counts = check_rows(paths, full_sids, args.tree)
        summary["rows"] = row_counts
        summary["history_events"] = history_counts
    return summary


def main() -> None:
    args = parse_args()
    summary = validate(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
