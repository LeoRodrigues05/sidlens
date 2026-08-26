#!/usr/bin/env python3
"""Demonstrate that the shipped RQ-KMeans index files store bit-packed bytes, not codes.

Everything is read straight out of OneDiffRec_data.tar.gz, so the result does not
depend on any working tree, extraction step, or local edit. Only the Python
standard library is required - no faiss, no torch, no GPU.

The proof is a round trip. `faiss.ResidualQuantizer.compute_codes()` returns codes
bit-packed at nbits = log2(codebook_size) bits each, little-endian, padded up to
whole bytes. rq/rqkmeans_faiss.py reshapes that byte string to (-1, num_levels)
and writes the bytes out as if they were codes, which is only correct at
codebook_size 256 (nbits == 8). If that is what happened, then for every item:

    unpack(stored_bytes) -> M codes, each < codebook_size
    repack(those codes)  -> exactly the stored bytes again

A bit-exact round trip over every item leaves no other reading of the file.

Usage:
    python scripts/demo_rqkmeans_packing.py [--archive PATH]

Exits 0 if every affected file round-trips (the defect is confirmed as described)
and 1 if any file fails to, which would mean the explanation is incomplete.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile

TOKEN_RE = re.compile(r"<[a-z]_(\d+)>")

CATEGORIES = ("Industrial_and_Scientific", "Office")
LEVELS = (3, 4, 5)
SIZES = (128, 256, 512)


def unpack(values: list[int], num_levels: int, nbits: int) -> list[int]:
    """Read a little-endian bit stream of `num_levels` codes of `nbits` bits."""
    stream = 0
    for i, byte in enumerate(values):
        stream |= byte << (8 * i)
    mask = (1 << nbits) - 1
    return [(stream >> (nbits * j)) & mask for j in range(num_levels)]


def repack(codes: list[int], nbits: int, width: int) -> list[int]:
    """Inverse of unpack: pack codes into `width` bytes, little-endian."""
    stream = 0
    for j, code in enumerate(codes):
        stream |= code << (nbits * j)
    return [(stream >> (8 * i)) & 0xFF for i in range(width)]


def harvest(archive_path: str, wanted: set[str]) -> dict[str, bytes]:
    """Pull every wanted member out in a single forward pass.

    A .tar.gz is a gzip stream, so seeking backwards to a member means
    re-inflating from the start. Asking for members one at a time turns an
    18-file read of a 1.6 GB archive into 18 full decompressions. Streaming
    once and stopping as soon as everything is collected keeps it to seconds.
    """
    found: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r|gz") as stream:   # "r|" = forward-only
        for member in stream:
            if member.name in wanted:
                handle = stream.extractfile(member)
                if handle is not None:
                    found[member.name] = handle.read()
                if len(found) == len(wanted):
                    break
    return found


def index_member(category: str, levels: int, size: int) -> str:
    return (f"data/Amazon18/{category}/{category}"
            f".rqkmeans.index_{levels}codebook_{size}.json")


def info_member(tree: str, levels: int, size: int) -> str:
    return (f"data/{tree}/info/Industrial_and_Scientific"
            f"_rqkmeans_{levels}codebook_{size}.txt")


def codes_of(sids: list[str]) -> list[int]:
    return [int(m) for token in sids for m in TOKEN_RE.findall(token)]


def check(blobs: dict[str, bytes], category: str, levels: int, size: int) -> dict:
    index = json.loads(blobs[index_member(category, levels, size)])
    rows = [codes_of(sids) for sids in index.values()]

    nbits = size.bit_length() - 1              # 128->7, 256->8, 512->9
    width = -(-levels * nbits // 8)            # ceil(levels * nbits / 8)
    stored_width = len(rows[0])

    result = {
        "category": category, "levels": levels, "size": size,
        "nbits": nbits, "packed_width": width, "stored_width": stored_width,
        "items": len(rows),
    }

    # A correctly written file has one value per level, each below the codebook size.
    if stored_width == levels and all(max(r) < size for r in rows):
        result["verdict"] = "correct"
        result["digits"] = levels
        result["alphabet"] = [size] * levels
        return result

    if stored_width != width:
        result["verdict"] = "unexplained"
        return result

    # Round trip: unpack to codes, repack, compare against the stored bytes.
    mismatches = 0
    out_of_range = 0
    per_level: list[set[int]] = [set() for _ in range(levels)]
    for row in rows:
        codes = unpack(row, levels, nbits)
        if any(c >= size for c in codes):
            out_of_range += 1
        for j, c in enumerate(codes):
            per_level[j].add(c)
        if repack(codes, nbits, width) != row:
            mismatches += 1

    result["verdict"] = "packed" if mismatches == 0 and out_of_range == 0 else "unexplained"
    result["mismatches"] = mismatches
    result["out_of_range"] = out_of_range
    result["digits"] = stored_width
    result["alphabet"] = [1 << max(v).bit_length() for v in
                          [set(r[i] for r in rows) for i in range(stored_width)]]
    result["recovered_alphabet"] = [len(v) for v in per_level]
    return result


def propagation(blobs: dict[str, bytes], affected: list[dict]) -> None:
    """Show that the mispacking reached the files SFT and evaluation actually read."""
    print("\n" + "=" * 78)
    print("  Propagation into the files SFT and evaluation actually read")
    print("=" * 78)
    print("\n  The info file is what evaluate.py builds its decoding trie from, and the")
    print("  train/valid/test CSVs carry the same SIDs. Both are generated from the")
    print("  index by convert_dataset.py, so the defect is baked into them too.\n")
    print(f"  {'index file':<22} {'tree':<9} {'first SID in info':<40} {'digits':>7} {'max':>5}")
    print("  " + "-" * 74)
    for r in affected:
        levels, size = r["levels"], r["size"]
        for tree in ("Amazon18", "two-item"):
            blob = blobs.get(info_member(tree, levels, size))
            if blob is None:
                continue
            text = blob.decode("utf-8").splitlines()
            sid = text[0].split("\t")[0].strip()
            n = len(TOKEN_RE.findall(sid))
            hi = max(int(c) for line in text
                     for c in TOKEN_RE.findall(line.split("\t")[0]))
            flag = ""
            if n != levels:
                flag = f"  <- claims {levels}"
            elif hi >= size:
                flag = f"  <- code {hi} >= {size}"
            print(f"  {f'{levels}codebook_{size}':<22} {tree:<9} {sid:<40} "
                  f"{n:>7} {hi:>5}{flag}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", default="data/OneDiffRec_data.tar.gz",
                        help="path to OneDiffRec_data.tar.gz")
    args = parser.parse_args()

    wanted = {index_member(c, l, s)
              for c in CATEGORIES for l in LEVELS for s in SIZES}
    wanted |= {info_member(t, l, s)
               for t in ("Amazon18", "two-item") for l in LEVELS for s in SIZES}

    print("=" * 78)
    print("  RQ-KMeans index files: filename claim vs. stored contents")
    print(f"  source: {args.archive} (read directly, nothing extracted)")
    print("=" * 78)
    print(f"\n  reading {len(wanted)} members in one pass ...", flush=True)
    blobs = harvest(args.archive, wanted)
    print(f"  got {len(blobs)}\n")

    print(f"  {'category':<26} {'file claims':<16} {'stored as':<22} verdict")
    print("  " + "-" * 74)

    affected: list[dict] = []
    unexplained: list[dict] = []

    for category in CATEGORIES:
        for levels in LEVELS:
            for size in SIZES:
                r = check(blobs, category, levels, size)
                claim = f"{levels} digits x {size}"
                if r["verdict"] == "correct":
                    stored = f"{r['digits']} digits x {size}"
                    note = "correct"
                else:
                    alpha = " x ".join(str(a) for a in r["alphabet"])
                    stored = f"{r['digits']} digits: {alpha}"
                    note = ("MISPACKED (round trip exact)"
                            if r["verdict"] == "packed" else "UNEXPLAINED")
                    (affected if r["verdict"] == "packed" else unexplained).append(r)
                print(f"  {category:<26} {claim:<16} {stored:<22} {note}")
        print()

    print("=" * 78)
    print("  Round-trip verification on the affected files")
    print("=" * 78)
    print("\n  For each item: unpack the stored bytes into N codes of `nbits` bits,")
    print("  then repack them. If the result is byte-identical to what is on disk,")
    print("  the stored values are bit-packed codes and nothing else.\n")
    print(f"  {'file':<34} {'items':>6} {'nbits':>6} {'bytes':>6} {'exact':>10} {'in range':>11}")
    print("  " + "-" * 74)
    for r in affected:
        name = f"{r['category'][:11]} {r['levels']}codebook_{r['size']}"
        exact = r["items"] - r["mismatches"]
        inrange = r["items"] - r["out_of_range"]
        print(f"  {name:<34} {r['items']:>6} {r['nbits']:>6} {r['packed_width']:>6} "
              f"{exact:>6}/{r['items']:<3} {inrange:>7}/{r['items']:<3}")

    if affected:
        propagation(blobs, affected)

    print("\n" + "=" * 78)
    total_items = sum(r["items"] for r in affected)
    bad_round_trips = sum(r["mismatches"] + r["out_of_range"] for r in affected)
    print(f"  {len(affected)} mispacked files, {total_items} items round-tripped, "
          f"{bad_round_trips} failures")
    if unexplained:
        print(f"  {len(unexplained)} file(s) NOT explained by bit packing - "
              "the diagnosis is incomplete")
        return 1
    if bad_round_trips:
        print("  round trip failed - the diagnosis is incomplete")
        return 1
    print("  Every mispacked file round-trips exactly. The stored values are the")
    print("  bit-packed form of the quantizer's codes, so the original codes are")
    print("  fully recoverable without retraining.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
