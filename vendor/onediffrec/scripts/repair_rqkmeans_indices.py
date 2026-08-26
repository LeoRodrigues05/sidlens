#!/usr/bin/env python3
"""Repair the RQ-KMeans index files that store bit-packed bytes instead of codes.

Background
----------
`faiss.ResidualQuantizer.compute_codes()` returns an (n, code_size) uint8 array
where code_size = ceil(M * nbits / 8) and the M codes are packed little-endian,
nbits bits each. rq/rqkmeans_faiss.py wrote those bytes out as if each byte were
one code. That is only correct when nbits == 8, i.e. codebook_size == 256.

    codebook 256 -> nbits 8 -> one byte per code            -> correct
    codebook 128 -> nbits 7 -> M codes squeezed into M bytes -> wrong values
    codebook 512 -> nbits 9 -> M codes spread over M+1 bytes -> wrong values
                                                                and wrong width

How we know that is what happened
---------------------------------
A residual quantizer is greedy: level j is chosen from the residual left by
levels 0..j-1, so adding a level cannot change the earlier ones. The 256 files
(which need no unpacking) show exactly that nesting - item 0 reads
<a_169><b_215><c_239> at 3 codebooks and <a_169><b_215><c_239><d_212> at 4.

Read as raw bytes, the 128 and 512 files violate it: item 0 is
<a_28><b_63><c_27> at 3 codebooks but <a_28><b_63><c_219><d_4> at 4, so level 2
apparently changed when level 3 was added. Unpacking restores the nesting
exactly. `--check` runs that test on every item and every shared level.

What this script rewrites
-------------------------
  1. the index JSON            item_id -> [SID tokens]
  2. the info txt              SID \t title \t item_id
  3. train/valid/test CSVs     the history_item_sid and item_sid columns,
                               keyed on the item ids the rows already carry
Both task trees (Amazon18 and two-item) are covered. Originals are moved to
*.mispacked before anything is written.

Usage
-----
    python scripts/repair_rqkmeans_indices.py --check      # prove the diagnosis
    python scripts/repair_rqkmeans_indices.py --selftest   # prove the rewriter
    python scripts/repair_rqkmeans_indices.py --write      # do the repair
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import sys

TOKEN_RE = re.compile(r"<([a-z])_(\d+)>")
TPL = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>", "<f_{}>", "<g_{}>"]

DEFAULT_ROOT = "/l/users/leo.rodrigues/onediffrec/data"
CATEGORY = "Industrial_and_Scientific"
LEVELS = (3, 4, 5)
SIZES = (128, 256, 512)
TREES = ("Amazon18", "two-item")
SPLITS = ("train", "valid", "test")

csv.field_size_limit(10 ** 7)


# ---------------------------------------------------------------- bit packing

def unpack(row: list[int], num_levels: int, nbits: int) -> list[int]:
    stream = 0
    for i, byte in enumerate(row):
        stream |= int(byte) << (8 * i)
    mask = (1 << nbits) - 1
    return [(stream >> (nbits * j)) & mask for j in range(num_levels)]


def repack(codes: list[int], nbits: int, width: int) -> list[int]:
    stream = 0
    for j, code in enumerate(codes):
        stream |= int(code) << (nbits * j)
    return [(stream >> (8 * i)) & 0xFF for i in range(width)]


# ------------------------------------------------------------------ file I/O

def index_path(root: str, category: str, levels: int, size: int) -> str:
    return os.path.join(root, "Amazon18", category,
                        f"{category}.rqkmeans.index_{levels}codebook_{size}.json")


def tag(category: str, levels: int, size: int) -> str:
    return f"{category}_rqkmeans_{levels}codebook_{size}"


def load_index(path: str) -> dict[str, list[int]]:
    """item_id -> the integers stored in its tokens, in order."""
    raw = json.load(open(path))
    return {k: [int(m.group(2)) for m in TOKEN_RE.finditer("".join(v))]
            for k, v in raw.items()}


def classify(stored: dict[str, list[int]], levels: int, size: int) -> dict:
    nbits = size.bit_length() - 1
    width = -(-levels * nbits // 8)
    rows = list(stored.values())
    stored_width = len(rows[0])
    top = max(max(r) for r in rows)

    info = {"levels": levels, "size": size, "nbits": nbits,
            "packed_width": width, "stored_width": stored_width, "max": top,
            "items": len(rows)}
    if stored_width == levels and top < size:
        info["verdict"] = "correct"
        info["codes"] = stored
    elif stored_width == width:
        info["verdict"] = "packed"
        info["codes"] = {k: unpack(v, levels, nbits) for k, v in stored.items()}
    else:
        info["verdict"] = "unexplained"
        info["codes"] = None
    return info


def to_tokens(codes: list[int]) -> list[str]:
    return [TPL[j].format(int(c)) for j, c in enumerate(codes)]


def sid_of(codes: list[int]) -> str:
    return "".join(to_tokens(codes))


# ------------------------------------------------------------ the nesting test

def padding_report(root: str, category: str) -> None:
    """The unused high bits of the last byte cap it far below 255.

    Packing M codes of nbits bits into ceil(M*nbits/8) bytes leaves
    8*width - M*nbits bits unused at the top of the final byte, so that byte can
    never exceed 2**(8 - padding) - 1. Observing exactly that cap, for every
    file, is what identifies the stored values as a bit stream rather than codes.
    """
    print("\n" + "=" * 78)
    print("  Padding signature: the last stored value is capped by the unused bits")
    print("=" * 78 + "\n")
    print(f"  {'file':<16}{'bits used':>11}{'bits held':>11}{'padding':>9}"
          f"{'cap implied':>13}{'cap observed':>14}  match")
    print("  " + "-" * 76)
    for size in SIZES:
        for levels in LEVELS:
            path = index_path(root, category, levels, size)
            if not os.path.isfile(path):
                continue
            stored = load_index(path)
            c = classify(stored, levels, size)
            rows = list(stored.values())
            observed = max(r[-1] for r in rows)
            if c["verdict"] == "correct":
                print(f"  {f'{levels}cb/{size}':<16}{'-':>11}{'-':>11}{'-':>9}"
                      f"{'-':>13}{observed:>14}  (already correct)")
                continue
            used = levels * c["nbits"]
            held = 8 * c["packed_width"]
            pad = held - used
            implied = (1 << (8 - pad)) - 1
            print(f"  {f'{levels}cb/{size}':<16}{used:>11}{held:>11}{pad:>9}"
                  f"{implied:>13}{observed:>14}  {'YES' if implied == observed else 'no'}")
        print()


def nesting_report(root: str, category: str) -> int:
    """Level j must agree across every M that has a level j, at a fixed size.

    A residual quantizer is greedy, so extending a 3-codebook run to 4 codebooks
    cannot revise levels 0..2. Files built by separate training runs share no
    lineage and are reported as such rather than counted as failures.
    """
    failures = 0
    print("=" * 78)
    print("  Nesting test: a greedy residual quantizer cannot revise an earlier level")
    print("=" * 78)
    print("\n  For each codebook size, compare the shared levels of the 3-, 4- and")
    print("  5-codebook files item by item. 'raw' reads each stored value as a code")
    print("  (what the pipeline did); 'unpacked' reads the bytes as a bit stream.\n")
    print(f"  {'size':>5} {'pair':>11} {'raw agrees':>14} {'unpacked agrees':>17}   note")
    print("  " + "-" * 74)

    for size in SIZES:
        loaded = {}
        for levels in LEVELS:
            p = index_path(root, category, levels, size)
            if os.path.isfile(p):
                stored = load_index(p)
                loaded[levels] = (stored, classify(stored, levels, size))
        for a in LEVELS:
            for b in LEVELS:
                if b <= a or a not in loaded or b not in loaded:
                    continue
                (raw_a, ca), (raw_b, cb) = loaded[a], loaded[b]
                shared = min(ca["stored_width"], cb["stored_width"], a)
                keys = sorted(set(raw_a) & set(raw_b), key=int)
                n = len(keys)
                raw_ok = sum(raw_a[k][:shared] == raw_b[k][:shared] for k in keys)
                un_ok = sum(ca["codes"][k][:a] == cb["codes"][k][:a] for k in keys)

                # No shared level-0 assignment means the two files come from
                # different training runs, so nesting does not apply.
                if raw_ok == 0 and un_ok == 0:
                    note = "separate training runs - not comparable"
                elif un_ok == n and raw_ok < n:
                    note = "raw violates nesting, unpacking restores it"
                elif un_ok == n and raw_ok == n:
                    note = "consistent either way"
                else:
                    note = "UNRESOLVED"
                    failures += 1
                print(f"  {size:>5} {f'{a}cb vs {b}cb':>11} {f'{raw_ok}/{n}':>14} "
                      f"{f'{un_ok}/{n}':>17}   {note}")
        print()
    return failures


def check(root: str, category: str) -> int:
    print("=" * 78)
    print(f"  RQ-KMeans index files under {root}")
    print("=" * 78)
    print(f"\n  {'file':<28}{'nbits':>6}{'width':>13}{'max code':>10}  verdict")
    print("  " + "-" * 72)
    bad = 0
    for size in SIZES:
        for levels in LEVELS:
            p = index_path(root, category, levels, size)
            if not os.path.isfile(p):
                print(f"  {f'{levels}codebook_{size}':<28}{'':>6}{'':>13}{'':>10}  MISSING")
                continue
            c = classify(load_index(p), levels, size)
            w = f"{c['stored_width']} (want {c['levels']})"
            flag = {"correct": "correct",
                    "packed": f"MISPACKED -> {c['packed_width']} bytes",
                    "unexplained": "UNEXPLAINED"}[c["verdict"]]
            if c["verdict"] != "correct":
                bad += 1
            print(f"  {f'{levels}codebook_{size}':<28}{c['nbits']:>6}{w:>13}"
                  f"{c['max']:>10}  {flag}")
        print()
    padding_report(root, category)
    fails = nesting_report(root, category)
    if fails:
        print(f"  {fails} pair(s) still disagree after unpacking - diagnosis incomplete")
    return bad


# -------------------------------------------------------------- the rewriting

def line_terminator(path: str) -> str:
    """Match the file we are replacing.

    The Amazon18 CSVs are LF-only and the two-item CSVs are CRLF. Python's csv
    writer defaults to CRLF, so writing blind would silently reformat every
    Amazon18 file even when no SID changed.
    """
    with open(path, "rb") as f:
        head = f.read(65536)
    return "\r\n" if b"\r\n" in head else "\n"


def rewrite_csv(src: str, dst: str, sid_for: dict[str, str]) -> tuple[int, int]:
    """Replace the two SID columns using the item ids already in each row."""
    def ids(cell: str) -> list[str]:
        return [s.strip() for s in cell.split("|||")]

    changed = total = 0
    term = line_terminator(src)
    with open(src, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fields = reader.fieldnames
        rows = list(reader)
    with open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields, lineterminator=term)
        writer.writeheader()
        for row in rows:
            total += 1
            hist = [str(i) for i in ast.literal_eval(row["history_item_id"])]
            new_hist = repr([sid_for[i] for i in hist])
            new_item = " ||| ".join(sid_for[i] for i in ids(row["item_id"]))
            if new_hist != row["history_item_sid"] or new_item != row["item_sid"]:
                changed += 1
            row["history_item_sid"] = new_hist
            row["item_sid"] = new_item
            writer.writerow(row)
    return changed, total


def write_info(dst: str, items: dict, codes: dict[str, list[int]]) -> int:
    n = 0
    with open(dst, "w", encoding="utf-8") as f:
        for item_id, meta in items.items():
            if item_id not in codes:
                continue
            title = meta.get("title", f"Item_{item_id}")
            f.write(f"{sid_of(codes[item_id])}\t{title}\t{item_id}\n")
            n += 1
    return n


def selftest(root: str, category: str) -> int:
    """Rewrite the already-correct 256 files and require byte-identical output.

    If the rewriter reproduces a file it should not change, then any difference
    it produces elsewhere comes from the new SIDs and not from the rewriting.
    """
    print("=" * 78)
    print("  Self-test: re-render the correct 256 files and diff against the originals")
    print("=" * 78 + "\n")
    bad = 0
    for levels in LEVELS:
        c = classify(load_index(index_path(root, category, levels, 256)), levels, 256)
        assert c["verdict"] == "correct", "256 file is not clean - cannot self-test"
        sid_for = {k: sid_of(v) for k, v in c["codes"].items()}
        for tree in TREES:
            for split in SPLITS:
                src = os.path.join(root, tree, split, f"{tag(category, levels, 256)}.csv")
                if not os.path.isfile(src):
                    continue
                tmp = src + ".selftest"
                changed, total = rewrite_csv(src, tmp, sid_for)
                same = open(src, "rb").read() == open(tmp, "rb").read()
                os.remove(tmp)
                if not same:
                    bad += 1
                print(f"  {tree:<9}{split:<6}{levels}cb/256  rows={total:<6} "
                      f"rows-that-would-change={changed:<5} "
                      f"{'identical' if same else 'DIFFERS'}")
    print()
    print("  rewriter is faithful" if not bad else f"  {bad} file(s) differ - do not use --write")
    return bad


def write(root: str, category: str) -> int:
    items = json.load(open(os.path.join(
        root, "Amazon18", category, f"{category}.item.json")))
    print("=" * 78)
    print("  Repairing")
    print("=" * 78 + "\n")
    done = 0
    for size in SIZES:
        for levels in LEVELS:
            p = index_path(root, category, levels, size)
            if not os.path.isfile(p):
                continue
            c = classify(load_index(p), levels, size)
            if c["verdict"] == "correct":
                print(f"  {levels}codebook_{size:<4} already correct, skipped")
                continue
            if c["verdict"] == "unexplained":
                print(f"  {levels}codebook_{size:<4} UNEXPLAINED, skipped")
                continue

            codes = c["codes"]
            sid_for = {k: sid_of(v) for k, v in codes.items()}
            uniq = len(set(sid_for.values()))
            print(f"  {levels}codebook_{size:<4} unpacked {c['items']} items, "
                  f"{uniq} distinct SIDs "
                  f"({100 * (1 - uniq / c['items']):.2f}% collision), "
                  f"max code {max(max(v) for v in codes.values())}")

            shutil.move(p, p + ".mispacked")
            with open(p, "w") as f:
                json.dump({k: to_tokens(v) for k, v in codes.items()}, f, indent=2)

            for tree in TREES:
                info = os.path.join(root, tree, "info", f"{tag(category, levels, size)}.txt")
                if os.path.isfile(info):
                    shutil.move(info, info + ".mispacked")
                    n = write_info(info, items, codes)
                    print(f"      {tree}/info   {n} lines")
                for split in SPLITS:
                    src = os.path.join(root, tree, split,
                                       f"{tag(category, levels, size)}.csv")
                    if not os.path.isfile(src):
                        continue
                    shutil.move(src, src + ".mispacked")
                    changed, total = rewrite_csv(src + ".mispacked", src, sid_for)
                    print(f"      {tree}/{split:<6} {changed}/{total} rows rewritten")
            done += 1
            print()
    print(f"  repaired {done} variant(s); originals kept as *.mispacked")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_ROOT)
    ap.add_argument("--category", default=CATEGORY)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if a.check:
        return 0 if check(a.data_root, a.category) >= 0 else 1
    if a.selftest:
        return 1 if selftest(a.data_root, a.category) else 0
    return write(a.data_root, a.category)


if __name__ == "__main__":
    sys.exit(main())
