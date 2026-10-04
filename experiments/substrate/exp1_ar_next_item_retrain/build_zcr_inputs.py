#!/usr/bin/env python
"""Build ZCR versions of the next-item training inputs (protocol.md, arm B).

ZCR (Zhang et al., arXiv 2605.25330) changes only the last digit of items that
share a SID. To train a ZCR model with exactly the native recipe, every place
the recipe reads a SID must use the ZCR table: the train / valid / test CSVs
(`history_item_sid`, `item_sid`), the index JSON and the decoding info file.
Titles, item ids, rows and their order stay byte-for-byte the same.

Traps this guards against
-------------------------
* **A bundle built from a different native table.** The bundle's
  `native_index.json` must equal the frozen index item by item.
* **A table that is not a ZCR of the native one.** Every item keeps its first
  D-1 tokens, and all 3,105 ZCR SIDs are distinct.
* **A CSV whose SIDs disagree with its item ids.** Every SID in every row is
  checked against the native table before it is replaced; any mismatch stops
  the build.
* **Silently reformatting list columns.** Lists are written back in the same
  Python-literal form upstream `eval()`s.

    python build_zcr_inputs.py --variant rqkmeans_3codebook_128 \
        --bundle /l/users/leo.rodrigues/onediffrec/handoff/zcr-release-v1/zcr/industrial/rqkmeans_3codebook_128 \
        --out $SIDLENS_WORK/derived/substrate/zcr_inputs/20261004
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from sidlens import paths
from sidlens.provenance.hashing import sha256_file

import run  # frozen_inputs, parse_variant (same folder)


def sid_str(tokens: list[str]) -> str:
    return "".join(tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    method, cb, sz = run.parse_variant(a.variant)
    if method != "rqkmeans":
        raise SystemExit("ZCR bundles exist for RQ-KMeans only")
    frozen = run.frozen_inputs(a.variant)
    native = json.loads(frozen["index"][0].read_text())
    bundle_native = json.loads((a.bundle / "native_index.json").read_text())
    zcr = json.loads((a.bundle / "index.json").read_text())
    if bundle_native != native:
        raise SystemExit("bundle native_index.json differs from the frozen index")
    if set(zcr) != set(native):
        raise SystemExit("ZCR index covers different item ids")
    for k, toks in zcr.items():
        if len(toks) != cb or toks[:-1] != native[k][:-1]:
            raise SystemExit(f"item {k}: ZCR changed a prefix digit or the length")
    if len({sid_str(t) for t in zcr.values()}) != len(zcr):
        raise SystemExit("ZCR SIDs are not all distinct")
    nat_s = {int(k): sid_str(v) for k, v in native.items()}
    zcr_s = {int(k): sid_str(v) for k, v in zcr.items()}
    changed = sorted(i for i in nat_s if nat_s[i] != zcr_s[i])

    out = a.out / a.variant
    out.mkdir(parents=True, exist_ok=False)
    report = {"variant": a.variant, "bundle": str(a.bundle),
              "bundle_manifest_sha256": sha256_file(a.bundle / "manifest.json"),
              "bundle_index_sha256": sha256_file(a.bundle / "index.json"),
              "items": len(zcr), "items_changed": len(changed), "files": {}}
    for split in ("train", "valid", "test"):
        src = frozen[split][0]
        df = pd.read_csv(src, dtype=str, keep_default_na=False)
        n_rows_changed = 0
        hist_sids, tgt_sids = [], []
        for hid, hsid, iid, isid in zip(df["history_item_id"], df["history_item_sid"],
                                        df["item_id"], df["item_sid"]):
            ids = ast.literal_eval(hid)
            sids = ast.literal_eval(hsid)
            if [nat_s[int(i)] for i in ids] != sids or nat_s[int(iid)] != isid:
                raise SystemExit(f"{split}: a row's SIDs disagree with its item ids under the native table")
            new_h = [zcr_s[int(i)] for i in ids]
            new_t = zcr_s[int(iid)]
            n_rows_changed += (new_h != sids) or (new_t != isid)
            hist_sids.append(repr(new_h))
            tgt_sids.append(new_t)
        df["history_item_sid"] = hist_sids
        df["item_sid"] = tgt_sids
        dst = out / f"{split}.csv"
        df.to_csv(dst, index=False)
        back = pd.read_csv(dst, dtype=str, keep_default_na=False)
        orig = pd.read_csv(src, dtype=str, keep_default_na=False)
        keep = [c for c in orig.columns if c not in ("history_item_sid", "item_sid")]
        if list(back.columns) != list(orig.columns) or not back[keep].equals(orig[keep]):
            raise SystemExit(f"{split}: rewriting changed a non-SID column")
        report["files"][split] = {"source": str(src), "source_sha256": sha256_file(src),
                                  "path": str(dst), "sha256": sha256_file(dst),
                                  "rows": len(df), "rows_changed": int(n_rows_changed)}
    info_src = frozen["info"][0]
    lines = []
    for ln in info_src.read_text().splitlines():
        sid, title, iid = ln.split("\t")
        if nat_s[int(iid)] != sid:
            raise SystemExit("info file disagrees with the native table")
        lines.append(f"{zcr_s[int(iid)]}\t{title}\t{iid}")
    (out / "info.txt").write_text("\n".join(lines) + "\n")
    (out / "index.json").write_text(json.dumps(zcr))
    for role, name in (("info", "info.txt"), ("index", "index.json")):
        report["files"][role] = {"path": str(out / name), "sha256": sha256_file(out / name)}
    (out / "manifest.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("variant", "items_changed")}),
          {s: report["files"][s]["rows_changed"] for s in ("train", "valid", "test")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
