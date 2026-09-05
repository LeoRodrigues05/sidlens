#!/usr/bin/env python
"""Recover the attribute labels upstream stripped, from the public 2018 dump.

Why this script has to exist
----------------------------
`frozen/data/item_meta/<category>.item.json` carries four fields -- title,
description, brand, categories -- and `categories` is the empty string for
3105/3105 Industrial items. Upstream's preprocessing kept the two fields the
embedding needed and blanked the rest. That is fine for training and fatal for
Experiment 1, which asks what each SID digit MEANS: with no attribute labels
there is nothing to measure digit alignment against, and the atlas can only
describe digit STRUCTURE.

The lineage is recoverable. Review timestamps in the frozen substrate run
2013-10-02 to 2018-09-28 and the upstream data tree is named `Amazon18`, so this
is Amazon Review Data 2018 (Ni, Li & McAuley), not the 2014 dump the DiffGRM
cache directory name claims. The per-category metadata file for that release
still joins on ASIN at 100%.

What is recovered, and what it is worth
---------------------------------------
    category      hierarchical breadcrumb. Level 0 is the constant
                  "Industrial & Scientific" and is dropped. Levels 1/2/3 give
                  26 / 182 / 373 classes -- the first real probe targets in the
                  project.
    main_cat      Amazon's single top-level store. Cuts across `category`:
                  an item can sit in the Industrial breadcrumb while being sold
                  under "Tools & Home Improvement".
    price, rank   the only continuous targets available.
    also_buy      co-purchase graph -- a COLLABORATIVE neighbour signal that
                  owes nothing to the item text, and therefore the one label
                  here that can separate "this digit encodes meaning" from
                  "this digit encodes the text encoder".
    feature,
    details       free text carrying material/size/colour for the items that
                  have it. Not labels yet; the raw material for extraction.

Discipline
----------
Output goes to $SIDLENS_WORK/external/, NEVER to frozen/. The frozen substrate
is defined as the bytes the models were trained on; this data was pulled after
the fact and joining it in would destroy that guarantee. The two are kept apart
so that "the model saw this" and "we looked this up later" can never be
confused. The emitted manifest records the source URL, its sha256, the join
rate, and per-field coverage.

The frozen title/description remain authoritative for anything embedding-
related. The 2018 dump's own title/description are recorded for comparison only
-- they are not always identical, and silently swapping them would change what
`z_i` is supposed to represent.

Usage
-----
    python scripts/build_labels.py --category Industrial_and_Scientific
    python scripts/build_labels.py --check      # re-join, report, write nothing
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidlens import paths                                    # noqa: E402
from sidlens.data.sids import load_item2id                   # noqa: E402
from sidlens.provenance import hashing                       # noqa: E402

SOURCE_URL = ("https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/"
              "metaFiles2/meta_{category}.json.gz")

# Level 0 of every breadcrumb is the category name itself and carries zero
# information. Kept out of the emitted fields rather than emitted and ignored,
# so nobody probes a constant and reports 100% accuracy.
BREADCRUMB_SKIP = 1

RE_PRICE = re.compile(r"^\$([\d,]+\.\d{2})$")
RE_RANK = re.compile(r"([\d,]+)\s+in\s+(.+?)\s*\(")
RE_RANK_ALT = re.compile(r">?#([\d,]+)\s+in\s+(.+?)\s*(?:\(|$)")


def parse_price(raw) -> float | None:
    """Dollars as a float, or None.

    2605/3105 items carry a price and 41 of those are a leaked CSS blob
    (`.a-box-inner{background-color:#fff}...`) rather than a number. 19 more are
    a range. Only the unambiguous `$X.XX` form is accepted; a range midpoint
    would be a fabricated value in a column meant to be measured.
    """
    if not isinstance(raw, str):
        return None
    m = RE_PRICE.match(raw.strip())
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def parse_rank(raw) -> tuple[int | None, str | None]:
    """(rank, store) for the item's PRIMARY sales rank.

    The field is a string for roughly half the catalogue and a list of
    "#N in Store > Sub > Sub" strings for the other half. The first entry is
    the store-wide rank; later entries are subcategory ranks whose denominators
    differ per item and are therefore not comparable across the catalogue.
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if not isinstance(raw, str) or not raw:
        return None, None
    for pat in (RE_RANK_ALT, RE_RANK):
        m = pat.search(raw)
        if m:
            return int(m.group(1).replace(",", "")), m.group(2).strip()
    return None, None


def build(category: str, write: bool = True) -> dict:
    src = paths.EXTERNAL / "amazon2018" / f"meta_{category}.json.gz"
    if not src.exists():
        raise SystemExit(
            f"missing {src}\n"
            f"download it first:\n"
            f"  mkdir -p {src.parent}\n"
            f"  curl -sSL -o {src} {SOURCE_URL.format(category=category)}")

    item2id = load_item2id(category)
    wanted = set(item2id)
    print(f"[labels] {category}: {len(wanted)} items in the frozen catalogue")
    print(f"[labels] scanning {src.name} ({src.stat().st_size / 1e6:.0f} MB)")

    found: dict[str, dict] = {}
    n_lines = n_dupe = 0
    with gzip.open(src, "rt") as fh:
        for line in fh:
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("asin")
            if asin not in wanted:
                continue
            if asin in found:
                # metaFiles2 is the de-duplicated release, but a repeat would
                # silently pick a winner, so it is counted and the first is kept.
                n_dupe += 1
                continue
            found[asin] = rec
    print(f"[labels] {n_lines} records scanned, {n_dupe} duplicate ASINs ignored")

    out: dict[str, dict] = {}
    for asin, rec in found.items():
        crumb = [c for c in (rec.get("category") or []) if isinstance(c, str) and c.strip()]
        crumb = crumb[BREADCRUMB_SKIP:]
        price = parse_price(rec.get("price"))
        rank, rank_store = parse_rank(rec.get("rank"))
        row = {
            "item_id": item2id[asin],
            "main_cat": (rec.get("main_cat") or "").strip(),
            "cat_depth": len(crumb),
            "cat_leaf": crumb[-1] if crumb else "",
            "brand_ext": (rec.get("brand") or "").strip(),
            "price_usd": price,
            "rank": rank,
            "rank_store": rank_store,
            "also_buy": list(rec.get("also_buy") or []),
            "also_view": list(rec.get("also_view") or []),
            "feature": list(rec.get("feature") or []),
            "details": dict(rec.get("details") or {}),
            "title_ext": (rec.get("title") or "").strip(),
        }
        for lvl in range(3):
            row[f"cat_l{lvl + 1}"] = crumb[lvl] if len(crumb) > lvl else ""
        out[asin] = row

    missing = sorted(wanted - set(out))
    joined = len(out)
    print(f"[labels] JOIN {joined}/{len(wanted)} = {joined / len(wanted):.1%}"
          + (f"  ({len(missing)} unmatched, e.g. {missing[:3]})" if missing else ""))

    # Coverage is computed over the WHOLE frozen catalogue, not over the joined
    # subset, because a field is only usable to the extent it covers the items
    # the SIDs actually partition.
    cov = {}
    for field in ("main_cat", "cat_l1", "cat_l2", "cat_l3", "cat_leaf",
                  "brand_ext", "price_usd", "rank", "also_buy", "feature", "details"):
        vals = [out[a].get(field) for a in out]
        nonempty = [v for v in vals if v not in (None, "", [], {})]
        distinct = (len({json.dumps(v, sort_keys=True) for v in nonempty})
                    if field in ("also_buy", "also_view", "feature", "details")
                    else len(set(nonempty)))
        cov[field] = {
            "n_items": len(wanted),
            "non_empty": len(nonempty),
            "coverage": round(len(nonempty) / len(wanted), 4),
            "distinct": distinct,
            "mean_items_per_value": round(len(nonempty) / distinct, 3) if distinct else 0.0,
        }

    print("[labels] coverage over the full catalogue")
    for f, s in cov.items():
        print(f"          {f:12s} {s['non_empty']:5d}/{s['n_items']} "
              f"({s['coverage']:6.1%})  distinct={s['distinct']:5d}  "
              f"items/value={s['mean_items_per_value']:8.2f}")

    top = {f: Counter(out[a][f] for a in out if out[a].get(f)).most_common(10)
           for f in ("main_cat", "cat_l1", "cat_l2", "cat_l3")}

    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "source": {
            "url": SOURCE_URL.format(category=category),
            "release": "Amazon Review Data (2018), Ni/Li/McAuley -- metaFiles2",
            "path": str(src),
            "sha256": hashing.sha256_file(src),
            "size": src.stat().st_size,
            "records_scanned": n_lines,
        },
        "join": {
            "catalogue_items": len(wanted),
            "joined": joined,
            "join_rate": round(joined / len(wanted), 4),
            "unmatched_asins": missing,
            "duplicate_asins_ignored": n_dupe,
            "key": "asin, via frozen/data/id_maps/<category>.item2id",
        },
        "coverage": cov,
        "top_values": top,
        "caveats": {
            "not_frozen": "External data, pulled after the freeze. The models "
                          "never saw it. It must never be joined into "
                          "frozen/ or treated as part of the training substrate.",
            "text": "title_ext/brand_ext are the 2018 dump's own strings and may "
                    "differ from the frozen item.json. The FROZEN text is what "
                    "the embeddings were built from and stays authoritative; "
                    "these are carried for comparison only.",
            "breadcrumb": f"Level 0 ('{category.replace('_and_', ' & ').replace('_', ' ')}') "
                          "is constant and dropped; cat_l1 is breadcrumb level 1.",
            "rank": "Primary store rank only. Subcategory ranks have per-item "
                    "denominators and are not comparable across the catalogue.",
            "price": "Only unambiguous $X.XX parsed. Ranges and CSS-blob values "
                     "are left None rather than guessed.",
        },
    }

    if not write:
        print("[labels] --check: nothing written")
        return manifest

    dest = paths.EXTERNAL / "labels"
    dest.mkdir(parents=True, exist_ok=True)
    lab_path = dest / f"{category}.labels.json"
    man_path = dest / f"{category}.labels.manifest.json"
    lab_path.write_text(json.dumps(out, sort_keys=True))
    manifest["labels_sha256"] = hashing.sha256_file(lab_path)
    manifest["labels_path"] = str(lab_path)
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"[labels] wrote {lab_path} ({lab_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[labels] wrote {man_path}")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="Industrial_and_Scientific")
    ap.add_argument("--check", action="store_true",
                    help="re-join and report, but write nothing")
    args = ap.parse_args(argv)
    man = build(args.category, write=not args.check)
    return 0 if man["join"]["join_rate"] >= 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
