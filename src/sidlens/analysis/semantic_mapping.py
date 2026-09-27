"""Lossless SID-prefix exports and descriptive category-coherence experiments.

Run ``python -m sidlens.analysis.semantic_mapping build --help`` or ``query --help``.
Uses frozen assignments, not lossy token-to-item caches or quantizer weights.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.data import labels, meta
from sidlens.data.sids import SidTable, available
from sidlens.provenance.hashing import sha256_file

CATEGORY = "Industrial_and_Scientific"
EXAMPLES = [
    {"variant": "rqkmeans_3codebook_128", "prefix": [14], "label": "3D-printer filaments"},
    {"variant": "rqvae_3codebook_128", "prefix": [10], "label": "Printer filaments and accessories"},
    {"variant": "MQ_3codebook_128", "prefix": [122], "label": "Hydrometers and test jars"},
]


def sid_text(codes):
    return "".join(f"({int(c)})" for c in codes)


def pair_agreement(groups: np.ndarray, y: np.ndarray) -> tuple[float | None, int]:
    """Fraction of unordered distinct-item pairs in a group sharing a label.

    The caller supplies only labelled items. Count combinations without
    enumerating O(n^2) pairs. No eligible pairs means undefined, not perfect.
    """
    groups, y = np.asarray(groups), np.asarray(y)
    if len(groups) != len(y):
        raise ValueError("groups and labels must have equal length")
    if not len(y):
        return None, 0
    _, g = np.unique(groups, return_inverse=True)
    _, label = np.unique(y, return_inverse=True)
    sizes = np.bincount(g).astype(np.int64)
    pairs = int(np.sum(sizes * (sizes - 1)) // 2)
    if not pairs:
        return None, 0
    _, counts = np.unique(g * (int(label.max()) + 1) + label, return_counts=True)
    same = int(np.sum(counts * (counts - 1)) // 2)
    return same / pairs, pairs


def coherence(codes, raw_labels, n_perm=200, seed=0):
    """Prefix coherence vs size-preserving label permutations, at every depth.

    Missing-label positions remain fixed. Null intervals are permutation-null
    ranges, not confidence intervals. These are descriptive catalogue results.
    """
    codes = np.asarray(codes)
    keep = np.array([isinstance(y, str) and bool(y.strip()) for y in raw_labels])
    _, y = np.unique(np.array(raw_labels, dtype=object)[keep].astype(str), return_inverse=True)
    baseline, _ = pair_agreement(np.zeros(len(y), dtype=int), y)
    rows = []
    for depth in range(1, codes.shape[1] + 1):
        _, group = np.unique(codes[:, :depth], axis=0, return_inverse=True)
        sizes = np.bincount(group)
        observed, pairs = pair_agreement(group[keep], y)
        # Reset to the same seed at each depth: paired null assignments without
        # storing an n_permutations x n_items matrix for larger catalogues.
        rng = np.random.default_rng(seed)
        null = np.array([pair_agreement(group[keep], rng.permutation(y))[0]
                         for _ in range(n_perm)], dtype=float) if pairs else np.array([])
        null_mean = float(null.mean()) if len(null) else None
        rows.append({
            "depth": depth, "n_items": len(codes), "n_labelled": int(keep.sum()),
            "n_groups": len(sizes), "n_multi_groups": int((sizes > 1).sum()),
            "singleton_item_share": float((sizes == 1).sum() / len(codes)),
            "n_labelled_pairs": pairs, "pair_category_agreement": observed,
            "catalogue_pair_agreement": baseline, "null_pair_mean": null_mean,
            "null_pair_low": float(np.quantile(null, .025)) if len(null) else None,
            "null_pair_high": float(np.quantile(null, .975)) if len(null) else None,
            "pair_excess": observed - null_mean if observed is not None and null_mean is not None else None,
            "n_permutations": len(null),
        })
    return rows


def write_csv(path, rows):
    with Path(path).open("w", newline="") as fh:
        iterator = iter(rows)
        first = next(iterator, None)
        if first is None:
            return 0
        writer = csv.DictWriter(fh, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        count = 1
        for row in iterator:
            writer.writerow(row)
            count += 1
    return count


def assignment_rows(table, items):
    for item in items:
        codes = table.asin2codes[item["asin"]]
        row = {"variant": table.variant.name, **item, "sid": sid_text(codes),
               "sid_shared_by": len(table.cb2items[codes])}
        for d in range(5):
            row[f"code{d + 1}"] = codes[d] if d < len(codes) else ""
            row[f"prefix{d + 1}"] = sid_text(codes[:d + 1]) if d < len(codes) else ""
        yield row


def group_rows(table, by_asin):
    for depth in range(1, table.variant.n_codebook + 1):
        children = Counter(p[:-1] for p in table.clusters(depth + 1)) if depth < table.variant.n_codebook else {}
        for prefix, asins in sorted(table.clusters(depth).items()):
            row = {"variant": table.variant.name, "depth": depth, "prefix": sid_text(prefix),
                   "parent": sid_text(prefix[:-1]), "n_items": len(asins),
                   "n_children": children.get(prefix, 0)}
            for field in ("cat_l1", "cat_l2"):
                counts = Counter(by_asin[a][field] for a in asins if by_asin[a][field])
                n = sum(counts.values())
                top, count = counts.most_common(1)[0] if counts else ("", 0)
                row.update({f"{field}_labelled": n, f"{field}_top": top,
                            f"{field}_top_count": count,
                            f"{field}_purity": count / n if n else None,
                            f"{field}_entropy_bits": -sum((v / n) * math.log2(v / n) for v in counts.values()) if n else None})
            yield row


def make_database(path, items, tables):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE items (asin TEXT PRIMARY KEY, item_id INTEGER, title TEXT,
            brand TEXT, cat_l1 TEXT, cat_l2 TEXT, cat_l3 TEXT);
        CREATE TABLE assignments (variant TEXT, asin TEXT, sid TEXT,
            code1 INTEGER, code2 INTEGER, code3 INTEGER, code4 INTEGER, code5 INTEGER,
            sid_shared_by INTEGER, PRIMARY KEY (variant, asin),
            FOREIGN KEY (asin) REFERENCES items(asin));
        CREATE INDEX sid_prefix ON assignments(variant,code1,code2,code3,code4,code5);
        CREATE INDEX complete_sid ON assignments(variant,sid);
    """)
    conn.executemany("INSERT INTO items VALUES (?,?,?,?,?,?,?)", [tuple(i[k] for k in ("asin", "item_id", "title", "brand", "cat_l1", "cat_l2", "cat_l3")) for i in items])
    for table in tables:
        conn.executemany("INSERT INTO assignments VALUES (?,?,?,?,?,?,?,?,?)", (
            (table.variant.name, asin, sid_text(c), *c, *([None] * (5 - len(c))), len(table.cb2items[c]))
            for asin, c in table.asin2codes.items()))
    conn.commit()
    check = conn.execute("PRAGMA integrity_check").fetchone()[0]
    count = conn.execute("SELECT count(*) FROM assignments").fetchone()[0]
    conn.close()
    return {"sqlite_integrity": check, "assignment_rows": count}


def query_database(database, variant, prefix=(), contains=""):
    if len(prefix) > 5 or any(v < 0 for v in prefix):
        raise ValueError("prefix must have zero to five nonnegative codes")
    conn = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    if not conn.execute("SELECT 1 FROM assignments WHERE variant=? LIMIT 1", (variant,)).fetchone():
        conn.close()
        raise ValueError(f"unknown variant: {variant}")
    conditions, args = ["a.variant=?"], [variant]
    for d, code in enumerate(prefix, 1):
        conditions.append(f"a.code{d}=?")
        args.append(code)
    if contains:
        conditions.append("(instr(lower(i.title),lower(?))>0 OR instr(lower(i.asin),lower(?))>0)")
        args.extend([contains, contains])
    rows = conn.execute("SELECT a.*,i.item_id,i.title,i.brand,i.cat_l1,i.cat_l2,i.cat_l3 FROM assignments a JOIN items i USING(asin) WHERE " + " AND ".join(conditions) + " ORDER BY a.code1,a.code2,a.code3,a.code4,a.code5,a.asin", args).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def render_examples(tables, by_asin):
    out = ["# Concrete SID prefix mappings", "", "These examples were selected after inspecting the catalogue to illustrate useful and imperfect groupings. They are exploratory, not a random sample. Every item in each selected first-digit group is listed below; the full exports cover every group.", "", "Code numbers have meaning only within a specified variant and position. A full SID can map to several items. MQ prefixes intersect parallel partitions; adding positions does not establish an intrinsic coarse-to-fine hierarchy.", ""]
    lookup = {t.variant.name: t for t in tables}
    for example in EXAMPLES:
        if example["variant"] not in lookup:
            continue
        table = lookup[example["variant"]]
        prefix = tuple(example["prefix"])
        members = table.clusters(len(prefix))[prefix]
        out += [f"## {table.variant.name}: {sid_text(prefix)} — {example['label']}", "", f"**All {len(members)} matching items:**", "", "| Full SID | ASIN | Item | Category |", "|---|---|---|---|"]
        for asin in sorted(members, key=lambda a: (table.asin2codes[a], a)):
            m = by_asin[asin]
            title = m["title"].replace("|", "\\|").replace("\n", " ")
            cat = (m["cat_l2"] or m["cat_l1"]).replace("|", "\\|")
            out.append(f"| {sid_text(table.asin2codes[asin])} | {asin} | {title} | {cat} |")
        out += ["", "### Every two-digit branch", "", "| Prefix | Items | Full-SID buckets |", "|---|---:|---:|"]
        for p, aa in sorted(table.clusters(2).items()):
            if p[:len(prefix)] == prefix:
                out.append(f"| {sid_text(p)} | {len(aa)} | {len({table.asin2codes[a] for a in aa})} |")
        out.append("")
    return "\n".join(out)


def build(args):
    if args.n_perm < 0:
        raise ValueError("n-perm must be nonnegative")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "status.txt").write_text("running\n")
    names = [v.name for v in available()] if args.variants == ["all"] else args.variants
    tables = [SidTable.load(v) for v in names]
    if not tables or len(names) != len(set(names)):
        raise ValueError("provide one or more distinct available variants")
    metadata, lab = meta.load(CATEGORY), labels.load(CATEGORY)
    keys = sorted(tables[0].keys)
    if set(keys) != set(metadata) or any(set(t.keys) != set(keys) for t in tables):
        raise ValueError("SID and metadata catalogue identities differ")
    if not set(keys).issubset(lab):
        raise ValueError("external label rows missing for catalogue items")
    items = [{"asin": a, "item_id": metadata[a].item_id, "title": metadata[a].title,
              "brand": metadata[a].brand,
              **{f: (lab[a].get(f) or "").strip() for f in ("cat_l1", "cat_l2", "cat_l3")}} for a in keys]
    by_asin = {i["asin"]: i for i in items}
    now = datetime.now(timezone.utc).isoformat()
    data = {"category": CATEGORY, "generated_at": now, "items": items, "variants": {},
            "examples": [e for e in EXAMPLES if e["variant"] in names]}
    summaries, counts = [], {}
    for table in tables:
        name = table.variant.name
        codes = np.array([table.asin2codes[a] for a in keys], dtype=np.int64)
        print(f"Mapping and permutation check: {name}", flush=True)
        local = []
        for field in ("cat_l1", "cat_l2"):
            seed = (args.seed + int(hashlib.sha256((name + field).encode()).hexdigest()[:8], 16)) % (2**32)
            rows = [{"variant": name, "field": field, "seed": seed, **r}
                    for r in coherence(codes, [by_asin[a][field] for a in keys], args.n_perm, seed)]
            summaries.extend(rows)
            if field == "cat_l1":
                local = rows
        data["variants"][name] = {"codes": codes.tolist(), "summary": local}
        sizes = Counter(table.asin2codes.values())
        counts[name] = {"items": len(keys), "unique_sids": len(sizes),
                        "ambiguous_items": sum(n for n in sizes.values() if n > 1),
                        "excess_items_over_unique_sids": len(keys) - len(sizes)}
    n_assignments = write_csv(out / "mappings.csv", itertools.chain.from_iterable(assignment_rows(t, items) for t in tables))
    n_groups = write_csv(out / "prefix_groups.csv", itertools.chain.from_iterable(group_rows(t, by_asin) for t in tables))
    write_csv(out / "coherence.csv", summaries)
    validation = make_database(out / "mappings.sqlite", items, tables)
    validation.update({"catalogue_identity": "passed", "metadata_rows": len(items), "label_rows_joined": len(keys),
                       "csv_assignment_rows": n_assignments, "prefix_group_rows": n_groups,
                       "all_items_preserved": n_assignments == len(keys) * len(tables), "counts": counts})
    # Independently compare SQL prefix results with the authoritative loader.
    checks = 0
    for table in tables:
        for depth in range(1, table.variant.n_codebook + 1):
            clusters = table.clusters(depth)
            for p in [min(clusters), max(clusters), max(clusters, key=lambda p: len(clusters[p]))]:
                found = {r["asin"] for r in query_database(out / "mappings.sqlite", table.variant.name, p)}
                if found != set(clusters[p]):
                    raise AssertionError(f"SQL prefix mismatch: {table.variant.name} {p}")
                checks += 1
    validation["sql_prefix_checks"] = checks
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    (out / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    from sidlens.viz.semantic_explorer import render_explorer
    (out / "explorer.html").write_text(render_explorer(data))
    (out / "examples.md").write_text(render_examples(tables, by_asin))
    source_files = [Path(__file__), paths.REPO / "src/sidlens/viz/semantic_explorer.py",
                    paths.REPO / "src/sidlens/data/sids.py", paths.REPO / "src/sidlens/data/meta.py",
                    paths.REPO / "src/sidlens/data/labels.py"]
    inputs = [t.source for t in tables] + [paths.FROZEN_DATA / "item_meta" / f"{CATEGORY}.item.json",
              paths.FROZEN_DATA / "id_maps" / f"{CATEGORY}.item2id",
              paths.EXTERNAL / "labels" / f"{CATEGORY}.labels.json",
              paths.EXTERNAL / "labels" / f"{CATEGORY}.labels.manifest.json"]
    (out / "source").mkdir()
    for source in source_files:
        destination = out / "source" / source.relative_to(paths.REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    manifest = {"schema_version": 1, "created_at": now, "arguments": vars(args), "command": sys.argv,
                "git_head": subprocess.check_output(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"], text=True).strip(),
                "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
                "python": sys.version, "numpy": np.__version__,
                "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
                "source_hashes": {str(p): sha256_file(p) for p in source_files},
                "input_hashes": {str(p): sha256_file(p) for p in inputs}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "README.md").write_text(f"""# SID semantic mapping experiment

Completed {now}. {len(items):,} items × {len(tables)} variants = {n_assignments:,} lossless assignments; {n_groups:,} observed prefix groups.

- Open `explorer.html` directly in a browser; it embeds every item and assignment, works offline, and supports prefix/position filtering and CSV export.
- `examples.md` lists every item in three illustrative first-digit groups.
- `mappings.csv` contains every assignment, item title, category, all prefixes, and full-SID collision multiplicity.
- `prefix_groups.csv` contains every observed prefix and descriptive category summaries.
- `mappings.sqlite` stores items once and indexes variant plus digit values. Use the query CLI below for large-scale exact lookups.
- `coherence.csv` compares category agreement among distinct item pairs sharing a prefix to {args.n_perm} label permutations. The permutations preserve prefix sizes and the labelled-item positions. Pair agreement weights large groups more strongly; singleton share and eligible pair counts are explicit. Null ranges are not confidence intervals. This is an exploratory finite-catalogue analysis, not a causal or held-out claim.
- `data.json` is the portable explorer payload; `manifest.json`, `source/`, and `validation.json` record inputs, code, and checks.

```bash
python -m sidlens.analysis.semantic_mapping query --db {out / 'mappings.sqlite'} --variant rqkmeans_3codebook_128 --prefix 14,16
```

The assignments come from the frozen repaired tables. Numeric codes do not transfer across independently fitted variants. MQ positions are parallel partitions. Full SIDs may identify multiple items; no member is discarded. Titles are training-lineage text; categories are the separately acquired Amazon-2018 labels. Text/category agreement alone does not establish model reasoning. Missing category values are excluded from pair metrics, with coverage reported. Per-prefix purity is descriptive and can rise simply because groups become smaller. Do not count singleton purity as semantic evidence.

Generation scales with observed assignments and prefixes, rather than enumerating the empty K^D code space. For a much larger catalogue, query SQLite or partition exports per variant; the embedded HTML is intended for this catalogue size.
""")
    (out / "status.txt").write_text("complete\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "output.sha256")
    (out / "output.sha256").write_text("".join(f"{sha256_file(p)}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"out": str(out), "items": len(items), "variants": len(tables),
                      "assignments": n_assignments, "prefix_groups": n_groups, "validation": "passed"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)
    b = sub.add_parser("build")
    b.add_argument("--variants", nargs="+", default=["all"])
    b.add_argument("--out", required=True)
    b.add_argument("--n-perm", type=int, default=200)
    b.add_argument("--seed", type=int, default=20260915)
    q = sub.add_parser("query")
    q.add_argument("--db", required=True)
    q.add_argument("--variant", required=True)
    q.add_argument("--prefix", default="", help="Comma-separated codes, e.g. 14,16")
    q.add_argument("--contains", default="")
    q.add_argument("--format", choices=["json", "csv"], default="json")
    args = parser.parse_args()
    if args.command_name == "build":
        build(args)
    else:
        prefix = tuple(int(v.strip()) for v in args.prefix.split(",") if v.strip())
        rows = query_database(args.db, args.variant, prefix, args.contains)
        if args.format == "json":
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        elif rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
