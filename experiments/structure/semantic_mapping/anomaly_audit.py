#!/usr/bin/env python
"""Reproduce the RQ-VAE collision audit using only frozen inputs.

Example::

    python experiments/structure/semantic_mapping/anomaly_audit.py --out /fresh/output/path

The output directory must not exist. This audits shared pre-quantization input
embeddings, not the unavailable RQ-VAE encoder latents or codebook vectors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.data import meta
from sidlens.data.sids import SidTable, available, load_item2id
from sidlens.provenance.hashing import sha256_file


CATEGORY = "Industrial_and_Scientific"


def sid_text(codes):
    return "".join(f"({int(c)})" for c in codes)


def geometry(vectors):
    """Compute bounded exact pair distances without a pairs-by-dimension array."""
    z = np.asarray(vectors, dtype=np.float64)
    if not len(z) or not np.isfinite(z).all():
        raise ValueError("geometry requires at least one finite vector")
    norms = np.linalg.norm(z, axis=1)
    distances = np.concatenate([
        np.linalg.norm(z[i + 1:] - z[i], axis=1)
        for i in range(len(z) - 1)
    ]) if len(z) > 1 else np.array([], dtype=np.float64)
    return {
        "n_items": len(z),
        "n_unique_input_vectors": int(len(np.unique(z, axis=0))),
        "n_zero_input_vectors": int((norms == 0).sum()),
        "all_finite": True,
        "input_norm_min": float(norms.min()),
        "input_norm_max": float(norms.max()),
        "input_rms_radius": float(np.sqrt(((z - z.mean(0)) ** 2).sum(1).mean())),
        "n_distinct_item_pairs": len(distances),
        "n_pairs_with_identical_input_vectors": int((distances == 0).sum()),
        "pair_euclidean_min": float(distances.min()) if len(distances) else None,
        "pair_euclidean_median": float(np.median(distances)) if len(distances) else None,
        "pair_euclidean_max": float(distances.max()) if len(distances) else None,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"output must be a fresh directory: {out}")
    target = SidTable.load(args.variant)
    sid = tuple(int(c.strip()) for c in args.sid.split(","))
    if len(sid) != target.variant.n_codebook or sid not in target.cb2items:
        raise ValueError(f"no complete SID {sid_text(sid)} in {args.variant}")
    members = sorted(target.cb2items[sid])
    if len(members) < 2:
        raise ValueError("the selected full SID must contain at least two items")
    tables = [SidTable.load(v) for v in available()]
    metadata = meta.load(CATEGORY)
    item2id = load_item2id(CATEGORY)
    keys = set(target.keys)
    if keys != set(metadata) or keys != set(item2id) or any(set(t.keys) != keys for t in tables):
        raise ValueError("the SID, metadata and item-ID catalogues differ")
    if len(set(item2id.values())) != len(item2id):
        raise ValueError("integer item IDs must map to distinct embedding rows")
    embedding_path = paths.FROZEN_DATA / "embeddings" / f"{CATEGORY}.emb-qwen-td.npy"
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[1] != 2560:
        raise ValueError(f"unexpected input embedding shape: {embeddings.shape}")
    if min(item2id.values()) < 0 or max(item2id.values()) >= len(embeddings):
        raise ValueError("integer item IDs are outside the embedding matrix")

    def vectors(asins):
        return embeddings[[item2id[a] for a in asins]]

    buckets = []
    audited_keys = set()
    for code, asins in sorted(target.cb2items.items()):
        if len(asins) < 2:
            continue
        audited_keys.update(asins)
        buckets.append({"sid": sid_text(code), "asins": sorted(asins), **geometry(vectors(asins))})
    selected = next(b for b in buckets if b["sid"] == sid_text(sid))
    others = [b for b in buckets if b["sid"] != sid_text(sid)]
    all_pairs = sum(b["n_distinct_item_pairs"] for b in buckets)
    parent_asins = target.clusters(len(sid) - 1)[sid[:-1]]
    audited_keys.update(parent_asins)
    parent = {"prefix": sid_text(sid[:-1]), **geometry(vectors(parent_asins))}
    fit_results = []
    for table in tables:
        counts = Counter(table.asin2codes[a] for a in members)
        fit_results.append({
            "variant": table.variant.name,
            "n_selected_items": len(members),
            "n_unique_sids_among_selected_items": len(counts),
            "max_selected_items_sharing_sid": max(counts.values()),
            "n_selected_pairs_sharing_sid": sum(n * (n - 1) // 2 for n in counts.values()),
        })
    rows = []
    for asin in members:
        row = {"asin": asin, "item_id": item2id[asin], "title": metadata[asin].title,
               "brand": metadata[asin].brand, "audited_sid": sid_text(sid)}
        row.update({t.variant.name: sid_text(t.asin2codes[asin]) for t in tables})
        rows.append(row)
    row_hashes = [{
        "asin": a,
        "embedding_row_index": item2id[a],
        "row_sha256": hashlib.sha256(np.asarray(embeddings[item2id[a]]).tobytes(order="C")).hexdigest(),
        "selected_anomaly_member": a in members,
    } for a in sorted(audited_keys)]

    result = {
        "category": CATEGORY,
        "target_variant": target.variant.name,
        "target_full_sid": sid_text(sid),
        "embedding_space": "Shared pre-quantization Qwen item input embeddings; not RQ-VAE encoder latent space",
        "embedding_shape": list(embeddings.shape),
        "stored_embedding_dtype": embeddings.dtype.str,
        "geometry_accumulation_dtype": "float64",
        "catalogue_items": len(keys),
        "selected_bucket": selected,
        "parent_prefix": parent,
        "other_multi_item_buckets": others,
        "other_multi_item_buckets_all_identical_inputs": all(b["n_unique_input_vectors"] == 1 for b in others),
        "all_multi_item_bucket_count": len(buckets),
        "all_distinct_item_pairs_sharing_full_sid": all_pairs,
        "selected_bucket_share_of_all_full_sid_pairs": selected["n_distinct_item_pairs"] / all_pairs,
        "pair_denominator_note": "All catalogue items, including those without category labels; these are not category-labelled pair counts",
        "same_items_across_fits": fit_results,
        "n_embedding_rows_audited": len(row_hashes),
    }
    inputs = [t.source for t in tables] + [embedding_path,
        paths.FROZEN_DATA / "id_maps" / f"{CATEGORY}.item2id",
        paths.FROZEN_DATA / "item_meta" / f"{CATEGORY}.item.json"]
    sources = [Path(__file__).resolve(), Path(paths.__file__).resolve(),
        Path(meta.__file__).resolve(), paths.REPO / "src/sidlens/data/sids.py",
        paths.REPO / "src/sidlens/provenance/hashing.py"]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arguments": vars(args), "command": sys.argv,
        "python": platform.python_version(), "numpy": np.__version__,
        "frozen_snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "input_sha256": {str(p): sha256_file(p) for p in inputs},
        "source_sha256": {str(p): sha256_file(p) for p in sources},
        "embedding_row_identity": {
            "mapping": "ASIN -> frozen OneDiffRec integer item ID -> zero-based embedding row",
            "hash_encoding": "C-order raw stored row bytes; dtype and matrix shape recorded in result.json",
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "status.txt").write_text("running\n")
    write_csv(out / "members.csv", rows)
    write_csv(out / "same_items_across_fits.csv", fit_results)
    write_csv(out / "embedding_rows.csv", row_hashes)
    write_csv(out / "multi_item_buckets.csv", [{k: v for k, v in b.items() if k != "asins"} for b in buckets])
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        dest = out / "source" / source.relative_to(paths.REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    report = [
        "# Full-SID collision audit", "",
        f"**{args.variant} {sid_text(sid)} maps {len(members)} items to one full SID, despite their distinct input representations.**" if selected["n_unique_input_vectors"] == len(members) else f"**{args.variant} {sid_text(sid)} contains {len(members)} items and {selected['n_unique_input_vectors']} distinct input vectors.**", "",
        f"This audit reads only frozen assignments, item titles, the item-ID crosswalk, and shared input embeddings. It recomputes every multi-item full-SID bucket in the selected fit. No existing atlas geometry is used.", "",
        "## Input-space geometry", "",
        f"- Selected bucket: {selected['n_unique_input_vectors']} distinct vectors, {selected['n_zero_input_vectors']} zero vectors; all coordinates finite.",
        f"- Vector norms: {selected['input_norm_min']:.4f}–{selected['input_norm_max']:.4f}.",
        f"- Euclidean distances across {selected['n_distinct_item_pairs']} distinct-item pairs: minimum {selected['pair_euclidean_min']:.4f}, median {selected['pair_euclidean_median']:.4f}, maximum {selected['pair_euclidean_max']:.4f}.",
        f"- RMS radius around the group centroid: {selected['input_rms_radius']:.4f}; parent {parent['prefix']} has {parent['n_items']} items and radius {parent['input_rms_radius']:.4f}.",
        f"- Other colliding buckets: {len(others)}; {sum(b['n_unique_input_vectors'] == 1 for b in others)} contain identical input vectors within each bucket. All bucket statistics are in `multi_item_buckets.csv` and `result.json`.",
        f"- This bucket contributes {selected['n_distinct_item_pairs']}/{all_pairs} = {selected['n_distinct_item_pairs']/all_pairs:.2%} of all distinct-item pairs sharing a full SID. These counts include every item, not only items with category labels.", "",
        "## The same items under other fits", "",
        "These counts concern the selected items only. A selected item can still share a SID with an item outside this set.", "",
        "| Variant | Distinct SIDs among selected items | Largest selected subset sharing one SID |", "|---|---:|---:|",
    ]
    report.extend(f"| {r['variant']} | {r['n_unique_sids_among_selected_items']} | {r['max_selected_items_sharing_sid']} |" for r in fit_results)
    report += ["", "## Interpretation and reproduction", "",
        "Distinct input vectors rule out identical or zero shared-input vectors as the explanation for this collision. This does not locate its cause: the learned RQ-VAE encoder latents and quantizer weights are unavailable. Shared-input distances are not distances in that latent space.", "",
        "Other widths and RQ-VAE depths are separate fits. Their assignments cannot establish an isolated effect of increasing width or adding a digit. This is a post hoc audit of a specific collision, not a random sample of code groups.", "",
        "`members.csv` lists every selected ASIN, full frozen title, integer embedding-row ID, and complete SID across every available fit. `embedding_rows.csv` identifies and hashes each row used by this audit. `manifest.json` hashes the full embedding file, ID crosswalk, title file, every SID file, and executable sources.", "",
        "Rerun into a fresh directory:", "", "```bash",
        f"/l/users/leo.rodrigues/sidlens/venv/bin/python experiments/structure/semantic_mapping/anomaly_audit.py --variant {args.variant} --sid {','.join(map(str, sid))} --out /path/to/fresh/anomaly-audit",
        "```", "",
    ]
    (out / "report.md").write_text("\n".join(report))
    (out / "status.txt").write_text("complete\n")
    (out / "output.sha256").write_text("".join(
        f"{sha256_file(p)}  {p.relative_to(out)}\n"
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "output.sha256"
    ))
    print(json.dumps({"out": str(out), "members": len(members), "fits": len(tables),
                      "audited_embedding_rows": len(row_hashes), "status": "complete"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="rqvae_3codebook_128")
    parser.add_argument("--sid", default="27,89,7", help="Complete SID as comma-separated code values")
    run(parser.parse_args())
