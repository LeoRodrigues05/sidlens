#!/usr/bin/env python
"""Experiment 1a -- the item-side semantic atlas of every SID digit.

CPU only, no model. Reads frozen bytes for structure and the external
Amazon-2018 label join for meaning. This is the half of Experiment 1 that can
run today; the probe/patching half needs the hook layer.

Outputs (to $SIDLENS_WORK/derived/exp1_atlas/<variant>/):
    cells.csv        one row per (prefix): size, radius, shrink vs parent,
                     branching, top brand, exemplar ASINs
    digits.csv       per-digit rollup -- how much each position tightens
    members.csv      the flat item table: item -> full SID -> every prefix
    attribute.csv    marginal + conditional AMI per digit per label field,
                     each conditional against a within-parent permutation null
    copurchase.csv   same-prefix co-purchase rate vs chance, per depth
    naming.csv       per code cell: the terms that distinguish it from its
                     siblings, and the held-out AUC of those terms
    naming_digits.csv  per-digit rollup of that AUC -- how readable each
                     position's codes are at all
    browse.txt       the nesting, rendered for reading by eye
    result.json      the above plus the provenance stamp

Default variants are the three that matter first: rqkmeans 3cb x 128 is the AR
next-item best AND has a matched diffusion checkpoint, so it is the cell every
paradigm comparison will run on; rqvae 4cb x 128 is the second matched cell; MQ
3cb x 128 is the non-residual contrast.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.analysis import atlas as A
from sidlens.analysis import attribute as AT
from sidlens.analysis import naming as NM
from sidlens.data import embeddings as E
from sidlens.data import labels as L
from sidlens.data import meta as M
from sidlens.data.sids import SidTable, available

CATEGORY = "Industrial_and_Scientific"
DEFAULT_VARIANTS = ["rqkmeans_3codebook_128", "rqvae_4codebook_128", "MQ_3codebook_128"]

# Fields the alignment table is computed over. `cat_l3` is included with an
# explicit force because its 78% coverage is worth having at depth 1-2 even
# though labels.py gates it off by default; every row carries n_labelled so the
# thinner support is visible rather than assumed away.
ALIGN_FIELDS = ["cat_l1", "cat_l2", "main_cat", "price_decile", "rank_decile"]
ALIGN_FIELDS_FORCED = ["cat_l3"]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cell_rows(cells: list[A.CodeCell], meta: dict) -> list[dict]:
    rows = []
    for c in cells:
        rows.append({
            "variant": c.variant, "quantizer": c.quantizer, "residual": c.residual,
            "digit": c.digit, "depth": c.depth,
            "prefix": "-".join(map(str, c.prefix)),
            "parent": "-".join(map(str, c.parent)),
            "code": c.code,
            "n_items": c.n_items, "n_children": c.n_children,
            "radius": round(c.radius, 4),
            "parent_radius": round(c.parent_radius, 4),
            "shrink": round(c.shrink, 4) if np.isfinite(c.shrink) else "",
            "top_brand": c.top_brand, "top_brand_count": c.top_brand_count,
            "brand_purity": round(c.brand_purity, 4),
            "exemplar_asins": " ".join(c.exemplars),
            "exemplar_titles": " | ".join(
                meta[a].short_title for a in c.exemplars if a in meta),
        })
    return rows


def member_rows(table: SidTable, meta: dict) -> list[dict]:
    """The flat item table: one row per item, its SID, and every prefix of it.

    This is the table to sort and eyeball. Sorting by d1 groups the catalogue by
    the first digit; sorting by (d1,d2) shows what the second digit split.
    """
    rows = []
    n = table.variant.n_codebook
    for asin, code in table.asin2codes.items():
        m = meta.get(asin)
        row = {
            "asin": asin,
            "item_id": m.item_id if m else "",
            "sid": "-".join(map(str, code)),
            "title": m.title if m else "",
            "brand": m.brand if m else "",
        }
        for d in range(n):
            row[f"d{d + 1}"] = code[d]
        for d in range(1, n + 1):
            row[f"prefix{d}"] = "-".join(map(str, code[:d]))
        row["sid_shared_by"] = len(table.cb2items[code])
        rows.append(row)
    rows.sort(key=lambda r: tuple(r[f"d{d + 1}"] for d in range(n)))
    return rows


def render_browse(cells: list[A.CodeCell], table: SidTable, meta: dict,
                  n_codes: int, n_items: int) -> str:
    """The nesting rendered for a human: digit 1 codes, then their splits.

    Codes are taken largest-first, because a code holding two items tells you
    almost nothing about what the digit means and a code holding seventy does.
    """
    v = table.variant
    out: list[str] = []
    out.append(f"{'=' * 100}")
    out.append(f"{v.name}   ({v.quantizer}, {v.n_codebook} digits x {v.codebook_size} codes, "
               f"{'residual' if A.RESIDUAL[v.quantizer] else 'PARALLEL -- digits are not a refinement'})")
    out.append(f"{'=' * 100}")

    by_prefix = {c.prefix: c for c in cells}
    d1 = sorted((c for c in cells if c.depth == 1),
                key=lambda c: -c.n_items)[:n_codes]

    for c1 in d1:
        out.append("")
        out.append(f"DIGIT 1 = {c1.code:<4d}  {c1.n_items} items, {c1.n_children} sub-codes, "
                   f"radius {c1.radius:.1f} (catalogue {c1.parent_radius:.1f}, "
                   f"shrink {c1.shrink:.2f}), top brand {c1.top_brand or '-'} "
                   f"({c1.top_brand_count}/{c1.n_items})")
        out.append("  " + "-" * 96)
        for a in c1.exemplars[:n_items]:
            out.append(f"    {'-'.join(map(str, table.asin2codes[a])):<14s} "
                       f"{meta[a].short_title if a in meta else a}")

        kids = sorted((c for c in cells if c.depth == 2 and c.parent == c1.prefix),
                      key=lambda c: -c.n_items)[:4]
        for c2 in kids:
            out.append("")
            out.append(f"    +-- DIGIT 2 = {c2.code:<4d}  {c2.n_items} items, "
                       f"radius {c2.radius:.1f}, shrink vs parent {c2.shrink:.2f}, "
                       f"top brand {c2.top_brand or '-'} ({c2.top_brand_count}/{c2.n_items})")
            for a in c2.exemplars[:n_items]:
                out.append(f"          {'-'.join(map(str, table.asin2codes[a])):<14s} "
                           f"{meta[a].short_title if a in meta else a}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS,
                    help="variant names, or 'all' for the full 27-variant grid")
    ap.add_argument("--category", default=CATEGORY)
    ap.add_argument("--out", default=None)
    ap.add_argument("--browse-codes", type=int, default=8,
                    help="digit-1 codes to render in browse.txt")
    ap.add_argument("--browse-items", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=AT.N_PERM,
                    help="permutations for the conditional-AMI null; 0 disables")
    ap.add_argument("--no-attributes", action="store_true",
                    help="skip the label-dependent half (atlas structure only)")
    ap.add_argument("--no-naming", action="store_true",
                    help="skip discriminative-term signatures and their held-out AUC")
    ap.add_argument("--naming-top", type=int, default=12,
                    help="terms per code-cell signature")
    args = ap.parse_args(argv)

    if len(args.variants) == 1 and args.variants[0] == "all":
        args.variants = [v.name for v in available()]

    out_root = Path(args.out) if args.out else paths.DERIVED / "exp1_atlas"
    meta = M.load(args.category)
    cov = M.coverage(args.category)
    print("[exp1] metadata coverage")
    for f, s in cov.items():
        print(f"        {f:12s} non_empty={s['non_empty']:5d}/{s['n_items']} "
              f"distinct={s['distinct']:5d} mean_items_per_value={s['mean_items_per_value']:.2f}")

    label_cov = label_gate = None
    if not args.no_attributes:
        label_cov = L.coverage(args.category)
        label_gate = L.usable(args.category)
        print("[exp1] recovered label coverage (external, Amazon-2018 join)")
        for f, s in sorted(label_cov.items()):
            ok, why = label_gate[f]
            print(f"        {f:14s} {s['non_empty']:5d}/{s['n_items']} "
                  f"({s['coverage']:6.1%})  distinct={s['distinct']:5d}  "
                  f"{'USABLE' if ok else 'gated: ' + why}")

    all_digits: list[dict] = []
    all_align: list[dict] = []
    all_copurchase: list[dict] = []
    all_naming: list[dict] = []
    all_naming_digits: list[dict] = []
    for name in args.variants:
        print(f"[exp1] {name}")
        table = SidTable.load(name)
        keys = table.keys
        z = E.load_aligned(args.category, keys)
        cells = A.build(table, z, keys, meta)
        digits = A.digit_summary(cells)
        all_digits.extend(digits)

        d = out_root / name
        write_csv(d / "cells.csv", cell_rows(cells, meta))
        write_csv(d / "digits.csv", digits)
        write_csv(d / "members.csv", member_rows(table, meta))
        (d / "browse.txt").write_text(
            render_browse(cells, table, meta, args.browse_codes, args.browse_items))

        if not args.no_attributes:
            align: list[dict] = []
            for field in ALIGN_FIELDS + ALIGN_FIELDS_FORCED:
                forced = field in ALIGN_FIELDS_FORCED
                y, vocab = L.codes(args.category, keys, field, force=forced)
                rows = AT.alignment(table, y, keys, field, n_perm=args.n_perm)
                for r in rows:
                    r["gated_field"] = forced
                align.extend(rows)
            titles = {k: meta[k].title for k in keys if k in meta}
            align.extend(AT.lexicon_crosscheck(table, keys, titles, args.category))
            cop = AT.copurchase_preservation(table, keys, args.category)
            write_csv(d / "attribute.csv", align)
            write_csv(d / "copurchase.csv", cop)
            all_align.extend(align)
            all_copurchase.extend(cop)

            print("        attribute alignment (AMI; cond = given digits 1..d-1)")
            for field in ALIGN_FIELDS + ALIGN_FIELDS_FORCED:
                sub = [r for r in align if r["field"] == field]
                cells_s = "  ".join(
                    f"d{r['digit']}:{r['marginal_ami']:.3f}/"
                    f"{r['conditional_ami'] if r['conditional_ami'] is not None else float('nan'):.3f}"
                    f"{'' if r['conditional_supported'] or r['digit'] == 0 else '~'}"
                    f"{'*' if (r['cond_p'] is not None and r['cond_p'] <= 0.05) else ' '}"
                    for r in sub)
                print(f"          {field:13s} {cells_s}")
            print("        co-purchase preservation (same-prefix rate / chance)")
            print("          " + "  ".join(
                f"d{r['digit']}:{r['lift'] if r['lift'] is not None else float('nan'):.1f}x"
                for r in cop))

        if not args.no_naming:
            texts = {k: meta[k].title for k in keys if k in meta}
            nm = NM.describe_cells(table, texts, top=args.naming_top)
            nmd = NM.depth_summary(nm)
            write_csv(d / "naming.csv", nm)
            write_csv(d / "naming_digits.csv", nmd)
            all_naming.extend(nm)
            all_naming_digits.extend(nmd)
            print("        code readability (held-out AUC of the cell's own terms)")
            print("          " + "  ".join(
                f"d{r['digit']}:{r['median_auc'] if r['median_auc'] is not None else float('nan'):.2f}"
                f"({r['n_cells_validated']}/{r['n_cells_described']})" for r in nmd))
        for row in digits:
            print(f"        digit {row['digit']}: {row['n_cells']:4d} cells "
                  f"({row['codes_used']:3d} codes used), "
                  f"{row['n_multi_cells']:4d} multi-item "
                  f"(median {row['median_multi_cell_size']:.0f} items, "
                  f"radius {row['median_radius']:.2f}, shrink {row['median_shrink']:.3f}, "
                  f"brand purity {row['mean_brand_purity']:.3f}); "
                  f"{row['singleton_item_share']:.0%} of catalogue now alone")

    result = {
        "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "category": args.category,
        "variants": args.variants,
        "metadata_coverage": cov,
        "label_coverage": label_cov,
        "label_gate": ({k: {"usable": v[0], "reason": v[1]}
                        for k, v in label_gate.items()} if label_gate else None),
        "label_manifest": (L.manifest(args.category)
                           if not args.no_attributes else None),
        "n_perm": args.n_perm,
        "caveats": {
            "attributes": "The frozen `categories` field is empty for every item, so "
                          "attribute labels come from OUTSIDE the frozen substrate: "
                          "the public Amazon-2018 metadata release, joined on ASIN at "
                          "100%. The models never saw these labels. `brand` remains an "
                          "identity vocabulary (~2.6 items/value) and is gated off.",
            "text_confound": "Every label here except `also_buy` is derived from item "
                             "text, and z_i is a text embedding. A digit aligning with "
                             "a text-derived label is therefore ambiguous between "
                             "encoding meaning and encoding the encoder. `also_buy` is "
                             "behavioural and is the column that breaks that tie.",
            "MQ": "MQ is not residual: digits independently quantize the full "
                  "embedding, so its depth axis is an intersection of parallel "
                  "partitions and `shrink` must not be read as refinement.",
            "rqvae": "RQ-VAE quantizes its own encoder latent, which is unrecoverable "
                     "(no RQ-VAE checkpoint exists on disk). Its geometry here is "
                     "measured in the shared input space.",
            "conditional_null": "Conditional AMI is compared against labels permuted "
                                "WITHIN each parent cell, which preserves cluster "
                                "sizes and the parent's own label marginal and "
                                "randomizes only what digit d could add.",
            "naming": "Signatures contrast a cell against its SIBLINGS, so they "
                      "describe what digit d added rather than what the prefix "
                      "already established. The AUC is measured on held-out items "
                      "of the same cell, so a term list that merely restates the "
                      "cell cannot pass. Cells under 4 items get no signature.",
        },
        "digit_summary": all_digits,
        "attribute_alignment": all_align,
        "copurchase": all_copurchase,
        "naming_digits": all_naming_digits,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[exp1] wrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
