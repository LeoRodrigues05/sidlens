#!/usr/bin/env python
"""Package the SidLens results cited by the WWW'27 draft, with source hashes.

The paper project (docs/paper/Leo_WebConf27_GenRecSys*.zip) keeps every plotted
number in a small CSV under data/, pinned by sha256 in a manifest, and draws
figures from those CSVs only (tools/plot_paper.py). This exporter produces the
SidLens part of that contract, in the same schema:

    data/sidlens_*.csv            packaged tables (verbatim copies or declared selections)
    data/sidlens_manifest.json    per-table: packaged/source path + sha256, rows, selection
    data/sidlens_claims.csv       every in-text SidLens number: value, source file + sha256, locator

Traps this guards against
-------------------------
* A figure drifting from its source. Every packaged table records the sha256
  of the derived file it came from, and the paper-side plotter re-hashes the
  packaged copy before drawing.
* Two collision definitions. The paper's "collision rate" is the share of items
  in non-singleton SID groups; SidLens's semantic-mapping `collision_rate` is
  duplicate excess, 1 - |unique SIDs| / |items|. Both columns are exported,
  under names that say which is which.
* Archived-decoder numbers standing in for controlled ones. `policy` in
  sidlens_decoding_cells.csv separates the shared-search conditions
  (confidence, fixed) from the archived decoder (legacy_confidence), which
  reproduces the historical metrics exactly.

Reads only $SIDLENS_WORK/derived and the frozen SID tables. CPU, seconds.

    python experiments/paper_www27/export_paper_data.py --out /tmp/www27_data
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sidlens import paths                                    # noqa: E402
from sidlens.data.sids import SidTable, available            # noqa: E402
from sidlens.provenance.hashing import sha256_file           # noqa: E402

D = paths.DERIVED
MATCHED = D / "controlled/exp1_matched_beam"
ORDERS = D / "controlled/exp2_fixed_orders/summary-194295-194297"
FIGS = D / "semantic_mapping/figures-20260922"
RETRO = D / "retrospective"


def rel(p: Path) -> str:
    return str(p.relative_to(paths.WORK))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def decoding_cells(out: Path) -> dict:
    """18 checkpoints x beams x policies: metrics and final-beam composition."""
    rows, sources = [], {}
    for f in sorted((MATCHED / "181944").glob("cell-*/result.json")):
        r = json.loads(f.read_text())
        sources[rel(f)] = sha256_file(f)
        for c in r["conditions"]:
            m, g = c["metrics"], c["generated_paths"]
            rows.append({
                "checkpoint": r["checkpoint"], "quantizer": r["quantizer"],
                "depth": r["depth"], "width": r["codebook_size"],
                "beam": c["beam_width"], "policy": c["policy"], "n_users": r["n_users"],
                "sid_hit10_pct": 100 * m["sid_hit@10"], "ndcg10_pct": 100 * m["ndcg@10"],
                "item_uniform_hit10_pct": 100 * m["item_uniform@10"],
                "generated_paths": g if g is not None else "",
                "duplicate_path_share": 1 - c["unique_final_sids"] / g if g else "",
                "invalid_path_share": c["invalid_final_paths"] / g if g else "",
                "users_with_ten_pct": 100 * c["rows_with_ten_predictions"] / r["n_users"],
            })
    p = out / "sidlens_decoding_cells.csv"
    write_csv(p, rows)
    return {"packaged_csv": p.name, "rows": len(rows), "sources_sha256": sources,
            "selection": "Every condition of all 18 cell result.json files. "
                         "duplicate/invalid shares are fractions of generated final paths "
                         "(users x beam); empty for legacy_confidence, which exposes no paths.",
            "notes": "policy=legacy_confidence is the archived decoder (greedy last digit); "
                     "confidence/fixed are the shared-search conditions."}


def decoding_contrasts(out: Path) -> dict:
    """Headline paired contrasts of both controlled decoding experiments."""
    a_path = MATCHED / "summary-181944-181946/result.json"
    b_path = ORDERS / "result.json"
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    rows = []
    for c in a["comparisons"]:
        rows.append({"experiment": "matched_beam_18_cells", "comparator": "fixed_seed42",
                     "beam": c["beam_width"], "outcome": c["metric"],
                     "confidence_pct": 100 * c["confidence_mean"],
                     "comparator_pct": 100 * c["fixed_mean"],
                     "difference_pp": 100 * c["difference"],
                     "ci95_low_pp": 100 * c["ci95"][0], "ci95_high_pp": 100 * c["ci95"][1],
                     "positive_cells": c["positive_cells"], "n_cells": c["n_cells"]})
    for c in b["overall"]:
        rows.append({"experiment": "fixed_orders_6_cells", "comparator": "mean_of_six_fixed_orders",
                     "beam": c["beam"], "outcome": c["outcome"],
                     "confidence_pct": c["confidence_pct"], "comparator_pct": c["mean_fixed_pct"],
                     "difference_pp": c["difference_pp"], "ci95_low_pp": c["ci95_pp"][0],
                     "ci95_high_pp": c["ci95_pp"][1], "positive_cells": "", "n_cells": b["n_cells"]})
    p = out / "sidlens_decoding_contrasts.csv"
    write_csv(p, rows)
    return {"packaged_csv": p.name, "rows": len(rows),
            "sources_sha256": {rel(a_path): sha256_file(a_path), rel(b_path): sha256_file(b_path)},
            "selection": "All `comparisons` of the matched-beam summary and all `overall` "
                         "rows of the fixed-order summary; fractions converted to percent.",
            "notes": "Paired-user bootstrap, 2,000 draws; seeds 20260909 and 20260915. "
                     "Intervals exclude training-seed variation."}


def verbatim(out: Path, src: Path, name: str, selection: str, notes: str = "") -> dict:
    dst = out / name
    shutil.copyfile(src, dst)
    with open(dst) as fh:
        n = sum(1 for _ in fh) - 1
    return {"packaged_csv": name, "rows": n, "source": rel(src),
            "source_sha256": sha256_file(src), "selection": selection, "notes": notes}


def map_collisions(out: Path) -> dict:
    """Both collision definitions for all 27 frozen maps."""
    rows = []
    for v in available():
        t = SidTable.load(v)
        groups = t.cb2items
        in_groups = sum(len(g) for g in groups.values() if len(g) > 1)
        rows.append({"variant": v.name, "quantizer": v.quantizer, "depth": v.n_codebook,
                     "width": v.codebook_size, "items": t.n_items, "unique_sids": len(groups),
                     "items_in_collision_groups_pct": 100 * in_groups / t.n_items,
                     "duplicate_excess_pct": 100 * (t.n_items - len(groups)) / t.n_items,
                     "sem_ids_sha256": sha256_file(t.source)})
    p = out / "sidlens_map_collisions.csv"
    write_csv(p, rows)
    return {"packaged_csv": p.name, "rows": len(rows),
            "source": "frozen/sids/sem_ids/diffgrm/*.sem_ids (hash per row)",
            "selection": "All 27 Industrial maps.",
            "notes": "items_in_collision_groups_pct is the paper's collision rate; "
                     "duplicate_excess_pct is 1 - unique/items."}


CLAIMS = [
    # (claim id, value, unit, source relative to derived/, locator)
    ("ar_prefix_first_error_digit1", "82.16 [80.84, 83.48]", "% of rows",
     "retrospective/exp2_prefix/report.md", "First-error process at rank 1, digit 1"),
    ("ar_prefix_exact_rank1", "8.75 [7.64, 9.85]", "%",
     "retrospective/exp2_prefix/report.md", "Headline findings, exact rank-1 full-SID accuracy"),
    ("ar_prefix_hazard_digit2_3", "35.89; 17.98", "% conditional hazard",
     "retrospective/exp2_prefix/report.md", "First-error process, digits 2 and 3"),
    ("ar_exact_vs_title_hr10", "16.59 vs 16.72", "%",
     "retrospective/exp2_prefix/report.md", "Headline findings, K=10"),
    ("ar_rows_prefix", "66,258 rows; 1,606 users; 18 configurations", "count",
     "retrospective/exp2_prefix/report.md", "Scope and validation"),
    ("collision_sid_minus_item_uniform_hr10", "4.54 [4.1, 5.0]", "pp",
     "retrospective/exp3_collisions/report.md", "Result, paragraph 2"),
    ("collision_gap_collided_vs_singleton", "14.1 vs 1.2", "pp",
     "retrospective/exp3_collisions/report.md", "Result, paragraph 2"),
    ("collided_minus_singleton_sid_hr10", "+42.5 [+40.0, +44.9]; 17/18 cells", "pp",
     "retrospective/exp3_collisions/report.md", "Result, paragraph 1"),
    ("next_two_conditional_top_pair", "4.03 vs 0.34; RD +3.69 [+1.14, +6.23]", "%, pp",
     "retrospective/exp4_next_two/report.md", "Result, primary conditional contingency"),
    ("next_two_cell_heterogeneity", "5/10 positive; -0.55 to +18.48", "pp",
     "retrospective/exp4_next_two/report.md", "Result, heterogeneity paragraph"),
    ("next_two_ordered_minus_reversed_hr20", "+0.23 equal-cell; +0.35 event-weighted", "pp",
     "retrospective/exp4_next_two/report.md", "Result, heterogeneity paragraph"),
    ("rqkmeans_5x512_identical_pairs", "separates 11 of 11 identical-embedding pairs; "
     "other 26 maps separate 0", "count",
     "semantic_mapping/figures-20260922/fig8_collisions.csv",
     "recomputed from frozen embeddings and .sem_ids (exporter check)"),
    ("rqkmeans_5x512_nesting", "0 / 3,105 items share the 3x512 prefix (4x512: 3,105)", "count",
     "semantic_mapping/figures-20260922/fig8_collisions.csv",
     "recomputed from frozen .sem_ids (exporter check)"),
]


def claims(out: Path) -> dict:
    import numpy as np
    from collections import defaultdict
    from sidlens.data import embeddings as E

    # Re-derive the two RQ-KMeans 5x512 claims rather than copying prose.
    base = SidTable.load("rqkmeans_3codebook_128")
    X = E.load_aligned("Industrial_and_Scientific", base.keys, dtype=np.float32)
    groups = defaultdict(list)
    for k, row in zip(base.keys, X):
        groups[row.tobytes()].append(k)
    ident = [g for g in groups.values() if len(g) > 1]
    split = {v.name: sum(len({SidTable.load(v).asin2codes[k] for k in g}) > 1 for g in ident)
             for v in available()}
    assert len(ident) == 11 and split["rqkmeans_5codebook_512"] == 11
    assert sum(split.values()) == 11, split
    a, b = SidTable.load("rqkmeans_5codebook_512"), SidTable.load("rqkmeans_3codebook_512")
    c = SidTable.load("rqkmeans_4codebook_512")
    assert sum(a.asin2codes[k][:3] == b.asin2codes[k] for k in a.keys) == 0
    assert sum(c.asin2codes[k][:3] == b.asin2codes[k] for k in a.keys) == 3105

    rows = []
    for cid, value, unit, src, loc in CLAIMS:
        f = D / src
        rows.append({"claim": cid, "value": value, "unit": unit, "source": rel(f),
                     "source_sha256": sha256_file(f), "locator": loc})
    p = out / "sidlens_claims.csv"
    write_csv(p, rows)
    return {"packaged_csv": p.name, "rows": len(rows),
            "selection": "Every SidLens number quoted in the text that is not plotted.",
            "notes": "Values copied from generated reports whose source of truth is the "
                     "adjacent result.json; the RQ-KMeans 5x512 claims are recomputed here."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="must not exist")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {
        "decoding_cells": decoding_cells(args.out),
        "decoding_contrasts": decoding_contrasts(args.out),
        "fixed_orders_cells": verbatim(
            args.out, ORDERS / "quality_cost.csv", "sidlens_fixed_orders_cells.csv",
            "All 42 rows (6 depth-3 cells x confidence + six fixed orders), verbatim."),
        "fixed_orders_per_cell": verbatim(
            args.out, ORDERS / "per_cell.csv", "sidlens_fixed_orders_per_cell.csv",
            "All rows verbatim: per-cell confidence minus mean fixed, with paired intervals."),
        "digit_structure": verbatim(
            args.out, FIGS / "fig5_digit_info.csv", "sidlens_digit_structure.csv",
            "All 27 maps x digits, verbatim.",
            "standalone_between_share = embedding variance explained by one digit alone; "
            "conditional_ami is within-parent, gated by conditional_supported."),
        "category_purity": verbatim(
            args.out, FIGS / "fig4_coherence.csv", "sidlens_category_purity.csv",
            "All rows verbatim; the figure uses field=cat_l1 at 4x128.",
            "Null = 200 label permutations within fixed prefix groups."),
        "prefix_groups": verbatim(
            args.out, FIGS / "fig1_icicle.csv", "sidlens_prefix_groups.csv",
            "All 12 rows verbatim (4x128 fits)."),
        "refinement": verbatim(
            args.out, FIGS / "fig3_geometry.csv", "sidlens_refinement.csv",
            "All rows verbatim.",
            "refinement_index = 1 - R2_within / E[R2 of a random partition with the same k]."),
        "map_collisions": map_collisions(args.out),
        "claims": claims(args.out),
    }
    for s in sources.values():
        s["packaged_sha256"] = sha256_file(args.out / s["packaged_csv"])
    doc = {"description": "SidLens results cited in the WWW'27 draft (controlled decoding, "
                          "retrospective AR audits, identifier-map structure). Generated by "
                          "sidlens experiments/paper_www27/export_paper_data.py.",
           "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
           "sources": sources}
    (args.out / "sidlens_manifest.json").write_text(json.dumps(doc, indent=2))
    for k, s in sources.items():
        print(f"{s['packaged_csv']:38s} {s['rows']:4d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
