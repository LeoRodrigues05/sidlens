#!/usr/bin/env python
"""Build the review page for the DiffGRM reveal trace (representation/exp4, run 310761).

    python docs/results/diffusion_trace/build_report.py

Reads only result files (no model, no GPU): the summary tables, each cell's
validation record, the archive-difference diagnosis of the cells that stopped
at V6, and one cell's `trace.npz` for the worked examples. Writes
`sidlens_diffusion_trace.html` beside this file from `report_template.html`,
and a `_standalone.html` copy that opens from disk. Every number on the page
comes from the files listed in its "Sources" section, with their sha256.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.data import labels as L                                     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

BASE = paths.DERIVED / "representation/exp4_diffusion_reveal_trace"
RUN, SUMMARY, DIAG = BASE / "310761", BASE / "summary-310761-16cells", BASE / "310761.v6-diagnosis"
QLAB = {"rqkmeans": "RQ-KMeans", "rqvae": "RQ-VAE", "MQ": "MQ"}
EXAMPLE_CELL = 14                                                        # diff-next1-rqvae-4cb-128, a matched cell


def label(q, depth, width):
    return f"{QLAB[q]} {depth}×{width}"


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def cells() -> list[dict]:
    h = pd.read_csv(SUMMARY / "per_cell.csv")
    step = pd.read_csv(SUMMARY / "by_step.csv")
    start = step[(step.path == "greedy") & (step.state == 0)].set_index("cell")
    out = []
    for _, c in h.iterrows():
        row = {"id": c.cell, "q": c.quantizer, "depth": int(c.depth), "width": int(c.width),
               "label": label(c.quantizer, c.depth, c.width), "v6": c.v6_archive}
        for k in ("sid_hit10_beam1", "sid_hit10_beam64", "sid_hit10_beam256", "sid_hit10_beam256_lo",
                  "sid_hit10_beam256_hi", "state0_catalogue_sid_hit10", "state0_catalogue_sid_hit10_lo",
                  "state0_catalogue_sid_hit10_hi", "greedy_first_is_digit1", "greedy_modal_order_share",
                  "greedy_states_in_trained_family", "greedy_sid_settled_at_state0", "top1_sid_settled_at_state0",
                  "beam256_final_duplicate_share", "beam256_final_unreachable_share", "regret_any_digit",
                  "regret_any_digit_lo", "regret_any_digit_hi", "state0_confidence_minus_accuracy",
                  "state0_confidence_minus_accuracy_lo", "state0_confidence_minus_accuracy_hi", "state0_top1_accuracy",
                  "state0_mean_max_probability", "target_best_order_differs_from_greedy",
                  "target_best_minus_greedy_order_score", "top1_found"):
            row[k] = r(c[k])
        row["modal_order"] = str(c.greedy_modal_order)
        row["state0_item_hit10_lo"], row["state0_item_hit10_hi"] = r(start.loc[c.cell, "item_hit10_lo"]), r(start.loc[c.cell, "item_hit10_hi"])
        out.append(row)
    return out


def tables() -> dict:
    first = pd.read_csv(SUMMARY / "first.csv")
    first = first[first.path == "greedy"]
    step = pd.read_csv(SUMMARY / "by_step.csv")
    beam = pd.read_csv(SUMMARY / "beam.csv")
    blocks = pd.read_csv(SUMMARY / "blocks.csv")
    blocks = blocks[blocks.path == "greedy"]
    gains = pd.read_csv(SUMMARY / "gains.csv")
    greedy = step[step.path == "greedy"]
    return {
        "first": {cid: [r(v) for v in g.sort_values("digit").share_revealed_first] for cid, g in first.groupby("cell")},
        "family_by_depth": {int(d): [r(v) for v in g.groupby("state").in_trained_family.mean()]
                            for d, g in greedy.groupby("depth")},
        "guess_by_depth": {int(d): {"final": [r(v) for v in g.groupby("state").guess_is_final_sid.mean()],
                                    "legal": [r(v) for v in g.groupby("state").guess_in_catalogue.mean()]}
                           for d, g in greedy.groupby("depth")},
        "beam_by_depth": {int(b): {int(d): {"dup": [r(v) for v in g.groupby("step").duplicate_share.mean()],
                                            "dead": [r(v) for v in g.groupby("step").unreachable_share.mean()]}
                                   for d, g in gb.groupby("depth")} for b, gb in beam.groupby("beam")},
        "blocks_by_q": {q: {"agree": [r(v) for v in g.groupby("block").top_code_equals_last_block.mean()],
                            "logp": [r(v) for v in g.groupby("block").target_logp.mean()],
                            "lo": [r(v) for v in g.groupby("block").top_code_equals_last_block.min()],
                            "hi": [r(v) for v in g.groupby("block").top_code_equals_last_block.max()]}
                        for q, g in blocks.groupby("quantizer")},
        "blocks": {cid: [r(v) for v in g.sort_values("block").top_code_equals_last_block] for cid, g in blocks.groupby("cell")},
        "gains": {cid: [[None if i == j else r(g[(g.revealed_digit == i + 1) & (g.predicted_digit == j + 1)]
                                              .target_logp_gain.iloc[0], 3) for j in range(int(g.depth.iloc[0]))]
                        for i in range(int(g.depth.iloc[0]))] for cid, g in gains.groupby("cell")},
        "gain_mean": {cid: r(g.target_logp_gain.mean(), 3) for cid, g in gains.groupby("cell")},
    }


def decode_gaps() -> dict:
    """Paired differences in SID Hit@10 from the beam-256 list, per saved checkpoint.

    Not on the page at present: the comparison of one forward pass, the greedy
    path and the beam was taken out of the co-author version on 2026-10-13 and
    may return. Kept so that it returns with the same numbers.

    The beam-256 number is controlled exp1's (V6 checks it list by list), so
    the page shows only what the trace adds: how far one forward pass and the
    single greedy path fall short of it, on the same users and bootstrap draws.
    """
    from sidlens.analysis.bootstrap import UserBootstrap

    boot, out = None, {}
    for d in sorted(RUN.glob("cell-*")):
        if not (d / "trace.npz").is_file():
            continue
        cid = json.loads((d / "inputs.json").read_text())["checkpoint"]
        with np.load(d / "trace.npz", allow_pickle=False) as z:
            users = z["users"]
            beam = (z["beam256_target_rank"] > 0).astype(float)
            one = (z["greedy_target_reachable"][:, 0] & (z["greedy_catalogue_sid_rank"][:, 0] <= 10)).astype(float)
            greedy = (z["beam1_target_rank"] > 0).astype(float)
        boot = boot or UserBootstrap(users, draws=2000, seed=20261013)
        out[cid] = {name: {k: r(100 * v, 3) for k, v in boot.paired_difference((users, x), (users, beam)).items()
                           if k in ("est", "lo", "hi")} for name, x in (("one", one), ("greedy", greedy))}
    return out


def validation() -> dict:
    rows, failed = [], []
    for d in sorted(RUN.glob("cell-*")):
        inp = json.loads((d / "inputs.json").read_text())
        _, _, q, depth, width = inp["checkpoint"].split("-")
        base = {"cell": int(d.name[5:]), "label": label(q, int(depth[0]), int(width)), "q": q}
        if (d / "validation.json").is_file():
            v = json.loads((d / "validation.json").read_text())
            rows.append({**base, "v3": v["V3_V4_V5"]["v3_path_score_max_abs_diff"]["value"],
                         "v4": v["V3_V4_V5"]["v4_last_block_max_abs_diff"]["value"],
                         "v6": v["V6_exp1_archive"]["status"], "seconds": v["seconds"],
                         "v6_score": max((c["score_max_absolute_difference"] for c in v["V6_exp1_archive"].get("conditions", [])),
                                         default=None)})
        else:
            diag = json.loads((DIAG / d.name / "archive_diff.json").read_text())
            entry = {**base, "archive_gpu": diag["archive_environment"].get("gpu"), "gpu": diag["device"], "beams": {}}
            for beam, b in diag["beams"].items():
                kinds = sorted({"same SIDs, other order" if x["same_sids_other_order"] else "another SID at the list boundary"
                                for x in b["rows"]})
                entry["beams"][beam] = {"users": b["users_with_a_different_list"], "kinds": kinds,
                                        "spread": b["max_score_spread_of_entries_involved"],
                                        "rank_changed": b["users_whose_target_rank_changed"],
                                        "hit_here": r(b["sid_hit10_here"], 5), "hit_archive": r(b["sid_hit10_archive"], 5),
                                        "score_diff": b["max_abs_score_diff_at_identical_entries"]}
            failed.append(entry)
    return {"passed": rows, "failed": failed}


def examples() -> dict:
    """Three users of one cell, each state of the first-ranked SID's best path at beam 256."""
    cell = RUN / f"cell-{EXAMPLE_CELL:02d}"
    inp = json.loads((cell / "inputs.json").read_text())
    entry = load_runtime()[inp["checkpoint"]]
    cohort = load_eval_cohort(entry)
    if cohort.cohort_sha256 != inp["cohort_sha256"]:
        raise ValueError("cohort differs from the traced one")
    lab = L.load("Industrial_and_Scientific")
    by_id = {int(v["item_id"]): v for v in lab.values()}

    def item(i):
        v = by_id[int(i)]
        return {"title": html_lib.unescape(v["title_ext"])[:90], "cat": html_lib.unescape(v["cat_leaf"])}

    with np.load(cell / "trace.npz", allow_pickle=False) as z:
        z = {k: z[k] for k in z.files}
    n = int(inp["n_digit"])
    order, codes = z["top1_order"].astype(int), z["top1_codes"].astype(int)
    final = np.full(order.shape, -1)
    ok = z["top1_found"]
    np.put_along_axis(final, np.where(ok[:, None], order, np.arange(n)[None]), np.where(ok[:, None], codes, -1), 1)
    rank = z["beam256_target_rank"]
    revised = (z["top1_guess"][:, 0] != final).any(1)
    picks = {"Hit after the guess changed": np.flatnonzero(ok & (rank == 1) & revised),
             "Hit that never changed": np.flatnonzero(ok & (rank == 1) & ~revised),
             "Miss": np.flatnonzero(ok & (rank == 0) & revised & z["top1_target_reachable"][:, 1])}
    out = []
    for kind, idx in picks.items():
        u = int(idx[0])
        L_ = int(cohort.history_lengths[u])
        last = tuple(int(c) for c in cohort.histories[u, L_ - 1])
        final_sid = tuple(int(c) for c in final[u])
        states = []
        for t in range(n):
            revealed = set(order[u, :t].tolist())
            digits = []
            for d in range(n):
                if d in revealed:
                    digits.append({"revealed": True, "code": int(final[u, d])})
                else:
                    digits.append({"revealed": False, "code": int(z["top1_top_codes"][u, t, d, 0]),
                                   "p": r(float(np.exp(z["top1_top_logp"][u, t, d, 0].astype(float))), 3),
                                   "target_rank": int(z["top1_target_rank"][u, t, d]) + 1})
            states.append({"digits": digits, "commit_digit": int(order[u, t]), "commit_code": int(codes[u, t]),
                           "guess_legal": bool(z["top1_guess_in_catalogue"][u, t]),
                           "target_reachable": bool(z["top1_target_reachable"][u, t]),
                           "catalogue_rank": int(z["top1_catalogue_sid_rank"][u, t])})
        bucket = cohort.sid_buckets.get(final_sid, ())
        out.append({"kind": kind, "user_row": u, "history_length": L_,
                    "target": [int(c) for c in cohort.target_sids[u]], "target_item": item(cohort.target_item_ids[u]),
                    "last": list(last), "last_item": item(cohort.sid_buckets[last][0]), "last_bucket": len(cohort.sid_buckets[last]),
                    "final": list(final_sid), "final_item": item(bucket[0]) if bucket else None, "final_bucket": len(bucket),
                    "final_is_last": final_sid == last, "target_is_last": tuple(int(c) for c in cohort.target_sids[u]) == last,
                    "beam_rank": int(rank[u]), "states": states})
    return {"cell": label(*(lambda p: (p[2], int(p[3][0]), int(p[4])))(inp["checkpoint"].split("-"))),
            "n": n, "K": int(inp["codebook_size"]), "rows": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=HERE.parent / "sidlens_diffusion_trace.html")
    a = ap.parse_args()
    exp = json.loads((SUMMARY / "expectations.json").read_text())
    sources = [SUMMARY / f for f in ("per_cell.csv", "by_step.csv", "first.csv", "beam.csv", "blocks.csv", "gains.csv")]
    sources += sorted(DIAG.glob("cell-*/archive_diff.json")) + [RUN / f"cell-{EXAMPLE_CELL:02d}" / "trace.npz"]
    val = validation()
    data = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "cells": cells(), **tables(),
            "validation": val, "examples": examples(), "expectations": exp, "users": 6297,
            "seconds_total": r(sum(v["seconds"] for v in val["passed"]), 0),
            "sources": [{"path": str(p.relative_to(paths.WORK)), "sha256": sha256_file(p)[:16]} for p in sources]}
    html = (HERE.parent / "report_template.html").read_text()
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = html.replace("/*__DATA__*/null", blob)
    a.out.write_text(page)
    # The artifact viewer wraps the published page in a document skeleton; the
    # standalone copy carries the same skeleton so it renders identically when
    # opened from disk (charset for × and –, [hidden] for tooltips, viewport).
    standalone = a.out.with_name(a.out.stem + "_standalone.html")
    standalone.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,-apple-system,'
        '"Segoe UI",sans-serif;background:#fafafa}img{max-width:100%}[hidden]{display:none!important}</style>\n'
        '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} KB) and {standalone.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
