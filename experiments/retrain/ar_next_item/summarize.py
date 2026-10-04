#!/usr/bin/env python
"""Summarize a retraining run: one row per configuration with its status,
training against the archived record, and acceptance against the archived
evaluation (protocol.md). Reads only the manifests that run.py wrote.

    python experiments/retrain/ar_next_item/summarize.py \
        --roots $SIDLENS_WORK/runs/ar_next_item_retrain/<group> [...] --out <dir>

The output directory must not exist (results are never overwritten).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for man in sorted(root.glob("nextitem__*/manifest.json")):
            m = json.loads(man.read_text())
            acc = m.get("acceptance", {})
            tr = m.get("training", {})
            tvr = acc.get("training_vs_record", {})
            new = acc.get("new_calc_metrics") or {}
            arc = acc.get("archived_calc_metrics") or {}
            agr = acc.get("agreement_with_archive", {})
            orig = acc.get("agreement_with_original_weights", {})
            out.append({
                "sid_name": m["sid_name"], "role": m["role"], "status": m["status"],
                "group": root.name, "train_seconds": tr.get("train_seconds"), "ngpu": tr.get("ngpu"),
                "best_eval_loss_new": (tvr.get("best_eval_loss") or [None, None])[0],
                "best_eval_loss_archived": (tvr.get("best_eval_loss") or [None, None])[1],
                "stopped_step_new": (tvr.get("stopped_at_step") or [None, None])[0],
                "stopped_step_archived": (tvr.get("stopped_at_step") or [None, None])[1],
                "HR@10_new": new.get("HR@10"), "HR@10_archived": arc.get("HR@10"),
                "NDCG@10_new": new.get("NDCG@10"), "NDCG@10_archived": arc.get("NDCG@10"),
                "dHR@10": (acc.get("delta") or {}).get("HR@10"),
                "dNDCG@10": (acc.get("delta") or {}).get("NDCG@10"),
                "exact_sid_HR@10_new": (acc.get("exact_sid_hr10") or {}).get("new"),
                "exact_sid_HR@10_archived": (acc.get("exact_sid_hr10") or {}).get("archived"),
                "top1_agree_archive": agr.get("top1_agree"),
                "top10_overlap_archive": agr.get("mean_top10_overlap"),
                "top1_agree_original_weights": orig.get("top1_agree"),
                "snapshots": len([s for s in tr.get("snapshots", []) if s.get("kept")]),
                "weights_sha256": ((m.get("export") or {}).get("files", {})
                                   .get("model.safetensors", {}).get("sha256")),
                "error": m.get("error"),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    table = rows(args.roots)
    if not table:
        raise SystemExit("no manifests found")
    args.out.mkdir(parents=True)
    with open(args.out / "retrain_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    n = {s: sum(r["status"] == s for r in table) for s in sorted({r["status"] for r in table})}
    lines = ["# Next-item Qwen-AR retraining: summary", "",
             f"Configurations: {len(table)}; status counts: {n}", "",
             "| Config | Role | Status | HR@10 new / archived | NDCG@10 new / archived | "
             "Best val. loss new / archived | Top-1 = archived |",
             "|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['sid_name']} | {r['role']} | {r['status']} | "
                     f"{r['HR@10_new']} / {r['HR@10_archived']} | {r['NDCG@10_new']} / "
                     f"{r['NDCG@10_archived']} | {r['best_eval_loss_new']} / "
                     f"{r['best_eval_loss_archived']} | {r['top1_agree_archive']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
