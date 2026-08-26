#!/usr/bin/env python3
"""Fold one variant's training and evaluation artefacts into a single metrics JSON.

Metrics are parsed out of calc.py's stdout rather than recomputed, so the numbers
this sweep reports are produced by exactly the same scorer as the results already
recorded in the project notes.

Two modes:
  (default)     collect artefacts for one variant into <metrics-dir>/<id>.json
  --is-best ID  print "best" if ID has the highest NDCG@10 among completed
                variants sharing its (tree, category), else "not-best"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

# calc.py prints: n_beam, the valid top-k list, then "NDCG:\t[...]" and "HR\t[...]".
TOPK_RE = re.compile(r"^\[(?P<body>[\d,\s]+)\]\s*$", re.MULTILINE)
ARRAY_RE = {
    "NDCG": re.compile(r"NDCG:\s*\[(?P<body>[^\]]*)\]", re.DOTALL),
    "HR": re.compile(r"HR\s*\[(?P<body>[^\]]*)\]", re.DOTALL),
}


# calc_two_item.py reports differently: one "  HR@k: v    NDCG@k: v" line per k.
TWO_ITEM_RE = re.compile(
    r"HR@(?P<k>\d+):\s*(?P<hr>[\d.]+)\s+NDCG@(?P=k):\s*(?P<ndcg>[\d.]+)"
)


def parse_eval_log(path: Path) -> dict:
    """Pull the HR/NDCG vectors out of the scorer's stdout.

    calc.py prints numpy arrays; calc_two_item.py prints one line per k. The
    two-item form is tried first since its lines are unambiguous.
    """
    text = path.read_text(errors="replace")

    pairs = TWO_ITEM_RE.findall(text)
    if pairs:
        out: dict[str, float] = {}
        for k, hr, ndcg in pairs:
            out[f"HR@{k}"] = round(float(hr) * 100, 4)
            out[f"NDCG@{k}"] = round(float(ndcg) * 100, 4)
        return out

    topks = None
    for match in TOPK_RE.finditer(text):
        candidate = [int(x) for x in match.group("body").replace(",", " ").split()]
        # calc.py's valid_topk is a subset of this fixed ladder.
        if candidate and set(candidate).issubset({1, 3, 5, 10, 20, 50}):
            topks = candidate
    if topks is None:
        raise ValueError(f"no top-k list found in {path}")

    out: dict[str, float] = {}
    for name, pattern in ARRAY_RE.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"no {name} array found in {path}")
        values = [float(x) for x in match.group("body").split()]
        if len(values) != len(topks):
            raise ValueError(
                f"{name} has {len(values)} values but top-k list has {len(topks)} in {path}"
            )
        for k, value in zip(topks, values):
            # Percentages, matching how the notes report them.
            out[f"{name}@{k}"] = round(value * 100, 4)
    return out


def parse_trainer_state(output_dir: Path) -> dict:
    """Read the newest checkpoint's trainer_state for the loss curve and stopping point."""
    candidates = sorted(
        glob.glob(str(output_dir / "checkpoint-*" / "trainer_state.json")),
        key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
    )
    if not candidates:
        return {}
    state = json.loads(Path(candidates[-1]).read_text())
    curve = [
        {"epoch": entry.get("epoch"), "eval_loss": entry["eval_loss"]}
        for entry in state.get("log_history", [])
        if "eval_loss" in entry
    ]
    return {
        "best_eval_loss": state.get("best_metric"),
        "best_model_checkpoint": state.get("best_model_checkpoint"),
        "stopped_at_step": state.get("global_step"),
        "stopped_at_epoch": state.get("epoch"),
        "max_steps": state.get("max_steps"),
        "eval_loss_curve": curve,
    }


def read_seconds(path: Path) -> int:
    """Training may span several jobs; the file accumulates one line per attempt."""
    if not path.is_file():
        return 0
    return sum(int(line) for line in path.read_text().split() if line.strip().isdigit())


def collect(args: argparse.Namespace) -> dict:
    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)

    record: dict = {"variant_id": args.variant_id}

    assets_path = metrics_dir / f"{args.variant_id}.assets.json"
    if assets_path.is_file():
        assets = json.loads(assets_path.read_text())
        assets.pop("paths", None)
        record.update(assets)

    record.update(parse_trainer_state(output_dir))
    record["metrics"] = parse_eval_log(metrics_dir / f"{args.variant_id}.eval.log")
    record["train_seconds"] = read_seconds(Path(args.train_seconds_file))
    record["eval_seconds"] = args.eval_seconds

    out_path = metrics_dir / f"{args.variant_id}.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def is_best(metrics_dir: Path, variant_id: str, key: str = "NDCG@10") -> str:
    target_path = metrics_dir / f"{variant_id}.json"
    if not target_path.is_file():
        return "not-best"
    target = json.loads(target_path.read_text())
    score = target.get("metrics", {}).get(key)
    if score is None:
        return "not-best"

    for path in metrics_dir.glob("*.json"):
        # .assets.json and .predictions.json are sidecars, not variant records;
        # predictions are a JSON list, so .get() below would blow up on them.
        if path.name.endswith((".assets.json", ".predictions.json")) or path.stem == variant_id:
            continue
        try:
            other = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if other.get("tree") != target.get("tree"):
            continue
        if other.get("category") != target.get("category"):
            continue
        rival = other.get("metrics", {}).get(key)
        if rival is not None and rival > score:
            return "not-best"
    return "best"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--is-best", dest="best_of", default=None)
    parser.add_argument("--variant-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-seconds-file", default="")
    parser.add_argument("--eval-seconds", type=int, default=0)
    args = parser.parse_args()

    if args.best_of:
        print(is_best(Path(args.metrics_dir), args.best_of))
        return

    if not (args.variant_id and args.output_dir):
        parser.error("--variant-id and --output-dir are required unless --is-best is given")
    record = collect(args)
    print(json.dumps(record.get("metrics", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
