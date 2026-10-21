#!/usr/bin/env python
"""Declared descriptive statistics S1-S8 and expectations E1-E3 of the DiffGRM reveal trace.

Reads `trace.npz` of completed cells, refuses a cell whose validation did not
pass, and writes CSV tables, `expectations.json` and `report.md`. One user
bootstrap (same seed, hence the same resampled users in every cell). Intervals
exclude training-seed uncertainty: there is one checkpoint per cell.

    python summarize.py --run <run dir with cell-XX> --out <new summary dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common                                                            # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402

DRAWS = 2000
PATHS = ("greedy", "top1")


def load_cell(directory: Path) -> dict:
    if (directory / "status.txt").read_text().split()[0] != "complete" or not (directory / "trace.npz").is_file():
        raise RuntimeError(f"{directory} is not a complete cell")
    validation = json.loads((directory / "validation.json").read_text())
    if not (validation["V2_observer_is_noop"] and all(v["ok"] for v in validation["V3_V4_V5"].values())):
        raise RuntimeError(f"{directory}: validation did not pass")
    inputs = json.loads((directory / "inputs.json").read_text())
    with np.load(directory / "trace.npz", allow_pickle=False) as z:
        data = {name: z[name] for name in z.files}
    from sidlens import grid
    if grid.is_grid(inputs["checkpoint"]):
        i = grid.cell_info(inputs["checkpoint"])
        quantizer, depth, width = i["quantizer"], i["depth"], i["width"]
    else:
        _, _, quantizer, d_, width = inputs["checkpoint"].split("-")
        depth = int(d_[0])
    return {"dir": directory, "id": inputs["checkpoint"], "quantizer": quantizer, "depth": int(depth),
            "width": int(width), "inputs": inputs, "v6": validation["V6_exp1_archive"]["status"], "z": data}


def final_sid(order, codes):
    out = np.full(order.shape, -1, dtype=np.int64)
    np.put_along_axis(out, order.astype(np.int64), codes.astype(np.int64), 1)
    return out


def revealed_bits(order):
    """[N, depth] bit set of the digits revealed BEFORE each state of a path."""
    bits = np.zeros(order.shape, dtype=np.int64)
    for state in range(1, order.shape[1]):
        bits[:, state] = bits[:, state - 1] | (1 << order[:, state - 1].astype(np.int64))
    return bits


def summarize_cell(cell: dict, boot: UserBootstrap) -> dict:
    z, n = cell["z"], cell["depth"]
    users, N = z["users"], len(z["users"])
    training = cell["inputs"].get("training") or {}
    guided_steps = int(training.get("guided_steps") or min(n, 4))
    select = training.get("guided_select") or "least"
    key = {"cell": cell["id"], "quantizer": cell["quantizer"], "depth": n, "width": cell["width"]}

    def mean(x, rows=None):
        rows = np.ones(N, dtype=bool) if rows is None else rows
        return boot.mean(users[rows], np.asarray(x, dtype=float)[rows])

    head = {**key, "users": N, "v6_archive": cell["v6"]}
    for beam in common.BEAMS:
        head[f"sid_hit10_beam{beam}"] = float((z[f"beam{beam}_target_rank"] > 0).mean())
    primary = mean(z[f"beam{common.PRIMARY_BEAM}_target_rank"] > 0)
    head.update(sid_hit10_beam256_lo=primary["lo"], sid_hit10_beam256_hi=primary["hi"])

    trainer_order = np.argsort(-z["trainer_confidence"] if select == "least" else z["trainer_confidence"],
                               axis=1, kind="stable")
    trainer_bits = revealed_bits(trainer_order)
    first, by_step, blocks = [], [], []
    for kind in PATHS:
        found = z[f"{kind}_found"]
        order, codes = z[f"{kind}_order"][found], z[f"{kind}_codes"][found]
        for digit in range(n):                                               # S1
            first.append({**key, "path": kind, "digit": digit + 1,
                          "share_revealed_first": float((order[:, 0] == digit).mean())})
        orders, counts = np.unique(order, axis=0, return_counts=True)
        head[f"{kind}_found"] = float(found.mean())
        head[f"{kind}_modal_order"] = "".join(str(d + 1) for d in orders[counts.argmax()])
        head[f"{kind}_modal_order_share"] = float(counts.max() / len(order))
        head[f"{kind}_first_is_digit1"] = float((order[:, 0] == 0).mean())
        bits = revealed_bits(order)
        in_family = (bits == trainer_bits[found]) & ((n - np.arange(n))[None, :] <= guided_steps)
        head[f"{kind}_states_in_trained_family"] = float(in_family.mean())

        final = final_sid(order, codes)                                      # S2
        agrees = z[f"{kind}_guess"][found] == final[:, None, :]
        settled = np.flip(np.logical_and.accumulate(np.flip(agrees, 1), 1), 1)   # true from that state on
        commit = n - settled.sum(1)                                          # [F, digit]; n = never predicted
        head[f"{kind}_sid_settled_at_state0"] = float((commit.max(1) == 0).mean())
        head[f"{kind}_mean_sid_commit_state"] = float(commit.max(1).mean())
        for digit in range(n):
            head[f"{kind}_mean_commit_state_digit{digit + 1}"] = float(commit[:, digit].mean())

        reach = z[f"{kind}_target_reachable"]                                # S3
        for state in range(n):
            row = {**key, "path": kind, "state": state, "in_trained_family": float(in_family[:, state].mean()),
                   "guess_in_catalogue": float(z[f"{kind}_guess_in_catalogue"][found, state].mean()),
                   "guess_is_final_sid": float(agrees[:, state].all(1).mean()),
                   "target_reachable": float(reach[found, state].mean())}
            for name, column in (("sid_hit10", "catalogue_sid_rank"), ("item_hit10_lo", "catalogue_item_rank_hi"),
                                 ("item_hit10_hi", "catalogue_item_rank_lo")):
                est = mean(reach[:, state] & (z[f"{kind}_{column}"][:, state] <= 10) & found)
                row.update({name: est["est"], f"{name}_ci_lo": est["lo"], f"{name}_ci_hi": est["hi"]})
            by_step.append(row)

        top = z[f"{kind}_block_top"][found]                                  # S5: [F, state, block, digit]
        masked = top[:, :, -1, :] >= 0
        logp = z[f"{kind}_block_target_logp"][found].astype(np.float64)
        for block in range(top.shape[2]):
            blocks.append({**key, "path": kind, "block": block,
                           "top_code_equals_last_block": float((top[:, :, block, :] == top[:, :, -1, :])[masked].mean()),
                           "target_logp": float(logp[:, :, block, :][masked].mean())})

    state0 = next(r for r in by_step if r["path"] == "greedy" and r["state"] == 0)
    head.update(state0_catalogue_sid_hit10=state0["sid_hit10"], state0_catalogue_sid_hit10_lo=state0["sid_hit10_ci_lo"],
                state0_catalogue_sid_hit10_hi=state0["sid_hit10_ci_hi"])

    beam_rows = []                                                           # S4
    for beam in common.BEAMS:
        paths = z[f"beam{beam}_n_paths"].astype(float)
        for step in range(n):
            beam_rows.append({**key, "beam": beam, "step": step, "paths": float(paths[:, step].mean()),
                              "duplicate_share": float((1 - z[f"beam{beam}_n_distinct"][:, step] / paths[:, step]).mean()),
                              "unreachable_share": float((1 - z[f"beam{beam}_n_reachable"][:, step] / paths[:, step]).mean())})
    head["beam256_final_duplicate_share"] = beam_rows[-1]["duplicate_share"]
    head["beam256_final_unreachable_share"] = beam_rows[-1]["unreachable_share"]

    lattice = z["lattice_target_logp"].astype(np.float64)                    # S6: [N, 2^n - 1, digit]
    gains = [{**key, "revealed_digit": i + 1, "predicted_digit": j + 1,
              "target_logp_gain": float((lattice[:, 1 << i, j] - lattice[:, 0, j]).mean())}
             for i in range(n) for j in range(n) if i != j]
    head["target_best_minus_greedy_order_score"] = float((z["target_best_score"] - z["target_greedy_score"]).mean())
    head["target_best_order_differs_from_greedy"] = float((z["target_best_order"] != z["target_greedy_order"]).any(1).mean())

    has_top1 = z["top1_found"]                                               # S7
    regret = mean((z["regret_rank"] > 0).any(1), has_top1)
    head.update(regret_any_digit=regret["est"], regret_any_digit_lo=regret["lo"], regret_any_digit_hi=regret["hi"])

    confident = np.exp(z["greedy_top_logp"][:, 0, :, 0].astype(np.float64))  # S8: state 0, every digit masked
    correct = z["greedy_target_rank"][:, 0, :] == 0
    gap = mean((confident - correct).mean(1))
    head.update(state0_confidence_minus_accuracy=gap["est"], state0_confidence_minus_accuracy_lo=gap["lo"],
                state0_confidence_minus_accuracy_hi=gap["hi"], state0_top1_accuracy=float(correct.mean()),
                state0_mean_max_probability=float(confident.mean()))
    return {"head": head, "first": first, "by_step": by_step, "blocks": blocks, "beam": beam_rows, "gains": gains}


def expectations(head: pd.DataFrame) -> dict:
    by = head.set_index(["quantizer", "depth", "width"])
    e1 = [{"cell": f"{q}-{d}cb-{w}", "rq": float(by.loc[(q, d, w), "greedy_first_is_digit1"]),
           "mq": float(by.loc[("MQ", d, w), "greedy_first_is_digit1"])}
          for q in ("rqkmeans", "rqvae") for (qq, d, w) in by.index if qq == q and ("MQ", d, w) in by.index]
    e2 = int((head.state0_catalogue_sid_hit10 < head.sid_hit10_beam256).sum())
    e3 = [{"pair": f"{q}-{w}", "depth5": float(by.loc[(q, 5, w), "state0_confidence_minus_accuracy"]),
           "depth3": float(by.loc[(q, 3, w), "state0_confidence_minus_accuracy"]),
           "depth4": float(by.loc[(q, 4, w), "state0_confidence_minus_accuracy"])}
          for q in ("MQ", "rqkmeans", "rqvae") for w in (128, 512) if all((q, d, w) in by.index for d in (3, 4, 5))]
    e3_held = sum(r["depth5"] > max(r["depth3"], r["depth4"]) for r in e3)
    return {"E1": {"held": bool(e1) and all(r["rq"] > r["mq"] for r in e1), "cells_where_rq_exceeds_mq":
                   sum(r["rq"] > r["mq"] for r in e1), "cells_compared": len(e1), "rows": e1},
            "E2": {"held": e2 >= 15, "cells_where_state0_below_final": e2, "cells": len(head)},
            "E3": {"held": e3_held >= 4, "pairs_where_depth5_largest": e3_held, "pairs": len(e3), "rows": e3}}


def markdown_table(frame: pd.DataFrame) -> str:
    """A plain markdown table (pandas' own needs the optional `tabulate`)."""
    def cell(value):
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    rows = [" | ".join(frame.columns), " | ".join("---" for _ in frame.columns)]
    rows += [" | ".join(cell(value) for value in row) for row in frame.itertuples(index=False)]
    return "\n".join(f"| {row} |" for row in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, help="a run dir holding cell-*/")
    ap.add_argument("--cells", type=Path, nargs="*", default=[], help="cell dirs (grid layout; one cohort)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--skip-failed", action="store_true",
                    help="summarize the complete cells and name the others in the report, instead of refusing")
    args = ap.parse_args(argv)
    directories = (sorted(args.run.glob("cell-*")) if args.run else []) + list(args.cells)
    skipped = [d for d in directories if (d / "status.txt").read_text().split()[0] != "complete"]
    if skipped and not args.skip_failed:
        raise RuntimeError(f"cells not complete: {skipped}; pass --skip-failed to summarize the rest")
    cells = [load_cell(d) for d in directories if d not in skipped]
    skipped = [str(d) for d in skipped]
    if not cells:
        raise FileNotFoundError(f"no cells under {args.run}")
    users = cells[0]["z"]["users"]
    if any(not np.array_equal(c["z"]["users"], users) for c in cells):
        raise ValueError("cells do not hold the same users in the same order")
    args.out.mkdir(parents=True, exist_ok=False)
    boot = UserBootstrap(users, draws=args.draws, seed=common.BOOTSTRAP_SEED)
    parts = [summarize_cell(c, boot) for c in cells]
    tables = {name: pd.DataFrame([row for p in parts for row in p[name]]) for name in ("first", "by_step", "blocks", "beam", "gains")}
    head = pd.DataFrame([p["head"] for p in parts])
    head.to_csv(args.out / "per_cell.csv", index=False)
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
    verdict = expectations(head) if len(cells) == len(common.CELLS) else {"note": "expectations need all 20 cells"}
    (args.out / "expectations.json").write_text(json.dumps(verdict, indent=1))
    show = ["cell", "sid_hit10_beam256", "state0_catalogue_sid_hit10", "greedy_first_is_digit1", "greedy_modal_order",
            "greedy_states_in_trained_family", "greedy_sid_settled_at_state0", "beam256_final_duplicate_share",
            "regret_any_digit", "state0_confidence_minus_accuracy"]
    lines = ["# DiffGRM reveal trace: declared descriptive statistics", "",
             f"Run `{args.run}`; {len(cells)} cells; {len(users)} users; user bootstrap {args.draws} draws, seed "
             f"{common.BOOTSTRAP_SEED}. Intervals (in the CSV files) exclude training-seed uncertainty. SID-level and "
             "item-level numbers are in separate columns of `by_step.csv`.", "",
             f"Archive check V6: {dict(head.v6_archive.value_counts())}. Cells left out because their run did "
             f"not complete: {skipped or 'none'}.", "",
             markdown_table(head[show]), "", "## Expectations", "",
             "```json", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v
                                    for k, v in verdict.items()}, indent=1), "```", ""]
    (args.out / "report.md").write_text("\n".join(lines))
    (args.out / "inputs.json").write_text(json.dumps({"run": str(args.run), "cells": [c["id"] for c in cells],
                                                      "skipped": skipped, "draws": args.draws, "seed": common.BOOTSTRAP_SEED}, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
