"""A grid run's saved test predictions and decoder settings, and the V2a reproduction check.

Every grid run's `evaluation/` holds `protocol.json` (the evaluator's arguments
and resolved dtype) and `predictions.jsonl` (one row per test example, keyed
by `example_id`, with the ranked list). An experiment that re-decodes a run
must first reproduce those lists with the run's own settings (V2a); only then
does a changed condition mean anything.

Traps, each with its guard
--------------------------
1. **Settings guessed instead of read.** Beam width, list length, batch sizes
   and dtype differ between the smoke and full grids. `ar_protocol` and
   `diffgrm_protocol` read them from the run; nothing is defaulted.
2. **A predictions file that is not this run's.** `load` checks the file's
   fingerprint against the evaluation provenance the registry already
   verified (`checks.predictions`) and refuses an unusable run.
3. **Ties read as failures, failures read as ties.** A row may differ from the
   saved list only if both lists have the same length, the same multiset of
   SIDs up to the last tied score, and sorted scores within `tie_atol`; at
   most `max_tie_share` of rows may differ at all (`decode_checks`' rule).
   Scores of identical lists must agree within `score_atol`.
4. **Sampled rows.** Office decodes cover the 6,297-user sample; the check
   compares the rows decoded, by example id, never by position.
5. **Pilots on partial evaluations.** The handoff smoke runs evaluated a few
   users. `require_full=False` and `only_saved=True` let a `--limit` pilot run
   against them; a full run never passes either.
"""

from __future__ import annotations

import json
from pathlib import Path

PARADIGM_SCORE = {"ar": "joint_logprob", "diffgrm": "score"}
# OneDiffRec's run_experiment.py passes --batch-size 8 to every DiffGRM test
# evaluation; protocol.json does not record it.
DIFFGRM_EVAL_BATCH = 8


def _eval_dir(e: dict) -> Path:
    return Path(e["run_dir"]) / "evaluation"


def ar_protocol(e: dict, require_full: bool = True) -> dict:
    """The AR evaluator's decode settings for this run (`evaluate_ar_joint` arguments)."""
    if e["paradigm"] != "ar":
        raise ValueError(f"{e['key']} is not an AR run")
    p = json.loads((_eval_dir(e) / "protocol.json").read_text())
    a = p["arguments"]
    if a.get("max_examples") and require_full:
        raise ValueError(f"{e['key']}: the saved evaluation covers only {a['max_examples']} rows")
    return {"num_beams": int(a["num_beams"]), "top_k": int(a["top_k"]), "batch_size": int(a["batch_size"]),
            "score_batch_size": int(a["score_batch_size"]), "dtype": p["resolved_dtype"].replace("torch.", ""),
            "horizon": 2 if a["predtype"] == "next2item" else 1, "protocol_hash": p["protocol_hash"]}


def diffgrm_protocol(e: dict, require_full: bool = True) -> dict:
    """The DiffGRM evaluator's decode settings (`evaluate_diff_joint` protocol)."""
    if e["paradigm"] != "diffgrm":
        raise ValueError(f"{e['key']} is not a DiffGRM run")
    p = json.loads((_eval_dir(e) / "protocol.json").read_text())
    if not p.get("full_cohort", False) and require_full:
        raise ValueError(f"{e['key']}: the saved evaluation is not the full cohort")
    return {"beam_size": int(p["beam_size"]), "top_k": int(p["top_k"]), "seed": int(p["seed"]),
            "reveal_order": [int(x) for x in p["reveal_order"]], "protocol": p["evaluation_protocol"],
            "batch_size": DIFFGRM_EVAL_BATCH, "device": p.get("device")}


def load(e: dict) -> dict[str, dict]:
    """example_id -> saved row (with its `predictions` list), after the provenance check."""
    if not e.get("usable"):
        raise RuntimeError(f"{e['key']} is not usable: {e.get('problems')}")
    if e.get("checks", {}).get("predictions") != "ok":
        raise RuntimeError(f"{e['key']}: predictions.jsonl does not match its evaluation provenance")
    rows = {}
    with (_eval_dir(e) / "predictions.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["example_id"] in rows:
                raise ValueError(f"{e['key']}: duplicate example id {r['example_id']}")
            rows[r["example_id"]] = r
    return rows


def _key(c: dict) -> tuple:
    return tuple(c["sids"])


def compare(mine: dict[str, list[dict]], saved: dict[str, dict], paradigm: str, *,
            score_atol: float = 1e-4, tie_atol: float = 1e-4, max_tie_share: float = 0.001,
            only_saved: bool = False) -> dict:
    """V2a: re-decoded lists against the saved ones (trap 3). Returns a report with `ok`."""
    field = PARADIGM_SCORE[paradigm]
    n_same, n_tie, bad, worst = 0, 0, [], 0.0
    if only_saved:                                       # pilots on partial evaluations (trap 5)
        mine = {k: v for k, v in mine.items() if k in saved}
    for eid, cands in mine.items():
        if eid not in saved:
            bad.append({"example_id": eid, "why": "not in predictions.jsonl"})
            continue
        old = saved[eid]["predictions"]
        a, b = [_key(c) for c in cands], [_key(c) for c in old]
        sa, sb = [float(c[field]) for c in cands], [float(c[field]) for c in old]
        if a == b:
            d = max((abs(x - y) for x, y in zip(sa, sb)), default=0.0)
            worst = max(worst, d)
            if d <= score_atol:
                n_same += 1
            else:
                bad.append({"example_id": eid, "why": f"same list, score diff {d:.2e}"})
            continue
        tied = len(a) == len(b) and all(abs(x - y) <= tie_atol for x, y in zip(sorted(sa), sorted(sb)))
        if tied:
            # SIDs above the last score tied with the boundary must be the same set
            cut = min(sa[-1], sb[-1]) + tie_atol
            tied = {k for k, s in zip(a, sa) if s > cut} == {k for k, s in zip(b, sb) if s > cut}
        if tied:
            n_tie += 1
        else:
            bad.append({"example_id": eid, "why": "different list", "mine": a[:3], "saved": b[:3]})
    n = len(mine)
    ok = not bad and n_tie <= max_tie_share * max(n, 1)
    return {"rows": n, "identical": n_same, "tie_reordered": n_tie, "failed": len(bad),
            "max_abs_score_diff_identical": worst, "score_atol": score_atol, "tie_atol": tie_atol,
            "max_tie_share": max_tie_share, "ok": bool(ok), "examples": bad[:20]}
