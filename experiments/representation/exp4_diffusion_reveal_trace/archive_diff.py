#!/usr/bin/env python
"""Describe how a cell's decoded lists differ from controlled exp1's archive (a V6 failure).

`run.py` stops when the beam-64 or beam-256 lists are not exactly the
archive's. This script re-decodes the same cell with the same settings and
says what the difference is, user by user: the same SIDs in another order, or
other SIDs; how far apart the scores of the entries involved are; and whether
the target's rank changed. It decides nothing. It exists so that any change to
V6 is made knowing the size and kind of the difference, not guessed.

    python archive_diff.py --cell 7 --out <new dir> [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common                                                            # noqa: E402
from sidlens import paths                                                # noqa: E402
from sidlens.analysis.matched_decode import decode                       # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(common.EXP1_CELLS)), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--archive", type=Path, default=paths.DERIVED / "controlled" / "exp1_matched_beam" / "181944")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    ckpt_id = common.CELLS[args.cell]
    entry = load_runtime()[ckpt_id]
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    cohort = load_eval_cohort(entry)
    model, _ = diffusion.load(entry, device=args.device)
    N, allowed = len(cohort), set(cohort.sid_buckets)
    archive = args.archive / f"cell-{args.cell:02d}"
    previous = json.loads((archive / "result.json").read_text())
    report = {"checkpoint": ckpt_id, "archive": str(archive), "users": N,
              "device": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
              "torch": torch.__version__, "archive_environment": {k: previous.get(k) for k in ("torch_version", "cuda_version", "gpu", "batch_size", "decoder_chunk_size")},
              "identity_equal": previous["checkpoint_sha256"] == entry["ckpt_sha256"]
              and previous["cohort_sha256"] == cohort.cohort_sha256, "beams": {}}
    with np.load(archive / "predictions.npz", allow_pickle=False) as old:
        axes = list(zip(old["condition_beams"].tolist(), old["condition_policies"].tolist()))
        for beam in (64, 256):
            j = axes.index((beam, "confidence"))
            codes, scores = [], []
            for start in range(0, N, common.BATCH):
                enc = diffusion.encode(model, torch.from_numpy(cohort.histories[start:start + common.BATCH]).to(args.device),
                                       torch.from_numpy(cohort.history_mask[start:start + common.BATCH]).to(args.device))
                result = decode(model, enc, beam_width=beam, policy="confidence", top_k=10, allowed_sids=allowed,
                                decoder_chunk_size=common.CHUNK)
                codes.append(result.codes.cpu().numpy())
                scores.append(result.scores.cpu().numpy())
            codes, scores = np.concatenate(codes), np.concatenate(scores)
            old_codes, old_scores = old["codes"][j, :N], old["scores"][j, :N]
            differs = np.flatnonzero((codes != old_codes).any((1, 2)))
            finite = np.isfinite(scores) & np.isfinite(old_scores) & (codes == old_codes).all(2)
            rows = []
            for user in differs:
                mine = [tuple(int(c) for c in sid) for sid in codes[user] if sid[0] >= 0]
                theirs = [tuple(int(c) for c in sid) for sid in old_codes[user] if sid[0] >= 0]
                my_score = {sid: float(s) for sid, s in zip(mine, scores[user])}
                their_score = {sid: float(s) for sid, s in zip(theirs, old_scores[user])}
                shared = set(mine) & set(theirs)
                moved = [sid for sid in shared if mine.index(sid) != theirs.index(sid)]
                only = sorted(set(mine) ^ set(theirs))
                target = tuple(int(c) for c in cohort.target_sids[user])
                involved = [my_score.get(sid, their_score.get(sid)) for sid in moved + only]
                rows.append({
                    "user_row": int(user), "same_sids_other_order": not only, "n_moved": len(moved),
                    "n_only_in_one_list": len(only),
                    "max_score_diff_on_shared_sids": max((abs(my_score[s] - their_score[s]) for s in shared), default=0.0),
                    "score_spread_of_entries_involved": (max(involved) - min(involved)) if involved else 0.0,
                    "first_differing_rank": int(next(i for i in range(10) if i >= len(mine) or i >= len(theirs)
                                                     or mine[i] != theirs[i]) + 1),
                    "target_rank_here": mine.index(target) + 1 if target in mine else 0,
                    "target_rank_archive": theirs.index(target) + 1 if target in theirs else 0})
            report["beams"][str(beam)] = {
                "users_with_a_different_list": int(len(differs)), "share": float(len(differs) / N),
                "pure_reorders": int(sum(r["same_sids_other_order"] for r in rows)),
                "users_whose_target_rank_changed": int(sum(r["target_rank_here"] != r["target_rank_archive"] for r in rows)),
                "users_whose_hit10_changed": int(sum((r["target_rank_here"] > 0) != (r["target_rank_archive"] > 0) for r in rows)),
                "max_score_spread_of_entries_involved": max((r["score_spread_of_entries_involved"] for r in rows), default=0.0),
                "max_abs_score_diff_at_identical_entries": float(np.abs(scores - old_scores)[finite].max(initial=0)),
                "sid_hit10_here": float(np.mean([(codes[u] == cohort.target_sids[u]).all(1).any() for u in range(N)])),
                "sid_hit10_archive": float(np.mean([(old_codes[u] == cohort.target_sids[u]).all(1).any() for u in range(N)])),
                "rows": rows}
            summary = {k: v for k, v in report["beams"][str(beam)].items() if k != "rows"}
            print(f"beam {beam}: {json.dumps(summary)}", flush=True)
    (args.out / "archive_diff.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
