"""Checks and hit counts shared by experiments that re-run the DiffGRM beam search.

Traps, each with the guard that closes it
-----------------------------------------
1. **A baseline that is not the archived one.** A new condition only means
   something against a decode that reproduces controlled exp1's archive.
   `archive_check` compares codes, counts and scores, and identity arrays
   (users, targets) before anything else.
2. **Ties on other hardware read as failures, or failures read as ties.** On
   another GPU model, exactly tied entries can come out in another order or
   swap at the list boundary (representation/exp4's V6 diagnosis). The
   tolerance is narrow and declared in the protocols that use it: a user may
   differ from the archive only if both lists have the same count and their
   sorted scores agree within ``tie_atol``, and at most ``max_tie_share`` of
   users may differ at all. Anything else raises.
3. **SID hits passed off as item hits.** `item_hits` expands each predicted
   SID to every item in its bucket and returns the lower and upper bound of
   item-level HR@k, never one number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sidlens.analysis.collisions import item_rank_bounds
from sidlens.provenance.hashing import sha256_file


class ArchiveMismatch(RuntimeError):
    pass


def archive_check(archive: Path, entry: dict, cohort, n: int, decoded: dict, *, tie_atol: float = 1e-5,
                  max_tie_share: float = 0.001, score_atol: float = 1e-4) -> dict:
    """Compare plain confidence decodes with controlled exp1's archive for one cell.

    ``decoded``: beam -> {"codes": [n, 10, depth], "counts": [n], "scores": [n, 10]} for the
    first ``n`` cohort users. Raises `ArchiveMismatch` on any difference outside trap 2.
    """
    previous = json.loads((archive / "result.json").read_text())
    if previous["checkpoint_sha256"] != entry["ckpt_sha256"]:
        raise ArchiveMismatch("archive checkpoint differs")
    if previous["cohort_sha256"] != cohort.cohort_sha256 or \
            sorted(previous["input_sha256"].values()) != sorted(cohort.input_sha256.values()):
        raise ArchiveMismatch("archive cohort/input hashes differ")
    checks = []
    with np.load(archive / "predictions.npz", allow_pickle=False) as old:
        for name, mine in (("users", np.asarray(cohort.users[:n])), ("user_ids", cohort.user_ids[:n]),
                           ("target_item_ids", cohort.target_item_ids[:n]), ("targets", cohort.target_sids[:n])):
            if not np.array_equal(mine, old[name][:n]):
                raise ArchiveMismatch(f"archive {name} identity differs")
        axes = list(zip(old["condition_beams"].tolist(), old["condition_policies"].tolist()))
        for beam, mine in decoded.items():
            if (beam, "confidence") not in axes:
                continue
            j = axes.index((beam, "confidence"))
            codes, counts, scores = old["codes"][j, :n], old["counts"][j, :n], old["scores"][j, :n]
            differ = ~((mine["codes"] == codes).all(axis=(1, 2)) & (mine["counts"] == counts))
            ties, bad = [], []
            for u in np.flatnonzero(differ):
                a, b = np.sort(mine["scores"][u][np.isfinite(mine["scores"][u])]), np.sort(scores[u][np.isfinite(scores[u])])
                if mine["counts"][u] == counts[u] and len(a) == len(b) and np.allclose(a, b, atol=tie_atol, rtol=0):
                    ties.append(int(u))
                else:
                    bad.append(int(u))
            same = ~differ
            fin = np.isfinite(scores[same])
            if not np.array_equal(np.isfinite(mine["scores"][same]), fin) or not np.allclose(
                    mine["scores"][same][fin], scores[same][fin], atol=score_atol, rtol=score_atol):
                raise ArchiveMismatch(f"beam {beam}: finite scores of identical lists differ")
            if bad or len(ties) > max_tie_share * n:
                raise ArchiveMismatch(f"beam {beam}: {len(bad)} users differ beyond ties, {len(ties)} tie-only "
                                      f"(allowed {max_tie_share * n:.1f})")
            checks.append({"beam": beam, "users_identical": int(same.sum()), "users_tie_only": ties,
                           "score_max_absolute_difference": float(np.max(np.abs(mine["scores"][same][fin] - scores[same][fin]), initial=0))})
    if not checks:
        raise ArchiveMismatch("no decoded beam is in the archive")
    return {"status": "passed", "users": n, "beams": checks, "archive": str(archive),
            "archive_sha256": {name: sha256_file(archive / name) for name in ("result.json", "predictions.npz")}}


def sid_rank(codes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """[n] 1-based rank of the target SID in each top list [n, k, depth]; 0 when absent."""
    match = (codes == targets[:, None, :]).all(-1)
    return np.where(match.any(1), match.argmax(1) + 1, 0)


def item_hits(codes: np.ndarray, counts: np.ndarray, target_items: np.ndarray, target_sids: np.ndarray,
              sid_buckets: dict, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Item-level HR@k bounds per user: (hit if the target is LAST in its bucket, hit if FIRST)."""
    lo, hi = np.zeros(len(codes), dtype=bool), np.zeros(len(codes), dtype=bool)
    for u in range(len(codes)):
        preds = [tuple(int(c) for c in codes[u, r]) for r in range(int(counts[u]))]
        b = item_rank_bounds(preds, int(target_items[u]), sid_buckets, target_sid=tuple(int(c) for c in target_sids[u]))
        lo[u] = b.worst is not None and b.worst <= k
        hi[u] = b.best is not None and b.best <= k
    return lo, hi
