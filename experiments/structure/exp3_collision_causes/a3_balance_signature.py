#!/usr/bin/env python
"""Amendment A3 (post hoc): is the frozen RQ-KMeans 5x512 table the output of
upstream's ``--uniform`` balancing?

The declared H1b rule expected exact capacities (6 or 7 items per code). That
was a misreading of ``rqkmeans_faiss.sinkhorn_balance_level``: when none of an
item's 32 best codes has capacity left, it falls back to
``argmin(remaining)``, which is the code already most over capacity, so the
overflow piles into one "dump" code per level. This script compares
fingerprints of that procedure between the frozen 5x512 table, the native
5x512 refit (seed 1234; its depths 3-4 equal the archived nested tables), and
the faithful reimplementation applied to that refit (Part D2):

* per-level count range, the share in the largest code, and the number of
  codes exactly at capacity
* the repeat rate P(digit l = digit l-1) (shared-seed signature)
* identical-input pairs split
* first-digit ARI between tables, and 10-NN first-digit agreement

Run at 96 threads (the native refit only reproduces there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402


def fingerprint(name, codes, K, ident_inv, knn10):
    n, D = codes.shape
    full = R.keys_of(codes, K)
    pairs = [(a, b) for a in range(n) for b in np.flatnonzero(ident_inv == ident_inv[a]) if a < b]
    rows = []
    for l in range(D):
        cnt = np.bincount(codes[:, l], minlength=K)
        rows.append({"table": name, "level": l, "n_used": int((cnt > 0).sum()),
                     "count_min": int(cnt.min()), "count_max": int(cnt.max()),
                     "max_code_share": float(cnt.max() / n),
                     "n_codes_at_capacity": int(((cnt == n // K) | (cnt == -(-n // K))).sum()),
                     "repeat_rate": float((codes[:, l] == codes[:, l - 1]).mean()) if l else np.nan})
    summary = {"table": name, **R.rates(codes, K),
               "identical_pairs_split": int(sum(full[a] != full[b] for a, b in pairs)),
               "n_identical_pairs": len(pairs),
               "knn10_agree_depth1": R.knn_agreement(knn10, codes[:, 0])}
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=96)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    faiss.omp_set_num_threads(args.threads)
    src = Path(__file__).read_bytes() + Path(R.__file__).read_bytes()
    raw, item2id, titles, brands, _ = R.load_inputs()
    X32 = np.ascontiguousarray(raw.astype(np.float32))
    X = raw.astype(np.float64)
    D2 = R.sqdist(X, X)
    np.fill_diagonal(D2, np.inf)
    knn10 = np.argsort(D2, axis=1, kind="stable")[:, :10]
    _, ident_inv = np.unique(raw, axis=0, return_inverse=True)
    ident_inv = ident_inv.ravel()

    K = 512
    frozen = R.load_codes("rqkmeans_5codebook_512", item2id, len(raw))
    native, cbs, _ = R.fit_rq(X32, 5, K, lambda l: R.NATIVE_SEED)
    for D in (3, 4):
        arch = R.load_codes(f"rqkmeans_{D}codebook_{K}", item2id, len(raw))
        if not (native[:, :D] == arch).all():
            raise RuntimeError(f"native refit does not reproduce {D}x{K}; threads?")
    balanced = R.uniform_mapping(X32, native, cbs, K)

    lvl, summ = [], []
    for name, c in (("frozen_5x512", frozen), ("native_refit_5x512", native),
                    ("uniform_of_native_5x512", balanced)):
        r, s = fingerprint(name, c, K, ident_inv, knn10)
        lvl += r
        summ.append(s)
    ari = {f"{a}|{b}": float(adjusted_rand_score(x[:, 0], y[:, 0]))
           for (a, x), (b, y) in [(("frozen", frozen), ("native", native)),
                                  (("frozen", frozen), ("uniform", balanced)),
                                  (("native", native), ("uniform", balanced))]}
    pd.DataFrame(lvl).to_csv(out / "a3_levels.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "a3_summary.csv", index=False)
    (out / "a3_result.json").write_text(json.dumps({
        "ari_first_digit": ari, "summary": summ,
        "source_sha256": hashlib.sha256(src).hexdigest(),
        "faiss_threads": faiss.omp_get_max_threads()}, indent=2, default=str))
    print(pd.DataFrame(lvl).round(4).to_string())
    print(pd.DataFrame(summ).round(4).to_string())
    print(json.dumps(ari, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
