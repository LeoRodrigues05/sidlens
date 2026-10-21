#!/usr/bin/env python
"""Why RQ-VAE's last-digit reassignment leaves some buckets of distinct items.

Structure exp3 found three old RQ-VAE buckets of distinct items the loop could
not split (26, 10 and 19 items) and could only infer that "the encoder put them
at (nearly) the same latent point", because the weights were not saved. The v3
checkpoints are published, so this measures it. Exploratory (post hoc, added
2026-10-20 after reading run 327442's bucket table).

For every published RQ-VAE bucket of >= 3 items with more than one distinct
input (``summary/b_rqvae_buckets.csv``), per catalog:

    latent_pair_bucket / _catalogue   median pairwise distance of encoder outputs,
                                      in the bucket and for 600 random items
    input_pair_bucket / _catalogue    the same in the 2,560-d input embedding
    dead_layer                        the first hidden ReLU layer at which every
                                      unit is zero for every bucket member (then
                                      the encoder output is the same constant for
                                      all of them), or "" if none
    active_units_min                  fewest active units in any hidden layer,
                                      over the bucket's items

Writes ``<run>/summary/rqvae_bucket_latent.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sidlens.data import codebooks_v3 as V
from sidlens.data import embeddings as E

SEED = 20261020


def med_pair(A) -> float:
    A = np.asarray(A, np.float64)
    s = (A * A).sum(1)
    d = np.sqrt(np.maximum(s[:, None] + s[None] - 2 * A @ A.T, 0))
    return float(np.median(d[np.triu_indices(len(A), 1)]))


def hidden_activity(sd: dict, x: np.ndarray) -> list[np.ndarray]:
    """(n, units) boolean 'unit active' per hidden ReLU layer of the encoder."""
    idx = V._linear_indices(sd, "encoder.mlp_layers")
    h = torch.from_numpy(np.array(x, dtype=np.float32))
    out = []
    with torch.no_grad():
        for j, i in enumerate(idx[:-1]):
            h = torch.relu(h @ sd[f"encoder.mlp_layers.{i}.weight"].T + sd[f"encoder.mlp_layers.{i}.bias"])
            out.append((h > 0).numpy())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    S = args.run / "summary"
    rb = pd.read_csv(S / "b_rqvae_buckets.csv")
    rb = rb[rb.n_distinct_inputs > 1]
    rng = np.random.default_rng(SEED)
    rows = []
    for cat, g in rb.groupby("catalog"):
        X = E.load(V.category(cat), np.float32)
        samp = rng.choice(len(X), 600, replace=False)
        for t in g.table.unique():
            codes = V.codes(cat, t)
            _, ck = V.rqvae_checkpoint(cat, t)
            sd = ck["state_dict"]
            z = V.rqvae_encode(sd, X)
            for sidtxt in g[g.table == t].sid:
                sid = np.array([int(x) for x in sidtxt.strip("()").split(")(")])
                mem = np.flatnonzero((codes == sid).all(1))
                act = hidden_activity(sd, X[mem])
                dead = [k for k, a in enumerate(act) if not a.any()]
                rows.append({"catalog": cat, "table": t, "sid": sidtxt, "size": len(mem),
                             "latent_pair_bucket": med_pair(z[mem]), "latent_pair_catalogue": med_pair(z[samp]),
                             "input_pair_bucket": med_pair(X[mem]), "input_pair_catalogue": med_pair(X[samp]),
                             "latent_max_abs_spread": float(np.abs(z[mem] - z[mem][0]).max()),
                             "dead_layer": (f"hidden layer {dead[0] + 1} of {len(act)} ({act[dead[0]].shape[1]} units)"
                                            if dead else ""),
                             "active_units_min": int(min(a.sum(1).min() for a in act))})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(S / "rqvae_bucket_latent.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
