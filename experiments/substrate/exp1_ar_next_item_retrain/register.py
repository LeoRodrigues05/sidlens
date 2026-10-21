#!/usr/bin/env python
"""Register the retrained models that check.py accepted, so analyses can load them.

    python register.py --check $SIDLENS_WORK/derived/substrate/exp1_ar_next_item_retrain/check-<...>

Writes `manifests/registry.ar_retrain.json`, which `sidlens.models.ar` reads to
resolve names like "retrain/r20261004/rqvae_4codebook_128". Entries from an
earlier registration are kept unless the same name is registered again.

Traps this guards against
-------------------------
* **Analysing a model that failed acceptance.** Only rows that check.py marked
  `accepted` (its registry.json) are written. A failed model has no name that
  resolves, so no run can load it by accident.
* **Weights that changed after acceptance.** The hashes come from the training
  run's manifest.json and are re-computed here; a mismatch refuses. Analyses
  re-check `model.safetensors` against this file before loading
  (`ar.checkpoint_sha`).
* **Absolute paths.** Paths are stored relative to `$SIDLENS_WORK`, so the
  registry stays valid on another cluster after a bundle pull.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sidlens import paths
from sidlens.models import ar
from sidlens.provenance import hashing

FILES = ("model.safetensors", "added_tokens.json", "generation_config.json", "tokenizer.json", "config.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", required=True, type=Path, help="a check.py output directory")
    a = ap.parse_args()
    accepted = json.loads((a.check / "registry.json").read_text())
    est = pd.read_csv(a.check / "estimates.csv").set_index("variant")
    reg = json.loads(ar.RETRAIN_REGISTRY.read_text()) if ar.RETRAIN_REGISTRY.exists() else {
        "doc": "Retrained AR next-item models accepted by experiments/substrate/exp1_ar_next_item_retrain/"
               "check.py. Written by register.py there; read by sidlens.models.ar. Paths are relative to "
               "$SIDLENS_WORK.", "models": {}}
    cache = hashing.HashCache()
    for row in accepted:
        weights = Path(row["weights"])
        vdir = weights.parents[1]
        run = vdir.parent.name
        name = f"{ar.RETRAIN_PREFIX}{run}/{row['variant']}"
        man = json.loads((vdir / "manifest.json").read_text())
        sha = {}
        for f in FILES:
            got = cache.get(weights / f)
            want = man["weights_sha256"].get(f)
            if want is not None and got != want:
                raise SystemExit(f"{name}: {f} sha256 {got[:12]} != training manifest {want[:12]}")
            sha[f] = got
        if sha["model.safetensors"] != row["weights_sha256"]:
            raise SystemExit(f"{name}: weights differ from the ones check.py accepted")
        pred = vdir / "eval" / "predictions.json"
        e = est.loc[row["variant"]]
        reg["models"][name] = {
            "variant": row["variant"], "run": run,
            "weights": str(weights.relative_to(paths.WORK)), "sha256": sha,
            "predictions": str(pred.relative_to(paths.WORK)), "predictions_sha256": hashing.sha256_file(pred),
            "acceptance": {"check": str(a.check.relative_to(paths.WORK)),
                           "report_sha256": hashing.sha256_file(a.check / "report.md"),
                           "d_hr10": float(e.d_hr10), "d_ndcg10": float(e.d_ndcg10),
                           "hr10_new": float(e.hr10_new), "hr10_archived": float(e.hr10_old)}}
        print(f"registered {name}")
    reg["models"] = dict(sorted(reg["models"].items()))
    ar.RETRAIN_REGISTRY.write_text(json.dumps(reg, indent=1) + "\n")
    print(f"{len(reg['models'])} models in {ar.RETRAIN_REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
