"""Canonical locations. Code lives on /home, bulk data on /l/users.

Every path in the project resolves through here so that a relocation is a
one-file change and so that no module ever hardcodes a source-tree path.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- the sidlens repo (code) -------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "vendor"
MANIFESTS = REPO / "manifests"

# --- sidlens bulk storage ----------------------------------------------------
WORK = Path(os.environ.get("SIDLENS_WORK", "/l/users/leo.rodrigues/sidlens"))
FROZEN = WORK / "frozen"
DERIVED = WORK / "derived"
RUNS = WORK / "runs"
CACHE = WORK / "cache"

# --- upstream, READ-ONLY -----------------------------------------------------
# Never written to. `freeze` reads from here exactly once; everything else
# reads from FROZEN.
UPSTREAM_REPO = Path(
    os.environ.get("ONEDIFFREC_REPO", "/home/leo.rodrigues/onediffrec/OneDiffRec")
)
UPSTREAM_WORK = Path(
    os.environ.get("ONEDIFFREC_WORK", "/l/users/leo.rodrigues/onediffrec")
)

# --- frozen substrate layout -------------------------------------------------
FROZEN_SIDS = FROZEN / "sids"
FROZEN_DATA = FROZEN / "data"
FROZEN_CKPT = FROZEN / "ckpt"
FROZEN_RESULTS = FROZEN / "results"
FROZEN_BASE_MODELS = FROZEN / "base_models"

# The live lineage. The legacy `data/Amazon` tree (3686 Industrial / 3459
# Office_Products items) is a DIFFERENT item set and is deliberately not frozen.
CATEGORY_ITEMS = {"Industrial_and_Scientific": 3105, "Office": 17696}
