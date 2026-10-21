#!/bin/bash
# run_best_state.sh plus the cheap set at every intermediate weight state
# (AR epochs 0.5 and 1.5; every DiffGRM snapshot): AR controlled exp3 Part A and
# exp4 Part R (test split); DiffGRM controlled exp7. Cells already finished by
# run_best_state.sh are skipped.
#   bash scripts/grid/run_full.sh [--dry-run]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SIDLENS_GRID_ENV:-${HOME}/.config/sidlens/grid.env}"
[[ -f "${SIDLENS_RESULTS}/registry/registry.json" ]] || bash "${HERE}/sidlens-grid.sh" register
bash "${HERE}/sidlens-grid.sh" plan --set full
bash "${HERE}/sidlens-grid.sh" run --set full "$@"
