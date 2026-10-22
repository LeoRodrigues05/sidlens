#!/bin/bash
# Optional case study (docs/plans/GRID_PAPER_SET.md): every experiment, in full, on
# the native 3x256 and 4x256 maps of each quantizer in both catalogs, plus the cheap
# set at their intermediate weight states. Queues ONE Slurm job (it requeues itself
# if its time runs out); cells the paper set already finished are skipped.
#   bash scripts/grid/run_case_study.sh [--dry-run]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SIDLENS_GRID_ENV:-${HOME}/.config/sidlens/grid.env}"
[[ -f "${SIDLENS_RESULTS}/registry/registry.json" ]] || bash "${HERE}/sidlens-grid.sh" register
bash "${HERE}/sidlens-grid.sh" plan --set case
bash "${HERE}/sidlens-grid.sh" run --set case "$@"
