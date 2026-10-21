#!/bin/bash
# Every experiment on every grid configuration, at each run's best weight state.
# Registers the grid first if no registry exists, prints the plan, then submits
# every stage as Slurm arrays (finished cells are skipped; rerunning is safe).
#   bash scripts/grid/run_best_state.sh [--dry-run]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SIDLENS_GRID_ENV:-${HOME}/.config/sidlens/grid.env}"
[[ -f "${SIDLENS_RESULTS}/registry/registry.json" ]] || bash "${HERE}/sidlens-grid.sh" register
bash "${HERE}/sidlens-grid.sh" plan --set best
bash "${HERE}/sidlens-grid.sh" run --set best "$@"
