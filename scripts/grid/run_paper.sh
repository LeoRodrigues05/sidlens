#!/bin/bash
# The paper set (docs/plans/GRID_PAPER_SET.md): best state only, every map for the
# experiments whose claims the paper makes on all configurations, the core maps for the
# costly ones. Registers the grid first if no registry exists, prints the plan, then
# queues ONE Slurm job that runs every cell (it requeues itself if its time runs out;
# finished cells are skipped, so rerunning is safe).
#   bash scripts/grid/run_paper.sh [--dry-run]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SIDLENS_GRID_ENV:-${HOME}/.config/sidlens/grid.env}"
[[ -f "${SIDLENS_RESULTS}/registry/registry.json" ]] || bash "${HERE}/sidlens-grid.sh" register
bash "${HERE}/sidlens-grid.sh" plan --set paper
bash "${HERE}/sidlens-grid.sh" run --set paper "$@"
