#!/bin/bash
# The quick set (docs/plans/GRID_PAPER_SET.md): the paper set cut to one test per claim,
# sized to 8-14 hours on 16 H200s (BATCH_NODES=2 with 8 GPUs each). Registers the grid
# first if no registry exists, prints the plan, then queues ONE Slurm job (it requeues
# itself if its time runs out). run_paper.sh afterwards continues from where it stops.
#   bash scripts/grid/run_quick.sh [--dry-run]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SIDLENS_GRID_ENV:-${HOME}/.config/sidlens/grid.env}"
[[ -f "${SIDLENS_RESULTS}/registry/registry.json" ]] || bash "${HERE}/sidlens-grid.sh" register
bash "${HERE}/sidlens-grid.sh" plan --set quick
bash "${HERE}/sidlens-grid.sh" run --set quick "$@"
