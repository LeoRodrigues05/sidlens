#!/bin/bash
# One Slurm array task of the grid pipeline: the cells [index*per, (index+1)*per)
# of a frozen task file. Submitted by `python -m sidlens.grid.pipeline run`
# (through scripts/grid/*.sh), which sets every resource flag; not for use by hand.
#   $1  task file (results_remote/registry/tasks/<spec>-<stamp>.json)
#   $2  cells per array task
# The job inherits the submitting shell's environment (sbatch's default
# --export=ALL), so the site file sourced there also applies here.

SIDLENS_REPO="${SIDLENS_REPO:?SIDLENS_REPO unset: submit through scripts/grid/sidlens-grid.sh}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
: "${ONEDIFFREC_WORK_ROOT:?ONEDIFFREC_WORK_ROOT unset (the OneDiffRec handoff work root)}"
export SIDLENS_VERIFY_PROFILE="${SIDLENS_VERIFY_PROFILE:-grid}"
sidlens_gate
exec "${PYBIN}" -m sidlens.grid.pipeline cell --task-file "${1:?task file}" \
    --index "${SLURM_ARRAY_TASK_ID:?submit as an array}" --per-task "${2:-1}"
