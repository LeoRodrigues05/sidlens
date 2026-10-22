#!/bin/bash
# The whole grid set as ONE Slurm job: `pipeline batch` runs every unfinished cell
# of the set on the GPUs and cores this job was given (one cell per GPU at a time,
# CPU cells beside them) and queues its successor if its time runs out. Submitted
# by `python -m sidlens.grid.pipeline run` (scripts/grid/run_*.sh), which sets the
# resource flags from grid.env (BATCH_GPUS, BATCH_CPUS, BATCH_MEM, BATCH_TIME); not
# for use by hand.
#   $1  set (paper | case | best | full)
#   $2  round (how many jobs of this set ran before this one)
#   $3… stages (optional)
# The job inherits the submitting shell's environment (sbatch's default
# --export=ALL), so the site file sourced there also applies here.

SIDLENS_REPO="${SIDLENS_REPO:?SIDLENS_REPO unset: submit through scripts/grid/sidlens-grid.sh}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export PYTHONUNBUFFERED=1
: "${ONEDIFFREC_WORK_ROOT:?ONEDIFFREC_WORK_ROOT unset (the OneDiffRec handoff work root)}"
export SIDLENS_VERIFY_PROFILE="${SIDLENS_VERIFY_PROFILE:-grid}"
sidlens_gate
set_="${1:?set}"; round="${2:-0}"; shift 2 || true
if [[ ! -f "${SIDLENS_RESULTS}/registry/registry.json" ]]; then
    "${PYBIN}" -m sidlens.grid.pipeline register || exit 1
fi
stages=()
[[ $# -gt 0 ]] && stages=(--stage "$@")
exec "${PYBIN}" -m sidlens.grid.pipeline batch --set "${set_}" --round "${round}" "${stages[@]}"
