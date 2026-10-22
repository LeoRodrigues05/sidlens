#!/bin/bash
# sidlens on the OneDiffRec grid: one entry point for the coauthor's cluster.
#
#   bash scripts/grid/sidlens-grid.sh setup --onediffrec-work <WORK_ROOT> [--sidlens-work DIR] [--python PY]
#   bash scripts/grid/sidlens-grid.sh register      # runs and weight states found; anything missing or failed
#   bash scripts/grid/sidlens-grid.sh pilot         # one short cell per experiment, timed (run on a GPU node)
#   bash scripts/grid/sidlens-grid.sh plan  [--set best|full|paper|case|quick]
#   bash scripts/grid/sidlens-grid.sh run   [--set best|full|paper|case|quick] [--stage ...] [--dry-run]   # ONE job
#   bash scripts/grid/sidlens-grid.sh status [--set best|full|paper|case|quick]
#   bash scripts/grid/sidlens-grid.sh summarize
#   bash scripts/grid/sidlens-grid.sh export        # packs results_remote/ with a hash manifest
#
# Settings live in ~/.config/sidlens/grid.env (written by setup; edit the Slurm
# lines there): SLURM_PARTITION, SLURM_ACCOUNT, SLURM_QOS, SLURM_EXCLUDE,
# CPU_PARTITION, CPU_QOS, GPU_TYPE; the one job: BATCH_NODES, BATCH_GPUS (per node),
# BATCH_CPUS, BATCH_MEM, BATCH_TIME, BATCH_CPU_WORKERS, BATCH_MARGIN_MIN (run --arrays only: GPU_GRES, MAX_GPUS, MAX_CPUS, CELLS_PER_TASK).
# Runbook: docs/clusters/GRID_RUNBOOK.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
GRID_ENV="${SIDLENS_GRID_ENV:-${HOME}/.config/sidlens/grid.env}"

cmd="${1:-}"; shift || true
if [[ "${cmd}" == "setup" ]]; then
    WORK_ROOT=""; SL_WORK=""; PY=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --onediffrec-work) WORK_ROOT="$(cd "$2" && pwd)"; shift 2 ;;
            --sidlens-work) SL_WORK="$2"; shift 2 ;;
            --python) PY="$2"; shift 2 ;;
            *) echo "unknown setup option $1" >&2; exit 2 ;;
        esac
    done
    [[ -n "${WORK_ROOT}" ]] || { echo "setup needs --onediffrec-work <WORK_ROOT>" >&2; exit 2; }
    SL_WORK="${SL_WORK:-$(dirname "${WORK_ROOT}")/sidlens_work}"
    mkdir -p "${SL_WORK}"
    if [[ -z "${PY}" ]]; then                     # the interpreter OneDiffRec's own runs recorded
        cell="$(find "${WORK_ROOT}" -maxdepth 9 -name cell.json -path '*seed-*' -print -quit)"
        [[ -n "${cell}" ]] || { echo "no cell.json under ${WORK_ROOT}; pass --python" >&2; exit 1; }
        PY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["python"])' "${cell}")"
    fi
    echo "[setup] OneDiffRec python: ${PY}"
    if [[ ! -x "${SL_WORK}/venv/bin/python" ]]; then
        "${PY}" -m venv --system-site-packages "${SL_WORK}/venv"
    fi
    VPY="${SL_WORK}/venv/bin/python"
    # sidlens on top of OneDiffRec's torch/transformers (the ones that produced the saved lists);
    # pip leaves already-satisfied packages alone.
    "${VPY}" -m pip install -q -e "${REPO}[test,hub]"
    "${VPY}" -c 'import torch, transformers; print("[setup] torch", torch.__version__, "transformers", transformers.__version__)'
    mkdir -p "$(dirname "${GRID_ENV}")"
    [[ -f "${GRID_ENV}" ]] && cp "${GRID_ENV}" "${GRID_ENV}.bak-$(date +%Y%m%dT%H%M%S)"
    cat > "${GRID_ENV}" <<ENV
# sidlens on the OneDiffRec grid (written by scripts/grid/sidlens-grid.sh setup, $(date -Is)).
export SIDLENS_REPO=${REPO}
export SIDLENS_WORK=${SL_WORK}
export SIDLENS_PYTHON=${VPY}
export SIDLENS_VERIFY_PROFILE=grid
export SIDLENS_SITE_ENV=${GRID_ENV}
export ONEDIFFREC_WORK_ROOT=${WORK_ROOT}
export SIDLENS_RESULTS=${SL_WORK}/results_remote
export SIDLENS_SCRATCH=${SL_WORK}/scratch
# Slurm (the names OneDiffRec's scripts use; fill in for this cluster):
export SLURM_PARTITION=\${SLURM_PARTITION:-}
export SLURM_ACCOUNT=\${SLURM_ACCOUNT:-}
export SLURM_QOS=\${SLURM_QOS:-}
export SLURM_EXCLUDE=\${SLURM_EXCLUDE:-}
export CPU_PARTITION=\${CPU_PARTITION:-}
export CPU_QOS=\${CPU_QOS:-}
export GPU_GRES=\${GPU_GRES:-gpu:1}
export MAX_GPUS=\${MAX_GPUS:-16}
export MAX_CPUS=\${MAX_CPUS:-32}
export CELLS_PER_TASK=\${CELLS_PER_TASK:-1}
# One job (what run_*.sh queue): BATCH_NODES nodes, each with these resources; it
# requeues itself. BATCH_MEM=0 asks for all of a node's memory; set a size if the
# cluster refuses it.
export BATCH_NODES=\${BATCH_NODES:-1}
export BATCH_GPUS=\${BATCH_GPUS:-8}
export GPU_TYPE=\${GPU_TYPE:-}
export BATCH_CPUS=\${BATCH_CPUS:-64}
export BATCH_MEM=\${BATCH_MEM:-0}
export BATCH_TIME=\${BATCH_TIME:-2-00:00:00}
export BATCH_CPU_WORKERS=\${BATCH_CPU_WORKERS:-4}
export BATCH_MARGIN_MIN=\${BATCH_MARGIN_MIN:-15}
ENV
    echo "[setup] wrote ${GRID_ENV}"
    source "${GRID_ENV}"
    for need in external/labels derived/grid_inputs/near_duplicates; do
        [[ -e "${SIDLENS_WORK}/${need}" ]] || echo "[setup] MISSING ${SIDLENS_WORK}/${need}: unpack the inputs tarball (runbook step 2)"
    done
    "${VPY}" -m sidlens.cli verify --strict --quick --profile grid
    "${VPY}" -m pytest -q "${REPO}/tests" -x -q -p no:cacheprovider || { echo "[setup] tests FAILED" >&2; exit 1; }
    echo "[setup] done. Next: bash scripts/grid/sidlens-grid.sh register"
    exit 0
fi

[[ -f "${GRID_ENV}" ]] || { echo "no ${GRID_ENV}; run setup first" >&2; exit 1; }
# shellcheck source=/dev/null
source "${GRID_ENV}"
cd "${SIDLENS_REPO}"
PY="${SIDLENS_PYTHON}"
case "${cmd}" in
    register|plan|run|status|summarize|export)
        exec "${PY}" -m sidlens.grid.pipeline "${cmd}" "$@" ;;
    pilot)                                        # needs a GPU: inside an allocation run here, else submit one job
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            exec "${PY}" -m sidlens.grid.pipeline pilot "$@"
        fi
        flags=(--gres="${GPU_GRES:-gpu:1}" --cpus-per-task=8 --mem=128G --time=12:00:00 --job-name=sl-pilot
               --output="${SIDLENS_RESULTS}/provenance/slurm/pilot-%j.out")
        [[ -n "${SLURM_PARTITION:-}" ]] && flags+=(--partition="${SLURM_PARTITION}")
        [[ -n "${SLURM_ACCOUNT:-}" ]] && flags+=(--account="${SLURM_ACCOUNT}")
        [[ -n "${SLURM_QOS:-}" ]] && flags+=(--qos="${SLURM_QOS}")
        [[ -n "${SLURM_EXCLUDE:-}" ]] && flags+=(--exclude="${SLURM_EXCLUDE}")
        mkdir -p "${SIDLENS_RESULTS}/provenance/slurm"
        sbatch "${flags[@]}" --wrap "source '${GRID_ENV}' && cd '${SIDLENS_REPO}' && source scripts/common/_preamble.sh \
            && sidlens_gate && '${PY}' -m sidlens.grid.pipeline pilot $*"
        echo "pilot submitted; log in ${SIDLENS_RESULTS}/provenance/slurm/, timings in results_remote/pilot/" ;;
    *) sed -n '2,19p' "$0"; exit 2 ;;
esac
