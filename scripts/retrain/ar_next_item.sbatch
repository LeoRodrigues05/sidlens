#!/bin/bash
#SBATCH --job-name=sidlens-ar-retrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# Retrain the next-item Qwen-AR models with the recorded sweep recipe
# (experiments/retrain/ar_next_item). Full mode: one array task per group of
# configurations in configs.tsv (4 GPUs, as in the sweep). Pilot: one task on
# one GPU (original weights through the eval pipeline + a smoke training).
# SITE-NEUTRAL: pass partition/QoS/excludes at submission (see site.env).
# Submit from the repo root:
#   pilot: sbatch --array=0 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=03:00:00 \
#            --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/retrain/ar_next_item.sbatch --mode pilot
#   full:  sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/retrain/ar_next_item.sbatch --mode full
#   rerun: sbatch --array=<task> ... --mode full --root-group <group> --resume
# Output: $SIDLENS_WORK/runs/ar_next_item_retrain/<array-id>/ (pilot-<array-id>/
# for the pilot). One directory per configuration; never reused except by --resume.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
ONEDIFFREC_ENV="${ONEDIFFREC_ENV:-/home/leo.rodrigues/onediffrec/OneDiffRec/.conda}"
export ONEDIFFREC_ENV
[[ -x "${ONEDIFFREC_ENV}/bin/torchrun" ]] || { echo "FATAL: no torchrun in ${ONEDIFFREC_ENV}" >&2; exit 1; }

EXP=experiments/retrain/ar_next_item
MODE=""
GROUP_OVERRIDE=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; ARGS+=(--mode "$2"); shift 2 ;;
        --root-group) GROUP_OVERRIDE="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
[[ "${MODE}" == "pilot" || "${MODE}" == "full" ]] || { echo "FATAL: --mode pilot|full" >&2; exit 2; }

RUN_GROUP="${GROUP_OVERRIDE:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?run under sbatch}}}"
[[ "${MODE}" == "pilot" && -z "${GROUP_OVERRIDE}" ]] && RUN_GROUP="pilot-${RUN_GROUP}"
TASK="${SLURM_ARRAY_TASK_ID:?submit as an array}"
ROOT="${SIDLENS_WORK}/runs/ar_next_item_retrain/${RUN_GROUP}"
OUT="${ROOT}/task-$(printf '%02d' "${TASK}")-${SLURM_JOB_ID}"
mkdir -p "${ROOT}"
mkdir "${OUT}"
printf 'running job=%s task=%s started=%s\n' "${RUN_GROUP}" "${TASK}" "$(date -Is)" > "${OUT}/status.txt"
# scancel sends SIGTERM; without this trap the EXIT trap saw rc=0 and wrote
# "complete" for a cancelled job (seen on pilot 294721).
trap 'exit 143' TERM
finish_run() {
    local rc=$?
    local state=failed
    [[ ${rc} -eq 0 ]] && state=complete
    [[ ${rc} -eq 143 ]] && state=cancelled
    printf '%s job=%s task=%s exit=%s finished=%s\n' "${state}" "${RUN_GROUP}" \
        "${TASK}" "${rc}" "$(date -Is)" > "${OUT}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    git ls-files -z src/sidlens vendor/onediffrec
    find "${EXP}" -type f \( -name '*.py' -o -name '*.md' -o -name '*.tsv' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/retrain/ar_next_item.sbatch pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
cp "${EXP}/protocol.md" "${OUT}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{
    date -Is; hostname; uname -a; "${PYBIN}" -VV; "${ONEDIFFREC_ENV}/bin/python" -VV
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null \
        || echo "no GPU"
} > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.sidlens.freeze.txt"
"${ONEDIFFREC_ENV}/bin/python" -m pip freeze > "${OUT}/requirements.onediffrec-conda.freeze.txt"

"${PYBIN}" -u "${EXP}/run.py" --task "${TASK}" --root "${ROOT}" "${ARGS[@]}" 2>&1 | tee "${OUT}/run.log"
exit "${PIPESTATUS[0]}"
