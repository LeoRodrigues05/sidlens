#!/bin/bash
#SBATCH --job-name=sidlens-matched-beam
#SBATCH --partition=ws-ia
#SBATCH --nodelist=ws-l3-020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-17%1
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
source "${SIDLENS_REPO}/scripts/_preamble.sh"
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

RUN_GROUP="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
OUT="${SIDLENS_WORK}/derived/controlled/exp1_matched_beam/${RUN_GROUP}/cell-$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
mkdir -p "${OUT}"
printf 'running job=%s task=%s started=%s\n' "${RUN_GROUP}" \
    "${SLURM_ARRAY_TASK_ID}" "$(date -Is)" > "${OUT}/status.txt"
finish_run() {
    local rc=$?
    local state=failed
    if [[ ${rc} -eq 0 ]]; then
        state=complete
        find "${OUT}" -maxdepth 1 -type f ! -name output.sha256 \
            ! -name status.txt -print0 | sort -z | xargs -0 sha256sum \
            > "${OUT}/output.sha256"
    fi
    printf '%s job=%s task=%s exit=%s finished=%s\n' "${state}" \
        "${RUN_GROUP}" "${SLURM_ARRAY_TASK_ID}" "${rc}" "$(date -Is)" \
        > "${OUT}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT
sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    find src/sidlens vendor/diffgrm vendor/diffgrm_new vendor/onediffrec \
        -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sbatch' \) -print0
    find experiments/controlled/exp1_matched_beam -type f \
        \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/_preamble.sh scripts/matched_beam.sbatch pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
printf '%s\n' "$@" > "${OUT}/arguments.txt"
{
    date -Is
    hostname
    uname -a
    "${PYBIN}" -VV
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    printf 'OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s\n' "${OMP_NUM_THREADS}" \
        "${MKL_NUM_THREADS}" "${OPENBLAS_NUM_THREADS}" "${NUMEXPR_NUM_THREADS}"
} > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

"${PYBIN}" -u experiments/controlled/exp1_matched_beam/run.py \
    --cell-index "${SLURM_ARRAY_TASK_ID}" --out "${OUT}" "$@"
