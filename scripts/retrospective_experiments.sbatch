#!/bin/bash
#SBATCH --job-name=sidlens-retro
#SBATCH --partition=ws-ia
#SBATCH --nodelist=ws-l3-020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=1-4%4
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# Four retained-output analyses recovered from the 2026-09-06 local planning
# chat.  These jobs are CPU-only: requesting the node's sole GPU would reserve
# scarce hardware without changing any calculation.  Experiments 5 onward use
# model inference and get their own --gres=gpu:1 wrappers.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/_preamble.sh" >&2
    exit 1
fi

cd "${SIDLENS_REPO}"
sidlens_gate

case "${SLURM_ARRAY_TASK_ID:?submit this wrapper as an array}" in
    1)
        EXP_NAME="exp1_reveal_order"
        ENTRY="experiments/retrospective/exp1_reveal_order/run.py"
        ;;
    2)
        EXP_NAME="exp2_prefix"
        ENTRY="experiments/retrospective/exp2_prefix/run.py"
        ;;
    3)
        EXP_NAME="exp3_collisions"
        ENTRY="experiments/retrospective/exp3_collisions/run.py"
        ;;
    4)
        EXP_NAME="exp4_next_two"
        ENTRY="experiments/retrospective/exp4_next_two/run.py"
        ;;
    *)
        echo "FATAL: unknown array task ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${ENTRY}" ]]; then
    echo "FATAL: missing entry point ${ENTRY}" >&2
    exit 1
fi

OUT="${SIDLENS_WORK}/derived/retrospective/${EXP_NAME}"
mkdir -p "${OUT}"
sidlens_provenance_stamp "${OUT}/.provenance"
printf 'running job=%s task=%s started=%s\n' \
    "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" "${SLURM_ARRAY_TASK_ID}" \
    "$(date -Is)" > "${OUT}/status.txt"

finish_run() {
    local rc=$?
    local state="failed"
    if [[ ${rc} -eq 0 ]]; then
        state="complete"
    fi
    printf '%s job=%s task=%s exit=%s finished=%s\n' \
        "${state}" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" \
        "${SLURM_ARRAY_TASK_ID}" "${rc}" "$(date -Is)" > "${OUT}/status.txt"
    if [[ ${rc} -eq 0 ]]; then
        find "${OUT}" -maxdepth 1 -type f \
            ! -name 'output.sha256' ! -name 'status.txt' -print0 \
            | sort -z | xargs -0 sha256sum > "${OUT}/output.sha256"
    fi
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

# Thread counts are bounded by the allocation even though most work is parsing
# and bootstrap aggregation.  Hash all sprint code so a dirty-tree run is still
# tied to the exact bytes that produced it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SOURCE_LIST="${OUT}/source-files.nul"
{
    find experiments/retrospective -type f \
        \( -name '*.py' -o -name '*.md' -o -name '*.sbatch' \) -print0
    find src/sidlens vendor/onediffrec vendor/diffgrm vendor/diffgrm_new \
        -type f -name '*.py' -print0
    find tests -maxdepth 1 -type f -name 'test_retrospective*.py' -print0
    printf '%s\0' scripts/_preamble.sh scripts/retrospective_experiments.sbatch
} | sort -zu > "${SOURCE_LIST}"
xargs -0 sha256sum < "${SOURCE_LIST}" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${SOURCE_LIST}"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{
    echo "captured=$(date -Is)"
    echo "node=$(hostname)"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
    echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
    echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
    uname -a
    "${PYBIN}" -VV
    "${PYBIN}" -c \
        'import numpy, scipy; print(f"numpy={numpy.__version__}"); print(f"scipy={scipy.__version__}")'
} > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

echo "[retro-${SLURM_ARRAY_TASK_ID}] ${EXP_NAME} starting $(date -Is) on $(hostname)"
"${PYBIN}" -u "${ENTRY}" --out "${OUT}"
echo "[retro-${SLURM_ARRAY_TASK_ID}] ${EXP_NAME} done $(date -Is)"
