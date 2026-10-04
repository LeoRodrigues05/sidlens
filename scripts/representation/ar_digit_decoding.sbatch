#!/bin/bash
#SBATCH --job-name=sidlens-ar-decoding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# Layer-wise decoding of SID digits from captured AR activations: logit lens + probes
# (experiments/representation/exp1_ar_digit_decoding). One array task per capture:
#   0 = derived/ar_capture/281053 (next-item_best), 1 = derived/ar_capture/281054 (oneoff_rqvae4cb128).
# SITE-NEUTRAL: pass partition/QoS/excludes at submission (see site.env).
#   sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/ar_digit_decoding.sbatch
# Output: $SIDLENS_WORK/derived/representation/exp1_ar_digit_decoding/<array-id>/cell-XX/

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

EXP=experiments/representation/exp1_ar_digit_decoding
RUN_GROUP="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?run under sbatch}}"
case " $* " in *" --limit "*) RUN_GROUP="pilot-${RUN_GROUP}" ;; esac
ROOT="${SIDLENS_WORK}/derived/representation/exp1_ar_digit_decoding/${RUN_GROUP}"
OUT="${ROOT}/cell-$(printf '%02d' "${SLURM_ARRAY_TASK_ID:?submit as an array}")"
mkdir -p "${ROOT}"
mkdir "${OUT}"
printf 'running job=%s task=%s started=%s\n' "${RUN_GROUP}" "${SLURM_ARRAY_TASK_ID}" "$(date -Is)" > "${OUT}/status.txt"
finish_run() {
    local rc=$?
    local state=failed
    if [[ ${rc} -eq 0 ]]; then
        state=complete
        (cd "${OUT}" && find . -type f ! -name output.sha256 ! -name status.txt -print0 \
            | sort -z | xargs -0 sha256sum > output.sha256)
    fi
    printf '%s job=%s task=%s exit=%s finished=%s\n' "${state}" "${RUN_GROUP}" \
        "${SLURM_ARRAY_TASK_ID}" "${rc}" "$(date -Is)" > "${OUT}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    git ls-files -z src/sidlens vendor/onediffrec
    find src/sidlens/interventions src/sidlens/hooks src/sidlens/analysis "${EXP}" -type f \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/representation/ar_digit_decoding.sbatch pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
cp "${EXP}/protocol.md" "${OUT}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{
    date -Is; hostname; uname -a; "${PYBIN}" -VV
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null \
        || echo "no GPU"
} > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

CAPTURES=(281053 281054)
"${PYBIN}" -u "${EXP}/run.py" --capture "${SIDLENS_WORK}/derived/ar_capture/${CAPTURES[${SLURM_ARRAY_TASK_ID}]}" --out "${OUT}" "$@"
