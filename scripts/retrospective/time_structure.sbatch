#!/bin/bash
#SBATCH --job-name=sidlens-time-structure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Time in the data and in existing intervention outputs, CPU only
# (experiments/retrospective/exp6_time_structure). It reads frozen data plus
# the exp3/exp4/exp6/exp7 result directories named in run.py, and writes a new
# job directory.
# SITE-NEUTRAL: pass a CPU partition at submission. On the CIAI cluster:
#   sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/retrospective/time_structure.sbatch
# Output: $SIDLENS_WORK/derived/retrospective/exp6_time_structure/<job-id>/

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

EXP=experiments/retrospective/exp6_time_structure
OUT="${SIDLENS_WORK}/derived/retrospective/exp6_time_structure/${SLURM_JOB_ID:?run under sbatch}"
mkdir -p "$(dirname "${OUT}")"
mkdir "${OUT}"
printf 'running job=%s started=%s\n' "${SLURM_JOB_ID}" "$(date -Is)" > "${OUT}/status.txt"
finish_run() {
    local rc=$?
    local state=failed
    if [[ ${rc} -eq 0 ]]; then
        state=complete
        (cd "${OUT}" && find . -type f ! -name output.sha256 ! -name status.txt -print0 \
            | sort -z | xargs -0 sha256sum > output.sha256)
    fi
    printf '%s job=%s exit=%s finished=%s\n' "${state}" "${SLURM_JOB_ID}" "${rc}" "$(date -Is)" > "${OUT}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    git ls-files -z src/sidlens vendor/onediffrec
    find src/sidlens/data src/sidlens/analysis "${EXP}" -type f \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/retrospective/time_structure.sbatch pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
cp "${EXP}/protocol.md" "${OUT}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{ date -Is; hostname; uname -a; "${PYBIN}" -VV; } > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

"${PYBIN}" -u "${EXP}/run.py" --out "${OUT}/result" "$@"
