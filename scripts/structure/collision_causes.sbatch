#!/bin/bash
#SBATCH --job-name=sidlens-collision-causes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Why items share a Semantic ID (experiments/structure/exp3_collision_causes).
# CPU only: frozen SID tables + embeddings, plus faiss refits of RQ-KMeans.
# 96 CPUs because faiss k-means only reproduces the archived RQ-KMeans tables
# at 96 OpenMP threads; run.py refuses to interpret refits otherwise.
# SITE-NEUTRAL: pass a CPU partition at submission. On the CIAI cluster:
#   sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/structure/collision_causes.sbatch
# Extra arguments go to run.py (e.g. --quick).
# Output: $SIDLENS_WORK/derived/structure/exp3_collision_causes/<job-id>/

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export PYTHONUNBUFFERED=1

EXP=experiments/structure/exp3_collision_causes
OUT="${SIDLENS_WORK}/derived/structure/exp3_collision_causes/${SLURM_JOB_ID:?run under sbatch}"
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
    printf '%s\0' scripts/common/_preamble.sh scripts/structure/collision_causes.sbatch pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
cp "${EXP}/protocol.md" "${OUT}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{ date -Is; hostname; uname -a; "${PYBIN}" -VV; } > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

"${PYBIN}" -u "${EXP}/run.py" --out "${OUT}/result" --threads 96 "$@"
