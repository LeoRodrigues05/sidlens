#!/bin/bash
#SBATCH --job-name=sidlens-final-maps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# The final (v3) SID maps of both catalogs (experiments/structure/exp4_final_maps).
# CPU only: v3 tables + their quantizer weights, frozen embeddings and metadata,
# Amazon-2018 labels, faiss refits of RQ-KMeans. Industrial runs first (minutes),
# then Office (hours: 17,696 items, permutation nulls, naming).
# SITE-NEUTRAL: pass a CPU partition at submission. On the CIAI cluster:
#   source ~/.config/sidlens/site.env
#   sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/structure/final_maps.sbatch
# Extra arguments go to run.py for both catalogs (e.g. --quick, --parts 0,M1).
# Output: $SIDLENS_WORK/derived/structure/exp4_final_maps/<job-id>/{industrial,office}/

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
# BLAS stays at 8 in the parent; run.py sets each pool worker's own limit.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

EXP=experiments/structure/exp4_final_maps
OUT="${SIDLENS_WORK}/derived/structure/exp4_final_maps/${SLURM_JOB_ID:?run under sbatch}"
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
    find src/sidlens/data src/sidlens/analysis "${EXP}" experiments/structure/exp3_collision_causes \
        experiments/structure/semantic_mapping -type f \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/structure/final_maps.sbatch \
        scripts/data/build_labels.py pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
cp "${EXP}/protocol.md" "${OUT}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{ date -Is; hostname; uname -a; lscpu | grep -E 'Model name|^CPU\(s\)'; "${PYBIN}" -VV; } > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

for CATALOG in industrial office; do
    "${PYBIN}" -u "${EXP}/run.py" --catalog "${CATALOG}" --out "${OUT}" \
        --workers "${SLURM_CPUS_PER_TASK}" --threads 8 "$@"
done
