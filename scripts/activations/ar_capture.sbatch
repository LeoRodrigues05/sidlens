#!/bin/bash
#SBATCH --job-name=sidlens-ar-capture
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# AR activation capture (scripts/activations/ar_capture.py) under the standard gate and
# provenance record. SITE-NEUTRAL: there are no partition, account, node or
# GPU lines here, so the same file runs on every cluster. Pass them at
# submission, or set SBATCH_PARTITION / SBATCH_ACCOUNT in site.env. Submit
# from the repo root (the log path is relative):
#
#   GPU, full cohort, all layers:
#     sbatch -p <partition> --gres=gpu:1 scripts/activations/ar_capture.sbatch \
#         --ckpt next-item_best --template eval --layers all
#   CPU pilot, 8 rows:
#     sbatch -p <partition> scripts/activations/ar_capture.sbatch \
#         --ckpt next-item_best --template eval --layers 0,13,27 --limit 8 --device cpu
#
# Output: $SIDLENS_WORK/derived/ar_capture/<job-id>/ -- a new directory per job,
# never reused. Needs the `ar` profile (weights) on this cluster.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

OUT="${SIDLENS_WORK}/derived/ar_capture/${SLURM_JOB_ID:?run under sbatch}"
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
    printf '%s job=%s exit=%s finished=%s\n' "${state}" "${SLURM_JOB_ID}" "${rc}" \
        "$(date -Is)" > "${OUT}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    git ls-files -z src/sidlens vendor/onediffrec
    printf '%s\0' scripts/common/_preamble.sh scripts/activations/ar_capture.py scripts/activations/ar_capture.sbatch \
        src/sidlens/data/ar_prompts.py src/sidlens/data/sids.py src/sidlens/interventions/history.py \
        src/sidlens/hooks/store.py pyproject.toml
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{
    date -Is; hostname; uname -a; "${PYBIN}" -VV
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null \
        || echo "no GPU"
} > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

"${PYBIN}" -u scripts/activations/ar_capture.py --out "${OUT}/capture" "$@"
