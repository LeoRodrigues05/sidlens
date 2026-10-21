#!/bin/bash
#SBATCH --job-name=sidlens-diff-trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# DiffGRM reveal trace: what the decode holds at every step
# (experiments/representation/exp4_diffusion_reveal_trace). One array task per checkpoint:
#   0-17 = controlled exp1's grid in its order (MQ, rqkmeans, rqvae x depth 3,4,5 x width 128,512),
#   18 = diff-next1-MQ-5cb-256, 19 = diff-next1-rqvae-3cb-256. GPU by default; --device cpu also works
#   (the model is 4 x 256) with -p <cpu partition> --gres=none.
# SITE-NEUTRAL: pass partition/QoS/excludes at submission (see site.env), with site.env sourced
# so SIDLENS_REPO is set. Submit from the repo root:
#   pilot:    sbatch --array=6,14 --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/diffusion_reveal_trace.sbatch --limit 64
#   primary:  sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/diffusion_reveal_trace.sbatch
#   no exp1 archive on this cluster (results provisional): add --allow-missing-archive
#   where the queue caps submitted jobs per user (CIAI: 4), submit the cells in blocks and keep them in
#   one run directory by passing the first block's array id to the later ones:
#     sbatch --array=0-3 ...            -> "Submitted batch job G"
#     SIDLENS_RUN_GROUP=G sbatch --array=4-7 ...   (and so on; a cell directory that exists is refused)
# Output: $SIDLENS_WORK/derived/representation/exp4_diffusion_reveal_trace/<array-id>/cell-XX/
# (pilot-<array-id>/ when --limit is given). New directory per job, never reused.

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

EXP=experiments/representation/exp4_diffusion_reveal_trace
RUN_GROUP="${SIDLENS_RUN_GROUP:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?run under sbatch}}}"
case " $* " in *" --limit "*) RUN_GROUP="pilot-${RUN_GROUP}" ;; esac
ROOT="${SIDLENS_WORK}/derived/representation/exp4_diffusion_reveal_trace/${RUN_GROUP}"
OUT="${ROOT}/cell-$(printf '%02d' "${SLURM_ARRAY_TASK_ID:?submit as an array}")"
mkdir -p "${ROOT}"
mkdir "${OUT}"
printf 'running job=%s task=%s started=%s\n' "${RUN_GROUP}" "${SLURM_ARRAY_TASK_ID}" "$(date -Is)" > "${OUT}/status.txt"
finish_run() {
    local rc=$?
    local state=failed
    # A cancelled or timed-out job reaches this trap with rc=0 unless the signal
    # is turned into an exit code (below), and would be marked complete with no
    # results. Completion therefore also requires the runner's last file.
    if [[ ${rc} -eq 0 && ! -f "${OUT}/validation.json" ]]; then
        rc=1
    fi
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
trap 'exit 143' TERM
trap 'exit 130' INT

sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    git ls-files -z src/sidlens vendor/onediffrec
    find src/sidlens/analysis src/sidlens/interventions src/sidlens/hooks "${EXP}" -type f \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/representation/diffusion_reveal_trace.sbatch \
        experiments/controlled/exp1_matched_beam/validate.py pyproject.toml
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

"${PYBIN}" -u "${EXP}/run.py" --cell "${SLURM_ARRAY_TASK_ID}" --out "${OUT}" "$@"
