#!/bin/bash
#SBATCH --job-name=sidlens-ar-retrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# Retrain the 18 Qwen-AR next-item models of the width-128/512 grid with the
# recorded August-2026 recipe (experiments/substrate/exp1_ar_next_item_retrain).
# Each array task takes every N-th variant of run.ORDER (N = number of tasks),
# so 4 tasks fit this cluster's 4-task submit cap. Tasks resume: a variant
# whose weights are hashed is never retrained, and an evaluated one is skipped.
# SITE-NEUTRAL: pass partition/QoS/excludes at submission (see site.env). Submit
# from the repo root, with RUN_NAME naming the run directory:
#   pilot: RUN_NAME=pilot-$(date +%Y%m%d) sbatch --array=0 --gres=gpu:1 --mem=64G --time=02:00:00 \
#            --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/substrate/ar_next_item_retrain.sbatch \
#            --smoke --eval-limit 64 --variants rqkmeans_3codebook_128
#   full:  RUN_NAME=r20261004 sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE \
#            scripts/substrate/ar_next_item_retrain.sbatch
#   fewer GPUs per task (global batch stays 1,024): add --gres=gpu:2 --mem=96G
# Output: $SIDLENS_WORK/runs/ar_next_item_retrain/$RUN_NAME/<variant>/ (persistent
# per variant, by design) and .../jobs/<array>_<task>/ (new per job, never reused).

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

EXP=experiments/substrate/exp1_ar_next_item_retrain
: "${RUN_NAME:?set RUN_NAME, e.g. RUN_NAME=r20261004}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?run under sbatch}}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
RUN_ROOT="${SIDLENS_WORK}/runs/ar_next_item_retrain/${RUN_NAME}"
JOBDIR="${RUN_ROOT}/jobs/${ARRAY_ID}_${TASK}"
mkdir -p "${RUN_ROOT}/jobs"
mkdir "${JOBDIR}"
printf 'running job=%s task=%s started=%s\n' "${ARRAY_ID}" "${TASK}" "$(date -Is)" > "${JOBDIR}/status.txt"
finish_run() {
    local rc=$?
    local state=failed
    [[ ${rc} -eq 0 ]] && state=complete
    printf '%s job=%s task=%s exit=%s finished=%s\n' "${state}" "${ARRAY_ID}" "${TASK}" \
        "${rc}" "$(date -Is)" > "${JOBDIR}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

sidlens_gate
sidlens_provenance_stamp "${JOBDIR}/.provenance"
{
    git ls-files -z vendor/onediffrec src/sidlens/provenance
    find "${EXP}" -type f \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/substrate/ar_next_item_retrain.sbatch src/sidlens/paths.py
} | sort -zu > "${JOBDIR}/source-files.nul"
xargs -0 sha256sum < "${JOBDIR}/source-files.nul" > "${JOBDIR}/source.sha256"
tar -czf "${JOBDIR}/source.tar.gz" --null -T "${JOBDIR}/source-files.nul"
cp "${EXP}/protocol.md" "${JOBDIR}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${JOBDIR}/git-status.txt"
git diff --binary HEAD > "${JOBDIR}/tracked-changes.diff"
{
    date -Is; hostname; uname -a; "${PYBIN}" -VV
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null \
        || echo "no GPU"
} > "${JOBDIR}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${JOBDIR}/requirements.freeze.txt"

# GPUs this task actually holds, and the wall time it has left (minus 10 min).
# Count the ALLOCATION, never `nvidia-smi -L`: on CIAI nvidia-smi lists all 4
# GPUs of the node, and torchrun then put 4 ranks on one allocated GPU ("NCCL
# Duplicate GPU detected"; pilot 294718, as upstream's job 171602 did in August).
GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${GPUS}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=, read -ra _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    GPUS=${#_gpu_ids[@]}
fi
[[ "${GPUS}" =~ ^[0-9]+$ ]] && (( GPUS >= 1 )) || { echo "FATAL: cannot tell how many GPUs this job holds" >&2; exit 1; }
echo "allocated GPUs: ${GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
# The allocation must also be usable. CIAI gpu-51 allocates 4 GPUs but exposes
# only 2, so 4 ranks shared 2 devices and NCCL aborted every variant within
# 2 minutes (jobs 294804, 295416). Stop before any variant starts.
SEEN=$("${PYBIN}" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "${SEEN}" != "${GPUS}" ]]; then
    echo "FATAL: $(hostname -s) has ${SEEN} usable GPU(s) for an allocation of ${GPUS}; resubmit with --exclude=$(hostname -s)" >&2
    exit 1
fi
LIMIT=$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null)
budget_seconds() {   # Slurm time left: [D-]HH:MM:SS, MM:SS or UNLIMITED
    local t=$1 d=0 h=0 m=0 s=0
    [[ "$t" == UNLIMITED || -z "$t" ]] && { echo 86400; return; }
    [[ "$t" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r a b c <<< "$t"
    if [[ -n "$c" ]]; then h=$a; m=$b; s=$c; else m=$a; s=$b; fi
    echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))
}
BUDGET=$(( $(budget_seconds "${LIMIT}") - 600 ))

# Variant list: explicit --variants in "$@" wins; otherwise ORDER[TASK::NTASKS].
case " $* " in
    *" --variants "*) VARIANT_ARGS=() ;;
    *) VARIANT_ARGS=(--variants "$(cd "${EXP}" && "${PYBIN}" -c \
            "import run; print(','.join(run.ORDER[${TASK}::${NTASKS}]))")") ;;
esac
echo "task ${TASK}/${NTASKS}: gpus=${GPUS} budget=${BUDGET}s ${VARIANT_ARGS[*]} $*" | tee "${JOBDIR}/arguments.txt"

"${PYBIN}" -u "${EXP}/run.py" --run-root "${RUN_ROOT}" --job "${ARRAY_ID}_${TASK}" \
    --gpus "${GPUS}" --budget-seconds "${BUDGET}" "${VARIANT_ARGS[@]}" "$@"
