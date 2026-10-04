#!/bin/bash
#SBATCH --job-name=sidlens-ar-infomap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# What the AR residual stream holds, layer by layer (experiments/representation/exp3_ar_information_map).
# One array task per cell: 0 = next-item_best (rqkmeans 3x128), 1 = oneoff_rqvae4cb128 (rqvae 4x128).
# The first argument names the stage:
#   pilot    everything below on small limits (64 test rows, 600 train rows, 1 SAE epoch, 2 layers)
#   gpu      self-greedy captures (test, train; all layers + final norm) -> SAEs -> lenses, in one allocation
#   census   CPU: label every live SAE latent (pass --group <group dir>; submit with
#            -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none on the CIAI cluster)
# SITE-NEUTRAL: pass partition/QoS/excludes at submission (see site.env). Submit from the repo root:
#   sbatch --exclude=$X scripts/representation/ar_information_map.sbatch pilot
#   sbatch --exclude=$X --dependency=afterok:<pilot> scripts/representation/ar_information_map.sbatch gpu
# Output: $SIDLENS_WORK/derived/representation/exp3_ar_information_map/<array-id>/cell-XX/
#   capture-test/ capture-train/ sae/ lens/ (and census-<job>/ from the census stage). New directory per job.

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

STAGE="${1:?first argument: pilot | gpu | census}"; shift
CELL="${SLURM_ARRAY_TASK_ID:?submit as an array}"
CKPTS=(next-item_best oneoff_rqvae4cb128)
CKPT="${CKPTS[$CELL]}"
EXP=experiments/representation/exp3_ar_information_map
BASE="${SIDLENS_WORK}/derived/representation/exp3_ar_information_map"
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"

if [[ "${STAGE}" == "census" ]]; then
    GROUP=""
    while [[ $# -gt 0 ]]; do case "$1" in --group) GROUP="$2"; shift 2 ;; *) shift ;; esac; done
    CELLDIR="${GROUP:?--group <group dir>}/cell-$(printf '%02d' "${CELL}")"
    OUT="${CELLDIR}/census-${JOB}"
else
    RUN_GROUP="${JOB}"; [[ "${STAGE}" == "pilot" ]] && RUN_GROUP="pilot-${JOB}"
    OUT="${BASE}/${RUN_GROUP}/cell-$(printf '%02d' "${CELL}")"
fi
mkdir -p "$(dirname "${OUT}")"
mkdir "${OUT}"
printf 'running job=%s task=%s stage=%s started=%s\n' "${JOB}" "${CELL}" "${STAGE}" "$(date -Is)" > "${OUT}/status.txt"
finish_run() {
    local rc=$?
    local state=failed
    if [[ ${rc} -eq 0 ]]; then
        state=complete
        (cd "${OUT}" && find . -type f ! -name output.sha256 ! -name status.txt ! -path './capture-*/store/shards/*' -print0 \
            | sort -z | xargs -0 sha256sum > output.sha256)
    fi
    printf '%s job=%s task=%s stage=%s exit=%s finished=%s\n' "${state}" "${JOB}" "${CELL}" "${STAGE}" "${rc}" \
        "$(date -Is)" > "${OUT}/status.txt"
    trap - EXIT
    exit "${rc}"
}
trap finish_run EXIT

sidlens_gate
sidlens_provenance_stamp "${OUT}/.provenance"
{
    git ls-files -z src/sidlens vendor/onediffrec
    find src/sidlens "${EXP}" experiments/representation/exp2_ar_sae -type f \( -name '*.py' -o -name '*.md' \) -print0
    find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
    printf '%s\0' scripts/common/_preamble.sh scripts/representation/ar_information_map.sbatch \
        scripts/activations/ar_capture.py pyproject.toml
} | sort -zu > "${OUT}/source-files.nul"
xargs -0 sha256sum < "${OUT}/source-files.nul" > "${OUT}/source.sha256"
tar -czf "${OUT}/source.tar.gz" --null -T "${OUT}/source-files.nul"
cp "${EXP}/protocol.md" "${OUT}/protocol.md"
git status --porcelain=v1 --untracked-files=all > "${OUT}/git-status.txt"
git diff --binary HEAD > "${OUT}/tracked-changes.diff"
{
    date -Is; hostname; uname -a; "${PYBIN}" -VV
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null || echo "no GPU"
} > "${OUT}/environment.txt"
"${PYBIN}" -m pip freeze --local > "${OUT}/requirements.freeze.txt"

if [[ "${STAGE}" == "census" ]]; then
    "${PYBIN}" -u "${EXP}/census.py" --cell "${CELL}" --test-capture "${CELLDIR}/capture-test" \
        --sae-run "${CELLDIR}/sae" --out "${OUT}/result"
    exit 0
fi

if [[ "${STAGE}" == "pilot" ]]; then
    TEST_LIM=(--limit 64); TRAIN_LIM=(--limit 600); SAE_ARGS=(--epochs 1 --layers 0,27)
    LENS_ARGS=(--lens-layers 0,27 --dec-layers 0,27)
else
    TEST_LIM=(); TRAIN_LIM=(); SAE_ARGS=(); LENS_ARGS=()
fi
"${PYBIN}" -u scripts/activations/ar_capture.py --ckpt "${CKPT}" --template eval --split test \
    --layers all,norm --target self-greedy ${TEST_LIM[@]+"${TEST_LIM[@]}"} --out "${OUT}/capture-test"
"${PYBIN}" -u scripts/activations/ar_capture.py --ckpt "${CKPT}" --template eval --split train \
    --layers all,norm --target self-greedy ${TRAIN_LIM[@]+"${TRAIN_LIM[@]}"} --out "${OUT}/capture-train"
"${PYBIN}" -u "${EXP}/train_sae.py" --cell "${CELL}" --train-capture "${OUT}/capture-train" \
    --test-capture "${OUT}/capture-test" ${SAE_ARGS[@]+"${SAE_ARGS[@]}"} --out "${OUT}/sae"
"${PYBIN}" -u "${EXP}/lens.py" --cell "${CELL}" --train-capture "${OUT}/capture-train" \
    --test-capture "${OUT}/capture-test" ${LENS_ARGS[@]+"${LENS_ARGS[@]}"} --out "${OUT}/lens"
