#!/bin/bash
#SBATCH --job-name=onediffrec-sft
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
DATA_ROOT=${WORK}/data/Amazon18
MODEL_DIR=${WORK}/models/Qwen2.5-1.5B-Instruct
CATEGORY=Industrial_and_Scientific
VARIANT=rqvae_4codebook_128
RUN_MODE=${RUN_MODE:-smoke}

TRAIN_FILE=${DATA_ROOT}/train/${CATEGORY}_${VARIANT}.csv
VALID_FILE=${DATA_ROOT}/valid/${CATEGORY}_${VARIANT}.csv
INDEX_FILE=${DATA_ROOT}/${CATEGORY}/${CATEGORY}.index_4codebook_128.json
ITEM_FILE=${DATA_ROOT}/${CATEGORY}/${CATEGORY}.item.json

case "${RUN_MODE}" in
    smoke)
        SAMPLE=64
        EPOCHS=1
        GLOBAL_BATCH=32
        MICRO_BATCH=8
        OUTPUT_DIR=${WORK}/outputs/sft-smoke-${SLURM_JOB_ID}
        ;;
    full)
        SAMPLE=-1
        EPOCHS=10
        GLOBAL_BATCH=1024
        MICRO_BATCH=${MICRO_BATCH:-8}
        OUTPUT_DIR=${WORK}/outputs/sft-full-${SLURM_JOB_ID}
        ;;
    *)
        echo "Unknown RUN_MODE=${RUN_MODE}; expected smoke or full" >&2
        exit 2
        ;;
esac

for path in "${ENV_PREFIX}/bin/python" "${MODEL_DIR}/model.safetensors" "${TRAIN_FILE}" "${VALID_FILE}" "${INDEX_FILE}" "${ITEM_FILE}"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done

WORLD_SIZE=4
GRAD_ACCUM=$((GLOBAL_BATCH / MICRO_BATCH / WORLD_SIZE))
if (( GRAD_ACCUM < 1 || GLOBAL_BATCH % (MICRO_BATCH * WORLD_SIZE) != 0 )); then
    echo "Invalid batch geometry: global=${GLOBAL_BATCH}, micro=${MICRO_BATCH}, world=${WORLD_SIZE}" >&2
    exit 1
fi

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_DIR=${WORK}/wandb
export HF_HOME=${WORK}/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MANIFEST_DIR=${WORK}/manifests/sft-${RUN_MODE}-${SLURM_JOB_ID}
mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${MANIFEST_DIR}"
cd "${REPO}"

git rev-parse HEAD > "${MANIFEST_DIR}/git-head.txt"
git status --short > "${MANIFEST_DIR}/git-status.txt"
git diff -- data.py sft.py > "${MANIFEST_DIR}/source.patch"
sha256sum data.py sft.py requirements-sft.txt scripts/run_sft.sbatch > "${MANIFEST_DIR}/source-sha256.txt"
"${ENV_PREFIX}/bin/python" -m pip freeze > "${MANIFEST_DIR}/pip-freeze.txt"

echo "[$(date --iso-8601=seconds)] mode=${RUN_MODE} host=$(hostname) job=${SLURM_JOB_ID}"
echo "output=${OUTPUT_DIR} global_batch=${GLOBAL_BATCH} micro_batch=${MICRO_BATCH} grad_accum=${GRAD_ACCUM}"
echo "manifest=${MANIFEST_DIR}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
"${ENV_PREFIX}/bin/python" "${REPO}/scripts/validate_sft_assets.py" \
    --data-root "${DATA_ROOT}" --category "${CATEGORY}" --method rqvae --codebooks 4 --codebook-size 128

srun --ntasks=1 --kill-on-bad-exit=1 \
    "${ENV_PREFIX}/bin/torchrun" --standalone --nproc_per_node=4 \
    "${REPO}/sft.py" \
    --base_model "${MODEL_DIR}" \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${VALID_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample "${SAMPLE}" \
    --batch_size "${GLOBAL_BATCH}" \
    --micro_batch_size "${MICRO_BATCH}" \
    --num_epochs "${EPOCHS}" \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --category "${CATEGORY}" \
    --seed 42 \
    --sid_index_path "${INDEX_FILE}" \
    --item_meta_path "${ITEM_FILE}" \
    --train_from_scratch False \
    --freeze_LLM False \
    --wandb_project OneDiffRec_Industrial \
    --wandb_run_name "${CATEGORY}_${VARIANT}_${RUN_MODE}_${SLURM_JOB_ID}"

test -f "${OUTPUT_DIR}/final_checkpoint/config.json"
test -f "${OUTPUT_DIR}/final_checkpoint/tokenizer.json"
du -sh "${OUTPUT_DIR}"
echo "[$(date --iso-8601=seconds)] ${RUN_MODE} SFT complete: ${OUTPUT_DIR}/final_checkpoint"
