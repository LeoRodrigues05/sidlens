#!/bin/bash
#SBATCH --job-name=onediffrec-setup
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=gpu-debug-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
ARCHIVE=${REPO}/data/OneDiffRec_data.tar.gz
DATA_ROOT=${WORK}/data/Amazon18
MODEL_DIR=${WORK}/models/Qwen2.5-1.5B-Instruct
MODEL_REVISION=989aa7980e4cf806f80c7fef2b1adb7bc71aa306
MODEL_SHA256=dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
ARCHIVE_SHA256=c3ab0bedf84bd52cfc79f281e358518792d01f96e1a58ab68bbdd56c8374ce89

export CONDA_PKGS_DIRS=${WORK}/cache/conda-pkgs
export PIP_CACHE_DIR=${WORK}/cache/pip
export HF_HOME=${WORK}/cache/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1

mkdir -p "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${MODEL_DIR}" "${WORK}/outputs" "${WORK}/wandb" "${WORK}/manifests"
cd "${REPO}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    /apps/local/anaconda3/bin/conda create --yes --prefix "${ENV_PREFIX}" python=3.11 pip
fi

"${ENV_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_PREFIX}/bin/python" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"${ENV_PREFIX}/bin/python" -m pip install --requirement "${REPO}/requirements-sft.txt"
"${ENV_PREFIX}/bin/python" -m pip check

echo "${ARCHIVE_SHA256}  ${ARCHIVE}" | sha256sum --check --strict

DATA_MEMBERS=(
    data/Amazon18/train/Industrial_and_Scientific_rqvae_4codebook_128.csv
    data/Amazon18/valid/Industrial_and_Scientific_rqvae_4codebook_128.csv
    data/Amazon18/test/Industrial_and_Scientific_rqvae_4codebook_128.csv
    data/Amazon18/info/Industrial_and_Scientific_rqvae_4codebook_128.txt
    data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.index_4codebook_128.json
    data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.item.json
)

needs_extract=0
for member in "${DATA_MEMBERS[@]}"; do
    [[ -f "${WORK}/${member}" ]] || needs_extract=1
done
if (( needs_extract )); then
    tar -xzf "${ARCHIVE}" -C "${WORK}" "${DATA_MEMBERS[@]}"
fi

"${ENV_PREFIX}/bin/python" "${REPO}/scripts/validate_sft_assets.py" \
    --data-root "${DATA_ROOT}" \
    --category Industrial_and_Scientific \
    --method rqvae \
    --codebooks 4 \
    --codebook-size 128

"${ENV_PREFIX}/bin/hf" download Qwen/Qwen2.5-1.5B-Instruct \
    --revision "${MODEL_REVISION}" \
    --local-dir "${MODEL_DIR}"
echo "${MODEL_SHA256}  ${MODEL_DIR}/model.safetensors" | sha256sum --check --strict

"${ENV_PREFIX}/bin/python" - <<'PY'
import accelerate
import datasets
import fire
import torch
import transformers
import wandb
from transformers import AutoConfig, AutoTokenizer

model_path = "/l/users/leo.rodrigues/onediffrec/models/Qwen2.5-1.5B-Instruct"
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "wandb": wandb.__version__,
    "model_type": config.model_type,
    "architectures": config.architectures,
    "layers": config.num_hidden_layers,
    "hidden_size": config.hidden_size,
    "tokenizer_size": len(tokenizer),
    "cuda_devices": torch.cuda.device_count(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
})
assert torch.cuda.device_count() == 1
assert torch.cuda.is_bf16_supported()
assert config.model_type == "qwen2"
PY

"${ENV_PREFIX}/bin/python" -m pip freeze > "${WORK}/manifests/sft-pip-freeze.txt"
/apps/local/anaconda3/bin/conda list --prefix "${ENV_PREFIX}" --explicit > "${WORK}/manifests/sft-conda-explicit.txt"

du -sh "${ENV_PREFIX}" "${DATA_ROOT}" "${MODEL_DIR}"
echo "[$(date --iso-8601=seconds)] setup complete"
