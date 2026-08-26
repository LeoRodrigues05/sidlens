#!/bin/bash
#SBATCH --job-name=onediffrec-verify
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=gpu-debug-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
ENV_PREFIX=${REPO}/.conda
CHECKPOINT=${CHECKPOINT:-/l/users/leo.rodrigues/onediffrec/outputs/sft-full-160793/final_checkpoint}

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "${REPO}"
echo "[$(date --iso-8601=seconds)] verifying ${CHECKPOINT} on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

srun --ntasks=1 --kill-on-bad-exit=1 "${ENV_PREFIX}/bin/python" - "${CHECKPOINT}" <<'PY'
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
).eval().cuda()

sid_tokens = ["<a_56>", "<b_30>", "<c_25>", "<d_124>"]
sid_ids = tokenizer.convert_tokens_to_ids(sid_tokens)
assert tokenizer.unk_token_id not in sid_ids
assert len(set(sid_ids)) == len(sid_ids)
for token, token_id in zip(sid_tokens, sid_ids):
    assert tokenizer.encode(token, add_special_tokens=False) == [token_id]

text = "User history: " + "".join(sid_tokens) + "\nNext item:"
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.inference_mode():
    logits = model(**inputs).logits

assert logits.shape == (1, inputs.input_ids.shape[1], len(tokenizer))
assert torch.isfinite(logits).all()
print(json.dumps({
    "status": "ok",
    "checkpoint": checkpoint,
    "cuda_device": torch.cuda.get_device_name(0),
    "dtype": str(next(model.parameters()).dtype),
    "tokenizer_size": len(tokenizer),
    "model_vocab_size": model.config.vocab_size,
    "sid_token_ids": sid_ids,
    "input_tokens": inputs.input_ids.shape[1],
    "logits_shape": list(logits.shape),
}, indent=2))
PY

echo "[$(date --iso-8601=seconds)] final checkpoint verification complete"
