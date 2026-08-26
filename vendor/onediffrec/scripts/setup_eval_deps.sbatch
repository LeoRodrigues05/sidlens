#!/bin/bash
#SBATCH --job-name=onediffrec-evaldeps
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Installs the extra deps that evaluate.py needs on top of requirements-sft.txt.
# evaluate.py:12 does `import bitsandbytes as bnb` (unused, but a hard import).

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda

export PIP_CACHE_DIR=${WORK}/cache/pip
export HF_HOME=${WORK}/cache/huggingface

cd "${REPO}"
echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"

"${ENV_PREFIX}/bin/python" -m pip install --requirement "${REPO}/requirements-eval.txt"
"${ENV_PREFIX}/bin/python" -m pip check || echo "pip check reported issues (non-fatal)"

"${ENV_PREFIX}/bin/python" - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "fire", "pandas", "tqdm", "numpy", "bitsandbytes"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"OK {m} {getattr(mod, '__version__', '?')}")
PY

"${ENV_PREFIX}/bin/python" -m pip freeze > "${WORK}/manifests/eval-pip-freeze.txt"
echo "[$(date --iso-8601=seconds)] eval deps ready"
