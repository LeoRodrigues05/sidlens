#!/bin/bash
#SBATCH --job-name=sidlens-diff-v6diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# Describe how a reveal-trace cell's lists differ from controlled exp1's archive
# (experiments/representation/exp4_diffusion_reveal_trace/archive_diff.py), for cells whose
# run stopped at V6. One array task per cell index (0-17). It decides nothing.
# SITE-NEUTRAL: pass partition/QoS/excludes at submission, with site.env sourced. From the repo root:
#   sbatch --array=5,7 --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/diffusion_reveal_trace_archive_diff.sbatch <label>
# Output: $SIDLENS_WORK/derived/representation/exp4_diffusion_reveal_trace/<label>/cell-XX/archive_diff.json
# (<label> defaults to v6-diagnosis-<array-id>). An existing cell directory is refused.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

LABEL="${1:-v6-diagnosis-${SLURM_ARRAY_JOB_ID:?submit as an array}}"
ROOT="${SIDLENS_WORK}/derived/representation/exp4_diffusion_reveal_trace/${LABEL}"
mkdir -p "${ROOT}"
sidlens_gate
"${PYBIN}" -u experiments/representation/exp4_diffusion_reveal_trace/archive_diff.py \
    --cell "${SLURM_ARRAY_TASK_ID}" --out "${ROOT}/cell-$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
