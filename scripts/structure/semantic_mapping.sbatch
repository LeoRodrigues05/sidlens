#!/bin/bash
#SBATCH --job-name=sidlens-mapping
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
source "${SIDLENS_REPO}/scripts/common/_preamble.sh" || exit 1
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
sidlens_gate
"${PYBIN}" scripts/data/build_labels.py --check
"${PYBIN}" -u -m sidlens.analysis.semantic_mapping build \
    --out "${SIDLENS_WORK}/derived/semantic_mapping/${SLURM_JOB_ID}" "$@"
