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
# The labels gate must check the catalogue being built, not always Industrial:
# `--catalog office` (only valid with `--source v3`) joins the Office labels.
label_category=Industrial_and_Scientific
prev=""
for arg in "$@"; do
    if [[ "${arg}" == "--catalog=office" || ( "${prev}" == "--catalog" && "${arg}" == "office" ) ]]; then
        label_category=Office
    fi
    prev="${arg}"
done
"${PYBIN}" scripts/data/build_labels.py --check --category "${label_category}"
"${PYBIN}" -u -m sidlens.analysis.semantic_mapping build \
    --out "${SIDLENS_WORK}/derived/semantic_mapping/${SLURM_JOB_ID}" "$@"
