#!/bin/bash
#SBATCH --job-name=sidlens-orders-summary
#SBATCH --partition=ws-ia
#SBATCH --nodelist=ws-l3-020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
source "${SIDLENS_REPO}/scripts/common/_preamble.sh"
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ARRAY_ID="${1:?supply completed fixed-order array ID}"
PILOT_ID="${2:?supply pilot array ID}"
[[ "${ARRAY_ID}" =~ ^[0-9]+$ && "${PILOT_ID}" =~ ^[0-9]+$ ]] || exit 2
ROOT="${SIDLENS_WORK}/derived/controlled/exp2_fixed_orders"
OUT="${ROOT}/summary-${ARRAY_ID}-${SLURM_JOB_ID}"
sidlens_gate
"${PYBIN}" -u experiments/controlled/exp2_fixed_orders/summarize.py \
    --input "${ROOT}/${ARRAY_ID}" --out "${OUT}" \
    --pilot "${ROOT}/${PILOT_ID}/cell-06" --bootstrap 2000 --seed 20260915
sidlens_provenance_stamp "${OUT}/.provenance"
cp scripts/controlled/fixed_orders_summary.sbatch "${OUT}/submission.sbatch"
cp experiments/controlled/exp2_fixed_orders/summarize.py "${OUT}/analysis.py"
cp experiments/controlled/exp2_fixed_orders/common.py "${OUT}/common.py"
cd "${OUT}"
find . -type f ! -name output.sha256 -print0 | sort -z | xargs -0 sha256sum > output.sha256
