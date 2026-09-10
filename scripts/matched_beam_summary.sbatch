#!/bin/bash
#SBATCH --job-name=sidlens-matched-summary
#SBATCH --partition=ws-ia
#SBATCH --nodelist=ws-l3-020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
source "${SIDLENS_REPO}/scripts/_preamble.sh"
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ARRAY_ID="${1:?supply completed array job ID}"
[[ "${ARRAY_ID}" =~ ^[0-9]+$ ]] || exit 2
INPUT="${SIDLENS_WORK}/derived/controlled/exp1_matched_beam/${ARRAY_ID}"
OUT="${SIDLENS_WORK}/derived/controlled/exp1_matched_beam/summary-${ARRAY_ID}-${SLURM_JOB_ID}"
sidlens_gate
"${PYBIN}" -u experiments/controlled/exp1_matched_beam/summarize.py \
    --input "${INPUT}" --out "${OUT}" --bootstrap 2000 --seed 20260909
sidlens_provenance_stamp "${OUT}/.provenance"
cp scripts/matched_beam_summary.sbatch "${OUT}/submission.sbatch"
cp experiments/controlled/exp1_matched_beam/summarize.py "${OUT}/analysis.py"
find "${OUT}" -maxdepth 1 -type f ! -name output.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "${OUT}/output.sha256"
