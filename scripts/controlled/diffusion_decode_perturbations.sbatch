#!/bin/bash
#SBATCH --job-name=sidlens-diff-perturb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# controlled/exp14: copy block, order swap and position shift inside DiffGRM decoding (experiments/controlled/exp14_diffusion_decode_perturbations).
# One job walks the cells (default all 20, exp4's order) because the CIAI GPU QoS caps jobs per user.
# SITE-NEUTRAL: source site.env first (SIDLENS_REPO), pass partition/QoS/excludes at submission. From the repo root:
#   pilot:    SIDLENS_CELLS=6,14 sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/diffusion_decode_perturbations.sbatch --limit 256
#   primary:  sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/diffusion_decode_perturbations.sbatch
# Output: $SIDLENS_WORK/derived/controlled/exp14_diffusion_decode_perturbations/<job>/cell-XX/ (pilot-<job>/ with --limit).

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
source "${SIDLENS_REPO}/scripts/common/_cells.sh"
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
export SIDLENS_WRAPPER=scripts/controlled/diffusion_decode_perturbations.sbatch

EXP=experiments/controlled/exp14_diffusion_decode_perturbations
RUN="${SLURM_JOB_ID:?run under sbatch}"
case " $* " in *" --limit "*) RUN="pilot-${RUN}" ;; esac
sidlens_gate
sidlens_run_cells "${EXP}" "${SIDLENS_WORK}/derived/controlled/exp14_diffusion_decode_perturbations/${RUN}" \
    "${SIDLENS_CELLS:-0-19}" "$@"
