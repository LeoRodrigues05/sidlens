#!/bin/bash
#SBATCH --job-name=sidlens-retro-exp2
#SBATCH --partition=ws-ia
#SBATCH --nodelist=ws-l3-020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Frozen-prediction analysis: CPU only.  The node is pinned at the user's
# request, but no GPU is requested or used.
SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/_preamble.sh" >&2
    exit 1
fi

cd "${SIDLENS_REPO}"
sidlens_gate
sidlens_provenance_stamp \
    "${SIDLENS_WORK}/derived/retrospective/exp2_prefix/.provenance"

echo "[retrospective-exp2] starting $(date -Is)"
"${PYBIN}" -u -m experiments.retrospective.exp2_prefix.run "$@"
echo "[retrospective-exp2] done $(date -Is)"

