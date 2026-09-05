#!/bin/bash
#SBATCH --job-name=sidlens-exp2
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Experiment 2 -- RQ cluster geometry. CPU only, no GPU, no model.
#
# Runs here rather than on a login node: the sweep is a few minutes of solid
# numpy across 27 variants, and the login node's policy daemon SIGTERMs it
# partway through (observed: killed after 16/27 variants, exit 143).

# SLURM copies the batch script into its spool directory, so BASH_SOURCE points
# there, not at the repo. The repo path has to be absolute (overridable for a
# relocated checkout).
SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/_preamble.sh" >&2
    exit 1
fi

cd "${SIDLENS_REPO}"
sidlens_gate
sidlens_provenance_stamp "${SIDLENS_WORK}/derived/exp2_geometry/.provenance"

echo "[exp2] starting $(date -Is)"
"${PYBIN}" -u experiments/exp2_geometry/run.py "$@"
echo "[exp2] done $(date -Is)"
