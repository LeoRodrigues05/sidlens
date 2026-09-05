#!/bin/bash
#SBATCH --job-name=sidlens-smoke
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Capture-layer smoke test on the real checkpoints. CPU is enough -- this is a
# handful of forwards, not a sweep. 48G because the AR checkpoint is ~3 GB of
# fp32 weights plus activations.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/_preamble.sh" >&2
    exit 1
fi

cd "${SIDLENS_REPO}"
sidlens_gate
sidlens_provenance_stamp "${SIDLENS_WORK}/derived/hooks_smoke/.provenance"

echo "[smoke] starting $(date -Is)"
"${PYBIN}" -u scripts/hooks_smoke.py "$@"
echo "[smoke] done $(date -Is)"
