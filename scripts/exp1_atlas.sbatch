#!/bin/bash
#SBATCH --job-name=sidlens-exp1
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Experiment 1a -- the semantic atlas, across the full 27-variant grid.
#
# CPU only, no GPU, no model. The cost is the permutation null: the conditional
# AMI is re-pooled over every parent cell for each of N_PERM shuffles, across
# six label fields and up to five digits. Measured at ~100 s for a 5-codebook
# variant at n-perm 200, so the grid is roughly 30-45 minutes -- well past what
# a login node tolerates, hence SLURM.
#
# Partition note: this cluster exposes `ws-ia` and `gpu`. The `cscc-cpu-p`
# partition the older exp2 script targeted no longer exists, and there is no
# accounting daemon, so no --account is passed.

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
sidlens_provenance_stamp "${SIDLENS_WORK}/derived/exp1_atlas/.provenance"

# The label join is external to the frozen substrate, so the gate above cannot
# vouch for it. Prove it separately, and refuse to produce an attribute table
# whose labels never joined.
echo "[exp1] checking the external label join"
"${PYBIN}" scripts/build_labels.py --check \
    || { echo "[exp1] label join below threshold; refusing to run" >&2; exit 1; }

echo "[exp1] starting $(date -Is)"
"${PYBIN}" -u experiments/exp1_atlas/run.py --variants all "$@"
echo "[exp1] done $(date -Is)"
