#!/bin/bash
#SBATCH --job-name=sidlens-pytest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# The test suite on a compute node. On the CIAI cluster the login node does not
# mount $SIDLENS_WORK, so every test that loads a frozen checkpoint or cohort
# can only run under sbatch there. CPU only; arguments are passed to pytest.
# SITE-NEUTRAL: pass partition/QoS at submission (see site.env). Submit from
# the repo root:
#   sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/common/pytest.sbatch
#   sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/common/pytest.sbatch tests/interventions -k diffusion
# Writes nothing under $SIDLENS_WORK.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

sidlens_gate
echo "[pytest] node=$(hostname) work=${SIDLENS_WORK} head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
"${PYBIN}" -m pytest -q "$@"
