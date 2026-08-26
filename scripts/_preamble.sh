# Shared preamble for every sidlens SLURM job.
#
# The verify gate is not optional. Upstream's own run directories symlink their
# code into a live mutable tree, which is how a "reproducible" run silently
# becomes unreproducible. Every job here proves its substrate first.
set -euo pipefail

# Sourcing this file must be all-or-nothing. A partial source previously left
# sidlens_gate undefined, and the job carried on past a "command not found"
# and reported success having done nothing.

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/onediffrec/sidlens}"
export SIDLENS_WORK="${SIDLENS_WORK:-/l/users/leo.rodrigues/sidlens}"
PYBIN="${SIDLENS_WORK}/venv/bin/python"

# Compute nodes have no outbound network; make that a loud failure rather than a
# multi-minute hang inside some library's auto-download path.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SIDLENS_WORK}/cache/hf"
export TOKENIZERS_PARALLELISM=false

sidlens_gate() {
    echo "[gate] verifying frozen substrate..."
    "${PYBIN}" -m sidlens.cli verify --strict --quick \
        || { echo "[gate] FAILED -- substrate drifted; refusing to run" >&2; exit 1; }
    echo "[gate] ok"
}

sidlens_provenance_stamp() {
    # Record what this job ran against, next to whatever it produces.
    local out="$1"
    mkdir -p "$(dirname "${out}")"
    {
        echo "snapshot=$(cat "${SIDLENS_REPO}/manifests/CURRENT")"
        echo "slurm_job=${SLURM_JOB_ID:-none}"
        echo "node=$(hostname)"
        echo "started=$(date -Is)"
        echo "git_head=$(git -C "${SIDLENS_REPO}" rev-parse HEAD)"
        echo "git_dirty=$(git -C "${SIDLENS_REPO}" status --porcelain | wc -l)"
    } > "${out}"
}
