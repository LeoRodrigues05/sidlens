# Shared preamble for every sidlens SLURM job.
#
# The verify gate is not optional. Upstream's own run directories symlink their
# code into a live mutable tree, which is how a "reproducible" run silently
# becomes unreproducible. Every job here proves its substrate first.
set -euo pipefail

# Sourcing this file must be all-or-nothing. A partial source previously left
# sidlens_gate undefined, and the job carried on past a "command not found"
# and reported success having done nothing.

# Per-cluster settings live OUTSIDE the repo, so one checkout serves every
# cluster: SIDLENS_WORK, SIDLENS_PYTHON, SIDLENS_VERIFY_PROFILE, module loads.
# See scripts/common/site.env.example. Absent file = this (original) cluster's defaults.
SIDLENS_SITE_ENV="${SIDLENS_SITE_ENV:-${HOME}/.config/sidlens/site.env}"
if [[ -f "${SIDLENS_SITE_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${SIDLENS_SITE_ENV}"
fi

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
export SIDLENS_WORK="${SIDLENS_WORK:-/l/users/leo.rodrigues/sidlens}"
PYBIN="${SIDLENS_PYTHON:-${SIDLENS_WORK}/venv/bin/python}"
[[ -x "${PYBIN}" ]] || { echo "FATAL: no python at ${PYBIN} (set SIDLENS_PYTHON)" >&2; exit 1; }

# Compute nodes have no outbound network; make that a loud failure rather than a
# multi-minute hang inside some library's auto-download path.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SIDLENS_WORK}/cache/hf"
export TOKENIZERS_PARALLELISM=false

# A cluster holding only part of the substrate (e.g. `core` without the 12 GB
# of weights) sets SIDLENS_VERIFY_PROFILE; the gate then checks exactly that
# profile's files and says so in the log. Unset = the whole snapshot.
sidlens_gate() {
    local profile_args=()
    if [[ -n "${SIDLENS_VERIFY_PROFILE:-}" ]]; then
        profile_args=(--profile "${SIDLENS_VERIFY_PROFILE}")
    fi
    echo "[gate] verifying frozen substrate${SIDLENS_VERIFY_PROFILE:+ (profile ${SIDLENS_VERIFY_PROFILE})}..."
    "${PYBIN}" -m sidlens.cli verify --strict --quick ${profile_args[@]+"${profile_args[@]}"} \
        || { echo "[gate] FAILED -- substrate drifted; refusing to run" >&2; exit 1; }
    echo "[gate] ok"
}

sidlens_provenance_stamp() {
    # Record what this job ran against, next to whatever it produces.
    local out="$1"
    mkdir -p "$(dirname "${out}")"
    {
        echo "snapshot=$(cat "${SIDLENS_REPO}/manifests/CURRENT")"
        echo "verify_profile=${SIDLENS_VERIFY_PROFILE:-full-snapshot}"
        echo "slurm_job=${SLURM_JOB_ID:-none}"
        echo "node=$(hostname)"
        echo "started=$(date -Is)"
        echo "git_head=$(git -C "${SIDLENS_REPO}" rev-parse HEAD)"
        echo "git_dirty=$(git -C "${SIDLENS_REPO}" status --porcelain | wc -l)"
    } > "${out}"
}
