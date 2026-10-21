#!/bin/bash
#SBATCH --job-name=sidlens-ar-copygrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.out

# Core copy estimates on the 12 accepted retrained Qwen-AR next-item models
# (experiments/controlled/exp11_ar_copy_grid). Task t of N takes cells t, t+N, ...
# so 3 tasks fit the CIAI submit cap. Pass 1 runs the teacher-forced stages
# (exp4 valid R,K; exp3 A; exp4 test R,K) for all of the task's cells, pass 2
# the beam-search stage (exp6), so the core estimates exist early.
# A failed stage is recorded in the cell's status.txt and the task moves on.
# SITE-NEUTRAL: pass partition/QoS/excludes at submission (see site.env). From the repo root:
#   pilot:  CELLS=10 sbatch --array=0 --time=02:00:00 --exclude=$SIDLENS_SBATCH_EXCLUDE \
#               scripts/controlled/ar_copy_grid.sbatch --limit 64
#   full:   sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_copy_grid.sbatch
#   CELLS="00 04" restricts the cell list (re-runs go to a new array id; nothing is reused).
# Output: $SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/<array-id>/cell-XX/<stage>/
# (pilot-<array-id>/ when --limit is given).

SIDLENS_REPO="${SIDLENS_REPO:-/home/leo.rodrigues/GenRecSys/sidlens/sidlens}"
if ! source "${SIDLENS_REPO}/scripts/common/_preamble.sh"; then
    echo "FATAL: cannot source ${SIDLENS_REPO}/scripts/common/_preamble.sh" >&2
    exit 1
fi
cd "${SIDLENS_REPO}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

EXP=experiments/controlled/exp11_ar_copy_grid
RUN_NAME=r20261004
VARIANTS=(rqkmeans_3codebook_128 rqkmeans_4codebook_128 rqkmeans_5codebook_128 rqkmeans_5codebook_512
          rqvae_3codebook_128 rqvae_3codebook_512 rqvae_4codebook_128 rqvae_4codebook_512
          rqvae_5codebook_128 rqvae_5codebook_512 MQ_3codebook_128 MQ_4codebook_512)
LIMIT=()
case " $* " in *" --limit "*) LIMIT=(--limit "$(sed -E 's/.*--limit ([0-9]+).*/\1/' <<< "$*")") ;; esac
RUN_GROUP="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?run under sbatch}}"
[[ ${#LIMIT[@]} -gt 0 ]] && RUN_GROUP="pilot-${RUN_GROUP}"
ROOT="${SIDLENS_WORK}/derived/controlled/exp11_ar_copy_grid/${RUN_GROUP}"
TASK="${SLURM_ARRAY_TASK_ID:?submit as an array}"
NTASK="${SLURM_ARRAY_TASK_COUNT:-1}"
if [[ -n "${CELLS:-}" ]]; then
    read -r -a ALL <<< "${CELLS}"
else
    ALL=()
    for ((i = 0; i < ${#VARIANTS[@]}; i++)); do ALL+=("${i}"); done
fi
MINE=()
for ((i = TASK; i < ${#ALL[@]}; i += NTASK)); do MINE+=("$(printf '%02d' $((10#${ALL[$i]})))"); done
echo "task ${TASK}/${NTASK}: cells ${MINE[*]}"
mkdir -p "${ROOT}"

FAILED=0
sidlens_gate

prepare_cell() {   # $1 = cell; provenance once per cell
    local out="${ROOT}/cell-$1"
    mkdir "${out}"
    printf 'cell=%s variant=%s job=%s task=%s started=%s\n' "$1" "${VARIANTS[$((10#$1))]}" "${RUN_GROUP}" \
        "${TASK}" "$(date -Is)" > "${out}/status.txt"
    sidlens_provenance_stamp "${out}/.provenance"
    {
        git ls-files -z src/sidlens vendor/onediffrec
        find src/sidlens/interventions src/sidlens/hooks src/sidlens/models "${EXP}" \
            experiments/controlled/exp3_ar_history_patching experiments/controlled/exp4_ar_copy_circuit \
            experiments/controlled/exp6_ar_copy_in_decoding -type f \( -name '*.py' -o -name '*.md' \) -print0
        find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
        printf '%s\0' scripts/common/_preamble.sh scripts/controlled/ar_copy_grid.sbatch pyproject.toml
    } | sort -zu > "${out}/source-files.nul"
    xargs -0 sha256sum < "${out}/source-files.nul" > "${out}/source.sha256"
    tar -czf "${out}/source.tar.gz" --null -T "${out}/source-files.nul"
    cp "${EXP}/protocol.md" "${out}/protocol.md"
    git status --porcelain=v1 --untracked-files=all > "${out}/git-status.txt"
    git diff --binary HEAD > "${out}/tracked-changes.diff"
    {
        date -Is; hostname; uname -a; "${PYBIN}" -VV
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null || echo "no GPU"
    } > "${out}/environment.txt"
    "${PYBIN}" -m pip freeze --local > "${out}/requirements.freeze.txt"
}

stage() {   # $1 = cell, $2 = stage name, rest = run.py and its arguments
    local cell="$1" name="$2" script="$3"; shift 3
    local out="${ROOT}/cell-${cell}/${name}" ckpt="retrain/${RUN_NAME}/${VARIANTS[$((10#${cell}))]}"
    local t0=$(date +%s)
    mkdir "${out}"
    if "${PYBIN}" -u "${script}" --ckpt "${ckpt}" --out "${out}" "$@" ${LIMIT[@]+"${LIMIT[@]}"} \
            > "${out}/stdout.log" 2>&1; then
        (cd "${out}" && find . -type f ! -name output.sha256 -print0 | sort -z | xargs -0 sha256sum > output.sha256)
        printf 'complete stage=%s seconds=%s finished=%s\n' "${name}" $(( $(date +%s) - t0 )) "$(date -Is)" \
            >> "${ROOT}/cell-${cell}/status.txt"
    else
        FAILED=$((FAILED + 1))
        printf 'failed stage=%s seconds=%s finished=%s (see %s/stdout.log)\n' "${name}" $(( $(date +%s) - t0 )) \
            "$(date -Is)" "${out}" >> "${ROOT}/cell-${cell}/status.txt"
    fi
}

for c in "${MINE[@]}"; do prepare_cell "${c}"; done
for c in "${MINE[@]}"; do
    echo "$(date -Is) cell ${c} pass 1"
    stage "${c}" exp4_valid experiments/controlled/exp4_ar_copy_circuit/run.py --split valid --parts R,K
    stage "${c}" exp3_A experiments/controlled/exp3_ar_history_patching/run.py --parts A
    stage "${c}" exp4_test experiments/controlled/exp4_ar_copy_circuit/run.py --split test --parts R,K
done
for c in "${MINE[@]}"; do
    echo "$(date -Is) cell ${c} pass 2"
    stage "${c}" exp6 experiments/controlled/exp6_ar_copy_in_decoding/run.py
done
echo "$(date -Is) done: ${FAILED} failed stages"
exit $(( FAILED > 0 ))
