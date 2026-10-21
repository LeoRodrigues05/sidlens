# Run one experiment's run.py over several cells inside ONE job, sequentially.
#
# Why: the CIAI GPU QoS accepts at most four submitted jobs per user, so a
# 20-cell array cannot be submitted at once. One job walks the cells and writes
# the usual <run-root>/cell-XX/ directories. The source archive, provenance and
# environment are recorded once at the run root (every cell ran the same code).
#
# Usage, after sourcing _preamble.sh and passing sidlens_gate:
#   sidlens_run_cells <experiment dir> <run root> "<cells, e.g. 0-19 or 6,14>" [run.py args...]
# SIDLENS_PARALLEL=N runs N cells at once, splitting SLURM_CPUS_PER_TASK between them (CPU runs).
# A cell is complete only if its validation.json exists (a cancelled job's EXIT
# trap sees rc=0, which is how older wrappers marked empty cells complete).

sidlens_expand_cells() {
    local spec="$1" part out=()
    IFS=',' read -ra parts <<< "${spec}"
    for part in "${parts[@]}"; do
        if [[ "${part}" == *-* ]]; then
            out+=($(seq "${part%-*}" "${part#*-}"))
        else
            out+=("${part}")
        fi
    done
    echo "${out[@]}"
}

sidlens_run_cells() {
    local exp="$1" root="$2" spec="$3"
    shift 3
    mkdir -p "$(dirname "${root}")"
    mkdir "${root}"                              # never reuse a run directory
    sidlens_provenance_stamp "${root}/.provenance"
    {
        git ls-files -z src/sidlens vendor/onediffrec
        find src/sidlens "${exp}" -type f \( -name '*.py' -o -name '*.md' \) -print0
        find manifests -maxdepth 1 -type f \( -name '*.json' -o -name CURRENT \) -print0
        printf '%s\0' scripts/common/_preamble.sh scripts/common/_cells.sh \
            experiments/controlled/exp1_matched_beam/validate.py pyproject.toml
        if [[ -n "${SIDLENS_WRAPPER:-}" ]]; then printf "%s\0" "${SIDLENS_WRAPPER}"; fi
    } | sort -zu > "${root}/source-files.nul"
    xargs -0 sha256sum < "${root}/source-files.nul" > "${root}/source.sha256"
    tar -czf "${root}/source.tar.gz" --null -T "${root}/source-files.nul"
    cp "${exp}/protocol.md" "${root}/protocol.md"
    git status --porcelain=v1 --untracked-files=all > "${root}/git-status.txt"
    git diff --binary HEAD > "${root}/tracked-changes.diff"
    {
        date -Is; hostname; uname -a; "${PYBIN}" -VV
        nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv 2>/dev/null || echo "no GPU"
    } > "${root}/environment.txt"
    # pip freeze hung for over an hour on gpu-02 (job 312973, 2026-10-14) and the job never reached a cell.
    timeout 300 "${PYBIN}" -m pip freeze --local > "${root}/requirements.freeze.txt" \
        || echo "pip freeze did not finish within 300 s" >> "${root}/requirements.freeze.txt"

    local cell failed=0 parallel="${SIDLENS_PARALLEL:-1}"
    if [[ "${parallel}" -gt 1 ]]; then
        # Several cells at once, each with its share of the allocation's cores and its own log
        # (<run root>/log-cell-XX.txt). The CPU QoS caps jobs per user, not cores per job.
        local threads=$(( ${SLURM_CPUS_PER_TASK:-${parallel}} / parallel ))
        (( threads < 1 )) && threads=1
        export OMP_NUM_THREADS="${threads}" MKL_NUM_THREADS="${threads}"
        local pids=() cells=()
        for cell in $(sidlens_expand_cells "${spec}"); do
            while (( $(jobs -rp | wc -l) >= parallel )); do wait -n || true; done
            _sidlens_one_cell "${exp}" "${root}" "${cell}" "$@" > "${root}/log-cell-$(printf "%02d" "${cell}").txt" 2>&1 &
            pids+=($!); cells+=("${cell}")
        done
        local i
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" || failed=$((failed + 1))
        done
    else
        for cell in $(sidlens_expand_cells "${spec}"); do
            _sidlens_one_cell "${exp}" "${root}" "${cell}" "$@" || failed=$((failed + 1))
        done
    fi
    echo "[cells] ${failed} failed"
    return "${failed}"
}

_sidlens_one_cell() {
    local exp="$1" root="$2" cell="$3" out rc
    shift 3
    out="${root}/cell-$(printf '%02d' "${cell}")"
    mkdir "${out}"
    printf 'running cell=%s started=%s\n' "${cell}" "$(date -Is)" > "${out}/status.txt"
    set +e
    "${PYBIN}" -u "${exp}/run.py" --cell "${cell}" --out "${out}" "$@"
    rc=$?
    set -e
    if [[ ${rc} -eq 0 && -f "${out}/validation.json" ]]; then
        (cd "${out}" && find . -type f ! -name output.sha256 ! -name status.txt -print0 \
            | sort -z | xargs -0 sha256sum > output.sha256)
        printf 'complete cell=%s finished=%s\n' "${cell}" "$(date -Is)" > "${out}/status.txt"
        return 0
    fi
    printf 'failed cell=%s exit=%s finished=%s\n' "${cell}" "${rc}" "$(date -Is)" > "${out}/status.txt"
    return 1
}
