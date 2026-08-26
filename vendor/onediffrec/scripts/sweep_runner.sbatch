#!/bin/bash
#SBATCH --job-name=onediffrec-sweep
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Walks a manifest of dataset variants, training and scoring each one in turn.
#
# The cluster caps a job at 24h, which is shorter than the whole sweep and can
# be shorter than a single Office run. Rather than letting the wall clock kill
# work in progress, this script:
#   * budgets each training call with `timeout` so it stops cleanly with a
#     checkpoint on disk before the job's own deadline,
#   * resubmits itself and exits when it runs out of time,
#   * resumes the interrupted variant from its last checkpoint on the next run.
#
# State lives on disk, so a resubmitted job (or a manual one) picks up exactly
# where the previous one stopped.
#
# Submit with:
#   sbatch --export=ALL,PHASE=next-item scripts/sweep_runner.sbatch
# Optional overrides: MANIFEST, RUN_MODE=smoke, MAX_VARIANTS, NEXT_PHASE.

set -uo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
DATA_ROOT=${WORK}/data
MODEL_DIR=${WORK}/models/Qwen2.5-1.5B-Instruct

PHASE=${PHASE:-next-item}
RUN_MODE=${RUN_MODE:-full}
SWEEP=${WORK}/sweep/${PHASE}
MANIFEST=${MANIFEST:-${SWEEP}/manifest.tsv}
STATE_DIR=${SWEEP}/state
RUNS_DIR=${SWEEP}/runs
METRICS_DIR=${SWEEP}/metrics
BEST_DIR=${SWEEP}/best
MAX_VARIANTS=${MAX_VARIANTS:-0}          # 0 = no limit
NEXT_PHASE=${NEXT_PHASE:-}               # phase to submit once this one finishes

# Leave room to finish eval bookkeeping, prune and resubmit before SLURM kills us.
RESERVE_SECONDS=${RESERVE_SECONDS:-2400}
# Don't start a fresh variant unless there is at least this much time left.
MIN_START_SECONDS=${MIN_START_SECONDS:-3600}

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export WANDB_MODE=offline
export WANDB_DIR=${WORK}/wandb
export HF_HOME=${WORK}/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DATA_ROOT

cd "${REPO}"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

log "phase=${PHASE} mode=${RUN_MODE} host=$(hostname) job=${SLURM_JOB_ID:-none}"

# Some compute nodes come up without the /l/users mount. Detect that before doing
# any work and hand the job to a different node, otherwise every path silently
# fails and the sweep state gets corrupted.
EXCLUDE_NODES=${EXCLUDE_NODES:-}
probe=${WORK}/.mount_probe.${SLURM_JOB_ID:-$$}
if ! mkdir -p "${WORK}" 2>/dev/null || ! touch "${probe}" 2>/dev/null; then
    this_node=$(hostname -s)
    log "FATAL: ${WORK} is not writable on ${this_node} (missing mount)"
    if [[ -n "${EXCLUDE_NODES}" ]]; then
        EXCLUDE_NODES="${EXCLUDE_NODES},${this_node}"
    else
        EXCLUDE_NODES="${this_node}"
    fi
    log "resubmitting with --exclude=${EXCLUDE_NODES}"
    sbatch --parsable --time="${SBATCH_TIME:-24:00:00}" --exclude="${EXCLUDE_NODES}" \
        --export=ALL,PHASE="${PHASE}",RUN_MODE="${RUN_MODE}",MANIFEST="${MANIFEST}",NEXT_PHASE="${NEXT_PHASE}",TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-86400}",SBATCH_TIME="${SBATCH_TIME:-24:00:00}",EXCLUDE_NODES="${EXCLUDE_NODES}" \
        "${REPO}/scripts/sweep_runner.sbatch"
    exit 1
fi
rm -f "${probe}"
if [[ ! -d "${DATA_ROOT}/Amazon18" ]]; then
    log "FATAL: dataset root ${DATA_ROOT}/Amazon18 not visible"
    exit 1
fi

mkdir -p "${STATE_DIR}" "${RUNS_DIR}" "${METRICS_DIR}" "${BEST_DIR}" "${WANDB_DIR}" "${REPO}/slurm_logs"

# ---------------------------------------------------------------- manifest ---
if [[ ! -f "${MANIFEST}" ]]; then
    log "building manifest at ${MANIFEST}"
    mkdir -p "$(dirname "${MANIFEST}")"
    "${ENV_PREFIX}/bin/python" "${REPO}/scripts/variants.py" \
        --tree "${PHASE}" --order cheap-first --write-manifest "${MANIFEST}" || exit 1
fi
log "manifest: ${MANIFEST} ($(( $(wc -l < "${MANIFEST}") - 1 )) variants)"

# ------------------------------------------------------------ time helpers ---
# Remaining walltime is tracked locally from the job's own clock rather than by
# polling the scheduler, so the runner makes no monitoring calls.
JOB_START_EPOCH=$(date +%s)
TIME_LIMIT_SECONDS=${TIME_LIMIT_SECONDS:-86400}
SBATCH_TIME=${SBATCH_TIME:-24:00:00}

job_seconds_left() {
    local now
    now=$(date +%s)
    echo $(( TIME_LIMIT_SECONDS - (now - JOB_START_EPOCH) ))
}

resubmit_self() {
    local reason=$1
    log "resubmitting (${reason})"
    local jid rc
    local exclude_arg=()
    [[ -n "${EXCLUDE_NODES}" ]] && exclude_arg=(--exclude="${EXCLUDE_NODES}")
    jid=$(sbatch --parsable --time="${SBATCH_TIME}" "${exclude_arg[@]}" \
        --export=ALL,PHASE="${PHASE}",RUN_MODE="${RUN_MODE}",MANIFEST="${MANIFEST}",NEXT_PHASE="${NEXT_PHASE}",TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS}",SBATCH_TIME="${SBATCH_TIME}",EXCLUDE_NODES="${EXCLUDE_NODES}" \
        "${REPO}/scripts/sweep_runner.sbatch" 2>&1)
    rc=$?
    if (( rc == 0 )); then
        log "successor job ${jid}"
    else
        log "RESUBMIT FAILED: ${jid}"
        return 1
    fi
}

state_of()  { cat "${STATE_DIR}/$1" 2>/dev/null || echo pending; }
set_state() { echo "$2" > "${STATE_DIR}/$1"; log "  state[$1] -> $2"; }

# ------------------------------------------------------------------- loop ----
processed=0

# The manifest is read on fd 3, not stdin: srun/torchrun inherit stdin and will
# swallow the remaining lines, silently ending the loop after one variant.
while read -r variant_id tree category method codebooks size <&3; do
    [[ "${variant_id}" == "variant_id" ]] && continue      # header
    [[ -z "${variant_id}" ]] && continue

    status=$(state_of "${variant_id}")
    [[ "${status}" == "done" || "${status}" == "failed" ]] && continue

    if (( MAX_VARIANTS > 0 && processed >= MAX_VARIANTS )); then
        log "hit MAX_VARIANTS=${MAX_VARIANTS}, stopping"
        break
    fi

    left=$(job_seconds_left)
    if (( left < MIN_START_SECONDS )); then
        resubmit_self "only ${left}s left, next variant needs ${MIN_START_SECONDS}s"
        exit 0
    fi

    OUTPUT_DIR=${RUNS_DIR}/${variant_id}
    mkdir -p "${OUTPUT_DIR}"
    log "=== ${variant_id} (status=${status}, ${left}s left) ==="

    # ------------------------------------------------------------- train ----
    if [[ "${status}" == "pending" || "${status}" == "training" ]]; then
        set_state "${variant_id}" training

        "${ENV_PREFIX}/bin/python" "${REPO}/scripts/validate_sft_assets.py" \
            --data-root "${DATA_ROOT}" --tree "${tree}" --category "${category}" \
            --method "${method}" --codebooks "${codebooks}" --codebook-size "${size}" \
            --skip-rows > "${METRICS_DIR}/${variant_id}.assets.json"
        if [[ $? -ne 0 ]]; then
            log "  asset validation failed"
            set_state "${variant_id}" failed
            continue
        fi

        read -r TRAIN_FILE VALID_FILE INDEX_FILE ITEM_FILE < <(
            "${ENV_PREFIX}/bin/python" - "$tree" "$category" "$method" "$codebooks" "$size" "$DATA_ROOT" <<'PY'
import sys
sys.path.insert(0, "/home/leo.rodrigues/onediffrec/OneDiffRec/scripts")
import variants as V
tree, category, method, codebooks, size, data_root = sys.argv[1:7]
p = V.resolve(data_root, V.Variant(tree, category, method, int(codebooks), int(size)))
print(p["train"], p["valid"], p["index"], p["item"])
PY
        )
        if [[ -z "${TRAIN_FILE:-}" || -z "${VALID_FILE:-}" || -z "${INDEX_FILE:-}" || -z "${ITEM_FILE:-}" ]]; then
            log "  could not resolve dataset paths"
            set_state "${variant_id}" failed
            continue
        fi

        if [[ "${RUN_MODE}" == "smoke" ]]; then
            SAMPLE=64; EPOCHS=1; GLOBAL_BATCH=32; MICRO_BATCH=8
        else
            SAMPLE=-1; EPOCHS=${EPOCHS:-10}; GLOBAL_BATCH=${GLOBAL_BATCH:-1024}; MICRO_BATCH=${MICRO_BATCH:-8}
        fi
        # Derive the rank count from what SLURM actually handed us. Hardcoding 4
        # here (and in --nproc_per_node below) made job 171602 launch 4 ranks on a
        # node where a concurrent 1-GPU job held one of the four devices; ranks 0
        # and 3 landed on the same GPU and NCCL aborted with "Duplicate GPU
        # detected". GRAD_ACCUM compensates so the effective global batch stays
        # ${GLOBAL_BATCH} regardless, keeping runs comparable across jobs.
        WORLD_SIZE=${SLURM_GPUS_ON_NODE:-}
        if [[ -z "${WORLD_SIZE}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            IFS=',' read -ra _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
            WORLD_SIZE=${#_gpu_ids[@]}
        fi
        WORLD_SIZE=${WORLD_SIZE:-4}
        log "  world size ${WORLD_SIZE} (SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset})"
        GRAD_ACCUM=$(( GLOBAL_BATCH / MICRO_BATCH / WORLD_SIZE ))
        if (( GRAD_ACCUM < 1 || GLOBAL_BATCH % (MICRO_BATCH * WORLD_SIZE) != 0 )); then
            log "  invalid batch geometry g=${GLOBAL_BATCH} m=${MICRO_BATCH} w=${WORLD_SIZE}"
            set_state "${variant_id}" failed
            continue
        fi

        # Resume from the newest checkpoint if a previous job was cut short.
        # Picked with pure bash: the submit filter rejects `tail` on GPU partitions.
        RESUME_ARG=()
        latest=""
        latest_step=-1
        for candidate in "${OUTPUT_DIR}"/checkpoint-*; do
            [[ -d "${candidate}" ]] || continue
            step=${candidate##*/checkpoint-}
            [[ "${step}" =~ ^[0-9]+$ ]] || continue
            if (( step > latest_step )); then
                latest_step=${step}
                latest=${candidate}
            fi
        done
        if [[ -n "${latest}" && -f "${latest}/trainer_state.json" ]]; then
            log "  resuming from ${latest}"
            RESUME_ARG=(--resume_from_checkpoint "${latest}")
        fi

        left=$(job_seconds_left)
        BUDGET=$(( left - RESERVE_SECONDS ))
        log "  training budget ${BUDGET}s (grad_accum=${GRAD_ACCUM})"

        train_start=$(date +%s)
        # Launch geometry matches scripts/run_sft.sbatch (the job that produced
        # 160793) exactly, so results stay comparable across the sweep.
        timeout --signal=TERM --kill-after=120 "${BUDGET}" \
            srun --ntasks=1 --kill-on-bad-exit=1 \
            "${ENV_PREFIX}/bin/torchrun" --standalone --nproc_per_node="${WORLD_SIZE}" \
            "${REPO}/sft.py" \
            --base_model "${MODEL_DIR}" \
            --train_file "${TRAIN_FILE}" \
            --eval_file "${VALID_FILE}" \
            --output_dir "${OUTPUT_DIR}" \
            --sample "${SAMPLE}" \
            --batch_size "${GLOBAL_BATCH}" \
            --micro_batch_size "${MICRO_BATCH}" \
            --num_epochs "${EPOCHS}" \
            --learning_rate 3e-4 \
            --cutoff_len 512 \
            --category "${category}" \
            --seed 42 \
            --sid_index_path "${INDEX_FILE}" \
            --item_meta_path "${ITEM_FILE}" \
            --train_from_scratch False \
            --freeze_LLM False \
            --save_total_limit 2 \
            --wandb_project "OneDiffRec_${PHASE}" \
            --wandb_run_name "${variant_id}" \
            "${RESUME_ARG[@]}"
        rc=$?
        train_elapsed=$(( $(date +%s) - train_start ))
        echo "${train_elapsed}" >> "${STATE_DIR}/${variant_id}.train_seconds"

        if (( rc == 124 || rc == 137 )); then
            log "  hit training budget after ${train_elapsed}s; checkpoint retained"
            resubmit_self "variant ${variant_id} needs more time"
            exit 0
        elif (( rc != 0 )); then
            log "  training failed rc=${rc}"
            set_state "${variant_id}" failed
            continue
        fi

        if [[ ! -f "${OUTPUT_DIR}/final_checkpoint/config.json" ]]; then
            log "  training returned 0 but final_checkpoint is missing"
            set_state "${variant_id}" failed
            continue
        fi

        # The top-level copy duplicates final_checkpoint and is charged ~46 GB of
        # quota against a 2.9 GB file, so drop it as soon as training succeeds.
        rm -f "${OUTPUT_DIR}/model.safetensors"
        set_state "${variant_id}" trained
        status=trained
    fi

    # -------------------------------------------------------------- eval ----
    if [[ "${status}" == "trained" ]]; then
        log "  evaluating"
        eval_start=$(date +%s)
        bash "${REPO}/scripts/eval_variant.sh" \
            "${tree}" "${category}" "${method}" "${codebooks}" "${size}" \
            "${OUTPUT_DIR}/final_checkpoint" "${OUTPUT_DIR}" \
            "${METRICS_DIR}/${variant_id}.predictions.json" \
            > "${METRICS_DIR}/${variant_id}.eval.log" 2>&1
        rc=$?
        eval_elapsed=$(( $(date +%s) - eval_start ))
        if (( rc != 0 )); then
            log "  eval failed rc=${rc}; see ${METRICS_DIR}/${variant_id}.eval.log"
            # awk stands in for `tail`, which the GPU submit filter rejects.
            awk 'BEGIN{n=20} {buf[NR%n]=$0} END{for(i=(NR>n?NR-n+1:1);i<=NR;i++) print buf[i%n]}' \
                "${METRICS_DIR}/${variant_id}.eval.log"
            set_state "${variant_id}" failed
            continue
        fi
        log "  eval done in ${eval_elapsed}s"

        "${ENV_PREFIX}/bin/python" "${REPO}/scripts/collect_metrics.py" \
            --variant-id "${variant_id}" \
            --output-dir "${OUTPUT_DIR}" \
            --metrics-dir "${METRICS_DIR}" \
            --train-seconds-file "${STATE_DIR}/${variant_id}.train_seconds" \
            --eval-seconds "${eval_elapsed}" || {
                log "  metric collection failed"; set_state "${variant_id}" failed; continue; }

        set_state "${variant_id}" evaluated
        status=evaluated
    fi

    # ------------------------------------------------------------- prune ----
    if [[ "${status}" == "evaluated" ]]; then
        rm -rf "${OUTPUT_DIR}"/checkpoint-*
        rm -f "${OUTPUT_DIR}/model.safetensors"

        # Keep only the best checkpoint per (tree, category); metrics are kept for all.
        keep=$("${ENV_PREFIX}/bin/python" "${REPO}/scripts/collect_metrics.py" \
                   --metrics-dir "${METRICS_DIR}" --is-best "${variant_id}")
        if [[ "${keep}" == "best" ]]; then
            log "  best so far for ${category}; retaining checkpoint"
            rm -rf "${BEST_DIR}/${category}"
            mkdir -p "${BEST_DIR}"
            mv "${OUTPUT_DIR}/final_checkpoint" "${BEST_DIR}/${category}"
            echo "${variant_id}" > "${BEST_DIR}/${category}.variant"
        fi
        rm -rf "${OUTPUT_DIR}/final_checkpoint"
        find "${OUTPUT_DIR}" -maxdepth 1 -type f -delete 2>/dev/null

        set_state "${variant_id}" done
        processed=$(( processed + 1 ))
        log "  quota: $(lfs quota -h -u "${USER}" /l/users 2>/dev/null | awk 'NR==3{print $2}')"
    fi
done 3< "${MANIFEST}"

# ------------------------------------------------------------- wrap up ------
remaining=0
while read -r variant_id _ <&3; do
    [[ "${variant_id}" == "variant_id" || -z "${variant_id}" ]] && continue
    s=$(state_of "${variant_id}")
    [[ "${s}" == "done" || "${s}" == "failed" ]] || remaining=$(( remaining + 1 ))
done 3< "${MANIFEST}"

if (( remaining > 0 )); then
    resubmit_self "${remaining} variants still outstanding"
else
    log "phase ${PHASE} complete"
    if [[ -n "${NEXT_PHASE}" ]]; then
        log "submitting next phase ${NEXT_PHASE}"
        sbatch --export=ALL,PHASE="${NEXT_PHASE}",RUN_MODE="${RUN_MODE}" \
            "${REPO}/scripts/sweep_runner.sbatch"
    fi
fi

log "runner exiting (processed ${processed} variants this job)"
