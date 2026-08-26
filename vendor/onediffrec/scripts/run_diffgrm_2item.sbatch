#!/bin/bash
#SBATCH --job-name=onediffrec-diffgrm2
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Masked-diffusion next-2-item baseline, for the "Inside Diff, Outside AR" rows
# of the research notes (pages 15-16).
#
# Differs from scripts/run_diffgrm.sbatch in four ways:
#   * runs against DiffGRM_new, the only tree carrying multi-target support
#     (genrec/dataset.py n_target_items, model.py n_target_digits),
#   * --n_target_items=2 so the decoder emits two concatenated SIDs,
#   * --two_pass_inference=True, which is what "Inside Diff, Outside AR" means:
#     pass 1 diffuses item 1, pass 2 re-diffuses item 2 on the augmented
#     history, so generation is autoregressive across items and diffusive
#     within each SID,
#   * --external_sem_ids takes a BARE NAME here, not a path. DiffGRM_new's
#     tokenizer joins it against its own cache dir and appends the suffix
#     (tokenizer.py: os.path.join(cache_dir, f'{external_name}.sem_ids')),
#     whereas the older tree read the config value as a full path.
#
# Everything else - hyperparameters, guided masking, external SIDs - matches the
# next-item runner so the two tasks stay comparable.
#
# Submit with:
#   sbatch --export=ALL,VARIANT=rqvae_3codebook_256,N_DIGIT=3 scripts/run_diffgrm_2item.sbatch
#
# The GPU QOS allows only 2 running and 4 submitted jobs per user, so variants
# cannot all be queued up front. Point LANE at a file holding one variant per
# line and each job pops the next line and submits it after finishing, so one
# chain keeps exactly one queue slot busy until its file is empty.
#
# A variant that fails stops its own chain (nothing is popped), leaving the file
# as the record of what still has to run.

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
DIFFGRM=${REPO}/DiffGRM_new

CATEGORY=${CATEGORY:-Industrial_and_Scientific}
VARIANT=${VARIANT:?set VARIANT, e.g. rqvae_3codebook_256}
N_DIGIT=${N_DIGIT:?set N_DIGIT to the codebook count, e.g. 3}
# DIFF_GRM takes codebook_size from config (default 256) rather than inferring it
# from the external sem_ids, and sizes its vocabulary as 3 + n_digit*codebook_size.
CODEBOOK_SIZE=${CODEBOOK_SIZE:-${VARIANT##*_}}

LANE=${LANE:-}
EXCLUDE_NODES=${EXCLUDE_NODES:-}

# gpu-50 comes up without the /l/users mount; anything writing to WORK dies with
# "cannot create directory '/l'". Probe first and bounce the job off that node,
# carrying the rest of the chain with it.
probe=${WORK}/.mount_probe.${SLURM_JOB_ID:-$$}
if ! mkdir -p "${WORK}" 2>/dev/null || ! touch "${probe}" 2>/dev/null; then
    this_node=$(hostname -s)
    echo "FATAL: ${WORK} is not writable on ${this_node} (missing mount)" >&2
    if [[ -n "${EXCLUDE_NODES}" ]]; then
        EXCLUDE_NODES="${EXCLUDE_NODES},${this_node}"
    else
        EXCLUDE_NODES="${this_node}"
    fi
    echo "resubmitting ${VARIANT} with --exclude=${EXCLUDE_NODES}" >&2
    sbatch --exclude="${EXCLUDE_NODES}" \
        --export=ALL,VARIANT="${VARIANT}",N_DIGIT="${N_DIGIT}",LANE="${LANE}",EXCLUDE_NODES="${EXCLUDE_NODES}" \
        "${REPO}/scripts/run_diffgrm_2item.sbatch"
    exit 1
fi
rm -f "${probe}"

SEM_IDS=${DIFFGRM}/cache/AmazonReviews2014/${CATEGORY}/processed/${VARIANT}.sem_ids
OUT_DIR=${WORK}/diffgrm_2item/${CATEGORY}
LOG=${OUT_DIR}/${VARIANT}.txt

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${WORK}/cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

[[ -f "${SEM_IDS}" ]] || { echo "missing sem_ids: ${SEM_IDS}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

# genrec derives its checkpoint directory from a timestamp accurate to the second
# (genrec/models/DIFF_GRM/trainer.py builds saved/<dataset>_<time>/pytorch_model.bin),
# so two jobs starting in the same second share one file and overwrite each
# other's weights. "saved" is relative to the working directory, so give each job
# its own directory and link the code and cache into it.
RUNDIR=${WORK}/diffgrm_2item/run/${SLURM_JOB_ID}
mkdir -p "${RUNDIR}"
ln -sfn "${DIFFGRM}/genrec"  "${RUNDIR}/genrec"
ln -sfn "${DIFFGRM}/main.py" "${RUNDIR}/main.py"
ln -sfn "${DIFFGRM}/cache"   "${RUNDIR}/cache"
cd "${RUNDIR}"
echo "rundir=${RUNDIR}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "category=${CATEGORY} variant=${VARIANT} n_digit=${N_DIGIT} codebook_size=${CODEBOOK_SIZE} n_target_items=2"
echo "sem_ids=${SEM_IDS}"
echo "log=${LOG}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

"${ENV_PREFIX}/bin/python" -c "import json,sys; d=json.load(open('${SEM_IDS}')); print(f'sem_ids: {len(d)} items, {len(next(iter(d.values())))} digits')"

srun --ntasks=1 --kill-on-bad-exit=1 "${ENV_PREFIX}/bin/python" main.py \
  --category="${CATEGORY}" \
  --train_batch_size=256 \
  --model=DIFF_GRM \
  --n_digit="${N_DIGIT}" \
  --codebook_size="${CODEBOOK_SIZE}" \
  --masking_strategy=guided \
  --guided_refresh_each_step=false \
  --guided_select=least \
  --guided_conf_metric=msp \
  --encoder_n_layer=1 \
  --decoder_n_layer=4 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --train_sliding=true \
  --min_hist_len=2 \
  --eval_start_epoch=20 \
  --lr=0.003 \
  --label_smoothing=0.1 \
  --sid_quantizer=external \
  --external_sem_ids="${VARIANT}" \
  --metadata=none \
  --force_regenerate_opq=false \
  --share_decoder_output_embedding=true \
  --n_target_items=2 \
  --two_pass_inference=True 2>&1 | tee "${LOG}"

echo "[$(date --iso-8601=seconds)] diffgrm-2item ${VARIANT} complete -> ${LOG}"

# Hand the next variant to a fresh job. Only on success: a failing variant should
# stop its own chain rather than burn the queue slot on everything after it.
if [[ -n "${LANE}" && -s "${LANE}" ]]; then
    # Pop the first line in pure bash. The GPU submit filter rejects several
    # log-following and queue-polling commands, so this avoids external tools.
    mapfile -t lane_variants < "${LANE}"
    NEXT=${lane_variants[0]}
    remaining=$(( ${#lane_variants[@]} - 1 ))
    if (( remaining > 0 )); then
        printf '%s\n' "${lane_variants[@]:1}" >| "${LANE}.tmp"
    else
        : >| "${LANE}.tmp"
    fi
    mv -f "${LANE}.tmp" "${LANE}"
    NEXT_DIGIT=$(sed -E 's/.*_([0-9]+)codebook_.*/\1/' <<<"${NEXT}")
    echo "[$(date --iso-8601=seconds)] chaining ${NEXT} (n_digit=${NEXT_DIGIT}); ${remaining} left in ${LANE}"
    sbatch ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
        --export=ALL,VARIANT="${NEXT}",N_DIGIT="${NEXT_DIGIT}",LANE="${LANE}",EXCLUDE_NODES="${EXCLUDE_NODES}" \
        "${REPO}/scripts/run_diffgrm_2item.sbatch"
elif [[ -n "${LANE}" ]]; then
    echo "[$(date --iso-8601=seconds)] lane ${LANE} drained"
fi
