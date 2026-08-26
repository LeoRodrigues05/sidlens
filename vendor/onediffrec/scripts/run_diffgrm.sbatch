#!/bin/bash
#SBATCH --job-name=onediffrec-diffgrm
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Masked-diffusion (DiffGRM) baseline, for the "Mask Diffusion" rows of the
# research notes.
#
# Hyperparameters are copied verbatim from DiffGRM/run_industrial.sh so results
# stay comparable with the DiffGRM rows already in the notes. Only three things
# change: the two hardcoded paths from the original authors' machine
# (/u/dnian LD_LIBRARY_PATH and /data1/dnian cache), and the GPU index.
#
# Submit with:
#   sbatch --export=ALL,VARIANT=rqvae_3codebook_512,N_DIGIT=3 scripts/run_diffgrm.sbatch
#   sbatch --export=ALL,VARIANT=rqvae_4codebook_512,N_DIGIT=4 scripts/run_diffgrm.sbatch
#
# The GPU QOS allows only 2 running and 4 submitted jobs per user, so variants
# cannot all be queued up front. Point LANE at a file holding one variant per
# line and each job pops the next line and submits it after finishing, so one
# chain keeps exactly one queue slot busy until its file is empty:
#
#   printf '%s\n' MQ_5codebook_128 rqvae_3codebook_128 > lane0.txt
#   sbatch --export=ALL,VARIANT=MQ_3codebook_128,N_DIGIT=3,LANE=lane0.txt ...
#
# A variant that fails stops its own chain (nothing is popped), leaving the file
# as the record of what still has to run.

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
DIFFGRM=${REPO}/DiffGRM

CATEGORY=${CATEGORY:-Industrial_and_Scientific}
VARIANT=${VARIANT:?set VARIANT, e.g. rqvae_3codebook_512}
N_DIGIT=${N_DIGIT:?set N_DIGIT to the codebook count, e.g. 3}
# DIFF_GRM takes codebook_size from config (default 256) rather than inferring it
# from the external sem_ids, and sizes its vocabulary as 3 + n_digit*codebook_size.
# Feeding it 512-codebook ids under the default trips an assertion, so derive the
# size from the variant name and pass it explicitly.
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
        "${REPO}/scripts/run_diffgrm.sbatch"
    exit 1
fi
rm -f "${probe}"

SEM_IDS=${DIFFGRM}/cache/AmazonReviews2014/${CATEGORY}/processed/${VARIANT}.sem_ids
OUT_DIR=${WORK}/diffgrm/${CATEGORY}
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
RUNDIR=${WORK}/diffgrm/run/${SLURM_JOB_ID}
mkdir -p "${RUNDIR}"
ln -sfn "${DIFFGRM}/genrec"  "${RUNDIR}/genrec"
ln -sfn "${DIFFGRM}/main.py" "${RUNDIR}/main.py"
ln -sfn "${DIFFGRM}/cache"   "${RUNDIR}/cache"
cd "${RUNDIR}"
echo "rundir=${RUNDIR}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "category=${CATEGORY} variant=${VARIANT} n_digit=${N_DIGIT} codebook_size=${CODEBOOK_SIZE}"
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
  --external_sem_ids="${SEM_IDS}" \
  --metadata=none \
  --force_regenerate_opq=false \
  --share_decoder_output_embedding=true 2>&1 | tee "${LOG}"

echo "[$(date --iso-8601=seconds)] diffgrm ${VARIANT} complete -> ${LOG}"

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
    # This job still counts against MaxSubmitJobsPerUser while it submits its
    # successor, so keep at most one chain in flight alongside the sweep: 3
    # submitted steady-state, 4 for the moment of hand-off, which is the cap.
    # Retrying here is not an option - the GPU submit filter also rejects the
    # scripts that would wait between attempts, so a refused hand-off leaves the
    # lane file as the record (queue_remaining.py lists whatever did not run).
    sbatch ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
        --export=ALL,VARIANT="${NEXT}",N_DIGIT="${NEXT_DIGIT}",LANE="${LANE}",EXCLUDE_NODES="${EXCLUDE_NODES}" \
        "${REPO}/scripts/run_diffgrm.sbatch"
elif [[ -n "${LANE}" ]]; then
    echo "[$(date --iso-8601=seconds)] lane ${LANE} drained"
fi
