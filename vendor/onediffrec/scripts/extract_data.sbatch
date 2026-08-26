#!/bin/bash
#SBATCH --job-name=onediffrec-extract
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Extracts the FULL dataset archive (both task trees) so the sweep can cover
# every variant. Supersedes the 6-member extraction in setup_sft.sbatch, which
# only unpacked Industrial rqvae_4codebook_128.
#
# No GPU needed - this is tar and a path check.

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
ARCHIVE=${REPO}/data/OneDiffRec_data.tar.gz
DATA_ROOT=${WORK}/data
ARCHIVE_SHA256=c3ab0bedf84bd52cfc79f281e358518792d01f96e1a58ab68bbdd56c8374ce89

mkdir -p "${DATA_ROOT}"
cd "${REPO}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"

echo "--- verifying archive checksum ---"
echo "${ARCHIVE_SHA256}  ${ARCHIVE}" | sha256sum --check --strict

echo "--- extracting full archive to ${DATA_ROOT} ---"
# Archive members are rooted at data/, so -C ${WORK} yields ${WORK}/data/...
tar -xzf "${ARCHIVE}" -C "${WORK}"

echo "--- extracted tree ---"
du -sh "${DATA_ROOT}"/* 2>/dev/null
for tree in Amazon18 two-item; do
    for split in train valid test info; do
        n=$(find "${DATA_ROOT}/${tree}/${split}" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  ${tree}/${split}: ${n} files"
    done
done
for category in Industrial_and_Scientific Office; do
    n=$(find "${DATA_ROOT}/Amazon18/${category}" -maxdepth 1 -name '*index*.json' 2>/dev/null | wc -l)
    echo "  Amazon18/${category}: ${n} index json"
done

echo "--- resolving every variant in both trees ---"
for tree in next-item two-item; do
    "${ENV_PREFIX}/bin/python" "${REPO}/scripts/variants.py" \
        --tree "${tree}" --data-root "${DATA_ROOT}"
done

echo "--- quota after extraction ---"
lfs quota -h -u "${USER}" /l/users 2>/dev/null | tail -2 || true

echo "[$(date --iso-8601=seconds)] extraction complete"
