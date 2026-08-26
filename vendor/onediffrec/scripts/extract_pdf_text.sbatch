#!/bin/bash
#SBATCH --job-name=onediffrec-pdftext
#SBATCH --account=cscc-users
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

# Dumps the research-notes PDF to per-page text so the tables can be read.
# No PDF tooling exists in the environment, so pypdf is installed here.

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
PDF=${PDF:-${REPO}/data/Research Notes - OneDiffRec (1).pdf}
OUT=${OUT:-${WORK}/notes_pdf_text}

export PIP_CACHE_DIR=${WORK}/cache/pip

mkdir -p "${OUT}"
cd "${REPO}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
"${ENV_PREFIX}/bin/python" -m pip install --quiet pypdf==6.1.1
"${ENV_PREFIX}/bin/python" -c "import pypdf; print('pypdf', pypdf.__version__)"

"${ENV_PREFIX}/bin/python" - "${PDF}" "${OUT}" <<'PY'
import sys
from pathlib import Path

from pypdf import PdfReader

pdf_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
reader = PdfReader(str(pdf_path))
print(f"pages: {len(reader.pages)}")

combined = []
for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    (out_dir / f"page_{i:02d}.txt").write_text(text)
    combined.append(f"\n{'=' * 30} PAGE {i} {'=' * 30}\n{text}")
    print(f"  page {i:2d}: {len(text):6d} chars, {len(text.splitlines()):4d} lines")

(out_dir / "all_pages.txt").write_text("".join(combined))
print(f"wrote {out_dir}/all_pages.txt")
PY

echo "[$(date --iso-8601=seconds)] pdf text extraction complete"
