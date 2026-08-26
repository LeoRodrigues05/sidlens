#!/bin/bash
#SBATCH --job-name=onediffrec-pdfblue
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

# Dumps the research-notes PDF to per-page text AND records the fill colour in
# force for every text run, so the blue cells (the ones we own) can be told
# apart from the co-author's black ones without eyeballing a rendering.

set -euo pipefail
umask 027

REPO=/home/leo.rodrigues/onediffrec/OneDiffRec
WORK=/l/users/leo.rodrigues/onediffrec
ENV_PREFIX=${REPO}/.conda
PDF=${PDF:-${REPO}/data/Research Notes - OneDiffRec (3).pdf}
OUT=${OUT:-${WORK}/notes_pdf_v3}

export PIP_CACHE_DIR=${WORK}/cache/pip

mkdir -p "${OUT}"
cd "${REPO}"

echo "[$(date --iso-8601=seconds)] host=$(hostname) job=${SLURM_JOB_ID}"
echo "pdf=${PDF}"
"${ENV_PREFIX}/bin/python" -m pip install --quiet pypdf==6.1.1
"${ENV_PREFIX}/bin/python" -c "import pypdf; print('pypdf', pypdf.__version__)"

"${ENV_PREFIX}/bin/python" - "${PDF}" "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

from pypdf import PdfReader

pdf_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
reader = PdfReader(str(pdf_path))
print(f"pages: {len(reader.pages)}")

all_runs = {}
combined = []

for i, page in enumerate(reader.pages, start=1):
    state = {"fill": (0.0, 0.0, 0.0)}
    runs = []

    def before(op, operands, cm, tm, state=state):
        try:
            name = op.decode() if isinstance(op, bytes) else str(op)
        except Exception:
            return
        try:
            if name == "rg":
                state["fill"] = tuple(float(x) for x in operands[-3:])
            elif name == "g":
                v = float(operands[-1])
                state["fill"] = (v, v, v)
            elif name == "k":
                c, m, y, k = (float(x) for x in operands[-4:])
                state["fill"] = ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
            elif name in ("sc", "scn") and len(operands) >= 3:
                state["fill"] = tuple(float(x) for x in operands[-3:])
        except Exception:
            pass

    def visit(text, cm, tm, font_dict, font_size, runs=runs, state=state):
        if text and text.strip():
            runs.append({
                "text": text,
                "color": [round(c, 4) for c in state["fill"]],
                "x": round(tm[4], 1) if tm else None,
                "y": round(tm[5], 1) if tm else None,
            })

    text = page.extract_text(
        visitor_operand_before=before, visitor_text=visit) or ""
    (out_dir / f"page_{i:02d}.txt").write_text(text)
    all_runs[i] = runs
    combined.append(f"\n{'=' * 30} PAGE {i} {'=' * 30}\n{text}")

    colors = {}
    for r in runs:
        colors[tuple(r["color"])] = colors.get(tuple(r["color"]), 0) + 1
    blue = sum(n for c, n in colors.items() if c[2] > c[0] + 0.15 and c[2] > 0.35)
    print(f"  page {i:2d}: {len(text):6d} chars, {len(runs):4d} runs, "
          f"{len(colors)} colours, {blue} blue-ish runs")

(out_dir / "all_pages.txt").write_text("".join(combined))
(out_dir / "runs.json").write_text(json.dumps(all_runs, indent=1))
print(f"wrote {out_dir}/all_pages.txt and runs.json")
PY

echo "[$(date --iso-8601=seconds)] pdf colour extraction complete"
