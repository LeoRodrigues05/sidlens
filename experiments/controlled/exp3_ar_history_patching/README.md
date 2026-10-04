# exp3: AR history interventions and residual-stream patching

This is the first causal experiment on the AR recommenders, in two parts:

- **Part A** replaces one history item at a time with a catalogue control that
  shares the first m SID digits. It measures the change in the golden code's
  teacher-forced log-prob at every target digit.
- **Part B** patches the clean residual stream into the run where the most
  recent item is replaced. The sweep covers 28 layers × 4 position groups, and
  shows at which layer the item's evidence leaves its tokens and reaches the
  positions that score each digit.

The protocol ([protocol.md](protocol.md)) was declared before any intervened
forward pass. Results are in [RESULTS.md](RESULTS.md).

| Cell | Checkpoint | SID | Digits |
|---|---|---|---|
| 00 | `next-item_best` | RQ-KMeans 3×128 | 3 |
| 01 | `oneoff_rqvae4cb128` | RQ-VAE 4×128 | 4 |

The cohort is all 3,681 historical AR next-item test rows (1,606 users), with
eval wording and teacher forcing. It is not the diffusion cohort.

## Code

| Piece | Where |
|---|---|
| Patch mechanism (fires-exactly-once guard, residual sites only, no duplicate writes) | `src/sidlens/interventions/residual.py` |
| Prefix-matched controls, re-encode, changed-position proof | `src/sidlens/interventions/history.py` |
| Per-digit scores: codes / legal / vocab log-prob, rank, top-1 | `src/sidlens/interventions/scoring.py` |
| Runner (fixed-shape forwards, exact controls) | `run.py` |
| Estimates, paired user bootstrap, figures | `summarize.py` |
| Tests (CPU; toy Qwen2 exactness plus frozen tokenizers) | `tests/interventions/` |

## Running

Submit from the repo root with `site.env` sourced. On the CIAI cluster that
file sets `SBATCH_PARTITION=cscc-gpu-p`, `SBATCH_QOS=cscc-gpu-qos` and the
bad-node list:

```bash
source ~/.config/sidlens/site.env
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_history_patching.sbatch --limit 64   # pilot
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_history_patching.sbatch             # primary
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_history_patching.sbatch --template sft --parts A
$SIDLENS_PYTHON experiments/controlled/exp3_ar_history_patching/summarize.py \
    --cells $SIDLENS_WORK/derived/controlled/exp3_ar_history_patching/<id>/cell-0{0,1} \
    --out   $SIDLENS_WORK/derived/controlled/exp3_ar_history_patching/summary-<id>
```

Measured on an A100 in bf16: about 0.3 s per 128-row forward. The primary run
takes about 25–30 minutes per cell.

## Numerical acceptance

Every forward uses one shape: B = 128 rows × T_pad tokens, where T_pad is 128
for RQ-KMeans and 144 for RQ-VAE. With the shape fixed, a row's output
depends only on its own tokens. On the 64-row pilots (`pilot-280955`) this
gave:

- **Controls:** 768/768 control forwards per cell bit-identical to their
  reference, with max |Δ| = 0.0:
  - no-op patch before the item
  - full-layer restore
  - self-patch
- **Batch invariance:** clean and corrupted rows scored identically in
  different batch compositions (Part A vs Part B), with max |Δ| = 0.0.
- **bf16 vs fp32:** max |Δ log p| of 0.10 and 0.07 nats; top-1 agreement
  100% and 98.4%.

## Site traps found on this cluster (2026-09-27)

- **`gpu-54`** cannot see the venv's Python (a symlink into `/home/.../.conda`),
  so jobs there die with "FATAL: no python". It shows as idle because every job
  sent there fails immediately.
- **`gpu-05`** GPUs report "Unknown Error" in `nvidia-smi`, and torch fails with
  "CUDA unknown error".
- **`long`** with `gpu-12` is rejected for account `cscc-users`. Use
  `cscc-gpu-p` / `cscc-gpu-qos`.
- **QoS caps:** about 4 submitted and 2 running jobs per user. A 96G memory
  request waited for hours while 32G starts at once.
