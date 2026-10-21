# Controlled exp11: the copy findings on the retrained Qwen-AR grid

The copy findings (exp3 recent-item replacement, exp4 prefix-matched copying
and the single-edge knockout, exp6 copy knockout inside beam search) were
measured on two checkpoints. This experiment reruns their core stages,
unchanged, on the 12 retrained next-item models that substrate exp1 accepted
(other depths, width 512 and MQ), and compares the retrained RQ-KMeans 3×128
with the original `next-item_best` on the same rows.

- Protocol: [protocol.md](protocol.md), declared before any intervened forward.
- Results: [RESULTS.md](RESULTS.md).

There is no `run.py`: each stage calls the source experiment's own `run.py`
with `--ckpt retrain/r20261004/<variant>`. Run from the repo root on the CIAI
cluster (GPU); 3 array tasks share the 12 models:

```bash
source ~/.config/sidlens/site.env
CELLS=10 sbatch --array=0 --time=02:00:00 --exclude=$SIDLENS_SBATCH_EXCLUDE \
    scripts/controlled/ar_copy_grid.sbatch --limit 64      # pilot
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/ar_copy_grid.sbatch
python experiments/controlled/exp11_ar_copy_grid/summarize.py \
    --groups $SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/<array-id> \
    --out $SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/summary-<array-id>
```

Output: `$SIDLENS_WORK/derived/controlled/exp11_ar_copy_grid/<array-id>/cell-XX/<stage>/`,
one directory per stage (`exp4_valid`, `exp3_A`, `exp4_test`, `exp6`) holding
that stage's own outputs, plus source, hashes and `status.txt` per cell. The
summary directory holds `estimates.csv`, `g1_counts.csv`,
`g2_retrained_vs_original.csv`, `report.md` and `figures/`.
