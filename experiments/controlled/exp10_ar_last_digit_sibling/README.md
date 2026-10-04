# Controlled exp10: the last-digit read of a prefix-matching item

At the last digit, the model reads the last-digit token of a history item h
that matches the target on every earlier digit (exp4's copy edge). This
experiment removes that one attention edge, in layers 14–27 or in all layers,
and measures what happens to h's own last code and to the target's. The
target is either h's SID (repurchase or collision partner) or a sibling of
h, often a near-duplicate variant (retrospective exp7). A matched control
removes the same edge to a non-matching item's last-digit token.

- Protocol: [protocol.md](protocol.md), declared before any forward.
- Results: [RESULTS.md](RESULTS.md).

Run it from the repo root on the CIAI cluster (GPU). One array task per
cell: 0 = `next-item_best`, 1 = `oneoff_rqvae4cb128`.

```bash
source ~/.config/sidlens/site.env
X=gpu-04,gpu-05,gpu-07,gpu-49,gpu-50,gpu-51,gpu-53,gpu-54,gpu-56
sbatch --exclude=$X scripts/controlled/ar_last_digit_sibling.sbatch --split test --limit 64   # pilot
sbatch --exclude=$X scripts/controlled/ar_last_digit_sibling.sbatch --split test             # primary
sbatch --exclude=$X scripts/controlled/ar_last_digit_sibling.sbatch --split valid            # replication
python experiments/controlled/exp10_ar_last_digit_sibling/summarize.py \
    --run test=<test group dir> --run valid=<valid group dir> --out <new dir>
```

Output:
`$SIDLENS_WORK/derived/controlled/exp10_ar_last_digit_sibling/<array-id>-<split>/cell-XX/`
(`pilot-` prefix with `--limit`). Each cell directory holds:

- `scores.parquet`: per row and condition, ℓh, ℓt, the legal-set versions,
  top-1 flags and ranks
- `validation.json`: the no-op controls and the clean agreement with exp4
- `inputs.json`: the eligible-row counts
- source and provenance files

The summary directory holds `estimates.csv` and `report.md`.
