# N4: no-training baselines

`protocol.md` declares the baselines and estimands. CPU only; reads a run's
saved test lists, prepared cohorts and SID table.

    python run.py --ckpt 'grid:<key>@best' --out <new dir>
    python summarize.py --cells <cell> [...] --out <new dir>

Outputs: `rows.parquet` (per row: each list's SID and item hit@10, NDCG@10,
model/copy-rule agreement), `estimates.csv`, `inputs.json`, `validation.json`.
