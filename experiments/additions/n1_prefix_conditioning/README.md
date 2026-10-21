# N1: prefix-conditioning test

`protocol.md` declares the question, designs and estimands. For each eligible
(row, digit d >= 2) the most recent history item h matching the target's first
d digits is replaced by h' under three conditions (suffix, none, prefix) and
digit d is scored at the golden prefix: AR through the fixed-shape runner, DiffGRM
through the projected cross-attention scorer (`interventions.diffusion.score_states`).

    python run.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 256] [--device cpu]
    python summarize.py --cells <cell> [...] --out <new dir>

Outputs: `records.parquet` (one row per design and condition: log-probs of h's,
h''s and the golden digit-d code, `synthetic`), `inputs.json`, `validation.json`.
