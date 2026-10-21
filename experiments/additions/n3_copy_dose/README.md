# N3: copy dose-response

`protocol.md` declares the doses, gates and estimands. AR: exp6's `C_all`
edges on the float-mask route (`grid.decode_ar.CopyEdit(alpha=...)`); DiffGRM:
exp14's r1 block as post-softmax reweighting (`grid.diff_setup.encode_slot_scaled`,
`slot_scale_context`). alpha = 0 reproduces exp6 / exp14's block (DiffGRM:
checked on the smoke run, identical lists).

    python run.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 64] [--device cpu]
    python summarize.py --cells <cell> [...] --out <new dir>

Outputs: `predictions_alpha-<a>.parquet`, `rows.parquet` (target classes),
`scores.parquet` (per row and alpha: SID and item hit@10, history share),
`inputs.json`, `validation.json`.
