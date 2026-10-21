# N2: how the AR model matches the prefix

`protocol.md` declares (a)-(f). `run.py` measures (a)-(e) on N1's designs with
the query/key/value hooks of `src/sidlens/interventions/qkv.py`; `induction.py`
computes (f)'s base-model induction scores once.

    python run.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 256]
    python induction.py --base-model <Qwen2.5-1.5B-Instruct dir> --out <new dir>
    python summarize.py --cells <cell> [...] --induction <dir> --exp4-root <results_remote/controlled/exp4_ar_copy_circuit> --out <new dir>

Outputs: `records.parquet` (per design: log p of h's code under every
condition, donor log-probs), `attention.npz` (per design, layer and head:
attention to h's / h2's digit token; late-head DLA), `inputs.json`, `validation.json`
(the key self-patch must be bit-exact).
