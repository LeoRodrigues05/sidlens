# exp6: does prefix-matched copying help or hurt real AR recommendations?

This experiment knocks out copy links inside the **archived AR decoder**:
trie-constrained `generate()`, 50 beams, reproduced batch for batch. The
knockout blocks attention in layers 14–27 from the step that picks digit d to
digit d of the history items matching that beam's own prefix. For the first
digit (`C_all`), it blocks every history item's first digit.

The outcome is exact-SID HR@K and NDCG@10, reported separately for **repeat**
targets (the target SID is in the history) and **new** targets.

[protocol.md](protocol.md) was declared first. It carries a dated amendment:
the archived decoder turned out to be beam *sampling* with repetition penalty
1.1, because transformers merges the checkpoint's `generation_config.json`
defaults. Results are in [RESULTS.md](RESULTS.md).

| Piece | Where |
|---|---|
| Knockout inside KV-cached beam search (per-beam plans, fire-once, live-edge guards) | `src/sidlens/interventions/generation.py` |
| Tests (empty plan = plain `generate()` bit for bit) | `tests/interventions/test_generation_knockout.py` |
| Runner (vendored trie + `ConstrainedLogitsProcessor`, `split.py` shards, batches of 8, `set_seed(42)` per shard) | `run.py` |
| Summary (exact-SID and `calc.py` ranks, strata, paired user bootstrap) | `summarize.py` |
| Wrapper | `scripts/controlled/ar_copy_in_decoding.sbatch` |

Measured on an A100 in bf16: about 1.4 s per 8-row batch, 11 min per condition,
and about 70 min per cell for six conditions.
