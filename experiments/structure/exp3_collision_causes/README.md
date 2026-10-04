# Structure exp3: why items share a Semantic ID

A CPU study of the 27 frozen Industrial SID tables. It tests five
hypotheses for collisions:

- a processing step (RQ-VAE's last-digit loop, RQ-KMeans 5×512 balancing)
- near-duplicate products
- shared residual codebooks
- embedding anisotropy
- redundant parallel digits (MQ)

The tests are permutation nulls on the tables, faiss refits of RQ-KMeans
under changed procedures, and simulations of the upstream post-processing.
No recommender is run.

- Protocol: [protocol.md](protocol.md), declared before any estimand was
  computed, with post-hoc amendments A1–A3 marked.
- Results: [RESULTS.md](RESULTS.md).

Run it from the repo root on a CPU node; it takes about 3 minutes. It needs
96 CPUs, because faiss k-means only reproduces the archived tables at 96
OpenMP threads, and `run.py` stops if the native refit does not reproduce
every nested archived table.

```bash
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/structure/collision_causes.sbatch
# A3 (post hoc): balancing fingerprints of RQ-KMeans 5x512
python experiments/structure/exp3_collision_causes/a3_balance_signature.py --out <new dir>
```

Output: `$SIDLENS_WORK/derived/structure/exp3_collision_causes/<job-id>/result/`.

| File | Contents |
|---|---|
| `partA_catalogue.json` | the catalogue: identical inputs, spectrum, distances |
| `partA_near_duplicate_pairs.csv` | the near-duplicate pairs |
| `partB_tables.csv`, `partB_levels.csv` | the 27 archived tables, per table and per level: nulls, merge ratios, near-duplicate shares, repeat rates, A1 |
| `partB_rqvae_buckets.csv` | every RQ-VAE bucket of ≥ 3 items |
| `partC_refits.csv`, `partC_summary.csv` | 8 refit conditions × 3 widths × 5 seeds, with paired differences |
| `partC_a2_hub_codes.csv` | A2: the hub code at each level |
| `partD1_rqvae_loop.csv` | the RQ-VAE loop applied to RQ-KMeans |
| `partD2_uniform_mapping.csv` | upstream `--uniform` balancing |
| `partE_parallel_refits.csv` | parallel-digit refits (exploratory) |
| `validation.json` | the native reproduction check |
| `report.md`, `result.json` | tables and the verdicts computed by the declared rules |
