# Structure exp4: the final (v3) SID maps

Recomputes the map analyses of the paper (RQ1; final plan M1, M2, M5) on the
SID tables of the final training run, for both catalogs, and adds the
analyses the new quantizer weights make possible. CPU only; no recommender is
run. Protocol: [protocol.md](protocol.md). Results: [RESULTS.md](RESULTS.md).

## Inputs

- `WFRaain/onediffrec-sid-codebooks` at revision `9733eda7` (2026-10-19),
  downloaded to `$SIDLENS_WORK/external/sid_codebooks/WFRaain-onediffrec-sid-codebooks-9733eda/`
  (790 files, 7.7 GB, sizes checked). Read through `sidlens.data.codebooks_v3`,
  which checks each table's sha256 against its build record and maps the item
  index to ASINs.
- Frozen embeddings, item metadata and interaction sequences (unchanged).
- Amazon-2018 labels: `external/labels/Industrial_and_Scientific.labels.json`
  (2026-09-05) and `external/labels/Office.labels.json` (built 2026-10-20 from
  `meta_Office_Products.json.gz`, HTML entities unescaped; 100 % join).

## Run

```bash
source ~/.config/sidlens/site.env
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none scripts/structure/final_maps.sbatch
# then, on the login node (seconds):
$SIDLENS_PYTHON experiments/structure/exp4_final_maps/summarize.py --run $SIDLENS_WORK/derived/structure/exp4_final_maps/<job>
$SIDLENS_PYTHON experiments/structure/exp4_final_maps/figures.py   --run $SIDLENS_WORK/derived/structure/exp4_final_maps/<job>
$SIDLENS_PYTHON experiments/structure/exp4_final_maps/compare_old.py --run $SIDLENS_WORK/derived/structure/exp4_final_maps/<job> \
    --old-exp3 /l/users/leo.rodrigues/sidlens/derived/structure/exp3_collision_causes/293555/result \
    --old-zcr /l/users/leo.rodrigues/sidlens/derived/substrate/zcr_inputs/20261004
$SIDLENS_PYTHON experiments/structure/exp4_final_maps/rqvae_buckets.py --run $SIDLENS_WORK/derived/structure/exp4_final_maps/<job>
$SIDLENS_PYTHON experiments/structure/exp4_final_maps/report.py    --run $SIDLENS_WORK/derived/structure/exp4_final_maps/<job>
$SIDLENS_PYTHON experiments/structure/exp4_final_maps/paper_report.py --run $SIDLENS_WORK/derived/structure/exp4_final_maps/<job>
```

The old exp3 and ZCR directories exist only on `/l`; `compare_old.py` copies
every file it reads from them into `summary/compare/` with its sha256.

`run.py --quick` is a smoke run (5 permutations, a few tables). Industrial takes
about 3 minutes on 128 cores; Office takes longer (17,696 items).

## Files

| File | What it does |
|---|---|
| `run.py` | One catalog: Part 0 validation, Parts A/B (exp3), D (post-processing), HUB, C (refits), MQ audit, X (exploratory), M1, Z (ZCR), M5, R (old vs v3) |
| `summarize.py` | Both catalogs: merged CSVs, expectations E1–E11, the draft's RQ1 numbers recomputed, the collision-cause table |
| `figures.py` | The paper's two map figures redrawn on v3 (one row per catalog), post-processing and exposure figures; CSV twin for each |
| `compare_old.py` | The old (frozen, Industrial) values of every map number the draft cites, recomputed from the paper data package and the old exp3/ZCR outputs with the v3 definitions; `summary/compare/` (claims, per-map and per-digit tables, the old-vs-v3 figure) |
| `rqvae_buckets.py` | Post hoc: encoder outputs and dead hidden layers of every RQ-VAE bucket of distinct items the reassignment loop left (`summary/rqvae_bucket_latent.csv`) |
| `report.py`, `narrative.py` | The full HTML report (`docs/results/final_maps/sidlens_final_maps.html`) and `facts.md`, every number cited in RESULTS.md |
| `paper_report.py` | The paper-facing page (`docs/results/final_maps/sidlens_final_maps_paper.html`): every map result the draft displays, old against v3, what to change |

Measures come from the code that produced the old numbers:
`sidlens.analysis.{geometry,refinement,attribute,naming}`, the helpers of
`experiments/structure/semantic_mapping/figures.py` and of
`experiments/structure/exp3_collision_causes/run.py`, imported, not copied.
