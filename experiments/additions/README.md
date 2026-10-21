# additions/: the four experiments added for the OneDiffRec grid

Declared in `docs/plans/COPYING_GRID_PLAN.md` §4.3; each folder holds its
`protocol.md` (written before any outcome), `run.py`, `summarize.py`,
`README.md` and `RESULTS.md`. They run only on grid weight states
(`grid:<key>@best`, see `src/sidlens/grid/`), through the pipeline
(`python -m sidlens.grid.pipeline`, `scripts/grid/`), and write to
`results_remote/additions/<folder>/<dataset>/<task>/<paradigm>/<map>/<state>/`.

| Folder | Plan | What it asks | Paradigms | Compute |
|---|---|---|---|---|
| `n1_prefix_conditioning` | N1 | Does the copy need the whole prefix or only the previous digit token? | AR, DiffGRM | GPU (AR), CPU/GPU (DiffGRM) |
| `n2_ar_prefix_match` | N2 | Where the AR model matches the prefix: query, key or value; early layers; head DLA; base-model induction heads | AR | GPU |
| `n3_copy_dose` | N3 | Copy reads scaled by alpha in {0, .25, .5, 1, 1.5, 2} inside the run's own beam search | AR, DiffGRM | GPU |
| `n4_no_training_baselines` | N4 | Copy rule and popularity against each run's saved lists | both (saved lists) | CPU |
