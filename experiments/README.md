# experiments/

Entry points, grouped by the kind of evidence they produce. Each group's
SLURM wrappers are in the matching `scripts/<group>/` folder, and results go
to `$SIDLENS_WORK/derived/`.

| Folder | Evidence | Contents |
|---|---|---|
| `structure/` | Static properties of the 27 SID maps; no model is run | `exp1_atlas/` (per-digit attributes, naming, co-purchase), `exp2_geometry/` (radius, refinement, digit deletion), `semantic_mapping/` (the eight structure figures; `anomaly_audit.py`) |
| `retrospective/` | CPU audits of archived model outputs | `exp1_reveal_order`, `exp2_prefix`, `exp3_collisions`, `exp4_next_two` (see `retrospective/README.md`) |
| `controlled/` | New GPU decoding on frozen checkpoints, protocol declared first | `exp1_matched_beam/`, `exp2_fixed_orders/`, each with `README.md` and `RESULTS.md`. These are the templates for new experiments; see `CLAUDE.md`. |
| `paper_www27/` | Paper packaging | `export_paper_data.py`: hashed tables behind the SidLens numbers in the WWW'27 draft |
