# scripts/

SLURM wrappers and command-line tools, grouped by task. The groups mirror
`experiments/`, so a wrapper sits beside the experiment group it runs. Submit
from the repository root, because each wrapper's log path
(`slurm_logs/%x-%j.out`) is relative to it:

```bash
sbatch scripts/controlled/fixed_orders.sbatch --beam-widths 64 --batch-size 32 --decoder-chunk-size 1024 --validate
```

| Folder | Contents |
|---|---|
| `common/` | `_preamble.sh`, sourced by every wrapper: the per-cluster site file, the verify gate (honours `SIDLENS_VERIFY_PROFILE`), offline HF settings, and the provenance stamp. `site.env.example` is the template for `~/.config/sidlens/site.env`. `pytest.sbatch` runs the test suite on a compute node, for clusters whose login node cannot reach `$SIDLENS_WORK`. |
| `data/` | `build_labels.py`: joins Amazon-2018 metadata into `external/labels/`. `--check` writes nothing. |
| `provenance/` | `audit_checkpoints.py`: which planned cells have weights. It exits 1 by design while cells are missing. |
| `structure/` | Wrappers for `experiments/structure/`: `exp1_atlas`, `exp2_geometry`, `semantic_mapping` (`--source v3 --catalog industrial|office` builds the explorer on the final-run tables), `collision_causes` (exp3; CPU, 96 threads), `final_maps` (exp4; both catalogs in one CPU job, 128 cores, about 20 min). |
| `retrospective/` | `retrospective_experiments.sbatch`: the CPU audits of archived outputs. It writes into fixed `derived/retrospective/*` directories, so do not rerun it casually. `time_structure.sbatch` (retrospective exp6) and `collision_credit.sbatch` (retrospective exp7): CPU, a new job directory per run, `-p cscc-cpu-p --qos=cscc-cpu-qos --gres=none` on the CIAI cluster. |
| `controlled/` | Wrappers for `experiments/controlled/`: matched-beam and fixed-order decoding, each with a summary job. They pin this cluster's GPU node. `diffusion_activation_patching.sbatch` (exp12; CPU is fine), `diffusion_step_switch.sbatch` (exp13) and `diffusion_decode_perturbations.sbatch` (exp14; GPU: beam search) walk the cells inside one job via `common/_cells.sh`. |
| `controlled/` (AR, site-neutral) | `ar_history_patching.sbatch`, `ar_copy_circuit.sbatch`, `ar_next_two_conditioning.sbatch`, `ar_copy_in_decoding.sbatch`, `ar_order_vs_time.sbatch` (exp9), `ar_last_digit_sibling.sbatch` (exp10), `diffusion_history_use.sbatch`, `diffusion_copy_route.sbatch` (the last two run on CPU with `--device cpu`): the AR intervention experiments; pass partition/QoS/`--exclude` at submission (see `site.env`). |
| `representation/` | `ar_digit_decoding.sbatch`: logit lens + probes over the AR activation captures (site-neutral). `ar_sae.sbatch <train|analyze|causal>`: representation/exp2 SAEs. The analyze stage runs on a CPU partition. `ar_information_map.sbatch <pilot|gpu|census>`: representation/exp3 (self-forced captures → SAEs → lenses in one GPU allocation; census on CPU). `diffusion_reveal_trace.sbatch`: representation/exp4, one array task per next-item DiffGRM checkpoint; `diffusion_reveal_trace_archive_diff.sbatch` describes a cell that stopped at its archive check. `diffusion_encoder_features.sbatch`: representation/exp5; like every 2026-10-14 diffusion wrapper it walks the cells inside one job (`common/_cells.sh`, `SIDLENS_CELLS=0-19`) because of the CIAI per-user job caps. |
| `grid/` | The OneDiffRec grid, for the coauthor's cluster: `sidlens-grid.sh` (`setup` builds the venv over OneDiffRec's Python and writes `~/.config/sidlens/grid.env`; `register`, `pilot`, `plan`, `run`, `status`, `summarize`, `export`), `run_best_state.sh` (every experiment, every configuration, best state), `run_full.sh` (plus the cheap set at every intermediate weight state), `cell.sbatch` (one array task; submitted by the pipeline, not by hand). Runbook: `docs/clusters/GRID_RUNBOOK.md`. |
| `substrate/` | `ar_next_item_retrain.sbatch`: experiments/substrate/exp1. 4 GPUs per task by default, array 0-3 (each task takes every 4th map); resumable per variant under `runs/ar_next_item_retrain/$RUN_NAME/`. Site-neutral. |
| `activations/` | `ar_capture.{py,sbatch}`: AR residual-stream capture into the activation store; the wrapper is site-neutral. `hooks_smoke.{py,sbatch}`: capture-layer smoke test on the real checkpoints. |

This repository has no training scripts: models are trained upstream (see
`docs/clusters/`). Intervention code belongs in `src/sidlens/interventions/`,
and its wrappers go in a new `interventions/` group here.
