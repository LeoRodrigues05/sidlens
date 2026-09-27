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
| `common/` | `_preamble.sh`, sourced by every wrapper: the per-cluster site file, the verify gate (honours `SIDLENS_VERIFY_PROFILE`), offline HF settings, and the provenance stamp. `site.env.example` is the template for `~/.config/sidlens/site.env`. |
| `data/` | `build_labels.py`: joins Amazon-2018 metadata into `external/labels/`. `--check` writes nothing. |
| `provenance/` | `audit_checkpoints.py`: which planned cells have weights. It exits 1 by design while cells are missing. |
| `structure/` | Wrappers for `experiments/structure/`: `exp1_atlas`, `exp2_geometry`, `semantic_mapping`. |
| `retrospective/` | `retrospective_experiments.sbatch`: the CPU audits of archived outputs. It writes into fixed `derived/retrospective/*` directories, so do not rerun it casually. |
| `controlled/` | Wrappers for `experiments/controlled/`: matched-beam and fixed-order decoding, each with a summary job. They pin this cluster's GPU node. |
| `activations/` | `ar_capture.{py,sbatch}`: AR residual-stream capture into the activation store; the wrapper is site-neutral. `hooks_smoke.{py,sbatch}`: capture-layer smoke test on the real checkpoints. |

This repository has no training scripts: models are trained upstream (see
`docs/clusters/`). Intervention code belongs in `src/sidlens/interventions/`,
and its wrappers go in a new `interventions/` group here.
