# Collision analysis scripts

This folder contains the reproducible, Python-only plotting pipeline for the
OneDiffRec result analysis. It combines the collision/performance workbook,
the Office codebook-256 table in the research-notes PDF, and local sweep/W&B
records for training-resource measurements. It writes only to `diagrams/new/`
by default, keeping the analysis isolated from the older diagram suite.

Run from the repository root:

```bash
.conda/bin/python scripts/diagrams/new/make_all.py
```

An alternate input or output location can be supplied explicitly:

```bash
.conda/bin/python scripts/diagrams/new/make_all.py \
  --workbook path/to/results.xlsx \
  --research-notes path/to/research-notes.pdf \
  --sweep-metrics-root path/to/sweep/metrics \
  --wandb-root path/to/wandb \
  --output-dir path/to/output
```

Use the existing resource CSV without reopening experiment logs:

```bash
.conda/bin/python scripts/diagrams/new/make_all.py --skip-resource-refresh
```

The pipeline validates the complete 2-dataset collision grid, paired 54-run
Industrial performance grid, 18-row Office codebook-256 grid, and (when
available) the 18 completed current SFT resource runs. Every figure is produced
with Matplotlib's non-interactive `Agg` backend and exported as a 400-dpi PNG
plus a vector PDF. No browser plotting or image-generation tool is involved.
Pandas and NumPy prepare tables, SciPy computes statistics, OpenPyXL reads the
workbook, pypdf extracts the Office table, and W&B's protobuf reader recovers
training telemetry. The shared academic-paper style uses a uniform white
background, serif typography, restrained blue/orange/green series colors,
subtle dotted grids, panel labels, and title-free image assets. Legends remain
15.3 points—exactly 1.8× the original 8.5-point size.

The 12-figure suite adds four analysis views beyond the original eight:

- balanced marginal effects for tokenizer, SID depth, and codebook size;
- a cross-dataset generator-paradigm interaction at codebook 256;
- estimated training FLOPs, measured end-to-end evaluation wall time, sampled
  training-GPU energy, and the observed NDCG@10/resource Pareto frontier.
- a current-grid SID-depth trade-off joining collision burden, the paired
  Mask-Diffusion performance gap, and measured end-to-end evaluation cost.

Resource semantics are deliberately conservative. `total_flos` is Hugging
Face Trainer's analytical estimate, not a hardware-counter measurement. Energy
integrates sampled GPU power only. The available traces cover autoregressive
SFT at codebooks 128/512 on 4 × A100-SXM4-40GB GPUs; they do not support a
direct AR-versus-DiffGRM efficiency claim. Evaluation time is measured around
the complete four-GPU evaluation script for 3,681 examples with 50-beam
constrained decoding; it includes model loading, split/merge, and scoring.
Existing artifacts contain no inference FLOP counter or inference-only GPU
telemetry, so an actual inference-FLOP result requires a new profiled run rather
than a conversion from wall time.

The workbook and research-notes PDF are local inputs ignored by this
repository's `.gitignore`; both must be present for a full regeneration. The
W&B dependency is used only to read offline protobuf logs, and the parser opens
those logs in read-only mode.
