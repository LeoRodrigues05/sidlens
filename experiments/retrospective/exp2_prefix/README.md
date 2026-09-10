# Retrospective Experiment 2 — SID prefix errors

This CPU-only analysis reads every retained next-item autoregressive prediction
file and asks where ranked SID predictions first diverge from the target.

The primary quantities are:

- rank-1 longest common prefix and one-based first-error digit;
- prefix hit@1/3/5/10/20/50 at every available digit;
- rank-1 digit accuracy and first-error hazard conditional on a correct preceding
  prefix;
- exact full-SID HR and a separately labelled reconstruction of the historical
  evaluator's legacy exact-SID-or-title HR.

Validation is fail-closed. Each prediction record must align exactly, in order,
with the frozen test CSV's reconstructed input and target output. Test item IDs
and SIDs must agree with the frozen info catalogue; all non-empty predictions
must be canonical catalogue SIDs; empty beam padding must be trailing; and all
recorded HR/NDCG values must be reproduced within their four-decimal rounding
tolerance before outputs are written.

Intervals are deterministic user-cluster percentile bootstraps. They retain all
rows and configurations belonging to a sampled user. They condition on the
single frozen evaluation for each retained configuration and must not be read as
uncertainty over model seeds or configurations. Conditional digit hazards are
descriptive survival quantities, not independent digit effects.

Outputs under `$SIDLENS_WORK/derived/retrospective/exp2_prefix/`:

- `examples.csv`: one row per example × configuration, including LCP, first
  error, exact/legacy ranks, effective beam count, and maximum prefix depth by K;
- `config_metrics.csv`: long-form estimates and bootstrap intervals per exact
  quantizer/depth/width configuration;
- `strata_metrics.csv`: the same metrics overall and by quantizer, depth, and
  width;
- `metric_reproduction.csv`: recorded vs reconstructed exact and legacy HR/NDCG;
- `result.json`: definitions, caveats, hashes, validation results, and all
  summaries.
- `report.md`: concise generated headline tables; `result.json` remains the
  source of truth.

Run through SLURM (the wrapper pins the requested node and does not request a
GPU):

```bash
sbatch experiments/retrospective/exp2_prefix/run.sbatch
```

For a local validation-only invocation with no bootstrap intervals:

```bash
python -m experiments.retrospective.exp2_prefix.run \
  --bootstrap-replicates 0 --out /tmp/sidlens-retro-exp2-check
```
