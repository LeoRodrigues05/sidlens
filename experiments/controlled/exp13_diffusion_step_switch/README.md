# controlled/exp13: at which reveal step does DiffGRM use the history?

The most recent item is replaced (exp7's m = 0 control) at chosen reveal steps
of the decode only (`history_switch_scorer`): none, all, each single step, all
but one. Greedy and beam 64, all 20 next-item checkpoints. Protocol:
[protocol.md](protocol.md). Results: [RESULTS.md](RESULTS.md).

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/controlled/diffusion_step_switch.sbatch
$SIDLENS_PYTHON experiments/controlled/exp13_diffusion_step_switch/summarize.py --run <run dir> --out <new dir>
```

Outputs per cell: `decodes.npz` (top-10 per schedule and beam; greedy reveal order and codes),
`users.csv.gz`, `inputs.json`, `validation.json` (V2 none = plain decode, V3 all = donor decode,
V4 exp1 archive), `output.sha256`.
