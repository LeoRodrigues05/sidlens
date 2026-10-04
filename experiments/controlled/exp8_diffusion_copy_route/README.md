# exp8: how DiffGRM routes the copied item

This follow-up to exp7 blocks two routes of the most recent item, alone and
together:

- the encoder spread: other history slots reading it in the encoder's
  self-attention (`encoder_attention_masks`);
- the decoder's direct read of its slot (cross-attention).

A non-matching item is the control.

[protocol.md](protocol.md) was declared after exp7 and before any exp8 run.
Results are in [RESULTS.md](RESULTS.md).

```bash
sbatch -p cscc-cpu-p --qos=cscc-cpu-qos --gres=none --mem=64G scripts/controlled/diffusion_copy_route.sbatch --device cpu
```

It takes about 1 minute per cell on CPU.
