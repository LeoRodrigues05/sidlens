# representation/exp1: layer-wise decoding of SID digits from AR activations

This is an observational read of the test-split activation captures
(`derived/ar_capture/281053`, `281054`), using two tools:

- **Logit lens:** final norm + tied unembedding applied to every layer, at the
  positions that predict each digit.
- **User-disjoint linear probes (P1–P4):** target digit 1 and the most recent
  item's digit 1 at the digit-1 readout; the most recent item's digit 1 at its
  own last token; target digit 2 at its readout.

[protocol.md](protocol.md) was declared first. It carries a dated amendment
(the acceptance check compares at bf16 logit precision) and a disclosure (the
smoke run saw partial numbers). Results are in [RESULTS.md](RESULTS.md).

```bash
sbatch --exclude=$SIDLENS_SBATCH_EXCLUDE scripts/representation/ar_digit_decoding.sbatch   # array: 0 = 281053, 1 = 281054
$SIDLENS_PYTHON experiments/representation/exp1_ar_digit_decoding/summarize.py --cells <run>/cell-00 <run2>/cell-01 --out <summary>
```

Measured: about 1.5 h per capture on an A100. The time goes to 2,720
full-batch L-BFGS probe fits.
