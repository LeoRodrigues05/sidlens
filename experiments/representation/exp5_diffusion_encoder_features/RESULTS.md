# representation/exp5: what the DiffGRM encoder holds (2026-10-14, all 20 cells)

Linear probes at three encoder sites, TopK SAEs on the encoder output (with
causal ablations), and a description autoencoder in the style of natural
language autoencoders, on all 20 next-item checkpoints.

**Findings.**

1. **The single encoder layer turns each history slot from "this item" into
   "this user's recent context".**
   - *Own identity.* At the encoder input a slot's own first code is linearly
     readable almost perfectly (0.98–1.00), and its category at 0.86–0.98.
     After the self-attention it falls to 0.43–0.85, and at the output, the
     only thing the decoder reads, to **0.31–0.53** (category 0.37–0.50).
     Later digits fare no better: 0.20–0.53 at the output.
   - *Context.* Meanwhile the output gains what the input cannot hold:
     - the slot's recency: **+0.20 to +0.31** accuracy;
     - the most recent item r1's first code, read from the *other* slots:
       **+0.15 to +0.41** (to 0.20–0.48);
     - r1's category from the other slots: +0.12 to +0.29.

     All 60 of these gains exclude 0. This is the spread exp8 found with
     knockouts, read directly, in every cell.
2. **The encoder holds no information about the target beyond the history's
   content.**
   - *Target's first code.* Probed from [r1's state, mean state], output
     minus input is −0.011 to +0.010 on new-target users (target's first code
     not in the history), and slightly negative on all users (−0.027 to
     −0.004). The same holds for the target category.
   - *Absolute level.* On new targets the probe reaches 1.9–5.4 % accuracy;
     the model's own fully masked guess reaches 1.0–6.5 %.
   - *Comparison.* This is the AR result (representation/exp1, exp2):
     nothing about new targets is encoded beyond "copy the recent context".
3. **SAEs reconstruct the encoder output well, and most of their latents are
   not single named attributes.**
   - *Fidelity.* Test FVU is 0.064–0.245; splicing the reconstruction into
     every slot keeps **0.87–0.99** of the decoder's behaviour relative to
     mean ablation.
   - *Census.* Of the live latents, 90–95 % match no single family (κ ≥ 0.5).
     The named ones are:
     - own category, 0.4–5 %;
     - r1's first code, as **spread latents** in the other slots, 1.2–6.4 %;
     - recency, 1–2 %.

     Latents for the slot's own SID digits are nearly absent (≤ 0.5 %),
     consistent with finding 1.
4. **Spread latents are used.** Ablating, in every non-r1 slot, the active
   spread latents whose preferred code is r1's first code (CS1) lowers the
   copy score by up to **0.36 nats** more than a matched control. The interval
   excludes 0 in **19 of 20** cells; RQ-KMeans 3×512 is the exception
   (−0.02). Such latents are active for only 3–33 % of users. On those users,
   exp8's encoder block costs 0.10–0.48 nats, so the named spread latents are
   part of the encoder route, not all of it. CS2 (identity latents in r1's own
   slot) could not be tested: such latents are active for at most 3 % of users.
5. **The description autoencoder explains half of the state and most of the
   behaviour, and the word the decoder needs is the recent item.**
   - *Reconstruction.* An 8-word description per slot rebuilds 24–56 % of
     the encoder output's variance (FVU 0.44–0.76). Splicing the
     reconstruction in keeps **0.39–0.85** of the decoder's behaviour.
   - *Verbalizer vs oracle.* The verbalizer's own words beat the true labels
     (oracle FVU 0.64–0.85; splice 0.25–0.71). The state encodes what is
     *predictable* about a slot's item, not its label.
   - *Field drop.* Dropping one word hurts most for r1's category (17 cells)
     or r1's first code (3 cells). Dropping one of the slot's own words, or
     its recency, costs 0.01–0.03 of mean-ablation KL on average; dropping
     r1's category costs 0.31.
   - *By quantizer.* RQ-VAE states are the hardest to describe (FVU
     0.67–0.76, splice 0.39–0.56). The easiest are MQ at depth 5 and
     RQ-KMeans 5×512 (FVU 0.44–0.48).
6. **Trends.**
   - *By digit.* RQ-VAE keeps its first digit (0.36–0.52 at the output)
     better than later digits (0.19–0.32). MQ keeps every digit equally
     (within 0.04). RQ-KMeans is in between.
   - *By width.* Own-digit retention is lower at width 512 for MQ and
     RQ-KMeans (more classes; 0.27–0.41 vs 0.33–0.53).

The protocol ([protocol.md](protocol.md)) was declared before any forward pass.
Cohort: 6,297 users, 36,783 analysis slots. Fit split: 83,854 slots from
training-target histories. Intervals: paired user bootstrap, 2,000 draws,
seed 20261014. They exclude training-seed uncertainty, both of the
recommenders and of the probes, SAEs and reconstructors (each fitted once).

| Run | Job | Output under `$SIDLENS_WORK/derived/representation/exp5_diffusion_encoder_features/` |
|---|---|---|
| Failed on a node without a working CUDA driver (gpu-55) | 312762 | `312762.failed-gpu55-no-cuda/` |
| CPU pilot, 600 users (cancelled: CPU probes too slow) | 312752 | `pilot-312752/` (incomplete) |
| CPU smoke, 150 users, cell 6 | 312768 | `pilot-312768/` |
| **Primary**, 20 cells, A100 | 312763 | `312763/cell-00…19` |
| **Summary** | login node | `summary-312763/` (`estimates.csv`, `census_shares.csv`, `report.md`) |

## Numerical acceptance (all 20 cells)

- **V1**: exp1's pilot checks passed.
- **V2**: `encoder_sites` reproduced the encoder block and the encoder output
  bit for bit. A padding-only encoder mask was an exact no-op.
- **V3**: the rebuilt analysis histories equal the cohort's.
- **V4**: checkpoint hashes matched.
- **V5**: an empty SAE ablation, scored in the conditions' batches, was
  bit-identical to the clean scores.

## Expectations recorded before outcomes (scored mechanically in the report)

- **X1** (own first code ≥ 0.9 at the output, every width-128 cell):
  **failed**, 0 of 9 (0.36–0.53). Finding 1.
- **X2** (RQ: own-digit accuracy falls from the first digit to the last; MQ
  falls ≤ 0.1): **failed narrowly**. It holds for all 7 MQ cells and 12 of 13
  RQ cells.
- **X3** (recency output − input ≥ 0.2, every cell): **failed narrowly**. It
  holds in 19 of 20; the lowest gain is 0.1997 (RQ-VAE 5×512).
- **X4** (spread > 0, interval excluding 0, every cell): **held**, 20 of 20.
- **X5** (no target information beyond content on new targets, ≤ 0.02 every
  cell): **held**. The maximum is +0.010.
- **X6** (SAE FVU ≤ 0.2 and splice ≥ 0.7, every cell): **failed**. 17 of 20
  have FVU ≤ 0.2 (the highest is 0.245); every splice is ≥ 0.87.
- **X7** (CS1 > 0, interval excluding 0, in ≥ 15 of 20): **held**, 19 of 20.
- **X8** (description FVU ≥ 0.5 every cell; most-needed field is recency or
  the own first code in ≥ 15 of 20): **failed**. The FVU is ≥ 0.5 in 15 of
  20. The most-needed field is r1's category or code in all 20.

## Limits

- Probes and the census are observational. Only CS1, the splices and the
  field drops are interventions.
- The census assigns a latent to one family by chance-corrected selectivity on
  the fit split. "Unassigned" means no single named attribute fits; it does
  not mean the latent is meaningless.
- The description vocabulary was ours and the verbalizer supervised. A natural
  language autoencoder proper (LLM verbalizer and reconstructor trained by RL)
  might name what the 44–76 % of unexplained variance holds.
- Fit and analysis splits share users (earlier prefixes of the same sequences).
- Up to 58 % of latents are dead on the analysis split in some cells
  (RQ-KMeans 5×512, MQ 5×512); the census uses live latents only.
- One checkpoint per cell. Widths also differ by SID table.
