# representation/exp5: what the DiffGRM encoder holds (probes, SAEs, a description autoencoder)

Declared 2026-10-14, before any forward pass of this experiment. Plan:
`docs/plans/DIFFUSION_MECHANISMS_PLAN.md` (X1).

## Why

exp7 and exp8 showed that DiffGRM copies the most recent item r1, and that the
copy runs mainly through the encoder: other history slots read r1 and the
decoder reads it there. Both used knockouts on two cells. Nothing yet says what
the encoder states *contain*, whether that content can be named, or how it
changes with SID digit and codebook width. This experiment reads the encoder
directly on all 20 next-item checkpoints, with three tools of increasing
interpretive ambition, and tests the SAE features causally.

## Cells, cohort, sites

- **Cells**: the 20 trained next-item checkpoints, in representation/exp4's
  order (0–17: MQ, RQ-KMeans, RQ-VAE × depth 3, 4, 5 × width 128, 512;
  18: MQ 5×256; 19: RQ-VAE 3×256).
- **Analysis split**: the 6,297-user diffusion test cohort
  (`load_eval_cohort`), 36,783 real slots. Never joined to the AR cohort.
- **Fit split**: every training-target history. For each user's sequence s of
  length n and each 2 ≤ t ≤ n − 3: history s[max(0, t−50):t], target s[t]
  (the validation target is s[n−2], the test target s[n−1]). Probes, SAEs and
  reconstructors are fitted on this split only. **Fit-validation users**:
  sha256("20261014|<user>") mod 10 = 0; used only to choose penalties and to
  report the SAE's held-out FVU.
- **Sites**, from `diffusion_lens.encoder_sites` (one encoder block):
  `input` (E0) = item_mlp(digit embeddings) + position embedding; `mid` (E1) =
  after the self-attention residual add; `output` (E2) = ln_f, padding zeroed,
  the only thing the decoder reads. Real slots only.
- **Notation**: slot j of a history of length L has recency r = L − j (r1 = the
  most recent item, slot L − 1). Recency bins: 1, 2, 3, 4, 5, 6–10, 11+.

## Part P: linear probes

Multinomial logistic regression (`sidlens.probes.linear`: standardized inputs,
L2 penalty from {1e-4, 1e-3, 1e-2} chosen on fit-validation users, refit on all
fit rows). Reported on the analysis split: accuracy, majority accuracy, and
normalized information gain 1 − CE/CE_prior.

| Id | Label | Rows | Sites |
|---|---|---|---|
| P1 | own SID digit d, every d | all slots | input, mid, output |
| P2 | own `cat_l2`, own `main_cat` | slots with the label | input, output |
| P3 | recency bin | all slots | input, output |
| P4a | r1's first code | non-r1 slots whose own first code differs from r1's | input, mid, output |
| P4b | r1's `cat_l2` | non-r1 slots whose own `cat_l2` differs from r1's (both known) | input, output |
| P5 | target's first code; target's `cat_l2` | one row per history; features [site(r1), mean of site over real slots] | input, output |

The input site has no route from another slot, and it carries absolute position
but not the history length. So for P3 and P4 it is the no-computation
baseline. P5 is also reported on **new-target rows** (the target's first code
is no history item's first code), against two references: the copy rule (predict
r1's value) and the model's own fully masked argmax (first code only).

## Part S: TopK SAEs on the encoder output

- `sidlens.sae.topk`, one per cell: 2,048 latents (8 × 256), k = 16, AuxK
  k_aux = 128 with coefficient 1/32, dead after 50,000 rows, Adam lr 5e-4,
  batch 1,024, 60 epochs, seed 20261014. Fitted on fit rows of
  non-validation users; FVU on validation users is reported. Fixed before any
  fit; nothing is tuned on the analysis split.
- **F1**: analysis-split FVU and the share of latents never active there.
- **F2 (splice)**: every real slot's output replaced by its SAE reconstruction;
  the fully masked state is scored (all digits, validated decoder path).
  Baseline: every real slot replaced by the fit-split mean output.
  Recovered = 1 − Σ KL(clean ‖ SAE) / Σ KL(clean ‖ mean), pooled over users and
  digits; also per digit, and the digit-1 top-1 agreement.
- **Census** (fit split). For latent f and label family ℓ: among fit rows where
  f is active and ℓ is defined, c\* is the most frequent value, s its share,
  p its base rate on all rows where ℓ is defined; κ = (s − p)/(1 − p). Families:
  own digit d (each d), own `cat_l2`, recency bin, r1's first code (non-r1
  rows only), target's first code. f is assigned the family with the largest κ
  if κ ≥ 0.5 and the support ≥ 20, else "unassigned"; latents never active on
  the fit split are "dead".
- **CS1, spread latents (causal)**. Users with L ≥ 2. c = r1's first code. In
  every non-r1 real slot, ablate (error-preserving, `TopKSAE.ablate`) the
  active latents of family "r1's first code" with c\* = c. **Control**: in the
  same slots, as many other active latents, excluding that family, each the
  unused active latent nearest in activation to an ablated one. Measures at the
  fully masked state: the copy score (log p of c at digit 1) and the golden
  first code. Δ = clean − ablated. **CS1** = mean Δcopy(spread) −
  mean Δcopy(control), paired over users with at least one ablated latent; the
  share of users with one is reported.
- **CS2, own-identity latents in r1's slot**. The same, in r1's slot, with the
  family "own digit 1" and c\* = c; control as above, excluding that family
  with c\* = c.
- **Reference**: exp8's encoder block (`eK`: every other slot stops reading r1)
  on the same users, so CS1 can be read as a share of the encoder route.

## Part N: a description autoencoder in the style of NLAs

Anthropic's natural language autoencoders (Transformer Circuits, 2026) pair an
LLM activation verbalizer with an LLM reconstructor trained by RL on
reconstruction. Here the verbalizer's language is a fixed, grounded
vocabulary of eight fields:

- recency bin
- own `cat_l1`, own `cat_l2`, own `main_cat`, own price decile
- own first code
- context fields: r1's `cat_l2` and r1's first code

A missing label is its own word.

- **Verbalizer**: one probe per field on the output site, fitted on all fit
  rows (λ as in Part P). A slot's description is the argmax word per field,
  rendered as one sentence.
- **Reconstructor**: ridge regression from the one-hot description to the
  output state. α is chosen from {0.1, 1, 10, 100} on fit-validation users, and
  the reconstructor is fitted on the verbalizer's own descriptions of the fit
  rows. An **oracle** reconstructor is fitted on the true words.
- **Measures**, on the analysis split:
  - FVU of output → description → reconstruction, and of the oracle.
  - The behavioural splice, scored as in F2: every real slot replaced by its
    reconstruction, reported as recovered vs mean ablation.
  - **Field drop**: one field set to its fit-split mode for every slot. The
    change in pooled KL against the full description says which named word the
    decoder needs.
- **Examples**: eight analysis users with 3 ≤ L ≤ 8, drawn with seed 20261014.
  Every slot shows its item title, its true words and its description.

## Numerical acceptance (all must pass before any estimand counts)

- **V1** exp1's pilot checks on 8 users (`verify_model_and_decoder`).
- **V2** `encoder_sites` internal checks (they raise on failure); the output
  site equals `diffusion.encode` bit for bit.
- **V3** analysis histories rebuilt from the sequences equal
  `cohort.histories` and `cohort.target_sids` for every user.
- **V4** checkpoint sha256 equals the registry.
- **V5** an ablation of an empty latent set, scored in the same batches as the
  conditions, is bit-identical to the clean scores.

fp32, TF32 off, eval mode, deterministic algorithms; GPU.

## Statistics

Paired user bootstrap (`UserBootstrap`), 2,000 draws, seed 20261014, over
analysis users; all conditions of a user stay together. Intervals exclude
training-seed uncertainty, both of the recommenders and of the SAEs, probes and
reconstructors (each fitted once). Trends over digit and width are descriptive:
width 128 vs 512 is compared within each of the nine quantizer × depth pairs.

## Expectations recorded before outcomes

- **X1**: P1 own first code accuracy at the output ≥ 0.9 in every width-128 cell.
- **X2**: in RQ cells, P1 output accuracy falls from digit 1 to the last digit;
  in MQ cells it does not fall by more than 0.1.
- **X3**: P3 output − input accuracy ≥ 0.2 in every cell (the encoder computes
  recency; the input cannot).
- **X4**: P4a output − input > 0, with the interval excluding 0, in every cell
  (exp8's spread, read directly).
- **X5**: P5 target first code on new-target rows: output − input ≤ 0.02 in
  every cell (no target information beyond the history content, as in AR).
- **X6**: SAE analysis FVU ≤ 0.2 and F2 recovered ≥ 0.7 in every cell.
- **X7**: CS1 > 0, interval excluding 0, in at least 15 of 20 cells.
- **X8**: description FVU ≥ 0.5 in every cell; the field with the largest
  field-drop KL is recency or the own first code in at least 15 of 20 cells.

## Interpretation limits

- Probes and the census are observational: decodable is not used. Only CS1,
  CS2 and the splices are interventions.
- The description vocabulary was chosen by us. It can find only what was named,
  and its verbalizer is supervised, not trained for reconstruction as an NLA's
  is. A full LLM NLA is future work.
- Fit and analysis splits share users (the fit split is earlier prefixes of the
  same sequences), as the recommender's own training did.
- One checkpoint per cell; width comparisons across independent fits also differ
  by the SID table.
