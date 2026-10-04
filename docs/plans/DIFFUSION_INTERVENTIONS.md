# Interventions for DiffGRM: what the literature suggests we implement

Written 2026-10-03. This surveys interpretability and intervention work on masked
(discrete) diffusion models and diffusion transformers, and maps each idea to
DiffGRM. It then ranks what to implement with the tools SidLens already has.

## What DiffGRM is, in the terms the literature uses

DiffGRM is an absorbing-mask discrete diffusion model over one SID.

- **Encoder.** One layer. It reads one token per history item: the item's
  digit embeddings through `item_mlp`, plus a recency position.
- **Decoder.** Four layers, 256 wide, 4 heads, with one position per SID
  digit and no positional code. A masked digit carries its digit's mask
  embedding. Every decoder block cross-attends to the encoder's item states.
- **Decoding.** A beam of reveal paths: confidence order or a fixed order,
  with legality checked only at the end.
- **Training facts that matter here:**
  - each training example follows one reveal order, the model's own
    confidence ranking
  - depth-5 models never see the fully masked SID, although every decode
    starts from it

We already have three things:

- Scores at any decoder state with the validated projected cross-attention
  cache (`interventions/diffusion.py`).
- Cross-attention and encoder self-attention knockouts.
- The validated decoder `analysis/matched_decode.py`.

What we already know:

- exp7: DiffGRM copies like the AR model.
- exp8: the copy runs mostly through the encoder (other item slots read the
  recent item), not through the decoder's direct read.
- Controlled 1 and 2: reveal order matters, and depth 5 loses most at
  beam 64.

## Ranked recommendations

### Tier 1: cheap, uses existing tools, answers open questions

**1. Copy knockout inside decoding (DiffGRM analogue of exp6).**

- **Basis.** Catruna & Radoi (2026) find a bidirectional induction circuit in
  masked diffusion LMs: previous- and next-token heads write local context,
  and induction heads copy from a matching source in either direction.
  exp7 and exp8 found the history copy in DiffGRM but never tested it in
  decoding.
- **Implement.** In `matched_decode`, apply exp8's "both routes" block for the
  history item that matches each beam's revealed digits:
  - the encoder slots must not read that item
  - the decoder must not read that item's slot

  The encoder runs once per user, so a beam-specific encoder block needs a
  per-beam encoder pass. That is cheap: one layer over 50 slots.
- **Measure.** HR@10, split as in retrospective exp7: same item, same-SID
  partner, near-duplicate variant, new. Compare with a matched control item.
  Also test copying *between digits*: block a decoder digit position from
  reading an already-revealed digit that matches a history item.
- **Why first.** It completes the AR vs diffusion comparison in RQ4 with the
  same estimand, and retrospective exp7 tells us where to expect the effect
  (variants and partners).

**2. The implicit step: does the decoder know how many digits are masked?**

- **Basis.** Catruna & Radoi (2026) show that masked diffusion LMs without a
  timestep input linearly encode the global mask fraction after the first
  attention layer. Patching that direction changes predictions.
- **For DiffGRM.** The decoder has no step input either. The depth-5 models
  are never trained on the all-masked state, but decoding starts there.
- **Implement.**
  - A linear probe for the number of revealed digits on decoder residuals at
    every state.
  - A difference-of-means direction per layer.
  - A patch that moves the all-masked state's representation to the
    "one digit revealed" value, then rescore with `score_states`.
- **Predicts.** At depth 5, the probe sees the all-masked state as out of
  range, and patching moves first-step choices toward what the model does
  from trained states. If so, it is a mechanism for the depth-5 loss in
  Controlled 1. At depth 3–4 (trained on all-masked), patching should do
  little.

**3. Commitment and regret along the reveal path.**

- **Basis.** Zhou, Roy & Gangadharaiah (2026) show that attributes commit at
  different denoising steps (topic in the first 2 %, sentiment over 20 %),
  and that interventions work best when timed to commitment. Zhou, Wang &
  Van de Cruys (2026) measure a per-prompt "commitment horizon" after which
  guidance can be switched off. DLM-Scope (Wang et al., 2026) tracks
  pre-mask stability and post-decode drift with SAE features.
- **Implement.** Two measures, from `score_states` along each decode path:
  - **Commitment step.** At each partial state on the path, take the argmax
    of every still-masked digit. A digit's commitment step is the first
    state from which that argmax equals its final value.
  - **Regret.** Once all digits are revealed, re-score each digit with the
    others fixed (leave-one-out). Regret is the rate at which a committed
    digit is no longer the argmax.
- **Use.** Regret measures what remasking samplers (ReMDM, Wang et al., 2025;
  self-correcting masked diffusion, 2026) could fix. Commitment steps say
  when to intervene in the experiments below. Together they are the
  "reveal-order traces" item in the research plan.

### Tier 2: new representation work, comparable with the AR side

**4. SAEs on decoder states, split by masked and revealed positions.**

- **Basis.** DLM-Scope (Wang et al., 2026) trains separate SAEs on masked and
  unmasked positions of LLaDA and Dream. Inserting SAEs into early layers
  can even lower the loss, and deep-layer features steer far better than in
  AR models.
- **Implement.** Capture decoder residuals for the test cohort at every
  state of the confidence path (6,297 users × D states × D positions, 256-d,
  small). Fit TopK SAEs per layer, for masked and revealed positions
  separately. Run the same census as representation/exp3: copy source,
  answer, golden, own digit, and revealed-prefix variables.
- **Why.** It gives an AR vs diffusion information map with one method, and
  tests Kong, Lee & Jo (2026): "sharp, localized specialization in ARMs gives
  way to distributed integration in MDMs".

**5. Dose–response attention control instead of knockout.**

- **Basis.**
  - Prompt-to-Prompt (Hertz et al., 2022) edits outputs by scaling or
    replacing cross-attention maps.
  - Zaleska et al. (2026) find that ambiguous generative choices in
    diffusion models are resolved mainly in self-attention, and intervene
    there only.
  - Basu et al. (2023, 2024) find that knowledge in text-to-image diffusion
    is spread over several components, not one.
- **Implement.** Scale, rather than remove, the cross-attention logit to one
  history slot (α ∈ {0, 0.25, 0.5, 2, 4}). Separately, scale the decoder
  self-attention between digit positions. Use the existing mask pre-hook,
  with an additive bias in place of a boolean mask.
- **Use.** Dose–response curves show whether the copy saturates or is
  linear, and whether the choice between sibling codes (exp10's question,
  for DiffGRM) is made in self- or cross-attention.

**6. Steering repeat vs explore with guidance.**

- **Basis.**
  - Shnaidman et al. (2025): a single contrastive direction steers masked
    diffusion LMs, applied over the whole reverse process, mostly in early
    steps.
  - Ye, Rojas & Tao (2025) characterise classifier-free guidance in masked
    discrete diffusion.
  - The guidance-schedule work (arXiv 2507.08965) finds that strong early
    guidance hurts and late guidance helps.
  - DLM-SWAI (2026) biases token distributions before unmasking, and finds
    token-level bias more reliable than activation steering.
- **Implement.** A negative-guidance decode: log p(digit | history) −
  w · log p(digit | history without the copy route). The second term uses
  the exp8 block, not an empty history, which the model never saw. Apply it
  only before the commitment step from item 3.
- **Use.** It is a controllable "explore" knob, and it causally tests whether
  copying crowds out good new-item candidates. Report new-target and variant
  HR separately (retrospective exp7).

### Tier 3: worth noting, lower priority

- **Attention sinks that move.** In diffusion LMs, sinks shift as tokens are
  unmasked (Attention Sinks in Diffusion Language Models, 2025). Check whether DiffGRM's decoder parks
  attention on one history slot or one digit position, and whether that
  moves with the reveal order. If it does, knockouts should control for it.
- **Masks as distractors.** Piskorz et al. (2025) show appended masks hurt
  context use in masked diffusion LMs. DiffGRM's SID length is fixed, but the
  next-two model's second-item masks could distract the first item. Compare
  first-item scores with the second item masked vs absent.
- **Learned unmasking order.** A learned order policy (Adaptive Order
  Policies, 2026) is a training change, not an intervention. Commitment
  traces (item 3) would show whether a learned order is worth training.

## Diffusion recommenders to cite beside DiffGRM

LLaDA-Rec (2025), MaskGR (Masked Diffusion for Generative Recommendation,
2025), MDGR (Mu et al., 2026) and "Diffusion Language Model for
Recommendation" (2026) all generate SIDs with masked diffusion. We found no
interpretability or causal analysis in any of them. The items above would be
the first for a diffusion recommender.

## References

- Catruna & Radoi (2026). *Induction in Both Directions: A Mechanistic
  Analysis of In-Context Learning in Masked Diffusion Language Models.*
  https://arxiv.org/abs/2607.15893
- Wang et al. (2026). *DLM-Scope: Mechanistic Interpretability of Diffusion
  Language Models via Sparse Autoencoders.* https://arxiv.org/abs/2602.05859
- Zhou, Roy & Gangadharaiah (2026). *Steering Without Breaking:
  Mechanistically Informed Interventions for Discrete Diffusion Language
  Models.* https://arxiv.org/abs/2605.10971
- Zhou, Wang & Van de Cruys (2026). *Commitment Before Realization: When
  Classifier-Free Guidance Becomes Unnecessary in Masked Diffusion Language
  Models.* https://arxiv.org/abs/2608.08082
- Shnaidman et al. (2025). *Activation Steering for Masked Diffusion Language
  Models.* https://arxiv.org/abs/2512.24143
- *DLM-SWAI: Steering Diffusion Language Models Before They Unmask* (2026).
  https://arxiv.org/abs/2605.29626
- Kong, Lee & Jo (2026). *Mechanism Shift During Post-training from
  Autoregressive to Masked Diffusion Language Models.*
  https://arxiv.org/abs/2601.14758
- Piskorz et al. (2025). *Masks Can Be Distracting: On Context Comprehension
  in Diffusion Language Models.* https://arxiv.org/abs/2511.21338
- *Attention Sinks in Diffusion Language Models* (2025).
  https://arxiv.org/abs/2510.15731
- Wang et al. (2025). *Remasking Discrete Diffusion Models with
  Inference-Time Scaling* (ReMDM). https://arxiv.org/abs/2503.00307
- *Learn from Your Mistakes: Self-Correcting Masked Diffusion Models* (2026).
  https://arxiv.org/abs/2602.11590
- *Adaptive Order Policies for Masked Diffusion* (2026).
  https://arxiv.org/abs/2606.00295
- Ye, Rojas & Tao (2025). *What Exactly Does Guidance Do in Masked Discrete
  Diffusion Models.* https://arxiv.org/abs/2506.10971
- *Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical
  Insights with High-Dim Impact* (2025). https://arxiv.org/abs/2507.08965
- Hertz et al. (2022). *Prompt-to-Prompt Image Editing with Cross Attention
  Control.* https://arxiv.org/abs/2208.01626
- Zaleska et al. (2026). *Attention, May I Have Your Decision? Localizing
  Generative Choices in Diffusion Models.* https://arxiv.org/abs/2604.06052
- Basu et al. (2023). *Localizing and Editing Knowledge in Text-to-Image
  Generative Models.* https://arxiv.org/abs/2310.13730
- Basu et al. (2024). *On Mechanistic Knowledge Localization in Text-to-Image
  Generative Models.* https://arxiv.org/abs/2405.01008
- LLaDA-Rec (2025) https://arxiv.org/abs/2511.06254
- MaskGR (2025) https://arxiv.org/abs/2511.23021
- MDGR (2026) https://arxiv.org/abs/2601.19501
- DiffGRM (WWW 2026) https://arxiv.org/abs/2510.21805
- *Diffusion Language Model for Recommendation* (2026)
  https://arxiv.org/abs/2607.21519
