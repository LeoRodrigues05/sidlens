# What a DiffGRM recommendation looks like while it is being decoded

Declared 2026-10-13, before any forward pass of this experiment. The code was
tested on a random-weight model only; no trained checkpoint was reachable when
this was written (the cluster's `/l` filesystem was down).

This is observational. It records the decoding process; it does not show which
parts of it matter. `controlled/exp12` and `exp13` (to be declared after this
experiment's descriptive results) test that.

## Why

DiffGRM is masked discrete diffusion over one SID. A step takes a partial SID
and returns a distribution over codes at every digit position; the search
commits one (position, code); the next step starts from the new partial SID
alone. Nothing else passes between steps, and the model has no step input.
`matched_decode.decode` returned only the final list, so no record exists of
which digit is revealed first, when the answer is settled, or what the beam
holds at each step. This experiment makes that record. It is item 3 of
`docs/paper/rewrite-20261004/RUN_PLAN.md`.

## Fixed design

**Cells.** All 20 trained next-item checkpoints. Indices 0–17 are controlled
exp1's grid in its order (MQ, RQ-KMeans, RQ-VAE × depth 3, 4, 5 × width 128,
512); 18 is `diff-next1-MQ-5cb-256` and 19 is `diff-next1-rqvae-3cb-256`.

**Cohort.** The 6,297-user diffusion next-item cohort, in loader order. It is
never joined to the AR cohort.

**Decoder.** `matched_decode.decode`, confidence policy, catalogue filter after
the last step, top 10, batch 32, decoder chunk 1024, fp32. Beam widths 1
(greedy), 64 and 256. Beam 256 is the primary beam.

**Paths.** Two paths per user are re-scored state by state:

- `greedy`: the single path of the beam-1 decode, legal or not.
- `top1`: the best path of the first-ranked legal SID at beam 256. It is
  missing for a user whose beam holds no catalogue SID.

A path of depth n has n states: state 0 is fully masked and state t has the
first t commits written.

**Recorded per path state**

- The digit and code committed next, and the accumulated score.
- At each still-masked position: the 8 most probable codes with their
  log-probabilities, the entropy, and the target code's log-probability and
  rank. Nothing is read at a revealed position: the training loss never
  covered it.
- The current guess (revealed codes kept, masked digits by argmax) and whether
  it is a catalogue SID.
- A catalogue ranking: every catalogue SID scored by the sum of its codes'
  log-probabilities over the still-masked digits, and excluded if it
  contradicts a revealed digit. Recorded: the best-scored SID, and the target's
  SID-level rank and item-level rank bounds. This is the recommendation the
  state would give if decoding stopped there. It is a product of per-digit
  marginals, not a probability over items.
- The same per-position readout after each of the four decoder blocks (`ln_f`
  and the output embedding applied to the block's output).

**Recorded per beam width and step.** Paths kept, distinct partial states, and
paths and distinct states that some catalogue SID still completes; how many
commits went to each digit position; the final top-10 list with scores.

**Target lattice.** For each user, all 2ⁿ − 1 partial states of the target
SID with at least one digit masked: the target code's log-probability and rank
and the entropy at every masked position. From it: the best reveal order of
the target (the maximum over all n! orders, which is the score an unbounded
confidence beam gives the target) and the greedy order.

**Regret.** For the `top1` SID, each digit re-masked in turn with the others
revealed: the committed code's log-probability and rank.

**Training-style order.** The per-digit confidence at the fully masked state
from the trainer's ranking pass (`use_cache=False`, max probability). With
`guided_select = least` the trainer visits, per example, the states of
revealing digits in descending order of this confidence, for 1 to
`guided_steps = min(n, 4)` masked digits. A decode-path state is "in the
trained family" if its revealed set is a prefix of that order and its number of
masked digits is at most `guided_steps`.

## Declared descriptive statistics

| Id | Statistic |
|---|---|
| S1 | Distribution of the first revealed digit, and of the full reveal order, on the `greedy` and `top1` paths, by cell. Share of path states in the trained family. |
| S2 | Commitment step of each digit: the first state from which the current guess at that digit equals the path's final code at every later state. SID commitment step: the maximum over digits. |
| S3 | HR@10 of the catalogue ranking at each state (SID level, and item-level bounds), next to the final beam-256 HR@10. |
| S4 | Share of kept paths that are duplicates of another path's partial state, and share that no catalogue SID completes, by step, beam width and depth. |
| S5 | By decoder block: share of masked positions whose top code equals the last block's, and the target code's mean log-probability. |
| S6 | From the lattice: the change in the target code's log-probability at digit j when digit i alone is revealed, by (i, j) and cell; best-order minus greedy-order target score. |
| S7 | Share of `top1` SIDs with at least one digit that is no longer the top code when re-masked. |
| S8 | At state 0: mean max probability minus top-1 accuracy per digit, by depth. |

SID-level and item-level numbers are reported separately.

## Expectations recorded before outcomes

- **E1.** The `greedy` path reveals digit 1 first for a larger share of users
  on every RQ-KMeans and RQ-VAE cell than on the MQ cell of the same depth and
  width (RQ digits are a hierarchy; MQ digits are parallel partitions).
- **E2.** The state-0 catalogue ranking has lower SID HR@10 than the final
  beam-256 list in at least 15 of 20 cells: iterating buys something over one
  factorized prediction.
- **E3.** S8 is larger at depth 5 than at depth 3 and 4 for the same quantizer
  and width in at least 4 of 6 quantizer × width pairs: depth-5 models never
  train on the fully masked state.

These are predictions, not hypotheses with a decision rule. Each is reported
as held or failed.

## Validation, before any number counts

- **V1.** exp1's pilot checks on 8 users (`verify_model_and_decoder`).
- **V2.** The observer changes nothing: beam-64 lists and scores with and
  without it are identical on the first batch.
- **V3.** Re-scored path states reproduce the search's accumulated path scores
  (`atol = 1e-4`) for every traced path.
- **V4.** The last block's readout equals the model's distribution
  (`atol = 1e-5`).
- **V5.** Where the target SID is in the beam-256 list, its score does not
  exceed the lattice's best-order score by more than 1e-4.
- **V6.** Cells 0–17: the beam-64 and beam-256 lists, counts and scores equal
  controlled exp1's archive (`derived/controlled/exp1_matched_beam/181944`),
  lists and counts exactly and scores within `atol = rtol = 1e-4`. If the
  archive is not on the cluster the run stops, unless
  `--allow-missing-archive` is passed; the run then records that V6 was not
  checked and its results are provisional.

## Statistics

Means and shares with a user bootstrap (`UserBootstrap`, 2,000 draws, seed
20261013), all cells of a sampled user kept together. Intervals exclude
training-seed uncertainty: there is one checkpoint per cell.

## Limits

- Observational. A readout shows what a state's distributions say, not what
  the model uses.
- The block-wise readout assumes later blocks add nothing; before the last
  block it is a description.
- Two paths per user are traced, not the whole beam.
- The lattice reveals target codes, so its states are teacher-forced.
- Training visits one reveal order per example, ranked by a different network
  from the one decoded with; most traced states are outside that family.
- One seed per checkpoint.
