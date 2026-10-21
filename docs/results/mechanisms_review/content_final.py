"""Final important experiments tab: the experiments to run on every kept checkpoint of the final training run.

Three lines (codebooks and collisions, Qwen-AR, DiffGRM), each with its main experiments first and its supplementary
experiments after, plus the shared steps that come before all of them. Every experiment on the other tabs must map
to a final experiment in COVERAGE; `page` raises if one is missing, if a final experiment refers to an unknown review
experiment, or if a claim names an unknown final experiment, so the plan cannot silently drop or invent an experiment.

Facts about the final run come from the OneDiffRec handoff (README "The full grid, on retrained SID tables",
scripts/handoff/common.sh, scripts/run_experiment.py; commits up to 7ae3512, 2026-10-16). Costs are the run times of
our own jobs (controlled exp11 status files; exp5/12/13/14 validation.json). Wording follows docs/paper/WRITING.md.
"""

from __future__ import annotations

import html

RUN = """
<p>We assume the final run is the full grid of the OneDiffRec handoff (<code>run_grid.sh</code>, run name
<code>grid-v1</code>, commits up to 16 October 2026). If it turns out narrower, for example only
<code>run_next2.sh</code> and <code>run_zcr.sh</code>, the same experiments apply to the configurations that exist.</p>
<ul>
<li><b>Four blocks:</b> Industrial and Office, each with next-item and next-two prediction.</li>
<li><b>Configurations per block:</b> 27 SID tables refitted from the item embeddings (v3: the RQ-KMeans and RQ-VAE
tables of v2, and MQ refitted on 15 October with the coauthors' TokenRec code), each with native SIDs. A table also gets
a ZCR model when more than 5&nbsp;% of its items share a SID, and RQ-VAE 3×256, 4×256 and 5×256 get one by name.
<code>build_tables.sh --status</code> gives the exact list. Every configuration has an AR and a DiffGRM model, seed 42.</li>
<li><b>Kept weight states.</b> AR: snapshots at epochs 0.5 and 1.5, and the best model by validation loss (training
stops early around epoch 4; the best model is near epoch 2.5 on Industrial). DiffGRM: the best model by validation NDCG,
a snapshot every 20 epochs, and the training loss and validation metrics of every epoch.</li>
<li><b>Cohorts:</b> <code>disjoint_horizon_v3</code>. AR and DiffGRM are tested on the same target events, and both
see up to 50 history items.</li>
<li><b>Test decoding:</b> exact SIDs, 50 beams for both models. AR uses plain beam search. DiffGRM reveals digits in a
fixed left-to-right order and keeps only codes that some catalog SID allows at each step (<code>joint_catalog_v1</code>);
a next-two DiffGRM model decodes both items in one pass. Each run saves its ranked predictions, scores, collision strata
and item-level (CCE) metrics.</li>
</ul>
"""

CHANGES = [
    ("The SID tables are new.", "Every map result is recomputed (line M). Results tied to the old tables, such as the "
     "balanced RQ-KMeans 5×512 table and the old MQ fit, may not carry over."),
    ("AR and DiffGRM are tested on the same events.", "Their estimates can be compared user by user with paired "
     "intervals. With the old cohorts we could only set them side by side."),
    ("ZCR twins come from the same pipeline.", "Each ZCR model has a native twin trained the same way, so M3 and AR4 "
     "can test what collisions cause."),
    ("Office is a second catalog.", "Category statistics and title-based variant detection need Office metadata, which "
     "we have not downloaded. The other measures need only the embeddings and the run outputs."),
    ("Weight states along training.", "The cheap teacher-forced experiments can run at every kept state: three states "
     "per AR run and one every 20 epochs per DiffGRM run."),
    ("Our code must read the v3 inputs first.", "That means v3 AR prompts with 50 history items, v3 cohorts, two-item "
     "DiffGRM checkpoints and the v3 DiffGRM decoder. The reproduction check becomes: our decoder reproduces each run's "
     "saved test predictions."),
    ("DiffGRM's test decoder changed.", "v3 reveals digits left to right and checks legality at every step. Our earlier "
     "search results (duplicate paths under confidence order, illegal SIDs under fixed orders) describe the old decoder. "
     "DG5 compares the two decoders on the same checkpoints."),
]

SCOPES = [
    ("Every configuration", "All 27 native tables of a block and every ZCR twin."),
    ("Focal set", "RQ-KMeans 3×128, RQ-VAE 4×128 and MQ 3×128, with their ZCR twins where they exist. The first two "
     "are the maps of all our earlier work, so their final numbers can be set beside the provisional ones."),
    ("Every kept state / best state", "All weight states listed above, or only the model the run selected."),
    ("User sample", "A fixed set of test users per catalog and task (2,000 proposed), drawn once with a recorded seed "
     "and shared by both paradigms and all weight states. The full cohort is every test user."),
]

COSTS = """
<p>Measured on our earlier jobs, one GPU unless noted:</p>
<ul>
<li>AR teacher-forced battery (controlled exp4 on one split plus exp3's replacement part): 20–31 minutes per model
for 3,680 rows with histories of at most 10 items.</li>
<li>AR beam search with copy blocks (controlled exp6, six decodes of 3,681 events): 62–80 minutes per model.</li>
<li>DiffGRM experiments (representation exp5; controlled exp12 on CPU, exp13 and exp14 on an A100): 4–13 minutes per
model for 6,297 users.</li>
</ul>
<p>v3 AR prompts hold up to 50 history items. We guess they cost 3–5 times as much per row; measure it on the first v3
model. A block has up to about 47 configurations (27 native and up to about 20 ZCR twins), so the AR experiments
dominate the cost: AR1 at every kept state on both next-item blocks would take a few hundred GPU-hours at the proposed
sample size. The scope proposed in AR1 (every state on Industrial, the best state on Office) cuts that by about 30 %.
DiffGRM experiments cost minutes per model and can run everywhere.</p>
"""

SHARED = [
    dict(id="S1", role="Shared", title="Register, load and check the final run",
         question="Can we load every model exactly and reproduce what the run reported?",
         measures="Hash and register every kept weight state, SID table and ZCR codebook. Make the loaders read v3 "
                  "prompts, v3 cohorts, two-item DiffGRM checkpoints and the v3 DiffGRM decoder. Reproduce each best "
                  "state's saved test predictions with our scoring and decoding code. Draw the user samples.",
         checks="No result may use a model whose predictions we cannot reproduce. This check replaces the archive "
                "check of the earlier experiments.",
         runs="Every run of the four blocks.",
         cost="CPU for hashing; one decode per best state (GPU for AR).",
         builds=[], builds_text="The retrain acceptance check (substrate exp1), exp11's list check, "
                                "<code>analysis/decode_checks.py</code>.",
         new="Needed before any experiment below.", current="—", paper="Methods: reproduction and registration."),
    dict(id="S2", role="Shared", title="Data audit for each catalog",
         question="Which test targets are repeats, SID partners, near-duplicate variants, duplicated records or copied "
                  "reviews?",
         measures="The review day of every event; duplicated records (same user, item and day); reviews copied across "
                  "variant ASINs. For every test target and table, its class: the item is in the history (repeat), "
                  "shares a history item's SID (SID partner), is a near-duplicate variant of a history item, or is new.",
         checks="Write the protocol first; the copied-review numbers so far are exploratory.",
         runs="Both catalogs, every table.",
         cost="CPU, about a day per catalog.",
         builds=["ar-time", "ar-credit"], builds_text="The copied-review lead in <code>RQ_PLAN.md</code> §5.",
         new="Office needs metadata for title-based variant detection.",
         current="Duplicated records are 2.7 % of AR test rows and 11 / 17 % of Hit@10 hits.",
         paper="Every accuracy number is reported with and without duplicated and copied targets. M3, M4, AR2, AR7 "
               "and DG4 split their results by these classes."),
    dict(id="S3", role="Shared", title="Score table",
         question="Do score comparisons between the paradigms change with catalog, task and map?",
         measures="SID-level and item-level Hit@10 and NDCG@10 of every run, read from the saved results.",
         checks="—", runs="Every run.", cost="CPU, minutes.",
         builds=[], builds_text="The system-level comparison in the paper (Figure 2).",
         new="v3 scores replace the archived ones.",
         current="In the archived runs, DiffGRM trails Qwen-AR by 1.78 points of Hit@10 on Industrial and leads by "
                 "6.65 on Office.",
         paper="The opening figure."),
]

LINES = [
    dict(key="m", short="Codebooks", title="Line M: codebooks and collisions", rq="RQ1; M4 also serves RQ4",
         intro="These experiments read the SID tables, the ZCR codebooks, the item embeddings and the labels. M3 and M4 "
               "also read each run's saved predictions. None of them needs model weights.",
         experiments=[
             dict(id="M1", role="Main", title="What each digit encodes",
                  question="How does each quantizer split items into digits?",
                  measures="For each table: the share of embedding variance explained by each digit alone and given "
                           "its prefix; the codes in use per digit; the category information each digit adds given the "
                           "prefix, against a null that shuffles items within each parent; category purity by prefix "
                           "length, against a shuffled null.",
                  checks="Nulls keep group sizes.",
                  runs="All 54 native tables and every ZCR codebook. Category measures on Industrial until Office "
                       "metadata is downloaded.",
                  cost="CPU, hours.",
                  builds=["maps-stats"],
                  new="Refitted tables; a new MQ fit.",
                  current="In the 4×128 tables, digit 1 alone explains 53 % (RQ-KMeans) and 34 % (RQ-VAE) of the "
                          "variance and each later digit 8–18 %; each MQ digit explains 36–38 %. Category purity stops rising after digit 3. RQ-VAE uses "
                          "2–31 % of its digit-1 codes in 8 of 9 fits.",
                  paper="The per-digit information figure (RQ1)."),
             dict(id="M2", role="Main", title="Why items share a SID",
                  question="What makes distinct items end up with the same SID in each family?",
                  measures="The share of items whose SID is shared; how many of those have identical or near-identical "
                           "embeddings; the rate before and after RQ-VAE's last-digit reassignment, applied to every "
                           "table; items on a near-origin code at each level; redundancy between MQ digits (NMI against "
                           "an independent-digit null).",
                  checks="Refit each RQ-KMeans table and confirm it reproduces (the fit depends on the FAISS thread "
                         "count).",
                  runs="All 54 native tables.",
                  cost="CPU, about a day with refits.",
                  builds=["maps-collisions"],
                  new="v3 RQ-KMeans tables are unbalanced FAISS fits, so the old 5×512 balancing result does not apply.",
                  current="RQ-VAE's near-zero rates come from the reassignment, which takes RQ-KMeans tables from 10–45 % "
                          "to the 22 items with identical inputs. Near-duplicates are 73–85 % of collided items in the "
                          "deepest RQ-KMeans fits. MQ digits repeat each other (NMI 0.50–0.70).",
                  paper="The collision-cause table (RQ1)."),
             dict(id="M3", role="Main", title="What removing collisions changes: native against ZCR twins",
                  question="Does giving every item its own SID change item-level accuracy, and how much of the SID-level "
                           "score was credit for collisions?",
                  measures="For each native/ZCR pair: SID-level and item-level Hit@10 and NDCG@10, their gap, and hits by "
                           "target class (S2), with paired user intervals.",
                  checks="Both twins come from the same pipeline and seed; compare only within a pair.",
                  runs="Every pair, both paradigms, both catalogs, best state. Item-level measures need next-item runs "
                       "(pair CCE is undefined); next-two pairs report exact ordered-pair scores.",
                  cost="CPU; saved predictions only.",
                  builds=["maps-collisions", "ar-collisions"],
                  new="New experiment; needs the ZCR twins. Write the protocol first.",
                  current="No ZCR model analysed yet. In the archived AR lists, SID Hit@10 exceeds item-level Hit@10 by "
                          "4.5 points.",
                  paper="A new RQ1 table or figure."),
             dict(id="M4", role="Main", title="What SID-level hits count, across the grid",
                  question="How much of each reported score comes from collisions and from targets the history already "
                           "holds?",
                  measures="For every run: SID-level against item-level Hit@10; hits on targets whose SID is shared; the "
                           "share of hits from each target class (S2), with and without duplicated and copied targets; "
                           "the SID minus item gap against the table's collision share, per paradigm.",
                  checks="Same target classes for both paradigms (shared events).",
                  runs="Every configuration, both paradigms, both catalogs, best state.",
                  cost="CPU; saved predictions only.",
                  builds=["ar-collisions", "ar-credit"],
                  new="DiffGRM gets the same audit as AR, on the same events.",
                  current="In the archived AR lists, SID-level Hit@10 exceeds item-level Hit@10 by 4.5 points on average "
                          "and by 14.1 points on targets whose SID is shared.",
                  paper="The RQ4 figure of SID-level against item-level credit."),
             dict(id="M5", role="Supplementary", title="Naming and the explorer",
                  question="Can a category name pick out a prefix, and do items under a prefix get bought together?",
                  measures="Naming AUC per prefix; co-purchase lift within prefixes; the offline explorer rebuilt on the "
                           "v3 tables.",
                  checks="Shuffled-label null.",
                  runs="Industrial tables; Office once its metadata is downloaded.",
                  cost="CPU, hours.",
                  builds=["maps-stats"],
                  new="v3 tables.", current="Computed for the 27 old tables (structure exp1); not used in the paper so far.",
                  paper="Appendix; the explorer as supplementary material."),
         ]),
    dict(key="ar", short="Qwen-AR", title="Line AR: Qwen-AR", rq="RQ2–RQ4",
         intro="These experiments load AR weights. AR1 is the only AR experiment proposed for every kept state; the others "
               "run on the best state.",
         experiments=[
             dict(id="AR1", role="Main", title="History use and prefix-matched copying, along training",
                  question="How much does each digit depend on the most recent history item and on history items that "
                           "match the prefix written so far, and when during training does this appear?",
                  measures="Under teacher forcing: the cost to the true digit-1 code of replacing the history item at "
                           "recency r (A1, r up to 50); the share left when the replacement keeps the item's first m "
                           "digits (A3); changing digit d of a history item that matches the target on the earlier "
                           "digits, minus the same change on a non-matching item (H1); knocking out the attention edge from "
                           "digit d's readout to the matching item's digit-d token, minus a control edge (H2b); the share "
                           "of H2b in the later half of the layers.",
                  checks="No-op knockouts are bit-exact on every row.",
                  runs="Industrial: every next-item configuration at every kept state. Office: every configuration at "
                       "the best state, the focal set at every state. Test-cohort user sample; the validation cohort as "
                       "a replication at the best state.",
                  cost="20–31 GPU-minutes per model and split on the old short prompts; longer with 50 items. The largest "
                       "cost in the plan.",
                  builds=["ar-history", "ar-copy", "ar-grid"],
                  new="50-item histories; ZCR twins; weight states; the same users as DG1.",
                  current="In all 12 retrained models: A1 0.37–0.78 nats, H1 0.89–2.18, H2b 0.20–0.60, and 98–100 % of "
                          "the edge effect in layers 14–27.",
                  paper="The history and copying table and the grid figure (RQ3); a training panel if the states differ."),
             dict(id="AR2", role="Main", title="What copying buys in beam search",
                  question="Which hits depend on copying?",
                  measures="During the v3 beam search, when a beam predicts digit d, block its reads of digit d of every "
                           "history item whose earlier digits equal the beam's own prefix (at digit 1, of every history "
                           "item), in the later half of the layers. Control: block the same number of non-matching "
                           "items. Change in Hit@10 by target "
                           "class (S2), at SID and item level.",
                  checks="The unblocked search reproduces the run's saved predictions.",
                  runs="Every next-item configuration (native and ZCR), both catalogs, best state; Office on the user "
                       "sample.",
                  cost="About 10–13 GPU-minutes per decode of 3,681 events on the old prompts; three decodes per model "
                       "(unblocked, blocked, control).",
                  builds=["ar-decoding", "ar-credit", "ar-grid"],
                  new="Plain 50-beam search is now the reported decoder; the target classes come from S2.",
                  current="The copy block costs 7.5–37.0 points of Hit@10 on repeat targets and 0.1–4.3 on new targets "
                          "(12 models). In RQ-KMeans 3×128, SID partners carry 56 % of the loss, and 28 % of the top-1 "
                          "benefit remains at item level.",
                  paper="The RQ4 copy-benefit figure, beside DG4 on the same events."),
             dict(id="AR3", role="Main", title="What each layer holds",
                  question="From which layer can the model's own answer be read, does any layer hold information about new "
                           "targets beyond the history, and when does the copied item become the answer?",
                  measures="Activations at the digit readout positions of every layer; logit lens and tuned lens for the "
                           "model's own answer, the true target and the copy source, per digit; whole-SID decoders; probes "
                           "for the target on new-target rows against the rule \"predict the recent item's digit\".",
                  checks="User-disjoint folds; majority and shuffled-label baselines.",
                  runs="Industrial: every native next-item configuration at the best state. The focal set at every kept "
                       "state and on Office.",
                  cost="Captures, estimated at about 15 GPU-minutes per model; lenses and probes on CPU.",
                  builds=["ar-lens", "ar-infomap"],
                  new="Readout curves across the grid instead of two models.",
                  current="The final first-digit choice equals the recent item's first digit in 64 / 53 % of rows "
                          "(RQ-KMeans 3×128 / RQ-VAE 4×128). The preference for the copy appears at layers 20–23 in "
                          "RQ-KMeans and at layers 11 and 20 in RQ-VAE. No layer reads the target above the copy rule.",
                  paper="The layer readout figure (RQ2)."),
             dict(id="AR4", role="Main", title="The last digit: copy or avoid, native against ZCR",
                  question="When a history item matches the target on every digit but the last, does the model copy its "
                           "last code, and does removing collisions change that?",
                  measures="Knock out the edge from the last digit's readout to that item's last-digit token, minus a "
                           "control edge. Change in the item's own code and in the target's code, in rows where the target "
                           "has that item's SID and in rows where it is a sibling. Native minus ZCR twin.",
                  checks="No-op knockouts bit-exact.",
                  runs="Every native/ZCR pair, next-item, both catalogs, best state.",
                  cost="Teacher forcing, about as much as AR1 per model.",
                  builds=["ar-lastdigit"],
                  new="The causal test the ZCR runs were commissioned for.",
                  current="RQ-KMeans 3×128 copies the last digit (−0.17 / −0.34 nats, test / validation). RQ-VAE 4×128 "
                          "avoids it in sibling rows: blocking raises top-1 = that item's code by 11.8 points. RQ-VAE's "
                          "declared pooled estimate includes 0.",
                  paper="An RQ3 paragraph; a figure if the twins change consistently."),
             dict(id="AR5", role="Supplementary", title="Where the recent item's information goes",
                  question="Through which positions and heads does the history item's information reach the answer?",
                  measures="Residual-stream patching by layer and position (item tokens, response header, target "
                           "positions); a screen of attention heads for the copy edge, with held-out users.",
                  checks="A patch must fire exactly once; no-op patches bit-exact.",
                  runs="Focal set, best state.", cost="GPU, hours per model.",
                  builds=["ar-history", "ar-copy"],
                  new="—",
                  current="The item's information stays at its tokens until about layers 16–20 and moves to the header "
                          "and target positions in layers 19–24; the top 5 heads carry 8–9 % of the edge effect.",
                  paper="Appendix, or one sentence in RQ3."),
             dict(id="AR6", role="Supplementary", title="Sparse features of copying",
                  question="Which features carry the copy, and are they used?",
                  measures="TopK SAEs on the residual stream at every second layer; a census of what each feature tracks; "
                           "error-preserving ablation of copy features and prefix-match features at the readout, against "
                           "control features.",
                  checks="Reconstruction error and splice tests before any ablation counts.",
                  runs="Focal set, best state; epoch 1.5 as well if AR1 shows copying changes between states.",
                  cost="Heavy: training-split captures and SAE training.",
                  builds=["ar-sae", "ar-infomap"],
                  new="—",
                  current="Copy features are causal from layer 20 (0.10–0.36 nats); RQ-KMeans' sharp prefix-match "
                          "features have no effect at the readout.",
                  paper="Appendix; one RQ2 sentence."),
             dict(id="AR7", role="Supplementary", title="Order and time",
                  question="Does the model follow list position or review time?",
                  measures="Swap the two most recent history items, split by whether they share a review day (S2); the "
                           "rate at which the model copies the recent item's first digit, by the item's age, against the "
                           "same rate in the data.",
                  checks="Swaps of items with identical SIDs are bit-exact.",
                  runs="Every native next-item configuration, best state, user sample.",
                  cost="Teacher forcing, below AR1.",
                  builds=["ar-order", "ar-time"],
                  new="50-item histories; repeat without copied-review pairs (S2).",
                  current="RQ-KMeans 3×128 follows position (position effect 0.36–0.38), RQ-VAE 4×128 weakly (0.07–0.10); "
                          "a swap that keeps the review days flips 50 % of first-digit choices.",
                  paper="Appendix."),
             dict(id="AR8", role="Supplementary", title="Next-two: item 2 from item 1",
                  question="Is the second predicted item copied from the first?",
                  measures="Replace item 1 or the most recent history item and measure item 2's digits; prefix-matched "
                           "copying from item 1 to item 2; patching item 1's tokens (focal set); exact ordered-pair hits "
                           "from the saved lists.",
                  checks="No-op controls bit-exact.",
                  runs="Every next-two configuration, both catalogs, best state, user sample; the focal set at every "
                       "kept state.",
                  cost="Teacher forcing, about AR1 per model.",
                  builds=["ar-nexttwo", "ar-nexttwo-assoc"],
                  new="The grid instead of one model.",
                  current="Replacing item 1 costs item 2's first digit 0.60 nats, against 0.20 for the most recent history "
                          "item (one model, MQ 4×256).",
                  paper="A short RQ3 subsection, beside DG9."),
             dict(id="AR9", role="Supplementary", title="Where the answer first goes wrong",
                  question="At which digit do wrong predictions first leave the target?",
                  measures="The first digit at which each saved top-1 prediction differs from the target.",
                  checks="—", runs="Every run of both paradigms, saved predictions.", cost="CPU, minutes.",
                  builds=["ar-prefix-errors"],
                  new="DiffGRM gets the same count.",
                  current="In 82 % of archived AR rows, the top-1 prediction first differs from the target at digit 1.",
                  paper="One sentence that motivates the focus on digit 1."),
         ]),
    dict(key="dg", short="DiffGRM", title="Line DG: DiffGRM", rq="RQ2–RQ4",
         intro="DiffGRM experiments cost minutes per model, so most of them can run on every configuration. DG1 and the "
               "blocks of DG2 run at every kept state.",
         experiments=[
             dict(id="DG1", role="Main", title="History use and prefix-matched copying, along training",
                  question="Does DiffGRM depend on the most recent item and copy from prefix-matching items as AR does, and "
                           "when does this appear?",
                  measures="The cost to the true digit-1 code of replacing the item at recency r; changing a matching "
                           "against a non-matching history item (H1); knocking out the decoder's direct cross-attention "
                           "read of the matching item.",
                  checks="No-op controls bit-exact.",
                  runs="Every next-item configuration (native and ZCR), both catalogs, every kept state, on the same user "
                       "sample as AR1, so AR minus DiffGRM differences get paired intervals.",
                  cost="Minutes per weight state.",
                  builds=["dm-history"],
                  new="Weight states; ZCR twins; paired with AR1.",
                  current="Replacing the recent item costs 0.56 / 0.48 nats (AR 0.57 / 0.51); copying 0.35 / 0.96 against "
                          "about 0; the direct read carries almost nothing (−0.007 / −0.004).",
                  paper="Beside AR1 in the history and copying table (RQ3)."),
             dict(id="DG2", role="Main", title="Where the copy travels",
                  question="How does the recent item reach the digit being predicted?",
                  measures="Block the other slots from reading the recent item in the encoder, block the decoder's direct "
                           "read, block both, and the same blocks on a control item. Paired runs with the recent item "
                           "replaced: restore the other slots' encoder states or the item's own slot, and one block's "
                           "cross-attention output at one digit. Probe the recent item's first code from the other slots.",
                  checks="Patching every cross-attention output reproduces the donor run; no-op controls bit-exact.",
                  runs="Every next-item configuration at the best state; the blocks and the restored shares also at every "
                       "kept state.",
                  cost="Minutes per model on CPU.",
                  builds=["dm-route", "dm-patching", "dm-encoder"],
                  new="Weight states: does the encoder spread form before the direct read?",
                  current="The other slots restore 0.83–0.90 of the first-digit effect, the item's own slot 0.32–0.53. "
                          "Blocking both routes costs 0.24 / 0.60 nats, about two thirds of the replacement effect.",
                  paper="The DiffGRM route figure (RQ3), beside AR1's edge knockout."),
             dict(id="DG3", role="Main", title="What each step and block holds, and which step uses the history",
                  question="When does the answer become readable, and at which reveal step does the history decide it?",
                  measures="At every reveal step and decoder block: the readout of the model's own answer, the target and "
                           "the copy source, and the catalog ranking from the partial SID. Step switch: replace the recent "
                           "item at one step only and count the affected users whose top-1 follows the switched history. "
                           "Encoder probes for each slot's own code and for the target on new-target rows.",
                  checks="Switching at no step reproduces the plain decode, and switching at every step the donor decode, "
                         "bit for bit.",
                  runs="Every next-item configuration at the best state, under the v3 decoder and under confidence order; "
                       "the focal set at every kept state.",
                  cost="Minutes per model on one GPU.",
                  builds=["dm-trace", "dm-steps", "dm-encoder"],
                  new="Under v3's fixed order, step t always reveals digit t, so step and digit effects separate.",
                  current="The first step alone gives the switched result for 16–68 % of affected users, the last step for "
                          "1–10 %. The encoder output holds no information about new targets beyond the history "
                          "(±0.011).",
                  paper="The DiffGRM readout panels (RQ2) and the step result (RQ3)."),
             dict(id="DG4", role="Main", title="What copying buys in beam search",
                  question="Which DiffGRM hits depend on copying?",
                  measures="Hold DG2's double block on the recent item through the whole v3 search (50 beams), against the "
                           "same block on a control item. Change in Hit@10 by target class (S2), at SID and item level.",
                  checks="The unblocked search reproduces the run's saved predictions.",
                  runs="Every next-item configuration (native and ZCR), both catalogs, best state.",
                  cost="Minutes per model on one GPU.",
                  builds=["dm-perturb"],
                  new="The v3 decoder; the same events as AR2.",
                  current="The block lowers SID Hit@10 by 1.0–3.7 points overall and by 11.5–38.9 on repeat targets; on "
                          "new targets the change is −0.7 to +0.4 (19 models).",
                  paper="Beside AR2 in the RQ4 copy-benefit figure."),
             dict(id="DG5", role="Main", title="Search: duplicate paths, reveal orders and merging",
                  question="How much do search settings change DiffGRM's reported score?",
                  measures="At equal beam sizes (50 and 256): v3's left-to-right decoder with legality checks; confidence "
                           "order; all six fixed orders at depth 3; confidence order with one path per partial SID "
                           "(merging, new). The share of duplicate paths and illegal SIDs in the final beam; SID-level and "
                           "item-level Hit@10.",
                  checks="Each decoder reproduces its reference first: v3 the saved predictions, confidence order the "
                         "vendored decoder.",
                  runs="Every native next-item configuration, best state, full cohort.",
                  cost="GPU, several decodes per model.",
                  builds=["dm-beam", "dm-orders", "dm-retro-order"],
                  new="The v3 decoder and merging are new conditions.",
                  current="At beam 64, confidence order loses 0.95 points of Hit@10 to a fixed order (2.6 at depth 5); at "
                          "beam 256 it wins by 0.22. Duplicates fill 57–91 % of the confidence beam.",
                  paper="The search figure (RQ4)."),
             dict(id="DG6", role="Supplementary", title="How revealed digits inform each other",
                  question="How does a revealed digit change the prediction of another digit?",
                  measures="Same history, one revealed digit set to a sibling's code: patch decoder states by block and "
                           "digit; knock out the self-attention edge from the predicted digit to the changed digit.",
                  checks="No-op controls bit-exact.",
                  runs="Every next-item configuration, best state.", cost="Minutes per model on CPU.",
                  builds=["dm-patching"],
                  new="—",
                  current="The effect reaches the predicted position after block 1 (0.65–0.96 restored); the direct edge "
                          "carries 0.37–0.89 of it, less at depth 5.",
                  paper="One RQ3 sentence; detail in the appendix."),
             dict(id="DG7", role="Supplementary", title="Encoder features",
                  question="Which encoder features carry the recent item, and can a short description rebuild a slot?",
                  measures="TopK SAEs on the encoder output, labelled by the attribute each feature tracks; ablating the "
                           "features that carry the recent item into other slots; the description autoencoder (eight "
                           "grounded words per slot) and how much of the decoder's behaviour the rebuilt states keep.",
                  checks="Splice test before ablation; an empty ablation returns the input exactly.",
                  runs="Focal set, best state.", cost="Minutes per model on one GPU.",
                  builds=["dm-encoder"],
                  new="—",
                  current="The spread features are causal in 19 of 20 models (up to 0.36 nats); rebuilt states keep "
                          "0.39–0.85 of the behaviour.",
                  paper="Appendix."),
             dict(id="DG8", role="Supplementary", title="Order and position",
                  question="Does DiffGRM follow the item or its slot, and do absolute positions matter?",
                  measures="Swap the two most recent items; shift the whole history by 1 and 5 slots, once with padding at "
                           "the start and once by dropping the oldest items so that no padding appears.",
                  checks="Declare the swap estimand on the scale used in the analysis (the old one was mis-scaled).",
                  runs="Every native next-item configuration, best state.", cost="Minutes per model on one GPU.",
                  builds=["dm-perturb"],
                  new="The shift without padding; a correctly scaled swap estimand.",
                  current="After a swap, 39–86 % of users whose top-1 copied the recent item keep its code; a 5-slot shift "
                          "changes the top-1 for 47–82 % of users.",
                  paper="Appendix, beside AR7."),
             dict(id="DG9", role="Supplementary", title="Next-two in one decode",
                  question="In a joint next-two decode, is item 2 copied from item 1, as in AR?",
                  measures="Change item 1's revealed digits and measure item 2's digits; knock out the self-attention edges "
                           "from item 2's positions to item 1's; replace the most recent history item against clamping "
                           "item 1.",
                  checks="No-op controls bit-exact; the v3 decoder reproduces the saved pair predictions.",
                  runs="Every next-two configuration, both catalogs, best state, user sample.",
                  cost="Minutes per model on one GPU.",
                  builds=["ar-nexttwo"],
                  new="New experiment and protocol. Revealed SID tokens carry no item-slot position embedding in this "
                      "architecture, so the model must tell the items apart by their mask slots.",
                  current="None: the old DiffGRM next-two models decode in two passes.",
                  paper="Beside AR8."),
         ]),
]

DYNAMICS = """
<p>The kept weight states let us ask when copying forms during training. AR has three states (epochs 0.5 and 1.5 and
the best model), which is enough to see whether copying is present early and how much it grows. DiffGRM has a state
every 20 epochs and the validation metrics of every epoch, so we can also ask whether copying strength rises with
validation accuracy.</p>
<ul>
<li><b>Every kept state:</b> AR1 (Industrial), DG1, and the blocks and restored shares of DG2.</li>
<li><b>Focal set only:</b> AR3, DG3 and AR8.</li>
<li><b>Best state only:</b> the decoding experiments (AR2, DG4, DG5), because intermediate states have no saved
predictions to check against. DiffGRM decoding is cheap enough to add the focal set's states later.</li>
</ul>
<p>This could become a short RQ3 subsection if the states differ. If copying is already at full strength by epoch 0.5,
it is one sentence.</p>
"""

# Claim cards of the storyline tab -> final experiments (main, supplementary), read from content_story so the two tabs
# cannot disagree; then the questions the final run adds.
NEW_CLAIMS = [
    ("New", "What removing collisions changes (native against ZCR).", ["M3", "AR4"], []),
    ("New", "When copying forms during training.", ["AR1", "DG1", "DG2"], ["AR3", "DG3"]),
]


def _claims() -> list[tuple]:
    import content_story
    return [(c["rq"], c["title"], c["final_main"], c["final_supp"]) for c in content_story.CLAIMS] + NEW_CLAIMS


# Every experiment on the other tabs -> its final experiments and what happens to it.
COVERAGE = {
    "maps-stats": (["M1", "M5"], "Kept; recomputed on the v3 tables."),
    "maps-collisions": (["M2", "M3"], "Kept; the ZCR twins add a causal test."),
    "ar-prefix-errors": (["AR9"], "Kept as supplementary, for both paradigms."),
    "ar-history": (["AR1", "AR5"], "Replacement is main (AR1); residual patching is supplementary (AR5)."),
    "ar-copy": (["AR1", "AR5"], "Copying and the edge knockout are main (AR1); the head screen is supplementary (AR5)."),
    "ar-lastdigit": (["AR4"], "Main, now with ZCR twins."),
    "ar-grid": (["AR1", "AR2"], "Its design becomes the default: every configuration instead of two models."),
    "ar-lens": (["AR3"], "Kept, across the grid."),
    "ar-sae": (["AR6"], "Supplementary, focal set."),
    "ar-infomap": (["AR3", "AR6"], "Lens and decoders join AR3; the SAE census joins AR6."),
    "ar-decoding": (["AR2"], "Kept, on the v3 decoder."),
    "ar-credit": (["AR2", "M4", "S2"], "The target classes move to S2 and are used by M4, AR2 and DG4."),
    "ar-collisions": (["M4"], "Kept for both paradigms."),
    "ar-time": (["S2", "AR7"], "The data part moves to S2; the model part joins AR7."),
    "ar-order": (["AR7"], "Supplementary."),
    "ar-nexttwo": (["AR8", "DG9"], "Supplementary, across the next-two grid; DG9 adds the DiffGRM side."),
    "ar-nexttwo-assoc": (["AR8"], "Folded into AR8: the v3 lists report exact ordered pairs directly."),
    "dm-retro-order": (["DG5"], "Not rerun. It compared archived runs with unequal beams; DG5 replaces it."),
    "dm-beam": (["DG5"], "Kept, with the v3 decoder and merging added."),
    "dm-orders": (["DG5"], "Kept inside DG5."),
    "dm-trace": (["DG3"], "Kept, under both decoders."),
    "dm-history": (["DG1"], "Main, across the grid and the weight states."),
    "dm-route": (["DG2"], "Main, across the grid."),
    "dm-encoder": (["DG2", "DG3", "DG7"], "Spread probes join DG2, own-code and target probes join DG3, SAEs and the "
                                          "description autoencoder join DG7."),
    "dm-patching": (["DG2", "DG6"], "The history contrast joins DG2 (main); the digit contrast is DG6 (supplementary)."),
    "dm-steps": (["DG3"], "Kept inside DG3."),
    "dm-perturb": (["DG4", "DG8"], "The copy block is DG4 (main); swap and shift are DG8 (supplementary)."),
}

DECISIONS = [
    "Confirm that the final run is the four-block grid, or tell us which blocks exist.",
    "AR1 scope: every kept state on Industrial and the best state on Office (proposed), or every state on both.",
    "User-sample size (2,000 proposed) and whether the declared estimates use the test cohort (proposed) or the "
    "validation cohort.",
    "Focal set: RQ-VAE 4×128 has no ZCR twin if its collision rate stays under 5 %. Add RQ-VAE 4×256, which gets a twin "
    "by name?",
    "Download the Office metadata, so that M1, M5 and S2 have categories and titles for Office.",
    "Main DiffGRM decoder: the v3 left-to-right decoder that the runs report (proposed), with confidence order as a "
    "condition in DG3 and DG5.",
    "Whether training dynamics becomes a sub-question of RQ3. Decide after AR1 and DG1 on the first block.",
    "If seed replicates are trained, run AR1, DG1, AR2 and DG4 on them to measure training noise.",
]


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def _all_final() -> list[dict]:
    return SHARED + [e for line in LINES for e in line["experiments"]]


def _anchor(fid: str) -> str:
    return f"final-{fid.lower()}"


TOC = ('<a href="#final-run">The final run</a><a href="#final-changes">What changes</a>'
       '<a href="#final-scopes">Terms</a><a href="#final-overview">Overview</a>'
       '<a href="#final-shared">Shared steps</a>'
       + "".join(f'<a href="#final-line-{line["key"]}">{esc(line["title"])}</a>'
                 + "".join(f'<a class="sub" href="#{_anchor(e["id"])}">{esc(e["id"])} {esc(e["title"])}</a>'
                           for e in line["experiments"]) for line in LINES)
       + '<a href="#final-dynamics">Training states</a><a href="#final-cost">Cost</a>'
       '<a href="#final-claims">Claims</a><a href="#final-coverage">Coverage</a><a href="#final-decisions">Decisions</a>')


def page(index: dict) -> str:
    """The tab's HTML. `index` maps review experiment ids to their entries (for links and checks)."""
    finals = {e["id"]: e for e in _all_final()}
    if len(finals) != len(_all_final()):
        raise ValueError("duplicate final experiment id")

    def review_pill(eid: str) -> str:
        if eid not in index:
            raise KeyError(f"final plan refers to unknown review experiment {eid!r}")
        return f'<a class="pill" href="#{eid}">{esc(index[eid]["code"])}: {esc(index[eid]["title"])}</a>'

    def final_pill(fid: str) -> str:
        if fid not in finals:
            raise KeyError(f"unknown final experiment {fid!r}")
        e = finals[fid]
        return f'<a class="pill" href="#{_anchor(fid)}">{esc(fid)} {esc(e["title"])}</a>'

    def role(r: str) -> str:
        return f'<span class="role role-{r.lower()}">{esc(r)}</span>'

    def card(e: dict) -> str:
        builds = "".join(review_pill(b) for b in e["builds"]) + (f" {e['builds_text']}" if e.get("builds_text") else "")
        rows = [("Question", esc(e["question"])), ("Measures", esc(e["measures"])), ("Checks", esc(e["checks"])),
                ("Runs on", esc(e["runs"])), ("Cost", esc(e["cost"])), ("Builds on", builds or "—"),
                ("What is new", esc(e["new"])), ("Current result", esc(e["current"])), ("In the paper", esc(e["paper"]))]
        dl = "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in rows)
        return (f'<section class="exp fx" id="{_anchor(e["id"])}"><header><span class="code">{esc(e["id"])}</span> '
                f'{role(e["role"])}<h2>{esc(e["title"])}</h2></header><dl class="fxd">{dl}</dl></section>')

    # Checks: every review experiment is covered, and coverage names real final experiments.
    if missing := sorted(set(index) - set(COVERAGE)):
        raise ValueError(f"review experiments missing from the final plan: {missing}")
    if extra := sorted(set(COVERAGE) - set(index)):
        raise ValueError(f"final plan covers unknown review experiments: {extra}")
    for eid, (fids, _) in COVERAGE.items():
        for fid in fids:
            final_pill(fid)
        if lost := [f for f in fids if eid not in finals[f]["builds"]]:
            raise ValueError(f"{eid} is covered by {lost}, which do not list it under 'Builds on'")

    def short(text: str) -> str:
        return text.split(";")[0].split(". ")[0]

    overview_rows = []
    for e in SHARED:
        overview_rows.append((e, "Shared"))
    for line in LINES:
        for e in line["experiments"]:
            overview_rows.append((e, line["short"]))
    overview = "".join(
        f'<tr><td><a href="#{_anchor(e["id"])}">{esc(e["id"])}</a></td><td>{role(e["role"])}</td><td>{esc(grp)}</td>'
        f'<td>{esc(e["title"])}</td><td>{esc(short(e["runs"]))}</td><td>{esc(short(e["cost"]))}</td></tr>'
        for e, grp in overview_rows)
    n_main = sum(e["role"] == "Main" for e in _all_final())
    n_supp = sum(e["role"] == "Supplementary" for e in _all_final())

    lines_html = ""
    for line in LINES:
        mains = [e for e in line["experiments"] if e["role"] == "Main"]
        supps = [e for e in line["experiments"] if e["role"] == "Supplementary"]
        lines_html += (f'<h2 id="final-line-{line["key"]}">{esc(line["title"])}</h2>'
                       f'<p class="meta">{esc(line["rq"])}</p><p>{esc(line["intro"])}</p>'
                       f'<h3>Main experiments</h3>{"".join(card(e) for e in mains)}'
                       f'<h3>Supplementary experiments</h3>{"".join(card(e) for e in supps)}')

    claims = "".join(
        f'<tr><td>{esc(rq)}</td><td>{esc(text)}</td><td>{"".join(final_pill(f) for f in main) or "—"}</td>'
        f'<td>{"".join(final_pill(f) for f in supp) or "—"}</td></tr>' for rq, text, main, supp in _claims())
    coverage = "".join(
        f'<tr><td>{review_pill(eid)}</td><td>{"".join(final_pill(f) for f in fids)}</td><td>{esc(note)}</td></tr>'
        for eid, (fids, note) in sorted(COVERAGE.items(), key=lambda kv: list(index).index(kv[0])))
    changes = "".join(f"<li><b>{esc(a)}</b> {esc(b)}</li>" for a, b in CHANGES)
    scopes = "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in SCOPES)
    decisions = "".join(f"<li>{esc(d)}</li>" for d in DECISIONS)

    return (
        '<h1 class="pagetitle">Final important experiments</h1>'
        f"<p>This tab proposes the experiments to run on the checkpoints of the final training run, for review before we "
        f"build the pipeline. They fall into three lines: codebooks and collisions (M), Qwen-AR (AR) and DiffGRM (DG). "
        f"Each line lists its main experiments first and its supplementary experiments after: {n_main} main and {n_supp} "
        f"supplementary, plus three shared steps that come first. Every experiment on the other tabs maps to one of them "
        f"(<a href=\"#final-coverage\">Coverage</a>). Nothing here has run on the final checkpoints. The \"Current "
        f"result\" lines quote the provisional numbers that the final run should confirm or replace.</p>"
        f'<h2 id="final-run">What the final run contains</h2>{RUN}'
        f'<h2 id="final-changes">What this changes for the analysis</h2><ol>{changes}</ol>'
        f'<h2 id="final-scopes">Terms used below</h2><dl class="fxd terms">{scopes}</dl>'
        f'<h2 id="final-overview">Overview</h2><div class="table-wrap"><table class="ov"><thead><tr><th>ID</th><th>Role</th>'
        f'<th>Line</th><th>Experiment</th><th>Runs on</th><th>Cost</th></tr></thead><tbody>{overview}</tbody></table></div>'
        f'<h2 id="final-shared">Shared steps</h2><p>These come before every line: they load and check the models and '
        f'build the target classes that several experiments split by.</p>{"".join(card(e) for e in SHARED)}'
        f'{lines_html}'
        f'<h2 id="final-dynamics">Experiments along training</h2>{DYNAMICS}'
        f'<h2 id="final-cost">Cost</h2>{COSTS}'
        f'<h2 id="final-claims">Claims and the experiments behind them</h2><p>The first nine rows are the claim cards of '
        f'the storyline tab; the last two are questions the final run adds.</p><div class="table-wrap"><table class="ov">'
        f'<thead><tr><th>RQ</th><th>Claim</th><th>Main</th><th>Supplementary</th></tr></thead><tbody>{claims}</tbody>'
        f'</table></div>'
        f'<h2 id="final-coverage">Coverage of the earlier experiments</h2><div class="table-wrap"><table class="ov cov">'
        f'<thead><tr><th>Earlier experiment</th><th>Final experiment</th><th>What happens to it</th></tr></thead>'
        f'<tbody>{coverage}</tbody></table></div>'
        f'<h2 id="final-decisions">Decisions for your review</h2><ol class="order">{decisions}</ol>')
