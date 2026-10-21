"""AR experiments for the mechanisms review page. Wording follows docs/paper/WRITING.md."""

from __future__ import annotations

from build_report import DERIVED, figs, numbers, png
from sidlens.viz import svgcharts as SC

KM, VAE, MQ = "--q-rqkmeans", "--q-rqvae", "--q-MQ"

INTRO = """
<h1 class="pagetitle">AR experiments</h1>
<p>The AR recommender is Qwen2.5-1.5B fine-tuned to write a SID digit by digit after a prompt that lists the user's
history as SIDs. Two next-item models have weights: <b>RQ-KMeans 3×128</b> (<code>next-item_best</code>) and
<b>RQ-VAE 4×128</b> (<code>oneoff_rqvae4cb128</code>, a separate run, not the sweep model). One next-two model
(MQ 4×256) predicts the next two items, and 12 retrained next-item models cover more maps (exp11). Unless stated, the
data are the next-item test split (3,681 events, 1,606 users) under teacher forcing. Effects are given as Δ: how much
the target code's log-probability falls (nats). r1 is the most recent history item. Intervals are 95 % user-bootstrap
intervals; they do not include training-seed variation.</p>
"""


def _prefix_errors():
    first = [("digit 1", 82.16), ("digit 2", 6.40), ("digit 3", 2.06), ("digit 4", 0.67), ("digit 5", 0.55)]
    cond = [("digit 2", 64.11), ("digit 3", 82.02), ("digit 4", 93.17), ("digit 5", 93.83)]
    return figs(
        SC.hbars([{"label": a, "est": v / 100, "var": KM} for a, v in first],
                 title="Share of rank-1 predictions whose first wrong digit is…", x_label="share of rows", pct=True, label_w=90),
        SC.hbars([{"label": a, "est": v / 100, "var": KM} for a, v in cond],
                 title="Accuracy of a digit when every earlier digit is right", x_label="accuracy", pct=True, label_w=90),
    ) + numbers([("Rank-1 exact SID accuracy", "8.75 % [7.64, 9.85]"),
                 ("Rank-1 digits matched before the first error", "0.48 of a digit on average"),
                 ("First digit right at rank 1", "17.8 %"),
                 ("SID HR@10 (exact)", "16.6 %")])


def _sibling():
    items = [{"label": "RQ-VAE: top-1 = h's code (test)", "est": 0.118, "lo": 0.053, "hi": 0.191, "var": VAE},
             {"label": "RQ-VAE: top-1 = h's code (valid)", "est": 0.116, "lo": 0.037, "hi": 0.203, "var": VAE},
             {"label": "RQ-VAE: top-1 = sibling target (test)", "est": -0.107, "lo": -0.182, "hi": -0.041, "var": VAE},
             {"label": "RQ-VAE: top-1 = sibling target (valid)", "est": -0.093, "lo": -0.186, "hi": -0.006, "var": VAE}]
    nats = [{"label": "RQ-KMeans: h's code (test)", "est": -0.17, "lo": -0.22, "hi": -0.13, "var": KM},
            {"label": "RQ-KMeans: h's code (valid)", "est": -0.34, "lo": -0.42, "hi": -0.26, "var": KM},
            {"label": "RQ-KMeans: sibling target (valid)", "est": 0.13, "var": KM},
            {"label": "RQ-KMeans: sibling target (test)", "est": 0.15, "lo": 0.06, "hi": 0.26, "var": KM}]
    return figs(
        SC.hbars(items, title="RQ-VAE, sibling rows: change in top-1 when the last-digit edge to h is blocked", x_label="percentage points",
                 pct=True, signed=True, label_w=250, width=620),
        SC.hbars(nats, title="RQ-KMeans: change in log-prob when the edge is blocked, beyond a control edge (nats)", x_label="nats",
                 signed=True, label_w=250, width=620),
    ) + numbers([("Clean: h's exact code ranked first in sibling rows", "RQ-VAE 18 %, RQ-KMeans 53 %"),
                 ("… if h was re-bought by some user in training", "RQ-VAE 92 %, RQ-KMeans 96 %"),
                 ("RQ-VAE declared pooled estimate (P1)", "+0.09 nats [−0.02, +0.25] (test): includes 0")])


def _collision_credit():
    rows = [{"label": "RQ-KMeans: share of rows (collision partner)", "est": 0.113, "var": KM},
            {"label": "RQ-KMeans: share of C_all HR@10 loss", "est": 0.56, "lo": 0.46, "hi": 0.68, "var": KM},
            {"label": "RQ-KMeans: share of baseline SID hits", "est": 0.42, "var": KM},
            {"label": "RQ-VAE: share of rows (near-duplicate variant)", "est": 0.099, "var": VAE},
            {"label": "RQ-VAE: share of C_all HR@10 loss", "est": 0.80, "lo": 0.68, "hi": 0.95, "var": VAE}]
    new = [{"label": "RQ-KMeans, SID level", "est": -0.010, "var": KM},
           {"label": "RQ-KMeans, item level", "est": -0.002, "lo": -0.008, "hi": 0.003, "var": KM},
           {"label": "RQ-VAE, SID level", "est": -0.002, "var": VAE},
           {"label": "RQ-VAE, item level", "est": -0.001, "var": VAE}]
    return figs(SC.hbars(rows, title="Where the copy benefit comes from", x_label="share", pct=True, label_w=280, width=640),
                SC.hbars(new, title="Truly new targets (83–85 % of rows): HR@10 change when copying is blocked", x_label="percentage points",
                         pct=True, signed=True, label_w=170)) + numbers(
        [("RQ-KMeans: share of the SID-level HR@1 copy benefit that remains at item level", "28 % [22, 36] (HR@10: 83 %)"),
         ("RQ-VAE: share of exp6's 'copying helps new targets' carried by variants", "94 %")])


def _collisions():
    items = [{"label": "HR@10, collided − singleton targets", "est": 0.425, "lo": 0.400, "hi": 0.449, "var": KM},
             {"label": "SID HR@10 − item HR@10 (all targets)", "est": 0.045, "lo": 0.041, "hi": 0.050, "var": KM},
             {"label": "… collided targets", "est": 0.141, "var": KM},
             {"label": "… singleton targets", "est": 0.012, "var": KM}]
    return SC.hbars(items, title="SID credit and collisions, mean over 18 AR configurations", x_label="percentage points", pct=True,
                    label_w=260, width=620) + numbers([("Configurations where collided targets are easier", "17 of 18"),
                                                        ("Pooled (not cell-averaged) collided − singleton", "+33.0 pp")])


def _next_two_assoc():
    items = [{"label": "P(item 2 correct | item 1 correct)", "est": 0.0403, "var": MQ},
             {"label": "P(item 2 correct | item 1 wrong)", "est": 0.0034, "var": MQ},
             {"label": "Difference", "est": 0.0369, "lo": 0.0114, "hi": 0.0623, "var": MQ}]
    return SC.hbars(items, title="Top-ranked pair, mean over 10 next-two configurations", x_label="percentage points", pct=True,
                    label_w=230) + numbers([("Configurations with a positive difference", "5 of 10 (−0.6 to +18.5 pp)"),
                                            ("By quantizer (mean)", "MQ +0.7 pp, RQ-KMeans +0.1 pp, RQ-VAE +11.3 pp")])


def experiments() -> list[dict]:
    C = DERIVED / "controlled"
    R = DERIVED / "representation"
    T = DERIVED / "retrospective"
    return [
        dict(id="ar-prefix-errors", code="Retrospective exp2", title="Where AR predictions first go wrong",
             models="18 archived AR next-item configurations (saved lists only)", data="test split", status="descriptive, no model run",
             source="derived/retrospective/exp2_prefix/",
             did="<p>For each of the 18 archived AR configurations (widths 128 and 512), we compared every rank-1 prediction with the "
                 "target digit by digit and recorded the first digit that differs. We also measured how often a digit is right when all "
                 "earlier digits are right.</p>",
             results=_prefix_errors,
             takeaways=["Most rank-1 errors start at the first digit (82 % of rows).",
                        "When the first digit is right, the later digits are usually right too (64 % at digit 2, 82 % at digit 3).",
                        "So the first digit is where history matters most, which exp3 tests."]),

        dict(id="ar-history", code="Controlled exp3", title="Replacing one history item, and where its information goes",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="test split, teacher forcing", status="declared protocol",
             source="derived/controlled/exp3_ar_history_patching/summary-280961/",
             did="<p>We replaced one history item with a catalogue item that shares its first m digits and differs at digit m+1 "
                 "(m = 0: a different first digit), re-encoded the prompt, and measured the fall in the target code's log-probability "
                 "at each digit. We did this for each of the 10 most recent items.</p>"
                 "<p>To find where the replaced item's information goes, we ran the corrupted prompt and copied the clean run's "
                 "activations back in at one layer and one group of positions: the item's own tokens, the tokens between it and the "
                 "answer, the response header, or the target tokens. Recovery is the share of the effect that comes back.</p>",
             results=lambda: figs(
                 png(C / "exp3_ar_history_patching/summary-280961/figures-r2/fig1_recency.png",
                     "Δ at the first digit when the item r positions back is replaced (m = 0)"),
                 png(C / "exp3_ar_history_patching/summary-280961/figures-r2/fig3_patching_recovery.png",
                     "Share of the effect recovered by patching one layer at one position group")) + numbers(
                 [("Replace r1, first digit (m = 0)", "RQ-KMeans 0.57 [0.53, 0.62] · RQ-VAE 0.51 [0.47, 0.55] nats"),
                  ("Top-1 first digit changes in", "75 % · 50 % of rows"),
                  ("Second most recent item", "0.16 · 0.22 nats"),
                  ("Effect left when the control keeps r1's first digit", "18 % · 36 %"),
                  ("Item tokens still hold the effect (recovery ≥ 0.95) through", "layer 20 · layer 16 (of 28)"),
                  ("Response header recovers ≥ 0.5 from", "layer 21 · layer 20")]),
             takeaways=["Both models rely mostly on the most recent item, and mostly on its first digit.",
                        "The item's information stays at its own tokens until about layer 20 and then moves to the answer positions in "
                        "layers 19–24. The tokens in between never carry it.",
                        "A post-hoc split found that changes to later digits matter almost only when the history item already matches "
                        "the target's earlier digits. This led to the prefix-matched copying hypothesis, tested in exp4."]),

        dict(id="ar-copy", code="Controlled exp4", title="Prefix-matched copying and the attention edge that carries it",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="validation split (3,680 rows); test split as secondary",
             status="declared protocol, tested on data not used to form the hypothesis",
             source="derived/controlled/exp4_ar_copy_circuit/summary-281261/",
             did="<p>We tested the copy hypothesis from exp3 on the validation split. First, we changed digit d onward of the most "
                 "recent item and compared rows where that item already matches the target's first d digits with rows where it does "
                 "not. Second, we blocked a single attention edge: from the position that predicts digit d to the matching item's "
                 "digit-d token, in all layers, in layers 0–13 only, or in layers 14–27 only. Controls block the same edge to a "
                 "non-matching item. Third, we picked the 5 heads with the largest edge effect on half of the users and tested "
                 "them on the other half against random heads from the same layers.</p>",
             results=lambda: figs(
                 png(C / "exp4_ar_copy_circuit/summary-281261/figures/fig1_match_vs_nonmatch.png",
                     "Δ at digit d when r1 matches the target's first d digits, and when it does not"),
                 png(C / "exp4_ar_copy_circuit/summary-281261/figures/fig2_heads.png", "Head screen: effect of the top-5 heads vs random heads")) + numbers(
                 [("Change r1's digits ≥ d, matching rows", "RQ-KMeans 1.35 · RQ-VAE 1.06 nats"),
                  ("Same, non-matching rows", "0.01 · 0.03 nats"),
                  ("Block the one edge to the matching digit-d token", "0.40 [0.32, 0.48] · 0.36 [0.31, 0.42] nats"),
                  ("Same block in layers 0–13 / layers 14–27", "0.000 / 0.395 · 0.001 / 0.355 nats"),
                  ("Share of the edge effect carried by the top-5 heads", "9 % · 8 %")]),
             takeaways=["To choose digit d, both models copy digit d of a history item that matches the prefix they have already written.",
                        "The copy is a direct attention read of that item's digit-d token, entirely in layers 14–27.",
                        "Many heads share the read. Five heads carry less than a tenth of it, so there is no small circuit to point to."]),

        dict(id="ar-lastdigit", code="Controlled exp10", title="The last digit: copy it or avoid it",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="test and validation splits", status="declared protocol",
             source="derived/controlled/exp10_ar_last_digit_sibling/summary-293619-293620/",
             did="<p>We took rows where a history item h matches the target on every digit before the last. We blocked the attention "
                 "edge from the last digit's prediction position to h's last-digit token (layers 14–27), against a matched control "
                 "edge. We split rows by whether the target is h's own SID or a sibling: same prefix, different last digit.</p>",
             results=_sibling,
             takeaways=["RQ-KMeans copies h's last digit. When the target is a sibling, this copy competes with it.",
                        "RQ-VAE does the opposite in sibling rows: blocking the edge makes it pick h's code more often and the sibling "
                        "less often, so the read pushes it away from h. The declared pooled estimate for RQ-VAE includes 0; this rests "
                        "on the top-1 changes.",
                        "The difference matches the maps: RQ-VAE's last digit mostly names one item, RQ-KMeans' last digit is shared by "
                        "near-duplicates (structure exp3)."]),

        dict(id="ar-grid", code="Controlled exp11", title="Copying across 12 retrained AR models",
             models="12 retrained Qwen-AR next-item models (MQ, RQ-KMeans, RQ-VAE; depth 3–5; width 128 and 512)",
             data="test and validation splits", status="declared protocol; run 2026-10-07, summarized 2026-10-15",
             source="derived/controlled/exp11_ar_copy_grid/summary-301495/",
             did="<p>We retrained the 18 AR next-item models of the width-128/512 grid with the recorded recipe (substrate exp1). "
                 "Twelve passed the acceptance check against their archived lists; MQ 3×512, MQ 4×128, MQ 5×128, MQ 5×512, "
                 "RQ-KMeans 3×512 and RQ-KMeans 4×512 did not. On each of the 12, we repeated four estimates from the two original "
                 "models: replacing r1 (exp3), prefix-matched copying and the single-edge knockout (exp4), and blocking copying "
                 "inside beam search on repeat targets (exp6). We also compared the retrained RQ-KMeans 3×128 with the original.</p>",
             results=lambda: figs(png(C / "exp11_ar_copy_grid/summary-301495/figures/fig1_core_estimates.png",
                                      "Four core estimates on the 12 retrained models")) + numbers(
                 [("Models whose interval excludes 0, all four estimates", "12 of 12"),
                  ("Replace r1, first digit", "0.37–0.78 nats"),
                  ("Prefix-matched copying (match − non-match)", "0.89–2.18 nats"),
                  ("Single-edge knockout (minus control edge)", "0.20–0.60 nats"),
                  ("Copy block in beam search, repeat targets", "−7.5 to −37.0 pp of HR@10"),
                  ("Retrained − original RQ-KMeans 3×128", "replace r1 −0.03 nats; copying −0.04 [−0.13, 0.05]; decoding −9.8 pp")]),
             takeaways=["Every retrained model copies, on every quantizer, depth and width tested. The copy findings are not specific to the "
                        "two original checkpoints.",
                        "A second training run of the same map gives the same history-use numbers within 0.04 nats. The effect on repeat "
                        "targets in decoding differs more (−37 vs −27 pp).",
                        "The six models that failed acceptance are not included."]),

        dict(id="ar-lens", code="Representation exp1", title="Layer-by-layer readout of the first digit",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="test split, activations at all 28 layers", status="declared protocol; observational",
             source="derived/representation/exp1_ar_digit_decoding/summary-281264-281411/",
             did="<p>At the position that predicts the first digit, we applied the model's output head to every layer (logit lens). "
                 "We also trained linear probes, with users split between training and test, for r1's first digit and the target's "
                 "first digit.</p>",
             results=lambda: figs(
                 png(R / "exp1_ar_digit_decoding/summary-281264-281411/figures/fig1_logit_lens.png", "Logit lens by layer"),
                 png(R / "exp1_ar_digit_decoding/summary-281264-281411/figures/fig2_probes.png", "Linear probes by layer")) + numbers(
                 [("Final top-1 first digit = r1's first digit", "RQ-KMeans 64 % · RQ-VAE 53 %"),
                  ("Final top-1 first digit = the target's", "18 % · 34 %"),
                  ("Layer where the copy appears in the logit lens", "20–23 · 11 and 20"),
                  ("Probe for r1's first digit after block 0", "84 % · 93 %"),
                  ("Probe for the target's first digit, any layer", "never above the rule 'predict r1's first digit'")]),
             takeaways=["The model's first digit is mostly a copy of the most recent item's first digit.",
                        "That digit is readable from the first layer, but the output starts to follow it only in late layers.",
                        "No layer holds information about the target beyond the copy rule."]),

        dict(id="ar-sae", code="Representation exp2", title="Sparse autoencoders on the AR residual stream",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="SAEs trained on train-split activations, analysed on the test split",
             status="declared protocol; amendment A1 post hoc",
             source="derived/representation/exp2_ar_sae/summary-287222-287231/",
             did="<p>We trained TopK sparse autoencoders (12,288 latents, k = 32) at layers 12, 16, 20 and 24. At the first-digit "
                 "prediction position we looked for <i>copy latents</i>, which fire for one code of r1's first digit, and at later "
                 "digits for <i>match latents</i>, which fire when r1 matches the prefix so far. We removed each set from the "
                 "activation (keeping the SAE's error) and compared with matched control latents. Ridge and logistic probes tested "
                 "for target content beyond the history.</p>",
             results=lambda: figs(png(R / "exp2_ar_sae/summary-287222-287231/figures/fig1_causal.png",
                                      "Effect of removing copy latents and match latents, minus controls")) + numbers(
                 [("Removing copy latents: drop in the copy score", "0.10 nats at layer 20; 0.21 · 0.36 at layer 24"),
                  ("Share of the readout's latent activation they carry", "≤ 3 % · ≤ 12 %"),
                  ("Match latents: held-out AUC for 'r1 matches the prefix'", "up to 0.97 (RQ-KMeans), about 0.66 (RQ-VAE)"),
                  ("Removing match latents", "0.00 ± 0.04 nats (RQ-KMeans)"),
                  ("Readout predicts a same-day burst beyond SID overlap", "AUC +0.08–0.09")]),
             takeaways=["SAEs find copy features that change the output from layer 20 on.",
                        "Features that detect a prefix match exist at the prediction position but have no effect there. With exp4, "
                        "this suggests the match is computed inside attention (query against key).",
                        "No target content beyond the history is readable."]),

        dict(id="ar-infomap", code="Representation exp3", title="Information map across all layers",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="test split, the model's own greedy answers", status="declared protocol; amendment A1",
             source="derived/representation/exp3_ar_information_map/summary-294711-294712/",
             did="<p>We let each model write its own greedy answer and captured every second layer. We trained SAEs at each of these "
                 "layers and labelled every latent by what it tracks (a census). We also fitted a tuned lens and decoders that read "
                 "the whole SID of the model's answer, the true target and r1 from each layer.</p>",
             results=lambda: figs(
                 png(R / "exp3_ar_information_map/summary-294711-294712/figures/fig2_sid_lens.png", "Tuned lens and logit lens: agreement with the final answer"),
                 png(R / "exp3_ar_information_map/summary-294711-294712/figures/fig3_sid_decoders.png", "Whole-SID decoders by layer")) + numbers(
                 [("Median first layer from which the answer stays readable: copy / other answers", "17 / 23 · 19 / 23"),
                  ("An item's last token holds its full SID from", "layer 2"),
                  ("True target decoded above 'output r1'", "at most +2.1 · +3.3 pp"),
                  ("Greedy answer = target", "10.9 % · 6.3 % of rows")]),
             takeaways=["Answers that copy the most recent item form earlier than other answers.",
                        "Each history item's identity is assembled in the first two layers.",
                        "As in representation exp1, no layer holds the true target beyond simple history rules."]),

        dict(id="ar-decoding", code="Controlled exp6", title="Blocking copying inside beam search",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="test split, archived decoder and plain beam search", status="declared protocol, two amendments",
             source="derived/controlled/exp6_ar_copy_in_decoding/summary-283114/",
             did="<p>We re-ran the archived decoder and blocked the copy reads (exp4's edges, layers 14–27) at every digit, at later "
                 "digits only, or, as a control, as many reads of non-matching items. Each beam is blocked from its own prefix. We "
                 "measured HR@10 on <i>repeat</i> targets (SID already in the history) and <i>new</i> targets. The archived "
                 "decoder samples (temperature 0.7, top-k 20, top-p 0.8, repetition penalty 1.1), because the checkpoint's generation "
                 "settings were merged in; we reproduced it exactly and also ran plain beam search.</p>",
             results=lambda: figs(png(C / "exp6_ar_copy_in_decoding/summary-283114/figures/fig1_delta_hr10.png",
                                      "Change in HR@10 by condition and target type")) + numbers(
                 [("Block at every digit: all targets", "RQ-KMeans −5.7 · RQ-VAE −3.3 pp"),
                  ("… repeat targets (HR@10 91 % with copying)", "−27.3 · −10.1 pp"),
                  ("… new targets", "−2.1 · −3.0 pp"),
                  ("Control block", "−0.03 · −0.11 pp"),
                  ("Archived lists reproduced", "3,680 of 3,681")]),
             takeaways=["Copying helps the AR lists and never hurts them.",
                        "Most of the gain is on items the user already had. Retrospective exp7 shows that most of the gain on 'new' targets "
                        "comes from items that share a SID with a history item or are near-duplicate variants."]),

        dict(id="ar-credit", code="Retrospective exp7", title="Which targets the copy benefit comes from",
             models="exp6 predictions of RQ-KMeans 3×128 and RQ-VAE 4×128", data="test split", status="declared protocol; amendment A1 post hoc",
             source="derived/retrospective/exp7_collision_credit/293597/result/",
             did="<p>We re-scored exp6's lists at item level: each SID was expanded to every item it names. We split targets by how "
                 "they relate to the history: the same item, a different item with the same SID (a collision partner), a near-duplicate "
                 "variant with a different SID, or a truly new item.</p>",
             results=_collision_credit,
             takeaways=["Copying recovers repeat purchases and product variants.",
                        "For RQ-KMeans, much of the benefit at rank 1 is credit for a SID that several items share; only 28 % of it "
                        "remains at item level.",
                        "For truly new items, copying changes item HR@10 by −0.2 / −0.1 pp."]),

        dict(id="ar-collisions", code="Retrospective exp3", title="Collisions and SID-level accuracy",
             models="18 archived AR next-item configurations", data="test split, saved lists", status="descriptive",
             source="derived/retrospective/exp3_collisions/",
             did="<p>We compared HR@10 for targets whose SID is shared by several items with targets whose SID is unique, within each "
                 "configuration. We also expanded each predicted SID into its items and computed item-level HR@10 bounds.</p>",
             results=_collisions,
             takeaways=["Targets that share a SID are much easier to hit (+42.5 pp).",
                        "SID-level HR@10 overstates item-level HR@10 by 4.5 pp on average, and by 14.1 pp for targets that share a SID.",
                        "Results should report item-level bounds beside SID-level numbers."]),

        dict(id="ar-time", code="Retrospective exp6", title="Time in the data and in the models' copying",
             models="RQ-KMeans 3×128, RQ-VAE 4×128 (and both DiffGRM matched models)", data="review days for every event",
             status="declared protocol", source="derived/retrospective/exp6_time_structure/287189/result/",
             did="<p>We recovered each event's review day and split the data, and the outputs of exp3, exp4, exp6 and exp7, by the gap "
                 "between r1 and the target.</p>",
             results=lambda: figs(
                 png(T / "exp6_time_structure/287189/result/figures/fig2_copy_calibration.png", "How often the model copies r1's first digit, and how often the data warrant it, by gap"),
                 png(T / "exp6_time_structure/287189/result/figures/fig3_exp6_by_target_type.png", "exp6 copy effect by target type and day")) + numbers(
                 [("Targets on the same day as r1", "41–50 %"),
                  ("RQ-KMeans copies r1's first digit: same day / more than a year", "67 % / 67 % (data: 33 % / 3 %)"),
                  ("Share of exp6's copy benefit from same-day rows", "87 % · 88 %"),
                  ("Duplicate review records: share of test rows / of HR@10 hits", "2.7 % / 11 % · 17 %")]),
             takeaways=["In the data, r1 predicts the target mainly when it is from the same day. Once an item's age is known, its list position "
                        "adds almost nothing.",
                        "The models cannot see time and copy r1 whatever its age.",
                        "Duplicated review records supply a sixth of RQ-VAE's hits; results should flag them."]),

        dict(id="ar-order", code="Controlled exp9", title="Order versus time: swapping the two most recent items",
             models="RQ-KMeans 3×128, RQ-VAE 4×128", data="test split (validation as replication)", status="declared protocol",
             source="derived/controlled/exp9_ar_order_vs_time/summary-287182/",
             did="<p>We swapped r1 and r2 and measured how much the model prefers whichever item is last (position effect A) and how "
                 "much it prefers the item that is truly later (item effect B). We split rows by whether the two items share a review "
                 "day; when they do, only ASIN order separates them.</p>",
             results=lambda: figs(png(C / "exp9_ar_order_vs_time/summary-287182/figures/fig1_position_item_mix.png",
                                      "Position effect, item effect and the order-averaged model")) + numbers(
                 [("Position effect A, first digit", "RQ-KMeans 0.36–0.39 · RQ-VAE 0.07–0.10"),
                  ("Swap of same-day items flips the first-digit decision", "50 % · 18 % of rows"),
                  ("… and the top-1 SID", "42 % (RQ-KMeans)"),
                  ("Averaging over the two tied orders, where time cannot order them", "+0.027 nats (RQ-KMeans)")]),
             takeaways=["RQ-KMeans AR copies whichever item is last in the list, regardless of time.",
                        "RQ-VAE AR follows position only weakly.",
                        "Where the order of two items is arbitrary, it still decides half of RQ-KMeans' first digits."]),

        dict(id="ar-nexttwo", code="Controlled exp5", title="Next-two prediction: item 2 is copied from item 1",
             models="next-two AR model, MQ 4×256", data="two-item test split (3,452 rows), teacher forcing", status="declared protocol",
             source="derived/controlled/exp5_ar_next_two_conditioning/summary-281263/",
             did="<p>Under teacher forcing on 'item 1 ||| item 2', we replaced item 1 with a catalogue item, or replaced r1, and measured "
                 "item 2's digits. We also tested prefix-matched copying from item 1 to item 2 and patched activations to find where "
                 "item 1's information moves.</p>",
             results=lambda: figs(png(C / "exp5_ar_next_two_conditioning/summary-281263/figures/fig1_item1_patching.png",
                                      "Where item 1's information sits, by layer")) + numbers(
                 [("Replace item 1: item 2's first digit", "0.60 nats [0.54, 0.66]"),
                  ("Replace r1: item 2's first digit", "0.20 nats"),
                  ("Copying from item 1 to item 2 (matching rows)", "0.43–0.73 nats; about 0 otherwise"),
                  ("Item 1's information moves to item 2's positions", "layers 20–26")]),
             takeaways=["The second item depends on the first item more than on the history.",
                        "It is computed by the same prefix-matched copying, now between the two predicted items, in the same late layers."]),

        dict(id="ar-nexttwo-assoc", code="Retrospective exp4", title="Next-two association in archived lists",
             models="10 archived next-two AR configurations", data="two-item test split", status="descriptive",
             source="derived/retrospective/exp4_next_two/",
             did="<p>In the saved next-two lists, we measured how often the second predicted item is correct, given whether the first is.</p>",
             results=_next_two_assoc,
             takeaways=["When item 1 is right, item 2 is right more often (4.0 % vs 0.3 %).",
                        "This is an association, and it varies across maps; exp5 tests the dependence causally."]),
    ]
