"""Diffusion (DiffGRM) experiments for the mechanisms review page. Wording follows docs/paper/WRITING.md."""

from __future__ import annotations

import pandas as pd

from build_report import DERIVED, figs, numbers, png
from sidlens.viz import diffusion_report as DR
from sidlens.viz import svgcharts as SC

KM, VAE = "--q-rqkmeans", "--q-rqvae"
QLEG = DR.Q_LEGEND
LLEG = SC.legend([("--q-rqkmeans", "RQ-KMeans"), ("--q-rqvae", "RQ-VAE"), ("--q-MQ", "MQ")]) + DR.LINE_LEGEND

S_EXP5 = DERIVED / "representation/exp5_diffusion_encoder_features/summary-312763"
S_EXP12 = DERIVED / "controlled/exp12_diffusion_activation_patching/summary-312754-312806"
S_EXP13 = DERIVED / "controlled/exp13_diffusion_step_switch/summary-312774"
S_EXP14 = DERIVED / "controlled/exp14_diffusion_decode_perturbations/summary-312775-313104"
S_EXP4 = DERIVED / "representation/exp4_diffusion_reveal_trace/summary-310761-16cells"

INTRO = """
<h1 class="pagetitle">Diffusion experiments</h1>
<p>DiffGRM is a masked-diffusion recommender. A one-layer encoder turns each history item (its SID digits, plus its slot
position) into one state. A four-layer decoder has one position per SID digit; each position holds either a revealed code
or a mask. Decoding starts with every digit masked, and at each step a beam search reveals one (position, code) pair; only
the partial SID passes from one step to the next. We have 20 trained next-item models: MQ, RQ-KMeans and RQ-VAE at depths
3, 4 and 5 and widths 128 and 512, plus MQ 5×256 and RQ-VAE 3×256. Two of them, RQ-KMeans 3×128 and RQ-VAE 4×128, use the
same SID maps as the two AR models. The test cohort is 6,297 users with their last item held out; it is a different cohort
from the AR test split, so AR and DiffGRM numbers are compared side by side, never joined. In the width charts an open dot
is width 128 and a filled dot width 512.</p>
"""


def _cells(df: pd.DataFrame, col: str = "checkpoint") -> pd.DataFrame:
    info = pd.DataFrame([DR.cell_info(c) for c in df[col]])
    return pd.concat([df.reset_index(drop=True).drop(columns=[c for c in ("quantizer", "depth", "width") if c in df]), info], axis=1)


def _retro_order():
    pc = pd.read_csv(DERIVED / "retrospective/exp1_reveal_order/per_cell.csv")
    pc = _cells(pc, "ckpt_id")
    rows = DR.grid_rows(pc, lambda s: (float(s.iloc[0].ndcg_at_10_delta_confidence_minus_fixed), float("nan"), float("nan")))
    return SC.dumbbell(rows, title="Archived runs: NDCG@10 of confidence order (beam 256) − fixed order (beam 64)",
                       x_label="percentage points", pct=True, signed=True) + QLEG + numbers(
        [("Mean difference, 18 balanced configurations", "+0.35 pp NDCG@10 (14 of 18 positive)"),
         ("Why this is not a test of reveal order", "the two runs used beams of 256 and 64")])


def _matched_beam():
    e1 = pd.read_csv(DERIVED / "controlled/exp1_matched_beam/summary-181944-181946/configuration.csv")
    e1 = _cells(e1[e1.metric == "sid_hit@10"])
    charts = []
    for beam in (64, 256):
        sub = e1[e1.beam_width == beam]
        rows = DR.grid_rows(sub, lambda s: (float(s.iloc[0].difference), float(s.iloc[0].ci_low), float(s.iloc[0].ci_high)))
        charts.append(SC.dumbbell(rows, title=f"Confidence − fixed order, SID HR@10, both at beam {beam}", x_label="percentage points",
                                  pct=True, signed=True))
    return figs(*charts) + QLEG + numbers([("Mean over 18 models, beam 64", "−0.95 pp [−1.07, −0.83]"),
                                            ("Mean over 18 models, beam 256", "+0.22 pp [+0.12, +0.34]"),
                                            ("Beam 64, by depth 3 / 4 / 5", "+0.4 / −0.7 / −2.6 pp"),
                                            ("Duplicate paths in the final confidence beam, depth 3 / 4 / 5", "57 / 79 / 91 %")])


def _fixed_orders():
    pc = pd.read_csv(DERIVED / "controlled/exp2_fixed_orders/summary-194295-194297/per_cell.csv")
    pc = pc[(pc.outcome == "sid_hit@10") & (pc.beam == 64)]
    items = [{"label": DR.cell_info(r.checkpoint)["quantizer"].replace("rqkmeans", "RQ-KMeans").replace("rqvae", "RQ-VAE")
              + f" 3×{DR.cell_info(r.checkpoint)['width']}", "est": r.difference_pp / 100, "lo": r.ci95_low_pp / 100,
              "hi": r.ci95_high_pp / 100, "var": f"--q-{DR.cell_info(r.checkpoint)['quantizer']}"} for r in pc.itertuples()]
    return figs(SC.hbars(items, title="Confidence − mean of the six fixed orders, SID HR@10, beam 64", x_label="percentage points",
                         pct=True, signed=True, label_w=130),
                png(DERIVED / "controlled/exp2_fixed_orders/summary-194295-194297/figures/order_effect.png", "HR@10 of every fixed order and of confidence")) + numbers(
        [("Confidence − mean fixed order (6 models)", "+0.40 pp [+0.29, +0.52]"),
         ("Best fixed order overall", "digit 2 → 3 → 1 (16.44 % HR@10), but it differs by quantizer")])


def _reveal_trace():
    pc = pd.read_csv(S_EXP4 / "per_cell.csv")
    pc = _cells(pc, "cell")
    charts = [SC.dumbbell(DR.grid_rows(pc, lambda s: (float(s.iloc[0].state0_catalogue_sid_hit10 - s.iloc[0].sid_hit10_beam256), float("nan"), float("nan"))),
                          title="Scoring the catalogue once from the fully masked state − the beam-256 list (SID HR@10)",
                          x_label="percentage points", pct=True, signed=True),
              SC.dumbbell(DR.grid_rows(pc, lambda s: (float(s.iloc[0].greedy_first_is_digit1), float("nan"), float("nan"))),
                          title="Share of users whose greedy path reveals digit 1 first", x_label="share"),
              SC.dumbbell(DR.grid_rows(pc, lambda s: (float(s.iloc[0].greedy_states_in_trained_family), float("nan"), float("nan"))),
                          title="Share of greedy-path states that training visited", x_label="share"),
              SC.dumbbell(DR.grid_rows(pc, lambda s: (float(s.iloc[0].beam256_final_duplicate_share), float("nan"), float("nan"))),
                          title="Duplicate paths in the final beam-256", x_label="share")]
    return figs(*charts) + QLEG + numbers([("Models accepted", "16 of 20 (four width-512 models differ from the archive by tie-breaks)"),
                                            ("One-shot scoring vs beam-256 list", "2.8 pp lower on average, lower in all 16"),
                                            ("A digit's top code after decoder block 1 / 2 / 3 equals the final one", "36–71 / 56–81 / 73–87 %")])


def _history_use():
    e7 = pd.read_csv(DERIVED / "controlled/exp7_diffusion_history_use/summary-283124-r2/estimates.csv")
    items = []
    for v, label, var in (("rqkmeans_3codebook_128", "RQ-KMeans 3×128", KM), ("rqvae_4codebook_128", "RQ-VAE 4×128", VAE)):
        a1 = e7[(e7.variant == v) & (e7.estimand == "delta S_full") & (e7.r == 1) & (e7.m == 0) & (e7.digit == 0)].iloc[0]
        h1 = e7[(e7.variant == v) & (e7.estimand == "delta | match (pooled)")].iloc[0]
        nm = e7[(e7.variant == v) & (e7.estimand == "delta | nonmatch (pooled)")].iloc[0]
        k1 = e7[(e7.variant == v) & (e7.estimand == "H2a K1 | match (pooled d>=1)")].iloc[0]
        for name, r in (("replace r1, first digit", a1), ("copying: r1 matches the prefix", h1),
                        ("copying: r1 does not match", nm), ("block digit d → r1's slot, all blocks", k1)):
            items.append({"label": f"{label}: {name}", "est": float(r.est), "lo": float(r.lo), "hi": float(r.hi), "var": var})
    return figs(SC.hbars(items, title="Δ golden log-prob (nats)", x_label="nats", label_w=330, width=760),
                png(DERIVED / "controlled/exp7_diffusion_history_use/summary-283124-r2/figures/fig1_recency.png",
                    "Δ at the first digit when the item r positions back is replaced"))


def _route():
    e8 = pd.read_csv(DERIVED / "controlled/exp8_diffusion_copy_route/summary-283159/estimates.csv")
    items = []
    for v, label, var in (("rqkmeans_3codebook_128", "RQ-KMeans 3×128", KM), ("rqvae_4codebook_128", "RQ-VAE 4×128", VAE)):
        for est, name in (("H1 both | match (pooled d>=1)", "block both routes"), ("enc | match (pooled d>=1)", "block the encoder route"),
                          ("dec | match (pooled d>=1)", "block the decoder route"), ("both_ctrl | match (pooled d>=1)", "both routes, control item")):
            r = e8[(e8.variant == v) & (e8.estimand == est)].iloc[0]
            items.append({"label": f"{label}: {name}", "est": float(r.est), "lo": float(r.lo), "hi": float(r.hi), "var": var})
    return SC.hbars(items, title="Copy rows: Δ golden log-prob when routes to r1 are blocked (nats)", x_label="nats", label_w=300, width=720) + numbers(
        [("Both routes / exp7's replacement effect", "0.24 / 0.35 · 0.60 / 0.96 nats (about two thirds)"),
         ("Both − (encoder + decoder)", "+0.08 · +0.30 nats: the routes back each other up")])


def _encoder():
    est = DR.with_cells(pd.read_csv(S_EXP5 / "estimates.csv"))
    acc = est[(est.estimand == "probe_acc") & (est.subset == "all")]
    p1 = SC.profiles(DR.depth_panels(acc, list(range(5)), lambda s, d: DR.first(s, probe=f"P1_own_d{d}", site="output")),
                     title="A slot's own SID digit, read linearly from the encoder output (at the input: 0.98–1.00)",
                     x_labels=["digit 1", "2", "3", "4", "5"], y_label="accuracy", y_domain=(0, 1))
    d = est[(est.estimand == "probe_acc_output_minus_input") & (est.site == "output")]
    sp = SC.dumbbell(DR.grid_rows(d[(d.probe == "P4a_r1_d0") & (d.subset == "all")], lambda s: DR.triple(s.iloc[0])),
                     title="r1's first code read from the OTHER slots: output − input accuracy", x_label="accuracy points", pct=True, signed=True)
    cs1 = SC.dumbbell(DR.grid_rows(est[est.estimand == "CS1"], lambda s: DR.triple(s.iloc[0])),
                      title="Removing spread latents: copy score lost, minus a matched control (nats)", x_label="nats", signed=True)
    rec = est[(est.estimand == "splice_recovered") & (est.probe == "desc") & (est.site == "all digits")]
    dsc = SC.dumbbell(DR.grid_rows(rec, lambda s: DR.triple(s.iloc[0])),
                      title="Every slot replaced by its rebuilt 8-word description: decoder behaviour kept", x_label="share kept")
    fd = est[est.estimand == "field_drop_kl_increase"].groupby("probe").est.mean().sort_values(ascending=False)
    names = {"r1_cat_l2": "recent item's category", "r1_d0": "recent item's first code", "own_d0": "own first code",
             "rec": "recency", "cat_l2": "own category (level 2)", "cat_l1": "own category (level 1)",
             "price_decile": "price decile", "main_cat": "store"}
    drop = SC.hbars([{"label": names.get(k, k), "est": float(v), "var": "--accent"} for k, v in fd.items()],
                    title="Behaviour lost when one description word is replaced by its most common value (mean over models)",
                    x_label="share of mean-ablation KL", label_w=190, width=620)
    return p1 + LLEG + figs(sp, cs1, dsc, drop) + QLEG + numbers(
        [("Own first code at the output (all models)", "0.31–0.53 (input 0.98–1.00)"),
         ("Recency, output − input accuracy", "+0.20 to +0.31"),
         ("Target's first code on new-target users, output − input", "−0.011 to +0.010"),
         ("SAE: unexplained variance / behaviour kept by the splice", "0.06–0.25 / 0.87–0.99"),
         ("Description: unexplained variance / behaviour kept", "0.44–0.76 / 0.39–0.85")])


def _patching():
    est = DR.with_cells(pd.read_csv(S_EXP12 / "estimates.csv"))
    R = est[est.estimand == "R_denoise"]
    o = SC.dumbbell(DR.grid_rows(R[(R.group == "H0") & (R.site == "E:others")], lambda s: DR.triple(s.iloc[0])),
                    title="r1 replaced: share restored by the clean states of all OTHER slots (first digit)", x_label="share of KL restored")
    r1 = SC.dumbbell(DR.grid_rows(R[(R.group == "H0") & (R.site == "E:r1")], lambda s: DR.triple(s.iloc[0])),
                     title="… and by r1's own clean slot", x_label="share of KL restored")
    x = SC.profiles(DR.depth_panels(R[R.group == "H0"], list(range(4)), lambda s, l: DR.first(s, site=f"X:L{l}:query")),
                    title="Share restored by one decoder block's clean cross-attention output at the predicted digit (first digit)",
                    x_labels=["block 1", "2", "3", "4"], y_label="share restored", y_domain=(-0.1, 1))
    q = SC.profiles(DR.depth_panels(R[R.group == "S_all"], list(range(4)), lambda s, l: DR.first(s, site=f"D:L{l}:query")),
                    title="Revealed parent digit changed: share restored by the predicted position's clean residual after each block",
                    x_labels=["block 1", "2", "3", "4"], y_label="share restored", y_domain=(0, 1.05))
    edge = est[(est.estimand == "edge_share") & (est.group == "S_all") & (est.site == "K:all")]
    e = SC.dumbbell(DR.grid_rows(edge, lambda s: DR.triple(s.iloc[0])),
                    title="Share of a revealed digit's effect carried by the direct self-attention edge", x_label="edge share")
    return figs(o, r1) + QLEG + x + q + LLEG + figs(e) + QLEG + numbers(
        [("Restored by other slots / r1's own slot (first digit)", "0.83–0.90 / 0.32–0.53"),
         ("Noising: damage from the replaced history in other slots / r1's slot", "0.47–0.68 / 0.10–0.17"),
         ("Edge share: depth 3 / depth 5", "0.68–0.89 / 0.37–0.66"),
         ("Exactness controls", "all bit-exact; replacement cost matches exp7 (0.56 / 0.48 nats)")])


def _step_switch():
    est = DR.with_cells(pd.read_csv(S_EXP13 / "estimates.csv"))
    est["t"] = pd.to_numeric(est["t"], errors="coerce")
    F = est[(est.beam == 64) & (est.estimand == "F")]
    prof = SC.profiles(DR.depth_panels(F, list(range(5)), lambda s, t: DR.first(s, t=t)),
                       title="F(t): share of affected users whose top-1 becomes the swapped-history one when only step t uses it (beam 64)",
                       x_labels=["step 1", "2", "3", "4", "5"], y_label="share", y_domain=(0, 1))
    f0 = SC.dumbbell(DR.grid_rows(F[F.t == 0], lambda s: DR.triple(s.iloc[0])), title="F(first step), by width", x_label="share")
    return prof + LLEG + figs(f0) + QLEG + numbers(
        [("Users whose top-1 changes when r1 is replaced", "56–88 %"),
         ("F(first step) / F(last step)", "0.16–0.68 / 0.01–0.10"),
         ("F(first step), width 128 / 512", "0.16–0.39 / 0.25–0.68 (512 higher in 9 of 9 pairs)"),
         ("Controls", "no switch = plain decode and full switch = donor decode, bit for bit")])


def _perturb():
    est = DR.with_cells(pd.read_csv(S_EXP14 / "estimates.csv"))
    b = est[est.beam == 256]
    pb = SC.dumbbell(DR.grid_rows(b[(b.estimand == "PB") & (b.level == "sid")], lambda s: DR.triple(s.iloc[0])),
                     title="Blocking every route to r1 for the whole decode: ΔHR@10 minus a control block (SID, beam 256)",
                     x_label="percentage points", pct=True, signed=True)
    rep = SC.dumbbell(DR.grid_rows(b[(b.estimand == "dHR_block_r1") & (b.level == "sid") & (b.group == "repeat")], lambda s: DR.triple(s.iloc[0])),
                      title="… on targets that are r1's own item (4 % of users)", x_label="percentage points", pct=True, signed=True)
    new = SC.dumbbell(DR.grid_rows(b[(b.estimand == "dHR_block_r1") & (b.level == "sid") & (b.group == "new")], lambda s: DR.triple(s.iloc[0])),
                      title="… on new targets (66–89 % of users)", x_label="percentage points", pct=True, signed=True)
    stay = SC.dumbbell(DR.grid_rows(b[(b.estimand == "posthoc_stay_with_item") & (b.group == "copied_r1")], lambda s: DR.triple(s.iloc[0])),
                       title="r1 and r2 swapped: users who copied r1 and still output r1's code (post hoc)", x_label="share")
    sh = SC.dumbbell(DR.grid_rows(b[(b.estimand == "top1_changed") & (b.group == "shift5")], lambda s: DR.triple(s.iloc[0])),
                     title="History moved 5 slots right: share of users whose top-1 changes", x_label="share")
    return figs(pb, rep, new, stay, sh) + QLEG + numbers(
        [("Copy block vs control, SID / item level", "−1.0 to −3.7 / −1.0 to −2.9 pp"),
         ("Swap: users who copied r1 and switch to the item now last", "0–18 %"),
         ("Shift by 5 slots: HR@10 change", "−0.5 to −3.5 pp"),
         ("Models", "19 of 20 (MQ 5×512 failed the archive check)")])


def experiments() -> list[dict]:
    return [
        dict(id="dm-retro-order", code="Retrospective exp1", title="Confidence vs fixed reveal order in the archived runs",
             models="20 archived DiffGRM evaluations", data="saved scores", status="descriptive; not a controlled comparison",
             source="derived/retrospective/exp1_reveal_order/",
             did="<p>DiffGRM's training runs evaluated two decoders: confidence order (reveal the most confident digit first, beam "
                 "256) and one fixed order (beam 64). We compared their saved scores for every run.</p>",
             results=_retro_order,
             takeaways=["Confidence order scored higher in 14 of 18 runs, but the two decoders used different beams.",
                        "This motivated the controlled reruns (controlled exp1 and exp2)."]),

        dict(id="dm-beam", code="Controlled exp1", title="Confidence vs fixed order at the same beam size",
             models="18 DiffGRM models", data="6,297 test users", status="primary analysis specified before results",
             source="derived/controlled/exp1_matched_beam/summary-181944-181946/",
             did="<p>We decoded every model with both policies at the same beam (64 and 256), using one search implementation in "
                 "which only the allowed next position differs.</p>",
             results=_matched_beam,
             takeaways=["At beam 64, confidence order loses to fixed order (−0.95 pp), mostly at depth 5 (−2.6 pp).",
                        "At beam 256 it wins slightly (+0.22 pp).",
                        "Most of a deep confidence beam is the same partial SID reached through different orders, which wastes beam "
                        "slots. Any claim about reveal order must state the beam size."]),

        dict(id="dm-orders", code="Controlled exp2", title="All six fixed orders at depth 3",
             models="6 depth-3 DiffGRM models", data="6,297 test users", status="declared protocol",
             source="derived/controlled/exp2_fixed_orders/summary-194295-194297/",
             did="<p>On the six depth-3 models, we decoded with each of the six fixed reveal orders and with confidence order, all at "
                 "beam 64.</p>",
             results=_fixed_orders,
             takeaways=["At depth 3, confidence order beats the average fixed order by 0.40 pp.",
                        "The best fixed order differs by quantizer, so no single order suits every map."]),

        dict(id="dm-trace", code="Representation exp4", title="Reveal trace: what the decode holds at each step",
             models="20 DiffGRM models (16 accepted)", data="6,297 test users", status="declared protocol; provisional",
             source="derived/representation/exp4_diffusion_reveal_trace/summary-310761-16cells/",
             did="<p>We recorded every reveal step of the search: which digit is revealed, the distributions at each step and decoder "
                 "block, and how many beam paths are duplicates. We also scored the whole catalogue from intermediate states, and "
                 "checked how many decode states belong to the family of states training visits.</p>",
             results=_reveal_trace,
             takeaways=["Decoding step by step adds accuracy over scoring the catalogue once from the fully masked state.",
                        "Which digit is revealed first depends on the quantizer: RQ-VAE most often reveals digit 1 first, MQ least often.",
                        "At depth 5, only 11 % of decode states are states training visited, because depth-5 models never train on "
                        "the fully masked start.",
                        "Four width-512 models are missing because of tie-breaks; the numbers are provisional."]),

        dict(id="dm-history", code="Controlled exp7", title="What DiffGRM uses from the history",
             models="DiffGRM RQ-KMeans 3×128, RQ-VAE 4×128 (the AR maps)", data="6,297 test users", status="declared protocol",
             source="derived/controlled/exp7_diffusion_history_use/summary-283124-r2/",
             did="<p>We repeated AR exp3 and exp4 on the two DiffGRM models that share the AR models' SID maps: replaced history "
                 "items, tested prefix-matched copying, and blocked cross-attention from a digit position to the matching item's "
                 "slot in all four decoder blocks.</p>",
             results=_history_use,
             takeaways=["DiffGRM uses the history the way AR does: the most recent item matters most (0.56 / 0.48 nats; AR: 0.57 / "
                        "0.51), and it copies from items that match the prefix.",
                        "Blocking the decoder's direct read of the matching item's slot changes nothing, so the copy must reach the "
                        "decoder another way (exp8)."]),

        dict(id="dm-route", code="Controlled exp8", title="The copy reaches the decoder through the encoder",
             models="DiffGRM RQ-KMeans 3×128, RQ-VAE 4×128", data="6,297 test users", status="declared protocol",
             source="derived/controlled/exp8_diffusion_copy_route/summary-283159/",
             did="<p>We blocked the encoder route (no other history slot may attend to the matching item), the decoder route (no digit "
                 "position may attend to its slot), or both, and compared with the same blocks on a non-matching item.</p>",
             results=_route,
             takeaways=["The encoder spreads the most recent item into the other history slots, and the decoder reads it there.",
                        "The direct read of the item's own slot matters only when the spread is blocked."]),

        dict(id="dm-encoder", code="Representation exp5", title="What the encoder holds: probes, SAEs and a description autoencoder",
             models="20 DiffGRM models", data="6,297 test users (36,783 slots); fitted on training-target histories",
             status="declared protocol, run 2026-10-14",
             source="derived/representation/exp5_diffusion_encoder_features/summary-312763/",
             did="<p>We read the encoder at three points: its input, after self-attention, and its output. Linear probes tested what "
                 "each slot holds about its own item, its recency, r1 and the target. TopK SAEs (2,048 latents, k = 16) on the "
                 "output were labelled by what each latent tracks; we then removed the latents that carry r1's first code in the "
                 "other slots (spread latents) and measured the copy score.</p>"
                 "<p>We also built a small version of a natural language autoencoder: a verbalizer writes each slot state as eight "
                 "words (recency, category, store, price, first code, and what r1 looks like), and a reconstructor rebuilds the "
                 "state from those words. We then measured how much decoder behaviour the rebuilt states keep, and which word the "
                 "decoder needs most.</p>",
             results=_encoder,
             takeaways=["The single encoder layer turns each slot from 'this item' into 'this user's recent context': a slot's own first "
                        "code drops from almost perfectly readable to 31–53 %, while recency and r1's first code appear in every slot.",
                        "The encoder holds nothing about new targets beyond the history's content, the same as AR.",
                        "Spread latents change the output in 19 of 20 models; they are part of the route exp8 found.",
                        "Eight words keep 39–85 % of the decoder's behaviour, and the words it needs most describe the recent item."]),

        dict(id="dm-patching", code="Controlled exp12", title="Activation patching between paired runs",
             models="20 DiffGRM models", data="6,297 test users", status="declared protocol, run 2026-10-14 (on CPU)",
             source="derived/controlled/exp12_diffusion_activation_patching/summary-312754-312806/",
             did="<p>We ran each model twice on inputs that differ in one thing. In part H, r1 is replaced (exp7's controls). In part S, "
                 "the history is the same and one revealed digit holds a sibling's code. We copied pieces of one run into the other: "
                 "encoder states of r1's slot or of the other slots, the decoder's state after each block at each digit position, "
                 "and each block's cross-attention output. In part S we also cut the self-attention edge from the predicted digit to "
                 "the changed digit.</p>",
             results=_patching,
             takeaways=["The replaced item reaches the decoder mostly through the other history slots, in every model.",
                        "Each digit position reads the history through its own cross-attention, spread over the four blocks; the best "
                        "single block restores 0.34–0.77 of the effect, and the same patch at other digits at most 0.23.",
                        "A revealed digit's information reaches the digit being predicted in the first decoder block, mostly over the "
                        "direct self-attention edge. The edge carries less at depth 5."]),

        dict(id="dm-steps", code="Controlled exp13", title="At which reveal step DiffGRM uses the history",
             models="20 DiffGRM models", data="6,297 test users, greedy and beam 64", status="declared protocol, run 2026-10-14",
             source="derived/controlled/exp13_diffusion_step_switch/summary-312774/",
             did="<p>Only the partial SID passes from one step to the next, so each step can be given a different history. We decoded "
                 "with r1 replaced at one step only, at every step but one, at all steps, or at none, and asked how often the final "
                 "top-1 matches the one from the replaced history.</p>",
             results=_step_switch,
             takeaways=["The first reveal step decides most, but no single step decides alone: the history is used at every step.",
                        "Wider codebooks decide earlier: the first step matters more at width 512 than at 128 in all 9 pairs.",
                        "When one step changes the result, it usually changes which digit is revealed, not only its code."]),

        dict(id="dm-perturb", code="Controlled exp14", title="Perturbations held through decoding: copy block, swap and shift",
             models="19 DiffGRM models", data="6,297 test users, beam 256 and 64", status="declared protocol, run 2026-10-14/15",
             source="derived/controlled/exp14_diffusion_decode_perturbations/summary-312775-313104/",
             did="<p>We held three changes for the whole beam search. First, we blocked every route to r1 (exp8's double block) and "
                 "compared with the same block on an older item. Second, we swapped r1 and r2. Third, we moved the whole history one "
                 "or five slots to the right, keeping the order.</p>",
             results=_perturb,
             takeaways=["Copying helps DiffGRM's lists, and only for items the user already had: −12 to −39 pp on repeat targets, about 0 "
                        "on new targets. This is the AR pattern (AR exp6, retrospective exp7).",
                        "After a swap, DiffGRM keeps copying the same item even though it is no longer last; RQ-KMeans AR follows the "
                        "position instead (AR exp9). The declared swap estimate was mis-scaled, so this rests on a post-hoc measure.",
                        "Moving the history changes the top-1 for 47–82 % of users: DiffGRM reads absolute slot positions. Training never "
                        "put padding first, so part of this effect may come from that."]),
    ]
