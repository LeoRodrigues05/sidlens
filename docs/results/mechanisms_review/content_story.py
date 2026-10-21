"""Storyline tab: research questions, recommended claims with their evidence, write-up order, next steps.

Wording follows docs/paper/WRITING.md. Claims reference experiments by their ids in the other content files; a
reference to an unknown id raises, so a renamed experiment cannot leave a dead link.
"""

from __future__ import annotations

import html

TOC = ('<a href="#story-questions">Opening questions</a><a href="#story-claims">Recommended claims</a>'
       '<a href="#story-order">Order of the write-up</a><a href="#story-context">Context-only experiments</a>'
       '<a href="#story-next">Next steps</a>')

QUESTIONS = [
    ("RQ1 · Maps", "How do RQ-KMeans, RQ-VAE and MQ assign items to SID digits, and why do different items end up with the same SID?"),
    ("RQ2 · Representation", "What does each layer of an AR and a masked-diffusion recommender hold about the final recommendation, "
     "and when does information from the history become the answer?"),
    ("RQ3 · Computation", "Which history information produces each digit, through which parts of the network, and how do the AR "
     "and diffusion recommenders differ across maps?"),
    ("RQ4 · From scores to reported results", "How do decoding and evaluation turn digit scores into reported accuracy, and which "
     "hits come from items the history already identifies?"),
]

CLAIMS = [
    dict(rq="RQ1", title="SID maps differ in what a digit means, and post-processing decides most collisions.",
         evidence="In RQ-KMeans and RQ-VAE, digit 1 alone explains 34–53 % of the embedding variance and later digits 8–18 %; "
                  "MQ's four digits explain 36–38 % each. RQ-VAE's near-zero collision rate comes from a last-digit "
                  "reassignment; the same step removes nearly all RQ-KMeans collisions.",
         support=["maps-stats", "maps-collisions"], preface=[],
         show="The per-digit information figure and the before/after collision chart.",
         caveat="One catalogue (Industrial). Office maps exist but are not analysed.",
         final_main=['M1', 'M2'], final_supp=['M5']),
    dict(rq="RQ2", title="Neither model holds information about new targets beyond the history; what is readable is the item to copy.",
         evidence="AR: the target's first digit is never probed above 'predict the recent item's first digit'; the copy appears in "
                  "the output from layer 20. DiffGRM: the encoder output predicts new targets no better than its input "
                  "(±0.011), while the recent item becomes readable from every slot.",
         support=["ar-lens", "ar-sae", "ar-infomap", "dm-encoder"], preface=["dm-trace"],
         show="AR logit lens by layer; DiffGRM own-digit and spread probes.",
         caveat="Probes and lenses are observational; the SAE ablations are the causal part.",
         final_main=['AR3', 'DG3'], final_supp=['AR6', 'DG7']),
    dict(rq="RQ3", title="Both recommenders copy each digit from a history item that matches the prefix written so far.",
         evidence="AR: changing a matching history item costs 1.35 / 1.06 nats, a non-matching one about 0; this holds in all 12 "
                  "retrained AR models. DiffGRM: 0.35 / 0.96 vs about 0. Both rely most on the most recent item "
                  "(0.57 / 0.51 nats AR, 0.56 / 0.48 DiffGRM).",
         support=["ar-copy", "ar-grid", "dm-history"], preface=["ar-prefix-errors", "ar-history"],
         show="AR match vs non-match; the 12-model grid; DiffGRM's replacement and copying beside AR's.",
         caveat="The AR and DiffGRM test cohorts differ; compare effect sizes side by side only.",
         final_main=['AR1', 'DG1'], final_supp=['AR9']),
    dict(rq="RQ3", title="The two paradigms copy through different routes.",
         evidence="AR reads the matching item's digit token directly, by attention in layers 14–27, spread over many heads "
                  "(top 5 heads: 8–9 %). DiffGRM's encoder spreads the recent item into the other slots: their clean states "
                  "restore 0.83–0.90 of a replaced item's effect against 0.32–0.53 for its own slot, and each decoder digit reads "
                  "the history through its own cross-attention.",
         support=["ar-copy", "ar-history", "dm-route", "dm-patching", "dm-encoder"], preface=["dm-history", "ar-sae"],
         show="AR edge knockout by layer group; DiffGRM other-slot vs own-slot restoration across 20 models.",
         caveat="The systems also differ in size, pretraining and depth; the route is compared, not the scores.",
         final_main=['AR1', 'DG2'], final_supp=['AR5', 'DG7']),
    dict(rq="RQ3", title="DiffGRM decides most of the SID at the first reveal step, and digits exchange information in the first decoder block.",
         evidence="Switching the history at the first step alone gives the swapped-history result for 16–68 % of affected users (more "
                  "at width 512 in 9 of 9 pairs), at the last step for 1–10 %. A revealed digit's effect reaches the predicted "
                  "digit after block 1 (0.65–0.96 restored), mostly over the direct edge (0.37–0.89, lower at depth 5).",
         support=["dm-steps", "dm-patching"], preface=["dm-trace"],
         show="F(t) by step and width; the edge share by depth.",
         caveat="One checkpoint per map; widths also differ by SID table.",
         final_main=['DG3'], final_supp=['DG6']),
    dict(rq="RQ3", title="Neither model uses time. AR follows list position; DiffGRM follows the item but reads absolute slot positions.",
         evidence="In the data, the recent item predicts the target mainly on the same day. RQ-KMeans AR copies it whatever its age "
                  "and prefers whichever item is last (position effect 0.36–0.39). After a swap, DiffGRM keeps copying the same item "
                  "(0–18 % follow the slot), but moving the history five slots changes its top-1 for 47–82 % of users.",
         support=["ar-order", "ar-time", "dm-perturb"], preface=[],
         show="AR copy rate by gap against the data; AR position effect; DiffGRM swap and shift.",
         caveat="DiffGRM's swap result rests on a post-hoc measure (the declared one was mis-scaled); shifted histories start "
                "with padding, which training never did.",
         final_main=[], final_supp=['AR7', 'DG8']),
    dict(rq="RQ3", title="In next-two prediction, the second item is copied from the first.",
         evidence="Replacing item 1 costs item 2's first digit 0.60 nats, against 0.20 for the most recent history item; copying "
                  "between the two predicted items costs 0.43–0.73 nats in matching rows.",
         support=["ar-nexttwo"], preface=["ar-nexttwo-assoc"],
         show="Short subsection or appendix.",
         caveat="One next-two model (MQ 4×256).",
         final_main=[], final_supp=['AR8', 'DG9']),
    dict(rq="RQ4", title="Search settings change the reported comparison of reveal orders.",
         evidence="At beam 64, confidence order loses 0.95 pp of HR@10 to fixed order (2.6 pp at depth 5); at beam 256 it wins by 0.22 pp. "
                  "Duplicate paths fill 57–91 % of the confidence beam. The best fixed order differs by quantizer.",
         support=["dm-beam", "dm-orders"], preface=["dm-retro-order", "dm-trace"],
         show="Confidence − fixed by depth at both beams.",
         caveat="SID-level HR; item-level is lower.",
         final_main=['DG5'], final_supp=[]),
    dict(rq="RQ4", title="Copying raises hit rates only on items the user already had, and SID-level metrics overstate the gain.",
         evidence="Blocking copying in beam search costs AR 27 / 10 pp on repeat targets and DiffGRM 12–39 pp, and about 0 on truly new "
                  "items. In AR, collision partners and near-duplicate variants carry 56 % and 80 % of the loss, and only 28 % of "
                  "RQ-KMeans' top-rank benefit remains at item level. SID HR@10 exceeds item HR@10 by 4.5 pp.",
         support=["ar-decoding", "ar-credit", "dm-perturb", "ar-grid"], preface=["ar-collisions", "ar-time"],
         show="ΔHR@10 by target relation for both paradigms; SID vs item credit.",
         caveat="Repeat and duplicate-record targets are a small share of users but a large share of hits.",
         final_main=['AR2', 'DG4', 'M4'], final_supp=['S2']),
]

ORDER = [
    ("Opening", "State the four questions. Explain SIDs, the two paradigms, and that we compare what they compute on shared maps, not their scores."),
    ("RQ1 · Maps", "maps-stats, then maps-collisions."),
    ("RQ2 · Representation", "ar-lens → ar-sae → ar-infomap for AR; dm-encoder and dm-trace for DiffGRM."),
    ("RQ3 · Computation", "ar-prefix-errors (motivation) → ar-history → ar-copy → ar-lastdigit → ar-grid; then dm-history → dm-route → "
                          "dm-patching → dm-steps; then ar-order, ar-time and the swap/shift part of dm-perturb; next-two (ar-nexttwo) last."),
    ("RQ4 · Scores to results", "dm-retro-order → dm-beam → dm-orders for search; then ar-decoding → ar-credit → ar-collisions and the copy-block "
                                "part of dm-perturb for what the hits measure."),
    ("Limits", "One checkpoint per map; provisional numbers until the final-run checkpoints are analysed; different AR and DiffGRM cohorts."),
]

CONTEXT = [
    ("ar-nexttwo-assoc", "An association in saved lists; exp5 is the causal test. Appendix."),
    ("dm-retro-order", "Archived runs with unequal beams; it motivates exp1 and exp2 but tests nothing on its own."),
    ("ar-collisions", "Supports the item-credit claim; one sentence and a table may be enough."),
    ("ar-lastdigit", "Shows how map post-processing changes copying of the last digit. RQ-VAE's declared estimate includes 0, so it "
                     "fits a discussion paragraph better than a main claim."),
]

NEXT = """
<ol class="order">
<li><b>Final-run checkpoints.</b> Every number on this page is provisional. The paper's numbers will come from rerunning the
analysis on the kept checkpoints of the final training run. The <a href="#final">Final important experiments</a> tab lists
the experiments to run on them, in three lines (codebooks and collisions, Qwen-AR, DiffGRM), main before supplementary. We
expect the trends to hold, but the write-up must use the rerun numbers.</li>
<li><b>Write-up.</b> Open with the four questions, then follow the order above. Use the claim cards as the outline: one claim per
subsection, its supporting experiments as the evidence, and the prefacing experiments as the setup.</li>
<li><b>Open items in the current results.</b> Four width-512 DiffGRM models are missing from the reveal trace (tie-breaks; the
gate needs an amendment), MQ 5×512 is missing from controlled exp14, and the 17 next-two AR recovery checkpoints are not
registered. The final run replaces all of these models.</li>
</ol>
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def page(index: dict) -> str:
    def link(eid: str) -> str:
        if eid not in index:
            raise KeyError(f"storyline refers to unknown experiment {eid!r}")
        e = index[eid]
        return f'<a class="pill" href="#{eid}">{esc(e["code"])}: {esc(e["title"])}</a>'

    def order_text(text: str) -> str:
        out = text
        for eid in sorted(index, key=len, reverse=True):
            if eid in out:
                out = out.replace(eid, f'<a href="#{eid}">{esc(index[eid]["code"])}</a>')
        return out

    def final_links(c: dict) -> str:
        main = "".join(f'<a class="pill" href="#final-{f.lower()}">{esc(f)}</a>' for f in c["final_main"])
        supp = "".join(f'<a class="pill" href="#final-{f.lower()}">{esc(f)}</a>' for f in c["final_supp"])
        return (f"main {main or '—'} · supplementary {supp or '—'}")

    qs = "".join(f"<li><b>{esc(a)}.</b> {esc(b)}</li>" for a, b in QUESTIONS)
    claims = []
    for i, c in enumerate(CLAIMS, 1):
        sup = "".join(link(e) for e in c["support"])
        pre = "".join(link(e) for e in c["preface"]) or "—"
        claims.append(f'<div class="claim"><span class="rq">{esc(c["rq"])} · claim {i}</span><h3>{esc(c["title"])}</h3>'
                      f'<p>{esc(c["evidence"])}</p><dl><dt>Supporting</dt><dd>{sup}</dd><dt>Sets it up</dt><dd>{pre}</dd>'
                      f'<dt>What to show</dt><dd>{esc(c["show"])}</dd><dt>Caveats</dt><dd>{esc(c["caveat"])}</dd>'
                      f'<dt>Final experiments</dt><dd>{final_links(c)}</dd></dl></div>')
    order = "".join(f"<li><b>{esc(a)}.</b> {order_text(esc(b))}</li>" for a, b in ORDER)
    context = "".join(f"<li>{link(e)} {esc(t)}</li>" for e, t in CONTEXT)
    used = {e for c in CLAIMS for e in c["support"] + c["preface"]} | {e for e, _ in CONTEXT}
    if missing := sorted(set(index) - used):
        raise ValueError(f"experiments not placed in the storyline: {missing}")
    return (
        '<h1 class="pagetitle">Storyline and recommendations</h1>'
        "<p>This tab proposes how to organize the write-up. It lists the questions to open with, the claims we recommend making, "
        "and for each claim the experiments that support it and the experiments that set it up. Every experiment on the other "
        "tabs appears here at least once. Click an experiment to open it.</p>"
        f'<h2 id="story-questions">Opening questions</h2><ol>{qs}</ol>'
        f'<h2 id="story-claims">Recommended claims</h2>{"".join(claims)}'
        f'<h2 id="story-order">Order of the write-up</h2><ol class="order">{order}</ol>'
        f'<h2 id="story-context">Experiments that serve as context only</h2><ul>{context}</ul>'
        f'<h2 id="story-next">Next steps</h2>{NEXT}')
