#!/usr/bin/env python
"""The paper-facing page of structure exp4: every map result the WWW'27 draft
displays, its value on the old maps and on the v3 maps, and what to change.

Reads only ``<run>/summary`` (summarize.py, figures.py) and
``<run>/summary/compare`` (compare_old.py) and writes

    docs/results/final_maps/sidlens_final_maps_paper.html   self-contained page
    docs/results/final_maps/sidlens_final_maps_paper_artifact.html
                                                            the same page as a claude.ai
                                                            page body (no document wrapper;
                                                            safe-area gutters; the viewer's
                                                            own theme switch instead of ours)
    docs/results/final_maps/figures/fig_v3_vs_old.*         the comparison figure

What the paper displays was read from the draft of 2026-10-05
(sections/05results.tex RQ1, 01introduction.tex, 00abstract.tex,
06discussion.tex, 04setup_20261019_long.tex, appendix_setup.tex) and the paper
plan of 2026-10-19 (Sections 5.1-5.4). Every number on the page is formatted
from a CSV here; the sentences around them were written after reading run
327442's numbers, so another run needs them reread.

    python experiments/structure/exp4_final_maps/paper_report.py --run <job dir>
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as RP  # noqa: E402  (facts: the same numbers as the full report)

from sidlens import paths  # noqa: E402
from sidlens.data import codebooks_v3 as V  # noqa: E402
from sidlens.viz.diffusion_report import page  # noqa: E402

DOCS = paths.REPO / "docs/results/final_maps"
I, O = "industrial", "office"
TAG_LABEL = {"holds": "holds", "update": "new numbers", "changed": "statement changes", "new": "new result",
             "drop": "drop", "reference": "reference"}
TAG_CSS = {"holds": "held", "update": "partial", "changed": "failed", "new": "held", "drop": "failed",
           "reference": "partial"}

TITLE = "Final-run SID maps"

# claude.ai wraps a published page in its own document and theme switch, and
# pads :root by the phone's safe areas; these overrides keep the side gutter
# clear of a side system bar and drop the page's own light/dark button.
ARTIFACT_CSS = """
.layout{padding-block:24px 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
nav.toc{top:calc(16px + env(safe-area-inset-top,0px))}
.theme-toggle{display:none}
"""

EXTRA_CSS = """
.tag{display:inline-block;font-size:11.5px;font-weight:600;padding:1px 7px;border-radius:9px;border:1px solid var(--border);white-space:nowrap}
.tag.held{color:var(--good)} .tag.failed{color:var(--bad)} .tag.partial{color:var(--ink-2)}
.legend-row{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}
figure img{width:100%;height:auto;background:#fff;border-radius:6px}
.cap{font-size:13px;color:var(--ink-2);margin-top:6px}
td.where{font-size:12px;color:var(--ink-2)}
"""


def e(x) -> str:
    return html.escape(str(x))


def tag(t: str) -> str:
    return f'<span class="tag {TAG_CSS[t]}">{e(TAG_LABEL[t])}</span>'


def img(path: Path, alt: str) -> str:
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{e(alt)}">'


def table(df: pd.DataFrame, raw_cols=(), cls=None) -> str:
    cls = cls or {}
    head = "".join(f"<th>{e(c)}</th>" for c in df.columns)
    rows = []
    for r in df.itertuples(index=False):
        cells = []
        for c, x in zip(df.columns, r):
            cells.append(f'<td class="{cls.get(c, "")}">{x if c in raw_cols else e(x)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def ul(items) -> str:
    return '<ul class="findings">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def build(run: Path) -> str:
    S = run / "summary"
    F, T = RP.compute(S)
    C = pd.read_csv(S / "compare/claims.csv").set_index("claim")
    maps = pd.read_csv(S / "compare/maps.csv")
    src = json.loads((S / "compare/sources.json").read_text())
    rb = pd.read_csv(S / "b_rqvae_buckets.csv")
    tj = {c: V.tables_json(c)["tables"] for c in (I, O)}

    def f(c, k):
        return e(F(c, k))

    def cl(k, col):
        return e(C.loc[k, col])

    def both(k):
        return f"{f(I, k)} (Industrial), {f(O, k)} (Office)"

    toc, body = [], []

    def sec(anchor, title, inner, level=2):
        toc.append((anchor, title, level))
        body.append(f'<h{level} id="{anchor}">{e(title)}</h{level}>{inner}')

    # ------------------------------------------------------------ intro --
    counts = C.tag.value_counts()
    legend = '<div class="legend-row">' + "".join(
        f"{tag(t)} <span class='muted'>{int(counts.get(t, 0))}</span>" for t in TAG_LABEL) + "</div>"
    sec("about", "What this page is", f"""
<div class="card"><p>The paper's map results (RQ1, the introduction's first finding, the abstract, the discussion's
design table and the setup) were computed on the old frozen Industrial maps. The final run trains on the v3 maps:
27 per catalog for Industrial and Office, with ZCR twins. This page lists every map result the paper displays, with
its value on the old maps, recomputed from the paper's own data package and the old collision-cause run, and on the
v3 maps of both catalogs. Each result is tagged with what the text needs:</p>{legend}
<p class="muted">Sources: run <code>{e(run)}</code> (structure exp4, job 327442); v3 tables
<code>{e(V.REPO_ID)}</code> at <code>{e(V.REVISION[:8])}</code>. The full analysis, with every part and expectation, is
<a href="sidlens_final_maps.html">sidlens_final_maps.html</a>. Every number below is computed from CSVs in the run's
<code>summary/</code> and <code>summary/compare/</code>.</p></div>""")

    bl = pd.read_csv(S / "rqvae_bucket_latent.csv")

    def distinct_buckets(c):
        g = bl[bl.catalog == c]
        if g.empty:
            return "none"
        dead = g[g.dead_layer.fillna("") != ""]
        live = g[g.dead_layer.fillna("") == ""]
        parts = [f"{t.replace('rqvae_', '').replace('codebook_', '×')}: {int(s)} items mapped to one latent point "
                 f"(every unit of the encoder's {d.split(' (')[0].replace('hidden layer 1 of 6', 'first hidden layer')} is "
                 f"zero for them)" for t, s, d in zip(dead.table, dead["size"], dead.dead_layer)]
        if len(live):
            parts.append(f"{len(live)} buckets of {int(live['size'].min())}–{int(live['size'].max())} items in "
                         f"{live.table.nunique()} maps, with close but distinct latents")
        return "; ".join(parts)

    # ------------------------------------------------------ statements --
    def trio(c):          # the three 4x128 purities at length 3, primary field, and the null range
        k = "purity3"
        return C.loc[k, "v3_" + c] if c in (I, O) else C.loc[k, "old_industrial"]

    mqm = T["mq_models"]
    n_init = int(mqm.all_at_init.astype(bool).sum())
    if n_init != len(mqm):
        raise AssertionError("an MQ model is not at initialisation; the MQ sentence would be wrong")
    statements = ul([
        f"<b>Three digits sort items into categories</b> (RQ1 ¶1; intro finding 1; Figure map-digits c). Share of items "
        f"in their group's main category among items that share their first three digits with another item, "
        f"RQ-KMeans / RQ-VAE / MQ, 4×128: Industrial (level-1 categories) {e(trio(I))}; Office (level-2 categories) "
        f"{e(trio(O))}; old maps {e(trio(None))}. The introduction's \"81–99 %\" becomes \"79–99 %\".",
        f"<b>Residual digits refine their prefix; MQ digits repeat each other</b> (RQ1 ¶2; Figure map-digits b). "
        f"Variance explained by digit 1 alone, RQ-KMeans / RQ-VAE, 4×128: Industrial {cl('digit1_alone', 'v3_industrial')}; "
        f"Office {cl('digit1_alone', 'v3_office')}; old {cl('digit1_alone', 'old_industrial')}. Later residual digits "
        f"alone: {cl('later_residual_alone', 'v3_industrial')} (Industrial), {cl('later_residual_alone', 'v3_office')} "
        f"(Office). Each MQ digit alone: {cl('mq_digits_alone', 'v3_industrial')} (Industrial), "
        f"{cl('mq_digits_alone', 'v3_office')} (Office).",
        f"<b>Add for RQ-VAE: given digit 1, digit 2 explains the most variance.</b> Variance explained by digit 2 given "
        f"digit 1 (digit 1 in brackets), 9 RQ-VAE maps: Industrial {cl('rqvae_digit2_given', 'v3_industrial')}; Office "
        f"{cl('rqvae_digit2_given', 'v3_office')}; old {cl('rqvae_digit2_given', 'old_industrial')}. This is not new in v3: "
        f"the draft reports digits alone only, which hides it.",
        f"<b>The quantizers place item identity in different digits</b> (RQ1 ¶3; plan P4–P5). Share of RQ-VAE's first "
        f"codebook in use, 9 maps: Industrial {cl('rqvae_first_codebook', 'v3_industrial')}; Office "
        f"{cl('rqvae_first_codebook', 'v3_office')}; old {cl('rqvae_first_codebook', 'old_industrial')}. RQ-VAE 4×128 items "
        f"with their own SID after four digits: {cl('rqvae_alone4', 'v3_industrial')} (Industrial), "
        f"{cl('rqvae_alone4', 'v3_office')} (Office). RQ-VAE 4×128 items whose full SID any three of their four digits "
        f"determine: {cl('rqvae_three_of_four', 'v3_industrial')} (Industrial), {cl('rqvae_three_of_four', 'v3_office')} "
        f"(Office), old {cl('rqvae_three_of_four', 'old_industrial')}. Items in a shared SID at 3×128, RQ-KMeans / MQ: "
        f"{cl('collision_3x128', 'v3_industrial')} (Industrial), {cl('collision_3x128', 'v3_office')} (Office).",
        f"<b>Near-duplicate products set the floor</b> (RQ1 ¶4; table row All). Identical embeddings: {f(I, 'identical')} "
        f"on Industrial, {f(O, 'identical')} on Office; all 27 maps of each catalog give every identical group one SID. "
        f"Collided items with a same-brand, near-identical-title bucket-mate in the 5-digit RQ-KMeans maps: Industrial "
        f"{cl('near_dup', 'v3_industrial')}; Office {cl('near_dup', 'v3_office')}.",
        f"<b>Post-processing sets RQ-VAE's rate; now shown directly</b> (RQ1 ¶5; plan P6). Items in a shared SID under "
        f"RQ-VAE's own nearest codes → in the published table: {f(I, 'rv_pre')} → {f(I, 'rv_pub')} (Industrial), "
        f"{f(O, 'rv_pre')} → {f(O, 'rv_pub')} (Office). Re-running the reassignment from the nearest codes gives back the "
        f"published table: {f(I, 'rv_loop_reproduces')} on Industrial, {f(O, 'rv_loop_reproduces')} on Office. The same "
        f"step applied to RQ-KMeans: {f(I, 'rk_loop')} on Industrial, changing up to {e(C.loc['loop_rqkmeans', 'v3_industrial'].split('up to ')[1])} "
        f"of last digits; {f(O, 'rk_loop')} on Office. What the loop leaves in RQ-VAE: {e(distinct_buckets(I))} "
        f"(Industrial); {e(distinct_buckets(O))} (Office). The old analysis could only infer such a collapse; the "
        f"checkpoints now show it.",
        f"<b>RQ-KMeans hub codes</b> (RQ1 ¶5). At levels 2–5, the code whose centroid is nearest the origin is the most "
        f"used at {f(I, 'hub_rank0')} levels on Industrial and {f(O, 'hub_rank0')} on Office, and holds {f(I, 'hub_share')} "
        f"and {f(O, 'hub_share')} of items.",
        f"<b>MQ digits repeat each other</b> (RQ1 ¶5). Mean pairwise AMI between digits, per map: {cl('mq_ami', 'v3_industrial')} "
        f"(Industrial), {cl('mq_ami', 'v3_office')} (Office). MQ 3×128, observed against independent digits: "
        f"{cl('mq_n1', 'v3_industrial')} (Industrial), {cl('mq_n1', 'v3_office')} (Office). Cause: MQ's encoders never "
        f"train (every encoder parameter of all {n_init} MQ models is at its initialisation), so each digit quantizes a "
        f"different random projection of the same item.",
        f"<b>ZCR</b> (RQ1 ¶6). Last digits ZCR changes in RQ-KMeans 3×128: {cl('zcr_3x128', 'v3_industrial')} on "
        f"Industrial; {cl('zcr_3x128', 'v3_office')} on Office; old {cl('zcr_3x128', 'old_industrial')}.",
        "<b>Implication</b> (RQ1 last paragraph). Unchanged. Add that the reassignment does not reach the identical-input "
        "floor on Office maps whose prefix groups are larger than the last codebook.",
    ])
    sec("statements", "What the RQ1 text should say", "<p>In the order of the draft's paragraphs; each item names "
        "where it goes. Draft wording follows <code>docs/paper/WRITING.md</code>.</p>" + statements)

    # ---------------------------------------------------------- figures --
    fd = S / "figures"
    figs = [
        ("fig_v3_map_digits", "Replaces fig_map_digits (RQ1)",
         "What each digit encodes in the 4×128 maps; top row Industrial, bottom row Office. (a, d) Codes in use. (b, e) "
         "Embedding variance explained by one digit alone. (c, f) Among items that share a prefix with another item, the "
         "share in their group's main category (Industrial level-1, Office level-2), solid, and after shuffling "
         "categories, dotted."),
        ("fig_v3_map_identity", "Replaces fig_map_identity (RQ1)",
         "Where the maps resolve item identity; top row Industrial, bottom row Office. (a, e) Bits resolved by the first k "
         "digits (4×128). (b, f) Refinement lost when one digit is deleted (4×128). (c, g) Share of items whose SID is "
         "shared, all 27 maps. (d, h) Refinement index by prefix length."),
        ("fig_v3_post_processing", "New: supports the RQ-VAE paragraph and the collision-cause table",
         "Share of items in a shared SID before and after RQ-VAE's last-digit reassignment. (a, d) RQ-VAE: its own "
         "nearest codes against the published table. (b, c, e, f) The same reassignment applied to RQ-KMeans and MQ. "
         "The grey line is the identical-input floor."),
    ]
    inner = ""
    for stem, role, cap in figs:
        inner += (f"<figure><figcaption>{e(role)}</figcaption>{img(fd / f'{stem}.png', stem)}"
                  f"<p class='cap'><b>Caption draft.</b> {e(cap)}</p><p class='cap muted'>Files: "
                  f"<code>docs/results/final_maps/figures/{stem}.pdf</code> and <code>.csv</code> (the plotted numbers).</p></figure>")
    sec("figures", "Figures for the paper", inner)

    # ----------------------------------------------------------- table --
    old_rows = {
        "All": "Floor: 11 identical-embedding pairs (0.71 %) share a SID in 26 of 27 maps; 73–85 % of collided items at "
               "the deepest nested RQ-KMeans fits have a same-brand, near-identical bucket-mate",
        "RQ-VAE": "Last-digit reassignment; applied to RQ-KMeans it takes every table from 10–45 % to 0.71 %; three RQ-VAE "
                  "tables keep buckets it could not split (26, 10, 19 items)",
        "RQ-KMeans": "A near-origin hub code at every level after the first holds 9–17 % of items; parents with a smaller "
                     "radius collide more (Spearman −0.41 to −0.68)",
        "RQ-KMeans 5×512": "Sinkhorn balancing: 0.64 %; first-digit neighbour agreement halves",
        "MQ": "Dependent digits (NMI 0.50–0.70); 3×128: 54.5 % observed, 7.4 % with independent digits",
        "Not causes": "Capacity (uniform codes: 0.15 % at 3×128), norms, whitening; the reused k-means seed lowers "
                      "collisions (+13.0 points at 3×128 without it)",
    }

    def new_row(c, k):
        if k == "All":
            return (f"Floor: {F(c, 'identical')} with identical embeddings, one SID in all 27 maps; near-duplicate "
                    f"bucket-mate for {C.loc['near_dup', 'v3_' + c]} of collided items in the 5-digit RQ-KMeans maps")
        if k == "RQ-VAE":
            return (f"Last-digit reassignment: {F(c, 'rv_pre')} before → {F(c, 'rv_pub')} published (from the checkpoints); "
                    f"applied to RQ-KMeans {C.loc['loop_rqkmeans', 'v3_' + c].split(';')[0]}; buckets of distinct items it "
                    f"did not split (≥ 3 items): {distinct_buckets(c)}")
        if k == "RQ-KMeans":
            return (f"Hub code nearest the origin holds {F(c, 'hub_share')} of items at levels 2–5; Spearman "
                    f"{F(c, 'radius_spearman')}")
        if k == "RQ-KMeans 5×512":
            return f"Drop the row: not balanced in v3 ({C.loc['rk_5x512', 'v3_' + c]})"
        if k == "MQ":
            return (f"Dependent digits: mean pairwise AMI {F(c, 'pair_ami_MQ')} (residual maps {F(c, 'pair_ami_rqkmeans')} "
                    f"RQ-KMeans, {F(c, 'pair_ami_rqvae')} RQ-VAE); 3×128: {F(c, 'mq_3x128_n1')}; encoders untrained")
        cs = C.loc["per_level_seed", "v3_" + c]
        return (f"Capacity (uniform codes {C.loc['capacity', 'v3_' + c]} at 3×128); L2 normalisation (within seed ranges); "
                f"the reused seed: {cs} at 3×128")
    tdf = pd.DataFrame([{"Maps": k, "Draft (old maps, Industrial)": old_rows[k],
                         "v3 Industrial": new_row(I, k), "v3 Office": new_row(O, k)} for k in old_rows])
    sec("table", "Table: why items share a SID", "<p>The draft's Table (tab:collision-causes) against the same rows on "
        "v3. Rates are shares of items in a shared SID.</p>" + table(tdf)
        + "<p class='muted'>Whitening is not a consistent cause on v3 either: whitening the top 256 PCs raises collisions "
          f"at width 128 ({e(F(I, 'c_PCA-256-white').split(',')[0])} on Industrial), whitening the top 64 lowers them at "
          "every width (Part C of the full report, refits on this cluster).</p>")

    # ---------------------------------------------------- claim by claim --
    cc = C.reset_index()
    cc = cc[cc.claim != "rqvae_unique_prefix3"]
    show = pd.DataFrame({"": [tag(t) for t in cc.tag], "Draft statement": cc.paper_text, "Where": cc["where"],
                         "Old maps": cc.old_industrial, "v3 Industrial": cc.v3_industrial, "v3 Office": cc.v3_office,
                         "Note": cc.status})
    sec("claims", "Claim by claim", "<p>Every map number in the draft and the paper plan, recomputed. Old values come "
        "from the files the draft's numbers came from (the paper data package of 2026-09-28 and the old collision-cause "
        "run), with the same definitions as the v3 values.</p>" + table(show, raw_cols=("",), cls={"Where": "where"}))

    # ------------------------------------------------- other paper spots --
    tw = {c: sorted(n for n, r in tj[c].items() if r["zcr"] == "built") for c in (I, O)}
    lat = T["x_latent"]
    other = ul([
        f"<b>Abstract</b> (\"depends mostly on post-processing and near-duplicate products\"): holds in both catalogs "
        f"(statements 5 and 6 above).",
        "<b>Introduction, finding 1</b> (\"share a product category in 81–99 % of cases\"): 79–99 % on v3 (statement 1). "
        "\"Collision rates depend mostly on post-processing\": holds; \"most collisions that remain in long RQ-KMeans SIDs "
        "are between near-duplicate products\": holds.",
        "<b>Discussion, design table, row Quantizer</b>: holds with the numbers above. <b>Row Post-processing</b>: drop "
        "\"Sinkhorn balancing RQ-KMeans 5×512's\"; v3 has no balanced table.",
        f"<b>Discussion, limitation</b> \"Map statistics use the input embedding, because RQ-VAE's latent space was not "
        f"saved\": no longer true. The v3 RQ-VAE checkpoints are published; in its own latent, RQ-VAE's digit 1 explains "
        f"{both('lat_var')} of the variance (input embedding: {both('rv_d1_alone')}). Keep the input-space statistics for "
        f"comparability and say the latent ones are in the appendix or supplement.",
        "<b>Setup, SID maps</b> (<code>\\pending{check on the v3 tables}</code>): RQ-KMeans 3/4/5 are nested at every "
        "width in both catalogs. RQ-VAE and MQ depths are separate fits.",
        f"<b>Setup, ZCR twins</b> (<code>\\pending{{list of ZCR twins}}</code>): {len(tw[I])} on Industrial "
        f"({e(', '.join(n.replace('codebook_', '×') for n in tw[I]))}) and {len(tw[O])} on Office "
        f"({e(', '.join(n.replace('codebook_', '×') for n in tw[O]))}). Office MQ 3×128, 4×128 and 5×128 are infeasible.",
        "<b>Setup, data</b> (<code>\\pending{Office metadata}</code>): Office labels are built (100 % join). Its level-1 "
        "breadcrumb has 3 classes, so Office purity uses level-2 categories (42 classes); say so in the setup.",
        "<b>Appendix, \"Redundant MQ digits\"</b>: measure with AMI, not normalized mutual information (claim mq_nmi).",
        "<b>Paper plan P5</b>: \"92.5 % of RQ-VAE 4×128 items have a unique 3-digit prefix\" misdescribes the number. "
        "It is the share of items whose full SID is determined by any three of their four digits; the unique-3-digit-"
        f"prefix share is {e(C.loc['rqvae_unique_prefix3', 'old_industrial'])} on the old map and "
        f"{e(C.loc['rqvae_unique_prefix3', 'v3_industrial'])} on v3.",
    ])
    sec("elsewhere", "Other places in the paper", other)

    # ----------------------------------------------------- old against v3 --
    rk = maps[maps.quantizer == "rqkmeans"]
    why = ul([
        f"<b>RQ-KMeans is the same procedure on other hardware.</b> v3 keeps the old first digit for {f(I, 'old_rk_d1')} of "
        f"items and the old second digit for {f(I, 'old_rk_d2')}; later digits differ because FAISS's k-means gives "
        "different results on different hardware from level 3 on. RQ-KMeans collision shares move by "
        f"{100 * (rk[rk.variant != 'rqkmeans_5codebook_512'].collision_v3 - rk[rk.variant != 'rqkmeans_5codebook_512'].collision_old).abs().max():.1f} "
        "points at most, except 5×512.",
        f"<b>RQ-KMeans 5×512 was a separate, balanced fit</b>; on v3 it is the nested 5-digit extension of 3×512 and "
        f"4×512 ({cl('rk_5x512', 'old_industrial')} → {cl('rk_5x512', 'v3_industrial')}).",
        f"<b>RQ-VAE and MQ are new fits</b> ({f(I, 'old_focal')}). MQ is also a new implementation (TokenRec's code). Their "
        "map-level statistics are close to the old ones (panels b–d), so the statements carry over, but results tied to "
        "particular old items or SIDs do not.",
    ])
    shutil.copyfile(S / "compare/fig_v3_vs_old.png", DOCS / "figures/fig_v3_vs_old.png")
    shutil.copyfile(S / "compare/fig_v3_vs_old.pdf", DOCS / "figures/fig_v3_vs_old.pdf")
    shutil.copyfile(S / "compare/fig_v3_vs_old.csv", DOCS / "figures/fig_v3_vs_old.csv")
    mshow = maps[["variant", "collision_old", "collision_v3", "digit1_codes_old", "digit1_codes_v3", "purity3_old",
                  "purity3_v3", "mean_pair_ami_old", "mean_pair_ami_v3", "nd_share_collided_old", "nd_share_collided_v3"]].copy()
    for col in mshow.columns[1:]:
        if "codes" not in col:
            mshow[col] = mshow[col].map(lambda x: "" if pd.isna(x) else (f"{100 * x:.1f}" if "ami" not in col else f"{x:.2f}"))
    mshow.columns = ["map", "shared SID % old", "v3", "digit-1 codes old", "v3", "purity@3 % old", "v3",
                     "pair AMI old", "v3", "near-dup % old", "v3"]
    sec("old", "Old maps against v3 (Industrial)",
        f"<figure><figcaption>Old against v3, Industrial</figcaption>{img(S / 'compare/fig_v3_vs_old.png', 'old vs v3')}"
        "<p class='cap'>(a) Share of items in a shared SID, one point per map. (b) Variance explained by one digit alone, "
        "4×128: v3 solid, old dotted. (c) Category purity at prefix length 3 (level-1 categories), per map. (d) Share of "
        "the first codebook in use, per map. Dotted diagonals mark equal values.</p></figure>"
        + why + f"<details><summary>All 27 Industrial maps, old against v3</summary>{table(mshow)}</details>")

    # --------------------------------------------------- not yet in reach --
    xe = T["x_exposure"]
    sec("pending", "Not yet available for the paper", ul([
        "<b>Plan 5.3 (how collisions change the reported score): P9–P12, and final plan M3/M4</b> need the final run's "
        "predictions (SID- against item-level Hit@10 per run, native minus ZCR twin, hits by target class). This analysis "
        "has none: <code>$SIDLENS_WORK/runs/</code> holds no exported final-run run yet, and the codebook dataset holds only maps.",
        f"<b>What the maps already give 5.3</b> (exploratory): the share of next-item test targets whose SID a different "
        f"history item carries is {f(I, 'x_partner_rqkmeans')} under the RQ-KMeans maps on Industrial and "
        f"{f(O, 'x_partner_rqkmeans')} on Office; repeats are {f(I, 'x_repeat')} and {f(O, 'x_repeat')}; the ZCR twins "
        f"have no SID partners. Copying a history item's full SID can earn a SID-level hit only on these targets and on repeats.",
        "<b>Seeds.</b> Each map is one fit; there is no fit-to-fit interval for any map statistic.",
    ]))

    # ------------------------------------------------------------ sources --
    srows = pd.DataFrame([{"input": k, "path": v_["path"], "sha256": v_["sha256"][:16] + "…"} for k, v_ in src.items()])
    sec("sources", "Sources", "<p>Old values: the files below (copies of the /l files are in "
        "<code>summary/compare/old_exp3/</code> and <code>old_zcr/</code>). v3 values: the run's <code>summary/</code>. "
        "Code: <code>experiments/structure/exp4_final_maps/</code> (<code>compare_old.py</code>, "
        "<code>paper_report.py</code>).</p>" + table(srows))
    out = page(TITLE, toc, "".join(body),
               f"RQ1 results on the final-run (v3) maps, against the draft's old maps · structure exp4, job 327442 · "
               f"built {date.today().isoformat()}")
    return out.replace("</style>", EXTRA_CSS + "</style>", 1)


def artifact_body(full: str) -> str:
    """The standalone page as a claude.ai page body: title, style, then content."""
    import re
    style = re.search(r"<style>(.*?)</style>", full, flags=re.S).group(1)
    body = re.search(r"<body>(.*)</body>", full, flags=re.S).group(1)
    if re.search(r"<html[\s>]|<head[\s>]|<body[\s>]|<!doctype", body, flags=re.I):
        raise ValueError("page body still holds document tags")
    return (f"<title>{html.escape(TITLE)}</title>\n"
            f'<meta name="claude:viewport-fit-sides" content="cover">\n'
            f"<style>{style}{ARTIFACT_CSS}</style>\n{body}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    (DOCS / "figures").mkdir(parents=True, exist_ok=True)
    html_ = build(args.run)
    (DOCS / "sidlens_final_maps_paper.html").write_text(html_)
    (DOCS / "sidlens_final_maps_paper_artifact.html").write_text(artifact_body(html_))
    print("wrote", DOCS / "sidlens_final_maps_paper.html", f"{len(html_) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
