"""SID-map experiments (no recommender involved) for the mechanisms review page. Wording follows docs/paper/WRITING.md."""

from __future__ import annotations

import pandas as pd

from build_report import DERIVED, REPO, figs, numbers, png
from sidlens.viz import svgcharts as SC

FIG = REPO / "docs/results/semantic_mapping/figures"
R3 = DERIVED / "structure/exp3_collision_causes/293555/result"

INTRO = """
<h1 class="pagetitle">SID maps (both models)</h1>
<p>Both recommenders write semantic IDs from the same 27 maps: three quantizers (RQ-KMeans, RQ-VAE, MQ) × depth 3, 4 or 5
× codebook width 128, 256 or 512, over the 3,105 Industrial items. These two analyses look at the maps alone; no
recommender is run. They come before the model experiments in the write-up because the model results depend on what a
digit means in each map and on which items share a SID.</p>
"""


def _collisions():
    loop = pd.read_csv(R3 / "partD1_rqvae_loop.csv")
    items = []
    for r in loop.itertuples():
        q, depth, width = r.table.split("_")[0], r.table.split("_")[1][0], r.table.split("_")[2]
        items.append({"label": f"RQ-KMeans {depth}×{width}: native", "est": r.rate_before, "var": "--q-rqkmeans"})
        items.append({"label": f"RQ-KMeans {depth}×{width}: after RQ-VAE's last-digit loop", "est": r.rate_after, "var": "--q-rqvae"})
    tb = pd.read_csv(R3 / "partB_tables.csv")
    mq = tb[tb.quantizer == "MQ"]
    mq_items = [{"label": f"MQ {r.depth}×{r.width}: observed", "est": r.collision_rate, "var": "--q-MQ"} for r in mq.itertuples()
                if r.width == 128] + [{"label": f"MQ {r.depth}×{r.width}: digits shuffled independently", "est": r.n1_rate_p50,
                                        "var": "--muted"} for r in mq.itertuples() if r.width == 128]
    return figs(SC.hbars(items, title="Share of items in a shared SID: RQ-KMeans tables before and after RQ-VAE's last-digit "
                                      "reassignment", x_label="share of items", pct=True, label_w=320, width=720),
                SC.hbars(mq_items, title="MQ: observed collisions vs digits shuffled independently (width 128)",
                         x_label="share of items", pct=True, label_w=300, width=680)) + numbers(
        [("RQ-KMeans tables after the last-digit loop", "all at the identical-input floor: 22 items, 0.71 %"),
         ("Collided items with a near-duplicate in their bucket (nested RQ-KMeans, deepest fits)", "73–85 %"),
         ("Items in the hub code nearest the origin, each level ≥ 2", "9–17 %"),
         ("MQ: mean pairwise NMI between digits", "0.50–0.70")])


def experiments() -> list[dict]:
    return [
        dict(id="maps-stats", code="Structure exp1, exp2 and semantic mapping", title="What each SID digit encodes",
             models="27 SID maps", data="3,105 items, item embeddings, Amazon category labels", status="descriptive, no model",
             source="derived/exp1_atlas/, derived/exp2_geometry/, derived/semantic_mapping/194268/; figures in docs/results/semantic_mapping/figures/",
             did="<p>For every map we measured how much of the item embedding and of the category one digit explains, alone and "
                 "given the earlier digits; how many codes each digit uses; how large prefix groups are; how pure their categories "
                 "are compared with a shuffled-label null; how much tighter they are than random groups of the same size; and how "
                 "many items share a full SID.</p>",
             results=lambda: figs(png(FIG / "fig5_digit_info.png", "Information per digit, alone and given the prefix"),
                                  png(FIG / "fig4_coherence.png", "Category purity of prefix groups against a shuffled null"),
                                  png(FIG / "fig8_collisions.png", "Items that share a full SID, by map")) + numbers(
                 [("Embedding variance explained by digit 1 alone (4×128)", "RQ-KMeans 53 %, RQ-VAE 34 %, MQ 38 %"),
                  ("… by each later digit alone", "RQ-KMeans 12–16 %, RQ-VAE 8–18 %, MQ 36–38 %"),
                  ("Category purity after 3 digits (null)", "RQ-KMeans 0.97 (0.45), RQ-VAE 0.99 (0.50), MQ 0.81 (0.39)"),
                  ("First-codebook codes used by RQ-VAE", "2–31 % in 8 of 9 fits"),
                  ("Items sharing a full SID, 3×128 (1 − unique/N)", "RQ-KMeans 29.3 %, RQ-VAE 1.2 %, MQ 40.1 %")]),
             takeaways=["In RQ-KMeans and RQ-VAE, digit 1 is a real partition of the catalogue and later digits refine it; a later digit "
                        "means something only given the earlier digits.",
                        "MQ's digits are parallel partitions of similar quality that overlap; an MQ prefix is an intersection, not a step "
                        "down a hierarchy.",
                        "The category is settled by digit 3; later digits separate items within a category.",
                        "RQ-VAE leaves most of its first codebook unused and makes its last digit an identity digit."]),

        dict(id="maps-collisions", code="Structure exp3", title="Why items share a SID",
             models="27 SID maps, refits over 5 seeds", data="3,105 items", status="declared protocol; amendments A1–A3 post hoc",
             source="derived/structure/exp3_collision_causes/293555/result/",
             did="<p>We refitted the quantizers (reproducing the archived RQ-KMeans tables exactly), applied each map's post-processing "
                 "to the others, and tested candidate causes of collisions: capacity, embedding norm, whitening, the shared k-means "
                 "seed, a hub code, near-duplicate products, MQ's dependent digits, and RQ-VAE's last-digit reassignment.</p>",
             results=_collisions,
             takeaways=["Each quantizer collides for a different reason. RQ-VAE's near-zero rate comes from a post-processing step that "
                        "reassigns the last digit of colliding items; applied to RQ-KMeans, the same step removes nearly all of its "
                        "collisions.",
                        "Near-duplicate products (same brand, nearly the same title) are the floor of collisions in every map.",
                        "MQ collides because its digits repeat each other. RQ-KMeans 5×512 is a separately balanced table, not the "
                        "extension of the other RQ-KMeans tables.",
                        "Collision rates, and so SID-level hit rates, depend on post-processing as much as on the quantizer."]),
    ]
