"""Fixed sentences of the exp4 report; every number comes from ``report.compute``.

Written after reading facts.md of run 327442. Each function gets the facts
``F(catalog, key)`` and ``both(key)`` (both catalogs, one line) and returns
HTML. Wording follows docs/paper/WRITING.md.
"""

from __future__ import annotations

import html


def e(x):
    return html.escape(str(x))


def ul(items):
    return '<ul class="findings">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def summary(F, both):
    I, O = "industrial", "office"
    return ul([
        "<b>The tables are what they claim to be.</b> All 54 native tables and 39 ZCR twins match their recorded hashes; "
        "every ZCR twin keeps its native prefixes, has no shared SID and changes the minimum number of last digits. "
        "RQ-KMeans maps of one width are truncations of one fit in both catalogs, which the draft marked as unchecked. "
        "The RQ-VAE and MQ weights reproduce their tables.",
        f"<b>Residual digits refine their prefix; MQ digits repeat each other.</b> In RQ-KMeans 4×128, digits 1–4 on their own "
        f"explain {e(F(I, 'var_alone_rqkmeans_4x128'))} of the embedding variance (Industrial; Office {e(F(O, 'var_alone_rqkmeans_4x128'))}). "
        f"Every MQ digit explains about the same share (largest/smallest {e(F(I, 'mq_var_ratio'))} on Industrial, {e(F(O, 'mq_var_ratio'))} on Office).",
        f"<b>In RQ-VAE, digit 2 carries the most meaning, as it did in the old maps.</b> On Industrial it uses {e(F(I, 'rv_util_d1'))} of its first codebook, "
        f"and given digit 1, digit 2 explains {e(F(I, 'rv_d2_given'))} of the variance against {e(F(I, 'rv_d1_given'))} for "
        f"digit 1 (Office {e(F(O, 'rv_d2_given'))} against {e(F(O, 'rv_d1_given'))}). The old maps show the same (paper page); "
        f"the draft reports digits alone only. Digits 4–5 add at most {e(F(I, 'rv_late_given_max'))}: they separate items.",
        f"<b>Three digits sort items into categories.</b> Among items that share their first three digits with another item "
        f"(4×128), the share in their group's main category is, on Industrial (level-1 categories): RQ-KMeans {e(F(I, 'purity3_rqkmeans'))}; "
        f"MQ {e(F(I, 'purity3_MQ'))}. On Office (level-2 categories): RQ-KMeans {e(F(O, 'purity3_rqkmeans'))}; MQ {e(F(O, 'purity3_MQ'))}.",
        f"<b>RQ-VAE's low collision rate is its post-processing, now shown directly.</b> Its own nearest codes collide "
        f"{e(F(I, 'rv_pre'))} of items (Office {e(F(O, 'rv_pre'))}); the published tables collide {e(F(I, 'rv_pub'))} "
        f"({e(F(O, 'rv_pub'))}). Re-running the reassignment loop from the nearest codes reproduces "
        f"{e(F(I, 'rv_loop_reproduces'))} on Industrial and {e(F(O, 'rv_loop_reproduces'))} on Office.",
        f"<b>Near-duplicate products are the floor.</b> In the 5-digit RQ-KMeans maps, {e(F(I, 'nd_deepest'))} of collided items "
        f"share their SID with a same-brand, near-identical-title neighbour (Office {e(F(O, 'nd_deepest'))}).",
        f"<b>MQ's dependent digits have a simple cause.</b> MQ's encoders never train: every encoder weight in all 18 MQ models is "
        f"still at its random initialisation, so each digit quantizes a different random projection of the same item. Mean "
        f"pairwise AMI between digits is {e(F(I, 'pair_ami_MQ'))} for MQ against {e(F(I, 'pair_ami_rqkmeans'))} for RQ-KMeans (Industrial).",
        f"<b>Many test targets share their SID with a different history item (exploratory).</b> RQ-KMeans maps: "
        f"{e(F(I, 'x_partner_rqkmeans'))} of Industrial next-item targets, {e(F(O, 'x_partner_rqkmeans'))} of Office's; "
        f"repeats are {e(F(I, 'x_repeat'))} / {e(F(O, 'x_repeat'))}. ZCR twins have none.",
    ])


def validation(F, both):
    return ul([
        "Hashes: every native table matches the sha256 in <code>done.json</code> and <code>tables.json</code>; every ZCR index matches its "
        "<code>manifest.json</code> and names its native table by hash.",
        "Nesting: the 3- and 4-digit RQ-KMeans tables are the prefixes of the 5-digit table for every item at every width, in both "
        "catalogs. RQ-VAE and MQ depths are separate fits.",
        "RQ-VAE: the checkpoint's encoder and codebooks give every table's digits 1 to D−1 for every item, and the exported last-level "
        "residuals to within 1.5e-5.",
        "MQ: the eval-mode encoders give every digit of every table, except one Office item in MQ 5×512 whose two nearest codewords "
        "differ in distance by a relative 1.8e-7 (float32 tie).",
        "RQ-KMeans: the tables cannot be reproduced by refitting on this cluster. On Industrial 3×128, per-level k-means and faiss's "
        "ResidualQuantizer give digits 1–2 exactly and digit 3 for about a third of items, at 1, 8 and 96 threads (amendment A1). The published codebooks "
        "reproduce every table by nearest centroid, so the post-processing and hub-code analyses use them; Part C is a "
        "counterfactual on this cluster's refits.",
    ])


def expectations(F, both):
    return ("<p>Declared in <code>protocol.md</code> before any estimand was computed. E2 was declared for Industrial only. "
            "Failures are reported as they are.</p>")


def digits(F, both):
    I, O = "industrial", "office"
    return ul([
        f"Variance explained by each digit alone, 4×128 (digits 1–4). RQ-KMeans: {both('var_alone_rqkmeans_4x128')}. "
        f"RQ-VAE: {both('var_alone_rqvae_4x128')}. MQ: {both('var_alone_MQ_4x128')}.",
        f"The same, given the earlier digits. RQ-KMeans: {both('var_given_rqkmeans_4x128')}. RQ-VAE: {both('var_given_rqvae_4x128')}. "
        f"MQ: {both('var_given_MQ_4x128')}.",
        f"Across all nine maps of each family: RQ-KMeans digit 1 alone {both('rk_d1_alone')}, later digits {both('rk_later_alone')}; "
        f"MQ digits {both('mq_var_alone_all')}. RQ-VAE digit 2 given digit 1 explains more than digit 1 in {both('rv_d2_beats_d1')} maps.",
        f"Codes in use. RQ-KMeans every code ({both('rk_util')}); MQ {both('mq_util')}; RQ-VAE's first digit {both('rv_util_d1')}.",
        f"Category purity at prefix length 3, 4×128 (Industrial level-1, Office level-2 categories): "
        f"RQ-KMeans {both('purity3_rqkmeans')}; RQ-VAE {both('purity3_rqvae')}; MQ {both('purity3_MQ')}. "
        f"Range over residual maps with D ≥ 4: {both('purity3_residual_all')}; MQ maps: {both('purity3_mq_all')}.",
        f"Conditional category AMI per digit, 4×128 († = too few items per parent): RQ-KMeans {both('ami_rqkmeans_4x128')}; "
        f"RQ-VAE {both('ami_rqvae_4x128')}; MQ {both('ami_MQ_4x128')}.",
        f"Refinement index at full length, 5×128 (0 = no tighter than a random split, 1 = no spread left): {both('ri_5x128')}.",
        f"Items alone after the last digit: RQ-KMeans {both('alone_last_rqkmeans')}; RQ-VAE {both('alone_last_rqvae')}; MQ "
        f"{both('alone_last_MQ')}. Share of RQ-VAE 4×128 items alone at prefix length 3: {both('unique3_rqvae_4x128')} "
        f"(79.6 % on the old map). The draft's 92.5 % is a different statistic, the share of items whose SID any three "
        f"of its four digits determine; see the paper page.",
        f"Bits of item identity resolved by the full SID (log2 N = {e(F(I, 'bits_budget'))} / {e(F(O, 'bits_budget'))}): "
        f"RQ-KMeans {both('bits_full_rqkmeans')}; RQ-VAE {both('bits_full_rqvae')}; MQ {both('bits_full_MQ')}.",
    ])


def collisions(F, both):
    I, O = "industrial", "office"
    return ul([
        f"Collision rates (share of items in a shared SID). RQ-KMeans {both('rate_rqkmeans')}; RQ-VAE {both('rate_rqvae')}; "
        f"MQ {both('rate_MQ')}. At 3×128: {both('rate_3x128')}.",
        f"Floor: {both('identical')} have identical embeddings. No map gives two identical items different SIDs "
        f"({both('identical_split')} maps), so the old exception, the balanced RQ-KMeans 5×512 table, is gone.",
        f"<b>RQ-VAE (direct test, new).</b> Before reassignment {both('rv_pre')}; published {both('rv_pub')}. The loop rewrote the "
        f"last digit of {both('rv_changed')} of items. Published within 0.5 points of the floor: {both('rv_at_floor')}.",
        f"<b>The same loop on the other families.</b> RQ-KMeans {both('rk_loop')}; MQ {both('mq_loop')}. Left more than 0.5 points "
        f"above the floor: Industrial {e(F(I, 'mq_loop_notfloor'))}; Office {e(F(O, 'rk_loop_notfloor'))}, {e(F(O, 'mq_loop_notfloor'))}. "
        f"On Office's MQ maps at W = 128 one prefix group holds more items than the last codebook has codes, so no reassignment can "
        f"separate them; ZCR is infeasible on the same maps ({e(F(O, 'zcr_infeasible'))}). The other maps left above the floor "
        f"are feasible: the loop stops at its 20-round limit there. Five Office MQ maps end slightly below the floor: MQ's "
        f"exported vectors of identical inputs differ by up to 2.7e-6 (GPU arithmetic), enough for the loop to split a few "
        f"identical groups. Each moved item's added last-level "
        f"squared distance is {both('rk_loop_cost')} the table's mean (RQ-KMeans).",
        f"<b>Near-duplicates.</b> Share of collided items with a near-duplicate bucket-mate in the nested RQ-KMeans maps, depths "
        f"3 → 4 → 5 (%): {both('nd_trend')}. Collided items' closest bucket-mate is among the closest 0.1 % of all catalogue pairs "
        f"for {both('closest_mate')} of them (RQ-KMeans). Near-duplicate pairs: {both('near_dup')}.",
        f"<b>Hub codes (RQ-KMeans).</b> At levels 2–5 the most-used code is the centroid nearest the origin at "
        f"{both('hub_rank0')} levels; it holds {both('hub_share')} of items. Parents with a smaller radius collide more: "
        f"Spearman {both('radius_spearman')}. The reused k-means seed leaves its mark: digit l equals digit l−1 for "
        f"{both('repeat_rate')} of items.",
        f"<b>MQ.</b> Mean pairwise AMI between digits (chance-corrected): MQ {both('pair_ami_MQ')}; RQ-KMeans "
        f"{both('pair_ami_rqkmeans')}; RQ-VAE {both('pair_ami_rqvae')}. The plug-in NMI the draft quotes overlaps between families "
        f"(MQ {both('pair_nmi_MQ')}, RQ-KMeans {both('pair_nmi_rqkmeans')}), so it should not carry the claim. Observed collision "
        f"rate against independent digits: {both('mq_obs_n1')}; at 3×128 {both('mq_3x128_n1')}.",
        f"<b>MQ encoders are untrained.</b> {both('mq_init')}: every encoder Linear weight and bias lies within PyTorch's "
        f"default ±1/√fan-in range with the spread of a uniform draw, BatchNorm affine parameters are exactly 1 and 0, and "
        f"the learned position vector is within its ±1/K initial range. <code>rq/mq.py</code> runs the encoders under "
        f"<code>no_grad</code>, so only the codebooks and decoder train.",
    ])


def refits(F, both):
    I, O = "industrial", "office"
    return ("<p>Counterfactual refits on this cluster (amendment A1): every condition is compared with this cluster's own "
            "native refit over 5 seeds; differences are in points, depth 3 / depth 5, and count only when the 5-seed ranges do "
            "not overlap. The native refit is close to, but not the same as, the published tables: "
            f"{e(F(I, 'c_native_vs_pub'))} (Industrial); {e(F(O, 'c_native_vs_pub'))} (Office).</p>"
            + ul([f"Per-level seed: {both('c_per-level seed')}. The old finding (the shared seed lowers collisions, "
                  "+13.0 points at 3×128) holds at depth 3 on Industrial at W = 128 and 256 and on Office only at W = 512; "
                  "at depth 5 no width shows it.",
                  f"Scale-normalised residuals: {both('c_scale-norm')}.",
                  f"L2-normalised embeddings: {both('c_L2')}.",
                  f"Top-256 PCs whitened (vs unwhitened): {both('c_PCA-256-white')}.",
                  f"Top-64 PCs whitened (vs top-256 unwhitened): {both('c_PCA-64-white')}."]))


def zcr(F, both):
    return ul([
        f"Twins: {both('zcr_n')}. Tables without a twin are below the 5 % threshold (RQ-VAE: {both('zcr_below5')}) or "
        f"infeasible (Office: {e(F('office', 'zcr_infeasible'))}).",
        f"Last digits changed: RQ-KMeans {both('zcr_changed_rqkmeans')}; MQ {both('zcr_changed_MQ')}; RQ-VAE {both('zcr_changed_rqvae')}.",
        f"Last-level squared distance, relative change: RQ-KMeans {both('zcr_cost_rqkmeans')}; MQ {both('zcr_cost_MQ')}; RQ-VAE "
        f"{both('zcr_cost_rqvae')} (its twin moves the few items the loop left colliding to nearer free codes).",
        f"RQ-KMeans 3×128: {both('zcr_rk3x128')}.",
    ])


def exposure(F, both):
    return ("<p>Exploratory. Next-item test target = each user's last interaction, history = up to 50 earlier items, from "
            "the frozen sequences. Classes in order: repeat (item in the history), SID partner (another history item has its "
            "SID), variant (a near-duplicate of a history item), new.</p>" + ul([
                f"Test targets: {both('x_targets')}; repeats {both('x_repeat')} (the same for every map).",
                f"SID partners: RQ-KMeans {both('x_partner_rqkmeans')}; MQ {both('x_partner_MQ')}; RQ-VAE {both('x_partner_rqvae')}.",
                f"RQ-KMeans 3×128: {both('x_rk3x128')}. ZCR turns SID partners into variants.",
                f"Collided items are bought more often than their share of the catalogue: RQ-VAE collides "
                f"{both('rate_rqvae')} of items but {both('x_inter_rqvae')} of interactions; RQ-KMeans "
                f"{both('rate_rqkmeans')} of items, {both('x_inter_rqkmeans')} of interactions; MQ {both('rate_MQ')} of items, "
                f"{both('x_inter_MQ')} of interactions.",
                f"New targets: {both('x_new')}.",
            ]))


def latent(F, both):
    return ul([
        f"First-level codewords that are nobody's nearest code: {both('lat_dead')}.",
        f"Median norm of codewords, unused against used: {both('lat_norm')}. The unused codewords sit near the latent origin, "
        "away from the data; this fits codewords that were initialised and never chosen again, but the training history is "
        "not saved, so that is a reading, not a test.",
        f"Digit 1 explains {both('lat_var')} of the latent's variance, against {both('rv_d1_alone')} of the input embedding's variance.",
    ])


def m5(F, both):
    return ul([
        f"Held-out AUC of a digit-1 cell's own title terms against its siblings (median over cells; range over 9 maps): {both('naming_d1')}.",
        f"Co-purchase lift of items sharing digit 1 (also_buy, the one label not derived from text): {both('copurchase')}.",
        "The explorer was rebuilt on the v3 tables for both catalogs (<code>derived/semantic_mapping/v3-industrial-20261020/</code>, "
        "<code>derived/semantic_mapping/327460/</code>).",
    ])


def old(F, both):
    I = "industrial"
    return ul([
        f"RQ-KMeans digit 1 equals the old table's for {e(F(I, 'old_rk_d1'))} of items; digit 2 for {e(F(I, 'old_rk_d2'))}.",
        f"{e(F(I, 'old_focal'))}. RQ-VAE and MQ maps are new fits, so results on the old RQ-VAE 4×128 and MQ maps do not transfer item by item.",
    ])


def claims(F, both):
    return "<p>Each RQ1 number of the 2026-10-05 draft (sections/05results.tex), recomputed on v3.</p>"


def caveats(F, both):
    return ul([
        "No model is run: nothing here measures what a collision costs a trained recommender (final plan M3, M4).",
        "Each map is one fit; Part C ranges cover k-means initialisation only, on this cluster's refits.",
        "Category labels and the embedding both come from item text; co-purchase is the one behavioural check.",
        "Office uses level-2 categories as its primary field because its level-1 breadcrumb has 3 classes.",
        "The exposure classes use the frozen interaction sequences, not the v3 split files.",
    ])


def files(run):
    return ul([f"Run: <code>{e(run)}</code> (<code>industrial/</code>, <code>office/</code>, <code>summary/</code>).",
               "Code: <code>experiments/structure/exp4_final_maps/</code>; loader <code>src/sidlens/data/codebooks_v3.py</code>.",
               "Every number on this page is in <code>summary/facts.md</code> with the table it comes from."])


NARRATIVE = {"summary": summary, "validation": validation, "expectations": expectations, "digits": digits,
             "collisions": collisions, "refits": refits, "zcr": zcr, "exposure": exposure, "latent": latent,
             "m5": m5, "old": old, "claims": claims, "caveats": caveats, "files": files}
