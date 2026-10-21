#!/usr/bin/env python
"""Old (frozen, Industrial) map results against the v3 results of structure exp4.

The old numbers are the ones the WWW'27 draft cites. They are recomputed here
from the files they came from, never typed:

    docs/paper/archive/rewrite-20260928/data/sidlens_*.csv   the paper's own data
        package (digit structure, purity, collisions, refinement), each checked
        against the sha256 in its sidlens_manifest.json
    .../data/new/exp2_refinement_*_recomputed.csv            bits, digit deletion,
        digit-pair AMI/NMI (checked against data/new/SHA256SUMS)
    --old-exp3 <dir>                                         structure exp3's
        result directory (job 293555: Parts B, C, D1)
    --old-zcr <dir>                                          the old ZCR inputs
        (derived/substrate/zcr_inputs/20261004), for the changed-tail count

The two exp3/ZCR directories lived on /l only; every file read from them is
copied into the output with its sha256, so the comparison survives /l going
away again.

Writes <run>/summary/compare/ (must not exist):
    sources.json            every old input: path, sha256
    old_exp3/, old_zcr/     copies of the old files used
    claims.csv              one row per paper number: old, v3 Industrial, v3 Office
    maps.csv                one row per Industrial map, old against v3
    digits.csv              one row per Industrial map and digit, old against v3
    fig_v3_vs_old.{pdf,png,csv}

Traps:
* Same definitions on both sides. The old collision rate is the paper's
  (items in a shared SID, `items_in_collision_groups_pct`), not the takeaways'
  1 - unique/N; purity is the multi-item purity of fig4; variance-alone is
  fig5's `standalone_between_share`.
* Old digit indices start at 1 in the paper package and exp2 files, as in v3's
  m1_digits; old exp3 levels start at 0, as v3's hub table does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from sidlens import paths  # noqa: E402
from sidlens.data.codebooks_v3 import load_table  # noqa: E402
from sidlens.data.sids import SidTable  # noqa: E402
from sidlens.viz import style as st  # noqa: E402


def _v3_keys() -> list[str]:
    """Industrial ASINs in item-index order (the order of every v3 table)."""
    return load_table("industrial", "rqkmeans_3codebook_128").keys

PKG = paths.REPO / "docs/paper/archive/rewrite-20260928/data"
QO, QC, QL = st.QUANTIZER_ORDER, st.QUANTIZER_COLOR, st.QUANTIZER_LABEL
WIDTH_MARKER = {128: "o", 256: "s", 512: "^"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def v(q, d, w):
    return f"{q}_{d}codebook_{w}"


def pct(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f} %"


def _num(x, nd):
    return f"{x:.{nd}f}".replace("-", "−")


def prng(s, nd=1):
    s = pd.Series(s).dropna()
    if s.empty:
        return "n/a"
    lo, hi = _num(100 * s.min(), nd), _num(100 * s.max(), nd)
    return f"{lo} %" if lo == hi else f"{lo}–{hi} %"


def rng(s, nd=2):
    s = pd.Series(s).dropna()
    if s.empty:
        return "n/a"
    lo, hi = _num(s.min(), nd), _num(s.max(), nd)
    return lo if lo == hi else f"{lo}{' to ' if s.min() < 0 else '–'}{hi}"


def overlap_words(disjoint) -> str:
    return "5-seed ranges do not overlap" if bool(disjoint) else "5-seed ranges overlap, so no change"


def sid_determined_by_d_minus_1(codes: np.ndarray) -> float:
    """Share of items whose full SID is fixed by any D-1 of its D digits.

    Mean over the D leave-one-digit-out subsets of
    `full_sid_determined_fraction_item_uniform` from the 2026-09-10 catalogue
    mask-support audit (OneDiffRec sidlensreview/.../catalogue_mask_support.py,
    `stats`): an item counts when every catalogue item sharing its visible
    digits has the same full SID. The draft's "92.5 %" for RQ-VAE 4x128 is this
    number; the paper plan paraphrases it as "a unique 3-digit prefix", which is
    a different statistic (the share of items alone at prefix length 3).
    """
    n, D = codes.shape
    _, rep = np.unique(codes, axis=0, return_index=True)
    vals = []
    for hide in range(D):
        visible = [d for d in range(D) if d != hide]
        _, inv, cnt = np.unique(codes[:, visible], axis=0, return_inverse=True, return_counts=True)
        inv = inv.reshape(-1)
        sid_count = np.bincount(inv[rep], minlength=len(cnt))
        vals.append(float((sid_count[inv] == 1).mean()))
    return float(np.mean(vals))


def load_old(old_exp3: Path, old_zcr: Path, out: Path) -> tuple[dict, dict]:
    src = {}
    man = json.loads((PKG / "sidlens_manifest.json").read_text())
    pkg = {}
    for key, rec in man["sources"].items():
        if key in ("digit_structure", "category_purity", "map_collisions", "refinement"):
            f = PKG / rec["packaged_csv"]
            if sha(f) != rec["packaged_sha256"]:
                raise ValueError(f"{f} does not match the paper manifest")
            pkg[key] = pd.read_csv(f)
            src[f"paper/{f.name}"] = {"path": str(f), "sha256": rec["packaged_sha256"], "came_from": rec.get("source")}
    sums = dict(line.split()[::-1] for line in (PKG / "new/SHA256SUMS").read_text().splitlines() if line.strip())
    for name in ("exp2_refinement_digits_recomputed.csv", "exp2_refinement_digit_pairs_recomputed.csv"):
        f = PKG / "new" / name
        if sha(f) != sums[name]:
            raise ValueError(f"{f} does not match data/new/SHA256SUMS")
        pkg[name.split("_recomputed")[0]] = pd.read_csv(f)
        src[f"paper/new/{name}"] = {"path": str(f), "sha256": sums[name]}
    (out / "old_exp3").mkdir()
    exp3 = {}
    for name in ("partB_tables.csv", "partC_a2_hub_codes.csv", "partC_summary.csv", "partD1_rqvae_loop.csv",
                 "partA_catalogue.json"):
        f = old_exp3 / name
        shutil.copyfile(f, out / "old_exp3" / name)
        src[f"exp3/{name}"] = {"path": str(f), "sha256": sha(f)}
        exp3[name.split(".")[0]] = (json.loads(f.read_text()) if name.endswith(".json") else pd.read_csv(f))
    (out / "old_zcr").mkdir()
    f = old_zcr / "rqkmeans_3codebook_128" / "manifest.json"
    shutil.copyfile(f, out / "old_zcr" / "rqkmeans_3codebook_128.manifest.json")
    src["zcr/rqkmeans_3codebook_128/manifest.json"] = {"path": str(f), "sha256": sha(f)}
    exp3["zcr_3x128"] = json.loads(f.read_text())
    (out / "sources.json").write_text(json.dumps(src, indent=1))
    return pkg, exp3


def build(run: Path, old_exp3: Path, old_zcr: Path) -> None:
    S = run / "summary"
    out = S / "compare"
    out.mkdir(exist_ok=False)
    pkg, e3 = load_old(old_exp3, old_zcr, out)
    T = {k: pd.read_csv(S / f"{k}.csv") for k in ("m1_digits", "m1_depths", "m1_pairs", "b_tables", "d_loop", "hub",
                                                  "c_summary", "zcr")}
    dg = T["m1_digits"][T["m1_digits"].kind == "native"]
    dp = T["m1_depths"][T["m1_depths"].kind == "native"]
    pa = T["m1_pairs"][T["m1_pairs"].kind == "native"]
    B, D, H, C, Z = T["b_tables"], T["d_loop"], T["hub"], T["c_summary"], T["zcr"]
    A = {c: json.loads((run / c / "partA_catalogue.json").read_text()) for c in ("industrial", "office")}
    od, opur, ocol, oref = pkg["digit_structure"], pkg["category_purity"], pkg["map_collisions"], pkg["refinement"]
    odig, opair = pkg["exp2_refinement_digits"], pkg["exp2_refinement_digit_pairs"]
    oB, oH, oC, oD = e3["partB_tables"], e3["partC_a2_hub_codes"], e3["partC_summary"], e3["partD1_rqvae_loop"]

    # ---------------------------------------------------------------- maps --
    rows = []
    for _, r in ocol.iterrows():
        name = r.variant
        q, D_, W = r.quantizer, int(r.depth), int(r.width)
        b = B[(B.catalog == "industrial") & (B.table == name)].iloc[0]
        ob = oB[oB.table == name].iloc[0]
        rows.append({
            "variant": name, "quantizer": q, "depth": D_, "width": W,
            "collision_old": r.items_in_collision_groups_pct / 100, "collision_v3": b.collision_rate,
            "digit1_codes_old": int(od[(od.variant == name) & (od.digit == 1)].codes_used.iloc[0]),
            "digit1_codes_v3": int(dg[(dg.catalog == "industrial") & (dg.variant == name) & (dg.digit == 1)].codes_used.iloc[0]),
            "purity3_old": float(opur[(opur.variant == name) & (opur.field == "cat_l1") & (opur.depth == 3)].purity.iloc[0]),
            "purity3_v3": float(dp[(dp.catalog == "industrial") & (dp.variant == name) & (dp.depth == 3)].cat_l1_purity.iloc[0]),
            "refinement_full_old": float(oref[(oref.variant == name) & (oref.depth == D_)].refinement_index.iloc[0]),
            "refinement_full_v3": float(dp[(dp.catalog == "industrial") & (dp.variant == name) & (dp.depth == D_)].refinement_index.iloc[0]),
            "nd_share_collided_old": ob.nd_share_collided, "nd_share_collided_v3": b.nd_share_collided,
            "mean_pair_nmi_old": ob.mean_pairwise_nmi, "mean_pair_nmi_v3": b.mean_pairwise_nmi,
            "mean_pair_ami_old": float(opair[opair.variant == name].ami.mean()),
            "mean_pair_ami_v3": float(pa[(pa.catalog == "industrial") & (pa.variant == name)].ami.mean()),
            "n1_median_old": ob.n1_rate_p50, "n1_median_v3": b.n1_rate_p50,
            "identical_pairs_split_old": int(ob.identical_pairs_split), "identical_pairs_split_v3": int(b.identical_pairs_split),
            "balanced_old": bool(ob.all_levels_balanced), "balanced_v3": bool(b.all_levels_balanced),
        })
    maps = pd.DataFrame(rows)
    maps.to_csv(out / "maps.csv", index=False)
    drow = []
    for _, r in od.iterrows():
        n = dg[(dg.catalog == "industrial") & (dg.variant == r.variant) & (dg.digit == r.digit)].iloc[0]
        drow.append({"variant": r.variant, "quantizer": r.quantizer, "digit": int(r.digit),
                     "var_alone_old": r.standalone_between_share, "var_alone_v3": n.var_alone,
                     "codes_used_old": int(r.codes_used), "codes_used_v3": int(n.codes_used)})
    digits = pd.DataFrame(drow)
    digits.to_csv(out / "digits.csv", index=False)

    # -------------------------------------------------------------- claims --
    claims = []

    # tag: holds (keep the text), update (same statement, new numbers), changed
    # (the statement must change), new (not in the draft), drop, reference.
    TAG = {"digit1_alone": "update", "later_residual_alone": "holds", "rqvae_digit2_given": "new",
           "e1_rule": "new", "mq_digits_alone": "holds", "refinement_5x128": "update", "purity3": "update",
           "rqvae_first_codebook": "update", "rqvae_alone4": "holds", "rqvae_three_of_four": "update",
           "rqvae_unique_prefix3": "reference", "collision_3x128": "update", "collision_ranges": "changed",
           "identical_floor": "changed", "near_dup": "update", "loop_rqkmeans": "changed", "rqvae_direct": "new",
           "hub": "update", "radius_spearman": "holds", "mq_nmi": "changed", "mq_ami": "new", "mq_n1": "update",
           "rk_5x512": "drop", "capacity": "holds", "per_level_seed": "changed", "zcr_3x128": "update"}

    def add(cid, paper_text, where, old, ind, off, status):
        claims.append({"claim": cid, "tag": TAG[cid], "paper_text": paper_text, "where": where, "old_industrial": old,
                       "v3_industrial": ind, "v3_office": off, "status": status})

    def va(df, name, digit, col):
        return float(df[(df.variant == name) & (df.digit == digit)][col].iloc[0])

    def v3d(cat, name, digit, col="var_alone"):
        return float(dg[(dg.catalog == cat) & (dg.variant == name) & (dg.digit == digit)][col].iloc[0])

    def v3p(cat, name, depth, col):
        return float(dp[(dp.catalog == cat) & (dp.variant == name) & (dp.depth == depth)][col].iloc[0])

    PF = {"industrial": "cat_l1", "office": "cat_l2"}
    add("digit1_alone", "Digit 1 alone explains 53 % (RQ-KMeans) and 34 % (RQ-VAE) of the variance (4×128)",
        "05results RQ1 ¶2; fig:map-digits(b); plan 5.1 P2",
        f"{pct(va(od, v('rqkmeans', 4, 128), 1, 'standalone_between_share'))} / {pct(va(od, v('rqvae', 4, 128), 1, 'standalone_between_share'))}",
        *[f"{pct(v3d(c, v('rqkmeans', 4, 128), 1))} / {pct(v3d(c, v('rqvae', 4, 128), 1))}" for c in ("industrial", "office")],
        "RQ-KMeans holds; RQ-VAE lower on Industrial, higher on Office")
    lat = lambda df, col, filt: df[filt & (df.digit > 1)][col]  # noqa: E731
    add("later_residual_alone", "Each later residual digit explains 8–18 % (4×128)", "05results RQ1 ¶2; plan 5.1 P2",
        prng(lat(od, "standalone_between_share", od.variant.isin([v("rqkmeans", 4, 128), v("rqvae", 4, 128)])), 0),
        *[prng(lat(dg, "var_alone", (dg.catalog == c) & dg.variant.isin([v("rqkmeans", 4, 128), v("rqvae", 4, 128)])), 0)
          for c in ("industrial", "office")], "holds (range shifts by 1–2 points)")
    og = []
    for name in [x for x in oref.variant.unique() if x.startswith("rqvae")]:
        r_ = oref[oref.variant == name].set_index("depth").r2_within
        og.append({"variant": name, "d1": 1 - r_[1], "d2": r_[1] - r_[2]})
    og = pd.DataFrame(og)
    add("rqvae_digit2_given", "(new) RQ-VAE: variance explained by digit 2 given digit 1 (digit 1 in brackets), all 9 maps",
        "replaces part of RQ1 ¶2",
        f"{prng(og.d2, 0)} ({prng(og.d1, 0)}); digit 2 > digit 1 in {int((og.d2 > og.d1).sum())}/9",
        *[(lambda g: f"{prng(g[g.digit == 2].var_given_prefix, 0)} ({prng(g[g.digit == 1].var_given_prefix, 0)}); digit 2 > digit 1 in "
                     f"{int((g[g.digit == 2].set_index('variant').var_given_prefix > g[g.digit == 1].set_index('variant').var_given_prefix).sum())}/9")(
            dg[(dg.catalog == c) & (dg.quantizer == "rqvae")]) for c in ("industrial", "office")],
        "not in the draft; the old maps show the same")
    def e1_fail(df, col, cat=None):
        bad = []
        sub = df if cat is None else df[df.catalog == cat]
        for name, g in sub.groupby("variant"):
            if not name.startswith("rqvae"):
                continue
            g = g.sort_values("digit")
            if not g[col].iloc[0] > g[col].iloc[1:].max():
                bad.append(name.replace("rqvae_", "").replace("codebook_", "×"))
        return f"{len(bad)}/9: " + ", ".join(bad) if bad else "0/9"
    add("e1_rule", "(check) RQ-VAE maps in which a later digit alone explains more variance than digit 1 (expectation E1)",
        "expectation E1; RQ1 ¶2", e1_fail(od, "standalone_between_share"),
        *[e1_fail(dg, "var_alone", c) for c in ("industrial", "office")],
        "E1 was miscalibrated: the old maps fail it too")
    add("mq_digits_alone", "Every MQ digit explains 36–38 % on its own (4×128)", "05results RQ1 ¶2; fig:map-digits(b)",
        prng(od[od.variant == v("MQ", 4, 128)].standalone_between_share, 0),
        *[prng(dg[(dg.catalog == c) & (dg.variant == v("MQ", 4, 128))].var_alone, 0) for c in ("industrial", "office")],
        "holds")
    add("refinement_5x128", "Five MQ digits reach a refinement index of 0.80 and five RQ-KMeans digits 0.95 (5×128)",
        "05results RQ1 ¶2; fig:map-identity(d)",
        f"{float(oref[(oref.variant == v('MQ', 5, 128)) & (oref.depth == 5)].refinement_index.iloc[0]):.2f} / "
        f"{float(oref[(oref.variant == v('rqkmeans', 5, 128)) & (oref.depth == 5)].refinement_index.iloc[0]):.2f}",
        *[f"{v3p(c, v('MQ', 5, 128), 5, 'refinement_index'):.2f} / {v3p(c, v('rqkmeans', 5, 128), 5, 'refinement_index'):.2f}"
          for c in ("industrial", "office")], "holds")
    op3 = lambda name: float(opur[(opur.variant == name) & (opur.field == "cat_l1") & (opur.depth == 3)].purity.iloc[0])  # noqa: E731
    on3 = lambda name: float(opur[(opur.variant == name) & (opur.field == "cat_l1") & (opur.depth == 3)].null_mean.iloc[0])  # noqa: E731
    add("purity3", "Items sharing three digits share their group's main category 97–99 % (residual), 81 % (MQ); "
                   "shuffled 39–50 % (4×128)", "05results RQ1 ¶1; intro finding 1 (81–99 %); fig:map-digits(c); plan 5.1 P3",
        f"{pct(op3(v('rqkmeans', 4, 128)))} / {pct(op3(v('rqvae', 4, 128)))} / {pct(op3(v('MQ', 4, 128)))}; null "
        f"{prng([on3(v(q, 4, 128)) for q in QO], 0)}",
        *[f"{pct(v3p(c, v('rqkmeans', 4, 128), 3, PF[c] + '_purity'))} / {pct(v3p(c, v('rqvae', 4, 128), 3, PF[c] + '_purity'))} / "
          f"{pct(v3p(c, v('MQ', 4, 128), 3, PF[c] + '_purity'))}; null "
          f"{prng([v3p(c, v(q, 4, 128), 3, PF[c] + '_null_mean') for q in QO], 0)}" for c in ("industrial", "office")],
        "holds; intro range becomes 79–99 %")
    u = od[(od.quantizer == "rqvae") & (od.digit == 1)].utilization
    add("rqvae_first_codebook", "RQ-VAE uses only 2–31 % of its first codebook in eight of nine maps",
        "05results RQ1 ¶3; discussion table (Quantizer); plan 5.1 P4",
        f"{prng(u, 0)}; {int((u <= 0.31).sum())}/9 at ≤ 31 %",
        *[(lambda s: f"{prng(s, 0)}; {int((s <= 0.5).sum())}/9 at ≤ 50 %")(
            dg[(dg.catalog == c) & (dg.quantizer == "rqvae") & (dg.digit == 1)].utilization) for c in ("industrial", "office")],
        "update the range")
    oalone = float(oref[(oref.variant == v("rqvae", 4, 128)) & (oref.depth == 4)].frac_items_in_singletons.iloc[0])
    add("rqvae_alone4", "After four digits 99 % of RQ-VAE items have a SID of their own (4×128)", "05results RQ1 ¶3",
        pct(oalone), *[pct(v3p(c, v("rqvae", 4, 128), 4, "frac_items_in_singletons")) for c in ("industrial", "office")], "holds")
    # The draft's 92.5 % is the catalogue mask-support number; recompute it on
    # the frozen table and refuse to go on unless it reproduces the archived
    # 92.54 % (archive_catalogue_mask_support_README.md).
    asins = [k for k in _v3_keys()]
    frozen = SidTable.load(v("rqvae", 4, 128))
    old_det = sid_determined_by_d_minus_1(np.array([frozen.asin2codes[k] for k in asins]))
    if round(100 * old_det, 2) != 92.54:
        raise AssertionError(f"mask-support statistic does not reproduce the archive: {100 * old_det:.2f} % vs 92.54 %")
    det = {c: sid_determined_by_d_minus_1(load_table(c, v("rqvae", 4, 128)).codes) for c in ("industrial", "office")}
    add("rqvae_three_of_four", "For 92.5 % of RQ-VAE 4×128 items, any three of their four digits determine the full SID "
                               "(the paper plan's 'unique 3-digit prefix' wording is a different statistic)",
        "plan 5.1 P5; old results.tex §6.2 ¶5", pct(old_det, 2), *[pct(det[c], 2) for c in ("industrial", "office")],
        "update; fix the plan's wording")
    ouniq = float(oref[(oref.variant == v("rqvae", 4, 128)) & (oref.depth == 3)].frac_items_in_singletons.iloc[0])
    add("rqvae_unique_prefix3", "(for reference) RQ-VAE 4×128 items alone at prefix length 3", "plan 5.1 P5 as worded",
        pct(ouniq), *[pct(v3p(c, v("rqvae", 4, 128), 3, "frac_items_in_singletons")) for c in ("industrial", "office")],
        "not the draft's statistic")
    oc = lambda name: float(ocol[ocol.variant == name].items_in_collision_groups_pct.iloc[0]) / 100  # noqa: E731
    vc = lambda c, name: float(B[(B.catalog == c) & (B.table == name)].collision_rate.iloc[0])  # noqa: E731
    add("collision_3x128", "44.8 % (RQ-KMeans) and 54.5 % (MQ) of items share a SID at 3×128",
        "05results RQ1 ¶3; fig:map-identity(c)", f"{pct(oc(v('rqkmeans', 3, 128)))} / {pct(oc(v('MQ', 3, 128)))}",
        *[f"{pct(vc(c, v('rqkmeans', 3, 128)))} / {pct(vc(c, v('MQ', 3, 128)))}" for c in ("industrial", "office")],
        "update the numbers")
    add("collision_ranges", "(table) collision share, 9 maps per family: RQ-KMeans / RQ-VAE / MQ", "fig:map-identity(c)",
        " / ".join(prng(ocol[ocol.quantizer == q].items_in_collision_groups_pct / 100) for q in QO),
        *[" / ".join(prng(B[(B.catalog == c) & (B.quantizer == q)].collision_rate) for q in QO) for c in ("industrial", "office")],
        "RQ-KMeans 5×512 moves from 0.6 % to 6.3 %; the rest similar")
    ofl = e3["partA_catalogue"]
    add("identical_floor", "Eleven pairs of items have identical embeddings, and every map except one gives each pair a single SID",
        "05results RQ1 ¶4; tab:collision-causes (All)",
        f"{ofl['n_identical_input_items']} items in {ofl['n_identical_input_groups']} groups; "
        f"{int((oB.identical_pairs_split == 0).sum())}/27 maps split none",
        *[f"{A[c]['n_identical_input_items']} items in {A[c]['n_identical_input_groups']} groups; "
          f"{int((B[B.catalog == c].identical_pairs_split == 0).sum())}/27 maps split none" for c in ("industrial", "office")],
        "drop 'except one'")
    nd = lambda tb, names: ", ".join(f"{n.split('_')[1][0]}×{n.split('_')[-1]} {100 * float(tb[tb.table == n].nd_share_collided.iloc[0]):.0f} %" for n in names)  # noqa: E731
    deep_old = [v("rqkmeans", 5, 128), v("rqkmeans", 5, 256), v("rqkmeans", 4, 512)]
    deep_v3 = [v("rqkmeans", 5, 128), v("rqkmeans", 5, 256), v("rqkmeans", 5, 512)]
    add("near_dup", "73–85 % of collided items at the deepest nested RQ-KMeans fits have a same-brand, near-identical bucket-mate",
        "05results RQ1 ¶4; tab:collision-causes (All); intro finding 1; plan 5.2 P7", nd(oB, deep_old),
        *[nd(B[B.catalog == c], deep_v3) for c in ("industrial", "office")], "update; 5×512 is now nested and counts")
    od1 = oD[oD.table.str.startswith("rqkmeans")]
    add("loop_rqkmeans", "Applying RQ-VAE's reassignment to RQ-KMeans takes every table from 10–45 % to the 0.71 % floor, "
                         "changing the last digit of up to 44 % of items", "05results RQ1 ¶5; tab:collision-causes (RQ-VAE); plan 5.2 P6",
        f"{prng(od1.rate_before)} → {prng(od1.rate_after, 2)}; up to {pct(od1.share_last_digit_changed.max(), 0)}",
        *[(lambda g: f"{prng(g.rate_start)} → {prng(g.rate_after_loop, 2)}; up to {pct(g.share_last_digit_changed.max(), 0)}")(
            D[(D.catalog == c) & (D.quantizer == "rqkmeans")]) for c in ("industrial", "office")],
        "holds on Industrial; on Office RQ-KMeans 3×128 stays at 3.21 %")
    add("rqvae_direct", "(new) RQ-VAE before its reassignment → published", "strengthens RQ1 ¶5 and plan 5.2 P6",
        "not measurable (weights not saved)",
        *[(lambda g: f"{prng(g.rate_start)} → {prng(g.rate_published, 2)}")(D[(D.catalog == c) & (D.quantizer == "rqvae")])
          for c in ("industrial", "office")], "new")
    oh = oH[oH.level >= 1]
    add("hub", "At every level after the first, the code nearest the origin holds 9–17 % of all items",
        "05results RQ1 ¶5; tab:collision-causes (RQ-KMeans)",
        f"{prng(oh.hub_share, 0)}; rank 0 at {int((oh.hub_norm_rank_from_smallest == 0).sum())}/{len(oh)} levels",
        *[(lambda h: f"{prng(h.hub_share, 0)}; rank 0 at {int((h.hub_norm_rank_from_smallest == 0).sum())}/{len(h)} levels")(
            H[(H.catalog == c) & (H.level >= 1)]) for c in ("industrial", "office")], "update the range")
    add("radius_spearman", "Parents with a smaller radius collide more (Spearman −0.41 to −0.68)", "tab:collision-causes (RQ-KMeans)",
        rng(oB[oB.quantizer == "rqkmeans"].radius_spearman),
        *[rng(B[(B.catalog == c) & (B.quantizer == "rqkmeans")].radius_spearman) for c in ("industrial", "office")],
        "holds on Industrial; weaker on Office")
    add("mq_nmi", "MQ digits are strongly dependent (normalized mutual information 0.50–0.70)",
        "05results RQ1 ¶5; tab:collision-causes (MQ); appendix (Redundant MQ digits); plan 5.2 P8",
        f"MQ {rng(oB[oB.quantizer == 'MQ'].mean_pairwise_nmi)}; RQ-KMeans {rng(oB[oB.quantizer == 'rqkmeans'].mean_pairwise_nmi)}",
        *[f"MQ {rng(B[(B.catalog == c) & (B.quantizer == 'MQ')].mean_pairwise_nmi)}; RQ-KMeans "
          f"{rng(B[(B.catalog == c) & (B.quantizer == 'rqkmeans')].mean_pairwise_nmi)}" for c in ("industrial", "office")],
        "NMI overlaps between families; switch to AMI")
    om = maps.groupby("quantizer").mean_pair_ami_old.agg(["min", "max"])
    add("mq_ami", "(replacement) mean pairwise AMI between digits, per map", "replaces the NMI in RQ1 ¶5 and the table",
        f"MQ {om.loc['MQ', 'min']:.2f}–{om.loc['MQ', 'max']:.2f}; residual "
        f"{min(om.loc['rqkmeans', 'min'], om.loc['rqvae', 'min']):.2f}–{max(om.loc['rqkmeans', 'max'], om.loc['rqvae', 'max']):.2f}",
        *[(lambda s: f"MQ {s['MQ'].min():.2f}–{s['MQ'].max():.2f}; residual "
                     f"{min(s['rqkmeans'].min(), s['rqvae'].min()):.2f}–{max(s['rqkmeans'].max(), s['rqvae'].max()):.2f}")(
            pa[pa.catalog == c].groupby(["quantizer", "variant"]).ami.mean()) for c in ("industrial", "office")],
        "holds; use it")
    r3o, = oB[oB.table == v("MQ", 3, 128)].itertuples()
    add("mq_n1", "With independent digits, MQ 3×128 would collide 7.4 % of items instead of 54.5 %",
        "05results RQ1 ¶5; tab:collision-causes (MQ)", f"{pct(r3o.collision_rate)} vs {pct(r3o.n1_rate_p50)}",
        *[(lambda r: f"{pct(r.collision_rate)} vs {pct(r.n1_rate_p50)}")(
            B[(B.catalog == c) & (B.table == v("MQ", 3, 128))].iloc[0]) for c in ("industrial", "office")], "update")
    r5o = oB[oB.table == v("rqkmeans", 5, 512)].iloc[0]
    add("rk_5x512", "RQ-KMeans 5×512 is a Sinkhorn-balanced table (0.64 %)",
        "05results RQ1 ¶5; tab:collision-causes; discussion table (Post-processing)",
        f"{pct(r5o.collision_rate, 2)}; separate fit; identical pairs split {int(r5o.identical_pairs_split)}",
        *[(lambda r: f"{pct(r.collision_rate, 2)}; nested; identical pairs split {int(r.identical_pairs_split)}")(
            B[(B.catalog == c) & (B.table == v("rqkmeans", 5, 512))].iloc[0]) for c in ("industrial", "office")],
        "drop: v3 has no balanced table")
    add("capacity", "Uniform codes would give at most 0.15 % of items a shared SID at 3×128", "tab:collision-causes (Not causes); setup",
        pct(float(oB[oB.table == v("rqkmeans", 3, 128)].n0_rate.iloc[0]), 2),
        *[pct(float(B[(B.catalog == c) & (B.table == v("rqkmeans", 3, 128))].n0_rate.iloc[0]), 2) for c in ("industrial", "office")],
        "holds; add Office")
    ocs = oC[(oC.condition == "per-level seed") & (oC.width == 128) & (oC.depth == 3)].iloc[0]
    add("per_level_seed", "The reused k-means seed lowers collisions (+13.0 points at 3×128 without it)",
        "tab:collision-causes (Not causes); RQ1 ¶5",
        f"{f'{100 * ocs.paired_diff_mean:+.1f}'.replace('-', '−')} points without it ({overlap_words(ocs.ranges_disjoint)})",
        *[(lambda r: f"{f'{100 * r.paired_diff_mean:+.1f}'.replace('-', '−')} points without it ({overlap_words(r.ranges_disjoint)})")(
            C[(C.catalog == c) & (C.condition == "per-level seed") & (C.width == 128) & (C.depth == 3)].iloc[0])
          for c in ("industrial", "office")], "holds on Industrial only (refits on this cluster)")
    zr = lambda c: Z[(Z.catalog == c) & (Z.variant == v("rqkmeans", 3, 128))].iloc[0]  # noqa: E731
    add("zcr_3x128", "ZCR gives every RQ-KMeans 3×128 item its own SID by changing the last digit of 910 items (29 %)",
        "05results RQ1 ¶6", f"{e3['zcr_3x128']['items_changed']} ({pct(e3['zcr_3x128']['items_changed'] / e3['zcr_3x128']['items'], 0)})",
        *[f"{int(round(zr(c).changed_tail_share * A[c]['n_items']))} ({pct(zr(c).changed_tail_share, 0)}); "
          f"+{100 * zr(c).sq_dist_rel_increase:.1f} % squared distance" for c in ("industrial", "office")], "update")
    pd.DataFrame(claims).to_csv(out / "claims.csv", index=False)

    # -------------------------------------------------------------- figure --
    st.apply()
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15))
    plt.subplots_adjust(wspace=0.55)
    t = []
    a = axes[0]
    for _, r in maps.iterrows():
        a.plot(100 * r.collision_old, 100 * r.collision_v3, WIDTH_MARKER[r.width], color=QC[r.quantizer], ms=3.6)
        t.append({"panel": "a", "variant": r.variant, "old": r.collision_old, "v3": r.collision_v3})
    a.set_xscale("log"); a.set_yscale("log")
    lim = (0.3, 80)
    a.plot(lim, lim, color=st.MUTED, lw=0.8, ls=":")
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30"]); a.set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
    r5 = maps[maps.variant == v("rqkmeans", 5, 512)].iloc[0]
    a.annotate("RQ-KMeans\n5×512", (100 * r5.collision_old, 100 * r5.collision_v3), xytext=(6, 2),
               textcoords="offset points", fontsize=5.6, color=st.INK_2)
    a.set_xlabel("Old map (%)"); a.set_ylabel("v3 map (%)")
    a.set_title("Items sharing a SID", fontsize=7, loc="left")
    a = axes[1]
    for q in QO:
        s = digits[digits.variant == v(q, 4, 128)].sort_values("digit")
        a.plot(s.digit, 100 * s.var_alone_old, ":", marker="o", mfc=st.SURFACE, color=QC[q], lw=1.0, ms=3.2)
        a.plot(s.digit, 100 * s.var_alone_v3, "-o", color=QC[q], ms=3.2)
        for _, r in s.iterrows():
            t.append({"panel": "b", "variant": r.variant, "digit": r.digit, "old": r.var_alone_old, "v3": r.var_alone_v3})
    a.set_xticks([1, 2, 3, 4]); a.set_ylim(0, 60)
    a.set_xlabel("Digit (alone), 4×128"); a.set_ylabel("Variance explained (%)")
    a.set_title("One digit alone", fontsize=7, loc="left")
    a = axes[2]
    for _, r in maps.iterrows():
        a.plot(100 * r.purity3_old, 100 * r.purity3_v3, WIDTH_MARKER[r.width], color=QC[r.quantizer], ms=3.6)
        t.append({"panel": "c", "variant": r.variant, "old": r.purity3_old, "v3": r.purity3_v3})
    a.plot((60, 101), (60, 101), color=st.MUTED, lw=0.8, ls=":")
    a.set_xlim(60, 101); a.set_ylim(60, 101)
    a.set_xlabel("Old map (%)"); a.set_ylabel("v3 map (%)")
    a.set_title("Purity, 3 shared digits", fontsize=7, loc="left")
    a = axes[3]
    for _, r in maps.iterrows():
        a.plot(r.digit1_codes_old / r.width * 100, r.digit1_codes_v3 / r.width * 100, WIDTH_MARKER[r.width],
               color=QC[r.quantizer], ms=3.6)
        t.append({"panel": "d", "variant": r.variant, "old": r.digit1_codes_old / r.width, "v3": r.digit1_codes_v3 / r.width})
    a.plot((0, 102), (0, 102), color=st.MUTED, lw=0.8, ls=":")
    a.set_xlim(0, 102); a.set_ylim(0, 102)
    a.set_xlabel("Old map (%)"); a.set_ylabel("v3 map (%)")
    a.set_title("Digit-1 codes in use", fontsize=7, loc="left")
    for ax_, l in zip(axes, "abcd"):
        ax_.text(-0.02, 1.13, f"({l})", transform=ax_.transAxes, ha="right", fontsize=8, fontweight="bold")
    h = [Line2D([], [], color=QC[q], marker="o", ls="", label=QL[q]) for q in QO]
    h += [Line2D([], [], marker=m, ls="", color=st.MUTED, label=f"$W$={w}") for w, m in WIDTH_MARKER.items()]
    h += [Line2D([], [], color=st.INK_2, label="v3"), Line2D([], [], color=st.INK_2, ls=":", label="old (b)")]
    fig.legend(handles=h, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.13), columnspacing=0.8)
    st.save(fig, out / "fig_v3_vs_old")
    pd.DataFrame(t).to_csv(out / "fig_v3_vs_old.csv", index=False)
    plt.close(fig)
    print("wrote", out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--old-exp3", type=Path, required=True, help="structure exp3 result dir (job 293555)")
    ap.add_argument("--old-zcr", type=Path, required=True, help="derived/substrate/zcr_inputs/20261004")
    args = ap.parse_args(argv)
    build(args.run, args.old_exp3, args.old_zcr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
