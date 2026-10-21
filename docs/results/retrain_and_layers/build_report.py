#!/usr/bin/env python
"""Build the review page for the 2026-10-06 results: retrain acceptance, ZCR scores,
the layer-by-layer information map, and the retrained-model copy grid (exp11).

    python docs/results/retrain_and_layers/build_report.py [--grid-summary <exp11 summary dir>]

Reads only result files (no model, no GPU); writes `sidlens_retrain_layers.html`
beside this file from `report_template.html`. Without --grid-summary the grid
section shows the original checkpoints' reference values and the pilot checks,
and says the full run is pending. Every number on the page comes from the files
listed in the page's "Sources" section; their sha256 is recorded there.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/controlled/exp11_ar_copy_grid"))

from sidlens import paths                                                # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402

CHECK = paths.DERIVED / "substrate/exp1_ar_next_item_retrain/check-r20261004-20261006-zcr"
INFOMAP = paths.DERIVED / "representation/exp3_ar_information_map/summary-294711-294712"
PILOT = paths.DERIVED / "controlled/exp11_ar_copy_grid/pilot-300856"
MODELS = {"rqkmeans_3codebook_128": "RQ-KMeans 3×128", "rqvae_4codebook_128": "RQ-VAE 4×128"}


def parse(v: str) -> dict:
    m = re.match(r"(rqkmeans|rqvae|MQ)_(\d)codebook_(\d+)", v)
    return {"q": m[1], "depth": int(m[2]), "width": int(m[3])}


def label(v: str) -> str:
    p = parse(v)
    return f"{dict(rqkmeans='RQ-KMeans', rqvae='RQ-VAE', MQ='MQ')[p['q']]} {p['depth']}×{p['width']}"


def r3(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def acceptance() -> tuple[list, list, float]:
    d = pd.read_csv(CHECK / "estimates.csv")
    acc, zcr = [], []
    nat = d[~d.variant.str.endswith("__zcr")].set_index("variant")
    for v, r in nat.iterrows():
        acc.append({"variant": v, "label": label(v), **parse(v), "hr_new": r3(100 * r.hr10_new, 2),
                    "hr_old": r3(100 * r.hr10_old, 2), "d": r3(100 * r.d_hr10, 2), "lo": r3(100 * r.d_hr10_lo, 2),
                    "hi": r3(100 * r.d_hr10_hi, 2), "d_ndcg": r3(100 * r.d_ndcg10, 2),
                    "top1": r3(r.top1_agree, 3), "accepted": bool(r.accepted),
                    "train_min": r3(r.train_seconds / 60, 1)})
    for v in d.variant[d.variant.str.endswith("__zcr")]:
        n = v.removesuffix("__zcr")
        z = d.set_index("variant").loc[v]
        zcr.append({"variant": n, "label": label(n), "native_hr": r3(100 * nat.loc[n, "hr10_new"], 2),
                    "native_strict": r3(100 * nat.loc[n, "strict_hr10_new"], 2),
                    "native_accepted": bool(nat.loc[n, "accepted"]), "zcr_hr": r3(100 * z.hr10_new, 2),
                    "zcr_strict": r3(100 * z.strict_hr10_new, 2), "zcr_ndcg": r3(100 * z.ndcg10_new, 2),
                    "native_ndcg": r3(100 * nat.loc[n, "ndcg10_new"], 2)})
    rho = pd.Series([a["hr_new"] for a in acc]).rank().corr(pd.Series([a["hr_old"] for a in acc]).rank())
    return acc, zcr, round(float(rho), 2)


def infomap() -> dict:
    e = pd.read_csv(INFOMAP / "estimates.csv")
    out = {"lens": {}, "decoders": {}, "d4": [], "census": {}, "headline": {}, "digits": {}}
    pts = lambda g: [{"x": int(r.layer), "y": r3(r.est), "lo": r3(r.lo), "hi": r3(r.hi)}       # noqa: E731
                     for r in g.assign(L=g.layer.astype(int)).sort_values("L").itertuples()]
    for m in MODELS:
        g = e[e.variant == m]
        d2 = g[g.estimand == "D2 whole SID"]
        out["lens"][m] = {lens: pts(d2[d2.lens == lens]) for lens in ("tuned", "logit")}
        d1 = g[(g.estimand == "D1 top1 = answer") & (g.lens == "tuned")]
        out["digits"][m] = {int(dd): pts(x) for dd, x in d1.groupby("d")}
        l2 = g[g.part == "L2"]
        dec = {}
        for task, x in l2.groupby("task"):
            num = x[x.layer.astype(str).str.isdigit()]
            item = {"exact": pts(num[num.estimand == "exact"])}
            if task == "answer_at_R0":
                item["noncopy"] = pts(num[num.estimand == "exact | non-copy answer"])
            pr = x[(x.layer == "prior") & (x.estimand == "exact")]
            item["prior"] = r3(pr.est.iloc[0]) if len(pr) else None
            rl = x[(x.layer == "rule") & (x.estimand.str.startswith("exact "))]
            item["rule"] = r3(rl.est.iloc[0]) if len(rl) else None
            item["rule_name"] = rl.estimand.iloc[0].removeprefix("exact ") if len(rl) else None
            dec[task] = item
        out["decoders"][m] = dec
        for r in g[g.estimand.str.startswith("D4")].itertuples():
            out["d4"].append({"model": m, "subset": r.estimand.split(", ")[1], "median": r3(r.est, 1),
                              "q1": r3(r.lo, 1), "q3": r3(r.hi, 1), "n": int(r.n_rows), "never": r3(r.never, 3)})
        c = g[g.part == "C"].set_index("estimand")
        out["headline"][m] = {"greedy_golden": r3(c.loc["greedy = golden", "est"], 3),
                              "greedy_beam": r3(c.loc["greedy = plain-beam top-1", "est"], 3)}
    cs = pd.read_csv(INFOMAP / "census_summary.csv")
    vin = json.loads((INFOMAP / "verdict_inputs.json").read_text())
    for m in MODELS:
        g = cs[(cs.variant == m) & (cs.group == "R_0")]
        sh = g.groupby(["layer", "kind"]).share_latents.sum().unstack(fill_value=0)
        nl = g.groupby("layer").n_live.first()
        beyond = {b["layer"]: b for b in vin[m]["census_answer_beyond_copy_R0"]}
        out["census"][m] = [{"x": int(L), "copy": r3(sh.loc[L].get("copy source", 0), 3),
                             "answer": r3(sh.loc[L].get("own answer", 0), 3),
                             "unexplained": r3(sh.loc[L].get("unexplained", 0), 3), "n_live": int(nl.loc[L]),
                             "n_answer": int(beyond[L]["n_answer_latents"]),
                             "n_beyond": int(beyond[L]["n_beyond_copy"])} for L in sh.index]
        h = cs[(cs.variant == m) & (cs.group == "H_0")]
        sh0 = h.groupby(["layer", "kind"]).share_latents.sum().unstack(fill_value=0)
        out["census"][m + "|H_0"] = [{"x": int(L), "token": r3(sh0.loc[L].get("token", 0), 3),
                                      "prev": r3(sh0.loc[L].get("previous item", 0), 3)} for L in sh0.index]
    return out


GRID_KEYS = {"A1": ("exp3_A", "A1 delta digit 0 (r=1, m=0)", 1),
             "A3": ("exp3_A", "A3 retained fraction digit 0 (m=1)", 1),
             "H1": ("exp4_valid", "H1 match - nonmatch (pooled d>=1)", 1),
             "H2a": ("exp4_valid", "H2a K1 | match (pooled d>=1)", 1),
             "H2b": ("exp4_valid", "H2b K1 - K3 | match (paired, pooled d>=1)", 1),
             "late": ("exp4_valid", "late share K5 / K1 | match (pooled d>=1)", 1),
             "P1": ("exp6", "P1 C_all - B | repeat (HR@10)", 100),
             "P2": ("exp6", "P2 C_all - B | new (HR@10)", 100),
             "N": ("exp6", "N_later - B | all (HR@10)", 100),
             "plain": ("exp6", "C_all_plain - B_plain | all (HR@10)", 100)}


def grid_rows(rows: pd.DataFrame) -> list[dict]:
    out = []
    for model, g in rows.groupby("model", sort=False):
        v = g.variant.iloc[0]
        rec = {"model": model, "variant": v, "label": label(v), **parse(v),
               "original": not str(model).startswith("retrain/")}
        for k, (st, nm, sc) in GRID_KEYS.items():
            x = g[(g.stage == st) & (g.estimand == nm)]
            rec[k] = None if x.empty else {"est": r3(sc * x.est.iloc[0]), "lo": r3(sc * x.lo.iloc[0]),
                                           "hi": r3(sc * x.hi.iloc[0])}
        b = g[(g.stage == "exp6") & (g.estimand == "level hr10") & (g.condition == "B")]
        rec["hr10"] = None if b.empty else r3(100 * b.est.iloc[0], 2)
        out.append(rec)
    return out


def grid(summary: Path | None) -> dict:
    pilot = []
    for cell in sorted(PILOT.glob("cell-*")):
        v = re.search(r"variant=(\S+)", (cell / "status.txt").read_text())[1]
        k = json.loads((cell / "exp4_valid/validation.json").read_text())["controls"]
        v1 = json.loads((cell / "exp6/validation.json").read_text())["V1_reproduction"]
        pilot.append({"label": label(v), "rows": v1["rows"],
                      "k0_exact": sum(c["n_exact"] for c in k.values()), "k0_n": sum(c["n"] for c in k.values()),
                      "v1": v1["identical_50"]})
    if summary is not None:
        est = pd.read_csv(summary / "estimates.csv")
        res = json.loads((summary / "result.json").read_text())
        return {"state": "complete", "rows": grid_rows(est), "g1": res["g1"], "g2": res["g2"],
                "g3": res["g3_spearman_vs_hr10"], "pilot": pilot, "summary": str(summary)}
    import summarize as S                                           # exp11's own estimators
    from sidlens.analysis.bootstrap import UserBootstrap
    from sidlens.data import ar_prompts as P
    boots = {s: UserBootstrap([e.user_id for e in P.load_examples("rqkmeans_3codebook_128", "next-item", s)],
                              S.DRAWS, S.SEED) for s in ("test", "valid")}
    rows = []
    for v, o in S.ORIGINAL.items():
        c = S.Cell(v, o["label"], boots)
        c.exp3(o["exp3_A"])
        c.exp4(o["exp4_valid"], "exp4_valid")
        c.exp6(o["exp6"], None)
        rows += c.rows
    return {"state": "pending", "rows": grid_rows(pd.DataFrame(rows)), "pilot": pilot}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid-summary", type=Path, default=None)
    ap.add_argument("--grid-job", default="301495")
    ap.add_argument("--out", type=Path, default=HERE.parent / "sidlens_retrain_layers.html")
    a = ap.parse_args()
    acc, zcr, rho = acceptance()
    sources = [CHECK / "estimates.csv", INFOMAP / "estimates.csv", INFOMAP / "census_summary.csv",
               INFOMAP / "verdict_inputs.json"]
    if a.grid_summary:
        sources += [a.grid_summary / "estimates.csv", a.grid_summary / "result.json"]
    data = {"generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "grid_job": a.grid_job,
            "acceptance": acc, "acceptance_rho": rho, "zcr": zcr, "models": MODELS, "info": infomap(),
            "grid": grid(a.grid_summary),
            "sources": [{"path": str(p.relative_to(paths.WORK)), "sha256": sha256_file(p)[:16]} for p in sources]}
    html = (HERE.parent / "report_template.html").read_text()
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = html.replace("/*__DATA__*/null", blob)
    a.out.write_text(page)
    # The artifact viewer wraps the published page in a document skeleton; the
    # standalone copy carries the same skeleton so it renders identically when
    # opened from disk (charset for ×/–, [hidden] for tooltips, viewport).
    standalone = a.out.with_name(a.out.stem + "_standalone.html")
    standalone.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0;font:14px system-ui,-apple-system,'
        '"Segoe UI",sans-serif;background:#fafafa}img{max-width:100%}[hidden]{display:none!important}</style>\n'
        '</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    print(f"wrote {a.out} ({a.out.stat().st_size / 1024:.0f} KB) and {standalone.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
