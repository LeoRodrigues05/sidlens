#!/usr/bin/env python
"""Estimates for representation/exp2 (SAEs): fidelity, copy and prefix-match ablations, probes.

Merges the three stages per cell (train -> fidelity.json, analyze ->
observational.csv, causal -> records.parquet). The causal estimands (F2, CC,
CM) are bootstrapped here, with one paired user bootstrap shared by both cells.

    python summarize.py --train <train group> --analysis <analyze group> --causal <causal group> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402

SEED, DRAWS = 20260930, 2000
LAYERS = (12, 16, 20, 24)


def causal_estimands(rec: pd.DataFrame, boot: UserBootstrap, variant: str) -> list[dict]:
    out = []
    tag = {"variant": variant}
    cl = rec[rec.part == "clean"].set_index(["row", "digit"])
    rec = rec[rec.part != "clean"].join(cl[["logp_codes", "copy_score", "top1"]].add_prefix("clean_"),
                                        on=["row", "digit"])
    rec["delta"] = rec.clean_logp_codes - rec.logp_codes
    # F2 recovered fraction per layer and digit
    f2 = rec[rec.part == "F2"]
    for (L, d), g in f2.groupby(["layer", "digit"]):
        s = g[g.condition == "sae"].set_index("row")
        m = g[g.condition == "mean"].set_index("row").loc[s.index]
        r = boot.ratio(s.user_id, s.delta, m.delta)
        out.append({**tag, "part": "F2", "layer": L, "digit": d, "estimand": "recovered fraction 1 - dSAE/dMEAN",
                    **r, "est": 1 - r["est"], "lo": 1 - r["hi"], "hi": 1 - r["lo"]})
        out.append({**tag, "part": "F2", "layer": L, "digit": d, "estimand": "delta splice SAE",
                    **boot.mean(s.user_id, s.delta)})
        out.append({**tag, "part": "F2", "layer": L, "digit": d, "estimand": "delta splice mean",
                    **boot.mean(m.user_id, m.delta)})
    # CC copy latents at readout(0)
    cc = rec[rec.part == "CC"]
    for L, g in cc.groupby("layer"):
        a = g[g.condition == "copy"].set_index("row")
        b = g[g.condition == "control"].set_index("row").loc[a.index]
        dcopy_a = a.clean_copy_score - a.copy_score
        dcopy_b = b.clean_copy_score - b.copy_score
        out.append({**tag, "part": "CC", "layer": L, "estimand": "CC1 dCopyScore(copy) - dCopyScore(control)",
                    **boot.mean(a.user_id, dcopy_a - dcopy_b)})
        out.append({**tag, "part": "CC", "layer": L, "estimand": "dCopyScore copy latents", **boot.mean(a.user_id, dcopy_a)})
        out.append({**tag, "part": "CC", "layer": L, "estimand": "dCopyScore control", **boot.mean(b.user_id, dcopy_b)})
        out.append({**tag, "part": "CC", "layer": L, "estimand": "dGolden d0 copy latents", **boot.mean(a.user_id, a.delta)})
        out.append({**tag, "part": "CC", "layer": L, "estimand": "dGolden d0 control", **boot.mean(b.user_id, b.delta)})
        c1v = a["c1"]
        cr_a = (a.top1 == c1v).astype(float)
        cr_b = (b.top1 == c1v).astype(float)
        cr_0 = (a.clean_top1 == c1v).astype(float)
        out.append({**tag, "part": "CC", "layer": L, "estimand": "copy rate clean", **boot.mean(a.user_id, cr_0)})
        out.append({**tag, "part": "CC", "layer": L, "estimand": "CC2 copy rate(copy) - copy rate(control)",
                    **boot.mean(a.user_id, cr_a - cr_b)})
        out.append({**tag, "part": "CC", "layer": L, "estimand": "latents removed per row (copy)",
                    **boot.mean(a.user_id, a.n_active_removed.astype(float))})
    # CM prefix-match latents at readout(d)
    cm = rec[rec.part == "CM"]
    for L, g in cm.groupby("layer"):
        s = g[g.condition == "S"]
        sm, sn = s[s.match], s[~s.match]
        out.append({**tag, "part": "CM", "layer": L, "estimand": "CM1 dS(match) - dS(nonmatch)",
                    **boot.paired_difference((sm.user_id, sm.delta), (sn.user_id, sn.delta))})
        out.append({**tag, "part": "CM", "layer": L, "estimand": "dS | match", **boot.mean(sm.user_id, sm.delta)})
        out.append({**tag, "part": "CM", "layer": L, "estimand": "dS | nonmatch", **boot.mean(sn.user_id, sn.delta)})
        ctl = g[g.condition.str.startswith("C")].groupby(["row", "d"]).delta.mean().rename("ctrl")
        j = sm.join(ctl, on=["row", "d"])
        out.append({**tag, "part": "CM", "layer": L, "estimand": "CM2 dS(match) - mean dControl(match)",
                    **boot.mean(j.user_id, j.delta - j.ctrl)})
        out.append({**tag, "part": "CM", "layer": L, "estimand": "dControl | match", **boot.mean(j.user_id, j.ctrl)})
        out.append({**tag, "part": "CM", "layer": L, "estimand": "share of match pairs with an S latent active",
                    **boot.mean(sm.user_id, (sm.n_active_removed > 0).astype(float))})
    return out


def figures(res: pd.DataFrame, obs: pd.DataFrame, fid: dict, out: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from sidlens.viz import style
    style.apply()
    made = []
    col = lambda v: style.QUANTIZER_COLOR["rqvae" if "rqvae" in v else "rqkmeans"]     # noqa: E731
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.8))
    for ax, (part, name, ttl) in zip(axes, [
            ("CC", "CC1 dCopyScore(copy) - dCopyScore(control)", "CC1: copy latents, readout(0)"),
            ("CM", "CM1 dS(match) - dS(nonmatch)", "CM1: match latents, readout(d)"),
            ("CM", "CM2 dS(match) - mean dControl(match)", "CM2: vs matched controls"),
            ("F2", "recovered fraction 1 - dSAE/dMEAN", "F2: splice recovery, digit 0")]):
        for k, v in enumerate(res.variant.unique()):
            q = res[(res.variant == v) & (res.part == part) & (res.estimand == name)]
            if q.empty:                                   # e.g. an amendment run with CM only
                continue
            if part == "F2":
                q = q[q.digit == 0]
            q = q.sort_values("layer")
            ax.errorbar(q.layer + (k - 0.5) * 0.6, q.est, yerr=[q.est - q.lo, q.hi - q.est], fmt="o-",
                        color=col(v), ms=style.MARKER, capsize=0, label=v)
        style.null_line(ax, 0, "")
        ax.set_xticks(LAYERS)
        ax.set_xlabel("layer")
        ax.set_title(ttl, fontsize=7)
    axes[0].set_ylabel("nats (or fraction)")
    style.legend_outside(axes[-1])
    made += [str(p) for p in style.save(fig, out / "fig1_causal")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for ax, (q_, name, ttl) in zip(axes, [
            ("Q1", "O1b share of readout(0) activation on on-copy latents", "Q1: readout(0) mass on copy latents"),
            ("Q3", "T1 R: R2(R+B0) - R2(B0) (new)", "Q3: target semantics beyond history (new rows)"),
            ("Q4", "T2 AUC(R+content) - AUC(content)", "Q4: same-day beyond content")]):
        for k, v in enumerate(obs.variant.unique()):
            q = obs[(obs.variant == v) & (obs.q == q_) & (obs.estimand == name)].sort_values("layer")
            ax.errorbar(q.layer + (k - 0.5) * 0.6, q.est, yerr=[q.est - q.lo, q.hi - q.est], fmt="o-",
                        color=col(v), ms=style.MARKER, capsize=0, label=v)
        style.null_line(ax, 0, "")
        ax.set_xticks(LAYERS)
        ax.set_xlabel("layer")
        ax.set_title(ttl, fontsize=7)
    style.legend_outside(axes[-1])
    made += [str(p) for p in style.save(fig, out / "fig2_observational")]
    plt.close("all")
    return made


def md_table(df: pd.DataFrame) -> list[str]:
    """A markdown table without the optional `tabulate` dependency."""
    cols = list(df.columns)
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
        ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--causal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    fid, obs, recs, variants = {}, [], [], []
    for c in ("cell-00", "cell-01"):
        tin = json.loads((args.train / c / "inputs.json").read_text())
        v = tin["variant"]
        variants.append(v)
        fid[v] = json.loads((args.train / c / "fidelity.json").read_text())
        obs.append(pd.read_csv(args.analysis / c / "observational.csv"))
        val = json.loads((args.causal / c / "validation.json").read_text())
        sp = val["self_patch"]
        if sp["n"] and sp["n_exact"] != sp["n"]:
            raise RuntimeError(f"{c}: self-patch controls not bit-exact: {sp}")
        r = pd.read_parquet(args.causal / c / "records.parquet")
        pos = pd.read_parquet(args.analysis / c / "positions.parquet")
        r = r.merge(pos[pos.d == 0][["example_id", "c1"]], on="example_id", validate="many_to_one")
        r["variant"] = v
        recs.append(r)
    obs = pd.concat(obs, ignore_index=True)
    boot = UserBootstrap(pd.concat([r.user_id for r in recs]), draws=DRAWS, seed=SEED)
    res = pd.DataFrame([e for r, v in zip(recs, variants) for e in causal_estimands(r, boot, v)])
    res.to_csv(args.out / "causal_estimates.csv", index=False)
    obs.to_csv(args.out / "observational_estimates.csv", index=False)
    rows = []
    for v, f in fid.items():
        for site, rec in f.items():
            rows.append({"variant": v, "site": site, "fvu_train_holdout": rec["fvu_train_holdout"],
                         "dead_frac_test": rec["dead_frac_test"],
                         **{f"fvu_test_{k}": x for k, x in rec["fvu_test"].items()},
                         **{f"fvu_holdout_{k}": x for k, x in rec.get("fvu_train_holdout_by_role", {}).items()}})
    fdf = pd.DataFrame(rows)
    fdf.to_csv(args.out / "fidelity.csv", index=False)
    figs = figures(res, obs, fid, args.out / "figures")
    L = ["# representation/exp2: SAEs on the AR residual stream", "",
         "Paired user bootstrap, 2,000 draws, seed 20260930; intervals exclude training-seed variance "
         "(recommenders and SAEs are each fitted once).", "", "## F1 fidelity", "",
         *md_table(fdf.round(4)), "", "## Causal estimands", "",
         "| variant | part | layer | digit | estimand | est | 95% CI | rows | users |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        L.append(f"| {r.variant} | {r.part} | {r.layer} | {'' if pd.isna(r.get('digit')) else int(r.digit)} | "
                 f"{r.estimand} | {r.est:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {r.n_rows} | {r.n_users} |")
    L += ["", "## Observational estimands", "",
          "| variant | q | layer | estimand | latent | est | 95% CI |", "|---|---|---|---|---|---|---|"]
    for _, r in obs.iterrows():
        ci = "" if pd.isna(r.get("lo")) else f"[{r.lo:.3f}, {r.hi:.3f}]"
        lat = "" if pd.isna(r.get("latent")) else int(r.latent)
        L.append(f"| {r.variant} | {r.q} | {r.layer} | {r.estimand} | {lat} | {r.est:.3f} | {ci} |")
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    (args.out / "result.json").write_text(json.dumps({"figures": figs, "inputs_sha256": {
        str(p): sha256_file(p) for g in (args.causal, args.analysis) for p in sorted(g.glob("cell-*/*.parquet"))}},
        indent=1))
    (args.out / "summarize.py.source").write_text(Path(__file__).read_text())
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
