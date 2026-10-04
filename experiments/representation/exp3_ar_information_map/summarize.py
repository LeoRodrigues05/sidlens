#!/usr/bin/env python
"""Summary of representation exp3: estimates, figures and the E1-E6 verdicts.

    python summarize.py --group <dir with cell-00/ cell-01/> --out <new dir>

Each cell dir must hold capture-test/, sae/, lens/ and exactly one census-*/.
Intervals: paired user bootstrap over test users (2,000 draws, seed 20260927),
95% percentile; they exclude training-seed variance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.viz import style as S                                       # noqa: E402

CELLS = {0: "rqkmeans_3codebook_128", 1: "rqvae_4codebook_128"}
EXP6_PLAIN = {0: "283586/cell-00", 1: "283621/cell-01"}
DRAWS, SEED = 2000, 20260927
KIND_COLOR = {"format": "#eda100", "token": "#2a78d6", "own item": "#1baf7a", "previous item": "#4a3aa7",
              "semantic": "#e87ba4", "copy source": "#eb6834", "own answer": "#e34948",
              "true target": "#008300", "match": S.INK_2, "burst": S.MUTED, "unexplained": S.OTHER}


def ci(boot, user, x) -> dict:
    return boot.mean(np.asarray(user), np.asarray(x, dtype=float))


def lens_estimates(lt: pd.DataFrame, boot, n, tag):
    out, per_ex = [], {}
    lt = lt.assign(hit=lt.top1 == lt.answer_d, gold=lt.top1 == lt.golden_d, r1=lt.top1 == lt.r1_d)
    for (lens, L, d), g in lt.groupby(["lens", "layer", "d"]):
        for m in ("hit", "gold", "r1"):
            out.append({**tag, "part": "L1", "lens": lens, "layer": int(L), "d": int(d),
                        "estimand": {"hit": "D1 top1 = answer", "gold": "D3 top1 = golden",
                                     "r1": "D3 top1 = r1"}[m], **ci(boot, g.user_id, g[m])})
    ex = lt.groupby(["lens", "layer", "example_id", "user_id"]).agg(all_hit=("hit", "all"),
                                                                    copy_ans=("copy_answer", "first")).reset_index()
    for (lens, L), g in ex.groupby(["lens", "layer"]):
        out.append({**tag, "part": "L1", "lens": lens, "layer": int(L), "d": -1, "estimand": "D2 whole SID",
                    **ci(boot, g.user_id, g.all_hit)})
        per_ex[(lens, int(L))] = g.set_index("example_id").all_hit
    # D4: earliest layer from which D2 holds at every later layer (tuned lens)
    layers = sorted(L for lens, L in per_ex if lens == "tuned")
    M = pd.concat([per_ex[("tuned", L)] for L in layers], axis=1)
    M.columns = layers
    ok_from = M.iloc[:, ::-1].cummin(axis=1).iloc[:, ::-1]               # True iff all later layers hit
    first = ok_from.apply(lambda r: next((L for L, v in r.items() if v), np.nan), axis=1)
    d4 = ex[ex.lens == "tuned"].drop_duplicates("example_id").set_index("example_id")[["copy_ans", "user_id"]]
    d4["first_layer"] = first
    return out, d4


def decoder_estimates(dt: pd.DataFrame, boot, tag, rule):
    out = []
    for (task, L), g in dt.groupby(["task", "layer"]):
        for m in ("exact", "top10"):
            out.append({**tag, "part": "L2", "task": task, "layer": L, "estimand": m,
                        **ci(boot, g.user_id, g[m])})
        if task in ("answer_at_R0",):
            nc = g[~g.copy_answer]
            out.append({**tag, "part": "L2", "task": task, "layer": L, "estimand": "exact | non-copy answer",
                        **ci(boot, nc.user_id, nc.exact)})
    for name, (user, x) in rule.items():
        out.append({**tag, "part": "L2", "task": name.split(":")[0], "layer": "rule",
                    "estimand": name.split(":")[1], **ci(boot, user, x)})
    return out


def rules(cell: int, variant: str, n: int, dt: pd.DataFrame):
    """Rule baselines on the same test tokens: copy r1; most frequent SID given digit 0."""
    ex = {e.example_id: e for e in P.load_examples(variant, "next-item", "test")}
    r = {}
    a = dt[(dt.task == "answer_at_R0") & (dt.layer == dt.layer.iloc[0])]
    copy = np.array([P.parse_sid(ex[i].history_sids[-1], n) == tuple(map(int, t.split(",")))
                     for i, t in zip(a.example_id, a.target)])
    r["answer_at_R0:exact copy rule"] = (a.user_id.to_numpy(), copy)
    gd = dt[(dt.task == "golden_at_R0") & (dt.layer == dt.layer.iloc[0])]
    copyg = np.array([ex[i].history_sids[-1] == ex[i].target_sids[0] for i in gd.example_id])
    r["golden_at_R0:exact copy rule"] = (gd.user_id.to_numpy(), copyg)
    train = P.load_examples(variant, "next-item", "train")
    cnt = pd.Series([P.parse_sid(s, n) for e in train for s in e.history_sids]).value_counts()
    mode = {}
    for sid in cnt.index:
        mode.setdefault(sid[0], sid)
    f = dt[(dt.task == "item_at_first") & (dt.layer == dt.layer.iloc[0])]
    tgt = [tuple(map(int, t.split(","))) for t in f.target]
    r["item_at_first:exact mode-given-digit0 rule"] = (f.user_id.to_numpy(),
                                                       np.array([mode.get(t[0]) == t for t in tgt]))
    return r


def fig_infomap(summ: dict, out: Path):
    import matplotlib.pyplot as plt
    S.apply()
    for cell, sm in summ.items():
        groups = sorted(sm.group.unique(), key=lambda g: (g[0], int(g.split("_")[1])))
        fig, axes = plt.subplots(1, len(groups), figsize=(1.9 * len(groups), 2.2), sharey=True)
        for ax, gname in zip(np.atleast_1d(axes), groups):
            g = sm[sm.group == gname].groupby(["layer", "kind"]).share_latents.sum().unstack(fill_value=0)
            kinds = [k for k in KIND_COLOR if k in g.columns]
            if not kinds:
                ax.set_title(f"{gname}: no live latents", fontsize=7)
                continue
            ax.stackplot(g.index, *[g[k] for k in kinds], colors=[KIND_COLOR[k] for k in kinds],
                         labels=kinds, linewidth=0)
            ax.set_title(gname, fontsize=8)
            ax.set_xlabel("layer")
        np.atleast_1d(axes)[0].set_ylabel("share of live latents")
        S.legend_outside(np.atleast_1d(axes)[-1], fontsize=6.5, frameon=False)
        fig.suptitle(CELLS[cell], fontsize=8)
        S.save(fig, out / f"fig1_information_map_{CELLS[cell]}")


def fig_lens(est: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3), sharey=True)
    for ax, (cell, name) in zip(axes, CELLS.items()):
        e = est[(est.variant == name) & (est.part == "L1") & (est.estimand == "D2 whole SID")]
        for lens, ls in (("tuned", "-"), ("logit", "--")):
            g = e[e.lens == lens].assign(L=lambda x: x.layer.astype(int)).sort_values("L")
            ax.plot(g.L, g.est.astype(float), ls, color=S.INK, lw=S.LINE_W, label=f"{lens} lens: whole answer")
            ax.fill_between(g.L, g.lo.astype(float), g.hi.astype(float), color=S.NULL_BAND, lw=0)
        e1 = est[(est.variant == name) & (est.part == "L1") & (est.estimand == "D1 top1 = answer")
                 & (est.lens == "tuned")]
        for d, g in e1.groupby("d"):
            g = g.assign(L=lambda x: x.layer.astype(int)).sort_values("L")
            ax.plot(g.L, g.est.astype(float), color=S.DEPTH_RAMP[min(int(d) + 1, 5)], lw=1.0, label=f"digit {int(d) + 1}")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("agreement with the model's answer")
    S.legend_outside(axes[-1], fontsize=6.5, frameon=False)
    S.save(fig, out / "fig2_sid_lens")


def fig_decoders(est: pd.DataFrame, out: Path):
    import matplotlib.pyplot as plt
    S.apply()
    tasks = ["item_at_last", "item_at_first", "prev_item_at_last", "r1_at_R0", "answer_at_R0", "golden_at_R0"]
    fig, axes = plt.subplots(2, len(tasks), figsize=(2.0 * len(tasks), 3.8), sharey=True)
    for row, (cell, name) in enumerate(CELLS.items()):
        for ax, task in zip(axes[row], tasks):
            e = est[(est.variant == name) & (est.part == "L2") & (est.task == task) & (est.estimand == "exact")]
            num = e[e.layer.astype(str).str.isdigit()].assign(L=lambda x: x.layer.astype(int)).sort_values("L")
            ax.plot(num.L, num.est.astype(float), color=S.INK, lw=S.LINE_W)
            ax.fill_between(num.L, num.lo.astype(float), num.hi.astype(float), color=S.NULL_BAND, lw=0)
            pri = e[e.layer == "prior"]
            if len(pri):
                S.null_line(ax, float(pri.est.iloc[0]), "prior")
            rl = est[(est.variant == name) & (est.part == "L2") & (est.task == task) & (est.layer == "rule")]
            if len(rl):
                ax.axhline(float(rl.est.iloc[0]), color=S.QUANTIZER_COLOR["rqvae"], lw=0.9, ls=":")
            if task == "answer_at_R0":
                nc = est[(est.variant == name) & (est.part == "L2") & (est.task == task)
                         & (est.estimand == "exact | non-copy answer")]
                nc = nc[nc.layer.astype(str).str.isdigit()].assign(L=lambda x: x.layer.astype(int)).sort_values("L")
                ax.plot(nc.L, nc.est.astype(float), color=S.QUANTIZER_COLOR["rqkmeans"], lw=1.0)
            ax.set_title(task if row == 0 else "", fontsize=7)
            if row == 1:
                ax.set_xlabel("layer")
        axes[row, 0].set_ylabel(f"{name}\nexact SID")
    S.save(fig, out / "fig3_sid_decoders")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    est, summ, verdict, d4s, sources = [], {}, {}, {}, {}
    for cell, name in CELLS.items():
        cd = args.group / f"cell-{cell:02d}"
        cens = sorted(cd.glob("census-*/result"))
        if len(cens) != 1:
            raise RuntimeError(f"{cd}: expected exactly one census run, found {len(cens)}")
        sources[name] = {"cell": str(cd), "census": str(cens[0])}
        n = 3 if cell == 0 else 4
        tag = {"variant": name}
        g = pd.read_csv(cd / "capture-test" / "self_greedy.csv")
        boot = UserBootstrap(g.user_id, draws=DRAWS, seed=SEED)
        est.append({**tag, "part": "C", "estimand": "greedy = golden",
                    **ci(boot, g.user_id, g.greedy_sid == g.golden_sid)})
        bp = pd.read_parquet(paths.DERIVED / "controlled" / "exp6_ar_copy_in_decoding" / EXP6_PLAIN[cell]
                             / "predictions_B_plain.parquet").set_index("example_id").predict
        est.append({**tag, "part": "C", "estimand": "greedy = plain-beam top-1",
                    **ci(boot, g.user_id, [s == list(bp.loc[i])[0] for i, s in zip(g.example_id, g.greedy_sid)])})
        sm = pd.read_csv(cens[0] / "summary.csv")
        summ[cell] = sm
        lt = pd.read_parquet(cd / "lens" / "lens_tokens.parquet")
        e1, d4 = lens_estimates(lt, boot, n, tag)
        est += e1
        ex = {e.example_id: e for e in P.load_examples(name, "next-item", "test")}
        d4["repeat_target"] = [ex[i].target_sids[0] in ex[i].history_sids for i in d4.index]
        d4s[name] = d4
        for k, sub in (("all", d4), ("copy answer", d4[d4.copy_ans]), ("non-copy answer", d4[~d4.copy_ans]),
                       ("repeat target", d4[d4.repeat_target]), ("new target", d4[~d4.repeat_target])):
            q = sub.first_layer
            est.append({**tag, "part": "L1", "estimand": f"D4 first stable layer, {k}",
                        "est": float(q.median()), "lo": float(q.quantile(0.25)), "hi": float(q.quantile(0.75)),
                        "n_rows": int(len(q)), "never": float(q.isna().mean())})
        dt = pd.read_parquet(cd / "lens" / "decoder_tokens.parquet")
        a = dt.task == "answer_at_R0"
        cp = {i: P.parse_sid(ex[i].history_sids[-1], n) for i in ex}
        dt["copy_answer"] = [bool(t == ",".join(map(str, cp[i]))) for i, t in zip(dt.example_id, dt.target)]
        dt.loc[~a, "copy_answer"] = False
        est += decoder_estimates(dt, boot, tag, rules(cell, name, n, dt))
        beyond = pd.read_json(cens[0] / "answer_beyond_copy.json")
        verdict[name] = {"census_answer_beyond_copy_R0": [] if beyond.empty else beyond[beyond.group == "R_0"][
            ["layer", "n_answer_latents", "n_beyond_copy"]].to_dict("records")}
        fid = json.loads((cd / "sae" / "fidelity.json").read_text())
        verdict[name]["sae_fvu_test_all"] = {k: v["fvu_test"]["all"] for k, v in fid.items()}
    est = pd.DataFrame(est)
    est.to_csv(args.out / "estimates.csv", index=False)
    pd.concat([s.assign(variant=CELLS[c]) for c, s in summ.items()]).to_csv(args.out / "census_summary.csv", index=False)
    for name, d4 in d4s.items():
        d4.to_csv(args.out / f"d4_first_stable_layer__{name}.csv")
    fig_infomap(summ, args.out / "figures")
    fig_lens(est, args.out / "figures")
    fig_decoders(est, args.out / "figures")
    (args.out / "verdict_inputs.json").write_text(json.dumps(verdict, indent=1, default=str))
    (args.out / "sources.json").write_text(json.dumps(sources, indent=1))
    L = ["# Representation exp3: generated summary", "",
         "Paired user bootstrap (2,000 draws, seed 20260927); 95% intervals exclude training-seed "
         "variance. D4 rows give median [IQR] of the first stable layer.", ""]
    for name, g in est.groupby("variant", sort=False):
        L += [f"## {name}", "", "| part | lens/task | layer | d | estimand | est [95% CI] | n |", "|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lt_ = getattr(r, "lens", "") if isinstance(getattr(r, "lens", ""), str) else ""
            tk = getattr(r, "task", "") if isinstance(getattr(r, "task", ""), str) else ""
            L.append(f"| {r.part} | {lt_ or tk} | {'' if pd.isna(getattr(r, 'layer', np.nan)) else r.layer} | "
                     f"{'' if pd.isna(getattr(r, 'd', np.nan)) else int(r.d)} | {r.estimand} | "
                     f"{r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}] | {r.n_rows} |")
        L.append("")
    (args.out / "report.md").write_text("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
