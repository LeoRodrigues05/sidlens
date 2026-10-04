#!/usr/bin/env python
"""Stage A: label every live SAE latent by the variable it tracks (protocol.md).

For each layer and token group, the active pattern of each latent (active =
non-zero after TopK) is scored against a fixed list of categorical variables
by the best one-value F1, against a permutation null. CPU only; sparse
co-occurrence counts make the full census a few minutes.

Traps guarded here:

* Readout groups are defined from the self-forced test capture: R_0 is the
  last response-header token; R_d (d >= 1) is the answer's digit-(d-1) token,
  which predicts answer digit d. The last answer token predicts end-of-text
  and is excluded.
* answer, r1 and golden coincide often (copying). A latent labelled answer_d
  is re-scored on rows where the answer digit differs from r1's digit, with
  its own null; only then is it counted as answer-beyond-copy.
* The store's rows must match the self-greedy CSV row for row (example ids),
  or the answer labels would be misaligned; this raises.

    python census.py --cell 0 --test-capture <dir> --sae-run <dir> --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from safetensors.numpy import load_file

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.data import meta as meta_mod                                # noqa: E402
from sidlens.data import timestamps as T                                 # noqa: E402
from sidlens.hooks.store import StoreReader                              # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
MIN_SUPPORT = 20
N_PERM = 5
SEED = 20261003
F1_MIN = 0.3
KIND = {"pos_recency": "format", "own_code": "token", "item_d0": "own item", "item_d1": "own item",
        "prev_item_d0": "previous item", "brand": "semantic", "r1_d": "copy source",
        "answer_d": "own answer", "answer_d0": "own answer", "golden_d": "true target",
        "match_prefix": "match", "same_day": "burst"}


def token_table(store: StoreReader, greedy: pd.DataFrame, variant: str, n: int) -> pd.DataFrame:
    """One row per stored token with its group and every variable (NaN where undefined)."""
    rows = store.rows.reset_index(drop=True)
    ex = {e.example_id: e for e in P.load_examples(variant, "next-item", "test")}
    g = greedy.set_index("example_id")
    if set(rows.example_id) - set(g.index):
        raise ValueError("store rows without a self-greedy answer")
    meta = meta_mod.load(P.PRIMARY_CATEGORY)
    brand_of = {m.item_id: " ".join(m.brand.lower().split()) for m in meta.values()}
    top = pd.Series(list(brand_of.values())).replace("", np.nan).dropna().value_counts().head(50).index
    rt, _ = T.ar_row_times(list(ex.values()), T.load_event_times())
    same_day = rt.set_index("example_id").gap1_days.eq(0)

    hdr = rows.role == "response_header"
    last_hdr = rows[hdr].groupby("example_id").pos.transform("max")
    is_r0 = np.zeros(len(rows), bool)
    is_r0[rows.index[hdr][rows[hdr].pos.to_numpy() == last_hdr.to_numpy()]] = True

    out = []
    for i, r in enumerate(rows.itertuples(index=False)):
        e = ex[r.example_id]
        hc = [P.parse_sid(s, n) for s in e.history_sids]
        ans = P.parse_sid(g.at[r.example_id, "greedy_sid"], n)
        gold = P.parse_sid(e.target_sids[0], n)
        r1 = hc[-1]
        rec = {"i": i, "example_id": r.example_id, "user_id": e.user_id}
        if r.role == "hist_sid":
            k, d = int(r.item), int(r.digit)
            rec_k = len(hc) - k                                   # 1 = most recent
            rec.update(group=f"H_{d}", pos_recency=rec_k if rec_k <= 4 else (5 if rec_k < 10 else 10),
                       own_code=hc[k][d], item_d0=hc[k][0] if d >= 1 else np.nan,
                       item_d1=hc[k][1] if d >= 2 else np.nan,
                       prev_item_d0=hc[k - 1][0] if k >= 1 else np.nan,
                       brand=(brand_of.get(e.history_item_ids[k], "") if brand_of.get(e.history_item_ids[k], "") in top
                              else "other") if d == n - 1 else np.nan)
        elif is_r0[i]:
            rec.update(group="R_0", r1_d=r1[0], answer_d=ans[0], golden_d=gold[0],
                       same_day=float(same_day.get(r.example_id, np.nan)))
        elif r.role == "target_sid" and 0 <= int(r.digit) < n - 1:
            d = int(r.digit) + 1                                  # the digit this token predicts
            if int(r.code) != ans[d - 1]:
                raise ValueError(f"{r.example_id}: stored target code is not the self-greedy answer")
            rec.update(group=f"R_{d}", own_code=ans[d - 1], r1_d=r1[d], answer_d=ans[d], answer_d0=ans[0],
                       golden_d=gold[d], match_prefix=float(r1[:d] == ans[:d]))
        else:
            continue
        out.append(rec)
    return pd.DataFrame(out)


def onehot(values: np.ndarray):
    """Sparse one-hot of the values with support >= MIN_SUPPORT; NaN rows are all-zero."""
    s = pd.Series(values)
    ok = s.notna()
    cnt = s[ok].value_counts()
    keep = cnt[cnt >= MIN_SUPPORT].index
    if len(keep) == 0:
        return None, []
    code = pd.Categorical(s.where(s.isin(keep)), categories=keep).codes
    r = np.flatnonzero(code >= 0)
    Y = sp.csr_matrix((np.ones(len(r)), (r, code[r])), shape=(len(s), len(keep)))
    return Y, list(keep)


def best_f1(A: sp.csr_matrix, Y: sp.csr_matrix, fa: np.ndarray):
    """Max over values of F1(active -> value), and the argmax value index, per latent."""
    C = (A.T @ Y).toarray()                                       # latents x values
    fy = np.asarray(Y.sum(0)).ravel()
    f1 = 2 * C / np.maximum(fa[:, None] + fy[None, :], 1)
    return f1.max(1), f1.argmax(1)


def census_group(A, tok: pd.DataFrame, variables: list[str], rng, live):
    res, nulls = {}, {}
    fa = np.asarray(A.sum(0)).ravel()
    for v in variables:
        Y, vals = onehot(tok[v].to_numpy())
        if Y is None:
            continue
        f1, arg = best_f1(A, Y, fa)
        ns = []
        for _ in range(N_PERM):
            ns.append(best_f1(A, Y[rng.permutation(Y.shape[0])], fa)[0][live])
        nulls[v] = float(np.percentile(np.concatenate(ns), 99))
        res[v] = (f1, np.array(vals, dtype=object)[arg])
    return res, nulls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--test-capture", type=Path, required=True, help="ar_capture --out dir (holds store/)")
    ap.add_argument("--sae-run", type=Path, required=True, help="train_sae.py output (one cell)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    root = args.test_capture
    store = StoreReader(root / "store", verify=False)
    if store.meta.get("target") != "self-greedy" or store.meta["split"] != "test":
        raise RuntimeError("census needs the self-greedy test capture")
    variant = store.meta["variant"]
    n = int(variant.split("_")[1][0])
    greedy = pd.read_csv(root / "self_greedy.csv")
    tok = token_table(store, greedy, variant, n)
    tok.drop(columns=[]).to_parquet(args.out / "tokens.parquet", index=False)
    layers = sorted(int(p.stem.split("_L")[1]) for p in args.sae_run.glob("test_latents_L*.safetensors"))
    rng = np.random.default_rng(SEED)
    per_latent, summary, beyond = [], [], []
    for L in layers:
        z = load_file(str(args.sae_run / f"test_latents_L{L}.safetensors"))
        idx, val = z["idx"], z["val"]
        n_lat = int(json.loads((args.sae_run / "sae" / f"L{L}" / "config.json").read_text())["n_latents"])
        for grp, g in tok.groupby("group"):
            ti = g.i.to_numpy()
            ii, vv = idx[ti], val[ti]
            rr = np.repeat(np.arange(len(ti)), ii.shape[1])
            act = vv.ravel() > 0
            A = sp.csr_matrix((np.ones(act.sum()), (rr[act], ii.ravel()[act])), shape=(len(ti), n_lat))
            M = sp.csr_matrix((vv.ravel()[act], (rr[act], ii.ravel()[act])), shape=(len(ti), n_lat))
            fa = np.asarray(A.sum(0)).ravel()
            live = fa >= MIN_SUPPORT
            if not live.any():
                summary.append({"layer": L, "group": grp, "label": "no live latents", "kind": "none",
                                "n_live": 0, "share_latents": np.nan, "share_mass": np.nan, "null99": np.nan})
                print(f"L{L} {grp}: no live latents (n_tokens={len(ti)})", flush=True)
                continue
            variables = [v for v in KIND if v in g and g[v].notna().any()]
            res, nulls = census_group(A, g.reset_index(drop=True), variables, rng, live)
            lat = np.flatnonzero(live)
            margin = np.stack([res[v][0][lat] - nulls[v] for v in res], 1)
            f1s = np.stack([res[v][0][lat] for v in res], 1)
            best = margin.argmax(1)
            names = list(res)
            lab = np.array([names[b] if margin[j, b] > 0 and f1s[j, b] >= F1_MIN else "unexplained"
                            for j, b in enumerate(best)], dtype=object)
            mass = np.asarray(M.sum(0)).ravel()[lat]
            for j, f in enumerate(lat):
                per_latent.append({"layer": L, "group": grp, "latent": int(f), "n_active": int(fa[f]),
                                   "mass": float(mass[j]), "label": lab[j],
                                   **{f"f1_{v}": float(res[v][0][f]) for v in names},
                                   **{f"best_{v}": str(res[v][1][f]) for v in names}})
            tot_m = mass.sum()
            for label in [*names, "unexplained"]:
                sel = lab == label
                summary.append({"layer": L, "group": grp, "label": label, "kind": KIND.get(label, "unexplained"),
                                "n_live": int(live.sum()), "share_latents": float(sel.mean()),
                                "share_mass": float(mass[sel].sum() / tot_m) if tot_m else np.nan,
                                "null99": nulls.get(label, np.nan)})
            # answer-beyond-copy, readout groups
            if grp.startswith("R_") and "answer_d" in res:
                sub = (g.answer_d != g.r1_d).to_numpy()
                ans_lat = lat[lab == "answer_d"]
                if sub.sum() >= MIN_SUPPORT and len(ans_lat):
                    As = A[np.flatnonzero(sub)]
                    Y, vals = onehot(g.answer_d.to_numpy()[sub])
                    fs = np.asarray(As.sum(0)).ravel()
                    f1, _ = best_f1(As, Y, fs) if Y is not None else (np.zeros(n_lat), None)
                    lv = fs >= MIN_SUPPORT
                    ns = np.concatenate([best_f1(As, Y[rng.permutation(Y.shape[0])], fs)[0][lv]
                                         for _ in range(N_PERM)]) if Y is not None and lv.any() else np.zeros(0)
                    n99 = float(np.percentile(ns, 99)) if len(ns) else np.inf     # no live latents: none pass
                    ok = [f for f in ans_lat if fs[f] >= MIN_SUPPORT and f1[f] - n99 > 0 and f1[f] >= F1_MIN]
                    beyond.append({"layer": L, "group": grp, "n_answer_latents": int(len(ans_lat)),
                                   "n_beyond_copy": len(ok), "n_noncopy_rows": int(sub.sum()),
                                   "null99_noncopy": n99, "latents": ok[:50]})
                else:
                    beyond.append({"layer": L, "group": grp, "n_answer_latents": int(len(ans_lat)),
                                   "n_beyond_copy": 0, "n_noncopy_rows": int(sub.sum())})
            print(f"{time.strftime('%H:%M:%S')} L{L} {grp}: live={live.sum()} "
                  f"labels={pd.Series(lab).value_counts().head(4).to_dict()}", flush=True)
    pd.DataFrame(per_latent).to_parquet(args.out / "latents.parquet", index=False)
    pd.DataFrame(summary).to_csv(args.out / "summary.csv", index=False)
    pd.DataFrame(beyond).to_json(args.out / "answer_beyond_copy.json", orient="records", indent=1)
    (args.out / "inputs.json").write_text(json.dumps({
        "test_capture": str(args.test_capture), "sae_run": str(args.sae_run), "layers": layers,
        "n_tokens": len(tok), "groups": tok.group.value_counts().to_dict(), "min_support": MIN_SUPPORT,
        "n_perm": N_PERM, "seed": SEED, "f1_min": F1_MIN, "seconds": round(time.time() - t0, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
