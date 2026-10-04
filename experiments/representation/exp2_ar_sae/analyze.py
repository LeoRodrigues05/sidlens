#!/usr/bin/env python
"""Observational SAE analyses (Q1-Q4) and the latent selections the causal runs use.

Protocol: `protocol.md` beside this file. One invocation = one cell. CPU only.
Every selection rule is applied to the screening half of test users
(sha256 parity 0). Observational estimands are reported on the held-out half
(Q1, Q2) or out of fold (Q3, Q4).

Traps, each with its guard
--------------------------
* **Selecting and reporting on the same users.** S_L and the copy-latent table
  are computed from half 0 only. Held-out estimands and the causal plans cover
  half 1 only, and `selection.json` records the users of each half.
* **Store rows vs examples.** Readout positions are found by role and pos in
  the store's own rows.parquet, then checked against `Encoded.predict_pos`
  on a sample, so a position map drift raises instead of reading the wrong
  token.
* **Dense probes that see the answer.** Under teacher forcing readout(0) comes
  before any target token, so its residual cannot contain the target tokens.
  Q3 and Q4 use readout(0) only.

    python analyze.py --cell 0 --sae-run <train run>/cell-00 --out <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.bootstrap import UserBootstrap                     # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.data import timestamps as T                                 # noqa: E402
from sidlens.hooks.store import StoreReader                              # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
TEST_STORE = {"next-item_best": "281053", "oneoff_rqvae4cb128": "281054"}
LAYERS = (12, 16, 20, 24)
SEED, DRAWS = 20260930, 2000
N_S, N_CTRL_SETS = 8, 5


def h(user: str, mod: int, salt: str = "") -> int:
    return int(hashlib.sha256(f"{SEED}|{salt}{user}".encode()).hexdigest(), 16) % mod


def load_latents(sae_run: Path, L: int):
    from safetensors.torch import load_file
    t = load_file(str(sae_run / f"test_latents_L{L}.safetensors"))
    return t["idx"].numpy(), t["val"].numpy()


def positions(rows: pd.DataFrame, examples, n: int) -> pd.DataFrame:
    """One row per (example, d): the store row index of readout(d)."""
    hdr = rows[rows.role == "response_header"]
    r0 = hdr.loc[hdr.groupby("example_id").pos.idxmax()]
    out = [{"example_id": e, "d": 0, "store_row": int(i), "pos": int(p)}
           for e, i, p in zip(r0.example_id, r0.index, r0.pos)]
    tg = rows[(rows.role == "target_sid") & (rows.digit < n - 1)]
    out += [{"example_id": e, "d": int(dg) + 1, "store_row": int(i), "pos": int(p)}
            for e, dg, i, p in zip(tg.example_id, tg.digit, tg.index, tg.pos)]
    df = pd.DataFrame(out)
    if len(df) != len(examples) * n:
        raise ValueError(f"expected {len(examples) * n} readout rows, found {len(df)}")
    return df


def check_positions(pos: pd.DataFrame, examples, ckpt: str, n: int, k: int = 50) -> None:
    from transformers import AutoTokenizer
    from sidlens.models import ar
    vocab = ar.load_vocab(ckpt)
    tok = AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / ckpt), local_files_only=True)
    by = pos.set_index(["example_id", "d"]).pos
    for ex in examples[:k]:
        e = P.encode(ex, tok, vocab, template="eval", with_target=True)
        for d in range(n):
            if by[(ex.example_id, d)] != e.predict_pos(0, d):
                raise ValueError(f"{ex.example_id} d={d}: store pos {by[(ex.example_id, d)]} != predict_pos")


def dense_acts(idx, val, rows_sel, latents):
    """(len(rows_sel), len(latents)) activations; 0 where inactive."""
    col = {f: j for j, f in enumerate(latents)}
    A = np.zeros((len(rows_sel), len(latents)), np.float32)
    ii, vv = idx[rows_sel], val[rows_sel]
    for r in range(len(rows_sel)):
        for f, v in zip(ii[r], vv[r]):
            j = col.get(int(f))
            if j is not None and v > 0:
                A[r, j] = v
    return A


def auc_columns(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    pos, neg = y.sum(), (~y).sum()
    out = np.empty(A.shape[1])
    for j in range(A.shape[1]):
        r = rankdata(A[:, j])
        out[j] = (r[y].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--sae-run", type=Path, required=True, help="train run cell dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--smoke", action="store_true", help="accept a pilot SAE run (code test only)")
    args = ap.parse_args(argv)
    ckpt = CELLS[args.cell]
    if (args.sae_run / f"cell-{args.cell:02d}").is_dir():          # a run group: take this cell
        args.sae_run = args.sae_run / f"cell-{args.cell:02d}"
    layers = tuple(int(x) for x in args.layers.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "selection.json").exists():
        raise FileExistsError(f"{args.out} already holds results")
    t0 = time.time()
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)       # noqa: E731
    tin = json.loads((args.sae_run / "inputs.json").read_text())
    if tin["checkpoint"] != f"ar/{ckpt}" or ((tin.get("pilot_epochs") or tin.get("pilot_max_train_tokens")) and not args.smoke):
        raise ValueError(f"{args.sae_run} is not a full training run for {ckpt}")
    st = StoreReader(paths.DERIVED / "ar_capture" / TEST_STORE[ckpt] / "capture" / "store", verify=True)
    variant = st.meta["variant"]
    n = int(variant.split("_")[1][0])
    examples = P.load_examples(variant, "next-item", "test")
    by = {e.example_id: e for e in examples}
    pos = positions(st.rows, examples, n)
    check_positions(pos, examples, ckpt, n)
    rows_t, _ = T.ar_row_times(examples)
    rows_t = rows_t.set_index("example_id")
    codes = {e.example_id: ([P.parse_sid(s, n) for s in e.history_sids], P.parse_sid(e.target_sids[0], n))
             for e in examples}
    pos["user_id"] = pos.example_id.map(lambda i: by[i].user_id)
    pos["half"] = pos.user_id.map(lambda u: h(u, 2))
    pos["c1"] = pos.example_id.map(lambda i: codes[i][0][-1][0])
    pos["ct"] = pos.example_id.map(lambda i: codes[i][1][0])
    pos["match"] = [tuple(codes[i][0][-1][:d]) == tuple(codes[i][1][:d]) for i, d in zip(pos.example_id, pos.d)]
    pos["r1_next"] = [codes[i][0][-1][d] for i, d in zip(pos.example_id, pos.d)]
    pos["gap1_days"] = pos.example_id.map(rows_t.gap1_days)
    pos.to_parquet(args.out / "positions.parquet", index=False)
    boot = UserBootstrap(pos.user_id, draws=DRAWS, seed=SEED)
    res, selection = [], {"cell": ckpt, "variant": variant, "layers": {}, "screening_users": int(
        pos[pos.half == 0].user_id.nunique()), "heldout_users": int(pos[pos.half == 1].user_id.nunique())}

    # item embeddings for Q3
    emb = np.load(paths.FROZEN_DATA / "embeddings" / f"{P.PRIMARY_CATEGORY}.emb-qwen-td.npy").astype(np.float32)
    mu = emb.mean(0)
    U, S, Vt = np.linalg.svd(emb - mu, full_matrices=False)
    pc = (emb - mu) @ Vt[:64].T
    pc = (pc - pc.mean(0)) / pc.std(0)
    r0 = pos[pos.d == 0].reset_index(drop=True)
    tgt_item = np.array([by[i].target_item_ids[0] for i in r0.example_id])
    r1_item = np.array([by[i].history_item_ids[-1] for i in r0.example_id])
    Y = pc[tgt_item]
    B0 = np.hstack([pc[r1_item], np.stack([pc[list(by[i].history_item_ids)].mean(0) for i in r0.example_id])])
    new = np.array([(by[i].target_sids[0] not in by[i].history_sids) and (ct != c1)
                    for i, ct, c1 in zip(r0.example_id, r0.ct, r0.c1)])
    fold = r0.user_id.map(lambda u: h(u, 5)).to_numpy()
    # content features for Q4
    def pref(a, b):
        k = 0
        while k < n and a[k] == b[k]:
            k += 1
        return k
    Bc = np.array([[float(len(codes[i][0]) >= 2 and codes[i][0][-1][0] == codes[i][0][-2][0]),
                    float(pref(codes[i][0][-1], codes[i][0][-2])) if len(codes[i][0]) >= 2 else 0.0,
                    float(len(codes[i][0]) >= 2 and codes[i][0][-1] == codes[i][0][-2]),
                    float(len(codes[i][0])), float(len({c[0] for c in codes[i][0]}))] for i in r0.example_id])
    y_same = (r0.gap1_days.to_numpy() == 0)

    for L in layers:
        idx, val = load_latents(args.sae_run, L)
        m = int(json.loads((args.sae_run / "sae" / f"L{L}" / "config.json").read_text())["n_latents"])
        sel = {}
        # ---------------- Q1 copy latents at readout(0) ----------------------
        rr = r0.store_row.to_numpy()
        act = [(r, int(f), float(v)) for r, (ii, vv) in enumerate(zip(idx[rr], val[rr])) for f, v in zip(ii, vv) if v > 0]
        a = pd.DataFrame(act, columns=["r", "f", "v"])
        a["half"] = r0.half.to_numpy()[a.r]
        a["c1"] = r0.c1.to_numpy()[a.r]
        scr = a[a.half == 0]
        cnt = scr.groupby(["f", "c1"]).size().rename("n").reset_index()
        tot = cnt.groupby("f").n.sum()
        best = cnt.sort_values(["f", "n"], ascending=[True, False]).drop_duplicates("f").set_index("f")
        table = pd.DataFrame({"support": tot, "c_star": best.c1, "selectivity": best.n / tot})
        copy = table[(table.selectivity >= 0.5) & (table.support >= 20)]
        table.reset_index().to_parquet(args.out / f"q1_latent_table_L{L}.parquet", index=False)
        a["copy_lat"] = a.f.isin(copy.index)
        a["c_star"] = a.f.map(copy.c_star)
        a["on_copy"] = a.copy_lat & (a.c_star == a.c1)
        ho = a[a.half == 1]
        per = ho.groupby("r").apply(lambda g: pd.Series({
            "share_on_copy": g.v[g.on_copy].sum() / g.v.sum(),
            "any_copy": float(g.copy_lat.any()),
            "top_copy_right": float(g[g.copy_lat].sort_values("v").c_star.iloc[-1] == g.c1.iloc[0])
            if g.copy_lat.any() else np.nan}), include_groups=False)
        per["user_id"] = r0.user_id.to_numpy()[per.index]
        tag = {"variant": variant, "layer": L}
        res.append({**tag, "q": "Q1", "estimand": "O1a n copy latents", "est": float(len(copy)), "n_rows": len(copy)})
        res.append({**tag, "q": "Q1", "estimand": "O1b share of readout(0) activation on on-copy latents",
                    **boot.mean(per.user_id, per.share_on_copy)})
        res.append({**tag, "q": "Q1", "estimand": "rows with any copy latent active", **boot.mean(per.user_id, per.any_copy)})
        q = per.dropna(subset=["top_copy_right"])
        res.append({**tag, "q": "Q1", "estimand": "O1c top copy latent's c* = c1", **boot.mean(q.user_id, q.top_copy_right)})
        plan = {}
        for r, g in ho.groupby("r"):
            oc = g[g.on_copy].sort_values("v", ascending=False)
            if not len(oc):
                continue
            others = g[~g.on_copy].copy()
            ctrl = []
            for v in oc.v:
                if not len(others):
                    break
                j = (others.v - v).abs().idxmin()
                ctrl.append(int(others.f[j]))
                others = others.drop(j)
            if len(ctrl) == len(oc):
                plan[r0.example_id[r]] = {"copy": [int(f) for f in oc.f], "control": ctrl}
        sel["copy_plan"] = plan
        # ---------------- Q2 prefix-match latents at readout(d>=1) ------------
        pd1 = pos[pos.d >= 1].reset_index(drop=True)
        rows_d = pd1.store_row.to_numpy()
        ii = idx[rows_d]
        vv = val[rows_d]
        support = np.bincount(ii[(vv > 0) & (pd1.half.to_numpy()[:, None] == 0)].ravel(), minlength=m)
        cand = np.flatnonzero(support >= 20)
        A = dense_acts(idx, val, rows_d, cand)
        y = pd1.match.to_numpy()
        s0, s1 = pd1.half.to_numpy() == 0, pd1.half.to_numpy() == 1
        auc = auc_columns(A[s0], y[s0])
        order = np.argsort(-auc)
        S = [int(cand[j]) for j in order[:N_S]]
        meanact = np.array([A[s0][:, j][A[s0][:, j] > 0].mean() if (A[s0][:, j] > 0).any() else 0 for j in range(len(cand))])
        pool = [j for j in range(len(cand)) if abs(auc[j] - 0.5) < 0.02 and cand[j] not in S]
        rng = np.random.default_rng(SEED + L)
        ctrl_sets = []
        for c_ in range(N_CTRL_SETS):
            used, cs = set(), []
            for f in S:
                j0 = int(np.flatnonzero(cand == f)[0])
                near = sorted(pool, key=lambda j: abs(meanact[j] - meanact[j0]))
                near = [j for j in near if j not in used][:20]
                pick = int(rng.choice(near))
                used.add(pick)
                cs.append(int(cand[pick]))
            ctrl_sets.append(cs)
        auc_ho = auc_columns(A[s1][:, [int(np.flatnonzero(cand == f)[0]) for f in S]], y[s1])
        sel["S"] = S
        sel["S_auc_screen"] = [float(auc[np.flatnonzero(cand == f)[0]]) for f in S]
        sel["S_auc_heldout"] = [float(x) for x in auc_ho]
        sel["control_sets"] = ctrl_sets
        sel["n_candidates"] = int(len(cand))
        for f, a_s, a_h in zip(S, sel["S_auc_screen"], sel["S_auc_heldout"]):
            res.append({**tag, "q": "Q2", "estimand": "O2a held-out AUC (match_d)", "latent": f, "est": a_h,
                        "screen_auc": a_s, "n_rows": int(s1.sum())})
        res.append({**tag, "q": "Q2", "estimand": "O2a best held-out AUC among S", "est": float(auc_ho.max()),
                    "n_rows": int(s1.sum())})
        res.append({**tag, "q": "Q2", "estimand": "screen AUC null band (|AUC-0.5|<0.02) pool size",
                    "est": float(len(pool)), "n_rows": int(len(cand))})
        selection["layers"][str(L)] = sel
        # ---------------- Q3 target semantics, Q4 same-day -------------------
        R = st.load(f"model.layers.{L}", rows=rr).float().numpy()
        act_cnt = np.bincount(idx[rr][val[rr] > 0].ravel(), minlength=m)
        zl = np.flatnonzero(act_cnt >= 20)
        Z = dense_acts(idx, val, rr, zl)
        for name, X in (("R", R), ("Z", Z)):
            sse = {}
            for feat, M in (("B0", B0), (name, X), (f"{name}+B0", np.hstack([X, B0]))):
                sse[feat] = ridge_oof(M, Y, fold)
            sst = ((Y - Y.mean(0)) ** 2).sum(1)
            for sub, msk in (("all", np.ones(len(Y), bool)), ("new", new)):
                u = r0.user_id[msk]
                res.append({**tag, "q": "Q3", "estimand": f"R2 B0 ({sub})", **ratio_r2(boot, u, sse["B0"][msk], sst[msk])})
                res.append({**tag, "q": "Q3", "estimand": f"R2 {name} ({sub})", **ratio_r2(boot, u, sse[name][msk], sst[msk])})
                res.append({**tag, "q": "Q3", "estimand": f"T1 {name}: R2({name}+B0) - R2(B0) ({sub})",
                            **boot.ratio(u, sse["B0"][msk] - sse[f"{name}+B0"][msk], sst[msk])})
        pb = logit_oof(Bc, y_same, fold)
        pr = logit_oof(np.hstack([R, Bc]), y_same, fold)
        res.append({**tag, "q": "Q4", "estimand": "AUC same-day | content", **boot_auc(boot, r0.user_id, y_same, pb)})
        res.append({**tag, "q": "Q4", "estimand": "AUC same-day | R+content", **boot_auc(boot, r0.user_id, y_same, pr)})
        res.append({**tag, "q": "Q4", "estimand": "T2 AUC(R+content) - AUC(content)",
                    **boot_auc(boot, r0.user_id, y_same, pr, pb)})
        sc0 = r0.half.to_numpy() == 0
        au = auc_columns(Z[sc0], y_same[sc0])
        top = np.argsort(-np.abs(au - 0.5))[:10]
        au_h = auc_columns(Z[~sc0][:, top], y_same[~sc0])
        for j, ah in zip(top, au_h):
            f = int(zl[j])
            res.append({**tag, "q": "Q4", "estimand": "same-day latent held-out AUC", "latent": f, "est": float(ah),
                        "screen_auc": float(au[j]), "c_star": float(table.c_star.get(f, np.nan)),
                        "selectivity": float(table.selectivity.get(f, np.nan)), "n_rows": int((~sc0).sum())})
        log(f"[L{L}] copy latents {len(copy)}, S auc_ho {np.round(auc_ho, 3).tolist()}, "
            f"copy plans {len(plan)}, {time.time() - t0:.0f}s")

    pd.DataFrame(res).to_csv(args.out / "observational.csv", index=False)
    (args.out / "selection.json").write_text(json.dumps(selection))
    (args.out / "inputs.json").write_text(json.dumps({"cell": ckpt, "sae_run": str(args.sae_run),
                                                      "test_store": TEST_STORE[ckpt], "seed": SEED}, indent=1))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


def ridge_oof(X, Y, fold, alphas=(1.0, 10.0, 100.0, 1000.0)) -> np.ndarray:
    """Per-row out-of-fold SSE (summed over Y dims); alpha by inner user-fold CV."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sse = np.empty(len(Y))
    for k in range(5):
        tr, te = fold != k, fold == k
        inner = fold[tr]
        best, best_e = None, np.inf
        for a in alphas:
            e = 0.0
            for j in [x for x in range(5) if x != k][:4]:
                itr, ite = inner != j, inner == j
                sc = StandardScaler().fit(X[tr][itr])
                mdl = Ridge(alpha=a).fit(sc.transform(X[tr][itr]), Y[tr][itr])
                e += ((mdl.predict(sc.transform(X[tr][ite])) - Y[tr][ite]) ** 2).sum()
            if e < best_e:
                best, best_e = a, e
        sc = StandardScaler().fit(X[tr])
        mdl = Ridge(alpha=best).fit(sc.transform(X[tr]), Y[tr])
        sse[te] = ((mdl.predict(sc.transform(X[te])) - Y[te]) ** 2).sum(1)
    return sse


def logit_oof(X, y, fold, Cs=(0.01, 0.1, 1.0)) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    p = np.empty(len(y))
    for k in range(5):
        tr, te = fold != k, fold == k
        inner = fold[tr]
        best, best_a = None, -1
        for C in Cs:
            sc_ = []
            for j in [x for x in range(5) if x != k][:4]:
                itr, ite = inner != j, inner == j
                s = StandardScaler().fit(X[tr][itr])
                mdl = LogisticRegression(C=C, max_iter=500).fit(s.transform(X[tr][itr]), y[tr][itr])
                sc_.append(roc_auc_score(y[tr][ite], mdl.predict_proba(s.transform(X[tr][ite]))[:, 1]))
            if np.mean(sc_) > best_a:
                best, best_a = C, np.mean(sc_)
        s = StandardScaler().fit(X[tr])
        mdl = LogisticRegression(C=best, max_iter=500).fit(s.transform(X[tr]), y[tr])
        p[te] = mdl.predict_proba(s.transform(X[te]))[:, 1]
    return p


def ratio_r2(boot, user, sse, sst) -> dict:
    r = boot.ratio(user, sse, sst)
    return {**r, "est": 1 - r["est"], "lo": 1 - r["hi"], "hi": 1 - r["lo"]}


def boot_auc(boot, user, y, p, p0=None) -> dict:
    """AUC (or AUC(p) - AUC(p0)) with users resampled by the shared bootstrap weights."""
    from sklearn.metrics import roc_auc_score
    u = boot.index.loc[np.asarray(user)].to_numpy()
    f = lambda w: roc_auc_score(y, p, sample_weight=w) - (roc_auc_score(y, p0, sample_weight=w) if p0 is not None else 0)  # noqa: E731
    est = f(None)
    bs = np.array([f(boot.W[b][u]) for b in range(boot.draws)])
    return UserBootstrap._summ(est, bs, len(y), len(np.unique(u)))


if __name__ == "__main__":
    raise SystemExit(main())
