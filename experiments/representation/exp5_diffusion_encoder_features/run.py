#!/usr/bin/env python
"""representation/exp5: what the DiffGRM encoder holds (probes, SAEs, description autoencoder).

Protocol: `protocol.md` beside this file, declared before any forward pass of
this experiment. One invocation = one checkpoint. The pure parts (histories,
census, control matching, description design) are in `common.py`; this file
loads, gates, fits and writes. Every digit score goes through
`interventions.diffusion.score_states` (the projected cross cache exp1
validated), at the fully masked state, in fixed-size batches.

    python run.py --cell 7 --out <new dir> [--limit 600] [--device cpu]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C                                                       # noqa: E402
from sidlens import paths                                                # noqa: E402
from sidlens.analysis import diffusion_lens as DL                        # noqa: E402
from sidlens.analysis.matched_decode import decode as matched_decode     # noqa: E402
from sidlens.data import labels as LB                                    # noqa: E402
from sidlens.data import meta as META                                    # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.data.sids import load_item2id                               # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.probes import linear as LP                                  # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402
from sidlens.sae.topk import SaeConfig, TopKSAE                          # noqa: E402

ENC_B, DEC_B = 512, 1024
SITES = ("input", "mid", "output")
ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEV = "cuda"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def _pad(a: np.ndarray, size: int) -> np.ndarray:
    return a if len(a) == size else np.concatenate([a, np.repeat(a[:1], size - len(a), 0)])


def encode_sites(model, H: C.Histories, keep_output=False, blocked=None):
    """Sites at real slots [rows, 3, W] (slot-table order) and optionally the full output [N, S, W].

    ``blocked`` [N] (slot or -1): exp8's encoder route block, every other real
    slot stops reading that slot. Every batch then carries an explicit mask,
    padding-derived where nothing is blocked.
    """
    rows, full = [], []
    for s in range(0, len(H), ENC_B):
        real = min(ENC_B, len(H) - s)
        h = torch.from_numpy(_pad(H.sids[s:s + ENC_B], ENC_B)).to(DEV)
        m_np = _pad(H.mask[s:s + ENC_B], ENC_B)
        m = torch.from_numpy(m_np).to(DEV)
        if blocked is None:
            sites = DL.encoder_sites(model, h, m)
        else:
            b = _pad(blocked[s:s + ENC_B], ENC_B)
            em = torch.from_numpy(DI.encoder_block_mask(m_np, [(r, int(k)) for r, k in enumerate(b[:real]) if k >= 0])).to(DEV)
            with DI.encoder_attention_masks(model, em):
                sites = {"output": diffusion.encode(model, h, m)}
        if blocked is None:
            rows.append(torch.stack([sites[k][:real][m[:real]] for k in SITES], 1))
        if keep_output:
            full.append(sites["output"][:real])
    return (torch.cat(rows) if rows else None), (torch.cat(full) if keep_output else None)


def score_full(model, enc):
    """Log-probs [B, n, K] at the fully masked state, in fixed DEC_B-row forwards."""
    n = int(model.n_digit)
    out = []
    for s in range(0, enc.shape[0], DEC_B):
        e = enc[s:s + DEC_B]
        real = e.shape[0]
        if real < DEC_B:
            e = torch.cat([e, e[:1].expand(DEC_B - real, -1, -1)])
        rows = torch.arange(DEC_B, device=DEV)
        codes = torch.zeros((DEC_B, n), dtype=torch.long, device=DEV)
        with torch.inference_mode():
            lp = DI.score_states(model, e, DI.project_cross_cache(model, e), rows, codes, torch.ones_like(codes, dtype=torch.bool))
        out.append(lp[:real])
    return torch.cat(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(C.CELLS)))
    ap.add_argument("--ckpt", help="a next-item DiffGRM grid weight state (grid:<key>@<state>) instead of --cell")
    ap.add_argument("--limit", type=int, default=None, help="first N cohort users (pilot)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    global DEV
    DEV = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    t0 = time.time()

    grid_setup = None
    if args.ckpt:                    # a grid run: its own cohort, catalog, sequences and model
        from sidlens.grid import diff_setup as DS, inputs as GI
        grid_setup = DS.load(args.ckpt, device=DEV)
        ckpt_id, entry, cohort = args.ckpt, grid_setup.compat, grid_setup.cohort
        model, report = grid_setup.model, grid_setup.report
        category, item2id = grid_setup.entry["category"], grid_setup.item2id
        root = GI.category_dir(grid_setup.entry)
        inter = json.loads((root / f"{category}.inter.json").read_text())
        id2user = {i: u for u, i in GI.user2id(grid_setup.entry).items()}
        id2item = {i: a for a, i in item2id.items()}
        seqs = {id2user[int(u)]: [id2item[int(x)] for x in s_] for u, s_ in inter.items()}
    else:
        ckpt_id = C.CELLS[args.cell]
        entry = load_runtime()[ckpt_id]
        cohort = load_eval_cohort(entry)
        model, report = diffusion.load(entry, device=DEV)
        category, item2id = C.CATEGORY, load_item2id(C.CATEGORY)
        seqs = json.loads((paths.FROZEN / "data" / "sequences" / "all_item_seqs.json").read_text())
    validation = {"V4_checkpoint_sha256": sha256_file(Path(entry["ckpt_path"])) == entry["ckpt_sha256"]}
    if not validation["V4_checkpoint_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    n, K, W = int(model.n_digit), int(model.codebook_size), int(model.n_embd)
    id2asin = {i: a for a, i in item2id.items()}
    asins = [id2asin[i] for i in range(len(id2asin))]
    item_sids = np.array([cohort.item_to_sid[i] for i in range(len(asins))], dtype=np.int64)
    item_labels, vocab = {}, {}
    for field in C.LABEL_FIELDS:
        item_labels[field], vocab[field] = LB.codes(category, asins, field)
    N_users = min(args.limit, len(cohort)) if args.limit else len(cohort)
    users = cohort.users[:N_users]

    # ---- data and gates -----------------------------------------------------------
    test = C.build_histories(users, seqs, item2id, item_sids, "test", cohort.histories.shape[1])
    fit = C.build_histories(users, seqs, item2id, item_sids, "fit", cohort.histories.shape[1])
    validation["V3_rebuilt_histories_equal_cohort"] = bool(
        np.array_equal(test.sids, cohort.histories[:N_users]) and np.array_equal(test.mask, cohort.history_mask[:N_users])
        and np.array_equal(test.target_sids, cohort.target_sids[:N_users])
        and np.array_equal(test.target_items, cohort.target_item_ids[:N_users]))
    if not validation["V3_rebuilt_histories_equal_cohort"]:
        raise RuntimeError("rebuilt analysis histories differ from the cohort")
    if grid_setup is not None:       # no exp1 archive: the run's own lists (V2a) and the scorer check
        validation["V1_grid_checks"] = DS.check(grid_setup, device=DEV)
        if not validation["V1_grid_checks"]["ok"]:
            raise RuntimeError(f"grid checks failed: {json.dumps(validation, default=str)[:1500]}")
    else:
        validation["V1_exp1_pilot_checks"] = exp1_validate()(
            model, torch.from_numpy(cohort.histories[:8]).to(DEV), torch.from_numpy(cohort.history_mask[:8]).to(DEV),
            cohort.sid_buckets, matched_decode)
    T = C.slot_table(test, item_sids, item_labels)
    F = C.slot_table(fit, item_sids, item_labels)
    fit_val_user = np.array([C.user_bucket(u, 10) == 0 for u in users])
    F_val = fit_val_user[F["user_index"]]
    inputs = {"checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
              "cohort_sha256": cohort.cohort_sha256, "input_sha256": cohort.input_sha256, "n_users": N_users,
              "limited": args.limit, "n_digit": n, "codebook_size": K, "width": W,
              "fit_histories": len(fit), "fit_rows": len(F["hist"]), "fit_val_rows": int(F_val.sum()),
              "test_rows": len(T["hist"]), "label_vocab": {k: len(v) for k, v in vocab.items()}, "seed": C.SEED,
              "enc_batch": ENC_B, "dec_batch": DEC_B, "category": category,
              "device": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "torch": torch.__version__,
              **({k: entry[k] for k in ("grid_key", "grid_state", "dataset")} if grid_setup else {})}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    log(f"{ckpt_id}: users={N_users} fit rows={len(F['hist'])} test rows={len(T['hist'])}; V1 V3 V4 passed")

    Xf, _ = encode_sites(model, fit)                                 # [fit rows, 3, W]
    Xt, out_test = encode_sites(model, test, keep_output=True)        # [test rows, 3, W], [N, S, W]
    validation["V2_encoder_sites_checked"] = True                     # encoder_sites raises otherwise
    log(f"encoded: fit {tuple(Xf.shape)} test {tuple(Xt.shape)}")

    def tt(a, dtype=torch.long):
        return torch.as_tensor(np.asarray(a), device=DEV, dtype=dtype)

    # ---- Part P: probes -------------------------------------------------------------
    probe_rows, correct = [], {}

    def run_probe(pid, site, f_rows, f_y, t_rows, t_y, n_labels, Xfit=None, Xtest=None, subsets=None):
        Xa = Xf[:, SITES.index(site)] if Xfit is None else Xfit
        Xb = Xt[:, SITES.index(site)] if Xtest is None else Xtest
        fr, tr_ = np.flatnonzero(f_rows), np.flatnonzero(t_rows)
        probe, info = LP.fit_selected(Xa[tt(fr)], tt(f_y[fr]), tt(F_val[fr] if Xfit is None else f_val_h[fr], torch.bool))
        for sub_name, sub in (subsets or {"all": np.ones(len(tr_), dtype=bool)}).items():
            rows = tr_[sub]
            if len(rows) == 0:
                continue
            ev = LP.evaluate(probe, Xb[tt(rows)], tt(t_y[rows]), tt(f_y[fr]), n_labels)
            correct[f"{pid}|{site}|{sub_name}"] = (ev.pop("correct").cpu().numpy(), rows)
            probe_rows.append({"probe": pid, "site": site, "subset": sub_name, "lam": info["lam"], **ev})

    for d in range(n):
        for site in SITES:
            run_probe(f"P1_own_d{d}", site, np.ones(len(F["hist"]), bool), F[f"own_d{d}"],
                      np.ones(len(T["hist"]), bool), T[f"own_d{d}"], K)
    for field in ("cat_l2", "main_cat"):
        for site in ("input", "output"):
            run_probe(f"P2_own_{field}", site, F[f"own_{field}"] >= 0, F[f"own_{field}"], T[f"own_{field}"] >= 0,
                      T[f"own_{field}"], len(vocab[field]))
    for site in ("input", "output"):
        run_probe("P3_rec", site, np.ones(len(F["hist"]), bool), F["rec"], np.ones(len(T["hist"]), bool), T["rec"], 7)
    for site in SITES:
        run_probe("P4a_r1_d0", site, ~F["is_r1"] & (F["own_d0"] != F["r1_d0"]), F["r1_d0"],
                  ~T["is_r1"] & (T["own_d0"] != T["r1_d0"]), T["r1_d0"], K)
    for site in ("input", "output"):
        def p4b(S):
            return ~S["is_r1"] & (S["own_cat_l2"] >= 0) & (S["r1_cat_l2"] >= 0) & (S["own_cat_l2"] != S["r1_cat_l2"])
        run_probe("P4b_r1_cat_l2", site, p4b(F), F["r1_cat_l2"], p4b(T), T["r1_cat_l2"], len(vocab["cat_l2"]))

    # P5: one row per history, [site(r1), mean site]
    def history_features(X, S, n_hist, site):
        x = X[:, SITES.index(site)]
        h = tt(S["hist"])
        mean = torch.zeros((n_hist, W), device=DEV).index_add_(0, h, x) / torch.bincount(h, minlength=n_hist)[:, None]
        r1 = torch.zeros((n_hist, W), device=DEV)
        r1[h[tt(S["is_r1"], torch.bool)]] = x[tt(S["is_r1"], torch.bool)]
        return torch.cat([r1, mean], 1)

    f_val_h = fit_val_user[fit.user_index]
    hist_d0 = [set(test.sids[i, :test.lengths[i], 0].tolist()) for i in range(len(test))]
    new_target = np.array([int(test.target_sids[i, 0]) not in hist_d0[i] for i in range(len(test))])
    tgt_cat = item_labels["cat_l2"][test.target_items]
    fit_tgt_cat = item_labels["cat_l2"][fit.target_items]
    for site in ("input", "output"):
        Hf, Ht = history_features(Xf, F, len(fit), site), history_features(Xt, T, len(test), site)
        run_probe("P5_tgt_d0", site, np.ones(len(fit), bool), fit.target_sids[:, 0], np.ones(len(test), bool),
                  test.target_sids[:, 0], K, Xfit=Hf, Xtest=Ht, subsets={"all": np.ones(len(test), bool), "new": new_target})
        run_probe("P5_tgt_cat_l2", site, fit_tgt_cat >= 0, fit_tgt_cat, tgt_cat >= 0, tgt_cat, len(vocab["cat_l2"]),
                  Xfit=Hf, Xtest=Ht, subsets={"all": np.ones(int((tgt_cat >= 0).sum()), bool),
                                              "new": new_target[tgt_cat >= 0]})
    pd.DataFrame(probe_rows).to_csv(args.out / "probes.csv", index=False)
    np.savez_compressed(args.out / "test_slots.npz", **{k: T[k] for k in ("hist", "slot", "L", "r", "rec", "is_r1", "item")})
    np.savez_compressed(args.out / "probe_correct.npz",
                        **{f"{k}|correct": v[0] for k, v in correct.items()},
                        **{f"{k}|rows": v[1] for k, v in correct.items()})
    log(f"probes done ({len(probe_rows)} rows)")

    # ---- Part S: SAE ----------------------------------------------------------------
    cfg = SaeConfig(d_in=W, n_latents=2048, k=16, k_aux=128, aux_coef=1 / 32, dead_after_tokens=50_000,
                    lr=5e-4, batch=1024, epochs=60, seed=C.SEED)
    sae = TopKSAE(cfg, device=DEV)
    fit_out = Xf[:, 2].contiguous()
    history = sae.fit(fit_out[tt(~F_val, torch.bool)], log=lambda m: None, X_val=fit_out[tt(F_val, torch.bool)])
    sae.save(args.out / "sae")
    (args.out / "sae_history.json").write_text(json.dumps(history))
    test_out = Xt[:, 2].contiguous()

    def sae_encode(X):
        idx, val = [], []
        with torch.no_grad():
            for s in range(0, X.shape[0], 16384):
                i, v = sae.encode(X[s:s + 16384])
                idx.append(i), val.append(v)
        return torch.cat(idx), torch.cat(val)

    fi, fv = sae_encode(fit_out)
    ti, tv = sae_encode(test_out)
    fi_np, fv_np, ti_np, tv_np = (a.cpu().numpy() for a in (fi, fv, ti, tv))
    test_fires = np.bincount(ti_np[tv_np > 0], minlength=cfg.n_latents)
    fidelity = {"fvu_test": sae.fvu(test_out), "fvu_fit_val": history[-1].get("fvu_val") if history else None,
                "fvu_fit_train_last": history[-1]["fvu"] if history else None,
                "dead_frac_test": float((test_fires == 0).mean()), "steps": history[-1]["step"] if history else 0}
    families = {f"own_d{d}": (np.ones(len(F["hist"]), bool), F[f"own_d{d}"], K) for d in range(n)}
    families.update({"own_cat_l2": (np.ones(len(F["hist"]), bool), F["own_cat_l2"], len(vocab["cat_l2"])),
                     "rec": (np.ones(len(F["hist"]), bool), F["rec"], 7),
                     "r1_d0": (~F["is_r1"], F["r1_d0"], K),
                     "tgt_d0": (np.ones(len(F["hist"]), bool), F["tgt_d0"], K)})
    per, assigned, fires = C.census(fi_np, fv_np, cfg.n_latents, families)
    census = pd.DataFrame({"latent": np.arange(cfg.n_latents), "assigned": assigned, "fires_fit": fires,
                           "fires_test": test_fires})
    for fam, stats in per.items():
        for key in ("c_star", "kappa", "support"):
            census[f"{fam}|{key}"] = stats[key]
    census.to_csv(args.out / "census.csv", index=False)
    log(f"SAE: fvu_test={fidelity['fvu_test']:.4f}; families {census['assigned'].value_counts().to_dict()}")

    # ---- conditions scored at the fully masked state --------------------------------
    rows_of = {}                                                      # condition -> (test row ids, values [r, W])
    all_rows = np.arange(len(T["hist"]))
    with torch.no_grad():
        rows_of["noop"] = (all_rows, sae.ablate(test_out, [[] for _ in all_rows])[0])
        rows_of["sae"] = (all_rows, sae.reconstruct(test_out))
        rows_of["mean"] = (all_rows, fit_out.mean(0, keepdim=True).expand(len(all_rows), -1))

    # CS1 / CS2 latent choices
    L_user = test.lengths
    c_user = test.sids[np.arange(len(test)), L_user - 1, 0]
    r1_fam = assigned == "r1_d0"
    own_fam = (assigned == "own_d0")
    cstar_r1, cstar_own = per["r1_d0"]["c_star"], per["own_d0"]["c_star"]
    cs = {"cs1": [], "cs1_ctrl": [], "cs2": [], "cs2_ctrl": []}
    counts = {k: np.zeros(len(test), dtype=np.int64) for k in cs}
    for r in all_rows:
        h = T["hist"][r]
        if L_user[h] < 2:
            continue
        c = c_user[h]
        idx_r, val_r = ti_np[r], tv_np[r]
        act = val_r > 0
        if T["is_r1"][r]:
            chosen = idx_r[act & own_fam[idx_r] & (cstar_own[idx_r] == c)]
            excluded = np.flatnonzero(own_fam & (cstar_own == c))
            keys = ("cs2", "cs2_ctrl")
        else:
            chosen = idx_r[act & r1_fam[idx_r] & (cstar_r1[idx_r] == c)]
            excluded = np.flatnonzero(r1_fam)
            keys = ("cs1", "cs1_ctrl")
        ctrl = C.matched_control(idx_r, val_r, chosen, excluded)
        cs[keys[0]].append((r, chosen)), cs[keys[1]].append((r, ctrl))
        counts[keys[0]][h] += len(chosen)
        counts[keys[1]][h] += len(ctrl)
    with torch.no_grad():
        for key, items in cs.items():
            rr = np.array([r for r, _ in items], dtype=np.int64)
            if len(rr):
                rows_of[key] = (rr, sae.ablate(test_out[tt(rr)], [lat for _, lat in items])[0])
            else:
                rows_of[key] = (rr, test_out[:0])

    # ---- Part N: description autoencoder ---------------------------------------------
    def words_of(S):
        return np.stack([S["rec"], C.with_unknown(S["own_cat_l1"], len(vocab["cat_l1"])),
                         C.with_unknown(S["own_cat_l2"], len(vocab["cat_l2"])),
                         C.with_unknown(S["own_main_cat"], len(vocab["main_cat"])),
                         C.with_unknown(S["own_price_decile"], len(vocab["price_decile"])), S["own_d0"],
                         C.with_unknown(S["r1_cat_l2"], len(vocab["cat_l2"])), S["r1_d0"]], 1)

    sizes = (7, len(vocab["cat_l1"]) + 1, len(vocab["cat_l2"]) + 1, len(vocab["main_cat"]) + 1,
             len(vocab["price_decile"]) + 1, K, len(vocab["cat_l2"]) + 1, K)
    true_f, true_t = words_of(F), words_of(T)
    pred_f, pred_t, verbalizer = np.zeros_like(true_f), np.zeros_like(true_t), {}
    for i, field in enumerate(C.DESCRIPTION_FIELDS):
        probe, info = LP.fit_selected(fit_out, tt(true_f[:, i]), tt(F_val, torch.bool))
        with torch.no_grad():
            pred_f[:, i] = probe.predict(fit_out).cpu().numpy()
            pred_t[:, i] = probe.predict(test_out).cpu().numpy()
        verbalizer[field] = {"lam": info["lam"], "acc_test": float((pred_t[:, i] == true_t[:, i]).mean()),
                             "majority_test": float((true_t[:, i] == np.bincount(true_f[:, i]).argmax()).mean())}
    ybar = fit_out.mean(0)

    def ridge(words_fit):
        X = torch.from_numpy(C.one_hot_design(words_fit, sizes)).to(DEV)
        Y = (fit_out - ybar).double()
        val = tt(F_val, torch.bool)

        def solve(Xa, Ya, alpha):
            G = (Xa.T @ Xa).double()
            G += alpha * torch.eye(G.shape[0], device=DEV, dtype=torch.float64)
            return torch.linalg.solve(G, Xa.T.double() @ Ya)

        scores = {}
        for a in ALPHAS:
            B = solve(X[~val], Y[~val], a)
            scores[a] = float(((X[val].double() @ B - Y[val]) ** 2).sum(1).mean())
        best = min(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return solve(X, Y, best), best, scores

    def reconstruct(B, words):
        X = torch.from_numpy(C.one_hot_design(words, sizes)).to(DEV).double()
        return (X @ B).float() + ybar

    B_desc, a_desc, s_desc = ridge(pred_f)
    B_orc, a_orc, s_orc = ridge(true_f)

    def fvu(xhat):
        x = test_out
        return float(((x - xhat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

    rows_of["desc"] = (all_rows, reconstruct(B_desc, pred_t))
    rows_of["oracle"] = (all_rows, reconstruct(B_orc, true_t))
    nla = {"fields": list(C.DESCRIPTION_FIELDS), "sizes": list(sizes), "verbalizer": verbalizer,
           "alpha_desc": a_desc, "alpha_oracle": a_orc, "ridge_val_mse_desc": s_desc, "ridge_val_mse_oracle": s_orc,
           "fvu_desc": fvu(rows_of["desc"][1]), "fvu_oracle": fvu(rows_of["oracle"][1]), "fvu_sae": fidelity["fvu_test"],
           "fvu_mean": fvu(rows_of["mean"][1]), "field_drop_fvu": {}}
    for i, field in enumerate(C.DESCRIPTION_FIELDS):
        dropped = pred_t.copy()
        dropped[:, i] = np.bincount(pred_f[:, i]).argmax()
        rows_of[f"drop_{field}"] = (all_rows, reconstruct(B_desc, dropped))
        nla["field_drop_fvu"][field] = fvu(rows_of[f"drop_{field}"][1])
    log(f"NLA: fvu desc={nla['fvu_desc']:.3f} oracle={nla['fvu_oracle']:.3f} sae={nla['fvu_sae']:.3f}")

    # ---- score every condition --------------------------------------------------------
    blocked = np.where(L_user >= 2, L_user - 1, -1)
    _, enc_block = encode_sites(model, test, keep_output=True, blocked=blocked)
    _, enc_noop_mask = encode_sites(model, test, keep_output=True, blocked=np.full(len(test), -1))
    validation["V2b_padding_only_encoder_mask_is_noop"] = bool(torch.equal(enc_noop_mask, out_test))
    if not validation["V2b_padding_only_encoder_mask_is_noop"]:
        raise RuntimeError("the padding-only encoder mask changed the encoder output")
    del enc_noop_mask
    clean = score_full(model, out_test)
    hist_of_row, slot_of_row = tt(T["hist"]), tt(T["slot"])
    tgt0, r1c = tt(test.target_sids[:, 0]), tt(c_user)
    ar = torch.arange(len(test), device=DEV)
    records = []

    def record(name, lp):
        kl = (clean.exp() * (clean - lp)).sum(-1)                     # [N, n]
        rec = {"user_index": np.arange(len(test)), "cond": name, "copy": lp[ar, 0, r1c].cpu().numpy(),
               "gold": lp[ar, 0, tgt0].cpu().numpy(), "top1_d0": lp[:, 0].argmax(-1).cpu().numpy()}
        for d in range(n):
            rec[f"kl_{d}"] = kl[:, d].cpu().numpy()
        if name in counts:
            rec["n_ablated"] = counts[name]
        records.append(pd.DataFrame(rec))

    record("clean", clean)
    for name, (rr, values) in rows_of.items():
        enc = out_test.clone()
        if len(rr):
            enc[hist_of_row[tt(rr)], slot_of_row[tt(rr)]] = values.to(enc.dtype)
        lp = score_full(model, enc)
        if name == "noop":
            validation["V5_empty_ablation_is_bit_exact"] = bool(torch.equal(lp, clean))
            if not validation["V5_empty_ablation_is_bit_exact"]:
                raise RuntimeError("an empty SAE ablation changed the scores")
        record(name, lp)
    record("enc_block", score_full(model, enc_block))
    cond = pd.concat(records, ignore_index=True)
    cond.to_csv(args.out / "conditions.csv.gz", index=False)
    r1_cat = item_labels["cat_l2"][test.items[np.arange(len(test)), L_user - 1]]
    users_tab = pd.DataFrame({"user_index": np.arange(len(test)), "user": list(users), "L": L_user,
                              "r1_d0": c_user, "tgt_d0": test.target_sids[:, 0], "new_target": new_target,
                              "copy_rule_d0": c_user == test.target_sids[:, 0],
                              "copy_rule_cat_l2": (r1_cat >= 0) & (tgt_cat >= 0) & (r1_cat == tgt_cat),
                              "tgt_cat_known": tgt_cat >= 0,
                              "model_top1_d0_correct": (clean[:, 0].argmax(-1) == tgt0).cpu().numpy()})
    users_tab.to_csv(args.out / "users.csv", index=False)

    # ---- examples -------------------------------------------------------------------------
    meta = META.load(C.CATEGORY)
    names = {"rec": list(C.REC_LABELS), "cat_l1": vocab["cat_l1"], "cat_l2": vocab["cat_l2"],
             "main_cat": vocab["main_cat"], "price_decile": vocab["price_decile"],
             "own_d0": [str(c) for c in range(K)], "r1_cat_l2": vocab["cat_l2"], "r1_d0": [str(c) for c in range(K)]}
    rng = np.random.default_rng(C.SEED)
    pool = np.flatnonzero((L_user >= 3) & (L_user <= 8))
    examples = []
    for h in sorted(rng.choice(pool, size=min(8, len(pool)), replace=False).tolist()):
        slots = []
        for r in np.flatnonzero(T["hist"] == h):
            item = int(T["item"][r])
            top = [(int(f), str(assigned[f]), float(v)) for f, v in zip(ti_np[r], tv_np[r]) if v > 0][:6]
            slots.append({"slot": int(T["slot"][r]), "recency": int(T["r"][r]), "item": item,
                          "title": meta[id2asin[item]].short_title if id2asin[item] in meta else "",
                          "sid": [int(x) for x in item_sids[item]],
                          "true": C.render(dict(zip(C.DESCRIPTION_FIELDS, true_t[r])), names),
                          "description": C.render(dict(zip(C.DESCRIPTION_FIELDS, pred_t[r])), names),
                          "words_correct": int((pred_t[r] == true_t[r]).sum()), "top_latents": top})
        target = int(test.target_items[h])
        examples.append({"user": users[h], "L": int(L_user[h]), "slots": slots,
                         "target": {"item": target, "title": meta[id2asin[target]].short_title if id2asin[target] in meta else "",
                                    "sid": [int(x) for x in item_sids[target]]},
                         "model_top_d0": int(clean[h, 0].argmax())})
    (args.out / "nla.json").write_text(json.dumps(nla, indent=1))
    (args.out / "nla_examples.json").write_text(json.dumps(examples, indent=1))
    (args.out / "sae_fidelity.json").write_text(json.dumps(fidelity, indent=1))
    validation.update({"model_report_ok": report["ok"], "seconds": round(time.time() - t0, 1)})
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
