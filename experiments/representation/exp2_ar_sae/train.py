#!/usr/bin/env python
"""Fit TopK SAEs on train-split AR captures; encode the test captures (F1).

Protocol: `protocol.md` beside this file. One invocation = one cell. Every
hyperparameter is fixed there. Test tokens are only encoded, never fitted:
they are the evaluation set of every later analysis.

Outputs (under --out):
    sae/L<layer>/{sae.safetensors, config.json, history.json}
    test_latents_L<layer>.safetensors   idx (N, k) int32, val (N, k) float32, aligned
                                        with the test store's rows.parquet
    fidelity.json                       FVU per role (train-holdout, test), dead fractions

    python train.py --cell 0 --out <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.hooks.store import StoreReader                              # noqa: E402
from sidlens.provenance import manifest as manifest_mod                  # noqa: E402
from sidlens.sae.topk import SaeConfig, TopKSAE                          # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
TRAIN_STORE = {"next-item_best": "287160", "oneoff_rqvae4cb128": "287161"}
TEST_STORE = {"next-item_best": "281053", "oneoff_rqvae4cb128": "281054"}
LAYERS = (12, 16, 20, 24)
SEED = 20260930
CFG = dict(n_latents=12288, k=32, k_aux=512, aux_coef=1.0 / 32, dead_after_tokens=200_000,
           lr=2e-4, batch=4096, epochs=20, seed=SEED)


def user_bucket(user: str, mod: int) -> int:
    return int(hashlib.sha256(f"{SEED}|{user}".encode()).hexdigest(), 16) % mod


def open_store(job: str, ckpt: str, split: str) -> StoreReader:
    st = StoreReader(paths.DERIVED / "ar_capture" / job / "capture" / "store", verify=True)
    m = st.meta
    want = manifest_mod.load()["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    if (m["model"], m["checkpoint_sha256"], m["split"], m["template"]) != (f"ar/{ckpt}", want, split, "eval"):
        raise RuntimeError(f"store {job}: {m['model']} {m['split']} {m['template']} is not {ckpt} {split} eval")
    missing = [f"model.layers.{L}" for L in LAYERS if f"model.layers.{L}" not in st.sites]
    if missing:
        raise KeyError(f"store {job} lacks {missing}")
    return st


def role_groups(rows) -> dict[str, np.ndarray]:
    """Row indices per evaluation role; readout0 = the last response_header token of each example."""
    last = rows[rows.role == "response_header"].groupby("example_id").pos.transform("max")
    hdr = rows.role == "response_header"
    r0 = np.zeros(len(rows), bool)
    r0[rows.index[hdr][rows[hdr].pos.to_numpy() == last.to_numpy()]] = True
    return {"all": np.arange(len(rows)), "readout0": np.flatnonzero(r0),
            "header_other": np.flatnonzero(hdr.to_numpy() & ~r0),
            "hist_sid": np.flatnonzero((rows.role == "hist_sid").to_numpy()),
            "target_sid": np.flatnonzero((rows.role == "target_sid").to_numpy())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--epochs", type=int, default=None, help="pilot only")
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train-tokens", type=int, default=None, help="pilot only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ckpt = CELLS[args.cell]
    layers = tuple(int(x) for x in args.layers.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "fidelity.json").exists():
        raise FileExistsError(f"{args.out} already holds results")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = args.device
    t0 = time.time()
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)       # noqa: E731

    tr, te = open_store(TRAIN_STORE[ckpt], ckpt, "train"), open_store(TEST_STORE[ckpt], ckpt, "test")
    variant = tr.meta["variant"]
    users = {e.example_id: e.user_id for e in P.load_examples(variant, "next-item", "train")}
    tr_user = tr.rows.example_id.map(users)
    if tr_user.isna().any():
        raise ValueError("train store rows whose example is not in the train CSV")
    hold = np.array([user_bucket(u, 20) == 0 for u in tr_user])
    groups = role_groups(te.rows)
    tr_groups = role_groups(tr.rows)
    ho_pos = np.full(len(tr.rows), -1)
    ho_pos[np.flatnonzero(hold)] = np.arange(int(hold.sum()))
    ho_groups = {g: ho_pos[ix][ho_pos[ix] >= 0] for g, ix in tr_groups.items()}
    cfg = SaeConfig(d_in=int(tr.manifest["sites"][f"model.layers.{layers[0]}"]["width"]),
                    **{**CFG, **({"epochs": args.epochs} if args.epochs else {})})
    inputs = {"checkpoint": f"ar/{ckpt}", "variant": variant, "train_store": TRAIN_STORE[ckpt],
              "test_store": TEST_STORE[ckpt], "layers": layers, "config": cfg.__dict__,
              "n_train_tokens": int((~hold).sum()), "n_holdout_tokens": int(hold.sum()),
              "n_train_users": int(tr_user[~hold].nunique()), "n_holdout_users": int(tr_user[hold].nunique()),
              "n_test_tokens": len(te.rows), "test_role_counts": {k: int(len(v)) for k, v in groups.items()},
              "pilot_epochs": args.epochs, "pilot_max_train_tokens": args.max_train_tokens,
              "device": torch.cuda.get_device_name() if dev.startswith("cuda") else "cpu", "torch": torch.__version__}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    log(f"inputs: {json.dumps({k: inputs[k] for k in ('n_train_tokens', 'n_holdout_tokens', 'n_test_tokens')})}")

    fidelity = {}
    from safetensors.torch import save_file
    for L in layers:
        site = f"model.layers.{L}"
        X = tr.load(site)
        tr_ix = np.flatnonzero(~hold)[:args.max_train_tokens] if args.max_train_tokens else np.flatnonzero(~hold)
        Xtr = X[torch.as_tensor(tr_ix)].to(dev)
        Xho = X[torch.as_tensor(np.flatnonzero(hold))].to(dev)
        del X
        sae = TopKSAE(cfg, device=dev)
        hist = sae.fit(Xtr, log=lambda m: log(f"[L{L}] {m}"), X_val=Xho[:50_000])
        sae.save(args.out / "sae" / f"L{L}")
        (args.out / "sae" / f"L{L}" / "history.json").write_text(json.dumps(hist, indent=1))
        Xte = te.load(site).to(dev)
        idx_all, val_all = [], []
        with torch.no_grad():
            for b in range(0, len(Xte), 16384):
                i, v = sae.encode(Xte[b:b + 16384])
                idx_all.append(i.int().cpu())
                val_all.append(v.float().cpu())
        idx, val = torch.cat(idx_all), torch.cat(val_all)
        save_file({"idx": idx.contiguous(), "val": val.contiguous()}, str(args.out / f"test_latents_L{L}.safetensors"))
        fired = torch.zeros(cfg.n_latents, dtype=torch.bool)
        fired[idx[val > 0].unique().long()] = True
        rec = {"fvu_train_holdout": sae.fvu(Xho), "fvu_train": sae.fvu(Xtr[:200_000]),
               "n_train_tokens_used": int(len(Xtr)),
               "fvu_train_holdout_by_role": {g: sae.fvu(Xho[torch.as_tensor(ix, device=dev)])
                                             for g, ix in ho_groups.items() if len(ix)},
               "dead_frac_test": float(1 - fired.float().mean()),
               "fvu_test": {g: sae.fvu(Xte[torch.as_tensor(ix, device=dev)]) for g, ix in groups.items()
                            if len(ix)}}
        fidelity[site] = rec
        log(f"[L{L}] {json.dumps(rec)}")
        del Xtr, Xho, Xte, sae
        torch.cuda.empty_cache()
    (args.out / "fidelity.json").write_text(json.dumps(fidelity, indent=1))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
