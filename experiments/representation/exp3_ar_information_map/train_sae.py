#!/usr/bin/env python
"""Stage S: fit TopK SAEs on the self-forced train capture; encode the test capture.

exp2's configuration, unchanged (protocol.md), at layers 0, 2, ..., 26, 27.
Test tokens are only encoded, never fitted.

Outputs (under --out):
    sae/L<layer>/{sae.safetensors, config.json, history.json}
    test_latents_L<layer>.safetensors   idx (N, k) int32, val (N, k) float32, aligned with
                                        the test store's rows.parquet
    fidelity.json                       FVU per role (train holdout, test), dead fractions

    python train_sae.py --cell 0 --train-capture <dir> --test-capture <dir> --out <new dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "exp2_ar_sae"))

from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.hooks.store import StoreReader                              # noqa: E402
from sidlens.provenance import manifest as manifest_mod                  # noqa: E402
from sidlens.sae.topk import SaeConfig, TopKSAE                          # noqa: E402
from train import CFG, role_groups, user_bucket                          # noqa: E402  (exp2, unchanged)

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
LAYERS = tuple(range(0, 27, 2)) + (27,)


def open_store(capture: Path, ckpt: str, split: str, layers) -> StoreReader:
    """Verified store of this checkpoint, split and template, captured self-forced."""
    st = StoreReader(Path(capture) / "store", verify=True)
    m = st.meta
    want = manifest_mod.load()["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    if (m["model"], m["checkpoint_sha256"], m["split"], m["template"]) != (f"ar/{ckpt}", want, split, "eval"):
        raise RuntimeError(f"store {capture}: {m['model']} {m['split']} {m['template']} is not {ckpt} {split} eval")
    if m.get("target") != "self-greedy":
        raise RuntimeError(f"store {capture} is not a self-greedy capture (target={m.get('target')})")
    missing = [f"model.layers.{L}" for L in layers if f"model.layers.{L}" not in st.sites]
    if missing:
        raise KeyError(f"store {capture} lacks {missing}")
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--train-capture", type=Path, required=True, help="ar_capture --out dir (holds store/)")
    ap.add_argument("--test-capture", type=Path, required=True)
    ap.add_argument("--layers", default=",".join(map(str, LAYERS)))
    ap.add_argument("--epochs", type=int, default=None, help="pilot only")
    ap.add_argument("--max-train-tokens", type=int, default=None, help="pilot only")
    ap.add_argument("--device", default="cuda")
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

    tr = open_store(args.train_capture, ckpt, "train", layers)
    te = open_store(args.test_capture, ckpt, "test", layers)
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
    inputs = {"checkpoint": f"ar/{ckpt}", "variant": variant, "train_capture": str(args.train_capture),
              "test_capture": str(args.test_capture), "layers": layers, "config": cfg.__dict__,
              "n_train_tokens": int((~hold).sum()), "n_holdout_tokens": int(hold.sum()),
              "n_test_tokens": len(te.rows), "pilot_epochs": args.epochs,
              "pilot_max_train_tokens": args.max_train_tokens,
              "device": torch.cuda.get_device_name() if dev.startswith("cuda") else "cpu",
              "torch": torch.__version__}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))

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
        save_file({"idx": idx.contiguous(), "val": val.contiguous()},
                  str(args.out / f"test_latents_L{L}.safetensors"))
        fired = torch.zeros(cfg.n_latents, dtype=torch.bool)
        fired[idx[val > 0].unique().long()] = True
        rec = {"fvu_train_holdout": sae.fvu(Xho), "n_train_tokens_used": int(len(Xtr)),
               "fvu_train_holdout_by_role": {g: sae.fvu(Xho[torch.as_tensor(ix, device=dev)])
                                             for g, ix in ho_groups.items() if len(ix)},
               "dead_frac_test": float(1 - fired.float().mean()),
               "fvu_test": {g: sae.fvu(Xte[torch.as_tensor(ix, device=dev)]) for g, ix in groups.items()
                            if len(ix)}}
        fidelity[site] = rec
        log(f"[L{L}] {json.dumps(rec)}")
        (args.out / "fidelity.json.partial").write_text(json.dumps(fidelity, indent=1))
        del Xtr, Xho, Xte, sae
        torch.cuda.empty_cache()
    (args.out / "fidelity.json").write_text(json.dumps(fidelity, indent=1))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
