#!/usr/bin/env python
"""N2 (f): induction scores of every head of the base Qwen2.5-1.5B-Instruct (computed once).

A sequence of T random tokens is repeated once ([x, x], length 2T). At every
position t of the second copy, an induction head attends to the token after
the previous occurrence of the current token, i.e. position t - T + 1. A
head's induction score is its mean attention probability there (Olsson et
al., 2022); the prefix-matching score (attention to t - T, the previous
occurrence itself) is kept beside it. Observation only (`hooks.attention_probs`).

The base model path is the one the grid's AR runs were fine-tuned from (the
coauthor's cluster has it; nothing is downloaded: local files only).

    python induction.py --base-model /path/to/Qwen2.5-1.5B-Instruct --out <new dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.hooks.attention_probs import attention_probs                 # noqa: E402

SEED, N_SEQ, T, LO, HI = 20261021, 64, 64, 1000, 30000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(args.base_model), local_files_only=True,
                                                 dtype=torch.float32).to(args.device).eval()
    nL, nH = model.config.num_hidden_layers, model.config.num_attention_heads
    rng = np.random.default_rng(SEED)
    seqs = rng.integers(LO, HI, size=(N_SEQ, T))
    ind = np.zeros((nL, nH))
    pm = np.zeros((nL, nH))
    count = 0
    for s in range(0, N_SEQ, args.batch):
        x = torch.as_tensor(np.concatenate([seqs[s:s + args.batch]] * 2, axis=1), device=args.device)
        B = x.shape[0]
        qs = list(range(T + 1, 2 * T))
        pairs_r = [b for b in range(B) for _ in qs]
        pairs_q = [q for _ in range(B) for q in qs]
        with attention_probs(model, list(range(nL)), pairs_r, pairs_q) as obs, torch.no_grad():
            model(input_ids=x, attention_mask=torch.ones_like(x))
        for L in range(nL):
            pr = obs[L].numpy()                                # (P, H, 2T)
            ind[L] += sum(pr[p, :, q - T + 1] for p, q in enumerate(pairs_q))
            pm[L] += sum(pr[p, :, q - T] for p, q in enumerate(pairs_q))
        count += len(pairs_q)
    ind, pm = ind / count, pm / count
    np.savez_compressed(args.out / "induction.npz", induction=ind, prefix_matching=pm)
    w = sorted(args.base_model.glob("*.safetensors"))
    (args.out / "inputs.json").write_text(json.dumps({
        "base_model": str(args.base_model), "weights_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                               for p in w},
        "seed": SEED, "n_sequences": N_SEQ, "half_length": T, "token_range": [LO, HI],
        "top10_induction_heads": [[int(a), int(b), float(ind[a, b])] for a, b in
                                  zip(*np.unravel_index(np.argsort(-ind, axis=None)[:10], ind.shape))]}, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
