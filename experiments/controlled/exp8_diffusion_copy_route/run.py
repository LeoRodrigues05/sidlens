#!/usr/bin/env python
"""DiffGRM copy route: block the encoder spread of the most recent item, its direct decoder read, or both.

Protocol: `protocol.md` beside this file (declared after exp7, before any exp8
forward). Every encoder forward carries an explicit [B, 1, S, S] mask (the
padding mask, plus any blocked (other slot -> k*) pairs) and every decoder
forward a cross-attention mask, so all conditions share one code path.
Fixed encoder/decoder batch shapes; fp32.

    python run.py --cell 0 --out <new dir> [--limit 64] [--device cpu]
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

from sidlens import paths                                                # noqa: E402
from sidlens.analysis.matched_decode import decode as matched_decode     # noqa: E402
from sidlens.data.diffusion_eval import load_eval_cohort                 # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

CELLS = ("diff-next1-rqkmeans-3cb-128", "diff-next1-rqvae-4cb-128")
ENC_B, DEC_B, USERS_PER_CHUNK = 512, 2048, 256
DEV = "cuda"


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def encode_masked(model, hist, hmask, blocked):
    """Encoder states with per-row encoder knockouts; blocked[i] = key slot or None."""
    out = []
    for s in range(0, len(hist), ENC_B):
        h, m, b = hist[s:s + ENC_B], hmask[s:s + ENC_B], blocked[s:s + ENC_B]
        real = len(h)
        if real < ENC_B:
            h = np.concatenate([h, np.repeat(h[:1], ENC_B - real, 0)])
            m = np.concatenate([m, np.repeat(m[:1], ENC_B - real, 0)])
            b = list(b) + [None] * (ENC_B - real)
        em = torch.from_numpy(DI.encoder_block_mask(m, [(r, k) for r, k in enumerate(b) if k is not None])).to(DEV)
        with DI.encoder_attention_masks(model, em):
            e = diffusion.encode(model, torch.from_numpy(h).to(DEV), torch.from_numpy(m).to(DEV))
        out.append(e[:real])
    return torch.cat(out)


def score_fixed(model, enc, cache, jobs, n, n_layers, S, enc_mask):
    outs = []
    for s in range(0, len(jobs), DEC_B):
        chunk = jobs[s:s + DEC_B]
        real = len(chunk)
        chunk = chunk + [chunk[0]] * (DEC_B - real)
        rows = torch.as_tensor([j["enc_row"] for j in chunk], device=DEV)
        codes = torch.as_tensor(np.stack([j["codes"] for j in chunk]), device=DEV)
        masked = torch.as_tensor(np.stack([j["masked"] for j in chunk]), device=DEV)
        edges = [(r, range(n_layers), list(range(n)), [j["xslot"]], j["enc_row"])
                 for r, j in enumerate(chunk[:real]) if j["xslot"] is not None]
        xm = torch.from_numpy(DI.knockout_mask(n_layers, DEC_B, n, S, edges, enc_mask)).to(DEV)
        with torch.inference_mode():
            lp = DI.score_states(model, enc, cache, rows, codes, masked, xmask=xm)
        outs.append(lp[:real].cpu())
    return torch.cat(outs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    global DEV
    DEV = args.device
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    ckpt_id = CELLS[args.cell]
    entry = load_runtime()[ckpt_id]
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    cohort = load_eval_cohort(entry)
    model, report = diffusion.load(entry, device=DEV)
    n, n_layers, S = int(model.n_digit), len(model.decoder_blocks), cohort.histories.shape[1]
    N = min(args.limit, len(cohort)) if args.limit else len(cohort)

    validation = {"model_report_ok": report["ok"], "V1_exp1_pilot_checks": exp1_validate()(
        model, torch.from_numpy(cohort.histories[:8]).to(DEV), torch.from_numpy(cohort.history_mask[:8]).to(DEV),
        cohort.sid_buckets, matched_decode)}
    h8, m8 = cohort.histories[:8], cohort.history_mask[:8]
    plain = diffusion.encode(model, torch.from_numpy(h8).to(DEV), torch.from_numpy(m8).to(DEV))
    viahook = encode_masked(model, h8, m8, [None] * 8)
    validation["V2_noop_encoder_mask"] = bool(torch.equal(plain, viahook))
    cache8 = DI.project_cross_cache(model, plain)
    z, allm = torch.zeros((8, n), dtype=torch.long, device=DEV), torch.ones((8, n), dtype=torch.bool, device=DEV)
    with torch.inference_mode():
        a = DI.score_states(model, plain, cache8, torch.arange(8, device=DEV), z, allm)
        b = DI.score_states(model, plain, cache8, torch.arange(8, device=DEV), z, allm,
                            xmask=torch.ones((n_layers, 8, n, S), device=DEV))
    validation["V2_noop_cross_mask"] = bool(torch.equal(a, b))
    if not (validation["V2_noop_encoder_mask"] and validation["V2_noop_cross_mask"]):
        raise RuntimeError(f"no-op masks changed the model: {validation}")
    (args.out / "inputs.json").write_text(json.dumps({
        "checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
        "cohort_sha256": cohort.cohort_sha256, "n_users": N, "limited": args.limit, "n_digit": n,
        "device": DEV, "torch": torch.__version__}, indent=1))
    print(f"inputs ok: {ckpt_id} users={N}; V1 passed; V2 {validation['V2_noop_encoder_mask']}, "
          f"{validation['V2_noop_cross_mask']}", flush=True)

    recs = []
    t0 = time.time()
    for u0 in range(0, N, USERS_PER_CHUNK):
        users = list(range(u0, min(N, u0 + USERS_PER_CHUNK)))
        hist, hmask, blocked, meta = [], [], [], []
        for i in users:
            L = int(cohort.history_lengths[i])
            k = L - 1
            tgt = cohort.target_sids[i]
            j = next((x for x in range(L - 2, -1, -1) if cohort.histories[i, x, 0] != tgt[0]), None)
            for kind, blk in (("base", None), ("base_dup", None), ("eK", k), ("eJ", j)):
                if kind == "eJ" and j is None:
                    continue
                hist.append(cohort.histories[i]); hmask.append(cohort.history_mask[i]); blocked.append(blk)
                meta.append({"i": i, "kind": kind, "k": k, "j": j})
        hist, hmask = np.stack(hist), np.stack(hmask)
        enc = encode_masked(model, hist, hmask, blocked)
        cache = DI.project_cross_cache(model, enc)
        jobs, jmeta = [], []
        for e, mt in enumerate(meta):
            tgt = cohort.target_sids[mt["i"]].astype(np.int64)
            plan = {"base": [("base", None), ("dec", mt["k"])], "base_dup": [("base_dup", None)],
                    "eK": [("enc", None), ("both", mt["k"])], "eJ": [("both_ctrl", mt["j"])]}[mt["kind"]]
            for d in range(n):
                masked = np.arange(n) >= d
                for cond, xs in plan:
                    jobs.append({"enc_row": e, "codes": np.where(masked, 0, tgt), "masked": masked, "xslot": xs})
                    jmeta.append({**mt, "d": d, "cond": cond})
        lp = score_fixed(model, enc, cache, jobs, n, n_layers, S, hmask)
        tg = torch.as_tensor(cohort.target_sids[[m["i"] for m in jmeta]], dtype=torch.long)
        gold = lp.gather(2, tg[:, :, None])[:, :, 0]
        rank = (lp > gold[:, :, None]).sum(-1)
        for x, mt in enumerate(jmeta):
            i, d = mt["i"], mt["d"]
            recs.append({"user": cohort.users[i], "user_row": i, "d": d, "digit": d, "cond": mt["cond"],
                         "state": "full" if d == 0 else "pref", "logp": float(gold[x, d]), "rank": int(rank[x, d]),
                         "has_j": mt["j"] is not None})
        print(f"{users[-1] + 1}/{N} users, {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(recs)
    b1 = df[df.cond == "base"].set_index(["user_row", "d"]).logp
    b2 = df[df.cond == "base_dup"].set_index(["user_row", "d"]).logp
    validation["V3_batch_invariance"] = {"compared": len(b1), "max_abs_diff_logp": float((b1 - b2).abs().max()),
                                         "exact": bool((b1 == b2.reindex(b1.index)).all())}
    df[df.cond != "base_dup"].to_parquet(args.out / "scores.parquet", index=False)
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    print(f"done: V3 {validation['V3_batch_invariance']}; {validation['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
