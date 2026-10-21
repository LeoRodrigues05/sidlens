#!/usr/bin/env python
"""controlled/exp12: paired-run activation patching in DiffGRM (history contrast H, state contrast S).

Protocol: `protocol.md` beside this file, declared before any forward pass of
this experiment. One invocation = one checkpoint. Two runs of the same model
differ in one input; every site is patched from one run into the other in both
directions, and the scored digit's distribution is compared with the clean run.

Traps, each with the guard that closes it
-----------------------------------------
1. **Patched and reference runs that differ in more than the patch.** Every
   chunk is scored at one fixed shape and row layout, so a patched run and the
   run it is compared with differ only in the patched values (V2-V4 are the
   bit-exact checks of that).
2. **Encoder patches on padding.** `mix_encoder_slots` refuses a padded slot and
   two runs whose padding differs; replacements keep the history length.
3. **Reading the wrong digit.** Each pair names its scored digit; only that
   digit's distribution is read (never a revealed position's).
4. **Silently missing controls.** Pairs without a catalogue control are counted
   in `missing.csv`.

    python run.py --cell 6 --out <new dir> [--limit 600] [--device cpu]
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
from sidlens.data.sids import SidTable, load_item2id                     # noqa: E402
from sidlens.hooks import capture                                        # noqa: E402
from sidlens.interventions import diffusion as DI                        # noqa: E402
from sidlens.interventions.history import ControlPool, condition_rng     # noqa: E402
from sidlens.interventions.residual import Patch                         # noqa: E402
from sidlens.models import diffusion                                     # noqa: E402
from sidlens.provenance.hashing import sha256_file                       # noqa: E402
from sidlens.registry.diffusion import load_runtime                      # noqa: E402

CELLS = tuple(f"diff-next1-{q}-{d}cb-{w}" for q in ("MQ", "rqkmeans", "rqvae")
              for d in (3, 4, 5) for w in (128, 512)) + ("diff-next1-MQ-5cb-256", "diff-next1-rqvae-3cb-256")
SEED_EXP7, SEED = 20260928, 20261014
ENC_B, CHUNK = 512, 1024
N_LAYERS = 4
DEV = "cuda"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def exp1_validate():
    spec = importlib.util.spec_from_file_location(
        "_exp1_validate", paths.REPO / "experiments" / "controlled" / "exp1_matched_beam" / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_model_and_decoder


def encode_fixed(model, hist: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    out = []
    for s in range(0, len(hist), ENC_B):
        h, m = hist[s:s + ENC_B], mask[s:s + ENC_B]
        real = len(h)
        if real < ENC_B:
            h = np.concatenate([h, np.repeat(h[:1], ENC_B - real, 0)])
            m = np.concatenate([m, np.repeat(m[:1], ENC_B - real, 0)])
        out.append(diffusion.encode(model, torch.from_numpy(h).to(DEV), torch.from_numpy(m).to(DEV))[:real])
    return torch.cat(out)


def site_names(part: str, n: int) -> list[str]:
    enc = ["E:r1", "E:others", "E:r2", "E:all"] if part == "H" else []
    dec = [f"{kind}:L{l}:p{p}" for kind in ("D", "X") for l in range(N_LAYERS) for p in range(n)]
    return [f"{direction}|{s}" for direction in ("denoise", "noise") for s in enc + dec]


def knockout_names(n_head: int) -> list[str]:
    return ["K:all"] + [f"K:L{l}" for l in range(N_LAYERS)] + [f"K:h{h}" for h in range(n_head)]


class Runner:
    """Fixed-shape scoring of one chunk of pairs, with captures and patches."""

    def __init__(self, model, chunk: dict):
        self.model = model
        self.c = chunk
        self.rows = torch.arange(CHUNK, device=DEV)
        self.n = int(model.n_digit)

    def score(self, enc, codes, masked, *, patches=None, cross=None, smask=None, cache=None, want_capture=False):
        names = [f"decoder_blocks.{l}" for l in range(N_LAYERS)] + [f"decoder_blocks.{l}.cross_attn" for l in range(N_LAYERS)]
        cache = DI.project_cross_cache(self.model, enc) if cache is None else cache
        with torch.inference_mode():
            if want_capture:
                with capture(self.model, names=names, to_cpu=False) as cap:
                    lp = DI.score_states(self.model, enc, cache, self.rows, codes, masked, smask=smask)
                if any(cap.call_counts.get(nm) != 1 for nm in names):
                    raise RuntimeError(f"capture fired {cap.call_counts}")
                res = torch.stack([cap[f"decoder_blocks.{l}"] for l in range(N_LAYERS)])
                xat = torch.stack([cap[f"decoder_blocks.{l}.cross_attn"] for l in range(N_LAYERS)])
                return lp, res, xat
            return DI.score_states(self.model, enc, cache, self.rows, codes, masked, patches=patches,
                                   cross_patches=cross, smask=smask)

    def digit(self, lp):
        return lp[self.rows, self.c["d"]]                              # [CHUNK, K] at the scored digit


def pad_chunk(arrays: dict, real: int) -> dict:
    out = {}
    for k, v in arrays.items():
        if real < CHUNK:
            reps = (CHUNK - real,) + (1,) * (v.ndim - 1)
            v = torch.cat([v, v[:1].repeat(*reps)])
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)))
    ap.add_argument("--ckpt", help="a next-item DiffGRM grid weight state (grid:<key>@<state>) instead of --cell")
    ap.add_argument("--limit", type=int, default=None)
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
    t_start = time.time()

    grid_setup = None
    if args.ckpt:                    # a grid run: its own cohort, table and model (sidlens.grid.diff_setup)
        from sidlens.grid import diff_setup as DS
        grid_setup = DS.load(args.ckpt, device=DEV)
        ckpt_id, entry, cohort = args.ckpt, grid_setup.compat, grid_setup.cohort
        model, report, table, item2id = grid_setup.model, grid_setup.report, grid_setup.table, grid_setup.item2id
    else:
        ckpt_id = CELLS[args.cell]
        entry = load_runtime()[ckpt_id]
        cohort = load_eval_cohort(entry)
        model, report = diffusion.load(entry, device=DEV)
        table, item2id = SidTable.load(entry["sem_ids_name"]), load_item2id("Industrial_and_Scientific")
    if sha256_file(Path(entry["ckpt_path"])) != entry["ckpt_sha256"]:
        raise RuntimeError("checkpoint hash drift")
    n, K, W = int(model.n_digit), int(model.codebook_size), int(model.n_embd)
    n_head = int(model.decoder_blocks[0].self_attn.n_head)
    pool = ControlPool(table, item2id)
    if pool.n_digits != n:
        raise ValueError("SID table depth differs from the model")
    N = min(args.limit, len(cohort)) if args.limit else len(cohort)
    if grid_setup is not None:       # no exp1 archive: the run's own lists (V2a) and the scorer check
        validation = {"V5_checkpoint_and_cohort": True, "model_report_ok": report["ok"],
                      "V1_grid_checks": DS.check(grid_setup, device=DEV)}
        if not validation["V1_grid_checks"]["ok"]:
            raise RuntimeError(f"grid checks failed: {json.dumps(validation, default=str)[:1500]}")
    else:
        validation = {"V5_checkpoint_and_cohort": True, "model_report_ok": report["ok"],
                      "V1_exp1_pilot_checks": exp1_validate()(
                          model, torch.from_numpy(cohort.histories[:8]).to(DEV),
                          torch.from_numpy(cohort.history_mask[:8]).to(DEV), cohort.sid_buckets, matched_decode)}
    inputs = {"checkpoint": ckpt_id, "checkpoint_sha256": entry["ckpt_sha256"], "sem_ids": entry["sem_ids_name"],
              "cohort_sha256": cohort.cohort_sha256, "input_sha256": cohort.input_sha256, "n_users": N,
              "limited": args.limit, "n_digit": n, "codebook_size": K, "n_head": n_head, "seed": SEED,
              "seed_exp7_controls": SEED_EXP7, "enc_batch": ENC_B, "chunk": CHUNK,
              "device": torch.cuda.get_device_name() if DEV == "cuda" else DEV, "torch": torch.__version__,
              **({k: entry[k] for k in ("grid_key", "grid_state", "dataset")} if grid_setup else {})}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))

    # ---- pairs ----------------------------------------------------------------------
    H_meta, H_hist, S_meta, missing = [], [], [], []
    for i in range(N):
        L = int(cohort.history_lengths[i])
        tgt = cohort.target_sids[i].astype(np.int64)
        user = cohort.users[i]
        codes_h = [tuple(int(c) for c in cohort.histories[i, j]) for j in range(L)]
        if L >= 2:
            k = L - 1
            excl_codes = set(codes_h) | {tuple(int(c) for c in tgt)}
            excl_ids = {int(cohort.target_item_ids[i])}
            for d in range(n):
                idx = pool.candidates(codes_h[k], d, excl_ids, excl_codes)
                if len(idx) == 0:
                    missing.append({"part": "H", "user": user, "d": d})
                    continue
                j = idx[condition_rng(SEED_EXP7, user, k, d).integers(len(idx))]
                h = cohort.histories[i].copy()
                h[k] = pool.codes[j]
                H_hist.append(h)
                H_meta.append({"user_index": i, "d": d, "match": tuple(codes_h[k][:d]) == tuple(tgt[:d]),
                               "control_item": int(pool.item_ids[j]), "n_candidates": int(len(idx)),
                               "gold": int(tgt[d]), "copy": int(codes_h[k][d]), "r2_slot": L - 2, "r1_slot": L - 1})
        for d in range(1, n):
            idx = pool.candidates(tuple(int(c) for c in tgt), d - 1, {int(cohort.target_item_ids[i])},
                                  {tuple(int(c) for c in tgt)})
            if len(idx) == 0:
                missing.append({"part": "S", "user": user, "d": d})
                continue
            j = idx[condition_rng(SEED, user, "s", d).integers(len(idx))]
            S_meta.append({"user_index": i, "d": d, "match": True, "sibling_item": int(pool.item_ids[j]),
                           "sibling_code": int(pool.codes[j][d - 1]), "n_candidates": int(len(idx)),
                           "gold": int(tgt[d]), "copy": -1})
    pd.DataFrame(missing, columns=["part", "user", "d"]).to_csv(args.out / "missing.csv", index=False)
    log(f"{ckpt_id}: H pairs {len(H_meta)}, S pairs {len(S_meta)}, missing {len(missing)}")

    enc_clean = encode_fixed(model, cohort.histories[:N], cohort.history_mask[:N])
    enc_H = encode_fixed(model, np.stack(H_hist), cohort.history_mask[[m["user_index"] for m in H_meta]]) \
        if H_meta else None
    hmask_all = torch.from_numpy(cohort.history_mask[:N]).to(DEV)
    targets = torch.from_numpy(cohort.target_sids[:N]).to(DEV)

    results = {}
    for part, meta in (("H", H_meta), ("S", S_meta)):
        sites = site_names(part, n)
        kos = knockout_names(n_head) if part == "S" else []
        P = len(meta)
        arr = {m: np.full((P, len(sites)), np.nan, dtype=np.float32) for m in ("kl", "gold", "copy")}
        ko = {m: np.full((P, len(kos), 2), np.nan, dtype=np.float32) for m in ("gold",)}
        base = {m: np.full((P,), np.nan, dtype=np.float32) for m in
                ("gold_clean", "gold_corr", "kl_corr", "copy_clean", "copy_corr")}
        ui = torch.as_tensor([m["user_index"] for m in meta], device=DEV, dtype=torch.long)
        d_all = torch.as_tensor([m["d"] for m in meta], device=DEV, dtype=torch.long)
        gold_all = torch.as_tensor([m["gold"] for m in meta], device=DEV, dtype=torch.long)
        copy_all = torch.as_tensor([max(m["copy"], 0) for m in meta], device=DEV, dtype=torch.long)
        sib_all = torch.as_tensor([m.get("sibling_code", 0) for m in meta], device=DEV, dtype=torch.long)
        for c0 in range(0, P, CHUNK):
            real = min(CHUNK, P - c0)
            sl = slice(c0, c0 + real)
            u = ui[sl]
            d = d_all[sl]
            tgt = targets[u]
            masked = torch.arange(n, device=DEV)[None, :] >= d[:, None]              # S_pref(d)
            codes = torch.where(masked, torch.zeros_like(tgt), tgt)
            chunk = {"enc_a": enc_clean[u], "enc_b": enc_H[sl] if part == "H" else enc_clean[u],
                     "hmask": hmask_all[u], "codes_a": codes, "masked_a": masked, "d": d,
                     "gold": gold_all[sl], "copy": copy_all[sl]}
            if part == "S":
                codes_b = codes.clone()
                codes_b[torch.arange(real, device=DEV), d - 1] = sib_all[sl]
                chunk["codes_b"], chunk["masked_b"] = codes_b, masked
            else:
                chunk["codes_b"], chunk["masked_b"] = codes, masked
            r1 = torch.as_tensor([meta[c0 + r].get("r1_slot", 0) for r in range(real)], device=DEV)
            r2 = torch.as_tensor([meta[c0 + r].get("r2_slot", 0) for r in range(real)], device=DEV)
            chunk = pad_chunk(chunk, real)
            r1, r2 = pad_chunk({"r1": r1, "r2": r2}, real).values()
            R = Runner(model, chunk)
            cache_a = DI.project_cross_cache(model, chunk["enc_a"])
            cache_b = DI.project_cross_cache(model, chunk["enc_b"])
            lp_a, res_a, x_a = R.score(chunk["enc_a"], chunk["codes_a"], chunk["masked_a"], cache=cache_a, want_capture=True)
            lp_b, res_b, x_b = R.score(chunk["enc_b"], chunk["codes_b"], chunk["masked_b"], cache=cache_b, want_capture=True)
            pa, pb = R.digit(lp_a), R.digit(lp_b)
            ar = torch.arange(CHUNK, device=DEV)

            def metrics(px):
                kl = (pa.exp() * (pa - px)).sum(-1)
                return kl, px[ar, chunk["gold"]], px[ar, chunk["copy"]]

            kl_b, g_b, c_b = metrics(pb)
            for name, v in (("gold_clean", pa[ar, chunk["gold"]]), ("gold_corr", g_b), ("kl_corr", kl_b),
                            ("copy_clean", pa[ar, chunk["copy"]]), ("copy_corr", c_b)):
                base[name][sl] = v[:real].cpu().numpy()

            first = c0 == 0
            for si, site in enumerate(sites):
                direction, what = site.split("|")
                # denoise: corrupted run (b) with clean (a) pieces; noise: clean run with corrupted pieces
                base_run, donor = ("b", "a") if direction == "denoise" else ("a", "b")
                enc_base, enc_donor = chunk[f"enc_{base_run}"], chunk[f"enc_{donor}"]
                codes_x, masked_x = chunk[f"codes_{base_run}"], chunk[f"masked_{base_run}"]
                cache_base = cache_b if base_run == "b" else cache_a
                res_d, x_d = (res_a, x_a) if donor == "a" else (res_b, x_b)
                if what.startswith("E:"):
                    S = enc_base.shape[1]
                    slot = torch.arange(S, device=DEV)[None, :]
                    take = {"E:r1": slot == r1[:, None], "E:r2": slot == r2[:, None],
                            "E:others": (slot != r1[:, None]) & chunk["hmask"],
                            "E:all": chunk["hmask"]}[what] & chunk["hmask"]
                    enc_mix = DI.mix_encoder_slots(enc_base, enc_donor, take, chunk["hmask"])
                    lp = R.score(enc_mix, codes_x, masked_x)
                    if what == "E:all" and direction == "denoise":
                        ok = bool(torch.equal(lp[:real], lp_a[:real]))
                        validation["V2_denoise_all_encoder_slots_is_clean"] = \
                            validation.get("V2_denoise_all_encoder_slots_is_clean", True) and ok
                        if not ok:
                            raise RuntimeError("denoising every encoder slot did not reproduce the clean run")
                else:
                    kind, layer, pos = what.split(":")
                    l, p = int(layer[1:]), int(pos[1:])
                    rows = torch.arange(CHUNK, device=DEV)
                    posv = torch.full((CHUNK,), p, device=DEV, dtype=torch.long)
                    if kind == "D":
                        lp = R.score(enc_base, codes_x, masked_x, cache=cache_base,
                                     patches=[Patch(f"decoder_blocks.{l}", rows, posv, res_d[l][:, p])])
                    else:
                        lp = R.score(enc_base, codes_x, masked_x, cache=cache_base, cross={l: (rows, posv, x_d[l][:, p])})
                kl, g, cp = metrics(R.digit(lp))
                arr["kl"][sl, si], arr["gold"][sl, si], arr["copy"][sl, si] = (
                    kl[:real].cpu().numpy(), g[:real].cpu().numpy(), cp[:real].cpu().numpy())
            if first:
                rows = torch.arange(CHUNK, device=DEV)

                def at(p):
                    return torch.full((CHUNK,), p, device=DEV, dtype=torch.long)

                v3 = all(
                    torch.equal(R.score(chunk["enc_a"], chunk["codes_a"], chunk["masked_a"], cache=cache_a,
                                        patches=[Patch("decoder_blocks.1", rows, at(p), res_a[1][:, p])]), lp_a)
                    and torch.equal(R.score(chunk["enc_a"], chunk["codes_a"], chunk["masked_a"], cache=cache_a,
                                            cross={1: (rows, at(p), x_a[1][:, p])}), lp_a)
                    for p in range(n))
                rr = rows.repeat_interleave(n)
                pp = torch.arange(n, device=DEV).repeat(CHUNK)
                v4 = torch.equal(R.score(chunk["enc_b"], chunk["codes_b"], chunk["masked_b"], cache=cache_b,
                                         patches=[Patch("decoder_blocks.3", rr, pp, res_a[3][rr, pp])]), lp_a)
                validation[f"V3_self_patch_block1_exact_{part}"] = bool(v3)
                validation[f"V4_last_block_full_patch_is_clean_{part}"] = bool(v4)
                if not (v3 and v4):
                    raise RuntimeError(f"part {part}: self-patch {v3}, full last-block patch {v4}")
            for ki, kname in enumerate(kos):
                edges = []
                for r in range(CHUNK):
                    dq = int(chunk["d"][r])
                    if kname == "K:all":
                        edges.append((r, range(N_LAYERS), [dq], [dq - 1]))
                    elif kname.startswith("K:L"):
                        edges.append((r, [int(kname[3:])], [dq], [dq - 1]))
                    else:
                        edges.append((r, range(N_LAYERS), [dq], [dq - 1], [int(kname[3:])]))
                mask = DI.self_knockout_mask(N_LAYERS, CHUNK, n, edges, n_head=n_head if kname.startswith("K:h") else None)
                sm = torch.from_numpy(mask).to(DEV)
                for run_i, run in enumerate(("a", "b")):
                    lp = R.score(chunk[f"enc_{run}"], chunk[f"codes_{run}"], chunk[f"masked_{run}"],
                                 cache=cache_a if run == "a" else cache_b, smask=sm)
                    ko["gold"][sl, ki, run_i] = R.digit(lp)[ar, chunk["gold"]][:real].cpu().numpy()
            log(f"  part {part}: {c0 + real}/{P}")
        meta_df = pd.DataFrame(meta)
        for k_, v in base.items():
            meta_df[k_] = v
        meta_df.to_csv(args.out / f"pairs_{part}.csv", index=False)
        np.savez_compressed(args.out / f"sites_{part}.npz", sites=np.array(sites), knockouts=np.array(kos),
                            ko_gold=ko["gold"], **arr)
        results[part] = P
    validation.update({"pairs": results, "seconds": round(time.time() - t_start, 1)})
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
