#!/usr/bin/env python
"""Prefix-matched copying: replication (R), attention-edge knockout (K), heads (H).

Protocol: `protocol.md` beside this file, declared before any valid-split
forward or knockout. One invocation = one cell (checkpoint x split).

The head set S is selected INSIDE this run by the declared rule (top 5 mean
effect on screening-half users) and then tested on held-out users, so the
selection cannot be tuned after seeing held-out outcomes. Everything numerical
goes through `sidlens.interventions.runner.FixedShapeRunner` (fixed shapes,
explicit per-layer masks), whose no-op controls must be bit-exact.

    python experiments/controlled/exp4_ar_copy_circuit/run.py --cell 0 --split valid --out <new dir>
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens import paths                                                # noqa: E402
from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.data.sids import SidTable, load_item2id                     # noqa: E402
from sidlens.interventions import history as Hi                          # noqa: E402
from sidlens.interventions.runner import FixedShapeRunner, Job, max_diff, records  # noqa: E402
from sidlens.interventions.scoring import DigitScorer                    # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
SEED = 20260927
TOP_K = 5
N_CONTROL_SETS = 20


def checkpoint_sha(ckpt: str) -> str:
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def user_half(user_id: str) -> int:
    """0 = screening, 1 = held-out (protocol: sha256 parity)."""
    return int(hashlib.sha256(f"{SEED}|{user_id}".encode()).hexdigest(), 16) % 2


def row_info(e: P.Encoded, n: int) -> dict:
    """k*, match_d, control item j_d, and the positions every condition needs."""
    ex = e.example
    L = len(ex.history_sids)
    hc = [P.parse_sid(s, n) for s in ex.history_sids]
    tc = P.parse_sid(ex.target_sids[0], n)
    k = L - 1
    tok = lambda j, d: e.positions("hist_sid", item=j, digit=d)[0]      # noqa: E731
    info = {"k": k, "match": [hc[k][:d] == tc[:d] for d in range(n)],
            "readout": [e.predict_pos(0, d) for d in range(n)],
            "next": [tok(k, d) for d in range(n)],
            "item": e.positions("hist_sid", item=k), "ctrl": [], "ctrl_item": [],
            "header0": e.positions("response_header")[0], "len": len(e)}
    for d in range(n):
        j = next((j for j in range(L - 2, -1, -1) if d == 0 or hc[j][:d] != tc[:d]), None)
        info["ctrl"].append(None if j is None else tok(j, d))
        info["ctrl_item"].append(-1 if j is None else j)
    if len(info["item"]) != n:
        raise ValueError(f"{ex.example_id}: most recent item is not {n} tokens")
    return info


class Stream:
    """Batch jobs to the runner in fixed-size forwards; collect records and code logits."""

    def __init__(self, runner: FixedShapeRunner, keep_cols=None, keep_logits=lambda job: False):
        self.runner, self.buf, self.recs, self.batch = runner, [], [], 0
        self.keep_cols, self.keep_logits, self.logits = keep_cols, keep_logits, {}

    def add(self, job: Job):
        self.buf.append(job)
        if len(self.buf) >= self.runner.B:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        s, cl, _, _ = self.runner.forward(self.buf)
        recs = records(self.runner, self.buf, s, self.batch)
        if self.keep_cols is not None:
            recs = [r for r in recs if self.keep_cols(r)]
        self.recs += recs
        for r, job in enumerate(self.buf):
            if self.keep_logits(job):
                self.logits[(job.meta["i"], job.meta["condition"], job.meta.get("d", -1))] = \
                    [c[r] for c in cl]
        self.batch += 1
        self.buf = []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)))
    ap.add_argument("--ckpt", choices=CELLS)
    ap.add_argument("--split", default="valid", choices=("valid", "test"))
    ap.add_argument("--parts", default="R,K,H")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if (args.cell is None) == (args.ckpt is None):
        ap.error("give exactly one of --cell or --ckpt")
    ckpt = args.ckpt or CELLS[args.cell]
    parts = set(args.parts.split(","))
    if not parts or not parts <= {"R", "K", "H"}:
        ap.error("--parts is a subset of R,K,H")
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json")):
        raise FileExistsError(f"{args.out} already holds results; results are never overwritten")
    logf = open(args.out / "run.log", "a")
    t_start = time.time()

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ---- inputs -------------------------------------------------------------
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(paths.FROZEN_CKPT / "ar" / ckpt), local_files_only=True)
    examples = P.load_examples(vocab.variant, "next-item", args.split)
    table = SidTable.load(vocab.variant)
    validation = {"table": P.check_against_table(examples, table),
                  "vocab_vs_table": ar.validate_against(vocab, table)}
    if args.split == "test":
        validation["archive"] = P.check_against_archive(examples)
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:2000]}")
    clean = [P.encode(ex, tok, vocab, template="eval", with_target=True) for ex in examples]
    t_pad = -(-max(len(e) for e in clean) // 8) * 8
    if args.limit:
        examples, clean = examples[:args.limit], clean[:args.limit]
    info = [row_info(e, n) for e in clean]
    half = [user_half(ex.user_id) for ex in examples]
    pool = Hi.ControlPool(table, load_item2id(P.PRIMARY_CATEGORY))
    csv = P.csv_path(vocab.variant, "next-item", args.split)
    sha = checkpoint_sha(ckpt)
    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": sha, "variant": vocab.variant.name,
              "n_digits": n, "split": args.split, "template": "eval", "csv": str(csv.relative_to(paths.WORK)),
              "csv_sha256": hashing.sha256_file(csv), "sem_ids_sha256": hashing.sha256_file(table.source),
              "n_rows": len(examples), "n_users": len({e.user_id for e in examples}),
              "limited": args.limit, "t_pad": t_pad, "batch_rows": args.rows, "seed": SEED,
              "parts": sorted(parts), "top_k": TOP_K, "n_control_sets": N_CONTROL_SETS,
              "n_match_by_digit": {d: int(sum(x["match"][d] for x in info)) for d in range(n)},
              "n_ctrl_key_by_digit": {d: int(sum(x["ctrl"][d] is not None for x in info)) for d in range(n)},
              "users_screening": len({e.user_id for e, h in zip(examples, half) if h == 0}),
              "users_heldout": len({e.user_id for e, h in zip(examples, half) if h == 1})}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"inputs ok: {ckpt} {args.split} rows={len(examples)} T_pad={t_pad} match={inputs['n_match_by_digit']}")

    model = ar.load_model(ckpt, device=args.device, dtype=args.dtype)
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    inputs.update({"dtype": args.dtype, "attn_implementation": model.config._attn_implementation,
                   "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                   "torch": torch.__version__, "transformers": __import__("transformers").__version__})
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    runner = FixedShapeRunner(model, DigitScorer(vocab, table), tok.pad_token_id, t_pad, args.rows, 1)
    all_layers = tuple(range(n_layers))

    def meta(i, cond, d=-1, **kw):
        x = info[i]
        return {"i": i, "condition": cond, "d": d, "half": half[i],
                "match": bool(x["match"][d]) if d >= 0 else None, **kw}

    # ---- clean rows, with the attention map observed ------------------------
    t0 = time.time()
    clean_recs, attn = [], {k: np.full((len(examples), n, n_layers, n_heads), np.nan, np.float32)
                            for k in ("p_next", "p_ctrl", "p_item")}
    clean_logits = {}
    for b in range(0, len(examples), runner.B):
        idx = list(range(b, min(len(examples), b + runner.B)))
        jobs = [Job(clean[i], meta=meta(i, "clean")) for i in idx]
        pairs = [(r, info[i]["readout"][d]) for r, i in enumerate(idx) for d in range(n)]
        s, cl, _, obs = runner.forward(jobs, observe={"layers": list(all_layers), "pairs": pairs})
        clean_recs += records(runner, jobs, s, b // runner.B)
        for r, i in enumerate(idx):
            clean_logits[i] = [c[r] for c in cl]
        for L in all_layers:
            pr = obs[L].numpy()                                        # (P, H, T)
            for p, (r, _) in enumerate(pairs):
                i, d = idx[r], p % n
                attn["p_next"][i, d, L] = pr[p, :, info[i]["next"][d]]
                attn["p_item"][i, d, L] = pr[p][:, info[i]["item"]].sum(-1)
                if info[i]["ctrl"][d] is not None:
                    attn["p_ctrl"][i, d, L] = pr[p, :, info[i]["ctrl"][d]]
    pd.DataFrame(clean_recs).to_parquet(args.out / "clean.parquet", index=False)
    np.savez_compressed(args.out / "attention_map.npz", **attn,
                        example_id=np.array([e.example_id for e in examples]),
                        match=np.array([x["match"] for x in info]))
    clean_s = pd.DataFrame(clean_recs).set_index(["i", "digit"])["logp_codes"]
    log(f"[clean+attention] {len(examples)} rows, {runner.n_forwards} forwards, {time.time() - t0:.0f}s")

    # ---- Part R: replication ------------------------------------------------
    missing = []
    if "R" in parts:
        t0 = time.time()
        st = Stream(runner)
        for i, ex in enumerate(examples):
            k = info[i]["k"]
            for d in range(1, n):
                c = pool.draw(ex, k, d, SEED)
                if c is None:
                    missing.append({"example_id": ex.example_id, "d": d})
                    continue
                e2 = P.encode(Hi.replace_history_item(ex, k, c), tok, vocab, template="eval", with_target=True)
                Hi.check_replacement(clean[i], e2, k, d)
                st.add(Job(e2, meta=meta(i, "replace_m_eq_d", d, control_item=c.item_id)))
        st.flush()
        pd.DataFrame(st.recs).to_parquet(args.out / "part_r.parquet", index=False)
        pd.DataFrame(missing, columns=["example_id", "d"]).to_csv(args.out / "part_r_no_control.csv", index=False)
        log(f"[R] {len(st.recs)} records, {len(missing)} missing controls, {time.time() - t0:.0f}s")

    # ---- Part K: knockouts + no-op controls ---------------------------------
    ctrl_recs = []
    if "K" in parts:
        t0 = time.time()
        early, late = tuple(range(0, n_layers // 2)), tuple(range(n_layers // 2, n_layers))
        st = Stream(runner, keep_logits=lambda j: j.meta["condition"].startswith("K0"))
        for i in range(len(examples)):
            x = info[i]
            for d in range(n):
                q = [x["readout"][d]]
                conds = {"K1_next": ([x["next"][d]], all_layers), "K2_item": (x["item"], all_layers),
                         "K4_next_early": ([x["next"][d]], early), "K5_next_late": ([x["next"][d]], late)}
                if x["ctrl"][d] is not None:
                    conds["K3_ctrl"] = ([x["ctrl"][d]], all_layers)
                for name, (keys, layers) in conds.items():
                    st.add(Job(clean[i], knockouts=[(q, keys, layers, None, False)], meta=meta(i, name, d)))
            st.add(Job(clean[i], knockouts=[(list(range(x["header0"], x["len"])), x["item"], all_layers,
                                             None, False)], meta=meta(i, "K6_readout_item")))
            st.add(Job(clean[i], meta=meta(i, "K0a_empty")))
            q = x["readout"][n - 1]
            st.add(Job(clean[i], knockouts=[([q], [q + 1], all_layers, None, True)], meta=meta(i, "K0b_future")))
        st.flush()
        for (i, cond, _), lg in st.logits.items():
            d_ = max_diff(lg, clean_logits[i])
            ctrl_recs.append({"example_id": examples[i].example_id, "control": cond,
                              "max_abs_diff_code_logits": d_, "exact": d_ == 0.0})
        recs = [r for r in st.recs if not r["condition"].startswith("K0")]
        pd.DataFrame(recs).to_parquet(args.out / "part_k.parquet", index=False)
        log(f"[K] {len(recs)} records, {time.time() - t0:.0f}s")

    # ---- Part H: head screen on half 0, held-out test on half 1 -------------
    if "H" in parts:
        t0 = time.time()
        screen = [(i, d) for i in range(len(examples)) for d in range(1, n)
                  if half[i] == 0 and info[i]["match"][d]]
        st = Stream(runner, keep_cols=lambda r: r["digit"] == r["d"])
        for i, d in screen:
            for L in all_layers:
                for h in range(n_heads):
                    st.add(Job(clean[i], knockouts=[([info[i]["readout"][d]], [info[i]["next"][d]], (L,), (h,), False)],
                               meta=meta(i, "H_screen", d, layer=L, head=h)))
        st.flush()
        sc = pd.DataFrame(st.recs).merge(clean_s.rename("clean_logp").reset_index(),
                                         on=["i", "digit"], validate="many_to_one")
        sc["delta"] = sc.clean_logp - sc.logp_codes
        sc.to_parquet(args.out / "part_h_screen.parquet", index=False)
        rank = sc.groupby(["layer", "head"]).delta.mean().reset_index() \
            .sort_values(["delta", "layer", "head"], ascending=[False, True, True])
        S = [(int(r.layer), int(r.head)) for r in rank.head(TOP_K).itertuples()]
        rng = np.random.default_rng(SEED)
        controls = [[(L, int(rng.choice([x for x in range(n_heads) if x != h]))) for L, h in S]
                    for _ in range(N_CONTROL_SETS)]
        (args.out / "heads.json").write_text(json.dumps(
            {"rule": f"top {TOP_K} by mean delta on screening-half matching (row, d>=1) pairs",
             "n_screen_pairs": len(screen), "S": S, "control_sets": controls,
             "ranking_top20": rank.head(20).to_dict("records")}, indent=1))
        log(f"[H] screen {len(screen)} pairs, {runner.n_forwards} forwards; S = {S}")

        test = [(i, d) for i in range(len(examples)) for d in range(n)
                if half[i] == 1 and (d == 0 or info[i]["match"][d])]
        st = Stream(runner, keep_cols=lambda r: r["digit"] == r["d"])
        for i, d in test:
            q, k = [info[i]["readout"][d]], [info[i]["next"][d]]
            for name, heads in [("S", S)] + [(f"C{c:02d}", hs) for c, hs in enumerate(controls)]:
                st.add(Job(clean[i], knockouts=[(q, k, (L,), (h,), False) for L, h in heads],
                           meta=meta(i, f"H_{name}", d)))
        st.flush()
        pd.DataFrame(st.recs).to_parquet(args.out / "part_h_test.parquet", index=False)
        log(f"[H] held-out {len(test)} pairs, {time.time() - t0:.0f}s")

    # ---- validation ---------------------------------------------------------
    if ctrl_recs:
        c = pd.DataFrame(ctrl_recs)
        c.to_parquet(args.out / "controls.parquet", index=False)
        validation["controls"] = {k: {"n": int(len(g)), "n_exact": int(g.exact.sum()),
                                      "max_abs_diff_code_logits": float(g.max_abs_diff_code_logits.max())}
                                  for k, g in c.groupby("control")}
    validation["part_r_no_control"] = len(missing)
    validation["clean_top1_by_digit"] = pd.DataFrame(clean_recs).groupby("digit")["rank"].apply(
        lambda r: float((r == 0).mean())).to_dict()
    validation["n_forwards"] = runner.n_forwards
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: {json.dumps(validation.get('controls'), default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
