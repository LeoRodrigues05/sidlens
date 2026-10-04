#!/usr/bin/env python
"""At the last digit, does reading a prefix-matching history item's last-digit
token push the model toward that item's own code (copy) or away from it?

Protocol: `protocol.md` beside this file, declared before any forward. One
invocation = one cell (checkpoint) x split. Reuses exp4's machinery: the
FixedShapeRunner (fixed shapes, explicit per-layer attention masks) and its
bit-exact no-op controls.

Traps guarded here:

* h's code is scored from the code logits, column ``vocab.codes(d).index(c)``,
  never ``c`` (codes are sparse and string-sorted in the vocabulary).
* T_pad is computed over the whole split, as in exp4, so clean scores can be
  compared with exp4's clean scores for the same rows.
* The edge q -> h's last-digit token must be live; the knockout guard raises
  if it is already masked. The no-op controls set expect_masked and raise if
  their edge is live.
* Rows without a control item are kept for the clean and KH conditions and
  marked, never silently dropped.

    python experiments/controlled/exp10_ar_last_digit_sibling/run.py --cell 1 --split test --out <new dir>
"""

from __future__ import annotations

import argparse
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
from sidlens.data.sids import SidTable                                   # noqa: E402
from sidlens.interventions.runner import FixedShapeRunner, Job, max_diff  # noqa: E402
from sidlens.interventions.scoring import DigitScorer                    # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing, manifest as manifest_mod         # noqa: E402

CELLS = ("next-item_best", "oneoff_rqvae4cb128")
ND_PAIRS = (paths.DERIVED / "structure" / "exp3_collision_causes" / "293555" / "result"
            / "partA_near_duplicate_pairs.csv")
SEED = 20260927


def checkpoint_sha(ckpt: str) -> str:
    man = manifest_mod.load()
    want = man["data"][f"ckpt/ar/{ckpt}"]["files"]["model.safetensors"]["sha256"]
    got = hashing.HashCache().get(paths.FROZEN_CKPT / "ar" / ckpt / "model.safetensors")
    if got != want:
        raise RuntimeError(f"{ckpt}/model.safetensors sha256 {got[:12]} != manifest {want[:12]}")
    return got


def rebought_items(variant) -> set[int]:
    """Items that some train row has both in its history and as its target."""
    out = set()
    for e in P.load_examples(variant, "next-item", "train"):
        if e.target_item_ids[0] in e.history_item_ids:
            out.add(e.target_item_ids[0])
    return out


def row_info(e: P.Encoded, n: int, nd_adj, rebought) -> dict | None:
    ex = e.example
    hc = [P.parse_sid(s, n) for s in ex.history_sids]
    tc = P.parse_sid(ex.target_sids[0], n)
    match = [k for k, c in enumerate(hc) if c[:n - 1] == tc[:n - 1]]
    if not match:
        return None
    k = max(match)
    j = next((j for j in range(len(hc) - 1, -1, -1) if hc[j][:n - 1] != tc[:n - 1]), None)
    tok = lambda kk: e.positions("hist_sid", item=kk, digit=n - 1)[0]      # noqa: E731
    h_item, t_item = ex.history_item_ids[k], ex.target_item_ids[0]
    return {"k": k, "j": j, "q": e.predict_pos(0, n - 1), "key_h": tok(k),
            "key_c": None if j is None else tok(j),
            "h_code": hc[k][n - 1], "t_code": tc[n - 1],
            "subset": "S_same" if hc[k][n - 1] == tc[n - 1] else "S_diff",
            "nd": t_item in nd_adj.get(h_item, ()), "h_is_r1": k == len(hc) - 1,
            "h_rebought": h_item in rebought, "prefix": tc[:n - 1]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    ap.add_argument("--split", default="test", choices=("valid", "test"))
    ap.add_argument("--limit", type=int, default=None, help="first N eligible rows (pilot only)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    ckpt = CELLS[args.cell]
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / f).exists() for f in ("inputs.json", "validation.json", "scores.parquet")):
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
    clean_all = [P.encode(ex, tok, vocab, template="eval", with_target=True) for ex in examples]
    t_pad = -(-max(len(e) for e in clean_all) // 8) * 8          # as exp4: over the whole split

    nd = pd.read_csv(ND_PAIRS)
    nd_adj: dict[int, set[int]] = {}
    for a, b in zip(nd.i, nd.j):
        nd_adj.setdefault(int(a), set()).add(int(b))
        nd_adj.setdefault(int(b), set()).add(int(a))
    rebought = rebought_items(vocab.variant)
    rows = [(i, x) for i, e in enumerate(clean_all)
            if (x := row_info(e, n, nd_adj, rebought)) is not None]
    if args.limit:
        rows = rows[:args.limit]
    cols = vocab.codes(n - 1)
    col_of = {c: j for j, c in enumerate(cols)}
    legal = {}
    codes = table.codes
    for _, x in rows:
        if x["prefix"] not in legal:
            sel = np.all(codes[:, :n - 1] == np.asarray(x["prefix"]), axis=1)
            m = np.zeros(len(cols), bool)
            m[[col_of[int(c)] for c in np.unique(codes[sel, n - 1])]] = True
            legal[x["prefix"]] = m

    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": checkpoint_sha(ckpt),
              "variant": vocab.variant.name, "n_digits": n, "split": args.split, "template": "eval",
              "csv_sha256": hashing.sha256_file(P.csv_path(vocab.variant, "next-item", args.split)),
              "sem_ids_sha256": hashing.sha256_file(table.source),
              "nd_pairs_sha256": hashing.sha256_file(ND_PAIRS),
              "n_rows_split": len(examples), "n_eligible": len(rows), "limited": args.limit,
              "n_by_subset": pd.Series([x["subset"] for _, x in rows]).value_counts().to_dict(),
              "n_without_control": int(sum(x["j"] is None for _, x in rows)),
              "n_rebought_items_train": len(rebought), "t_pad": t_pad, "batch_rows": args.rows}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"inputs ok: {ckpt} {args.split} eligible={len(rows)} {inputs['n_by_subset']} T_pad={t_pad}")

    model = ar.load_model(ckpt, device=args.device, dtype=args.dtype)
    n_layers = model.config.num_hidden_layers
    late, all_layers = tuple(range(n_layers // 2, n_layers)), tuple(range(n_layers))
    inputs.update({"dtype": args.dtype, "late_layers": list(late),
                   "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                   "torch": torch.__version__, "transformers": __import__("transformers").__version__})
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    runner = FixedShapeRunner(model, DigitScorer(vocab, table), tok.pad_token_id, t_pad, args.rows, 1)

    jobs = []
    for i, x in rows:
        e, q = clean_all[i], [x["q"]]
        jobs.append(Job(e, meta={"i": i, "condition": "clean"}))
        jobs.append(Job(e, knockouts=[(q, [x["key_h"]], late, None, False)], meta={"i": i, "condition": "KH_late"}))
        jobs.append(Job(e, knockouts=[(q, [x["key_h"]], all_layers, None, False)], meta={"i": i, "condition": "KH_all"}))
        if x["key_c"] is not None:
            jobs.append(Job(e, knockouts=[(q, [x["key_c"]], late, None, False)], meta={"i": i, "condition": "KC_late"}))
        jobs.append(Job(e, meta={"i": i, "condition": "K0a_empty"}))
        jobs.append(Job(e, knockouts=[(q, [q[0] + 1], all_layers, None, True)], meta={"i": i, "condition": "K0b_future"}))

    info = dict(rows)
    recs, logits = [], {}
    t0 = time.time()
    for b in range(0, len(jobs), runner.B):
        batch = jobs[b:b + runner.B]
        _, cl, _, _ = runner.forward(batch)
        lg = cl[n - 1]                                            # (R, C) last-digit code logits
        for r, job in enumerate(batch):
            i, cond = job.meta["i"], job.meta["condition"]
            x = info[i]
            z = lg[r].double()
            lp = (z - torch.logsumexp(z, -1)).numpy()
            m = legal[x["prefix"]]
            zl = z.clone()
            zl[~torch.as_tensor(m)] = float("-inf")
            lpl = (zl - torch.logsumexp(zl, -1)).numpy()
            hj, tj = col_of[x["h_code"]], col_of[x["t_code"]]
            top = int(np.argmax(lp))
            if cond in ("clean", "K0a_empty", "K0b_future"):
                logits[(i, cond)] = lg[r].clone()
            if cond.startswith("K0"):
                continue
            ex = clean_all[i].example
            recs.append({"example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id, "i": i,
                         "condition": cond, "subset": x["subset"], "nd": x["nd"], "h_is_r1": x["h_is_r1"],
                         "h_rebought": x["h_rebought"], "has_control": x["key_c"] is not None,
                         "h_code": x["h_code"], "t_code": x["t_code"],
                         "lh": lp[hj], "lt": lp[tj], "lh_legal": lpl[hj], "lt_legal": lpl[tj],
                         "top1_is_h": top == hj, "top1_is_t": top == tj,
                         "rank_h": int((lp > lp[hj]).sum()), "rank_t": int((lp > lp[tj]).sum()),
                         "batch": b // runner.B})
        if (b // runner.B) % 10 == 0:
            log(f"  batch {b // runner.B + 1}/{-(-len(jobs) // runner.B)}")
    pd.DataFrame(recs).to_parquet(args.out / "scores.parquet", index=False)
    log(f"[forwards] {len(jobs)} jobs, {runner.n_forwards} forwards, {time.time() - t0:.0f}s")

    ctrl = []
    for (i, cond), lg in logits.items():
        if cond == "clean":
            continue
        d_ = max_diff([lg], [logits[(i, "clean")]])
        ctrl.append({"i": i, "control": cond, "max_abs_diff": d_, "exact": d_ == 0.0})
    c = pd.DataFrame(ctrl)
    validation["controls"] = {k: {"n": int(len(g)), "n_exact": int(g.exact.sum()),
                                  "max_abs_diff": float(g.max_abs_diff.max())}
                              for k, g in c.groupby("control")}
    # clean agreement with exp4 (test only: exp4's test run is 281394)
    if args.split == "test":
        e4 = pd.read_parquet(paths.DERIVED / "controlled" / "exp4_ar_copy_circuit" / "281394"
                             / f"cell-{args.cell:02d}" / "clean.parquet")
        e4 = e4[e4.digit == n - 1].set_index("example_id").logp_codes
        mine = pd.DataFrame(recs)
        mine = mine[mine.condition == "clean"].set_index("example_id")["lt"]
        diff = (mine - e4.loc[mine.index]).abs()
        validation["clean_vs_exp4"] = {"n": int(len(diff)), "max_abs_diff": float(diff.max()),
                                       "mean_abs_diff": float(diff.mean()), "ok": bool(diff.max() <= 0.05)}
    validation["n_forwards"] = runner.n_forwards
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done: controls {json.dumps(validation['controls'])} exp4 {validation.get('clean_vs_exp4')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
