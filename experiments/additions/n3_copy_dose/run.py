#!/usr/bin/env python
"""N3: copy reads scaled by alpha inside one grid run's own beam search (AR or DiffGRM).

Protocol: `protocol.md` beside this file, declared before any scaled decode.
One invocation = one next-item grid run at its best state; the paradigm is the
run's. Every alpha decodes the same rows in the same batches.

Traps, each with its guard
--------------------------
1. **A dose that is not exp6's / exp14's edges.** AR uses `decode_ar.CopyEdit`
   (`C_all`), DiffGRM `diff_setup.encode_slot_scaled` + `slot_scale_context`,
   the same edges those experiments remove; alpha = 0 is their block.
2. **alpha = 1 that is not the baseline.** V3 checks it bit for bit against the
   unscaled route (module doc of `protocol.md`).
3. **A decoder that is not the run's.** V2a gates on the run's saved lists.

    python run.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 64]
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

from sidlens import grid                                                  # noqa: E402
from sidlens.data import ar_prompts as P                                  # noqa: E402
from sidlens.grid import events, predictions as GP, registry              # noqa: E402
from sidlens.models import ar                                             # noqa: E402

ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)


def tag_of(a: float) -> str:
    return f"alpha-{a:g}"


def run_ar(args, entry, log, validation) -> tuple[list, dict]:
    from transformers import AutoTokenizer
    from sidlens.grid import decode_ar as DA, inputs as GI

    ckpt = args.ckpt
    vocab = ar.load_vocab(ckpt)
    tok = AutoTokenizer.from_pretrained(str(ar.checkpoint_dir(ckpt)), local_files_only=True)
    gin = GI.ArSetup(ckpt, "test", tok, vocab, task="next-item")
    validation.update(gin.validation)
    if not all(v["ok"] for v in gin.validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(gin.validation, default=str)[:1500]}")
    examples = sorted(gin.examples, key=lambda e: e.row)
    if args.limit:
        examples = examples[:args.limit]
    proto = GP.ar_protocol(entry, require_full=not args.limit)
    if proto["batch_size"] != 1:
        raise RuntimeError("the run was evaluated in batches > 1; reproducing it needs its full row order")
    saved = GP.load(entry)
    model = ar.load_model(ckpt, device=args.device, dtype=proto["dtype"] if args.device == "cuda" else "float32")
    grammar, _ = DA.catalog_grammar(tok, entry["sid_index"]["path"], proto["horizon"])
    torch.use_deterministic_algorithms(False)        # the evaluator's numerics
    torch.backends.cudnn.allow_tf32 = True
    plain = DA.decode_rows(model, tok, grammar, examples[:args.v2_rows], proto, log=log)
    validation["V2a_reproduction"] = GP.compare(plain, saved, "ar", only_saved=bool(args.limit))
    if not validation["V2a_reproduction"]["ok"]:
        raise RuntimeError("the plain decoder does not reproduce the run's saved lists")
    encoded = [P.encode(ex, tok, vocab, template=gin.template) for ex in examples]
    lists, counts = {}, {}
    for a in ALPHAS:
        out, c = DA.run_conditions(model, tok, grammar, examples, encoded, vocab, proto, ["C_all"], alpha=a, log=log)
        lists[a], counts[tag_of(a)] = out["C_all"], c["C_all"]
    m = min(64, len(examples))
    base, _ = DA.run_conditions(model, tok, grammar, examples[:m], encoded[:m], vocab, proto, ["B"], alpha=1.0)
    validation["V3_alpha1_is_unscaled_route"] = all(lists[1.0][e.example_id] == base["B"][e.example_id]
                                                    for e in examples[:m])
    validation["V2b_alpha1_vs_saved_plain"] = GP.compare(
        lists[1.0], saved, "ar", score_atol=1e-2, tie_atol=1e-2, max_tie_share=1.0, only_saved=True)
    validation["edges"] = counts
    frames = [DA.lists_frame(lists[a], examples, tag_of(a)) for a in ALPHAS]
    return frames, {"examples": [e.example_id for e in examples], "proto": proto}


def run_diffgrm(args, entry, log, validation) -> tuple[list, dict]:
    from sidlens.grid import decode_diff as DD, diff_setup as DS

    S = DS.load(args.ckpt, device=args.device)
    validation["V2a_grid_checks"] = DS.check(S, device=args.device)
    if not validation["V2a_grid_checks"]["ok"]:
        raise RuntimeError("grid checks failed")
    proto = GP.diffgrm_protocol(entry, require_full=not args.limit)
    c = S.cohort
    N = min(args.limit, len(c.users)) if args.limit else len(c.users)
    L = c.history_lengths[:N]
    slots = np.where(L >= 2, L - 1, -1)
    n = int(S.model.n_digit)
    frames, plain = [], None
    for a in ALPHAS:
        t0 = time.time()
        hyps = []
        for start in range(0, N, proto["batch_size"]):
            rows = list(range(start, min(start + proto["batch_size"], N)))
            sl = torch.as_tensor(slots[rows], device=args.device)
            enc = DS.encode_slot_scaled(S.model, c.histories[rows], c.history_mask[rows], slots[rows], a)
            hyps += DD.decode(S.model, None, S.catalog, beam_size=proto["beam_size"], top_k=proto["top_k"],
                              seed=proto["seed"], encoded=enc, decoder_context=DS.slot_scale_context(S.model, sl, a))
        lists = {c.example_ids[i]: [{"sids": [DS._sid_text(row) for row in h["codes"]], "score": h["score"]}
                                    for h in hyps[i]] for i in range(N)}
        if a == 1.0:
            plain = lists
        frames.append(pd.DataFrame([{"example_id": c.example_ids[i], "row": i, "user_id": c.users[i],
                                     "condition": tag_of(a), "eligible": bool(L[i] >= 2),
                                     "predict": [x["sids"][0] for x in lists[c.example_ids[i]]],
                                     "scores": [x["score"] for x in lists[c.example_ids[i]]]} for i in range(N)]))
        log(f"[{tag_of(a)}] {N} users, {time.time() - t0:.0f}s")
    v3 = GP.compare(plain, GP.load(entry), "diffgrm", score_atol=1e-4, tie_atol=1e-5, only_saved=bool(args.limit))
    validation["V3_alpha1_vs_saved_lists"] = {**v3, "gate": not S.sampled}
    if not v3["ok"] and not S.sampled:
        raise RuntimeError("alpha = 1 does not reproduce the run's saved lists")
    return frames, {"examples": list(c.example_ids[:N]), "proto": proto}


def score_rows(entry, rows: pd.DataFrame, frames) -> pd.DataFrame:
    """Per row and alpha: SID and item (uniform, lower) hit@10 and the history share of the top 10."""
    from sidlens.analysis.collisions import item_rank_bounds
    from sidlens.grid import inputs as GI
    table, items = GI.sid_table(entry), GI.item2id(entry)
    n = table.variant.n_codebook
    buckets = {sid: tuple(sorted(items[a] for a in asins)) for sid, asins in table.cb2items.items()}
    r = rows.set_index("example_id")
    out = []
    for f in frames:
        for eid, pred, user in zip(f.example_id, f.predict, f.user_id):
            codes = [P.parse_sid(x, n) for x in list(pred)[:10]]
            tgt, hist = tuple(r.at[eid, "target_codes"]), set(r.at[eid, "history_sids"])
            b = item_rank_bounds(list(dict.fromkeys(codes)), int(r.at[eid, "target_item"]), buckets,
                                 target_sid=tgt).hit_bounds(10)
            out.append({"example_id": eid, "user_id": user, "condition": f.condition.iloc[0],
                        "alpha": float(f.condition.iloc[0].split("-", 1)[1]), "cls": r.at[eid, "cls"],
                        "eligible": bool(f.eligible.loc[f.example_id == eid].iloc[0]) if "eligible" in f else True,
                        "sid_hit": float(tgt in codes), "item_uniform": b["uniform"], "item_lower": b["lower"],
                        "history_share": float(np.mean([x in hist for x in list(pred)[:10]])) if len(pred) else 0.0})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a next-item grid weight state, best (grid:<key>@best)")
    ap.add_argument("--v2-rows", type=int, default=512, help="AR: rows decoded plainly for V2a")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (pilot only)")
    ap.add_argument("--device", default="cuda", help="cpu runs float32 (smoke tests; bit-exact controls hold on GPU only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if grid.split_name(args.ckpt)[1] != grid.BEST:
        ap.error("N3 is declared for the best state (V2a needs the saved lists)")
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

    entry = registry.load().entry(args.ckpt)
    if entry["predtype"] != "next-item":
        ap.error("N3 is declared for next-item runs")
    validation: dict = {}
    frames, info = (run_ar if entry["paradigm"] == "ar" else run_diffgrm)(args, entry, log, validation)
    ids = info["examples"]
    cls = events.target_classes(entry, "test", ids)
    cls["target_codes"] = cls.target_codes.map(list)
    cls["variant_codes"] = cls.variant_codes.map(lambda v: [list(x) for x in v])
    from sidlens.grid import diff_setup as DS, inputs as GI
    sid_txt = {a: DS._sid_text(cd) for a, cd in GI.sid_table(entry).asin2codes.items()}
    rows_by = {r["example_id"]: r for r in events.cohort_rows(entry, "test")}
    cls["history_sids"] = [[sid_txt[a] for a in rows_by[e]["history_item_ids"]] for e in cls.example_id]
    cls["target_sid"] = [sid_txt[rows_by[e]["target_item_ids"][0]] for e in cls.example_id]
    cls.to_parquet(args.out / "rows.parquet", index=False)
    for f in frames:
        f.to_parquet(args.out / f"predictions_{f.condition.iloc[0]}.parquet", index=False)
    score_rows(entry, cls, frames).to_parquet(args.out / "scores.parquet", index=False)
    (args.out / "inputs.json").write_text(json.dumps({
        "checkpoint": args.ckpt, "grid_key": entry["key"], "paradigm": entry["paradigm"], "alphas": list(ALPHAS),
        "n_rows": len(ids), "limited": args.limit, "protocol": info["proto"],
        "device": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
        "torch": torch.__version__, **grid.cell_info(args.ckpt)},
        indent=1, default=str))
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {validation['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
