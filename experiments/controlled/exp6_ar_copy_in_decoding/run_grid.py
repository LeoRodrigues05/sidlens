#!/usr/bin/env python
"""exp6 on a OneDiffRec grid run: copy reads blocked inside the run's own AR decoder.

Same question, conditions and edges as `run.py` (protocol.md): when a beam
predicts digit d, its reads of digit d of history items whose first d digits
equal the beam's prefix are removed in the later half of the layers (C_later;
C_all also removes every first-digit read; N_later removes as many reads of
non-matching items). What differs is the decoder, because the grid's saved
lists come from OneDiffRec's evaluator, not the archived sampler: plain
constrained beam search, then raw-vocabulary rescoring of every returned SID
(`sidlens.grid.decode_ar`). The edges are applied in both phases.

Gates, in order
---------------
V1   the run's inputs (rows, SID table, vocabulary, prompts token for token).
V2a  the plain decoder reproduces the run's saved lists on the first
     `--v2-rows` rows in CSV order (`grid.predictions.compare`); otherwise the
     run stops. `B_plain` is then the run's saved list for every row: it is
     exactly the plain decode, and decoding it again would add a fifth decode.
V2b  B (the edit route with an empty plan) against B_plain: same lists up to
     float noise of the explicit-mask kernel; reported, not a gate. Every
     contrast is against B, so conditions differ only by their edges.

    python run_grid.py --ckpt 'grid:<key>@best' --out <new dir> [--limit 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sidlens.data import ar_prompts as P                                 # noqa: E402
from sidlens.models import ar                                            # noqa: E402
from sidlens.provenance import hashing                                   # noqa: E402

CONDITIONS = ("B", "C_later", "C_all", "N_later")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a next-item AR grid weight state (grid:<key>@best)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--v2-rows", type=int, default=512, help="rows decoded plainly for V2a (CSV order)")
    ap.add_argument("--limit", type=int, default=None, help="first N rows in CSV order (pilot only)")
    ap.add_argument("--device", default="cuda", help="cpu runs float32 (smoke tests; bit-exact controls hold on GPU only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    conds = args.conditions.split(",")
    if not set(conds) <= set(CONDITIONS) or "B" not in conds:
        ap.error(f"--conditions is a subset of {CONDITIONS} that contains B")
    if not ar.is_grid(args.ckpt):
        ap.error("run_grid.py takes grid weight states; frozen cells use run.py")
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

    from transformers import AutoTokenizer
    from sidlens.grid import decode_ar as DA, events as GE, inputs as GI, predictions as GP

    ckpt = args.ckpt
    vocab = ar.load_vocab(ckpt)
    n = vocab.variant.n_codebook
    tok = AutoTokenizer.from_pretrained(str(ar.checkpoint_dir(ckpt)), local_files_only=True)
    gin = GI.ArSetup(ckpt, "test", tok, vocab, task="next-item")
    validation = dict(gin.validation)
    if not all(v["ok"] for v in validation.values()):
        raise RuntimeError(f"input validation failed: {json.dumps(validation, default=str)[:2000]}")
    examples = sorted(gin.examples, key=lambda e: e.row)             # the evaluator's order
    if args.limit:
        examples = examples[:args.limit]
    proto = GP.ar_protocol(gin.entry, require_full=not args.limit)
    if proto["batch_size"] != 1:
        raise RuntimeError("the run was evaluated in batches > 1; reproducing it needs its full row order")
    saved = GP.load(gin.entry)
    missing = [e.example_id for e in examples if e.example_id not in saved]
    if missing and not args.limit:
        raise RuntimeError(f"{len(missing)} rows have no saved list, e.g. {missing[:3]}")

    # strata fixed before any decode: target class, review days, history SIDs
    cls = GE.target_classes(gin.entry, "test", [e.example_id for e in examples])
    days, _ = GE.row_times(gin.entry, "test", [e.example_id for e in examples])
    rows = cls.drop(columns=["row"]).merge(
        days[["example_id", "gap1_days", "same_day_as_target", "dup_target", "time_ambiguous"]],
        on="example_id", validate="one_to_one")
    rows["row"] = [e.row for e in examples]
    rows["target_sid"] = [e.target_sids[0] for e in examples]
    rows["history_sids"] = [list(e.history_sids) for e in examples]
    rows["target_codes"] = rows.target_codes.map(list)
    rows["variant_codes"] = rows.variant_codes.map(lambda v: [list(x) for x in v])
    rows.to_parquet(args.out / "rows.parquet", index=False)

    sha = ar.checkpoint_sha(ckpt)
    inputs = {"checkpoint": f"ar/{ckpt}", "checkpoint_sha256": sha, "variant": vocab.variant.name,
              "n_digits": n, "split": "test", "task": "next-item", "template": gin.template,
              "csv": str(gin.csv), "csv_sha256": hashing.sha256_file(gin.csv),
              "sid_index_sha256": gin.entry["sid_index"]["sha256"], "decoder": "onediffrec evaluate_ar_joint",
              "protocol": proto, "n_rows": len(examples), "n_users": len({e.user_id for e in examples}),
              "limited": args.limit, "conditions": conds, "v2_rows": args.v2_rows,
              "classes": rows.cls.value_counts().to_dict(), **gin.info}
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))
    (args.out / "arguments.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    log(f"inputs ok: {ckpt} rows={len(examples)} classes={inputs['classes']} protocol={proto}")

    model = ar.load_model(ckpt, device=args.device, dtype=proto["dtype"] if args.device == "cuda" else "float32")
    grammar, _ = DA.catalog_grammar(tok, gin.entry["sid_index"]["path"], proto["horizon"])
    torch.use_deterministic_algorithms(False)       # the evaluator ran with torch's default numerics
    torch.backends.cudnn.allow_tf32 = True
    inputs.update({"device": torch.cuda.get_device_name() if args.device == "cuda" else args.device,
                   "torch": torch.__version__,
                   "transformers": __import__("transformers").__version__,
                   "late_layers": list(DA.late_layers(model))})
    (args.out / "inputs.json").write_text(json.dumps(inputs, indent=1, default=str))

    # ---- V2a ---------------------------------------------------------------
    t0 = time.time()
    v2 = examples[:args.v2_rows]
    plain = DA.decode_rows(model, tok, grammar, v2, proto, log=log)
    validation["V2a_reproduction"] = GP.compare(plain, saved, "ar", only_saved=bool(args.limit))
    log(f"[V2a] {json.dumps({k: validation['V2a_reproduction'][k] for k in ('rows', 'identical', 'tie_reordered', 'failed', 'ok')})}"
        f" {time.time() - t0:.0f}s")
    if not validation["V2a_reproduction"]["ok"]:
        (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
        raise RuntimeError("the plain decoder does not reproduce the run's saved lists; exp6 stops")
    b_plain = {e.example_id: saved[e.example_id]["predictions"] for e in examples if e.example_id in saved}
    DA.lists_frame(b_plain, examples, "B_plain").to_parquet(args.out / "predictions_B_plain.parquet", index=False)

    # ---- conditions ----------------------------------------------------------
    encoded = [P.encode(ex, tok, vocab, template=gin.template) for ex in examples]
    lists, counts = DA.run_conditions(model, tok, grammar, examples, encoded, vocab, proto, conds, log=log)
    for cond in conds:
        DA.lists_frame(lists[cond], examples, cond).to_parquet(args.out / f"predictions_{cond}.parquet", index=False)
    validation["V2b_edit_route_vs_plain"] = GP.compare(lists["B"], {k: {"predictions": v} for k, v in b_plain.items()},
                                                       "ar", score_atol=1e-2, tie_atol=1e-2, max_tie_share=1.0)
    validation["V3_edges"] = counts
    validation["seconds"] = round(time.time() - t_start, 1)
    (args.out / "validation.json").write_text(json.dumps(validation, indent=1, default=str))
    log(f"done in {validation['seconds']}s; V2b {json.dumps({k: validation['V2b_edit_route_vs_plain'][k] for k in ('identical', 'tie_reordered', 'failed')})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
