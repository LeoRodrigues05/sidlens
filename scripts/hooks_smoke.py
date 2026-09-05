#!/usr/bin/env python
"""Prove the capture layer works on the REAL checkpoints, not just toy modules.

`tests/test_hooks.py` covers the structural behaviour on synthetic stacks in a
second. What it cannot cover is whether the registered read positions actually
exist and fire on a 3 GB Qwen2 and on the vendored DIFF_GRM -- and that is the
part that breaks when an architecture assumption is wrong.

This runs under SLURM rather than on a login node. Loading the AR safetensors
off Lustre from a login node was measured at 35 minutes of uninterruptible disk
wait before being killed; on a compute node it is a normal read.

Writes a JSON report so a later run can be diffed against it: if a capture that
used to yield 28 layers starts yielding 1, that is visible without rerunning
anything.

    python scripts/hooks_smoke.py                 # both paradigms
    python scripts/hooks_smoke.py --skip-ar       # diffusion only, fast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch                                              # noqa: E402

from sidlens import hooks as H, paths                     # noqa: E402
from sidlens.data.sids import SidTable                    # noqa: E402
from sidlens.models import ar as AR                       # noqa: E402
from sidlens.models import diffusion as DIF               # noqa: E402


def smoke_diffusion(ckpt_id: str, n_hist: int = 6) -> dict:
    t0 = time.time()
    model, report = DIF.load_by_id(ckpt_id)
    variant = SidTable.load(
        json.loads((paths.MANIFESTS / "registry.diffusion.json").read_text()
                   )[ckpt_id]["sem_ids_name"])
    K = variant.variant.codebook_size
    nd = variant.variant.n_codebook

    # A real history: the first few catalogue items, as codebook ids.
    hist = torch.tensor(
        [[list(variant.asin2codes[k]) for k in variant.keys[:n_hist]]],
        dtype=torch.long)

    enc = DIF.encode(model, hist)
    dec_in = torch.zeros(1, nd, dtype=torch.long)
    out, cap = H.run_capture(
        model, lambda: DIF.digit_logits(model, enc, dec_in),
        names=[p for p in H.residual_points(model) if p.startswith("decoder")])

    # The decoder must actually fire. `forward(return_loss=False)` returns after
    # the encoder, so a capture taken through it yields nothing -- which is the
    # exact failure this asserts against.
    assert len(cap) > 0, "no decoder blocks captured -- wrong forward path"
    assert tuple(out.shape) == (1, nd, K), out.shape
    return {
        "checkpoint": ckpt_id,
        "paradigm": "diffusion",
        "shapes_ok": report["ok"],
        "encoder_hidden": list(enc.shape),
        "digit_logits": list(out.shape),
        "captured": cap.summary(),
        "call_counts": cap.call_counts,
        "seconds": round(time.time() - t0, 1),
    }


def smoke_ar(ckpt: str, n_hist: int = 3) -> dict:
    t0 = time.time()
    vocab = AR.load_vocab(ckpt)
    val = AR.validate_against(vocab)
    model = AR.load_model(ckpt)
    table = SidTable.load(vocab.variant)

    ids: list[int] = []
    for k in table.keys[:n_hist]:
        ids.extend(vocab.tokens_for(table.asin2codes[k]))
    x = torch.tensor([ids], dtype=torch.long)

    pts = H.residual_points(model)
    out, cap = H.run_capture(model, lambda: model(x), names=pts, dtype="float16")
    assert len(cap) == len(pts), f"{len(cap)} of {len(pts)} points fired"

    last = out.logits[0, -1]
    per_digit = {d: list(AR.digit_logits(last, vocab, d).shape)
                 for d in vocab.digits}
    return {
        "checkpoint": ckpt,
        "paradigm": "ar",
        "variant": vocab.variant.name,
        "vocab_ok": val["ok"],
        "n_codes_per_digit": {d: vocab.n_codes(d) for d in vocab.digits},
        "nominal_codebook_size": vocab.variant.codebook_size,
        "input_len": len(ids),
        "logits": list(out.logits.shape),
        "digit_logit_shapes": per_digit,
        "n_captured": len(cap),
        "captured_first": cap.summary()[:2],
        "captured_last": cap.summary()[-1:],
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ar", action="store_true")
    ap.add_argument("--skip-diffusion", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    results = []
    if not args.skip_diffusion:
        for variant, (_, diff_id) in sorted(AR.MATCHED_CELLS.items()):
            print(f"[smoke] diffusion {diff_id}", flush=True)
            r = smoke_diffusion(diff_id)
            print(f"        {r['digit_logits']} logits, "
                  f"{len(r['captured'])} decoder blocks, {r['seconds']}s", flush=True)
            results.append(r)
    if not args.skip_ar:
        for ckpt in sorted(AR.AR_CHECKPOINTS):
            print(f"[smoke] ar {ckpt}", flush=True)
            r = smoke_ar(ckpt)
            print(f"        {r['n_captured']} read points, logits {r['logits']}, "
                  f"codes/digit {r['n_codes_per_digit']}, {r['seconds']}s", flush=True)
            results.append(r)

    out = Path(args.out) if args.out else paths.DERIVED / "hooks_smoke" / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "snapshot": (paths.MANIFESTS / "CURRENT").read_text().strip(),
        "torch": torch.__version__,
        "results": results,
    }, indent=2))
    print(f"[smoke] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
