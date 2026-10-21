"""A grid DiffGRM weight state, in the form the DiffGRM experiments already consume.

The DiffGRM experiments (controlled exp1, exp2, exp7, exp8, exp12-exp14,
representation exp4-exp5) start from a sidlens registry entry, a
`DiffusionEvalCohort` and a loaded model, then check the model against
controlled exp1's archive. A grid state has none of those archives; it has the
run's own saved test lists instead. `load` builds the same three objects from
the grid (`sidlens.grid.inputs`), plus `compat`, a dict with the entry fields
the experiments read (`ckpt_path`, `ckpt_sha256`, `config`, ...), and `check`
replaces the archive check.

Checks (`check`)
----------------
V2a      the run's own decoder (`grid.decode_diff`, OneDiffRec's
         `joint_catalog_v1`) reproduces the saved lists of the first
         `n_users` cohort users. These come from the FULL cohort in its file
         order, so the evaluator's batches of 8 are the same even where the
         experiment samples users (Office).
scorer   the scorer sidlens's intervention code uses (`matched_decode.score_partial`,
         projected cross-attention cache, all positions at once) gives the
         same log-probabilities as the evaluator's per-digit forward along each
         user's top hypothesis, revealed left to right. This is what lets
         `interventions.diffusion` results speak about the grid's decoder.

Traps
-----
1. **Weight states without saved lists.** Only the best state was evaluated
   by OneDiffRec. For an intermediate snapshot V2a is not defined; `check`
   then runs the scorer check alone and says so.
2. **Smoke evaluations.** A run evaluated on a few users (smoke) has saved
   lists for those users only; V2a compares the users it has.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sidlens.grid import decode_diff as DD, inputs as GI, predictions as GP, registry


@dataclass
class DiffSetup:
    name: str
    entry: dict
    state: str
    compat: dict
    cohort: object
    model: object
    report: dict
    table: object
    item2id: dict
    n_full: int                      # users in the run's whole test cohort (sampled: fewer in `cohort`)

    @property
    def sampled(self) -> bool:
        return len(self.cohort.users) != self.n_full

    @property
    def catalog(self) -> list[tuple[int, ...]]:
        return list(self.table.asin2codes.values())


def _metrics(e: dict) -> dict:
    p = Path(e["run_dir"]) / "evaluation" / "metrics.json"
    if not p.exists():
        return {}
    m = json.loads(p.read_text())
    return {k: v for k, v in m.items() if isinstance(v, (int, float))}


def load(name: str, device: str = "cuda", *, sample: bool = True) -> DiffSetup:
    reg = registry.load()
    e, st = reg.state(name)
    if e["paradigm"] != "diffgrm" or e["predtype"] != "next-item":
        raise ValueError(f"{name}: the DiffGRM experiments are defined for next-item DiffGRM runs")
    model, report = GI.diffgrm_model(name, device)
    state = name.split("@")[-1]
    compat = {"ckpt_path": st["path"], "ckpt_sha256": report["sha256"], "sem_ids_name": e["variant"],
              "config": dict(e["config"]), "task": "next1", "status": "trained", "quantizer": e["method"],
              "codebook_size": int(e["size"]), "recorded_metrics": _metrics(e) if st.get("best") else {},
              "training": {"grid_key": e["key"], "state": state, "epoch": st.get("epoch"),
                           "best_epoch": e.get("best_epoch")},
              "grid_key": e["key"], "grid_state": state, "dataset": e["dataset"]}
    from sidlens.grid import events
    return DiffSetup(name, e, state, compat, GI.diffgrm_cohort(e, "test", sample=sample), model, report,
                     GI.sid_table(e), GI.item2id(e), len(events.cohort_rows(e, "test")))


def _sid_text(codes) -> str:
    return "".join(f"<{chr(ord('a') + d)}_{int(c)}>" for d, c in enumerate(codes))


def check(S: DiffSetup, n_users: int = 64, device: str = "cuda") -> dict:
    """V2a and the scorer check (module doc). Returns a report with `ok`."""
    import torch
    from sidlens.analysis import matched_decode as MD

    out = {"n_users": n_users}
    full = GI.diffgrm_cohort(S.entry, "test", sample=False)
    n = min(n_users, len(full.users))
    best = bool(S.entry["states"][S.state].get("best"))
    decoded = None
    if best:
        proto = GP.diffgrm_protocol(S.entry, require_full=False)
        p = json.loads((Path(S.entry["run_dir"]) / "evaluation" / "protocol.json").read_text())
        if not p.get("full_cohort", False):         # a smoke run: compare the users it has
            n = min(n, int(p["examples"]))
        decoded = DD.decode_cohort(S.model, full.histories[:n], full.history_mask[:n], S.catalog,
                                   beam_size=proto["beam_size"], top_k=proto["top_k"], seed=proto["seed"],
                                   batch_size=proto["batch_size"], device=device)
        mine = {full.example_ids[r]: [{"sids": [_sid_text(row) for row in h["codes"]], "score": h["score"]}
                                      for h in hyps] for r, hyps in enumerate(decoded)}
        out["V2a"] = GP.compare(mine, GP.load(S.entry), "diffgrm", score_atol=1e-4, tie_atol=1e-5)
        out["protocol"] = proto
    else:
        out["V2a"] = {"ok": True, "skipped": "an intermediate weight state has no saved test lists"}
    # scorer check along each user's top (or golden) SID, left to right
    m = min(n, 16)
    h = torch.as_tensor(full.histories[:m]).to(device)
    msk = torch.as_tensor(full.history_mask[:m]).to(device)
    with torch.inference_mode():
        enc = S.model.forward({"history_sid": h, "history_mask": msk}, return_loss=False).hidden_states
        cache = MD._project_cross_cache(S.model, enc)
        tops = (np.array([hyps[0]["codes"][0] for hyps in decoded[:m]]) if decoded is not None
                else np.asarray(full.target_sids[:m]))
        tops = torch.as_tensor(tops, device=device)
        depth = tops.shape[1]
        worst = 0.0
        for col in range(depth):
            partial = tops.clone()
            partial[:, col:] = -1
            a = MD.score_partial(S.model, enc, cache, partial, torch.arange(m, device=device))[:, col]
            b = S.model.forward_decoder_only({"decoder_input_ids": partial.clamp_min(0),
                                              "mask_positions": partial.eq(-1), "encoder_hidden": enc},
                                             digit=col, use_cache=True).logits.float().log_softmax(-1)
            worst = max(worst, float((a - b).abs().max()))
    out["scorer"] = {"users": m, "max_abs_diff_logprob": worst, "ok": worst <= 1e-4}
    out["ok"] = bool(out["V2a"]["ok"] and out["scorer"]["ok"])
    return out


def history_item_ids(S: DiffSetup) -> list[list[int]]:
    """Item ids of each cohort row's history, oldest first, as the model sees it
    (the most recent `max_history_len`); rows in cohort order."""
    from sidlens.grid import events
    rows = {r["example_id"]: r for r in events.cohort_rows(S.entry, "test")}
    H = S.cohort.histories.shape[1]
    out = []
    for eid, L in zip(S.cohort.example_ids, S.cohort.history_lengths):
        ids = [S.item2id[a] for a in rows[eid]["history_item_ids"]][-H:]
        if len(ids) != int(L):
            raise RuntimeError(f"{eid}: history length {len(ids)} != cohort {int(L)}")
        out.append(ids)
    return out


def as_arrays(hyps: list[list[dict]], n_digit: int, top_k: int = 10):
    """`decode_diff` hypothesis lists -> (codes [B, top_k, n] -1 padded, scores [B, top_k] -inf, counts [B])."""
    B = len(hyps)
    codes = np.full((B, top_k, n_digit), -1, np.int64)
    scores = np.full((B, top_k), -np.inf, np.float32)
    counts = np.zeros(B, np.int64)
    for b, hs in enumerate(hyps):
        for j, h in enumerate(hs[:top_k]):
            codes[b, j] = h["codes"][0]
            scores[b, j] = h["score"]
        counts[b] = min(len(hs), top_k)
    return codes, scores, counts


def slot_block_context(model, slots, n_slots: int):
    """`decode_diff` decoder_context blocking every digit's cross-attention to slot
    `slots[user]` (-1: none) in every decoder block, for the rows of each step."""
    import torch
    from sidlens.interventions import diffusion as DI

    def ctx(owners, column, states):
        sl = slots[owners]
        R, W = states.shape
        xm = torch.ones((len(model.decoder_blocks), R, W, n_slots), device=states.device)
        has = (sl >= 0).nonzero()[:, 0]
        if len(has):
            xm[:, has, :, sl[has]] = 0
        return DI.cross_attention_masks(model, xm)
    return ctx


def encode_slot_scaled(model, hist, mask, slots, alpha: float):
    """Encoder states with every OTHER real slot's attention to slot `slots[b]`
    (-1: none) scaled by `alpha` after the softmax and renormalised (exp14's
    encoder-side block at alpha = 0). alpha = 1 is the plain encoder bit for bit."""
    import torch
    from sidlens.interventions.attn_scale import reweight_attention
    from sidlens.models import diffusion

    h, m = torch.as_tensor(hist), torch.as_tensor(mask, dtype=torch.bool)
    dev = next(model.parameters()).device
    if alpha == 1:
        return diffusion.encode(model, h.to(dev), m.to(dev))
    sl = torch.as_tensor(slots, device=dev)
    mm = m.to(dev)

    def wfn(name, att):                                     # att [B, H, S, S]
        w = torch.ones_like(att)
        for b in (sl >= 0).nonzero()[:, 0].tolist():
            k = int(sl[b])
            q = mm[b].clone()
            q[k] = False                                     # the slot itself and padding keep their reads
            w[b, :, q, k] = alpha
        return w

    mods = {f"enc{L}": blk.attn for L, blk in enumerate(model.encoder_blocks)}
    with reweight_attention(mods, wfn):
        return diffusion.encode(model, h.to(dev), m.to(dev))


def slot_scale_context(model, slots, alpha: float):
    """`decode_diff` decoder_context scaling every digit's cross-attention to slot
    `slots[user]` by `alpha` (exp14's decoder-side block at alpha = 0), or None at alpha = 1."""
    import torch
    from sidlens.interventions.attn_scale import reweight_attention

    if alpha == 1:
        return None
    mods = {f"x{L}": blk.cross_attn for L, blk in enumerate(model.decoder_blocks)}

    def ctx(owners, column, states):
        sl = slots[owners]

        def wfn(name, att):                                 # att [R, H, W, S]
            w = torch.ones_like(att)
            has = (sl >= 0).nonzero()[:, 0]
            if len(has):
                w[has, :, :, sl[has]] = alpha
            return w
        return reweight_attention(mods, wfn)
    return ctx
