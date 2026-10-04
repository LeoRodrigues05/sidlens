"""Fixed-shape forwards that apply interventions and score every target digit.

Shared by the AR intervention experiments (exp4 onward; exp3 keeps its own
copy so its archived source stays what ran). One forward = B jobs, each job an
encoded input plus optional residual patches and attention knockouts, scored
at `predict_pos(slot, digit)` for every target digit of every slot.

Why the shape is fixed
----------------------
bf16 kernels are not invariant to batch shape, but for a fixed (B, T) shape a
row's output depends only on that row. Every forward therefore has B rows (a
short batch is filled with copies of a real row whose outputs are dropped) and
T_pad tokens (right padding, explicit `position_ids`), and passes the same
explicit (B, H, T, T) boolean mask kind to every layer, whether or not it
knocks anything out. That is what makes the no-op, self-patch and full-restore
controls bit-exact, and differences between conditions attributable to the
intervention alone.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field

import numpy as np

from sidlens import hooks as H
from sidlens.data import ar_prompts as P
from sidlens.hooks.attention_probs import attention_probs
from sidlens.interventions.attention import Knockout, base_mask, knockout_attention
from sidlens.interventions.residual import Patch, patch_residual
from sidlens.interventions.scoring import MEASURES, DigitScorer


@dataclass
class Job:
    """One row of a forward.

    patches    [(site, positions, source)]: residual values taken from
               `caches[source][site][cache_row, positions]`
    knockouts  [(queries, keys, layers, heads, expect_masked)]: attention edges
               removed for this row (see `interventions.attention`)
    """
    enc: P.Encoded
    cache_row: int = 0
    patches: list = field(default_factory=list)
    knockouts: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class FixedShapeRunner:
    def __init__(self, model, scorer: DigitScorer, pad_id: int, t_pad: int, rows: int,
                 n_slots: int):
        self.model, self.scorer, self.pad_id = model, scorer, pad_id
        self.t_pad, self.B = t_pad, rows
        self.dev = next(model.parameters()).device
        self.n = scorer.n_digits
        self.cols = [(s, d) for s in range(n_slots) for d in range(self.n)]
        self.n_heads = model.config.num_attention_heads
        self.n_forwards = 0

    def _tensors(self, encs):
        import torch
        c = P.collate(encs, pad_id=self.pad_id, side="right")
        if int(c["offset"].abs().max()) != 0:
            raise AssertionError("right padding must leave every offset at 0")
        extra = self.t_pad - c["input_ids"].shape[1]
        if extra < 0:
            raise ValueError(f"sequence longer than T_pad={self.t_pad}")
        pad = torch.nn.functional.pad
        return (pad(c["input_ids"], (0, extra), value=self.pad_id).to(self.dev),
                pad(c["attention_mask"], (0, extra), value=0).to(self.dev),
                pad(c["position_ids"], (0, extra), value=1).to(self.dev))

    def _patches(self, jobs, caches):
        import torch
        acc: dict[str, list] = {}
        for r, job in enumerate(jobs):
            for site, positions, source in job.patches:
                p = torch.as_tensor(positions, dtype=torch.long)
                vals = caches[source][site][job.cache_row, p.to(self.dev)]
                a = acc.setdefault(site, [[], [], []])
                a[0].append(torch.full((len(p),), r, dtype=torch.long))
                a[1].append(p)
                a[2].append(vals)
        return [Patch(s, torch.cat(a[0]), torch.cat(a[1]), torch.cat(a[2])) for s, a in acc.items()]

    def targets(self, enc: P.Encoded) -> list[int]:
        codes = [P.parse_sid(s, self.n) for s in enc.example.target_sids]
        return [codes[s][d] for s, d in self.cols]

    def forward(self, jobs: list[Job], caches: dict | None = None,
                capture: list[str] | None = None, observe: dict | None = None):
        """Score `jobs`. Returns (scores, code_logits, capture, observed).

        scores       measure -> (R, J) array, J = len(self.cols)
        code_logits  list over columns of (R, C_d) float32 tensors (for exactness checks)
        observe      {"layers": [...], "pairs": [(job_index, query_pos), ...]} to
                     record per-head attention probabilities (observation only)
        """
        import torch

        real = len(jobs)
        if not 0 < real <= self.B:
            raise ValueError(f"batch of {real} jobs, capacity {self.B}")
        if capture and any(j.patches or j.knockouts for j in jobs):
            raise ValueError("capture during an intervened forward would record intervened values")
        encs = [j.enc for j in jobs] + [jobs[0].enc] * (self.B - real)
        ids, am, pos = self._tensors(encs)
        mask = base_mask(am, self.n_heads)
        pred = torch.tensor([[e.predict_pos(s, d) for s, d in self.cols] for e in encs],
                            device=self.dev)
        kos = [Knockout(r, tuple(q), tuple(k), tuple(layers), None if heads is None else tuple(heads),
                        expect_masked=bool(noop))
               for r, job in enumerate(jobs) for q, k, layers, heads, noop in job.knockouts]
        with ExitStack() as st:
            st.enter_context(knockout_attention(self.model, mask, kos))
            st.enter_context(patch_residual(self.model, self._patches(jobs, caches)))
            cap = st.enter_context(H.capture(self.model, names=capture, to_cpu=False)) if capture else None
            obs = st.enter_context(attention_probs(
                self.model, observe["layers"], [p[0] for p in observe["pairs"]],
                [p[1] for p in observe["pairs"]])) if observe else None
            with torch.no_grad():
                h = self.model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                                     use_cache=False).last_hidden_state
                logits = self.model.lm_head(h[torch.arange(self.B, device=self.dev)[:, None], pred])
        self.n_forwards += 1
        logits = logits[:real]
        tg = np.array([self.targets(j.enc) for j in jobs])
        J = len(self.cols)
        scores = {k: np.empty((real, J), dtype=np.float64 if k.startswith("logp") else np.int64)
                  for k in MEASURES}
        for s in sorted({s for s, _ in self.cols}):
            idx = [j for j, (ss, _) in enumerate(self.cols) if ss == s]
            sc = self.scorer.score(logits[:, idx], tg[:, idx])
            for k in MEASURES:
                scores[k][:, idx] = sc[k]
        code_logits = []
        for j, (_, d) in enumerate(self.cols):
            ids_d = torch.as_tensor(self.scorer.vocab.ids(d), device=logits.device)
            code_logits.append(logits[:, j].index_select(-1, ids_d).float().cpu())
        return scores, code_logits, cap, obs


def records(runner: FixedShapeRunner, jobs: list[Job], scores: dict, batch: int) -> list[dict]:
    """One dict per (job, scored column) with the job's meta and every measure."""
    out = []
    for r, job in enumerate(jobs):
        ex = job.enc.example
        tg = runner.targets(job.enc)
        for j, (s, d) in enumerate(runner.cols):
            rec = {"example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id,
                   "hist_len": len(ex.history_item_ids), **job.meta, "slot": s, "digit": d,
                   "target_code": tg[j]}
            rec.update({k: scores[k][r, j].item() for k in MEASURES})
            rec["batch"] = batch
            out.append(rec)
    return out


def max_diff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b))
