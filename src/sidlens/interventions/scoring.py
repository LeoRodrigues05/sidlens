"""Per-digit target scores at the positions that decide each digit.

An intervention's outcome is the change in how strongly the model backs the
golden code at each digit. Three denominators answer different questions, so
all three are kept:

    codes   log-softmax over the digit's code tokens (`vocab.ids(d)`): the
            model's actual decision set, which is smaller than codebook_size
            when codes are unused (RQ-VAE 4x128 digit 0 has 39).
    legal   log-softmax over the codes that extend the golden prefix to some
            catalogue SID: what a trie-constrained decoder chooses among.
    vocab   full-vocabulary log-prob: includes mass the model puts on text.

Traps, each with the guard that closes it
-----------------------------------------
1. **Wrong column.** Code c of digit d is column `vocab.codes(d).index(c)`,
   never `c`; codes are sparse and ids are string-sorted. `DigitScorer` maps
   through the vocabulary.
2. **Wrong position.** Logits for digit d are read at `predict_pos(0, d)`,
   passed in by the caller from `Encoded`, never `prompt_len + d`.
3. **Ties.** Rank counts codes scoring strictly higher than the target (same
   convention as `scripts/activations/ar_capture.py`), so tied codes share
   the better rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sidlens.data.sids import SidTable
from sidlens.models.ar import SidVocab

MEASURES = ("logp_codes", "rank", "top1", "logp_legal", "logp_vocab")


@dataclass
class DigitScorer:
    vocab: SidVocab
    table: SidTable
    _col: list[dict[int, int]] = field(init=False)
    _legal: list[dict[tuple, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        n = self.vocab.variant.n_codebook
        self._col = [{c: j for j, c in enumerate(self.vocab.codes(d))} for d in range(n)]
        self._legal = []
        codes = self.table.codes
        for d in range(n):
            groups: dict[tuple, np.ndarray] = {}
            for prefix in {tuple(r) for r in codes[:, :d].tolist()}:
                sel = np.all(codes[:, :d] == np.asarray(prefix, dtype=np.int64), axis=1) if d else \
                    np.ones(len(codes), bool)
                mask = np.zeros(len(self._col[d]), bool)
                for c in np.unique(codes[sel, d]).tolist():
                    mask[self._col[d][c]] = True    # KeyError = table code without a token
                groups[prefix] = mask
            self._legal.append(groups)

    @property
    def n_digits(self) -> int:
        return self.vocab.variant.n_codebook

    def score(self, logits, targets: np.ndarray) -> dict[str, np.ndarray]:
        """logits (R, n, V) at each digit's predict position; targets (R, n) codes.

        Returns measure -> (R, n) array. Float math is done in float32.
        """
        import torch

        R, n = targets.shape
        if tuple(logits.shape[:2]) != (R, n) or n != self.n_digits:
            raise ValueError(f"logits {tuple(logits.shape)} vs targets {targets.shape}")
        lg = logits.float()
        out = {k: np.empty((R, n), dtype=np.float64 if k.startswith("logp") else np.int64)
               for k in MEASURES}
        full_lse = torch.logsumexp(lg, dim=-1)                                   # (R, n)
        for d in range(n):
            ids = torch.as_tensor(self.vocab.ids(d), device=lg.device)
            sub = lg[:, d].index_select(-1, ids)                                  # (R, C_d)
            j = torch.as_tensor([self._col[d][int(t)] for t in targets[:, d]], device=lg.device)
            ar = torch.arange(R, device=lg.device)
            tgt = sub[ar, j]
            out["logp_codes"][:, d] = (tgt - torch.logsumexp(sub, -1)).cpu().numpy()
            out["rank"][:, d] = (sub > tgt[:, None]).sum(-1).cpu().numpy()
            codes_d = np.asarray(self.vocab.codes(d))
            out["top1"][:, d] = codes_d[sub.argmax(-1).cpu().numpy()]
            legal = np.stack([self._legal[d][tuple(int(c) for c in targets[r, :d])] for r in range(R)])
            masked = sub.masked_fill(~torch.as_tensor(legal, device=lg.device), float("-inf"))
            out["logp_legal"][:, d] = (tgt - torch.logsumexp(masked, -1)).cpu().numpy()
            vid = torch.as_tensor([self.vocab.id_of(d, int(t)) for t in targets[:, d]], device=lg.device)
            out["logp_vocab"][:, d] = (lg[ar, d, vid] - full_lse[:, d]).cpu().numpy()
        return out

    def code_logits(self, logits, d: int):
        """(R, C_d) logits over digit d's codes, code-ordered: what the exactness
        controls compare bit for bit."""
        import torch
        ids = torch.as_tensor(self.vocab.ids(d), device=logits.device)
        return logits[:, d].index_select(-1, ids)
