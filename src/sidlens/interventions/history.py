"""Replace one history item of an AR prompt with a prefix-matched catalogue control.

Experiment 1's "history patching" replaces one past interaction at a time and
asks which output digits move. For a SID recommender the replacement has a
natural dial: the control can share the first m digits of the item it
replaces and differ from digit m on. Because every SID digit is exactly one
added token and the separators are unchanged, the intervened prompt has the
same length and the same token at every position except item k's digits >= m.
That makes the input intervention token-aligned with the clean run, which is
what activation patching between the two needs.

Traps, each with the guard that closes it
-----------------------------------------
1. **Hand-edited token ids.** Editing ids directly skips the prompt checks in
   `ar_prompts.encode` (trie key, SID order, table agreement). The intervened
   example is rebuilt as an `ArExample` and re-encoded; `check_replacement`
   then proves the ids differ from the clean ids only at item k's digit
   positions >= m, and do differ at digit m.
2. **Leaking the answer.** A control carrying the target's item id or SID
   would inject the answer into the history. Candidates whose item id or full
   SID occurs anywhere in the row's history or target are excluded.
3. **Out-of-catalogue SIDs.** A made-up code combination is a string the model
   never saw in any history. Controls are real catalogue items, so every
   intervened history is one the training distribution could contain.
4. **"Shares m digits" versus "shares at least m digits".** The control
   equals item k on digits 0..m-1 and DIFFERS at digit m, so level m isolates
   the contribution of digits >= m. Later digits may coincide by chance.
5. **Irreproducible draws.** Each draw is seeded by sha256 of (seed, example
   id, k, m), so a condition is reproducible alone, independent of iteration
   order or of which other conditions were run.
6. **No candidate.** Deep prefixes can have no sibling (RQ-VAE's last digit is
   nearly an identity digit). `draw` returns None; callers must record the
   condition as missing, never drop it silently.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass

import numpy as np

from sidlens.data import ar_prompts as P
from sidlens.data.sids import SidTable

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def sid_string(codes) -> str:
    """(28, 33, 113) -> '<a_28><b_33><c_113>', the form the CSVs and prompts use."""
    return "".join(f"<{LETTERS[d]}_{int(c)}>" for d, c in enumerate(codes))


def condition_rng(seed: int, example_id: str, k: int | str, m: int) -> np.random.Generator:
    h = hashlib.sha256(f"{seed}|{example_id}|{k}|{m}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


@dataclass(frozen=True)
class Control:
    item_id: int
    codes: tuple[int, ...]
    n_candidates: int

    @property
    def sid(self) -> str:
        return sid_string(self.codes)


class ControlPool:
    """Catalogue items, ordered by item id, for prefix-matched draws."""

    def __init__(self, table: SidTable, item2id: dict[str, int]):
        missing = [a for a in table.asin2codes if a not in item2id]
        if missing:
            raise KeyError(f"{len(missing)} SID-table items have no integer id, e.g. {missing[:3]}")
        ids = np.array([item2id[a] for a in table.asin2codes], dtype=np.int64)
        codes = np.array(list(table.asin2codes.values()), dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        self.item_ids = ids[order]
        self.codes = codes[order]
        self.n_digits = table.variant.n_codebook
        if len(np.unique(self.item_ids)) != len(self.item_ids):
            raise ValueError("duplicate item ids in the catalogue")

    def candidates(self, orig: tuple[int, ...], m: int, exclude_ids, exclude_codes) -> np.ndarray:
        """Row indices of items sharing exactly digits 0..m-1 with `orig`."""
        if not 0 <= m < self.n_digits:
            raise ValueError(f"shared-prefix level m must be in [0, {self.n_digits}), got {m}")
        o = np.asarray(orig, dtype=np.int64)
        mask = np.all(self.codes[:, :m] == o[:m], axis=1) & (self.codes[:, m] != o[m])
        mask &= ~np.isin(self.item_ids, np.fromiter(exclude_ids, dtype=np.int64))
        for c in exclude_codes:
            mask &= ~np.all(self.codes == np.asarray(c, dtype=np.int64), axis=1)
        return np.flatnonzero(mask)

    def draw(self, ex: P.ArExample, k: int, m: int, seed: int) -> Control | None:
        """The control for replacing history item k at shared-prefix level m, or None."""
        return self._draw(ex, ex.history_sids[k], k, m, seed)

    def draw_target(self, ex: P.ArExample, slot: int, m: int, seed: int) -> Control | None:
        """The control for replacing target slot `slot` (e.g. item 1 of a next-two
        row). Seeded under the key "t<slot>", so it never coincides with a
        history draw for the same example."""
        return self._draw(ex, ex.target_sids[slot], f"t{slot}", m, seed)

    def _draw(self, ex: P.ArExample, sid: str, key, m: int, seed: int) -> Control | None:
        orig = P.parse_sid(sid, self.n_digits)
        exclude_ids = {*ex.history_item_ids, *ex.target_item_ids}
        exclude_codes = {P.parse_sid(s, self.n_digits) for s in (*ex.history_sids, *ex.target_sids)}
        idx = self.candidates(orig, m, exclude_ids, exclude_codes)
        if len(idx) == 0:
            return None
        j = idx[condition_rng(seed, ex.example_id, key, m).integers(len(idx))]
        return Control(int(self.item_ids[j]), tuple(int(c) for c in self.codes[j]), len(idx))


def replace_history_item(ex: P.ArExample, k: int, control: Control) -> P.ArExample:
    """`ex` with history item k swapped for `control`. The example id is kept:
    it names the clean row, and the condition is recorded beside it."""
    if not 0 <= k < len(ex.history_item_ids):
        raise IndexError(f"{ex.example_id}: history index {k} outside 0..{len(ex.history_item_ids) - 1}")
    ids = list(ex.history_item_ids)
    sids = list(ex.history_sids)
    ids[k], sids[k] = control.item_id, control.sid
    return dataclasses.replace(ex, history_item_ids=tuple(ids), history_sids=tuple(sids))


def replace_target_slot(ex: P.ArExample, slot: int, control: Control) -> P.ArExample:
    """`ex` with target slot `slot` set to `control` (teacher-forced clamp)."""
    if not 0 <= slot < len(ex.target_sids):
        raise IndexError(f"{ex.example_id}: target slot {slot} outside 0..{len(ex.target_sids) - 1}")
    ids, sids = list(ex.target_item_ids), list(ex.target_sids)
    ids[slot], sids[slot] = control.item_id, control.sid
    return dataclasses.replace(ex, target_item_ids=tuple(ids), target_sids=tuple(sids))


def check_replacement(clean: P.Encoded, new: P.Encoded, k: int, m: int,
                      role: str = "hist_sid") -> list[int]:
    """Prove `new` is `clean` with only item k's digits >= m changed.

    `role="target_sid"` checks a target-slot clamp (k is then the slot).

    Returns the changed positions. Raises if the lengths, role map or prompt
    length differ, if any other position changed, or if digit m did not.
    """
    if len(clean) != len(new) or clean.prompt_len != new.prompt_len:
        raise ValueError(f"{clean.example.example_id}: intervened length "
                         f"{len(new)}/{new.prompt_len} != clean {len(clean)}/{clean.prompt_len}")
    if (clean.role, clean.item, clean.digit) != (new.role, new.item, new.digit):
        raise ValueError(f"{clean.example.example_id}: intervened position map differs")
    changed = [i for i, (a, b) in enumerate(zip(clean.input_ids, new.input_ids)) if a != b]
    allowed = {i for i in clean.positions(role, item=k) if clean.digit[i] >= m}
    if stray := sorted(set(changed) - allowed):
        raise ValueError(f"{clean.example.example_id}: positions {stray[:5]} changed outside "
                         f"item {k} digits >= {m}")
    must = clean.positions(role, item=k, digit=m)
    if len(must) != 1 or must[0] not in changed:
        raise ValueError(f"{clean.example.example_id}: item {k} digit {m} did not change")
    return changed
