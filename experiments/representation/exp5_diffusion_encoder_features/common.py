"""Declared design, data construction and pure helpers for representation/exp5.

Everything here takes arrays and returns arrays (no model, no file writes), so
the parts that decide what a number means are tested on toy inputs in
`tests/experiments/test_encoder_features_experiment.py`. `run.py` adds the
model, the gates and the writing.

Traps, each with the guard that closes it
-----------------------------------------
1. **Rebuilt histories that are not the cohort's.** The analysis histories are
   rebuilt from the sequences (they need item ids, which the cohort does not
   keep); `run.py` refuses them unless their SIDs equal `cohort.histories`
   (V3). The fit split uses the same cut rule on earlier targets.
2. **A census family that is only a correlate.** A latent that fires on items
   with first code c also fires, in other slots, more often when r1's first
   code is c, because a user's items are similar. Each latent is assigned the
   ONE family with the largest chance-corrected selectivity, so a latent is
   called a spread latent only if r1's code is better explained than its own.
3. **A control that ablates less.** The control removes as many latents as the
   condition, chosen nearest in activation, from the same slots; a row whose
   condition removes nothing contributes no pair, and the count is reported.
4. **A missing label read as a class.** Labels use -1 for missing. Probes
   filter it; the description autoencoder turns it into its own word
   ("unknown") because a description must say something about every slot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

CATEGORY = "Industrial_and_Scientific"
CELLS = tuple(f"diff-next1-{q}-{d}cb-{w}" for q in ("MQ", "rqkmeans", "rqvae")
              for d in (3, 4, 5) for w in (128, 512)) + ("diff-next1-MQ-5cb-256", "diff-next1-rqvae-3cb-256")
SEED = 20261014
REC_LABELS = ("1", "2", "3", "4", "5", "6-10", "11+")
LABEL_FIELDS = ("cat_l1", "cat_l2", "main_cat", "price_decile")
DESCRIPTION_FIELDS = ("rec", "cat_l1", "cat_l2", "main_cat", "price_decile", "own_d0", "r1_cat_l2", "r1_d0")
KAPPA_MIN, SUPPORT_MIN = 0.5, 20


def user_bucket(user: str, mod: int) -> int:
    return int(hashlib.sha256(f"{SEED}|{user}".encode()).hexdigest(), 16) % mod


def rec_bin(r: np.ndarray) -> np.ndarray:
    """Recency r >= 1 (1 = most recent) -> bins 1, 2, 3, 4, 5, 6-10, 11+ as 0..6."""
    r = np.asarray(r)
    if bool((r < 1).any()):
        raise ValueError("recency starts at 1")
    return np.where(r <= 5, r - 1, np.where(r <= 10, 5, 6)).astype(np.int64)


@dataclass(frozen=True)
class Histories:
    """One row per (user, cut). Items are canonical item ids, -1 on padding."""

    user_index: np.ndarray        # [N] index into the cohort's user order
    items: np.ndarray             # [N, S]
    sids: np.ndarray              # [N, S, depth], -1 on padding
    mask: np.ndarray              # [N, S] bool
    lengths: np.ndarray           # [N]
    target_items: np.ndarray      # [N]
    target_sids: np.ndarray       # [N, depth]

    def __len__(self) -> int:
        return len(self.lengths)


def build_histories(users, sequences: dict, item2id: dict, item_sids: np.ndarray, split: str,
                    slots: int = 50) -> Histories:
    """Histories of the analysis ("test") or fit ("fit") split; see protocol.md.

    ``item_sids`` [n_items, depth] is the SID of each canonical item id.
    test: the final item is the target. fit: every target index t with
    2 <= t <= n - 3 (history length >= 2, before the validation target).
    """
    if split not in ("test", "fit"):
        raise ValueError(f"split must be 'test' or 'fit', got {split!r}")
    depth = item_sids.shape[1]
    rows = []
    for u, user in enumerate(users):
        ids = [item2id[a] for a in sequences[user]]
        n = len(ids)
        cuts = [n - 1] if split == "test" else range(2, n - 2)
        for t in cuts:
            rows.append((u, ids[max(0, t - slots):t], ids[t]))
    N = len(rows)
    items = np.full((N, slots), -1, dtype=np.int64)
    for i, (_, hist, _) in enumerate(rows):
        items[i, :len(hist)] = hist
    mask = items >= 0
    sids = np.where(mask[..., None], item_sids[np.maximum(items, 0)], -1)
    targets = np.array([t for _, _, t in rows], dtype=np.int64)
    return Histories(user_index=np.array([u for u, _, _ in rows], dtype=np.int64), items=items, sids=sids,
                     mask=mask, lengths=mask.sum(1), target_items=targets, target_sids=item_sids[targets].reshape(N, depth))


def slot_table(h: Histories, item_sids: np.ndarray, item_labels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """One entry per real slot, in row-major (history, slot) order: who it is and every label."""
    hist, slot = np.nonzero(h.mask)
    L = h.lengths[hist]
    item = h.items[hist, slot]
    r1 = h.items[hist, L - 1]
    target = h.target_items[hist]
    out = {"hist": hist, "slot": slot, "L": L, "user_index": h.user_index[hist], "r": L - slot,
           "rec": rec_bin(L - slot), "item": item, "r1_item": r1, "is_r1": slot == L - 1, "target_item": target,
           "r1_d0": item_sids[r1, 0], "tgt_d0": item_sids[target, 0]}
    for d in range(item_sids.shape[1]):
        out[f"own_d{d}"] = item_sids[item, d]
    for name, codes in item_labels.items():
        out[f"own_{name}"] = codes[item]
        out[f"r1_{name}"] = codes[r1]
        out[f"tgt_{name}"] = codes[target]
    return out


def census(idx: np.ndarray, val: np.ndarray, n_latents: int, families: dict[str, tuple[np.ndarray, np.ndarray, int]]):
    """Assign each SAE latent one label family (protocol, Part S "Census").

    idx/val [rows, k] are the active latents per row. families: name ->
    (row mask, labels [rows] with -1 missing, n_labels). Returns per-family
    (c_star, share, base rate, kappa, support) arrays [n_latents], the assigned
    family name per latent ("dead", "unassigned" or a family), and the fire
    count per latent.
    """
    active = val > 0
    fires = np.bincount(idx[active], minlength=n_latents)
    per = {}
    for name, (rows, labels, n_labels) in families.items():
        ok = rows & (labels >= 0)
        base = np.bincount(labels[ok], minlength=n_labels).astype(float)
        base /= max(base.sum(), 1.0)
        a = active & ok[:, None]
        lat = idx[a]
        lab = np.broadcast_to(labels[:, None], idx.shape)[a]
        counts = np.zeros((n_latents, n_labels), dtype=np.int64)
        np.add.at(counts, (lat, lab), 1)
        support = counts.sum(1)
        c_star = counts.argmax(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = counts[np.arange(n_latents), c_star] / support
            p = base[c_star]
            kappa = (share - p) / (1 - p)
        kappa = np.where(support > 0, kappa, np.nan)
        per[name] = {"c_star": c_star, "share": share, "base": p, "kappa": kappa, "support": support}
    names = list(families)
    K = np.stack([np.where(per[n]["support"] >= SUPPORT_MIN, per[n]["kappa"], -np.inf) for n in names], 1)
    K = np.nan_to_num(K, nan=-np.inf)
    best = K.argmax(1)
    assigned = np.where(K[np.arange(n_latents), best] >= KAPPA_MIN, np.array(names, dtype=object)[best], "unassigned")
    assigned = np.where(fires == 0, "dead", assigned)
    return per, assigned, fires


def matched_control(idx_row: np.ndarray, val_row: np.ndarray, chosen: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """Latents of one row to ablate as the control for `chosen` (trap 3).

    For each chosen latent, largest activation first, the unused active latent
    that is neither chosen nor excluded and is nearest in activation. Returns
    fewer only when the row has fewer candidates.
    """
    value = dict(zip(idx_row.tolist(), val_row.tolist()))
    chosen_set, excluded_set = set(chosen.tolist()), set(excluded.tolist())
    if not chosen_set <= {f for f, v in value.items() if v > 0}:
        raise ValueError("a chosen latent is not active in this row")
    pool = [(f, v) for f, v in value.items() if v > 0 and f not in chosen_set and f not in excluded_set]
    want = sorted((value[f] for f in chosen_set), reverse=True)
    out = []
    for v in want:
        if not pool:
            break
        j = min(range(len(pool)), key=lambda i: (abs(pool[i][1] - v), pool[i][0]))
        out.append(pool.pop(j)[0])
    return np.array(out, dtype=np.int64)


def one_hot_design(words: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    """[rows, n_fields] word ids -> [rows, sum(sizes)] float32 one-hot design matrix."""
    rows, n_fields = words.shape
    if n_fields != len(sizes) or bool((words < 0).any()) or any(bool((words[:, i] >= s).any()) for i, s in enumerate(sizes)):
        raise ValueError("description words outside their field vocabularies")
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    X = np.zeros((rows, int(sum(sizes))), dtype=np.float32)
    X[np.arange(rows)[:, None], words + offsets[None, :]] = 1.0
    return X


def with_unknown(codes: np.ndarray, vocab_size: int) -> np.ndarray:
    """Missing (-1) -> its own word, index vocab_size (trap 4)."""
    return np.where(codes < 0, vocab_size, codes)


def render(words: dict, vocab: dict) -> str:
    """One slot's description as a sentence; `words` field -> word id, `vocab` field -> names."""
    def w(field):
        names = vocab[field]
        i = int(words[field])
        return names[i] if i < len(names) else "unknown"
    return (f"recency {w('rec')}: {w('cat_l1')} > {w('cat_l2')} (store: {w('main_cat')}), "
            f"price decile {w('price_decile')}, first code {w('own_d0')}; "
            f"recent item looks like {w('r1_cat_l2')}, first code {w('r1_d0')}")
