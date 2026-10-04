"""The day of every interaction the models were trained and tested on.

Neither model is given a timestamp. The AR prompt lists the history "in
chronological order", and DiffGRM gets item slots plus a recency position. Both
see only ORDER. Time survives only in the frozen `review.json`, whose keys are
"(user_idx, item_idx, unixReviewTime)". This module puts a day back on every
position of every user's `inter.json` sequence. Each AR window, each DiffGRM
row and each intervention record can then be stratified by how long ago each
history item happened.

How upstream built the sequences (vendor/onediffrec/data/amazon18_data_process.py)
-----------------------------------------------------------------------------------
* Each user's reviews are stable-sorted by `unixReviewTime`. Amazon 2018 stores
  a date, so every time is 00:00 UTC and ties are whole days.
* Duplicates are kept. The raw dump repeats some (user, item, day) reviews, and
  each copy becomes an event (43,102 events against 40,449 distinct keys).
* `review.json` holds one key per distinct review, in raw-file order.
* Next-item rows are sliding windows (at most 10 history items) ending at every
  position >= 1. They are globally stable-sorted by target time and cut 8:1:1
  into train / valid / test. That is why the three splits are consecutive time
  periods.

Traps, each with its guard
--------------------------
1. **Duplicates break a one-to-one join.** Events outnumber keys, so a lookup
   by (user, item) is not enough. `_align` searches exhaustively for every time
   assignment that is non-decreasing, uses every key, and has its first
   occurrences within each day in review-file order, which is exactly what a
   stable sort produces. A user with no such assignment raises.
2. **Ambiguous duplicate days.** A copy of an item reviewed on two different
   days may belong to either day (32 users, 72 positions). Every position >= 1
   is the target of exactly one next-item row, and the rows are sorted by
   target time. The row's place in train -> valid -> test therefore bounds its
   day by its neighbours, and `_resolve` keeps the one assignment that fits.
   Anything still ambiguous raises. Nothing is guessed.
3. **Same-day order is not time order.** Within a day the stable sort keeps
   raw-file order, and the raw dump is about 97% ASIN-sorted. The "most recent"
   item of a same-day burst is therefore mostly the one with the next-lower
   ASIN, not a later purchase. The row tables carry `tie12` and
   `same_day_as_target` so no recency analysis can forget this.
4. **Duplicate records look like repeat purchases.** A target that is the same
   item on the same day as a history event is one review recorded twice, not a
   re-purchase. `dup_target` and `dup_of_target` mark them.
5. **Two cohorts.** AR rows are windows. The DiffGRM cohort is each user's last
   event, read from `all_item_seqs.json`. `diffusion_row_times` proves that file
   equals `inter.json` under the id maps before attaching any time. Rows are
   joined through (user, position), never by row number.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sidlens import paths

CATEGORY = "Industrial_and_Scientific"
DAY = 86400
SPLIT_ORDER = ("train", "valid", "test")
MAX_AR_HISTORY = 10          # upstream's sliding window (st = max(i - 10, 0))

# The one binning of "how long before the target", used by every temporal analysis.
# Edges are inclusive upper bounds in days; the first bin is the target's own day.
GAP_BINS = ((0, "same day"), (30, "1-30 d"), (365, "31-365 d"), (10**9, ">365 d"))


def gap_bin(days) -> np.ndarray:
    """Label each gap (in whole days, >= 0) with its GAP_BINS name."""
    d = np.asarray(days)
    if np.any(d < 0):
        raise ValueError("negative gap: a history item after its target")
    out = np.empty(d.shape, dtype=object)
    lo = -1
    for hi, name in GAP_BINS:
        out[(d > lo) & (d <= hi)] = name
        lo = hi
    return out


@dataclass(frozen=True)
class EventTimes:
    """Unix seconds (a UTC midnight) for every position of every user's sequence.

    `times[u][p]` is the day of `sequences[u][p]`, with u the canonical user
    index as a string (the `inter.json` key).
    """
    sequences: dict[str, tuple[int, ...]]
    times: dict[str, np.ndarray]
    report: dict

    def gap_days(self, user: str, later: int, earlier: int) -> int:
        return int(self.times[user][later] - self.times[user][earlier]) // DAY


def _align(seq: list[int], keys: list[tuple[int, int]], cap: int = 64) -> list[tuple[int, ...]]:
    """Every assignment of key times to `seq` consistent with a stable time sort.

    keys: the user's distinct (item, time) reviews in review-file order.
    """
    srt = sorted(keys, key=lambda x: x[1])          # stable: file order within a day
    by_item: dict[int, list[int]] = collections.defaultdict(list)
    for i, t in srt:
        by_item[i].append(t)
    sols: list[tuple[int, ...]] = []
    out: list[int] = []
    used: set[tuple[int, int]] = set()

    def dfs(p: int, cur: int, nxt: int) -> None:
        if len(sols) >= cap:
            return
        if p == len(seq):
            if nxt == len(srt):
                sols.append(tuple(out))
            return
        i = seq[p]
        for t in by_item.get(i, ()):
            if t < cur:
                continue
            key = (i, t)
            if key in used:                        # a duplicate of an earlier event
                out.append(t)
                dfs(p + 1, t, nxt)
                out.pop()
            elif nxt < len(srt) and srt[nxt] == key:   # the next first occurrence
                used.add(key)
                out.append(t)
                dfs(p + 1, t, nxt + 1)
                out.pop()
                used.discard(key)

    dfs(0, -1, 0)
    if len(sols) >= cap:
        raise RuntimeError(f"more than {cap} alignments; refusing to enumerate further")
    return sols


def _read_inter(split: str) -> list[tuple[str, tuple[int, ...], int]]:
    path = paths.FROZEN_DATA / "interactions" / f"{CATEGORY}.{split}.inter"
    lines = path.read_text().splitlines()
    if lines[0] != "user_id:token\titem_id_list:token_seq\titem_id:token":
        raise ValueError(f"{path.name}: unexpected header {lines[0]!r}")
    rows = []
    for ln in lines[1:]:
        u, h, t = ln.split("\t")
        rows.append((u, tuple(int(x) for x in h.split()), int(t)))
    return rows


def window_position(seq: tuple[int, ...], history: tuple[int, ...], target: int) -> int:
    """The one position i with seq[i] == target and seq[max(i-10,0):i] == history."""
    hits = [i for i in range(1, len(seq))
            if seq[i] == target and seq[max(i - MAX_AR_HISTORY, 0):i] == history]
    if len(hits) != 1:
        raise ValueError(f"window matches {len(hits)} positions, expected exactly 1")
    return hits[0]


def _resolve(sols: dict[str, list[tuple[int, ...]]], seqs: dict[str, tuple[int, ...]]):
    """Pick each ambiguous user's assignment from the global target-time order."""
    order = []                                           # (user, pos) in train->valid->test order
    for split in SPLIT_ORDER:
        for u, h, t in _read_inter(split):
            order.append((u, window_position(seqs[u], h, t)))
    if len(set(order)) != len(order) or len(order) != sum(len(s) - 1 for s in seqs.values()):
        raise ValueError("next-item rows are not one per position >= 1")
    fixed = np.array([sols[u][0][p] if len({s[p] for s in sols[u]}) == 1 else -1
                      for u, p in order], dtype=np.int64)
    lo = np.maximum.accumulate(np.where(fixed >= 0, fixed, -1))
    hi = np.minimum.accumulate(np.where(fixed >= 0, fixed, np.iinfo(np.int64).max)[::-1])[::-1]
    where = {k: j for j, k in enumerate(order)}
    chosen, n_pos = {}, 0
    for u, cand in sols.items():
        if len(cand) == 1:
            chosen[u] = cand[0]
            continue
        ok = [s for s in cand if all(
            lo[where[(u, p)]] <= s[p] <= hi[where[(u, p)]] for p in range(1, len(s)))]
        if len({tuple(s) for s in ok}) != 1:
            raise ValueError(f"user {u}: {len(ok)} assignments fit the global order; expected 1")
        chosen[u] = ok[0]
        n_pos += sum(len({s[p] for s in cand}) > 1 for p in range(len(cand[0])))
    t_order = np.array([chosen[u][p] for u, p in order])
    if np.any(np.diff(t_order) < 0):
        raise ValueError("resolved times are not sorted along train -> valid -> test")
    return chosen, n_pos, order


@lru_cache(maxsize=1)
def load_event_times() -> EventTimes:
    """Align, resolve and verify the day of every event. Raises on any inconsistency."""
    d = paths.FROZEN_DATA
    rev_path = d / "reviews" / f"{CATEGORY}.review.json"
    inter_path = d / "interactions" / f"{CATEGORY}.inter.json"
    rev = json.loads(rev_path.read_text())
    inter = json.loads(inter_path.read_text())
    keys: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for k in rev:
        u, i, t = ast.literal_eval(k)
        if t % DAY:
            raise ValueError(f"review time {t} is not a UTC midnight")
        keys[str(u)].append((int(i), int(t)))
    if set(keys) != set(inter):
        raise ValueError("review users differ from inter.json users")
    seqs = {u: tuple(int(x) for x in s) for u, s in inter.items()}
    sols = {}
    for u, s in seqs.items():
        sols[u] = _align(list(s), keys[u])
        if not sols[u]:
            raise ValueError(f"user {u}: no stable-sort alignment of review times")
    chosen, n_amb_pos, order = _resolve(sols, seqs)
    times = {u: np.asarray(chosen[u], dtype=np.int64) for u in seqs}
    n_events = sum(len(s) for s in seqs.values())
    report = {
        "review_json_sha256": hashlib.sha256(rev_path.read_bytes()).hexdigest(),
        "inter_json_sha256": hashlib.sha256(inter_path.read_bytes()).hexdigest(),
        "n_users": len(seqs), "n_events": n_events, "n_review_keys": len(rev),
        "n_duplicate_events": n_events - len(rev),
        "n_users_ambiguous_before_resolution": sum(len(s) > 1 for s in sols.values()),
        "n_positions_ambiguous_before_resolution": int(n_amb_pos),
        "n_next_item_rows": len(order),
        "first_day": int(min(t.min() for t in times.values())),
        "last_day": int(max(t.max() for t in times.values())),
    }
    return EventTimes(seqs, times, report)


# ------------------------------------------------------------ row tables --
def ar_row_times(examples, ev: EventTimes | None = None):
    """One row per next-item AR example, plus one row per history item.

    Returns (rows, hist) DataFrames:
      rows  example_id, row, user_id, user, pos, hist_len, t_target, gap1_days
            (target minus most recent item), gap12_days (most recent minus second),
            tie12, same_day_as_target (most recent item on the target's day),
            n_same_day (history items on the target's day), span_days,
            dup_target (the target repeats a history event: same item, same day),
            repeat_item (target item id anywhere in the history)
      hist  example_id, k (0 = oldest), recency (1 = most recent), item_id,
            t, gap_days (target minus item), same_day_as_target, dup_of_target
    """
    import pandas as pd

    ev = ev or load_event_times()
    rows, hist = [], []
    for ex in examples:
        if ex.task != "next-item":
            raise ValueError("ar_row_times covers next-item windows only")
        if not ex.user_id.startswith("A") or not ex.user_id[1:].isdecimal():
            raise ValueError(f"{ex.example_id}: user id {ex.user_id!r} is not 'A<index>'")
        u = ex.user_id[1:]
        seq, t = ev.sequences[u], ev.times[u]
        i = window_position(seq, tuple(ex.history_item_ids), ex.target_item_ids[0])
        L = len(ex.history_item_ids)
        st = i - L
        tt = int(t[i])
        gaps = (tt - t[st:i]) // DAY
        dup = [(seq[p] == seq[i]) and (t[p] == tt) for p in range(st, i)]
        rows.append({
            "example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id, "user": u,
            "pos": i, "hist_len": L, "t_target": tt, "gap1_days": int(gaps[-1]),
            "gap12_days": int((t[i - 1] - t[i - 2]) // DAY) if L >= 2 else -1,
            "tie12": bool(L >= 2 and t[i - 1] == t[i - 2]),
            "same_day_as_target": bool(gaps[-1] == 0), "n_same_day": int((gaps == 0).sum()),
            "span_days": int(gaps[0]), "dup_target": bool(any(dup)),
            "repeat_item": ex.target_item_ids[0] in ex.history_item_ids})
        for k in range(L):
            hist.append({"example_id": ex.example_id, "k": k, "recency": L - k,
                         "item_id": ex.history_item_ids[k], "t": int(t[st + k]),
                         "gap_days": int(gaps[k]), "same_day_as_target": bool(gaps[k] == 0),
                         "dup_of_target": bool(dup[k])})
    return pd.DataFrame(rows), pd.DataFrame(hist)


def diffusion_row_times(cohort, ev: EventTimes | None = None):
    """The same two tables for the DiffGRM leave-last-out cohort (target = last event).

    `cohort` is a `diffusion_eval.DiffusionEvalCohort`. Its sequences come from
    `all_item_seqs.json` (ASINs keyed by raw user id). Each is checked against
    `inter.json` under the frozen id maps before any time is attached.
    """
    import pandas as pd

    ev = ev or load_event_times()
    d = paths.FROZEN_DATA
    seqs = json.loads((d / "sequences" / "all_item_seqs.json").read_text())
    user2id = {a: int(b) for a, b in (ln.split("\t") for ln in
               (d / "id_maps" / f"{CATEGORY}.user2id").read_text().splitlines() if ln)}
    item2id = {a: int(b) for a, b in (ln.split("\t") for ln in
               (d / "id_maps" / f"{CATEGORY}.item2id").read_text().splitlines() if ln)}
    rows, hist = [], []
    H = cohort.histories.shape[1]
    for r, raw in enumerate(cohort.users):
        u = str(user2id[raw])
        mapped = tuple(item2id[a] for a in seqs[raw])
        if mapped != ev.sequences[u]:
            raise ValueError(f"user {raw}: all_item_seqs.json differs from inter.json")
        if int(cohort.target_item_ids[r]) != mapped[-1] or int(cohort.user_ids[r]) != int(u):
            raise ValueError(f"cohort row {r} does not end at the user's last event")
        t = ev.times[u]
        i = len(mapped) - 1
        L = int(cohort.history_lengths[r])
        if L != min(i, H):
            raise ValueError(f"cohort row {r}: history length {L} != min({i}, {H})")
        st = i - L
        tt = int(t[i])
        gaps = (tt - t[st:i]) // DAY
        dup = [(mapped[p] == mapped[i]) and (t[p] == tt) for p in range(st, i)]
        rows.append({
            "user": raw, "user_row": r, "user_idx": u, "pos": i, "hist_len": L, "t_target": tt,
            "gap1_days": int(gaps[-1]),
            "gap12_days": int((t[i - 1] - t[i - 2]) // DAY) if L >= 2 else -1,
            "tie12": bool(L >= 2 and t[i - 1] == t[i - 2]),
            "same_day_as_target": bool(gaps[-1] == 0), "n_same_day": int((gaps == 0).sum()),
            "span_days": int(gaps[0]), "dup_target": bool(any(dup)),
            "repeat_item": mapped[i] in mapped[st:i]})
        for k in range(L):
            hist.append({"user": raw, "k": k, "recency": L - k, "item_id": mapped[st + k],
                         "t": int(t[st + k]), "gap_days": int(gaps[k]),
                         "same_day_as_target": bool(gaps[k] == 0), "dup_of_target": bool(dup[k])})
    return pd.DataFrame(rows), pd.DataFrame(hist)
