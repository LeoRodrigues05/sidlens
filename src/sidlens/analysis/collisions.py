"""Collision-aware utilities for retained Semantic-ID predictions.

The autoregressive sweep ranked *SIDs*, not items.  A full SID can name more
than one catalogue item, so an exact SID hit is an oracle upper bound on item
retrieval rather than proof that the target item was identified.  The helpers
here keep that ambiguity explicit and never use the upstream lossy
``tokens2item`` map.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"<([a-z])_(\d+)>")


def parse_sid(value: str, depth: int | None = None) -> tuple[int, ...]:
    """Parse ``<a_1><b_2>...`` into codebook IDs, rejecting extra text.

    Whitespace and surrounding quotes are tolerated because the retained
    evaluator stripped those characters.  Digit letters must be consecutive;
    accepting ``<a_1><c_2>`` would silently turn a malformed prediction into a
    valid-looking two-vector.
    """

    if not isinstance(value, str):
        raise ValueError(f"SID must be text, got {type(value).__name__}")
    compact = re.sub(r"\s+", "", value).strip('"\'')
    matches = TOKEN_RE.findall(compact)
    rebuilt = "".join(f"<{letter}_{code}>" for letter, code in matches)
    if not matches or rebuilt != compact:
        raise ValueError(f"malformed SID: {value!r}")
    letters = [letter for letter, _ in matches]
    expected = [chr(ord("a") + i) for i in range(len(matches))]
    if letters != expected:
        raise ValueError(
            f"non-consecutive SID digits in {value!r}: {letters} != {expected}")
    if depth is not None and len(matches) != depth:
        raise ValueError(
            f"SID has {len(matches)} digits, expected {depth}: {value!r}")
    return tuple(int(code) for _, code in matches)


def exact_rank(predictions: Sequence[tuple[int, ...]],
               target: tuple[int, ...]) -> int | None:
    """One-indexed rank of the first exact SID match, or ``None``."""

    try:
        return predictions.index(target) + 1
    except ValueError:
        return None


@dataclass(frozen=True)
class ItemRankBounds:
    """What a SID ranking implies about one target item's rank.

    ``best`` and ``worst`` put the target first or last inside its collision
    bucket. ``catalogue`` uses the stable item-ID order, which is reproducible
    but is not model evidence.  A missing target SID makes every field ``None``.
    """

    best: int | None
    worst: int | None
    catalogue: int | None
    multiplicity: int

    def hit_bounds(self, k: int) -> dict[str, float]:
        """Lower/expected/upper hit probability at an *item* cutoff.

        Expected assumes a uniformly random tie-break inside the target SID's
        bucket.  It is an explicit no-information baseline, not an estimate of
        an unobserved model preference.
        """

        if self.best is None or self.worst is None or self.catalogue is None:
            return {"lower": 0.0, "uniform": 0.0,
                    "catalogue": 0.0, "upper": 0.0}
        available = max(0, min(self.multiplicity, k - self.best + 1))
        return {
            "lower": float(self.worst <= k),
            "uniform": available / self.multiplicity,
            "catalogue": float(self.catalogue <= k),
            "upper": float(self.best <= k),
        }


def item_rank_bounds(
    predictions: Iterable[tuple[int, ...]],
    target_item: int,
    items_by_sid: Mapping[tuple[int, ...], Sequence[int]],
    *,
    target_sid: tuple[int, ...] | None = None,
) -> ItemRankBounds:
    """Expand unique predicted SIDs and locate ``target_item``.

    Each legal SID is expanded to every item in its one-to-many bucket.  A SID
    repeated by a padded beam is counted once.  Illegal SIDs are ignored, as
    they do not identify any catalogue item.
    """

    if target_sid is None:
        target_bucket: list[int] | None = None
        for sid, raw_items in items_by_sid.items():
            if target_item in raw_items:
                target_sid = sid
                target_bucket = sorted(int(i) for i in raw_items)
                break
        if target_sid is None or target_bucket is None:
            raise KeyError(f"target item {target_item} has no SID bucket")
    else:
        raw_target_bucket = items_by_sid.get(target_sid)
        if raw_target_bucket is None or target_item not in raw_target_bucket:
            raise KeyError(
                f"target item {target_item} is absent from supplied SID bucket "
                f"{target_sid}")
        target_bucket = sorted(int(i) for i in raw_target_bucket)

    seen: set[tuple[int, ...]] = set()
    preceding = 0
    for sid in predictions:
        if sid in seen:
            continue
        seen.add(sid)
        raw_bucket = items_by_sid.get(sid)
        if not raw_bucket:
            continue
        bucket = sorted(int(i) for i in raw_bucket)
        if sid == target_sid:
            offset = bucket.index(target_item)
            return ItemRankBounds(
                best=preceding + 1,
                worst=preceding + len(bucket),
                catalogue=preceding + offset + 1,
                multiplicity=len(bucket),
            )
        preceding += len(bucket)

    return ItemRankBounds(None, None, None, len(target_bucket))


def exact_sign_test(n_positive: int, n_negative: int) -> float:
    """Two-sided exact sign-test p-value, dropping zero differences."""

    n = n_positive + n_negative
    if n == 0:
        return 1.0
    tail = min(n_positive, n_negative)
    p = 2.0 * sum(math.comb(n, i) for i in range(tail + 1)) / (2 ** n)
    return min(1.0, p)
