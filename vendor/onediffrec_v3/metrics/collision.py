"""Collision-aware evaluation for generative recommendation.

The ItemHit and ItemNDCG implementations follow the collision-aware candidate
expansion (CCE) definition: every predicted semantic ID (SID) represents its
entire collision group, and groups are expanded in beam order.  SIDs are
canonicalized to tuples so they are never compared through ambiguous string
concatenation.

The paper assumes catalog-valid, unique beam entries.  Strict validation is
therefore the default.  ``"trim-padding"`` is the paper-safe compatibility
mode for historical OneDiffRec output: it removes only a contiguous suffix of
blank padding and validates every remaining entry strictly.  The more general
``"preserve"`` and ``"filter"`` policies are diagnostic, non-paper modes.
Preserve retains raw beam positions and assigns malformed/out-of-catalog slots
unit width (the reference implementation's fallback); filter drops them and
keeps only the first duplicate.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


SID = Tuple[int, ...]


class SIDParseError(ValueError):
    """Raised when a value cannot be interpreted as one semantic ID."""


_TOKEN_RE = re.compile(r"<([A-Za-z][A-Za-z0-9]*)_(-?\d+)>")
_INTEGER_RE = re.compile(r"-?\d+")


def parse_sid(value: object) -> SID:
    """Return a canonical integer tuple for common repository SID formats.

    Accepted examples are ``"<a_1><b_2><c_3>"``, a list of those individual
    tokens, and integer/string sequences such as ``[1, 2, 3]``.  Token labels
    are presentation metadata; tuple position identifies the codebook level.
    """

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise SIDParseError("SID is empty")

        # Also accept JSON arrays passed on a command line or stored as text.
        if text.startswith("[") and text.endswith("]"):
            try:
                return parse_sid(json.loads(text))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise SIDParseError(f"invalid JSON SID: {value!r}") from exc

        matches = list(_TOKEN_RE.finditer(text))
        if matches:
            remainder = _TOKEN_RE.sub("", text)
            if remainder.replace(",", "").strip():
                raise SIDParseError(f"malformed token SID: {value!r}")
            return tuple(int(match.group(2)) for match in matches)

        if _INTEGER_RE.fullmatch(text):
            return (int(text),)
        raise SIDParseError(f"unrecognized SID string: {value!r}")

    if not isinstance(value, (list, tuple)) and hasattr(value, "tolist"):
        value = value.tolist()  # numpy/torch scalar arrays, without dependency

    if isinstance(value, (list, tuple)):
        if not value:
            raise SIDParseError("SID sequence is empty")
        codes = []
        for component in value:
            if isinstance(component, bool):
                raise SIDParseError("boolean is not a SID code")
            if isinstance(component, int):
                codes.append(component)
                continue
            # numpy integer scalars expose __index__.
            if not isinstance(component, str) and hasattr(component, "__index__"):
                codes.append(int(component))
                continue
            if isinstance(component, str):
                component = component.strip()
                if _INTEGER_RE.fullmatch(component):
                    codes.append(int(component))
                    continue
                token = _TOKEN_RE.fullmatch(component)
                if token:
                    codes.append(int(token.group(2)))
                    continue
            raise SIDParseError(f"invalid SID component: {component!r}")
        return tuple(codes)

    raise SIDParseError(f"unsupported SID value: {value!r}")


def build_group_sizes(item_to_sid: Mapping[Any, object]) -> Dict[SID, int]:
    """Build ``SID -> number of catalog items`` from a forward item mapping."""

    counts: Counter[SID] = Counter()
    for item_id, raw_sid in item_to_sid.items():
        try:
            counts[parse_sid(raw_sid)] += 1
        except SIDParseError as exc:
            raise SIDParseError(f"invalid SID for catalog item {item_id!r}: {exc}") from exc
    return dict(counts)


@dataclass(frozen=True)
class CollisionStats:
    item_count: int
    unique_sid_count: int
    collision_group_count: int
    collision_item_count: int
    duplicate_excess_count: int
    max_group_size: int
    collision_item_rate: float
    duplicate_excess_rate: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def collision_stats(group_sizes: Mapping[SID, int]) -> CollisionStats:
    """Return both common collision-rate definitions.

    ``collision_item_rate`` is the paper's fraction of items belonging to a
    non-singleton SID group.  ``duplicate_excess_rate`` is ``(N-U)/N``, where
    U is the number of unique SIDs; it is reported separately because the two
    quantities are often (incorrectly) interchanged.
    """

    sizes = []
    for sid, size in group_sizes.items():
        if not isinstance(sid, tuple):
            raise TypeError("group_sizes keys must be tuple SIDs")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"invalid collision-group size for {sid!r}: {size!r}")
        sizes.append(size)

    item_count = sum(sizes)
    unique_count = len(sizes)
    colliding_sizes = [size for size in sizes if size > 1]
    collision_items = sum(colliding_sizes)
    duplicate_excess = item_count - unique_count
    return CollisionStats(
        item_count=item_count,
        unique_sid_count=unique_count,
        collision_group_count=len(colliding_sizes),
        collision_item_count=collision_items,
        duplicate_excess_count=duplicate_excess,
        max_group_size=max(sizes, default=0),
        collision_item_rate=(collision_items / item_count if item_count else 0.0),
        duplicate_excess_rate=(duplicate_excess / item_count if item_count else 0.0),
    )


@dataclass(frozen=True)
class BeamDiagnostics:
    raw_count: int
    kept_count: int
    padding_count: int
    invalid_count: int
    unknown_count: int
    duplicate_count: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RankingScore:
    metrics: Dict[str, float]
    target_status: str
    beam: BeamDiagnostics
    ranked_sids: Tuple[SID | None, ...]

    def to_dict(self, include_ranking: bool = False) -> Dict[str, object]:
        result: Dict[str, object] = {
            "metrics": dict(self.metrics),
            "target_status": self.target_status,
            "beam": self.beam.to_dict(),
        }
        if include_ranking:
            result["ranked_sids"] = [list(sid) if sid is not None else None for sid in self.ranked_sids]
        return result


def _normalise_ks(ks: Iterable[int]) -> Tuple[int, ...]:
    normalized = []
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError(f"K must be a positive integer, got {k!r}")
        normalized.append(k)
    if not normalized:
        raise ValueError("at least one K is required")
    return tuple(sorted(set(normalized)))


def _validate_group_sizes(group_sizes: Mapping[SID, int]) -> None:
    # Reuse the complete validation in one place.
    collision_stats(group_sizes)


def _prepare_beam(
    predictions: Sequence[object],
    group_sizes: Mapping[SID, int],
    *,
    policy: str,
) -> Tuple[Tuple[SID | None, ...], BeamDiagnostics]:
    if policy not in {"strict", "trim-padding", "preserve", "filter"}:
        raise ValueError(f"unknown beam policy: {policy!r}")

    raw_count = len(predictions)
    padding = 0
    if policy == "trim-padding":
        end = raw_count
        while end and (
            predictions[end - 1] is None
            or (
                isinstance(predictions[end - 1], str)
                and not predictions[end - 1].strip()
            )
        ):
            end -= 1
        padding = raw_count - end
        predictions = predictions[:end]
        # Padding is allowed only in the verified suffix above.  Everything in
        # the retained ranking must satisfy the paper's beam assumptions.
        entry_policy = "strict"
    else:
        entry_policy = policy

    prepared = []
    seen = set()
    invalid = unknown = duplicate = 0

    for rank, raw_sid in enumerate(predictions, start=1):
        try:
            sid = parse_sid(raw_sid)
        except SIDParseError as exc:
            invalid += 1
            if entry_policy == "strict":
                raise SIDParseError(f"invalid beam entry at rank {rank}: {exc}") from exc
            if entry_policy == "preserve":
                prepared.append(None)
            continue

        if sid not in group_sizes:
            unknown += 1
            if entry_policy == "strict":
                raise ValueError(f"out-of-catalog SID at rank {rank}: {sid!r}")
            if entry_policy == "preserve":
                prepared.append(sid)
            continue
        if sid in seen:
            duplicate += 1
            if entry_policy == "strict":
                raise ValueError(f"duplicate SID at rank {rank}: {sid!r}")
            if entry_policy == "filter":
                continue
            prepared.append(sid)
            continue
        seen.add(sid)
        prepared.append(sid)

    diagnostics = BeamDiagnostics(
        raw_count=raw_count,
        kept_count=sum(sid in group_sizes for sid in prepared if sid is not None),
        padding_count=padding,
        invalid_count=invalid,
        unknown_count=unknown,
        duplicate_count=duplicate,
    )
    return tuple(prepared), diagnostics


def _metrics_for_entries(
    ranked_sids: Sequence[SID | None],
    target_sid: SID,
    group_sizes: Mapping[SID, int],
    k: int,
) -> Dict[str, float]:
    try:
        target_index = next(i for i, sid in enumerate(ranked_sids) if sid == target_sid)
    except StopIteration:
        target_index = -1

    prefix = f"@{k}"
    if target_index < 0:
        return {
            "Hit" + prefix: 0.0,
            "NDCG" + prefix: 0.0,
            "ItemHit" + prefix: 0.0,
            "ItemNDCG" + prefix: 0.0,
        }

    # Exact SID metrics retain the beam's original positions in preserve mode.
    sid_rank = target_index + 1
    hit = float(sid_rank <= k)
    ndcg = (1.0 / math.log2(sid_rank + 1)) if hit else 0.0

    # Preserve is an explicitly non-paper diagnostic mode.  Its unknown and
    # malformed entries occupy one item slot, matching the reference code's
    # fallback group size and, critically, never promoting a later target.
    # Repeated valid groups deliberately expand again in that mode.
    first_item_rank = 1 + sum(
        group_sizes.get(sid, 1) if sid is not None else 1
        for sid in ranked_sids[:target_index]
    )
    group_size = group_sizes[target_sid]
    included = min(group_size, max(0, k - first_item_rank + 1))
    item_hit = included / group_size
    item_ndcg = sum(
        1.0 / math.log2(first_item_rank + offset + 1)
        for offset in range(included)
    ) / group_size

    return {
        "Hit" + prefix: hit,
        "NDCG" + prefix: ndcg,
        "ItemHit" + prefix: item_hit,
        "ItemNDCG" + prefix: item_ndcg,
    }


def metrics_at_k(
    ranked_sids: Sequence[SID],
    target_sid: SID,
    group_sizes: Mapping[SID, int],
    k: int,
) -> Dict[str, float]:
    """Compute exact-SID and CCE item metrics for one already-clean ranking."""

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"K must be a positive integer, got {k!r}")
    if target_sid not in group_sizes:
        raise KeyError(f"target SID is not in the catalog: {target_sid!r}")

    for sid in ranked_sids:
        if sid not in group_sizes:
            raise KeyError(f"ranked SID is not in the catalog: {sid!r}")
    return _metrics_for_entries(ranked_sids, target_sid, group_sizes, k)


def score_ranking(
    predictions: Sequence[object],
    target: object,
    group_sizes: Mapping[SID, int],
    ks: Sequence[int],
    *,
    beam_policy: str = "strict",
    validate: bool = True,
) -> RankingScore:
    """Score one ranked beam against a target SID.

    Unknown or malformed targets receive zero for every metric and an explicit
    ``target_status`` (``"unknown"`` or ``"invalid"``).  This keeps the metric
    denominator stable and prevents malformed examples from inflating results.
    """

    if validate:
        _validate_group_sizes(group_sizes)
    normalized_ks = _normalise_ks(ks)
    ranked_sids, diagnostics = _prepare_beam(
        predictions, group_sizes, policy=beam_policy
    )

    try:
        target_sid = parse_sid(target)
    except SIDParseError as exc:
        if beam_policy in {"strict", "trim-padding"}:
            raise SIDParseError(f"invalid target SID: {exc}") from exc
        target_sid = ()
        target_status = "invalid"
    else:
        target_status = "ok" if target_sid in group_sizes else "unknown"
        if beam_policy in {"strict", "trim-padding"} and target_status != "ok":
            raise ValueError(f"target SID is not in the catalog: {target_sid!r}")

    result: Dict[str, float] = {}
    if target_status == "ok":
        for k in normalized_ks:
            result.update(_metrics_for_entries(ranked_sids, target_sid, group_sizes, k))
    else:
        for k in normalized_ks:
            result.update(
                {
                    f"Hit@{k}": 0.0,
                    f"NDCG@{k}": 0.0,
                    f"ItemHit@{k}": 0.0,
                    f"ItemNDCG@{k}": 0.0,
                }
            )

    return RankingScore(
        metrics=result,
        target_status=target_status,
        beam=diagnostics,
        ranked_sids=ranked_sids,
    )


class CollisionEvaluator:
    """Prevalidated, reusable scorer for a single catalog and set of cutoffs."""

    def __init__(
        self,
        group_sizes: Mapping[SID, int],
        ks: Sequence[int],
        *,
        beam_policy: str = "strict",
    ) -> None:
        _validate_group_sizes(group_sizes)
        if beam_policy not in {"strict", "trim-padding", "preserve", "filter"}:
            raise ValueError(f"unknown beam policy: {beam_policy!r}")
        self.group_sizes = group_sizes
        self.ks = _normalise_ks(ks)
        self.beam_policy = beam_policy

    def score(self, predictions: Sequence[object], target: object) -> RankingScore:
        return score_ranking(
            predictions,
            target,
            self.group_sizes,
            self.ks,
            beam_policy=self.beam_policy,
            validate=False,
        )
