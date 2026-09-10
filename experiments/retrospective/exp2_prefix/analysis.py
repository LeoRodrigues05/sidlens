"""Strict loaders and statistics for the retained AR next-item predictions.

The unit of prediction is a Semantic ID (SID), not an item.  This module keeps
that distinction explicit: prefix metrics always use exact SID digits, while a
separate compatibility metric reproduces the historical evaluator's
exact-SID-or-title matching rule.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TOP_K = (1, 3, 5, 10, 20, 50)
EXPECTED_TEST_COLUMNS = (
    "user_id",
    "history_item_title",
    "item_title",
    "history_item_id",
    "item_id",
    "history_item_sid",
    "item_sid",
)
VARIANT_RE = re.compile(
    r"^nextitem__(?P<category>.+)__(?P<quantizer>MQ|rqkmeans|rqvae)__"
    r"(?P<depth>[1-9][0-9]*)cb__(?P<width>[1-9][0-9]*)$"
)
SID_TOKEN_RE = re.compile(r"<([a-z])_([0-9]+)>")
EVAL_HEADER_RE = re.compile(
    r"eval: tree=next-item (?P<category>\S+) (?P<quantizer>\S+) "
    r"(?P<depth>[0-9]+)cb/(?P<width>[0-9]+) beams=(?P<beams>[0-9]+)"
)


class ValidationError(ValueError):
    """A retained artifact violates the experiment's data contract."""


@dataclass(frozen=True)
class Variant:
    variant_id: str
    category: str
    quantizer: str
    depth: int
    width: int

    @classmethod
    def parse(cls, variant_id: str) -> "Variant":
        match = VARIANT_RE.fullmatch(variant_id)
        if match is None:
            raise ValidationError(f"invalid next-item variant id: {variant_id!r}")
        return cls(
            variant_id=variant_id,
            category=match.group("category"),
            quantizer=match.group("quantizer"),
            depth=int(match.group("depth")),
            width=int(match.group("width")),
        )

    @property
    def data_stem(self) -> str:
        return (
            f"{self.category}_{self.quantizer}_{self.depth}codebook_{self.width}"
        )


@dataclass(frozen=True)
class TestRow:
    row_index: int
    user_id: str
    history_titles: tuple[str, ...]
    target_title: str
    history_item_ids: tuple[int, ...]
    target_item_id: int
    history_sids: tuple[str, ...]
    target_sid: str
    target_codes: tuple[int, ...]

    @property
    def identity_key(self) -> tuple[Any, ...]:
        """Configuration-invariant key used to prove cross-file row order."""

        return (
            self.user_id,
            self.history_titles,
            self.target_title,
            self.history_item_ids,
            self.target_item_id,
        )


@dataclass(frozen=True)
class InfoCatalog:
    valid_sids: frozenset[str]
    sid_by_item_id: Mapping[int, str]
    # These intentionally use last-write-wins, matching vendor/onediffrec/calc.py.
    sid_to_title_last: Mapping[str, str]
    sid_to_item_id_last: Mapping[str, str]
    n_rows: int
    n_unique_sids: int


@dataclass(frozen=True)
class ExampleResult:
    variant: Variant
    row_index: int
    user_id: str
    target_item_id: int
    target_sid: str
    target_codes: tuple[int, ...]
    top1_sid: str
    top1_codes: tuple[int, ...]
    lcp_length: int
    first_error_digit: int | None
    exact_rank: int | None  # one-based
    legacy_rank: int | None  # one-based
    legacy_match_reason: str
    effective_beam_count: int
    prefix_max_lcp_by_k: tuple[int, ...]


@dataclass(frozen=True)
class RunData:
    variant: Variant
    examples: tuple[ExampleResult, ...]
    test_rows: tuple[TestRow, ...]
    validation: Mapping[str, Any]


@dataclass(frozen=True, order=True)
class MetricKey:
    metric: str
    digit: int | None = None
    k: int | None = None


METRIC_DEFINITIONS = {
    "top1_lcp_length": "Mean number of leading target digits matched by rank 1.",
    "top1_lcp_fraction": "Mean rank-1 LCP divided by that configuration's SID depth.",
    "top1_full_sid_correct": "Rank-1 exact full-SID equality.",
    "top1_first_error_rate": "Fraction whose first rank-1 mismatch is this 1-based digit.",
    "top1_conditional_digit_accuracy": (
        "Rank-1 digit correctness conditional on all preceding digits being correct."
    ),
    "top1_first_error_hazard": (
        "Rank-1 mismatch at this digit conditional on all preceding digits being correct."
    ),
    "prefix_hit": "Any of the first K beam slots matches the target's first d digits.",
    "full_sid_hr_exact": "Any of the first K beam slots exactly equals the target SID.",
    "full_sid_hr_legacy": (
        "Historical evaluator HR: exact SID, else last-write-wins title equality; "
        "item-ID fallback is only attempted when title maps are unavailable."
    ),
}


CAVEATS = (
    "This is a retrospective observational analysis of frozen predictions; it does "
    "not identify a causal role for any digit.",
    "The same test interactions recur across configurations and there is one retained "
    "evaluation per configuration, not independent model seeds. User-cluster bootstrap "
    "intervals condition on the fixed retained configurations and are not uncertainty "
    "intervals over configurations or training seeds.",
    "Prefix and exact-HR metrics are SID-level. An exact SID can name multiple catalogue "
    "items when codes collide, so these metrics are not collision-aware item identity.",
    "Historical reported HR used exact SID equality followed by title equality. The "
    "title and item-ID maps overwrite earlier items that share a SID; for valid known "
    "SIDs the evaluator's item-ID branch is unreachable after a title mismatch. Exact "
    "SID HR and legacy-compatible HR are therefore both reported.",
    "Some generation records contain fewer than 50 non-empty candidates followed by "
    "empty beam-padding strings. Padding remains in its original rank slots and never "
    "counts as a hit.",
    "Conditional digit accuracy/hazard selects examples that survived the preceding "
    "digits. It describes the autoregressive error process, not independent per-digit "
    "difficulty or semantic contribution. This is especially important for MQ, whose "
    "SID digits are parallel partitions rather than a residual hierarchy.",
)


def _fail(context: str, message: str) -> ValidationError:
    return ValidationError(f"{context}: {message}")


def parse_sid(value: str, *, depth: int, width: int, context: str = "SID") -> tuple[int, ...]:
    """Parse one canonical ``<a_1><b_2>`` SID without normalization."""

    if not isinstance(value, str):
        raise _fail(context, f"expected string, got {type(value).__name__}")
    matches = list(SID_TOKEN_RE.finditer(value))
    if not matches or "".join(m.group(0) for m in matches) != value:
        raise _fail(context, f"not a canonical SID: {value!r}")
    if len(matches) != depth:
        raise _fail(context, f"expected {depth} digits, found {len(matches)} in {value!r}")
    codes: list[int] = []
    for digit, match in enumerate(matches):
        expected_tag = chr(ord("a") + digit)
        if match.group(1) != expected_tag:
            raise _fail(
                context,
                f"digit {digit + 1} must use tag {expected_tag!r}, got {match.group(1)!r}",
            )
        code = int(match.group(2))
        if not 0 <= code < width:
            raise _fail(
                context,
                f"digit {digit + 1} code {code} is outside [0, {width})",
            )
        codes.append(code)
    return tuple(codes)


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) != len(b):
        raise ValidationError(f"cannot compare SID depths {len(a)} and {len(b)}")
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return len(a)


def _literal_list(value: str, *, context: str, item_type: type) -> tuple[Any, ...]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise _fail(context, f"invalid Python-list literal: {exc}") from exc
    if not isinstance(parsed, list):
        raise _fail(context, f"expected list, got {type(parsed).__name__}")
    if not parsed:
        raise _fail(context, "history must not be empty")
    for index, item in enumerate(parsed):
        if item_type is int:
            valid = isinstance(item, int) and not isinstance(item, bool)
        else:
            valid = isinstance(item, item_type)
        if not valid:
            raise _fail(
                context,
                f"element {index} must be {item_type.__name__}, got {type(item).__name__}",
            )
    return tuple(parsed)


def load_info_catalog(path: Path, variant: Variant) -> InfoCatalog:
    valid_sids: set[str] = set()
    sid_by_item_id: dict[int, str] = {}
    sid_to_title_last: dict[str, str] = {}
    sid_to_item_id_last: dict[str, str] = {}
    n_rows = 0

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise _fail(str(path), f"cannot open info file: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            if raw.endswith("\r"):
                raw = raw[:-1]
            context = f"{path}:{line_number}"
            if not raw:
                raise _fail(context, "blank info row")
            try:
                left, item_id_text = raw.rsplit("\t", 1)
                sid, title = left.split("\t", 1)
            except ValueError as exc:
                raise _fail(context, "expected SID<TAB>title<TAB>item_id") from exc
            sid = sid.strip()
            title = title.strip()
            item_id_text = item_id_text.strip()
            parse_sid(sid, depth=variant.depth, width=variant.width, context=context)
            if not title:
                raise _fail(context, "empty title")
            try:
                item_id = int(item_id_text)
            except ValueError as exc:
                raise _fail(context, f"invalid item id {item_id_text!r}") from exc
            if item_id < 0 or item_id in sid_by_item_id:
                raise _fail(context, f"duplicate or negative item id {item_id}")
            sid_by_item_id[item_id] = sid
            valid_sids.add(sid)
            # This overwrite is required to reproduce the historical evaluator.
            sid_to_title_last[sid] = title
            sid_to_item_id_last[sid] = str(item_id)
            n_rows += 1

    if not n_rows:
        raise _fail(str(path), "empty info file")
    return InfoCatalog(
        valid_sids=frozenset(valid_sids),
        sid_by_item_id=sid_by_item_id,
        sid_to_title_last=sid_to_title_last,
        sid_to_item_id_last=sid_to_item_id_last,
        n_rows=n_rows,
        n_unique_sids=len(valid_sids),
    )


def load_test_rows(path: Path, variant: Variant, catalog: InfoCatalog) -> tuple[TestRow, ...]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise _fail(str(path), f"cannot open test CSV: {exc}") from exc

    rows: list[TestRow] = []
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_TEST_COLUMNS:
            raise _fail(
                str(path),
                f"expected columns {EXPECTED_TEST_COLUMNS}, got {tuple(reader.fieldnames or ())}",
            )
        for row_index, raw in enumerate(reader):
            line = row_index + 2
            context = f"{path}:{line}"
            if None in raw:
                raise _fail(context, "row contains excess CSV fields")
            user_id = raw["user_id"].strip()
            if not user_id:
                raise _fail(context, "empty user_id")
            history_titles = _literal_list(
                raw["history_item_title"],
                context=f"{context}:history_item_title",
                item_type=str,
            )
            history_item_ids = _literal_list(
                raw["history_item_id"],
                context=f"{context}:history_item_id",
                item_type=int,
            )
            history_sids = _literal_list(
                raw["history_item_sid"],
                context=f"{context}:history_item_sid",
                item_type=str,
            )
            if not (len(history_titles) == len(history_item_ids) == len(history_sids)):
                raise _fail(context, "history title/item-id/SID lengths differ")
            try:
                target_item_id = int(raw["item_id"])
            except ValueError as exc:
                raise _fail(context, f"invalid target item id {raw['item_id']!r}") from exc
            target_sid = raw["item_sid"]
            target_codes = parse_sid(
                target_sid,
                depth=variant.depth,
                width=variant.width,
                context=f"{context}:item_sid",
            )
            if target_item_id not in catalog.sid_by_item_id:
                raise _fail(context, f"target item id {target_item_id} is absent from info")
            if catalog.sid_by_item_id[target_item_id] != target_sid:
                raise _fail(context, "target item id and SID disagree with info mapping")
            for pos, (item_id, sid) in enumerate(zip(history_item_ids, history_sids)):
                parse_sid(
                    sid,
                    depth=variant.depth,
                    width=variant.width,
                    context=f"{context}:history_item_sid[{pos}]",
                )
                if catalog.sid_by_item_id.get(item_id) != sid:
                    raise _fail(
                        context,
                        f"history item id/SID disagree with info at position {pos}",
                    )
            rows.append(
                TestRow(
                    row_index=row_index,
                    user_id=user_id,
                    history_titles=history_titles,
                    target_title=raw["item_title"],
                    history_item_ids=history_item_ids,
                    target_item_id=target_item_id,
                    history_sids=history_sids,
                    target_sid=target_sid,
                    target_codes=target_codes,
                )
            )
    if not rows:
        raise _fail(str(path), "empty test CSV")
    return tuple(rows)


def expected_input(row: TestRow) -> str:
    history = ", ".join(row.history_sids)
    return (
        "Can you predict the next possible item the user may expect, given the "
        f"following chronological interaction history: {history}"
    )


def legacy_match_reason(prediction: str, target: str, catalog: InfoCatalog) -> str | None:
    """Mirror the branch order and last-write-wins maps in upstream ``calc.py``."""

    if prediction == target:
        return "exact_sid"
    if prediction in catalog.sid_to_title_last and target in catalog.sid_to_title_last:
        if catalog.sid_to_title_last[prediction] == catalog.sid_to_title_last[target]:
            return "title"
        # The original ``elif`` chain does not fall through to item id here.
        return None
    if prediction in catalog.sid_to_item_id_last and target in catalog.sid_to_item_id_last:
        if catalog.sid_to_item_id_last[prediction] == catalog.sid_to_item_id_last[target]:
            return "item_id"
    return None


def _load_json(path: Path, *, expected_type: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise _fail(str(path), f"cannot read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _fail(str(path), f"invalid JSON: {exc}") from exc
    if not isinstance(value, expected_type):
        raise _fail(
            str(path),
            f"expected JSON {expected_type.__name__}, got {type(value).__name__}",
        )
    return value


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise _fail(context, "expected finite number, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(context, f"expected finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise _fail(context, f"expected finite number, got {value!r}")
    return result


def _parse_eval_beams(path: Path, variant: Variant) -> int:
    try:
        first_line = path.open("r", encoding="utf-8", errors="replace").readline().rstrip("\r\n")
    except OSError as exc:
        raise _fail(str(path), f"cannot read evaluation log: {exc}") from exc
    match = EVAL_HEADER_RE.fullmatch(first_line)
    if match is None:
        raise _fail(str(path), f"invalid evaluation header {first_line!r}")
    observed = (
        match.group("category"),
        match.group("quantizer"),
        int(match.group("depth")),
        int(match.group("width")),
    )
    expected = (variant.category, variant.quantizer, variant.depth, variant.width)
    if observed != expected:
        raise _fail(str(path), f"evaluation header config {observed} != {expected}")
    beams = int(match.group("beams"))
    if beams < max(TOP_K):
        raise _fail(str(path), f"beam count {beams} is smaller than requested K={max(TOP_K)}")
    return beams


def load_prediction_rows(
    path: Path,
    *,
    variant: Variant,
    test_rows: Sequence[TestRow],
    catalog: InfoCatalog,
    expected_beams: int,
) -> tuple[ExampleResult, ...]:
    payload = _load_json(path, expected_type=list)
    if len(payload) != len(test_rows):
        raise _fail(
            str(path),
            f"prediction count {len(payload)} != test row count {len(test_rows)}",
        )

    results: list[ExampleResult] = []
    required_keys = {"input", "output", "predict"}
    for row_index, (record, test_row) in enumerate(zip(payload, test_rows)):
        context = f"{path}:record[{row_index}]"
        if not isinstance(record, dict) or set(record) != required_keys:
            keys = sorted(record) if isinstance(record, dict) else type(record).__name__
            raise _fail(context, f"expected exactly keys {sorted(required_keys)}, got {keys}")
        if record["input"] != expected_input(test_row):
            raise _fail(
                context,
                "input does not exactly match the frozen test row; order or history drifted",
            )
        if record["output"] != test_row.target_sid + "\n":
            raise _fail(
                context,
                "output does not exactly match the frozen target SID plus newline",
            )
        predictions = record["predict"]
        if not isinstance(predictions, list) or len(predictions) != expected_beams:
            length = len(predictions) if isinstance(predictions, list) else type(predictions).__name__
            raise _fail(context, f"expected {expected_beams} beam slots, got {length}")

        candidate_codes: list[tuple[int, ...] | None] = []
        nonempty_sids: list[str] = []
        saw_padding = False
        for rank, prediction in enumerate(predictions, start=1):
            pcontext = f"{context}:predict[{rank - 1}]"
            if not isinstance(prediction, str):
                raise _fail(pcontext, f"expected string, got {type(prediction).__name__}")
            if prediction == "":
                saw_padding = True
                candidate_codes.append(None)
                continue
            if saw_padding:
                raise _fail(pcontext, "non-empty candidate appears after empty beam padding")
            codes = parse_sid(
                prediction,
                depth=variant.depth,
                width=variant.width,
                context=pcontext,
            )
            if prediction not in catalog.valid_sids:
                raise _fail(pcontext, "candidate SID is absent from the frozen info catalogue")
            candidate_codes.append(codes)
            nonempty_sids.append(prediction)
        if not nonempty_sids:
            raise _fail(context, "all beam slots are empty")
        if len(nonempty_sids) != len(set(nonempty_sids)):
            raise _fail(context, "non-empty beam candidates are not unique")

        lcps = [
            0 if codes is None else longest_common_prefix(codes, test_row.target_codes)
            for codes in candidate_codes
        ]
        exact_rank: int | None = None
        legacy_rank: int | None = None
        legacy_reason = "none"
        for rank, prediction in enumerate(predictions, start=1):
            if prediction == "":
                break
            if exact_rank is None and prediction == test_row.target_sid:
                exact_rank = rank
            if legacy_rank is None:
                reason = legacy_match_reason(prediction, test_row.target_sid, catalog)
                if reason is not None:
                    legacy_rank = rank
                    legacy_reason = reason
        lcp = lcps[0]
        first_error = None if lcp == variant.depth else lcp + 1
        prefix_max = tuple(max(lcps[:k]) for k in TOP_K)
        results.append(
            ExampleResult(
                variant=variant,
                row_index=row_index,
                user_id=test_row.user_id,
                target_item_id=test_row.target_item_id,
                target_sid=test_row.target_sid,
                target_codes=test_row.target_codes,
                top1_sid=nonempty_sids[0],
                top1_codes=candidate_codes[0],  # type: ignore[arg-type]
                lcp_length=lcp,
                first_error_digit=first_error,
                exact_rank=exact_rank,
                legacy_rank=legacy_rank,
                legacy_match_reason=legacy_reason,
                effective_beam_count=len(nonempty_sids),
                prefix_max_lcp_by_k=prefix_max,
            )
        )
    return tuple(results)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_metric_identity(metric: Mapping[str, Any], variant: Variant, path: Path) -> None:
    expected = {
        "variant_id": variant.variant_id,
        "tree": "next-item",
        "category": variant.category,
        "method": variant.quantizer,
        "codebooks": variant.depth,
        "codebook_size": variant.width,
        "status": "ok",
    }
    for key, value in expected.items():
        if metric.get(key) != value:
            raise _fail(str(path), f"{key}={metric.get(key)!r}, expected {value!r}")


def _reproduction_rows(
    examples: Sequence[ExampleResult],
    metric: Mapping[str, Any],
    *,
    tolerance_points: float,
    path: Path,
) -> list[dict[str, Any]]:
    recorded = metric.get("metrics")
    if not isinstance(recorded, dict):
        raise _fail(str(path), "missing metrics object")
    rows: list[dict[str, Any]] = []
    n = len(examples)
    for k in TOP_K:
        exact_hits = sum(e.exact_rank is not None and e.exact_rank <= k for e in examples)
        legacy_hits = sum(e.legacy_rank is not None and e.legacy_rank <= k for e in examples)
        exact_hr = exact_hits / n
        legacy_hr = legacy_hits / n
        legacy_ndcg = sum(
            0.0
            if e.legacy_rank is None or e.legacy_rank > k
            else 1.0 / math.log2(e.legacy_rank + 1)
            for e in examples
        ) / n
        recorded_hr = _finite_number(recorded.get(f"HR@{k}"), context=f"{path}:HR@{k}")
        recorded_ndcg = _finite_number(
            recorded.get(f"NDCG@{k}"), context=f"{path}:NDCG@{k}"
        )
        hr_delta = legacy_hr * 100.0 - recorded_hr
        ndcg_delta = legacy_ndcg * 100.0 - recorded_ndcg
        if abs(hr_delta) > tolerance_points:
            raise _fail(
                str(path),
                f"legacy HR@{k}={legacy_hr * 100:.8f} does not reproduce "
                f"recorded {recorded_hr:.8f} (delta {hr_delta:+.8f} points)",
            )
        if abs(ndcg_delta) > tolerance_points:
            raise _fail(
                str(path),
                f"legacy NDCG@{k}={legacy_ndcg * 100:.8f} does not reproduce "
                f"recorded {recorded_ndcg:.8f} (delta {ndcg_delta:+.8f} points)",
            )
        full_prefix_hits = sum(
            e.prefix_max_lcp_by_k[TOP_K.index(k)] >= e.variant.depth for e in examples
        )
        if full_prefix_hits != exact_hits:
            raise _fail(str(path), f"full-prefix hit@{k} disagrees with exact SID HR")
        rows.append(
            {
                "k": k,
                "n_examples": n,
                "exact_hits": exact_hits,
                "legacy_hits": legacy_hits,
                "exact_hr_fraction": exact_hr,
                "legacy_hr_fraction": legacy_hr,
                "recorded_hr_percent": recorded_hr,
                "legacy_hr_delta_points": hr_delta,
                "legacy_ndcg_fraction": legacy_ndcg,
                "recorded_ndcg_percent": recorded_ndcg,
                "legacy_ndcg_delta_points": ndcg_delta,
            }
        )
    return rows


def load_run(
    metric_path: Path,
    *,
    test_dir: Path,
    info_dir: Path,
    tolerance_points: float = 5.1e-5,
) -> RunData:
    variant = Variant.parse(metric_path.stem)
    metrics_dir = metric_path.parent
    prediction_path = metrics_dir / f"{variant.variant_id}.predictions.json"
    assets_path = metrics_dir / f"{variant.variant_id}.assets.json"
    eval_log_path = metrics_dir / f"{variant.variant_id}.eval.log"
    for path in (prediction_path, assets_path, eval_log_path):
        if not path.is_file():
            raise _fail(variant.variant_id, f"missing sidecar {path.name}")

    metric = _load_json(metric_path, expected_type=dict)
    _validate_metric_identity(metric, variant, metric_path)
    assets = _load_json(assets_path, expected_type=dict)
    if assets.get("variant_id", variant.variant_id) != variant.variant_id:
        raise _fail(str(assets_path), "variant_id disagrees with filename")
    paths = assets.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("test"), str) or not isinstance(paths.get("info"), str):
        raise _fail(str(assets_path), "paths.test and paths.info must be strings")
    expected_test_name = f"{variant.data_stem}.csv"
    expected_info_name = f"{variant.data_stem}.txt"
    if Path(paths["test"]).name != expected_test_name:
        raise _fail(str(assets_path), f"test basename is not {expected_test_name}")
    if Path(paths["info"]).name != expected_info_name:
        raise _fail(str(assets_path), f"info basename is not {expected_info_name}")
    test_path = test_dir / expected_test_name
    info_path = info_dir / expected_info_name
    if not test_path.is_file() or not info_path.is_file():
        raise _fail(variant.variant_id, "rebased frozen test or info file is missing")

    catalog = load_info_catalog(info_path, variant)
    expected_items = int(_finite_number(metric.get("items"), context=f"{metric_path}:items"))
    if catalog.n_rows != expected_items:
        raise _fail(str(info_path), f"{catalog.n_rows} rows != recorded items {expected_items}")
    recorded_unique = int(
        _finite_number(metric.get("unique_full_sids"), context=f"{metric_path}:unique_full_sids")
    )
    if catalog.n_unique_sids != recorded_unique:
        raise _fail(
            str(info_path),
            f"{catalog.n_unique_sids} unique SIDs != recorded {recorded_unique}",
        )
    test_rows = load_test_rows(test_path, variant, catalog)
    beams = _parse_eval_beams(eval_log_path, variant)
    examples = load_prediction_rows(
        prediction_path,
        variant=variant,
        test_rows=test_rows,
        catalog=catalog,
        expected_beams=beams,
    )
    reproduction = _reproduction_rows(
        examples,
        metric,
        tolerance_points=tolerance_points,
        path=metric_path,
    )
    effective = [e.effective_beam_count for e in examples]
    title_fallback = sum(e.legacy_match_reason == "title" for e in examples)
    item_id_fallback = sum(e.legacy_match_reason == "item_id" for e in examples)
    validation: dict[str, Any] = {
        "variant_id": variant.variant_id,
        "category": variant.category,
        "quantizer": variant.quantizer,
        "depth": variant.depth,
        "width": variant.width,
        "n_examples": len(examples),
        "n_users": len({e.user_id for e in examples}),
        "beam_slots": beams,
        "min_effective_beam_count": min(effective),
        "max_effective_beam_count": max(effective),
        "rows_with_empty_beam_padding": sum(n < beams for n in effective),
        "empty_beam_slots": sum(beams - n for n in effective),
        "earliest_title_fallback_hits": title_fallback,
        "earliest_item_id_fallback_hits": item_id_fallback,
        "info_rows": catalog.n_rows,
        "unique_full_sids": catalog.n_unique_sids,
        "hashes": {
            "metric_json_sha256": sha256_file(metric_path),
            "assets_json_sha256": sha256_file(assets_path),
            "predictions_json_sha256": sha256_file(prediction_path),
            "eval_log_sha256": sha256_file(eval_log_path),
            "test_csv_sha256": sha256_file(test_path),
            "info_txt_sha256": sha256_file(info_path),
        },
        "metric_reproduction": reproduction,
    }
    return RunData(
        variant=variant,
        examples=examples,
        test_rows=test_rows,
        validation=validation,
    )


def discover_metric_paths(metrics_dir: Path) -> list[Path]:
    if not metrics_dir.is_dir():
        raise _fail(str(metrics_dir), "metrics directory does not exist")
    prediction_stems = {
        path.name.removesuffix(".predictions.json")
        for path in metrics_dir.glob("*.predictions.json")
    }
    completed_stems: set[str] = set()
    for path in sorted(metrics_dir.glob("*.json")):
        if path.name.endswith((".predictions.json", ".assets.json")):
            continue
        payload = _load_json(path, expected_type=dict)
        if payload.get("tree") == "next-item" and payload.get("status") == "ok":
            completed_stems.add(path.stem)
    if prediction_stems != completed_stems:
        missing_predictions = sorted(completed_stems - prediction_stems)
        orphan_predictions = sorted(prediction_stems - completed_stems)
        raise _fail(
            str(metrics_dir),
            f"completed/prediction sidecars differ; missing={missing_predictions}, "
            f"orphan={orphan_predictions}",
        )
    if not prediction_stems:
        raise _fail(str(metrics_dir), "no retained prediction JSONs")
    paths = [metrics_dir / f"{stem}.json" for stem in sorted(prediction_stems)]
    variants = [Variant.parse(path.stem) for path in paths]
    if len({v.variant_id for v in variants}) != len(variants):
        raise _fail(str(metrics_dir), "duplicate variant ids")
    return paths


def load_all_runs(
    metrics_dir: Path,
    *,
    test_dir: Path,
    info_dir: Path,
    tolerance_points: float = 5.1e-5,
) -> list[RunData]:
    runs = [
        load_run(
            path,
            test_dir=test_dir,
            info_dir=info_dir,
            tolerance_points=tolerance_points,
        )
        for path in discover_metric_paths(metrics_dir)
    ]
    reference = runs[0]
    reference_keys = tuple(row.identity_key for row in reference.test_rows)
    for run in runs[1:]:
        keys = tuple(row.identity_key for row in run.test_rows)
        if keys != reference_keys:
            mismatch = next(
                (i for i, (left, right) in enumerate(zip(reference_keys, keys)) if left != right),
                min(len(reference_keys), len(keys)),
            )
            raise ValidationError(
                f"{run.variant.variant_id}: cross-configuration test identity/order "
                f"differs from {reference.variant.variant_id} at row {mismatch}"
            )
    return runs


def metric_keys(max_depth: int) -> tuple[MetricKey, ...]:
    keys = [
        MetricKey("top1_lcp_length"),
        MetricKey("top1_lcp_fraction"),
        MetricKey("top1_full_sid_correct"),
    ]
    for digit in range(1, max_depth + 1):
        keys.extend(
            (
                MetricKey("top1_first_error_rate", digit=digit),
                MetricKey("top1_conditional_digit_accuracy", digit=digit),
                MetricKey("top1_first_error_hazard", digit=digit),
            )
        )
    for k in TOP_K:
        keys.extend(
            (
                MetricKey("full_sid_hr_exact", k=k),
                MetricKey("full_sid_hr_legacy", k=k),
            )
        )
        for digit in range(1, max_depth + 1):
            keys.append(MetricKey("prefix_hit", digit=digit, k=k))
    return tuple(keys)


def metric_contribution(example: ExampleResult, key: MetricKey) -> tuple[float, float]:
    """Return numerator and denominator for one example and one estimand."""

    depth = example.variant.depth
    if key.metric == "top1_lcp_length":
        return float(example.lcp_length), 1.0
    if key.metric == "top1_lcp_fraction":
        return example.lcp_length / depth, 1.0
    if key.metric == "top1_full_sid_correct":
        return float(example.lcp_length == depth), 1.0
    if key.metric == "top1_first_error_rate":
        if key.digit is None or key.digit > depth:
            return 0.0, 0.0
        return float(example.first_error_digit == key.digit), 1.0
    if key.metric in {"top1_conditional_digit_accuracy", "top1_first_error_hazard"}:
        if key.digit is None or key.digit > depth or example.lcp_length < key.digit - 1:
            return 0.0, 0.0
        correct = example.lcp_length >= key.digit
        if key.metric == "top1_conditional_digit_accuracy":
            return float(correct), 1.0
        return float(not correct), 1.0
    if key.metric == "prefix_hit":
        if key.digit is None or key.k is None or key.digit > depth:
            return 0.0, 0.0
        max_lcp = example.prefix_max_lcp_by_k[TOP_K.index(key.k)]
        return float(max_lcp >= key.digit), 1.0
    if key.metric == "full_sid_hr_exact":
        if key.k is None:
            raise ValidationError("full_sid_hr_exact requires k")
        return float(example.exact_rank is not None and example.exact_rank <= key.k), 1.0
    if key.metric == "full_sid_hr_legacy":
        if key.k is None:
            raise ValidationError("full_sid_hr_legacy requires k")
        return float(example.legacy_rank is not None and example.legacy_rank <= key.k), 1.0
    raise ValidationError(f"unknown metric {key.metric!r}")


def make_bootstrap_weights(
    user_ids: Sequence[str], *, replicates: int, seed: int
) -> tuple[tuple[str, ...], np.ndarray | None]:
    users = tuple(sorted(set(user_ids)))
    if not users:
        raise ValidationError("cannot bootstrap an empty user set")
    if replicates < 0:
        raise ValidationError("bootstrap replicates must be non-negative")
    if replicates == 0:
        return users, None
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(users), 1.0 / len(users), dtype=np.float64)
    weights = rng.multinomial(len(users), probabilities, size=replicates)
    return users, weights


def summarize_group(
    examples: Sequence[ExampleResult],
    *,
    group_fields: Mapping[str, Any],
    all_users: Sequence[str],
    bootstrap_weights: np.ndarray | None,
) -> list[dict[str, Any]]:
    if not examples:
        raise ValidationError(f"empty summary group {group_fields}")
    user_index = {user: index for index, user in enumerate(all_users)}
    max_depth = max(e.variant.depth for e in examples)
    keys = metric_keys(max_depth)
    numerators = np.zeros((len(all_users), len(keys)), dtype=np.float64)
    denominators = np.zeros_like(numerators)
    contributing_examples = np.zeros(len(keys), dtype=np.int64)
    contributing_variants: list[set[str]] = [set() for _ in keys]
    for example in examples:
        ui = user_index[example.user_id]
        for ki, key in enumerate(keys):
            numerator, denominator = metric_contribution(example, key)
            numerators[ui, ki] += numerator
            denominators[ui, ki] += denominator
            if denominator:
                contributing_examples[ki] += 1
                contributing_variants[ki].add(example.variant.variant_id)

    total_num = numerators.sum(axis=0)
    total_den = denominators.sum(axis=0)
    if bootstrap_weights is None:
        bootstrap_values = None
    else:
        bootstrap_num = bootstrap_weights @ numerators
        bootstrap_den = bootstrap_weights @ denominators
        bootstrap_values = np.divide(
            bootstrap_num,
            bootstrap_den,
            out=np.full_like(bootstrap_num, np.nan),
            where=bootstrap_den > 0,
        )

    rows: list[dict[str, Any]] = []
    variants = {e.variant.variant_id for e in examples}
    for index, key in enumerate(keys):
        if total_den[index] <= 0:
            continue
        estimate = total_num[index] / total_den[index]
        if bootstrap_values is None:
            ci_low = ci_high = None
            valid_replicates = 0
        else:
            values = bootstrap_values[:, index]
            values = values[np.isfinite(values)]
            valid_replicates = len(values)
            if not len(values):
                ci_low = ci_high = None
            else:
                ci_low, ci_high = (float(v) for v in np.quantile(values, (0.025, 0.975)))
        row = {
            **group_fields,
            "metric": key.metric,
            "digit": key.digit,
            "k": key.k,
            "numerator": float(total_num[index]),
            "denominator": float(total_den[index]),
            "estimate": float(estimate),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_examples_total": len(examples),
            "n_examples_contributing": int(contributing_examples[index]),
            "n_users_total": len({e.user_id for e in examples}),
            "n_users_contributing": int(np.count_nonzero(denominators[:, index])),
            "n_configs_total": len(variants),
            "n_configs_contributing": len(contributing_variants[index]),
            "bootstrap_valid_replicates": valid_replicates,
        }
        rows.append(row)
    return rows


def summarize_all(
    runs: Sequence[RunData], *, bootstrap_replicates: int, bootstrap_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not runs:
        raise ValidationError("no runs to summarize")
    examples = [example for run in runs for example in run.examples]
    users, weights = make_bootstrap_weights(
        [e.user_id for e in examples],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    config_rows: list[dict[str, Any]] = []
    for run in runs:
        v = run.variant
        config_rows.extend(
            summarize_group(
                run.examples,
                group_fields={
                    "stratum_type": "config",
                    "stratum_value": v.variant_id,
                    "variant_id": v.variant_id,
                    "quantizer": v.quantizer,
                    "depth": v.depth,
                    "width": v.width,
                },
                all_users=users,
                bootstrap_weights=weights,
            )
        )

    strata: list[tuple[str, str, list[ExampleResult]]] = [("overall", "all", examples)]
    for field in ("quantizer", "depth", "width"):
        values = sorted({getattr(e.variant, field) for e in examples}, key=str)
        for value in values:
            subset = [e for e in examples if getattr(e.variant, field) == value]
            strata.append((field, str(value), subset))

    strata_rows: list[dict[str, Any]] = []
    for field, value, subset in strata:
        strata_rows.extend(
            summarize_group(
                subset,
                group_fields={
                    "stratum_type": field,
                    "stratum_value": value,
                    "variant_id": None,
                    "quantizer": value if field == "quantizer" else None,
                    "depth": int(value) if field == "depth" else None,
                    "width": int(value) if field == "width" else None,
                },
                all_users=users,
                bootstrap_weights=weights,
            )
        )
    return config_rows, strata_rows


def example_csv_rows(runs: Sequence[RunData]) -> Iterable[dict[str, Any]]:
    for run in runs:
        for example in run.examples:
            row: dict[str, Any] = {
                "variant_id": example.variant.variant_id,
                "category": example.variant.category,
                "quantizer": example.variant.quantizer,
                "depth": example.variant.depth,
                "width": example.variant.width,
                "row_index": example.row_index,
                "user_id": example.user_id,
                "target_item_id": example.target_item_id,
                "target_sid": example.target_sid,
                "target_codes": "-".join(map(str, example.target_codes)),
                "top1_sid": example.top1_sid,
                "top1_codes": "-".join(map(str, example.top1_codes)),
                "lcp_length": example.lcp_length,
                "lcp_fraction": example.lcp_length / example.variant.depth,
                "first_error_digit": example.first_error_digit,
                "exact_rank": example.exact_rank,
                "legacy_rank": example.legacy_rank,
                "legacy_match_reason": example.legacy_match_reason,
                "effective_beam_count": example.effective_beam_count,
            }
            for index, k in enumerate(TOP_K):
                row[f"prefix_max_lcp_at_{k}"] = example.prefix_max_lcp_by_k[index]
            yield row


def validation_csv_rows(runs: Sequence[RunData]) -> Iterable[dict[str, Any]]:
    for run in runs:
        base = {
            "variant_id": run.variant.variant_id,
            "category": run.variant.category,
            "quantizer": run.variant.quantizer,
            "depth": run.variant.depth,
            "width": run.variant.width,
        }
        for row in run.validation["metric_reproduction"]:
            yield {**base, **row}
