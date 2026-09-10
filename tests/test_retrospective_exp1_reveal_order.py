"""Focused tests for the retrospective Exp1 analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/retrospective/exp1_reveal_order/analysis.py"
)
SPEC = importlib.util.spec_from_file_location("exp1_reveal_order_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def test_exact_sign_test_known_cases() -> None:
    result = A.exact_two_sided_sign_test([1.0] * 14 + [-1.0] * 4)
    assert result == {
        "test": "exact_two_sided_sign_test",
        "n_nonzero": 18,
        "n_positive": 14,
        "n_negative": 4,
        "n_ties": 0,
        "p_value": pytest.approx(0.0308837890625),
    }
    tied = A.exact_two_sided_sign_test([0.0, 1e-16, -1e-16])
    assert tied["n_nonzero"] == 0
    assert tied["n_ties"] == 3
    assert tied["p_value"] == 1.0


def test_holm_adjust_preserves_input_order_and_monotonicity() -> None:
    assert A.holm_adjust([0.04, 0.01, 0.03]) == pytest.approx(
        [0.06, 0.03, 0.06])


def _metric_payload(offset: float = 0.0) -> dict[str, float]:
    payload = {"weighted_score": 0.2 + offset}
    for index, metric in enumerate(A.METRICS):
        payload[metric] = 0.2 + index * 0.001 + offset
        payload[f"{metric}_random"] = 0.19 + index * 0.001 + offset
    return payload


def _entry(quantizer: str, depth: int, width: int) -> dict:
    ckpt_id = f"diff-next1-{quantizer}-{depth}cb-{width}"
    return {
        "ckpt_id": ckpt_id,
        "task": "next1",
        "quantizer": quantizer,
        "n_codebook": depth,
        "codebook_size": width,
        "config": {"n_digit": depth, "codebook_size": width},
        "status": "trained",
        "recorded_metrics": _metric_payload(),
        "training": {
            **A.EXPECTED_TRAINING,
            "guided_steps": min(depth, 4),
        },
    }


def test_registry_validation_enforces_18_primary_and_20_total_cells(
    tmp_path: Path,
) -> None:
    payload = {}
    for quantizer, depth, width in A.EXPECTED_PRESENT_GRID:
        entry = _entry(quantizer, depth, width)
        payload[entry["ckpt_id"]] = entry
    for depth, status in ((3, "trained"), (4, "trained"), (5, "incomplete")):
        ckpt_id = f"diff-next2-rqvae-{depth}cb-256"
        payload[ckpt_id] = {
            "ckpt_id": ckpt_id,
            "task": "next2",
            "status": status,
        }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))

    _, entries, validation = A.load_and_validate_registry(path)

    assert len(entries) == 20
    assert len(validation["primary_grid"]) == 18
    assert len(validation["missing_grid"]) == 7
    assert validation["n_next2_trained"] == 2
    assert validation["n_next2_incomplete"] == 1


def test_registry_validation_rejects_missing_paired_metric(tmp_path: Path) -> None:
    payload = {}
    for quantizer, depth, width in A.EXPECTED_PRESENT_GRID:
        entry = _entry(quantizer, depth, width)
        payload[entry["ckpt_id"]] = entry
    for depth, status in ((3, "trained"), (4, "trained"), (5, "incomplete")):
        ckpt_id = f"diff-next2-rqvae-{depth}cb-256"
        payload[ckpt_id] = {"ckpt_id": ckpt_id, "task": "next2", "status": status}
    first = next(value for value in payload.values() if value["task"] == "next1")
    del first["recorded_metrics"]["ndcg@10_random"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(A.AuditError, match="metric schema mismatch"):
        A.load_and_validate_registry(path)


def test_artifact_validation_recomputes_hash_and_detects_drift(
    tmp_path: Path,
) -> None:
    # Use the same tiny artifact for all fields; this unit test targets hash
    # behavior, while the real audit validates the canonical path sections.
    recorded_metrics = _metric_payload()
    metric_text = ", ".join(
        f"('{name}', np.float64({value!r}))"
        for name, value in recorded_metrics.items())
    content = (
        "Number of users: 6298\n"
        "Number of items: 3106\n"
        "Number of interactions: 43102\n"
        f"Test Results: OrderedDict([{metric_text}])\n"
    )
    roots = {}
    for section in ("ckpt", "sids", "results"):
        path = tmp_path / section / "artifact.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        roots[section] = path
    digest = A.sha256_file(roots["ckpt"])
    entry = {
        "ckpt_id": "diff-next1-MQ-3cb-128",
        "ckpt_path": str(roots["ckpt"]),
        "ckpt_sha256": digest,
        "sem_ids_path": str(roots["sids"]),
        "sem_ids_sha256": digest,
        "log_path": str(roots["ckpt"]),
        "log_sha256": digest,
        "transcript_path": str(roots["results"]),
        "transcript_sha256": digest,
        "recorded_metrics": recorded_metrics,
    }

    records, summary = A.validate_artifacts([entry], tmp_path)
    assert len(records) == 4
    assert summary["n_hashes_matching"] == 4

    entry["transcript_sha256"] = "0" * 64
    with pytest.raises(A.AuditError, match="hash mismatch"):
        A.validate_artifacts([entry], tmp_path)


def test_artifact_resolution_never_bypasses_the_supplied_frozen_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside" / "checkpoint.bin"
    outside.parent.mkdir()
    outside.write_bytes(b"weights")
    with pytest.raises(A.AuditError, match="cannot safely rebase"):
        A._resolve_artifact_path(
            str(outside), expected_section="ckpt", frozen_root=tmp_path / "frozen")


def test_transcript_metric_parser_rejects_registry_drift() -> None:
    metrics = _metric_payload()
    line = ", ".join(
        f"('{name}', np.float64({value!r}))" for name, value in metrics.items())
    parsed = A._parse_final_transcript_metrics(
        f"INFO: Test Results: OrderedDict([{line}])\n", "fixture")
    assert parsed == metrics
    del metrics["ndcg@10_random"]
    with pytest.raises(A.AuditError, match="metric schema mismatch"):
        A._parse_final_transcript_metrics(
            "INFO: Test Results: OrderedDict([('weighted_score', 0.2)])\n",
            "fixture",
        )


def test_strata_use_only_balanced_widths_for_primary() -> None:
    entries = [
        {
            **_entry(quantizer, depth, width),
            "source_jobid": "1",
            "source_stamp": "stamp",
            "trained_at": "2026-01-01T00:00:00",
        }
        for quantizer, depth, width in sorted(
            A.EXPECTED_PRESENT_GRID,
            key=lambda cell: (A.QUANTIZERS.index(cell[0]), cell[1], cell[2]),
        )
    ]
    rows = A.build_cell_rows(entries)
    strata = A.build_strata_rows(rows)

    overall = {
        (row["subset"], row["metric"]): row
        for row in strata
        if row["group_by"] == "overall"
    }
    assert overall[("balanced18", "ndcg@10")]["n_cells"] == 18
    assert overall[("all20_sensitivity", "ndcg@10")]["n_cells"] == 20
    primary_ids = {row["ckpt_id"] for row in rows if row["primary_balanced18"]}
    assert all("-256" not in ckpt_id for ckpt_id in primary_ids)


def test_canonical_registry_balanced18_regression() -> None:
    repo = Path(__file__).resolve().parents[1]
    _, entries, validation = A.load_and_validate_registry(
        repo / "manifests/registry.diffusion.json")
    rows = A.build_cell_rows(entries)
    primary = [row for row in rows if row["primary_balanced18"]]

    ndcg = A.summarize_metric(primary, "ndcg@10")
    recall = A.summarize_metric(primary, "recall@10")
    tests = A.hypothesis_rows(primary, "balanced18")

    assert validation["registry_sha256"] == (
        "db7741d4e50c2cb3faedafd8b66ef2740e48499cf100ad36f39e9ebeea7072ee")
    assert ndcg["mean_delta"] == pytest.approx(0.0035197434319571117)
    assert ndcg["median_delta"] == pytest.approx(0.0032149392242410563)
    assert (ndcg["n_positive"], ndcg["n_negative"]) == (14, 4)
    assert recall["mean_delta"] == pytest.approx(0.003440791911492246)
    assert (recall["n_positive"], recall["n_negative"]) == (13, 5)
    assert tests[0]["exact_two_sided_sign_p"] == pytest.approx(
        0.0308837890625)
    assert tests[0]["holm_p_across_ndcg10_recall10"] == pytest.approx(
        0.061767578125)
