"""Focused contracts for retrospective AR prefix-error decomposition."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.retrospective.exp2_prefix import analysis as A
from experiments.retrospective.exp2_prefix import run as runner


VARIANT_ID = "nextitem__Industrial_and_Scientific__MQ__3cb__8"


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    metrics = root / "metrics"
    tests = root / "test"
    info = root / "info"
    metrics.mkdir()
    tests.mkdir()
    info.mkdir()

    data_stem = "Industrial_and_Scientific_MQ_3codebook_8"
    info_path = info / f"{data_stem}.txt"
    info_path.write_text(
        "<a_0><b_0><c_0>\tAlpha\t0\n"
        "<a_0><b_0><c_1>\tBeta\t1\n"
        "<a_0><b_1><c_0>\tGamma\t2\n"
        "<a_1><b_0><c_0>\tAlpha\t3\n",
        encoding="utf-8",
    )

    test_path = tests / f"{data_stem}.csv"
    rows = [
        {
            "user_id": "u1",
            "history_item_title": repr(["Beta"]),
            "item_title": "Alpha",
            "history_item_id": repr([1]),
            "item_id": "0",
            "history_item_sid": repr(["<a_0><b_0><c_1>"]),
            "item_sid": "<a_0><b_0><c_0>",
        },
        {
            "user_id": "u2",
            "history_item_title": repr(["Alpha"]),
            "item_title": "Beta",
            "history_item_id": repr([0]),
            "item_id": "1",
            "history_item_sid": repr(["<a_0><b_0><c_0>"]),
            "item_sid": "<a_0><b_0><c_1>",
        },
    ]
    with test_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(A.EXPECTED_TEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    prompt = (
        "Can you predict the next possible item the user may expect, given the "
        "following chronological interaction history: "
    )
    predictions = [
        {
            "input": prompt + "<a_0><b_0><c_1>",
            "output": "<a_0><b_0><c_0>\n",
            # Rank 1 is a title-only legacy hit; exact SID is rank 2.
            "predict": [
                "<a_1><b_0><c_0>",
                "<a_0><b_0><c_0>",
                "<a_0><b_1><c_0>",
            ]
            + [""] * 47,
        },
        {
            "input": prompt + "<a_0><b_0><c_0>",
            "output": "<a_0><b_0><c_1>\n",
            "predict": [
                "<a_0><b_1><c_0>",
                "<a_1><b_0><c_0>",
                "<a_0><b_0><c_1>",
            ]
            + [""] * 47,
        },
    ]
    (metrics / f"{VARIANT_ID}.predictions.json").write_text(
        json.dumps(predictions), encoding="utf-8"
    )
    metric_values = {
        "HR@1": 50.0,
        "NDCG@1": 50.0,
    }
    for k in A.TOP_K[1:]:
        metric_values[f"HR@{k}"] = 100.0
        metric_values[f"NDCG@{k}"] = 75.0
    (metrics / f"{VARIANT_ID}.json").write_text(
        json.dumps(
            {
                "variant_id": VARIANT_ID,
                "tree": "next-item",
                "category": "Industrial_and_Scientific",
                "method": "MQ",
                "codebooks": 3,
                "codebook_size": 8,
                "status": "ok",
                "items": 4,
                "unique_full_sids": 4,
                "metrics": metric_values,
            }
        ),
        encoding="utf-8",
    )
    (metrics / f"{VARIANT_ID}.assets.json").write_text(
        json.dumps(
            {
                "paths": {
                    "test": f"/old/cluster/test/{data_stem}.csv",
                    "info": f"/old/cluster/info/{data_stem}.txt",
                }
            }
        ),
        encoding="utf-8",
    )
    (metrics / f"{VARIANT_ID}.eval.log").write_text(
        "eval: tree=next-item Industrial_and_Scientific MQ 3cb/8 beams=50\n",
        encoding="utf-8",
    )
    return metrics, tests, info


def test_sid_parser_is_fail_closed():
    assert A.parse_sid("<a_1><b_2><c_3>", depth=3, width=8) == (1, 2, 3)
    for bad in (
        " <a_1><b_2><c_3>",
        "<a_1><c_2><b_3>",
        "<a_1><b_2>",
        "<a_1><b_2><c_8>",
        "<a_1><b_x><c_3>",
    ):
        with pytest.raises(A.ValidationError):
            A.parse_sid(bad, depth=3, width=8)


def test_load_run_aligns_rows_and_reproduces_legacy_metrics(tmp_path):
    metrics, tests, info = _write_fixture(tmp_path)
    run = A.load_run(metrics / f"{VARIANT_ID}.json", test_dir=tests, info_dir=info)

    assert len(run.examples) == 2
    first, second = run.examples
    assert (first.lcp_length, first.first_error_digit) == (0, 1)
    assert (first.exact_rank, first.legacy_rank, first.legacy_match_reason) == (
        2,
        1,
        "title",
    )
    assert (second.lcp_length, second.first_error_digit) == (1, 2)
    assert second.exact_rank == second.legacy_rank == 3
    assert run.validation["rows_with_empty_beam_padding"] == 2
    assert run.validation["empty_beam_slots"] == 94

    reproduced = {row["k"]: row for row in run.validation["metric_reproduction"]}
    assert reproduced[1]["exact_hr_fraction"] == 0.0
    assert reproduced[1]["legacy_hr_fraction"] == 0.5
    assert reproduced[3]["exact_hr_fraction"] == 1.0
    assert reproduced[3]["legacy_ndcg_fraction"] == 0.75


def test_summary_has_prefix_hits_hazards_and_deterministic_user_bootstrap(tmp_path):
    metrics, tests, info = _write_fixture(tmp_path)
    run = A.load_run(metrics / f"{VARIANT_ID}.json", test_dir=tests, info_dir=info)
    rows1, strata1 = A.summarize_all(
        [run], bootstrap_replicates=250, bootstrap_seed=7
    )
    rows2, strata2 = A.summarize_all(
        [run], bootstrap_replicates=250, bootstrap_seed=7
    )
    assert rows1 == rows2
    assert strata1 == strata2

    keyed = {(row["metric"], row["digit"], row["k"]): row for row in rows1}
    assert keyed[("top1_conditional_digit_accuracy", 1, None)]["estimate"] == 0.5
    assert keyed[("top1_first_error_hazard", 1, None)]["estimate"] == 0.5
    # Only the second example survived digit 1, and it fails at digit 2.
    d2 = keyed[("top1_conditional_digit_accuracy", 2, None)]
    assert d2["estimate"] == 0.0
    assert d2["denominator"] == 1.0
    assert keyed[("prefix_hit", 3, 1)]["estimate"] == 0.0
    assert keyed[("prefix_hit", 3, 3)]["estimate"] == 1.0
    assert keyed[("full_sid_hr_exact", None, 1)]["estimate"] == 0.0
    assert keyed[("full_sid_hr_legacy", None, 1)]["estimate"] == 0.5

    report = runner.render_report(
        {
            "scope": {
                "n_configs": 1,
                "n_example_config_rows": 2,
                "n_users": 2,
                "top_k": list(A.TOP_K),
            },
            "validation": [run.validation],
            "strata_metrics": strata1,
            "caveats": list(A.CAVEATS),
        }
    )
    assert "`result.json` is the source of truth" in report
    assert "First-error process at rank 1" in report
    assert "Prefix hit rate" in report


def test_wrong_prediction_order_is_rejected(tmp_path):
    metrics, tests, info = _write_fixture(tmp_path)
    prediction_path = metrics / f"{VARIANT_ID}.predictions.json"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction_path.write_text(json.dumps(list(reversed(payload))), encoding="utf-8")
    with pytest.raises(A.ValidationError, match="order or history drifted"):
        A.load_run(metrics / f"{VARIANT_ID}.json", test_dir=tests, info_dir=info)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda beams: beams.__setitem__(4, "<a_0><b_1><c_0>"), "after empty beam padding"),
        (lambda beams: beams.__setitem__(0, "not-a-sid"), "not a canonical SID"),
        (lambda beams: beams.__setitem__(0, "<a_7><b_7><c_7>"), "absent from the frozen info"),
    ],
)
def test_malformed_or_unregistered_beams_are_rejected(tmp_path, mutate, message):
    metrics, tests, info = _write_fixture(tmp_path)
    prediction_path = metrics / f"{VARIANT_ID}.predictions.json"
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    mutate(payload[0]["predict"])
    prediction_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(A.ValidationError, match=message):
        A.load_run(metrics / f"{VARIANT_ID}.json", test_dir=tests, info_dir=info)


def test_discovery_requires_sidecars_for_every_completed_metric(tmp_path):
    metrics, _, _ = _write_fixture(tmp_path)
    (metrics / f"{VARIANT_ID}.predictions.json").unlink()
    with pytest.raises(A.ValidationError, match="completed/prediction sidecars differ"):
        A.discover_metric_paths(metrics)


def test_expected_balanced_grid_is_fail_closed():
    runs = [
        SimpleNamespace(
            variant=A.Variant(
                variant_id=f"nextitem__Industrial_and_Scientific__{q}__{d}cb__{w}",
                category="Industrial_and_Scientific",
                quantizer=q,
                depth=d,
                width=w,
            )
        )
        for q, d, w in sorted(runner.EXPECTED_GRID)
    ]
    runner.validate_expected_grid(runs)
    with pytest.raises(A.ValidationError, match="grid changed"):
        runner.validate_expected_grid(runs[:-1])
