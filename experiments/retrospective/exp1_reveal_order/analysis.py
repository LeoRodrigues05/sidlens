"""Retrospective audit of the archived DiffGRM reveal-policy comparison.

This module intentionally analyses only the aggregate metrics already recorded
in ``registry.diffusion.json``.  It does not load a model, submit a job, or
claim that the historical comparison isolates reveal order: the confidence
branch used 256 beams while the seed-42 fixed-order branch used 64.

The unit of analysis is one trained configuration/checkpoint.  The primary
design is the complete 3 quantizers x 3 depths x 2 widths (128 and 512) lattice
(18 paired cells).  The two available width-256 cells are included only in an
all-20 sensitivity analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


TITLE = (
    "Retrospective saved-artifact audit: dynamic confidence policy vs "
    "inferred seed-42 fixed-order policy (historical beam 256 vs 64)"
)

QUANTIZERS = ("MQ", "rqkmeans", "rqvae")
DEPTHS = (3, 4, 5)
WIDTHS = (128, 256, 512)
PRIMARY_WIDTHS = (128, 512)

# Metrics with a genuinely paired archived value.  ``weighted_score`` is not
# included because the evaluator emitted it only for the confidence branch.
METRICS = (
    "ndcg@10",
    "recall@10",
    "ndcg@3",
    "ndcg@5",
    "recall@3",
    "recall@5",
    "dup@10",
    "legal_ratio",
    "duplicate_ratio",
)
PRIMARY_METRIC = "ndcg@10"
SECONDARY_METRIC = "recall@10"

EXPECTED_PRESENT_GRID = {
    ("MQ", 3, 128), ("MQ", 3, 512),
    ("MQ", 4, 128), ("MQ", 4, 512),
    ("MQ", 5, 128), ("MQ", 5, 256), ("MQ", 5, 512),
    ("rqkmeans", 3, 128), ("rqkmeans", 3, 512),
    ("rqkmeans", 4, 128), ("rqkmeans", 4, 512),
    ("rqkmeans", 5, 128), ("rqkmeans", 5, 512),
    ("rqvae", 3, 128), ("rqvae", 3, 256), ("rqvae", 3, 512),
    ("rqvae", 4, 128), ("rqvae", 4, 512),
    ("rqvae", 5, 128), ("rqvae", 5, 512),
}
FULL_GRID = {
    (quantizer, depth, width)
    for quantizer in QUANTIZERS
    for depth in DEPTHS
    for width in WIDTHS
}
PRIMARY_GRID = {
    (quantizer, depth, width)
    for quantizer in QUANTIZERS
    for depth in DEPTHS
    for width in PRIMARY_WIDTHS
}
MISSING_GRID = FULL_GRID - EXPECTED_PRESENT_GRID

HASH_FIELDS = (
    ("checkpoint", "ckpt_path", "ckpt_sha256", "ckpt"),
    ("semantic_ids", "sem_ids_path", "sem_ids_sha256", "sids"),
    ("training_log", "log_path", "log_sha256", "ckpt"),
    ("transcript", "transcript_path", "transcript_sha256", "results"),
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# The order is inferred by replaying torch.randperm(D) after manual_seed(42)
# under the archived torch 2.6.0+cu124 environment.  Batch-size 32 prevented
# the historical code from printing it, so this is evidence about the frozen
# implementation, not a per-run logged fact.
INFERRED_FIXED_ORDERS = {
    3: (0, 2, 1),
    4: (2, 3, 0, 1),
    5: (2, 4, 3, 0, 1),
}

EXPECTED_TRAINING = {
    "masking_strategy": "guided",
    "guided_conf_metric": "msp",
    "guided_select": "least",
    "guided_refresh_each_step": False,
    "lr": 0.003,
    "label_smoothing": 0.1,
    "train_batch_size": 256,
    "train_sliding": True,
    "min_hist_len": 2,
    "eval_start_epoch": 20,
    "metadata": "none",
    "sid_quantizer": "external",
}

CAVEATS = (
    "The historical test is not a reveal-order-only intervention: dynamic "
    "confidence decoding used beam_act=beam_max=256, while the fixed-order "
    "branch overrode both to 64.",
    "All checkpoints were trained with guided masking. This audit compares "
    "two inference policies, not guided training against random training.",
    "The so-called random branch sampled one digit permutation from seed 42 "
    "and then used deterministic top-k token expansion. It did not sample "
    "tokens and did not average over reveal orders.",
    "Best-checkpoint selection used confidence-only weighted validation score "
    "(0.8*NDCG@10 + 0.2*Recall@10), asymmetrically favoring that branch.",
    "There is one training seed (2024) and one inferred decode seed/order (42) "
    "per depth, so neither training-seed nor random-order uncertainty is "
    "identified.",
    "Only aggregate metrics survived. The paired unit is therefore one "
    "configuration (18 primary cells), not any of the 6,297 users; user-level "
    "paired bootstrap, permutation, or McNemar tests are impossible here.",
    "The full 3x3x3 grid is incomplete (20/27 cells). The primary analysis "
    "uses the complete width-{128,512} lattice; two width-256 cells are "
    "sensitivity-only and seven cells are absent.",
    "The runs report the same dataset counts and point to the same evaluator "
    "substrate, but per-run user identities were not retained. Configurations "
    "are not independent training replicates; sign-test p-values are only a "
    "binomial reference across the finite archived cells, not population inference.",
    "Archived recall/NDCG are exact-SID metrics with item expansion disabled. "
    "SID collisions can inflate item interpretation across quantizers/depths.",
    "The frozen source snapshot is hashed but was captured after these jobs and "
    "there is no per-run source commit/hash. Beam settings and fixed orders are "
    "therefore implementation inferences, not transcript-recorded provenance.",
    "For five-digit models the recorded guided training augmentation used four "
    "steps, unlike the three- and four-digit models where steps equalled depth.",
    "duplicate_ratio is a cross-user, batch-level statistic averaged without "
    "batch-size weighting; dup@10 is the interpretable per-user duplicate "
    "measure and both remain descriptive.",
)


class AuditError(RuntimeError):
    """Raised when the frozen registry fails an audit invariant."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_slug(metric: str) -> str:
    return metric.replace("@", "_at_")


def grid_label(cell: tuple[str, int, int]) -> str:
    quantizer, depth, width = cell
    return f"{quantizer}:{depth}x{width}"


def _sort_key(entry: Mapping) -> tuple[int, int, int]:
    return (
        QUANTIZERS.index(entry["quantizer"]),
        int(entry["n_codebook"]),
        int(entry["codebook_size"]),
    )


def _resolve_artifact_path(
    recorded: str,
    *,
    expected_section: str,
    frozen_root: Path,
) -> Path:
    """Resolve a registry path without rewriting its provenance value."""
    path = Path(recorded)
    section_root = (frozen_root / expected_section).resolve(strict=False)
    resolved_recorded = path.resolve(strict=False)
    try:
        resolved_recorded.relative_to(section_root)
    except ValueError:
        pass
    else:
        return resolved_recorded
    parts = path.parts
    markers = [
        index
        for index in range(len(parts) - 1)
        if parts[index] == "frozen" and parts[index + 1] == expected_section
    ]
    if len(markers) != 1:
        raise AuditError(
            f"cannot safely rebase {recorded!r} below frozen/{expected_section}")
    relative = Path(*parts[markers[0] + 1 :])
    candidate = (frozen_root / relative).resolve(strict=False)
    try:
        candidate.relative_to(section_root)
    except ValueError as exc:
        raise AuditError(
            f"artifact path escapes frozen/{expected_section}: {recorded!r}") from exc
    return candidate


TRANSCRIPT_METRIC_RE = re.compile(
    r"\('(?P<name>[^']+)',\s*(?:np\.float64\()?"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)?\)")


def _parse_final_transcript_metrics(text: str, ckpt_id: str) -> dict[str, float]:
    """Parse the final one-line OrderedDict without executing transcript text."""

    lines = [line for line in text.splitlines()
             if "Test Results: OrderedDict(" in line]
    if not lines:
        raise AuditError(f"{ckpt_id}: transcript lacks final Test Results")
    pairs = TRANSCRIPT_METRIC_RE.findall(lines[-1])
    metrics = {name: float(value) for name, value in pairs}
    if len(metrics) != len(pairs):
        raise AuditError(f"{ckpt_id}: duplicate metric in final Test Results")
    _validate_metric_schema(ckpt_id, metrics)
    return metrics


def _expected_metric_fields() -> set[str]:
    fields = {"weighted_score"}
    for metric in METRICS:
        fields.add(metric)
        fields.add(f"{metric}_random")
    return fields


def _validate_metric_schema(ckpt_id: str, metrics: Mapping) -> None:
    expected = _expected_metric_fields()
    actual = set(metrics)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AuditError(
            f"{ckpt_id}: metric schema mismatch; missing={missing}, extra={extra}")
    for name, value in metrics.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise AuditError(f"{ckpt_id}: non-finite metric {name}={value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise AuditError(f"{ckpt_id}: out-of-range metric {name}={value!r}")


def _validate_training(ckpt_id: str, entry: Mapping) -> None:
    training = entry.get("training")
    if not isinstance(training, Mapping):
        raise AuditError(f"{ckpt_id}: missing training metadata")
    for key, expected in EXPECTED_TRAINING.items():
        if training.get(key) != expected:
            raise AuditError(
                f"{ckpt_id}: training[{key}]={training.get(key)!r}, "
                f"expected {expected!r}")
    expected_steps = min(int(entry["n_codebook"]), 4)
    if training.get("guided_steps") != expected_steps:
        raise AuditError(
            f"{ckpt_id}: guided_steps={training.get('guided_steps')!r}, "
            f"expected archived value {expected_steps}")


def load_and_validate_registry(registry_path: Path) -> tuple[dict, list[dict], dict]:
    """Load the canonical registry and enforce the archived Exp1 design."""
    payload = json.loads(registry_path.read_text())
    if not isinstance(payload, dict):
        raise AuditError("diffusion registry must be a JSON object keyed by ckpt_id")

    next1: list[dict] = []
    for key, raw_entry in payload.items():
        if not isinstance(raw_entry, dict):
            raise AuditError(f"registry entry {key!r} is not an object")
        if raw_entry.get("ckpt_id") != key:
            raise AuditError(f"registry key/id mismatch for {key!r}")
        if raw_entry.get("task") == "next1":
            entry = dict(raw_entry)
            if entry.get("status") != "trained":
                raise AuditError(f"{key}: next1 entry is not trained")
            cell = (
                entry.get("quantizer"),
                entry.get("n_codebook"),
                entry.get("codebook_size"),
            )
            expected_id = f"diff-next1-{cell[0]}-{cell[1]}cb-{cell[2]}"
            if key != expected_id:
                raise AuditError(f"{key}: expected ckpt_id {expected_id!r}")
            config = entry.get("config", {})
            if config.get("n_digit") != cell[1]:
                raise AuditError(f"{key}: config n_digit does not match registry cell")
            if config.get("codebook_size") != cell[2]:
                raise AuditError(
                    f"{key}: config codebook_size does not match registry cell")
            _validate_metric_schema(key, entry.get("recorded_metrics", {}))
            _validate_training(key, entry)
            next1.append(entry)

    if len(payload) != 23:
        raise AuditError(f"expected 23 total diffusion entries, found {len(payload)}")
    if len(next1) != 20:
        raise AuditError(f"expected 20 trained next1 entries, found {len(next1)}")

    cells = {
        (entry["quantizer"], entry["n_codebook"], entry["codebook_size"])
        for entry in next1
    }
    if len(cells) != len(next1):
        raise AuditError("duplicate next1 configuration cells in registry")
    if cells != EXPECTED_PRESENT_GRID:
        raise AuditError(
            "next1 grid mismatch; "
            f"missing={sorted(map(grid_label, EXPECTED_PRESENT_GRID - cells))}, "
            f"unexpected={sorted(map(grid_label, cells - EXPECTED_PRESENT_GRID))}")
    if not PRIMARY_GRID.issubset(cells):
        raise AuditError("complete 18-cell primary lattice is not available")

    next1.sort(key=_sort_key)
    counts = defaultdict(int)
    for entry in payload.values():
        counts[(entry.get("task"), entry.get("status"))] += 1
    expected_inventory = {
        ("next1", "trained"): 20,
        ("next2", "trained"): 2,
        ("next2", "incomplete"): 1,
    }
    if dict(counts) != expected_inventory:
        raise AuditError(
            f"diffusion task/status inventory changed: {dict(counts)!r}")
    validation = {
        "registry_sha256": sha256_file(registry_path),
        "n_total_entries": len(payload),
        "n_next1_trained": len(next1),
        "n_next2_trained": counts[("next2", "trained")],
        "n_next2_incomplete": counts[("next2", "incomplete")],
        "metric_field_count": len(_expected_metric_fields()),
        "metric_fields": sorted(_expected_metric_fields()),
        "present_grid": sorted(map(grid_label, cells)),
        "primary_grid": sorted(map(grid_label, PRIMARY_GRID)),
        "missing_grid": sorted(map(grid_label, MISSING_GRID)),
        "checks": {
            "registry_shape": "pass",
            "exact_task_status_inventory": "pass",
            "next1_status": "pass",
            "metric_schema_and_ranges": "pass",
            "training_metadata": "pass",
            "expected_20_cell_grid": "pass",
            "complete_18_cell_primary_lattice": "pass",
        },
    }
    return payload, next1, validation


def validate_artifacts(
    entries: Sequence[Mapping],
    frozen_root: Path,
    *,
    recompute_hashes: bool = True,
) -> tuple[list[dict], dict]:
    """Validate the four registry-bound artifacts for every next1 entry."""
    records: list[dict] = []
    digest_cache: dict[Path, str] = {}
    transcript_counts: set[tuple[int, int, int]] = set()
    transcripts_matching_registry = 0
    failures: list[str] = []

    for entry in entries:
        for artifact, path_field, hash_field, section in HASH_FIELDS:
            ckpt_id = entry["ckpt_id"]
            recorded_path = entry.get(path_field)
            recorded_hash = entry.get(hash_field)
            if not isinstance(recorded_path, str) or not recorded_path:
                failures.append(f"{ckpt_id}: missing {path_field}")
                continue
            hash_well_formed = bool(
                isinstance(recorded_hash, str) and SHA256_RE.fullmatch(recorded_hash))
            if not hash_well_formed:
                failures.append(f"{ckpt_id}: malformed {hash_field}")

            resolved = _resolve_artifact_path(
                recorded_path,
                expected_section=section,
                frozen_root=frozen_root,
            )
            exists = resolved.is_file()
            actual_hash = ""
            hash_matches: bool | str = "not_recomputed"
            if not exists:
                failures.append(f"{ckpt_id}: missing artifact {resolved}")
            elif recompute_hashes:
                if resolved not in digest_cache:
                    digest_cache[resolved] = sha256_file(resolved)
                actual_hash = digest_cache[resolved]
                hash_matches = actual_hash == recorded_hash
                if not hash_matches:
                    failures.append(
                        f"{ckpt_id}: {artifact} hash mismatch "
                        f"({actual_hash} != {recorded_hash})")

            if artifact == "transcript" and exists:
                text = resolved.read_text(errors="replace")
                fields = []
                for label in (
                    "Number of users",
                    "Number of items",
                    "Number of interactions",
                ):
                    match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
                    if match is None:
                        failures.append(f"{ckpt_id}: transcript lacks {label}")
                        fields.append(-1)
                    else:
                        fields.append(int(match.group(1)))
                transcript_counts.add(tuple(fields))
                try:
                    transcript_metrics = _parse_final_transcript_metrics(text, ckpt_id)
                except AuditError as exc:
                    failures.append(str(exc))
                else:
                    registry_metrics = entry["recorded_metrics"]
                    mismatches = {
                        key: (transcript_metrics[key], float(registry_metrics[key]))
                        for key in transcript_metrics
                        if not math.isclose(
                            transcript_metrics[key], float(registry_metrics[key]),
                            rel_tol=0.0, abs_tol=1e-15)
                    }
                    if mismatches:
                        failures.append(
                            f"{ckpt_id}: final transcript metrics differ from registry: "
                            f"{mismatches}")
                    else:
                        transcripts_matching_registry += 1

            records.append({
                "ckpt_id": ckpt_id,
                "artifact": artifact,
                "recorded_path": recorded_path,
                "resolved_path": str(resolved),
                "exists": exists,
                "hash_metadata_well_formed": hash_well_formed,
                "recorded_sha256": recorded_hash or "",
                "actual_sha256": actual_hash,
                "hash_matches": hash_matches,
            })

    if transcript_counts != {(6298, 3106, 43102)}:
        failures.append(
            "transcript dataset counts differ from expected "
            f"(6298, 3106, 43102): {sorted(transcript_counts)}")
    if failures:
        raise AuditError("artifact validation failed:\n- " + "\n- ".join(failures))

    summary = {
        "status": "pass",
        "n_artifacts": len(records),
        "n_existing": sum(record["exists"] is True for record in records),
        "n_hash_metadata_well_formed": sum(
            record["hash_metadata_well_formed"] is True for record in records),
        "hashes_recomputed": recompute_hashes,
        "n_hashes_matching": (
            sum(record["hash_matches"] is True for record in records)
            if recompute_hashes else None
        ),
        "unique_transcript_dataset_counts": [
            {"n_users_including_pad": u, "n_items_including_pad": i,
             "n_interactions": n}
            for u, i, n in sorted(transcript_counts)
        ],
        "n_transcripts_metrics_matching_registry": transcripts_matching_registry,
        "n_transcript_metric_fields_matched": (
            transcripts_matching_registry * len(_expected_metric_fields())),
    }
    return records, summary


def _manifest_expected_hash(
    manifest: Mapping,
    tree: str,
    relative_path: str,
) -> str | None:
    try:
        return manifest["vendor"]["trees"][tree]["files"][relative_path]["sha256"]
    except KeyError:
        return None


def validate_implementation_evidence(
    repo_root: Path,
    frozen_root: Path,
) -> tuple[list[dict], dict]:
    """Hash the frozen source used to infer the historical decoder settings."""
    current_path = repo_root / "manifests" / "CURRENT"
    snapshot_id = current_path.read_text().strip()
    manifest_path = repo_root / "manifests" / f"provenance.{snapshot_id}.json"
    manifest = json.loads(manifest_path.read_text())

    source_specs = (
        ("diffgrm", "genrec/default.yaml"),
        ("diffgrm", "genrec/models/DIFF_GRM/config.yaml"),
        ("diffgrm", "genrec/models/DIFF_GRM/trainer.py"),
        ("diffgrm", "genrec/models/DIFF_GRM/evaluator.py"),
        ("diffgrm", "genrec/models/DIFF_GRM/beam.py"),
        ("onediffrec", "scripts/run_diffgrm.sbatch"),
    )
    evidence: list[dict] = []
    failures: list[str] = []
    for tree, relative in source_specs:
        path = repo_root / "vendor" / tree / relative
        actual = sha256_file(path)
        expected = _manifest_expected_hash(manifest, tree, relative)
        matches = actual == expected
        if not matches:
            failures.append(f"vendor/{tree}/{relative}: snapshot hash mismatch")
        evidence.append({
            "tree": tree,
            "relative_path": relative,
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "hash_matches_snapshot": matches,
        })

    environment_path = frozen_root / "results/job_manifests/eval-pip-freeze.txt"
    environment_record = manifest["data"]["results/job_manifests"]["files"][
        "eval-pip-freeze.txt"]
    environment_hash = sha256_file(environment_path)
    environment_matches = environment_hash == environment_record["sha256"]
    if not environment_matches:
        failures.append("eval-pip-freeze.txt: snapshot hash mismatch")
    evidence.append({
        "tree": "frozen_environment",
        "relative_path": "results/job_manifests/eval-pip-freeze.txt",
        "path": str(environment_path),
        "expected_sha256": environment_record["sha256"],
        "actual_sha256": environment_hash,
        "hash_matches_snapshot": environment_matches,
    })
    environment_text = environment_path.read_text()
    torch_match = re.search(r"^torch==([^\n]+)$", environment_text, re.MULTILINE)
    numpy_match = re.search(r"^numpy==([^\n]+)$", environment_text, re.MULTILINE)
    if torch_match is None or numpy_match is None:
        failures.append("eval-pip-freeze.txt lacks torch or numpy version")

    default_config = yaml.safe_load(
        (repo_root / "vendor/diffgrm/genrec/default.yaml").read_text())
    model_config = yaml.safe_load(
        (repo_root / "vendor/diffgrm/genrec/models/DIFF_GRM/config.yaml").read_text())
    random_beam = model_config["random_beam"]
    vectorized = model_config["vectorized_beam_search"]
    settings = {
        "training_seed": default_config["rand_seed"],
        "reproducibility": default_config["reproducibility"],
        "beam_modes": model_config["beam_search_modes"],
        "decode_seed": random_beam["seed"],
        "confidence_test_beam_act": vectorized["test"]["beam_act"],
        "confidence_test_beam_max": vectorized["test"]["beam_max"],
        "fixed_test_beam_act": random_beam["beam_act"],
        "fixed_test_beam_max": random_beam["beam_max"],
        "top_k_final": vectorized["top_k_final"],
        "validation_metric": model_config["val_metric"],
        "eval_expand_sid_to_items": model_config["eval_expand_sid_to_items"],
        "archived_torch_version": torch_match.group(1) if torch_match else None,
        "archived_numpy_version": numpy_match.group(1) if numpy_match else None,
        "inferred_fixed_orders": {
            str(depth): list(order)
            for depth, order in INFERRED_FIXED_ORDERS.items()
        },
    }
    expected_settings = {
        "training_seed": 2024,
        "decode_seed": 42,
        "confidence_test_beam_act": 256,
        "confidence_test_beam_max": 256,
        "fixed_test_beam_act": 64,
        "fixed_test_beam_max": 64,
        "top_k_final": 10,
        "validation_metric": "weighted_score",
        "eval_expand_sid_to_items": False,
        "archived_torch_version": "2.6.0+cu124",
    }
    for key, expected in expected_settings.items():
        if settings[key] != expected:
            failures.append(
                f"implementation setting {key}={settings[key]!r}, expected {expected!r}")
    if settings["beam_modes"] != ["confidence", "random"]:
        failures.append(
            f"beam_modes={settings['beam_modes']!r}, expected ['confidence', 'random']")
    if failures:
        raise AuditError("implementation evidence failed:\n- " + "\n- ".join(failures))

    return evidence, {
        "status": "pass",
        "snapshot_id": snapshot_id,
        "snapshot_manifest": str(manifest_path),
        "snapshot_manifest_sha256": sha256_file(manifest_path),
        "source_and_environment_hashes_match_snapshot": True,
        "settings": settings,
        "provenance_limit": (
            "The snapshot hashes these files after the jobs; no run-specific "
            "source hash was logged. Settings/orders are inferred, not observed.")
    }


def validate_dataset(repo_root: Path, frozen_root: Path) -> dict:
    """Validate and count the shared leave-last-item test substrate."""
    snapshot_id = (repo_root / "manifests/CURRENT").read_text().strip()
    manifest_path = repo_root / "manifests" / f"provenance.{snapshot_id}.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["data"]["data/sequences"]["files"]["all_item_seqs.json"]
    sequence_path = frozen_root / "data/sequences/all_item_seqs.json"
    actual_hash = sha256_file(sequence_path)
    if actual_hash != record["sha256"]:
        raise AuditError("all_item_seqs.json does not match the frozen manifest")
    sequences = json.loads(sequence_path.read_text())
    if not isinstance(sequences, dict):
        raise AuditError("all_item_seqs.json must be an object keyed by user")
    n_examples = len(sequences)
    n_interactions = sum(len(items) for items in sequences.values())
    if (n_examples, n_interactions) != (6297, 43102):
        raise AuditError(
            "unexpected sequence counts: "
            f"users={n_examples}, interactions={n_interactions}")
    return {
        "status": "pass",
        "path": str(sequence_path),
        "sha256": actual_hash,
        "manifest_sha256": record["sha256"],
        "n_test_examples": n_examples,
        "n_interactions": n_interactions,
        "split": (
            "One test row per real user; history is all but the final item and "
            "the final item is the target (leave-one-out)."),
    }


def build_cell_rows(entries: Sequence[Mapping]) -> list[dict]:
    rows: list[dict] = []
    for entry in entries:
        metrics = entry["recorded_metrics"]
        depth = int(entry["n_codebook"])
        width = int(entry["codebook_size"])
        row = {
            "ckpt_id": entry["ckpt_id"],
            "primary_balanced18": width in PRIMARY_WIDTHS,
            "analysis_role": (
                "primary_balanced_lattice"
                if width in PRIMARY_WIDTHS
                else "width256_sensitivity_only"
            ),
            "quantizer": entry["quantizer"],
            "n_codebook": depth,
            "codebook_size": width,
            "source_jobid": entry["source_jobid"],
            "source_stamp": entry["source_stamp"],
            "trained_at": entry["trained_at"],
            "training_seed_inferred": 2024,
            "decode_seed_inferred": 42,
            "fixed_order_inferred": "-".join(
                map(str, INFERRED_FIXED_ORDERS[depth])),
            "confidence_beam_act": 256,
            "fixed_seed42_beam_act": 64,
            "guided_training_steps": entry["training"]["guided_steps"],
        }
        for metric in METRICS:
            slug = metric_slug(metric)
            confidence = float(metrics[metric])
            fixed = float(metrics[f"{metric}_random"])
            row[f"{slug}_confidence"] = confidence
            row[f"{slug}_fixed_seed42"] = fixed
            row[f"{slug}_delta_confidence_minus_fixed"] = confidence - fixed
        rows.append(row)
    return rows


def _quantile(values: Sequence[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("quantile requires at least one value")
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def summarize_metric(rows: Sequence[Mapping], metric: str) -> dict:
    slug = metric_slug(metric)
    confidence = [float(row[f"{slug}_confidence"]) for row in rows]
    fixed = [float(row[f"{slug}_fixed_seed42"]) for row in rows]
    deltas = [float(row[f"{slug}_delta_confidence_minus_fixed"]) for row in rows]
    n = len(deltas)
    if not n:
        raise ValueError("cannot summarize an empty cell set")
    mean_confidence = math.fsum(confidence) / n
    mean_fixed = math.fsum(fixed) / n
    mean_delta = math.fsum(deltas) / n
    return {
        "metric": metric,
        "n_cells": n,
        "mean_confidence": mean_confidence,
        "mean_fixed_seed42": mean_fixed,
        "mean_delta": mean_delta,
        "relative_mean_delta_vs_fixed": (
            mean_delta / mean_fixed if mean_fixed != 0 else None),
        "median_delta": statistics.median(deltas),
        "q1_delta": _quantile(deltas, 0.25),
        "q3_delta": _quantile(deltas, 0.75),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "n_positive": sum(value > 0 for value in deltas),
        "n_negative": sum(value < 0 for value in deltas),
        "n_zero": sum(value == 0 for value in deltas),
        "proportion_positive_among_nonzero": (
            sum(value > 0 for value in deltas)
            / sum(value != 0 for value in deltas)
            if any(value != 0 for value in deltas) else None
        ),
    }


def exact_two_sided_sign_test(values: Iterable[float], tolerance: float = 1e-15) -> dict:
    """Exact two-sided binomial sign test, dropping numerical ties.

    The implementation doubles the smaller symmetric Binomial(n, 0.5) tail.
    It is exact for p=0.5 and deterministic; no Monte Carlo seed is involved.
    """
    values = list(values)
    positive = sum(value > tolerance for value in values)
    negative = sum(value < -tolerance for value in values)
    ties = len(values) - positive - negative
    n = positive + negative
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2 ** n))
    return {
        "test": "exact_two_sided_sign_test",
        "n_nonzero": n,
        "n_positive": positive,
        "n_negative": negative,
        "n_ties": ties,
        "p_value": p_value,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm adjusted p-values in the input order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def hypothesis_rows(rows: Sequence[Mapping], subset: str) -> list[dict]:
    endpoints = (
        (PRIMARY_METRIC, "analysis_designated_primary"),
        (SECONDARY_METRIC, "analysis_designated_secondary"),
    )
    output: list[dict] = []
    for metric, role in endpoints:
        slug = metric_slug(metric)
        deltas = [
            float(row[f"{slug}_delta_confidence_minus_fixed"])
            for row in rows
        ]
        test = exact_two_sided_sign_test(deltas)
        summary = summarize_metric(rows, metric)
        output.append({
            "subset": subset,
            "analysis_role": (
                "primary_complete_lattice"
                if subset == "balanced18" else "all20_sensitivity"
            ),
            "endpoint_role": role,
            "metric": metric,
            "unit": "configuration_cell",
            "n_cells": summary["n_cells"],
            "mean_delta": summary["mean_delta"],
            "median_delta": summary["median_delta"],
            "n_positive": test["n_positive"],
            "n_negative": test["n_negative"],
            "n_ties": test["n_ties"],
            "exact_two_sided_sign_p": test["p_value"],
        })
    adjusted = holm_adjust([row["exact_two_sided_sign_p"] for row in output])
    for row, p_value in zip(output, adjusted):
        row["holm_p_across_ndcg10_recall10"] = p_value
        row["inference_scope"] = (
            "Descriptive finite-configuration evidence only; shared users, "
            "single training seed, and unequal beam budgets preclude a causal "
            "or population-level claim."
        )
    return output


def build_strata_rows(cell_rows: Sequence[Mapping]) -> list[dict]:
    output: list[dict] = []
    subsets = {
        "balanced18": [row for row in cell_rows if row["primary_balanced18"]],
        "all20_sensitivity": list(cell_rows),
    }
    for subset_name, subset_rows in subsets.items():
        groupings: list[tuple[str, str, list[Mapping]]] = [
            ("overall", "all", subset_rows),
        ]
        for quantizer in QUANTIZERS:
            groupings.append((
                "quantizer",
                quantizer,
                [row for row in subset_rows if row["quantizer"] == quantizer],
            ))
        for depth in DEPTHS:
            groupings.append((
                "depth",
                str(depth),
                [row for row in subset_rows if row["n_codebook"] == depth],
            ))
        for group_kind, stratum, rows in groupings:
            if not rows:
                continue
            for metric in METRICS:
                summary = summarize_metric(rows, metric)
                output.append({
                    "subset": subset_name,
                    "group_by": group_kind,
                    "stratum": stratum,
                    **summary,
                })
    return output


def _summary_map(rows: Sequence[Mapping]) -> dict[str, dict]:
    return {metric: summarize_metric(rows, metric) for metric in METRICS}


def build_result(
    *,
    repo_root: Path,
    registry_path: Path,
    cell_rows: Sequence[Mapping],
    registry_validation: Mapping,
    artifact_summary: Mapping,
    implementation_summary: Mapping,
    dataset_summary: Mapping,
) -> dict:
    primary_rows = [row for row in cell_rows if row["primary_balanced18"]]
    hypothesis_tests = (
        hypothesis_rows(primary_rows, "balanced18")
        + hypothesis_rows(cell_rows, "all20_sensitivity")
    )
    return {
        "experiment_id": "retrospective_exp1_reveal_order",
        "title": TITLE,
        "analysis_kind": "saved_aggregate_artifact_audit",
        "repository": str(repo_root),
        "inputs": {
            "registry": str(registry_path),
            "registry_sha256": registry_validation["registry_sha256"],
            "snapshot_id": implementation_summary["snapshot_id"],
            "dataset": dataset_summary,
        },
        "design": {
            "comparison": (
                "Same archived checkpoint and common evaluator substrate: dynamic "
                "confidence position policy versus inferred seed-42 fixed digit order. "
                "Per-run user identities were not retained."),
            "primary_subset": (
                "18 cells: 3 quantizers x depths {3,4,5} x widths {128,512}."),
            "sensitivity_subset": (
                "All 20 next1 cells, adding MQ 5x256 and rqvae 3x256."),
            "unit": "configuration/checkpoint cell",
            "n_test_examples_per_cell": dataset_summary["n_test_examples"],
            "primary_endpoint": PRIMARY_METRIC,
            "secondary_endpoint": SECONDARY_METRIC,
            "descriptive_endpoints": [
                metric for metric in METRICS
                if metric not in {PRIMARY_METRIC, SECONDARY_METRIC}
            ],
            "effect": (
                "confidence metric minus fixed-seed42 metric on the native [0,1] "
                "scale; report tables express absolute deltas in percentage points"),
            "test": (
                "Exact-arithmetic two-sided configuration-level sign statistic, "
                "shown only as a binomial reference diagnostic. NDCG@10 was "
                "analysis-designated, not preregistered; Holm values across it and "
                "Recall@10 are also reported."),
            "fixed_orders_inferred": {
                str(depth): list(order)
                for depth, order in INFERRED_FIXED_ORDERS.items()
            },
        },
        "validation": {
            "status": "pass",
            "registry": registry_validation,
            "artifacts": artifact_summary,
            "implementation_evidence": implementation_summary,
        },
        "results": {
            "balanced18": {
                "n_cells": len(primary_rows),
                "metric_summaries": _summary_map(primary_rows),
            },
            "all20_sensitivity": {
                "n_cells": len(cell_rows),
                "metric_summaries": _summary_map(cell_rows),
            },
            "hypothesis_tests": hypothesis_tests,
        },
        "interpretation": (
            "Across the complete 18-cell lattice, the archived dynamic-confidence "
            "condition has higher average NDCG@10 and Recall@10 than the archived "
            "fixed seed-42 condition. Because beam budget and checkpoint selection "
            "also favored confidence, this is an implementation-level association, "
            "not an estimate of the causal effect of reveal order."),
        "caveats": list(CAVEATS),
    }


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _fmt_p(value: float) -> str:
    return f"{value:.6g}"


def render_report(
    result: Mapping,
    cell_rows: Sequence[Mapping],
    strata_rows: Sequence[Mapping],
) -> str:
    main = result["results"]["balanced18"]["metric_summaries"]
    sensitivity = result["results"]["all20_sensitivity"]["metric_summaries"]
    tests = result["results"]["hypothesis_tests"]
    validation = result["validation"]

    lines = [
        f"# {TITLE}",
        "",
        "## Outcome",
        "",
        "The saved aggregates show a small, broadly positive association for the "
        "dynamic-confidence branch, but they do **not** identify the effect of reveal "
        "order. The historical branches used unequal test beam budgets (256 versus "
        "64), and checkpoint selection used the confidence-only validation score.",
        "",
        "| Subset | Metric | Mean confidence [0,1] | Mean fixed seed-42 [0,1] | "
        "Mean delta (pp) | Median delta (pp) | Positive cells | Relative delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, summaries in (("balanced 18 (primary)", main),
                              ("all 20 (sensitivity)", sensitivity)):
        for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
            summary = summaries[metric]
            lines.append(
                f"| {subset} | {metric} | {_fmt(summary['mean_confidence'])} | "
                f"{_fmt(summary['mean_fixed_seed42'])} | "
                f"{_fmt(100 * summary['mean_delta'])} | "
                f"{_fmt(100 * summary['median_delta'])} | "
                f"{summary['n_positive']}/{summary['n_cells']} | "
                f"{summary['relative_mean_delta_vs_fixed']:.2%} |")

    lines.extend([
        "",
        "The test-statistic rows below use configurations—not users—as the paired "
        "unit. "
        "They are deterministic, exact-arithmetic two-sided sign statistics. NDCG@10 "
        "is the analysis-designated endpoint, not a preregistered one; Holm values show "
        "the joint reading with Recall@10. The p-values are binomial references only: "
        "the finite configurations reuse data and are not independent training replicates.",
        "",
        "| Subset | Role | Metric | Signs (+/-/tie) | Exact p | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for test in tests:
        lines.append(
            f"| {test['subset']} | {test['endpoint_role']} | {test['metric']} | "
            f"{test['n_positive']}/{test['n_negative']}/{test['n_ties']} | "
            f"{_fmt_p(test['exact_two_sided_sign_p'])} | "
            f"{_fmt_p(test['holm_p_across_ndcg10_recall10'])} |")

    lines.extend([
        "",
        "## Balanced-lattice strata",
        "",
        "These strata are effect summaries only; no subgroup hypothesis tests were "
        "performed.",
        "",
        "| Group | Stratum | Cells | NDCG@10 delta (pp) | Recall@10 delta (pp) |",
        "|---|---:|---:|---:|---:|",
    ])
    index = {
        (row["group_by"], row["stratum"], row["metric"]): row
        for row in strata_rows
        if row["subset"] == "balanced18"
    }
    for group_by, strata in (("quantizer", QUANTIZERS),
                             ("depth", tuple(map(str, DEPTHS)))):
        for stratum in strata:
            ndcg = index[(group_by, stratum, PRIMARY_METRIC)]
            recall = index[(group_by, stratum, SECONDARY_METRIC)]
            lines.append(
                f"| {group_by} | {stratum} | {ndcg['n_cells']} | "
                f"{_fmt(100 * ndcg['mean_delta'])} | "
                f"{_fmt(100 * recall['mean_delta'])} |")

    lines.extend([
        "",
        "## Per-cell primary deltas",
        "",
        "| Quantizer | Depth | Width | NDCG@10 delta (pp) | Recall@10 delta (pp) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in cell_rows:
        if not row["primary_balanced18"]:
            continue
        lines.append(
            f"| {row['quantizer']} | {row['n_codebook']} | "
            f"{row['codebook_size']} | "
            f"{_fmt(100 * row['ndcg_at_10_delta_confidence_minus_fixed'])} | "
            f"{_fmt(100 * row['recall_at_10_delta_confidence_minus_fixed'])} |")

    artifacts = validation["artifacts"]
    registry = validation["registry"]
    implementation = validation["implementation_evidence"]
    lines.extend([
        "",
        "## Frozen-input validation",
        "",
        f"- Registry: `{result['inputs']['registry']}` "
        f"(`{result['inputs']['registry_sha256']}`).",
        f"- Registry inventory: {registry['n_total_entries']} total; "
        f"{registry['n_next1_trained']} trained next-item; "
        f"{registry['n_next2_trained']} trained and "
        f"{registry['n_next2_incomplete']} incomplete next-two.",
        f"- Artifact hashes: {artifacts['n_hashes_matching']}/"
        f"{artifacts['n_artifacts']} recomputed hashes match registry metadata.",
        f"- Transcript metrics: "
        f"{artifacts['n_transcript_metric_fields_matched']} fields across "
        f"{artifacts['n_transcripts_metrics_matching_registry']} transcripts exactly "
        "match the registry values used for analysis.",
        f"- Test substrate: {result['inputs']['dataset']['n_test_examples']:,} "
        "leave-last-item examples in the common frozen source; per-run user IDs were "
        "not retained for an identity-level check.",
        f"- Frozen source snapshot: `{implementation['snapshot_id']}`; all six "
        "source files plus the archived environment manifest match its hashes "
        f"(torch {implementation['settings']['archived_torch_version']}).",
        f"- Missing grid cells ({len(registry['missing_grid'])}): "
        + ", ".join(f"`{cell}`" for cell in registry["missing_grid"]) + ".",
        "",
        "Outputs: [per_cell.csv](per_cell.csv), [strata.csv](strata.csv), "
        "[hypothesis_tests.csv](hypothesis_tests.csv), "
        "[artifact_validation.csv](artifact_validation.csv), and "
        "[implementation_evidence.csv](implementation_evidence.csv).",
        "",
        "## Interpretation boundary",
        "",
        result["interpretation"],
        "",
        "A clean follow-up must hold beam width, scoring, filtering, test rows and "
        "checkpoint fixed; enumerate all 6 orders at depth 3, all 24 at depth 4, "
        "and all 120 at depth 5 (or preregister a sample), then save per-user ranks "
        "for genuinely paired tests.",
        "",
        "## Caveats",
        "",
    ])
    lines.extend(f"{index}. {caveat}" for index, caveat in enumerate(CAVEATS, 1))
    lines.append("")
    return "\n".join(lines)


def run_audit(
    *,
    repo_root: Path,
    registry_path: Path,
    frozen_root: Path,
    out_dir: Path,
    recompute_hashes: bool = True,
) -> dict:
    _, entries, registry_validation = load_and_validate_registry(registry_path)
    artifact_rows, artifact_summary = validate_artifacts(
        entries,
        frozen_root,
        recompute_hashes=recompute_hashes,
    )
    implementation_rows, implementation_summary = validate_implementation_evidence(
        repo_root, frozen_root)
    dataset_summary = validate_dataset(repo_root, frozen_root)
    cell_rows = build_cell_rows(entries)
    strata_rows = build_strata_rows(cell_rows)
    test_rows = (
        hypothesis_rows(
            [row for row in cell_rows if row["primary_balanced18"]],
            "balanced18",
        )
        + hypothesis_rows(cell_rows, "all20_sensitivity")
    )
    result = build_result(
        repo_root=repo_root,
        registry_path=registry_path,
        cell_rows=cell_rows,
        registry_validation=registry_validation,
        artifact_summary=artifact_summary,
        implementation_summary=implementation_summary,
        dataset_summary=dataset_summary,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_cell.csv", cell_rows)
    _write_csv(out_dir / "strata.csv", strata_rows)
    _write_csv(out_dir / "hypothesis_tests.csv", test_rows)
    _write_csv(out_dir / "artifact_validation.csv", artifact_rows)
    _write_csv(out_dir / "implementation_evidence.csv", implementation_rows)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    (out_dir / "report.md").write_text(
        render_report(result, cell_rows, strata_rows))
    return result


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_frozen_root() -> Path:
    work = Path(os.environ.get("SIDLENS_WORK", "/l/users/leo.rodrigues/sidlens"))
    return work / "frozen"
