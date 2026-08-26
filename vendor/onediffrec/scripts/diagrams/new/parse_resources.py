#!/usr/bin/env python3
"""Build a reproducible resource table for completed next-item sweep runs.

The performance metrics and wall-clock timings come from the sweep JSON files.
Hugging Face's FLOP estimate and the more precise training runtime come from the
corresponding offline W&B protobuf log.  The two sources are joined only when
``variant_id`` exactly equals the W&B run ``display_name``.

Evaluation duration is a measured end-to-end wall-clock interval around the
four-GPU evaluation script.  The parser also validates the prediction count and
beam count from the retained evaluation artifacts.  Existing artifacts did not
record inference FLOPs or inference-only accelerator telemetry; the output
keeps that absence explicit instead of deriving a misleading value from wall
time.

W&B's :class:`DataStore` normally opens logs with ``r+b`` even when scanning.
``ReadOnlyDataStore`` deliberately overrides that behavior and uses ``rb`` so
running this analysis can never alter the experiment logs.

``gpu_hours`` and ``hf_estimated_achieved_tflops_per_gpu`` use the W&B
``train_runtime`` and the four GPUs recorded in the W&B environment metadata::

    gpu_hours = train_runtime_seconds * gpu_count / 3600
    achieved_TFLOP/s/GPU = HF_total_FLOPs / train_runtime_seconds
                           / gpu_count / 1e12

The throughput column is explicitly labelled as HF-estimated because
``total_flos`` is Transformers' analytical estimate, not a hardware-counter
measurement.  The table also includes direct W&B system telemetry summaries
for utilization, allocated memory, power, and sampled GPU energy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_ROOT = Path(
    "/l/users/leo.rodrigues/onediffrec/sweep/next-item/metrics"
)
DEFAULT_WANDB_ROOT = Path("/l/users/leo.rodrigues/onediffrec/wandb/wandb")
DEFAULT_OUTPUT = REPO_ROOT / "diagrams" / "new" / "data" / "resource_runs.csv"

EXPECTED_CATEGORY = "Industrial_and_Scientific"
EXPECTED_TREE = "next-item"
EXPECTED_GPU_MODEL = "NVIDIA A100-SXM4-40GB"
EXPECTED_GPU_COUNT = 4

METHOD_ORDER = {"rqvae": 0, "rqkmeans": 1, "MQ": 2}
TERMINAL_WANDB_KEYS = {
    "total_flos",
    "train_runtime",
    "_step",
    "train/global_step",
    "train/epoch",
}

CSV_FIELDS = (
    "variant_id",
    "tree",
    "category",
    "method",
    "codebooks",
    "codebook_size",
    "component_tokens",
    "items",
    "max_steps",
    "hr_at_10",
    "ndcg_at_10",
    "stopped_at_step",
    "stopped_at_epoch",
    "train_wall_seconds",
    "eval_seconds",
    "evaluation_examples",
    "evaluation_num_beams",
    "evaluation_seconds_per_example",
    "evaluation_allocated_gpu_hours",
    "inference_flops",
    "inference_flops_status",
    "hf_total_flops",
    "wandb_train_runtime_seconds",
    "wandb_step",
    "wandb_train_global_step",
    "wandb_train_epoch",
    "model_parameters",
    "base_model",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "effective_global_batch_size",
    "learning_rate",
    "seed",
    "gpu_model",
    "gpu_count",
    "gpu_memory_gib_per_gpu",
    "cuda_version",
    "gpu_hours",
    "hf_estimated_achieved_tflops_per_gpu",
    "gpu_telemetry_samples",
    "gpu_telemetry_duration_seconds",
    "mean_gpu_util_percent",
    "p05_gpu_util_percent",
    "peak_gpu_memory_allocated_gib",
    "mean_power_watts_per_gpu",
    "measured_gpu_energy_kwh",
    "wandb_run_id",
    "wandb_run_started_at_utc",
    "wandb_run_path",
)


class DependencyError(RuntimeError):
    """Raised when the optional W&B log-reading dependency is unavailable."""


warnings.filterwarnings(
    "ignore",
    message=r"The '(repr|frozen)' attribute .* was provided to the `Field\(\)` function.*",
    category=UserWarning,
)

try:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
except ImportError as exc:  # Let JSON-only imports work without W&B installed.
    wandb_internal_pb2 = None  # type: ignore[assignment]
    DataStore = None  # type: ignore[assignment,misc]
    _WANDB_IMPORT_ERROR: ImportError | None = exc
else:
    _WANDB_IMPORT_ERROR = None


if DataStore is not None:

    class ReadOnlyDataStore(DataStore):  # type: ignore[misc,valid-type]
        """W&B LevelDB-log reader that never requests write access."""

        def open_for_scan(self, fname: str | os.PathLike[str]) -> None:
            self._fname = os.fspath(fname)
            self._fp = open(fname, "rb")
            self._index = 0
            self._size_bytes = os.stat(fname).st_size
            self._opened_for_scan = True
            self._read_header()

else:

    class ReadOnlyDataStore:  # pragma: no cover - used only without W&B
        """Placeholder that reports how to enable offline W&B parsing."""

        def __init__(self) -> None:
            _require_wandb()


def _require_wandb() -> None:
    if _WANDB_IMPORT_ERROR is not None:
        raise DependencyError(
            "Reading offline .wandb files requires the optional `wandb` Python "
            "package. Install it with `python -m pip install wandb`, or run this "
            "repository's `.conda/bin/python`. Original import error: "
            f"{_WANDB_IMPORT_ERROR}"
        ) from _WANDB_IMPORT_ERROR


def _item_key(item: Any) -> str:
    """Support both scalar and nested keys used by recent W&B protobufs."""

    if item.key:
        return item.key
    return ".".join(item.nested_key)


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _linear_percentile(values: Iterable[float], q: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


@dataclass
class Telemetry:
    """Direct per-GPU samples decoded from W&B stats records."""

    utilization: list[float] = field(default_factory=list)
    allocated_memory_bytes: list[float] = field(default_factory=list)
    power_watts: list[float] = field(default_factory=list)
    # Each entry is timestamp plus a mapping from GPU index to its power.
    summed_power_samples: list[tuple[float, dict[int, float]]] = field(
        default_factory=list
    )

    def add(self, stats: Any) -> None:
        values = {item.key: _json_value(item.value_json) for item in stats.item}

        # W&B alternates GPU and host-stat records at nearly identical times.
        # Filtering this way avoids treating host-only rows as zero-GPU rows and
        # halving the trapezoidal energy estimate.
        if "gpu.0.gpu" not in values:
            return

        timestamp = stats.timestamp.seconds + stats.timestamp.nanos / 1e9
        sample_power: dict[int, float] = {}
        for key, value in values.items():
            match = re.fullmatch(
                r"gpu\.(\d+)\.(gpu|memoryAllocatedBytes|powerWatts)", key
            )
            if match is None:
                continue
            number = _optional_number(value)
            if number is None:
                continue
            gpu_index = int(match.group(1))
            metric = match.group(2)
            if metric == "gpu":
                self.utilization.append(number)
            elif metric == "memoryAllocatedBytes":
                self.allocated_memory_bytes.append(number)
            elif metric == "powerWatts":
                self.power_watts.append(number)
                sample_power[gpu_index] = number
        if sample_power:
            self.summed_power_samples.append((timestamp, sample_power))

    def summarize(self, gpu_count: int) -> dict[str, float | int | None]:
        complete_power_samples = sorted(
            (
                (timestamp, sum(power_by_gpu.values()))
                for timestamp, power_by_gpu in self.summed_power_samples
                if set(power_by_gpu) == set(range(gpu_count))
            ),
            key=lambda sample: sample[0],
        )

        energy_watt_seconds = 0.0
        for (time_a, power_a), (time_b, power_b) in zip(
            complete_power_samples, complete_power_samples[1:]
        ):
            elapsed = time_b - time_a
            if elapsed > 0:
                energy_watt_seconds += (power_a + power_b) * 0.5 * elapsed

        duration = None
        if len(complete_power_samples) >= 2:
            duration = complete_power_samples[-1][0] - complete_power_samples[0][0]

        return {
            "gpu_telemetry_samples": len(complete_power_samples),
            "gpu_telemetry_duration_seconds": duration,
            "mean_gpu_util_percent": (
                statistics.fmean(self.utilization) if self.utilization else None
            ),
            "p05_gpu_util_percent": _linear_percentile(self.utilization, 0.05),
            "peak_gpu_memory_allocated_gib": (
                max(self.allocated_memory_bytes) / 1024**3
                if self.allocated_memory_bytes
                else None
            ),
            "mean_power_watts_per_gpu": (
                statistics.fmean(self.power_watts) if self.power_watts else None
            ),
            "measured_gpu_energy_kwh": (
                energy_watt_seconds / 3_600_000
                if len(complete_power_samples) >= 2
                else None
            ),
        }


@dataclass
class WandbRun:
    path: Path
    display_name: str | None = None
    run_id: str | None = None
    started_at: float = 0.0
    exit_code: int | None = None
    scan_error: str | None = None
    terminal: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    gpu_model: str | None = None
    gpu_count: int | None = None
    gpu_memory_bytes_per_gpu: int | None = None
    cuda_version: str | None = None
    telemetry: Telemetry = field(default_factory=Telemetry)

    def complete(self) -> bool:
        if self.scan_error is not None or self.exit_code != 0 or not self.display_name:
            return False
        for key in ("total_flos", "train_runtime", "_step"):
            number = _optional_number(self.terminal.get(key))
            if number is None or number < 0:
                return False
        return (
            self.gpu_count == EXPECTED_GPU_COUNT
            and self.gpu_model == EXPECTED_GPU_MODEL
        )


@dataclass(frozen=True)
class BuildReport:
    sweep_json_files: int
    completed_sweep_rows: int
    wandb_files: int
    completed_wandb_runs: int
    incomplete_wandb_runs: int
    duplicate_completed_display_names: int
    unpaired_sweep_rows: int


def _update_proto_items(state: dict[str, Any], update: Any, remove: Any) -> None:
    for item in update:
        key = _item_key(item)
        if key:
            state[key] = _json_value(item.value_json)
    for item in remove:
        key = _item_key(item)
        if key:
            state.pop(key, None)


def scan_wandb_run(path: Path) -> WandbRun:
    """Decode one offline W&B log without modifying it."""

    _require_wandb()
    run = WandbRun(path=path)
    store = ReadOnlyDataStore()
    try:
        store.open_for_scan(path)
        while True:
            raw = store.scan_data()
            if raw is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(raw)
            record_type = record.WhichOneof("record_type")

            if record_type == "run":
                if record.run.display_name:
                    run.display_name = record.run.display_name
                if record.run.run_id:
                    run.run_id = record.run.run_id
                if record.run.HasField("start_time"):
                    run.started_at = (
                        record.run.start_time.seconds
                        + record.run.start_time.nanos / 1e9
                    )
                _update_proto_items(
                    run.config, record.run.config.update, record.run.config.remove
                )
            elif record_type == "config":
                _update_proto_items(
                    run.config, record.config.update, record.config.remove
                )
            elif record_type == "summary":
                for item in record.summary.update:
                    key = _item_key(item)
                    if key in TERMINAL_WANDB_KEYS:
                        run.terminal[key] = _json_value(item.value_json)
                for item in record.summary.remove:
                    run.terminal.pop(_item_key(item), None)
            elif record_type == "history":
                for item in record.history.item:
                    key = _item_key(item)
                    if key in TERMINAL_WANDB_KEYS:
                        run.terminal[key] = _json_value(item.value_json)
            elif record_type == "environment":
                environment = record.environment
                if environment.gpu_type:
                    run.gpu_model = environment.gpu_type
                if environment.gpu_count:
                    run.gpu_count = int(environment.gpu_count)
                if environment.gpu_nvidia:
                    memory_sizes = {
                        int(gpu.memory_total)
                        for gpu in environment.gpu_nvidia
                        if gpu.memory_total
                    }
                    if len(memory_sizes) == 1:
                        run.gpu_memory_bytes_per_gpu = memory_sizes.pop()
                if environment.cuda_version:
                    run.cuda_version = environment.cuda_version
            elif record_type == "stats":
                run.telemetry.add(record.stats)
            elif record_type == "exit":
                run.exit_code = int(record.exit.exit_code)
    except (AssertionError, OSError, ValueError) as exc:
        # A live/truncated log normally fails a final record assertion.  Keep
        # that condition on the object so it is excluded as incomplete.
        run.scan_error = f"{type(exc).__name__}: {exc}"
    finally:
        store.close()
    return run


def load_completed_wandb_runs(
    wandb_root: Path,
) -> tuple[dict[str, WandbRun], int, int, int]:
    if not wandb_root.is_dir():
        raise FileNotFoundError(f"W&B root not found: {wandb_root}")

    paths = sorted(wandb_root.glob("offline-run-*/run-*.wandb"))
    if not paths:
        raise FileNotFoundError(
            f"No offline-run-*/run-*.wandb files found under: {wandb_root}"
        )

    all_runs = [scan_wandb_run(path) for path in paths]
    completed = [run for run in all_runs if run.complete()]
    by_name: dict[str, list[WandbRun]] = {}
    for run in completed:
        assert run.display_name is not None
        by_name.setdefault(run.display_name, []).append(run)

    duplicate_names = sum(len(candidates) > 1 for candidates in by_name.values())
    selected = {
        name: max(candidates, key=lambda run: (run.started_at, str(run.path)))
        for name, candidates in by_name.items()
    }
    return selected, len(paths), len(completed), duplicate_names


def load_completed_sweep_metrics(sweep_root: Path) -> tuple[list[dict[str, Any]], int]:
    if not sweep_root.is_dir():
        raise FileNotFoundError(f"Sweep metrics root not found: {sweep_root}")

    paths = [
        path
        for path in sorted(sweep_root.glob("*.json"))
        if not path.name.endswith((".assets.json", ".predictions.json"))
    ]
    if not paths:
        raise FileNotFoundError(f"No sweep metric JSON files found under: {sweep_root}")

    completed: list[dict[str, Any]] = []
    seen_variant_ids: set[str] = set()
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        if data.get("status") != "ok":
            continue
        if data.get("tree") != EXPECTED_TREE:
            continue
        if data.get("category") != EXPECTED_CATEGORY:
            continue

        variant_id = str(data.get("variant_id", ""))
        if not variant_id:
            raise ValueError(f"Completed metric JSON has no variant_id: {path}")
        if variant_id in seen_variant_ids:
            raise ValueError(f"Duplicate completed variant_id: {variant_id}")

        method = str(data.get("method", ""))
        codebooks = int(_finite_float(data.get("codebooks"), f"{path}: codebooks"))
        codebook_size = int(
            _finite_float(data.get("codebook_size"), f"{path}: codebook_size")
        )
        expected_variant_id = (
            f"nextitem__{EXPECTED_CATEGORY}__{method}__"
            f"{codebooks}cb__{codebook_size}"
        )
        if variant_id != expected_variant_id:
            raise ValueError(
                f"variant_id/configuration mismatch in {path}: expected "
                f"{expected_variant_id!r}, got {variant_id!r}"
            )

        metrics = data.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Completed metric JSON has no metrics object: {path}")

        # Validate every field used downstream now, close to its provenance.
        _finite_float(metrics.get("HR@10"), f"{path}: HR@10")
        _finite_float(metrics.get("NDCG@10"), f"{path}: NDCG@10")
        for key in (
            "stopped_at_step",
            "stopped_at_epoch",
            "train_seconds",
            "eval_seconds",
        ):
            _finite_float(data.get(key), f"{path}: {key}")

        completed.append(data)
        seen_variant_ids.add(variant_id)

    return completed, len(paths)


def _config_value(run: WandbRun, key: str) -> Any:
    value = run.config.get(key)
    return "" if value is None else value


def _iso_utc(timestamp: float) -> str:
    if timestamp <= 0:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _evaluation_observations(
    sweep_root: Path, variant_id: str
) -> tuple[int, int]:
    """Validate retained predictions and recover the logged beam count."""

    predictions_path = sweep_root / f"{variant_id}.predictions.json"
    eval_log_path = sweep_root / f"{variant_id}.eval.log"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Evaluation predictions not found: {predictions_path}")
    if not eval_log_path.is_file():
        raise FileNotFoundError(f"Evaluation log not found: {eval_log_path}")

    try:
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid predictions JSON in {predictions_path}: {exc}") from exc
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"Expected a non-empty prediction list in {predictions_path}")

    header = eval_log_path.read_text(encoding="utf-8", errors="replace")[:4096]
    match = re.search(r"^eval:.*\bbeams=(\d+)\s*$", header, re.MULTILINE)
    if match is None:
        raise ValueError(f"Could not recover beam count from {eval_log_path}")
    return len(predictions), int(match.group(1))


def build_resource_rows(
    sweep_root: Path, wandb_root: Path
) -> tuple[list[dict[str, Any]], BuildReport]:
    sweep_rows, sweep_file_count = load_completed_sweep_metrics(sweep_root)
    wandb_by_name, wandb_file_count, completed_wandb_count, duplicate_count = (
        load_completed_wandb_runs(wandb_root)
    )

    rows: list[dict[str, Any]] = []
    unpaired = 0
    for sweep in sweep_rows:
        variant_id = sweep["variant_id"]
        run = wandb_by_name.get(variant_id)
        if run is None:
            unpaired += 1
            continue

        metrics = sweep["metrics"]
        total_flops = _finite_float(
            run.terminal["total_flos"], f"{variant_id}: W&B total_flos"
        )
        train_runtime = _finite_float(
            run.terminal["train_runtime"], f"{variant_id}: W&B train_runtime"
        )
        wandb_step = int(
            _finite_float(run.terminal["_step"], f"{variant_id}: W&B _step")
        )
        assert run.gpu_count is not None
        if train_runtime <= 0 or total_flops <= 0:
            unpaired += 1
            continue

        evaluation_examples, evaluation_num_beams = _evaluation_observations(
            sweep_root, variant_id
        )
        eval_seconds = _finite_float(
            sweep["eval_seconds"], f"{variant_id}: eval_seconds"
        )
        if eval_seconds <= 0:
            raise ValueError(f"{variant_id}: eval_seconds must be positive")

        per_device_batch = _optional_number(
            run.config.get("per_device_train_batch_size")
        )
        accumulation = _optional_number(run.config.get("gradient_accumulation_steps"))
        effective_batch = None
        if per_device_batch is not None and accumulation is not None:
            effective_batch = int(per_device_batch * accumulation * run.gpu_count)

        telemetry = run.telemetry.summarize(run.gpu_count)
        row: dict[str, Any] = {
            "variant_id": variant_id,
            "tree": sweep["tree"],
            "category": sweep["category"],
            "method": sweep["method"],
            "codebooks": int(sweep["codebooks"]),
            "codebook_size": int(sweep["codebook_size"]),
            "component_tokens": int(sweep["component_tokens"]),
            "items": int(sweep["items"]),
            "max_steps": int(sweep["max_steps"]),
            "hr_at_10": _finite_float(metrics["HR@10"], f"{variant_id}: HR@10"),
            "ndcg_at_10": _finite_float(
                metrics["NDCG@10"], f"{variant_id}: NDCG@10"
            ),
            "stopped_at_step": int(sweep["stopped_at_step"]),
            "stopped_at_epoch": _finite_float(
                sweep["stopped_at_epoch"], f"{variant_id}: stopped_at_epoch"
            ),
            "train_wall_seconds": _finite_float(
                sweep["train_seconds"], f"{variant_id}: train_seconds"
            ),
            "eval_seconds": eval_seconds,
            "evaluation_examples": evaluation_examples,
            "evaluation_num_beams": evaluation_num_beams,
            "evaluation_seconds_per_example": eval_seconds / evaluation_examples,
            "evaluation_allocated_gpu_hours": (
                eval_seconds * run.gpu_count / 3600
            ),
            "inference_flops": None,
            "inference_flops_status": (
                "not measured; retained evaluation artifacts record end-to-end "
                "wall time but no profiler or hardware-counter FLOPs"
            ),
            "hf_total_flops": total_flops,
            "wandb_train_runtime_seconds": train_runtime,
            "wandb_step": wandb_step,
            "wandb_train_global_step": run.terminal.get("train/global_step", ""),
            "wandb_train_epoch": run.terminal.get("train/epoch", ""),
            "model_parameters": _config_value(run, "model/num_parameters"),
            "base_model": _config_value(run, "_name_or_path"),
            "per_device_train_batch_size": _config_value(
                run, "per_device_train_batch_size"
            ),
            "gradient_accumulation_steps": _config_value(
                run, "gradient_accumulation_steps"
            ),
            "effective_global_batch_size": effective_batch,
            "learning_rate": _config_value(run, "learning_rate"),
            "seed": _config_value(run, "seed"),
            "gpu_model": run.gpu_model,
            "gpu_count": run.gpu_count,
            "gpu_memory_gib_per_gpu": (
                run.gpu_memory_bytes_per_gpu / 1024**3
                if run.gpu_memory_bytes_per_gpu is not None
                else None
            ),
            "cuda_version": run.cuda_version,
            "gpu_hours": train_runtime * run.gpu_count / 3600,
            "hf_estimated_achieved_tflops_per_gpu": (
                total_flops / train_runtime / run.gpu_count / 1e12
            ),
            **telemetry,
            "wandb_run_id": run.run_id,
            "wandb_run_started_at_utc": _iso_utc(run.started_at),
            "wandb_run_path": str(run.path.resolve()),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            METHOD_ORDER.get(str(row["method"]), len(METHOD_ORDER)),
            int(row["codebooks"]),
            int(row["codebook_size"]),
            str(row["variant_id"]),
        )
    )
    report = BuildReport(
        sweep_json_files=sweep_file_count,
        completed_sweep_rows=len(sweep_rows),
        wandb_files=wandb_file_count,
        completed_wandb_runs=completed_wandb_count,
        incomplete_wandb_runs=wandb_file_count - completed_wandb_count,
        duplicate_completed_display_names=duplicate_count,
        unpaired_sweep_rows=unpaired,
    )
    return rows, report


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge completed Industrial next-item sweep metrics with read-only "
            "offline W&B FLOP, runtime, hardware, and GPU telemetry records."
        )
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help=f"directory containing sweep metric JSON files (default: {DEFAULT_SWEEP_ROOT})",
    )
    parser.add_argument(
        "--wandb-root",
        type=Path,
        default=DEFAULT_WANDB_ROOT,
        help=f"directory containing offline-run-* folders (default: {DEFAULT_WANDB_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows, report = build_resource_rows(args.sweep_root, args.wandb_root)
    except (DependencyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("error: no completed, exactly matched resource runs found", file=sys.stderr)
        return 2

    write_csv(rows, args.output)
    print(
        f"Wrote {len(rows)} rows to {args.output} "
        f"({report.completed_sweep_rows} completed sweep JSONs; "
        f"{report.completed_wandb_runs}/{report.wandb_files} completed W&B logs; "
        f"{report.unpaired_sweep_rows} unpaired sweep rows)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
