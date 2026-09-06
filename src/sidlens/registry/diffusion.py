"""Registry of the trained diffusion checkpoints.

A checkpoint on disk is a bare state_dict. Turning one back into a runnable
model requires config that lives in three different places:

    shapes       <- the run .log module repr        (parse_logs)
    training     <- the run .txt transcript         (parse_logs)
    n_head       <- the vendored sbatch             (here)

`DiffGRMConfig` is frozen and has NO DEFAULTS. Constructing one requires naming
every field. That is deliberate: `n_head` cannot be recovered from weights or
logs, and a wrong value loads without error and silently computes different
attention. A default would let that happen quietly.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from sidlens import paths
from sidlens.provenance import hashing
from sidlens.registry.parse_logs import parse_run, ParsedRun

TASKS = {"next1": "diffgrm", "next2": "diffgrm_new"}
SBATCH_FOR_TASK = {
    "next1": "scripts/run_diffgrm.sbatch",
    "next2": "scripts/run_diffgrm_2item.sbatch",
}
RE_VARIANT = re.compile(r"^(?P<q>rqvae|rqkmeans|MQ)_(?P<cb>\d+)codebook_(?P<size>\d+)$")
RE_STAMP = re.compile(r"AmazonReviews2014_(\w{3})-(\d{2})-(\d{4})_(\d{2})-(\d{2})-(\d{2})")

# These paths are recorded as absolute paths because they are provenance from
# the machine on which the registry was built.  At runtime they all belong to
# the frozen snapshot, though, and must follow SIDLENS_WORK when that snapshot
# is moved to another cluster.  Keeping the expected top-level section beside
# each field prevents a malformed entry from being silently redirected to an
# unrelated file under the new root.
_FROZEN_PATH_FIELDS = {
    "ckpt_path": "ckpt",
    "sem_ids_path": "sids",
    "log_path": "ckpt",
    "transcript_path": "results",
}


@dataclass(frozen=True)
class DiffGRMConfig:
    """Complete DIFF_GRM construction config. No defaults, by design."""
    vocab_size: int
    n_embd: int
    n_head: int
    n_inner: int
    n_digit: int
    codebook_size: int
    n_target_items: int
    n_target_digits: int
    encoder_n_layer: int
    decoder_n_layer: int
    max_history_len: int
    dropout: float
    layer_norm_eps: float
    share_decoder_output_embedding: bool
    sid_offset: int

    def __post_init__(self):
        expected = 3 + self.n_digit * self.codebook_size
        if self.vocab_size != expected:
            raise ValueError(
                f"vocab_size {self.vocab_size} != 3 + n_digit*K = {expected}")
        if self.n_target_digits != self.n_digit * self.n_target_items:
            raise ValueError(
                f"n_target_digits {self.n_target_digits} != "
                f"n_digit*n_target_items = {self.n_digit * self.n_target_items}")
        if self.n_embd % self.n_head:
            raise ValueError(f"n_embd {self.n_embd} not divisible by n_head {self.n_head}")


def sbatch_defaults(task: str) -> dict:
    """Read config out of the VENDORED sbatch.

    Parsed from the frozen copy rather than hardcoded here, so the value stays
    bound to bytes that `sidlens verify` checks. If vendor drifts, verify goes
    red before anything reads a stale constant.
    """
    path = paths.VENDOR / "onediffrec" / SBATCH_FOR_TASK[task]
    text = path.read_text()
    out: dict[str, str] = {}
    for m in re.finditer(r"--([a-z_]+)=([^\s\\\"]+)", text):
        key, val = m.group(1), m.group(2)
        if val.startswith("$") or "${" in val:
            continue
        out.setdefault(key, val)
    return out


def _coerce(v: str):
    if v in ("true", "false"):
        return v == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@dataclass(frozen=True)
class DiffusionCheckpoint:
    ckpt_id: str
    task: str
    quantizer: str
    n_codebook: int
    codebook_size: int
    config: DiffGRMConfig
    ckpt_path: str
    ckpt_sha256: str
    sem_ids_name: str
    sem_ids_path: str
    sem_ids_sha256: str
    log_path: str
    log_sha256: str
    transcript_path: str | None
    transcript_sha256: str | None
    source_jobid: str
    source_stamp: str
    trained_at: str
    status: str
    recorded_metrics: dict
    best_epoch: int | None
    best_val_score: float | None
    training: dict
    warnings: tuple[str, ...]

    def to_json(self) -> dict:
        d = asdict(self)
        d["config"] = asdict(self.config)
        d["warnings"] = list(self.warnings)
        return d


def _stamp_to_iso(stamp: str) -> str:
    m = RE_STAMP.search(stamp)
    if not m:
        return ""
    mon, day, year, hh, mm, ss = m.groups()
    return datetime.strptime(
        f"{mon} {day} {year} {hh}:{mm}:{ss}", "%b %d %Y %H:%M:%S").isoformat()


def build(task: str, cache: hashing.HashCache | None = None) -> list[DiffusionCheckpoint]:
    cache = cache or hashing.HashCache()
    defaults = {k: _coerce(v) for k, v in sbatch_defaults(task).items()}
    ckpt_root = paths.FROZEN_CKPT / "diffusion" / task
    log_root = paths.FROZEN_CKPT / "diffusion" / f"{task}_logs"
    tx_root = paths.FROZEN_RESULTS / "diffusion_transcripts" / task
    sem_root = paths.FROZEN_SIDS / "sem_ids" / TASKS[task]

    out: list[DiffusionCheckpoint] = []
    for bin_path in sorted(ckpt_root.glob("*/saved/*/pytorch_model.bin")):
        jobid = bin_path.relative_to(ckpt_root).parts[0]
        stamp = bin_path.parent.name
        logs = sorted((log_root / jobid).glob(f"logs/**/{stamp}.log"))
        if not logs:
            raise FileNotFoundError(
                f"checkpoint {jobid}/{stamp} has no matching log -- config is "
                f"unrecoverable, refusing to guess")
        log_path = logs[0]

        parsed: ParsedRun = parse_run(log_path, None)
        if not parsed.sem_ids_path:
            raise ValueError(f"{log_path} names no external sem_ids file")
        sem_name = Path(parsed.sem_ids_path).stem
        m = RE_VARIANT.match(sem_name)
        if not m:
            raise ValueError(f"unparseable variant name {sem_name!r} from {log_path}")

        # The transcript carries the GUIDED banner and the final Test Results,
        # neither of which appears in the .log.
        tx_path = tx_root / f"{sem_name}.txt"
        parsed = parse_run(log_path, tx_path if tx_path.exists() else None)

        sem_path = sem_root / f"{sem_name}.sem_ids"
        if not sem_path.exists():
            raise FileNotFoundError(f"frozen sem_ids missing for {sem_name}: {sem_path}")

        cfg = DiffGRMConfig(
            vocab_size=parsed.vocab_size,
            n_embd=parsed.n_embd,
            # Not in any log or weight: supplied by the vendored, hashed sbatch.
            n_head=defaults["n_head"],
            n_inner=parsed.n_inner,
            n_digit=parsed.n_digit,
            codebook_size=parsed.codebook_size,
            n_target_items=parsed.n_target_items,
            n_target_digits=parsed.n_target_digits,
            encoder_n_layer=parsed.encoder_n_layer,
            decoder_n_layer=parsed.decoder_n_layer,
            max_history_len=parsed.max_history_len,
            dropout=parsed.dropout,
            layer_norm_eps=parsed.layer_norm_eps,
            share_decoder_output_embedding=defaults.get(
                "share_decoder_output_embedding", True),
            sid_offset=3,
        )

        training = {
            "masking_strategy": parsed.masking_strategy or defaults.get("masking_strategy"),
            "guided_steps": parsed.guided_steps,
            "guided_conf_metric": parsed.guided_conf_metric or defaults.get("guided_conf_metric"),
            "guided_select": parsed.guided_select or defaults.get("guided_select"),
            "guided_refresh_each_step": defaults.get("guided_refresh_each_step"),
            "lr": defaults.get("lr"),
            "label_smoothing": defaults.get("label_smoothing"),
            "train_batch_size": defaults.get("train_batch_size"),
            "train_sliding": defaults.get("train_sliding"),
            "min_hist_len": defaults.get("min_hist_len"),
            "eval_start_epoch": defaults.get("eval_start_epoch"),
            "metadata": defaults.get("metadata"),
            "sid_quantizer": defaults.get("sid_quantizer"),
        }

        # A checkpoint with no Test Results is mid-training, not broken. Say so
        # explicitly -- an empty metrics dict read as 0.0 would silently corrupt
        # any comparison, and it is also what blocks the behavioral n_head check.
        status = "trained" if parsed.test_results else "incomplete"

        out.append(DiffusionCheckpoint(
            ckpt_id=f"diff-{task}-{m['q']}-{m['cb']}cb-{m['size']}",
            task=task,
            quantizer=m["q"],
            n_codebook=int(m["cb"]),
            codebook_size=int(m["size"]),
            config=cfg,
            ckpt_path=str(bin_path),
            ckpt_sha256=cache.get(bin_path),
            sem_ids_name=sem_name,
            sem_ids_path=str(sem_path),
            sem_ids_sha256=cache.get(sem_path),
            log_path=str(log_path),
            log_sha256=cache.get(log_path),
            transcript_path=str(tx_path) if tx_path.exists() else None,
            transcript_sha256=cache.get(tx_path) if tx_path.exists() else None,
            source_jobid=jobid,
            source_stamp=stamp,
            trained_at=_stamp_to_iso(stamp),
            status=status,
            recorded_metrics=parsed.test_results,
            best_epoch=parsed.best_epoch,
            best_val_score=parsed.best_val_score,
            training=training,
            warnings=tuple(parsed.warnings),
        ))
    return out


def registry_path() -> Path:
    return paths.MANIFESTS / "registry.diffusion.json"


def build_all() -> dict:
    cache = hashing.HashCache()
    entries: dict[str, dict] = {}
    for task in TASKS:
        for ck in build(task, cache):
            if ck.ckpt_id in entries:
                raise ValueError(f"duplicate ckpt_id {ck.ckpt_id}")
            entries[ck.ckpt_id] = ck.to_json()
    return entries


def save(entries: dict) -> Path:
    p = registry_path()
    p.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return p


def _resolve_frozen_path(value: str, field: str, expected_section: str,
                         frozen_root: Path) -> Path:
    """Map one recorded frozen path onto ``frozen_root``.

    Absolute paths in the tracked registry retain where the artifact was
    originally frozen.  Resolution uses only the suffix beginning at the
    field's expected frozen section (for example ``ckpt/...``), never an
    arbitrary suffix supplied by the manifest.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string")

    recorded = Path(value)
    if recorded.is_absolute():
        parts = recorded.parts
        markers = [
            i for i in range(len(parts) - 1)
            if parts[i] == "frozen" and parts[i + 1] == expected_section
        ]
        if len(markers) != 1:
            raise ValueError(
                f"{field} is not an unambiguous frozen/{expected_section} path: "
                f"{value!r}")
        relative = Path(*parts[markers[0] + 1:])
    else:
        # Also accept a future registry that records paths relative to FROZEN.
        relative = recorded

    if (not relative.parts or relative.parts[0] != expected_section
            or ".." in relative.parts):
        raise ValueError(
            f"{field} must stay below frozen/{expected_section}: {value!r}")

    root = frozen_root.resolve(strict=False)
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{field} escapes the configured frozen root: {value!r}") from exc
    return resolved


def resolve_entry_paths(entry: dict, frozen_root: Path | None = None) -> dict:
    """Return an entry whose artifact paths follow the active SIDLENS_WORK.

    The input is never mutated.  Hashes and all other recorded provenance stay
    byte-for-byte identical; only the four runtime path fields are replaced.
    Pass ``frozen_root`` mainly for tooling/tests, otherwise ``paths.FROZEN`` is
    read at call time so environment-specific path configuration is respected.
    """
    root = Path(frozen_root) if frozen_root is not None else paths.FROZEN
    resolved = deepcopy(entry)
    for field, section in _FROZEN_PATH_FIELDS.items():
        value = resolved.get(field)
        if value is None and field == "transcript_path":
            continue
        resolved[field] = str(_resolve_frozen_path(value, field, section, root))
    return resolved


def load() -> dict:
    """Load the tracked registry exactly as recorded.

    Keeping this API raw preserves its original semantics for provenance and
    audit callers.  Code that opens artifacts should use ``load_runtime``.
    """
    return json.loads(registry_path().read_text())


def load_runtime() -> dict:
    """Load entries with artifact paths rebased for this installation.

    The tracked JSON is deliberately not rewritten: its absolute paths remain
    provenance for the source snapshot while runtime consumers follow the
    active ``SIDLENS_WORK`` root.
    """
    return {
        ckpt_id: resolve_entry_paths(entry)
        for ckpt_id, entry in load().items()
    }
