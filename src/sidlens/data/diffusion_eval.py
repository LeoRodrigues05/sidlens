"""Read-only reconstruction of the frozen DiffGRM next-item test cohort.

The upstream dataset emits one full sequence per user, in JSON insertion order.
Its tokenizer takes the final item as the target, keeps the most recent H
history items, and pads on the RIGHT with -1 (mask=True denotes real history).
Histories and labels are raw codebook IDs, not offset vocabulary token IDs.

Loading the upstream tokenizer would write mapping caches. This module instead
reconstructs its test semantics from the frozen inputs, retaining both raw and
canonical identities and a lossless, one-to-many SID catalogue. Unknown items
are errors; the upstream silent unknown-to-padding fallback is never needed on
the complete frozen Industrial catalogue.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.data.sids import SidTable, SidVariant
from sidlens.provenance.hashing import sha256_file
from sidlens.registry.diffusion import resolve_entry_paths

CATEGORY = "Industrial_and_Scientific"


@dataclass(frozen=True)
class DiffusionEvalCohort:
    users: tuple[str, ...]
    user_ids: np.ndarray
    histories: np.ndarray
    history_mask: np.ndarray
    history_lengths: np.ndarray
    sequence_lengths: np.ndarray
    target_sids: np.ndarray
    target_item_ids: np.ndarray
    target_asins: tuple[str, ...]
    sid_buckets: dict[tuple[int, ...], tuple[int, ...]]
    item_to_sid: dict[int, tuple[int, ...]]
    input_sha256: dict[str, str]
    cohort_sha256: str
    sid_lineage: dict

    def __len__(self) -> int:
        return len(self.users)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def _read_id_map(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1].isdecimal():
            raise ValueError(f"{path}:{line_number}: malformed ID-map row")
        raw, number = fields[0], int(fields[1])
        if raw in result or number in seen:
            raise ValueError(f"{path}:{line_number}: duplicate ID-map identity")
        result[raw] = number
        seen.add(number)
    if seen != set(range(len(result))):
        raise ValueError(f"{path}: canonical IDs must be contiguous and zero-based")
    return result


def _check_upstream_ids(mapping: dict, kind: str, canonical: dict[str, int]):
    forward, inverse = mapping.get(f"{kind}2id"), mapping.get(f"id2{kind}")
    expected = {"[PAD]": 0, **{raw: number + 1 for raw, number in canonical.items()}}
    if (not isinstance(forward, dict) or forward != expected
            or any(type(value) is not int for value in forward.values())):
        raise ValueError(f"DiffGRM {kind} IDs must equal canonical IDs + 1, plus PAD=0")
    if not isinstance(inverse, list) or len(inverse) != len(expected):
        raise ValueError(f"DiffGRM id2{kind} has inconsistent cardinality")
    if any(inverse[number] != raw for raw, number in expected.items()):
        raise ValueError(f"DiffGRM id2{kind} does not invert {kind}2id")


def load_eval_cohort(
    entry: dict,
    *,
    frozen_root: Path | None = None,
    expected_users: int = 6297,
    expected_items: int = 3105,
) -> DiffusionEvalCohort:
    """Load every held-out final item, without downloading or writing caches.

    ``entry`` is a diffusion-registry dictionary. The registry's exact SID file
    and SHA-256 are mandatory: repaired tables are not substituted by name, and
    quarantined pre-repair AR ``.mispacked`` indices are not accepted. The
    frozen-substrate gate remains the caller's responsibility; this loader also
    hashes every consumed input and checks SID contents against the registry.

    Canonical item/user IDs are the project's zero-based OneDiffRec IDs; the
    upstream DiffGRM IDs add one for its PAD row. ``sid_buckets`` retains ALL
    canonical items for each raw-code SID, including collisions.
    """
    root = Path(frozen_root) if frozen_root is not None else paths.FROZEN
    resolved = resolve_entry_paths(entry, frozen_root=root)
    if entry.get("task") != "next1" or entry.get("status") != "trained":
        raise ValueError("evaluation cohort requires a trained next1 checkpoint")
    config = entry["config"]
    variant = SidVariant.parse(entry["sem_ids_name"])
    d, width, history_limit = (
        config["n_digit"], config["codebook_size"], config["max_history_len"])
    if (type(d) is not int or type(width) is not int
            or type(history_limit) is not int or history_limit < 1
            or config["n_target_items"] != 1 or config["n_target_digits"] != d
            or (d, width) != (variant.n_codebook, variant.codebook_size)
            or (entry["quantizer"], entry["n_codebook"], entry["codebook_size"])
            != (variant.quantizer, d, width)):
        raise ValueError("checkpoint config and SID variant disagree")
    sem_path = Path(resolved["sem_ids_path"])
    required_sem_path = (
        root / "sids" / "sem_ids" / "diffgrm" / f"{variant.name}.sem_ids")
    if sem_path != required_sem_path.resolve():
        raise ValueError("next1 cohort requires the registry's canonical diffgrm .sem_ids path")

    data_root = root / "data"
    seq_path = data_root / "sequences" / "all_item_seqs.json"
    mapping_path = data_root / "sequences" / "id_mapping.json"
    user_path = data_root / "id_maps" / f"{CATEGORY}.user2id"
    item_path = data_root / "id_maps" / f"{CATEGORY}.item2id"
    input_paths = (seq_path, mapping_path, user_path, item_path, sem_path)
    hashes = {str(path.resolve()): sha256_file(path) for path in input_paths}
    if hashes[str(sem_path)] != entry["sem_ids_sha256"]:
        raise ValueError("semantic-ID SHA-256 differs from the checkpoint registry")

    sequences, mapping = _read_json(seq_path), _read_json(mapping_path)
    user_map, item_map = _read_id_map(user_path), _read_id_map(item_path)
    if not isinstance(sequences, dict) or not isinstance(mapping, dict):
        raise ValueError("sequences and id_mapping must be JSON objects")
    if len(sequences) != expected_users or len(user_map) != expected_users:
        raise ValueError(f"expected the complete {expected_users}-user test cohort")
    if len(item_map) != expected_items:
        raise ValueError(f"expected the complete {expected_items}-item catalogue")
    if set(sequences) != set(user_map):
        raise ValueError("sequence users differ from canonical user identities")
    _check_upstream_ids(mapping, "user", user_map)
    _check_upstream_ids(mapping, "item", item_map)

    raw_sids = _read_json(sem_path)
    if not isinstance(raw_sids, dict) or set(raw_sids) != set(item_map):
        raise ValueError("SID catalogue must cover exactly the canonical item identities")
    if any(not isinstance(code, list) or any(type(c) is not int for c in code)
           for code in raw_sids.values()):
        raise ValueError("SIDs must be lists of integer codebook IDs")
    table = SidTable(variant, {k: tuple(v) for k, v in raw_sids.items()}, sem_path)
    buckets = {
        sid: tuple(sorted(item_map[asin] for asin in asins))
        for sid, asins in table.cb2items.items()
    }
    item_to_sid = {item_map[asin]: sid for asin, sid in table.asin2codes.items()}

    users = tuple(sequences)
    histories = np.full((len(users), history_limit, d), -1, dtype=np.int64)
    history_mask = np.zeros((len(users), history_limit), dtype=np.bool_)
    history_lengths = np.empty(len(users), dtype=np.int64)
    sequence_lengths = np.empty(len(users), dtype=np.int64)
    target_sids = np.empty((len(users), d), dtype=np.int64)
    target_item_ids = np.empty(len(users), dtype=np.int64)
    user_ids = np.array([user_map[user] for user in users], dtype=np.int64)
    targets = []
    identity_hash = hashlib.sha256()
    for row, user in enumerate(users):
        seq = sequences[user]
        if (not isinstance(seq, list) or len(seq) < 2
                or any(not isinstance(asin, str) or asin not in item_map for asin in seq)):
            raise ValueError(f"user {user!r}: invalid sequence or unknown item; no PAD fallback")
        target, history = seq[-1], seq[:-1][-history_limit:]
        length = len(history)
        histories[row, :length] = [table.asin2codes[asin] for asin in history]
        history_mask[row, :length] = True
        history_lengths[row], sequence_lengths[row] = length, len(seq)
        target_sids[row] = table.asin2codes[target]
        target_item_ids[row] = item_map[target]
        targets.append(target)
        # Independent of the SID variant, but sensitive to user/row order,
        # history truncation, raw sequence contents, and canonical identity.
        identity_hash.update(json.dumps(
            [user, user_map[user], history_limit, seq, [item_map[a] for a in seq]],
            separators=(",", ":"), ensure_ascii=True).encode() + b"\n")

    return DiffusionEvalCohort(
        users=users, user_ids=user_ids, histories=histories,
        history_mask=history_mask, history_lengths=history_lengths,
        sequence_lengths=sequence_lengths, target_sids=target_sids,
        target_item_ids=target_item_ids, target_asins=tuple(targets),
        sid_buckets=buckets, item_to_sid=item_to_sid, input_sha256=hashes,
        cohort_sha256=identity_hash.hexdigest(),
        sid_lineage={
            "sem_ids_name": variant.name, "sem_ids_path": str(sem_path),
            "sem_ids_sha256": entry["sem_ids_sha256"],
            "checkpoint_trained_at": entry.get("trained_at"),
            "source": "registry-bound frozen next1 .sem_ids; no replacement or bit unpacking",
            "quarantined_mispacked_indices_used": False,
            "n_catalogue_items": len(item_map), "n_unique_sids": len(buckets),
        },
    )
