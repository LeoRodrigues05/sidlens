"""Activation store: one on-disk format for captures made on any cluster.

Captures from this cluster, from the upstream cluster, and conversions of
upstream's training-time activation logs should all land in this format. One
reader then serves probes, SAEs and patching regardless of origin, and the
directory can go to a Hub dataset repo as-is (see `provenance.bundle`).

Layout -- nothing in it needs pickle to load
    <root>/rows.parquet               one row per stored vector (see ROW_KEYS)
    <root>/shards/00000.safetensors   {site: (n, width)}, same rows for every site
    <root>/manifest.json              written LAST by close(): schema, sites,
                                      widths, dtypes, shard sha256s, meta, status

Traps, each with its guard
--------------------------
* An interrupted capture looks like a smaller complete one. Only `close()`
  writes `manifest.json` (status "complete", every shard's sha256), and the
  reader refuses a directory without it. An exception inside `with
  StoreWriter(...)` leaves a FAILED file instead.
* A batch index is not an identity. Rows carry `example_id` and `pos`, plus
  whatever semantic labels the caller supplies (role, item, digit, code, step,
  mask, ...). After filtering, concatenation or re-sharding, a row still says
  which example and token it came from.
* Sites that drift apart. Every `add` supplies every site for the same rows,
  and lengths, widths and dtypes are checked. Otherwise row 5 of layer 12 and
  row 5 of layer 20 could describe different tokens.
* Pickle. `torch.save` and pickled `.npy` files execute code on load, and a
  store pulled from a shared repo must not. The store is safetensors + parquet.
* Hub limits. Shards roll over at `shard_bytes` (default 1 GiB), which keeps
  files well under the Hub's per-file ceiling and the file count low.
* Overwrite. `root` must not exist; `mkdir` fails on a collision, following
  the derived/ convention.

`meta` should name the model and checkpoint (with its sha256), the input
template, the dtype the model ran in, and the capture config. The snapshot id,
git head and host are added automatically.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ROW_KEYS = ("example_id", "pos")          # required on every row
_INTERNAL = ("row", "shard", "shard_row")


def _provenance() -> dict:
    from sidlens import paths
    head = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(paths.REPO), "status", "--porcelain"],
                           capture_output=True, text=True, check=False).stdout
    cur = paths.MANIFESTS / "CURRENT"
    return {"snapshot_id": cur.read_text().strip() if cur.exists() else None,
            "git_head": head, "git_dirty_paths": len(dirty.splitlines()),
            "host": socket.gethostname(),
            "created_utc": datetime.now(timezone.utc).isoformat()}


class StoreWriter:
    def __init__(self, root: str | Path, meta: dict, shard_bytes: int = 1 << 30):
        if "model" not in meta:
            raise ValueError("meta must name the model (and its checkpoint sha256)")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "shards").mkdir()
        self.meta = {**meta, "provenance": _provenance()}
        self.shard_bytes = shard_bytes
        self._rows: list[dict] = []
        self._buf: dict[str, list] = {}
        self._buf_rows = 0
        self._buf_bytes = 0
        self._sites: dict[str, dict] | None = None
        self._columns: tuple[str, ...] | None = None
        self._shards: list[dict] = []
        self._closed = False

    # ---------------------------------------------------------------- write --
    def add(self, rows: list[dict], tensors: dict) -> None:
        """Append `len(rows)` vectors for every site. Tensors are (n, width)."""
        import torch

        if self._closed:
            raise RuntimeError("store already closed")
        n = len(rows)
        if n == 0:
            return
        cols = tuple(rows[0])
        for k in ROW_KEYS:
            if k not in cols:
                raise KeyError(f"row field {k!r} is required on every row")
        if clash := set(cols) & set(_INTERNAL):
            raise KeyError(f"row fields {sorted(clash)} are reserved")
        if any(tuple(r) != cols for r in rows):
            raise ValueError("every row must have the same fields in the same order")
        if self._columns is None:
            self._columns = cols
        elif cols != self._columns:
            raise ValueError(f"row fields changed: {cols} vs {self._columns}")
        spec = {}
        for site, t in tensors.items():
            if not isinstance(t, torch.Tensor) or t.ndim != 2 or t.shape[0] != n:
                raise ValueError(f"site {site!r}: expected a ({n}, width) tensor, got "
                                 f"{getattr(t, 'shape', type(t))}")
            spec[site] = {"width": int(t.shape[1]), "dtype": str(t.dtype).replace("torch.", "")}
        if self._sites is None:
            self._sites = spec
        elif spec != self._sites:
            raise ValueError(f"sites/widths/dtypes changed between adds: {spec} vs {self._sites}")
        for site, t in tensors.items():
            t = t.detach().to("cpu").contiguous()
            self._buf.setdefault(site, []).append(t)
            self._buf_bytes += t.numel() * t.element_size()
        shard = len(self._shards)
        for i, r in enumerate(rows):
            self._rows.append({**r, "row": len(self._rows), "shard": shard,
                               "shard_row": self._buf_rows + i})
        self._buf_rows += n
        if self._buf_bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        import torch
        from safetensors.torch import save_file
        from sidlens.provenance.hashing import sha256_file

        if not self._buf_rows:
            return
        idx = len(self._shards)
        path = self.root / "shards" / f"{idx:05d}.safetensors"
        save_file({s: torch.cat(ts) for s, ts in self._buf.items()}, str(path))
        self._shards.append({"file": path.relative_to(self.root).as_posix(),
                             "n_rows": self._buf_rows, "sha256": sha256_file(path),
                             "bytes": path.stat().st_size})
        self._buf, self._buf_rows, self._buf_bytes = {}, 0, 0

    def close(self) -> Path:
        import pandas as pd

        if self._closed:
            return self.root
        self._flush()
        if not self._rows:
            raise RuntimeError("nothing was added; refusing to write an empty store")
        pd.DataFrame(self._rows).to_parquet(self.root / "rows.parquet", index=False)
        doc = {"schema_version": SCHEMA_VERSION, "status": "complete",
               "n_rows": len(self._rows), "row_fields": list(self._columns),
               "sites": self._sites, "shards": self._shards, "meta": self.meta}
        (self.root / "manifest.json").write_text(json.dumps(doc, indent=1))
        self._closed = True
        return self.root

    def __enter__(self) -> "StoreWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            (self.root / "FAILED").write_text(f"{exc_type.__name__}: {exc}\n")


class StoreReader:
    def __init__(self, root: str | Path, verify: bool = False):
        import pandas as pd

        self.root = Path(root)
        mf = self.root / "manifest.json"
        if not mf.exists():
            why = (self.root / "FAILED").read_text().strip() if (self.root / "FAILED").exists() \
                else "no manifest.json (capture interrupted or still running)"
            raise RuntimeError(f"{self.root} is not a complete store: {why}")
        self.manifest = json.loads(mf.read_text())
        if self.manifest.get("schema_version") != SCHEMA_VERSION \
                or self.manifest.get("status") != "complete":
            raise RuntimeError(f"{self.root}: unsupported or incomplete store manifest")
        if verify:
            from sidlens.provenance.hashing import sha256_file
            for sh in self.manifest["shards"]:
                if sha256_file(self.root / sh["file"]) != sh["sha256"]:
                    raise RuntimeError(f"{sh['file']}: sha256 mismatch")
        self.rows = pd.read_parquet(self.root / "rows.parquet")
        if len(self.rows) != self.manifest["n_rows"]:
            raise RuntimeError("rows.parquet and manifest disagree on the row count")

    @property
    def sites(self) -> list[str]:
        return list(self.manifest["sites"])

    @property
    def meta(self) -> dict:
        return self.manifest["meta"]

    def load(self, site: str, rows=None):
        """(k, width) tensor for global row indices `rows` (default: all), in that order."""
        import numpy as np
        import torch
        from safetensors import safe_open

        if site not in self.manifest["sites"]:
            raise KeyError(f"no site {site!r}; have {self.sites}")
        spec = self.manifest["sites"][site]
        idx = np.arange(len(self.rows)) if rows is None else np.asarray(rows, dtype=np.int64)
        sel_shard = self.rows["shard"].to_numpy()[idx]
        sel_local = self.rows["shard_row"].to_numpy()[idx]
        out = torch.empty((len(idx), spec["width"]), dtype=getattr(torch, spec["dtype"]))
        for shard in np.unique(sel_shard):
            f = self.root / self.manifest["shards"][int(shard)]["file"]
            with safe_open(str(f), framework="pt") as fh:
                t = fh.get_tensor(site)
            where = np.flatnonzero(sel_shard == shard)
            out[torch.as_tensor(where)] = t.index_select(
                0, torch.as_tensor(sel_local[where]))
        return out
