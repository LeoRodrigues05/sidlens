"""The final-run (v3) SID tables, their ZCR twins and the quantizers behind them.

Published by the OneDiffRec handoff as the Hub dataset
`WFRaain/onediffrec-sid-codebooks` and downloaded, pinned to one revision, into
`$SIDLENS_WORK/external/sid_codebooks/`. They are NOT the frozen tables: the
models of the final run are trained on these, the archived models on those.
Never mix the two in one analysis without saying which is which.

Layout (per catalog, `industrial` or `office`):

    tokenizers/tables.json                     build settings + one record per table
    tokenizers/<name>/<table>.json             native table, item index -> ["<a_i>", ...]
    tokenizers/<name>/done.json                sha256 of the table, collisions
    tokenizers/<name>/geometry/                RQ-VAE and MQ only: last-level input
                                               vectors + last codebook (+ MQ model)
    tokenizers/<name>/checkpoints/*/best_collision_model.pth   RQ-VAE weights
    zcr/<name>/                                ZCR twin: index.json, native and ZCR
                                               codes, geometry; RQ-KMeans bundles add
                                               all codebooks (codebooks.npy)

Traps this module guards against:

* Item keys. Tables are keyed by the item index as a string. That index is the
  frozen `item2id` index and the embedding row; `load_table` turns it into the
  ASIN-keyed `SidTable` every analysis expects, in item-index order, and refuses
  a table whose keys are not exactly 0..N-1.
* Silent substitution. Every table is checked against the sha256 its build
  recorded (`done.json` for native tables, `manifest.json` for ZCR twins), and
  a ZCR twin must name the same native table by hash.
* Pickles from a download. The `.pth` files are loaded with an unpickler that
  admits only the four globals they contain (checked with pickletools on
  2026-10-20); anything else raises instead of running.
* RQ-KMeans has no `geometry/` folder: its residuals and codebooks live only in
  the ZCR bundle, which every RQ-KMeans table has. `geometry()` reads the right
  place for each family.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import re
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.data.sids import SidTable, SidVariant, load_item2id
from sidlens.provenance.hashing import sha256_file

REPO_ID = "WFRaain/onediffrec-sid-codebooks"
REVISION = "9733eda7cf4781063cf655597434d4b601de79b5"
ROOT = paths.EXTERNAL / "sid_codebooks" / "WFRaain-onediffrec-sid-codebooks-9733eda"

CATALOGS = {"industrial": "Industrial_and_Scientific", "office": "Office"}
RE_TOKEN = re.compile(r"<([a-z])_(\d+)>")


def category(catalog: str) -> str:
    """Frozen category name ('Industrial_and_Scientific' / 'Office')."""
    if catalog not in CATALOGS:
        raise KeyError(f"unknown catalog {catalog!r}; have {sorted(CATALOGS)}")
    return CATALOGS[catalog]


@lru_cache(maxsize=4)
def tables_json(catalog: str) -> dict:
    doc = json.loads((ROOT / catalog / "tokenizers" / "tables.json").read_text())
    if not doc.get("complete"):
        raise ValueError(f"{catalog}: tables.json says the build is not complete")
    return doc


def names(catalog: str) -> list[str]:
    """The 27 native table names, sorted like `sids.available`."""
    out = sorted(tables_json(catalog)["tables"],
                 key=lambda n: (SidVariant.parse(n).quantizer, SidVariant.parse(n).n_codebook,
                                SidVariant.parse(n).codebook_size))
    if len(out) != 27:
        raise ValueError(f"{catalog}: expected 27 tables, found {len(out)}")
    return out


def zcr_names(catalog: str) -> list[str]:
    """Tables that have a ZCR twin (a `zcr/<name>/` bundle)."""
    have = {p.name for p in (ROOT / catalog / "zcr").iterdir() if p.is_dir()}
    return [n for n in names(catalog) if n in have]


@lru_cache(maxsize=4)
def _id2asin(catalog: str) -> list[str]:
    item2id = load_item2id(category(catalog))
    n = len(item2id)
    if sorted(item2id.values()) != list(range(n)):
        raise ValueError(f"{catalog}: item2id is not a bijection onto 0..{n - 1}")
    out = [""] * n
    for asin, i in item2id.items():
        out[i] = asin
    return out


def _codes_from_index(raw: dict, source: Path, n_items: int) -> np.ndarray:
    ids = sorted(int(k) for k in raw)
    if ids != list(range(n_items)):
        raise ValueError(f"{source}: keys are not exactly 0..{n_items - 1}")
    rows = []
    for i in ids:
        toks = raw[str(i)]
        letters = [RE_TOKEN.match(t).group(1) for t in toks]
        if letters != [chr(ord("a") + d) for d in range(len(toks))]:
            raise ValueError(f"{source}: item {i} has digit letters {letters}")
        rows.append([int(RE_TOKEN.match(t).group(2)) for t in toks])
    return np.asarray(rows, dtype=np.int64)


def _native_path(catalog: str, name: str) -> tuple[Path, str]:
    rec = tables_json(catalog)["tables"][name]
    path = ROOT / catalog / "tokenizers" / name / rec["table"]
    return path, rec["sha256"]


def codes(catalog: str, name: str, kind: str = "native") -> np.ndarray:
    """(N, D) codes in item-index order, hash-checked. kind: 'native' | 'zcr'."""
    n = len(_id2asin(catalog))
    if kind == "native":
        path, want = _native_path(catalog, name)
        done = json.loads((path.parent / "done.json").read_text())
        if done["sha256"] != want:
            raise ValueError(f"{name}: done.json and tables.json disagree on the table hash")
    elif kind == "zcr":
        bundle = ROOT / catalog / "zcr" / name
        man = json.loads((bundle / "manifest.json").read_text())
        path, want = bundle / man["files"]["index"]["path"], man["files"]["index"]["sha256"]
        native_sha = _native_path(catalog, name)[1]
        if man["files"]["native_index"]["sha256"] != native_sha:
            raise ValueError(f"{name}: ZCR bundle was built from a different native table")
    else:
        raise ValueError(f"kind must be 'native' or 'zcr', got {kind!r}")
    got = sha256_file(path)
    if got != want:
        raise ValueError(f"{path}: sha256 {got} != recorded {want}")
    c = _codes_from_index(json.loads(path.read_text()), path, n)
    v = SidVariant.parse(name)
    if c.shape[1] != v.n_codebook or c.min() < 0 or c.max() >= v.codebook_size:
        raise ValueError(f"{path}: shape {c.shape} / range [{c.min()},{c.max()}] "
                         f"does not fit {name}")
    return c


def load_table(catalog: str, name: str, kind: str = "native") -> SidTable:
    """ASIN-keyed `SidTable`, keys in item-index order (= embedding rows)."""
    c = codes(catalog, name, kind)
    asins = _id2asin(catalog)
    path = (_native_path(catalog, name)[0] if kind == "native"
            else ROOT / catalog / "zcr" / name / "index.json")
    t = SidTable(SidVariant.parse(name), {a: tuple(int(x) for x in row)
                                          for a, row in zip(asins, c)}, path)
    if t.keys != asins:
        raise AssertionError("SidTable reordered its keys")
    return t


# ------------------------------------------------------------- geometry --
def geometry(catalog: str, name: str) -> dict:
    """Last-level geometry of a native table.

    `vectors` (N, d) is what the last codebook quantizes and `centroids` (K, d)
    is that codebook. Spaces differ by family, so distances are comparable only
    within a table:
      RQ-KMeans  residual in the 2,560-d embedding (from the ZCR bundle)
      RQ-VAE     residual in the 512-d encoder latent
      MQ         the last digit's encoder output (512-d), not a residual
    """
    v = SidVariant.parse(name)
    if v.quantizer == "rqkmeans":
        d = ROOT / catalog / "zcr" / name
        man = json.loads((d / "manifest.json").read_text())
        space = man["geometry_space"]
        files = {k: (d / man["files"][k]["path"], man["files"][k]["sha256"])
                 for k in ("last_stage_vectors", "final_centroids", "item_ids", "codebooks")}
    else:
        d = ROOT / catalog / "tokenizers" / name / "geometry"
        space = json.loads((d.parent / "done.json").read_text())["geometry"]["space"]
        files = {k: (d / f"{k}.{'json' if k == 'item_ids' else 'npy'}", None)
                 for k in ("last_stage_vectors", "final_centroids", "item_ids")}
    for k, (p, sha) in files.items():
        if sha is not None and sha256_file(p) != sha:
            raise ValueError(f"{p}: sha256 does not match its manifest")
    ids = json.loads(files["item_ids"][0].read_text())
    if [int(i) for i in ids] != list(range(len(_id2asin(catalog)))):
        raise ValueError(f"{name}: geometry item_ids are not 0..N-1 in order")
    out = {"space": space,
           "vectors": np.load(files["last_stage_vectors"][0], allow_pickle=False),
           "centroids": np.load(files["final_centroids"][0], allow_pickle=False)}
    if "codebooks" in files:
        out["codebooks"] = np.load(files["codebooks"][0], allow_pickle=False)
    if out["centroids"].shape[0] != v.codebook_size:
        raise ValueError(f"{name}: {out['centroids'].shape[0]} centroids for width {v.codebook_size}")
    return out


def zcr_bundle(catalog: str, name: str) -> dict:
    d = ROOT / catalog / "zcr" / name
    return {"manifest": json.loads((d / "manifest.json").read_text()),
            "stats": json.loads((d / "stats.json").read_text()),
            "native_codes": np.load(d / "native_codes.npy", allow_pickle=False),
            "zcr_codes": np.load(d / "zcr_codes.npy", allow_pickle=False)}


# ------------------------------------------------------------ weights --
# Exactly the globals the published .pth pickles reference (pickletools scan of
# the RQ-VAE and MQ checkpoints, 2026-10-20). A torch-written checkpoint also
# needs nothing else; a file that asks for more is not one of these.
_ALLOWED_GLOBALS = {("argparse", "Namespace"), ("collections", "OrderedDict"),
                    ("torch._utils", "_rebuild_tensor_v2"), ("torch", "FloatStorage"),
                    ("torch", "LongStorage")}


class _AllowlistUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")
        return super().find_class(module, name)


_restricted = types.ModuleType("sidlens_restricted_pickle")
_restricted.Unpickler = _AllowlistUnpickler
_restricted.load = pickle.load
_restricted.__name__ = "sidlens_restricted_pickle"


def safe_torch_load(path: Path) -> dict:
    """torch.load with an allowlist unpickler.

    torch's own `weights_only=True` rejects the RQ-VAE checkpoints (protocol-4
    FRAME opcode, `argparse.Namespace`), and `weights_only=False` would run any
    global the file names. This admits only `_ALLOWED_GLOBALS`.
    """
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_restricted)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError(f"{path}: not a {{state_dict, ...}} checkpoint")
    args = ck.get("args")
    if isinstance(args, argparse.Namespace):
        ck["args"] = vars(args)
    if not isinstance(ck["state_dict"], (dict, collections.OrderedDict)):
        raise ValueError(f"{path}: state_dict is {type(ck['state_dict'])}")
    return ck


def rqvae_checkpoint(catalog: str, name: str) -> tuple[Path, dict]:
    if SidVariant.parse(name).quantizer != "rqvae":
        raise ValueError(f"{name} is not an RQ-VAE table")
    done = json.loads((ROOT / catalog / "tokenizers" / name / "done.json").read_text())
    path = ROOT / catalog / "tokenizers" / name / done["geometry"]["checkpoint"]
    return path, safe_torch_load(path)


def mq_checkpoint(catalog: str, name: str) -> tuple[Path, dict]:
    if SidVariant.parse(name).quantizer != "MQ":
        raise ValueError(f"{name} is not an MQ table")
    path = ROOT / catalog / "tokenizers" / name / "geometry" / "mq_model.pth"
    return path, safe_torch_load(path)


def _linear_indices(sd: dict, prefix: str) -> list[int]:
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    return sorted(int(m.group(1)) for k in sd if (m := pat.match(k)) and sd[k].ndim == 2)


def rqvae_encode(sd: dict, x: np.ndarray) -> np.ndarray:
    """RQ-VAE encoder of `rq/models/layers.MLPLayers` (dropout 0, no BN).

    Linear layers at `encoder.mlp_layers.{1,4,7,...}` with ReLU after every one
    but the last. float32 on the CPU, as the data loader feeds the model.
    """
    import torch
    idx = _linear_indices(sd, "encoder.mlp_layers")
    if any(f"encoder.mlp_layers.{i}.running_mean" in sd for i in range(idx[-1] + 2)):
        raise ValueError("encoder has BatchNorm; rqvae_encode assumes bn=False")
    h = torch.from_numpy(np.array(x, dtype=np.float32))
    with torch.no_grad():
        for j, i in enumerate(idx):
            h = h @ sd[f"encoder.mlp_layers.{i}.weight"].T + sd[f"encoder.mlp_layers.{i}.bias"]
            if j < len(idx) - 1:
                h = torch.relu(h)
    return h.numpy()


def rqvae_codebooks(sd: dict) -> list[np.ndarray]:
    out, l = [], 0
    while f"rq.vq_layers.{l}.embedding.weight" in sd:
        out.append(sd[f"rq.vq_layers.{l}.embedding.weight"].numpy())
        l += 1
    return out


def rqvae_nearest(z: np.ndarray, codebooks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Nearest-code indices at every level and the residual entering each level.

    Mirrors `VectorQuantizer.forward(use_sk=False)`: d = |r|^2 + |e|^2 - 2 r.e in
    float32, argmin, residual -= chosen codeword.
    """
    import torch
    r = torch.from_numpy(np.array(z, dtype=np.float32))
    idx, residuals = [], []
    with torch.no_grad():
        for cb in codebooks:
            e = torch.from_numpy(cb)
            residuals.append(r.numpy().copy())
            d = (r ** 2).sum(1, keepdim=True) + (e ** 2).sum(1, keepdim=True).T - 2 * r @ e.T
            a = d.argmin(1)
            idx.append(a.numpy())
            r = r - e[a]
    return np.stack(idx, 1).astype(np.int64), residuals


def mq_encode(sd: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`MQ.encode` in eval mode: every digit's code and the last digit's latent.

    Each encoder is Linear, BatchNorm (running statistics), Dropout (off), ReLU,
    Linear, BatchNorm, Dropout, ReLU, Linear, applied to x + pos.
    """
    import torch
    with torch.no_grad():
        h0 = torch.from_numpy(np.array(x, dtype=np.float32)) + sd["pos.weight"]
        n_m = len([k for k in sd if re.match(r"^codebooks\.\d+\.weight$", k)])
        codes_, last = [], None
        for m in range(n_m):
            p = f"encoders.{m}"
            h = h0
            for lin, bn in ((0, 1), (4, 5), (8, None)):
                h = h @ sd[f"{p}.{lin}.weight"].T + sd[f"{p}.{lin}.bias"]
                if bn is not None:
                    h = ((h - sd[f"{p}.{bn}.running_mean"])
                         / torch.sqrt(sd[f"{p}.{bn}.running_var"] + 1e-5)
                         * sd[f"{p}.{bn}.weight"] + sd[f"{p}.{bn}.bias"])
                    h = torch.relu(h)
            codes_.append(torch.cdist(h, sd[f"codebooks.{m}.weight"]).argmin(1).numpy())
            last = h.numpy()
    return np.stack(codes_, 1).astype(np.int64), last


def mq_init_audit(sd: dict, codebook_size: int) -> list[dict]:
    """Are the MQ encoders and `pos` still at their initialisation?

    nn.Linear's default init draws weight and bias from U(-1/sqrt(fan_in),
    1/sqrt(fan_in)); BatchNorm starts at weight 1, bias 0; `pos` is drawn from
    U(-1/K, 1/K). A parameter that trained would leave these bounds or, for a
    BatchNorm affine, move off 1/0. Within bounds is necessary, not sufficient:
    the std is reported beside the uniform's bound/sqrt(3).
    """
    rows = []
    for k, t in sd.items():
        if not k.startswith(("encoders.", "pos.")) or "running" in k or "num_batches" in k:
            continue
        a = t.detach().double().numpy()
        if k == "pos.weight":
            bound, kind = 1.0 / codebook_size, "uniform"
        elif a.ndim == 2:
            bound, kind = 1.0 / np.sqrt(a.shape[1]), "uniform"
        else:
            lin = k.rsplit(".", 1)[0]
            w = sd.get(f"{lin}.weight")
            if w is not None and w.ndim == 2:          # Linear bias
                bound, kind = 1.0 / np.sqrt(w.shape[1]), "uniform"
            else:                                      # BatchNorm affine
                bound = None
                kind = "one" if k.endswith("weight") else "zero"
        row = {"param": k, "n": int(a.size), "kind": kind, "max_abs": float(np.abs(a).max()),
               "std": float(a.std())}
        if kind == "uniform":
            row.update(bound=bound, within_bound=bool(np.abs(a).max() <= bound * (1 + 1e-6)),
                       std_over_uniform_std=float(a.std() / (bound / np.sqrt(3))))
        else:
            target = 1.0 if kind == "one" else 0.0
            row.update(bound=None, within_bound=bool(np.abs(a - target).max() == 0.0),
                       std_over_uniform_std=None)
        rows.append(row)
    return rows
