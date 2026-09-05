"""Attribute labels, with a usability gate in front of them.

`meta.py` serves what the frozen substrate holds: title, description, brand, and
an empty `categories`. This module serves what `scripts/build_labels.py`
recovered from the public Amazon-2018 dump -- the category hierarchy, price,
rank, and the co-purchase graph -- joined to the frozen catalogue at 100%.

The gate is the point
---------------------
A label field is not usable just because it exists. Two failure modes recur and
both produce numbers that look like findings:

  too sparse     a field covering 30% of the catalogue conditions every
                 depth-2 statistic on a few dozen items.
  too fine       `rank` has 3008 distinct values over 3105 items. An AMI
                 between a digit and a near-unique field measures item identity
                 and will look impressively high for any quantizer.

So every field carries `min_coverage` and `max_distinct_ratio`, and `usable()`
answers yes or no with a reason. Callers are free to override, but they have to
do it explicitly -- the default path cannot silently probe a near-identity
column and report it as semantics.

What is worth knowing about each field
--------------------------------------
    cat_l1     25 classes, 97% coverage, ~120 items/class. The headline
               attribute target: coarse enough that a digit CAN align with it,
               fine enough that alignment is not trivial.
    cat_l2     181 classes, 94%. The interesting one for later digits.
    cat_l3     372 classes, 78%. Marginal -- gated off by default.
    cat_leaf   657 classes, 97%. The deepest breadcrumb node each item has, so
               its granularity is inconsistent ACROSS items; use l1/l2/l3 for
               anything that compares digits at fixed depth.
    main_cat   19 stores, 100%. Cuts across the breadcrumb rather than nesting
               inside it -- an item can be in the Industrial breadcrumb and sold
               under Tools & Home Improvement. That makes it a useful second,
               non-nested view rather than a coarser cat_l1.
    price_usd  82%, continuous. Deciles are the categorical form.
    rank       98%, continuous, but near-unique -- gated off as a categorical
               and usable only as deciles or as a regression target.
    also_buy   78%. The one signal here that is NOT derived from item text.
               Everything else in this file ultimately traces back to strings a
               text encoder could have read, so a digit aligning with them is
               ambiguous between "encodes meaning" and "encodes the encoder".
               Co-purchase is behavioural, so it breaks that tie.

The frozen text stays authoritative. `title_ext`/`brand_ext` are the dump's own
strings, carried for comparison; `meta.py` remains the source for anything the
embeddings were built from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from sidlens import paths

MISSING = -1


@dataclass(frozen=True)
class LabelField:
    """One probeable column, and the conditions under which it means anything."""
    name: str
    kind: str                     # "categorical" | "continuous" | "graph"
    note: str
    min_coverage: float = 0.60
    max_distinct_ratio: float = 0.25   # distinct values / items covered
    default_on: bool = True

    def check(self, cov: dict) -> tuple[bool, str]:
        n = cov["n_items"]
        if cov["coverage"] < self.min_coverage:
            return False, (f"coverage {cov['coverage']:.1%} < "
                           f"{self.min_coverage:.0%} required")
        if self.kind == "categorical" and cov["non_empty"]:
            ratio = cov["distinct"] / cov["non_empty"]
            if ratio > self.max_distinct_ratio:
                return False, (f"{cov['distinct']} distinct over {cov['non_empty']} "
                               f"labelled items (ratio {ratio:.2f} > "
                               f"{self.max_distinct_ratio:.2f}) -- this is an "
                               f"identity column, not a class column")
        if not self.default_on:
            return False, "off by default; pass force=True to use it"
        _ = n
        return True, "ok"


FIELDS: dict[str, LabelField] = {
    f.name: f for f in (
        LabelField("cat_l1", "categorical",
                   "breadcrumb level 1 -- the primary attribute target"),
        LabelField("cat_l2", "categorical",
                   "breadcrumb level 2 -- the target for middle digits"),
        LabelField("cat_l3", "categorical",
                   "breadcrumb level 3 -- sparse and fine; opt in explicitly",
                   min_coverage=0.75, max_distinct_ratio=0.20, default_on=False),
        LabelField("cat_leaf", "categorical",
                   "deepest breadcrumb node; granularity varies BETWEEN items",
                   max_distinct_ratio=0.25, default_on=False),
        LabelField("main_cat", "categorical",
                   "Amazon store -- cuts across the breadcrumb, does not nest in it"),
        LabelField("brand_ext", "categorical",
                   "2018-dump brand. ~2.6 items/value: an identity vocabulary",
                   max_distinct_ratio=0.25, default_on=False),
        LabelField("price_usd", "continuous",
                   "USD, unambiguous $X.XX only", min_coverage=0.60),
        LabelField("rank", "continuous",
                   "primary store sales rank, lower is better", min_coverage=0.60),
        LabelField("also_buy", "graph",
                   "co-purchase neighbours -- the only non-text signal here",
                   min_coverage=0.60),
    )
}

# Deciles turn the two continuous fields into categorical ones without inventing
# a threshold: they are defined by the catalogue's own distribution.
DECILE_OF = {"price_decile": "price_usd", "rank_decile": "rank"}


@lru_cache(maxsize=4)
def load(category: str) -> dict[str, dict]:
    """ASIN -> label row. Raises with the fix if the table was never built."""
    path = paths.EXTERNAL / "labels" / f"{category}.labels.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist.\n"
            f"Build it:  python scripts/build_labels.py --category {category}")
    return json.loads(path.read_text())


@lru_cache(maxsize=4)
def manifest(category: str) -> dict:
    path = paths.EXTERNAL / "labels" / f"{category}.labels.manifest.json"
    return json.loads(path.read_text())


def coverage(category: str, keys: list[str] | None = None) -> dict[str, dict]:
    """Per-field coverage over `keys` (default: the whole label table).

    Recomputed against the caller's key list rather than read off the manifest,
    because a SidTable's keys and the catalogue can differ and the number that
    matters is coverage over the items actually being partitioned.
    """
    lab = load(category)
    keys = keys if keys is not None else list(lab)
    out = {}
    for name in list(FIELDS) + list(DECILE_OF):
        src = DECILE_OF.get(name, name)
        vals = [lab.get(k, {}).get(src) for k in keys]
        nonempty = [v for v in vals if v not in (None, "", [], {})]
        distinct = (len({json.dumps(v, sort_keys=True) for v in nonempty})
                    if src == "also_buy" else len(set(nonempty)))
        out[name] = {
            "n_items": len(keys),
            "non_empty": len(nonempty),
            "coverage": len(nonempty) / len(keys) if keys else 0.0,
            "distinct": distinct,
        }
    return out


def usable(category: str, keys: list[str] | None = None) -> dict[str, tuple[bool, str]]:
    """{field: (ok, reason)} -- the gate, evaluated against real coverage."""
    cov = coverage(category, keys)
    out = {}
    for name, f in FIELDS.items():
        out[name] = f.check(cov[name])
    for name, src in DECILE_OF.items():
        # A decile column is 10 classes by construction, so only coverage of the
        # underlying continuous field can disqualify it.
        c = cov[name]
        out[name] = ((c["coverage"] >= FIELDS[src].min_coverage),
                     "ok" if c["coverage"] >= FIELDS[src].min_coverage
                     else f"coverage {c['coverage']:.1%} < "
                          f"{FIELDS[src].min_coverage:.0%} required")
    return out


def codes(category: str, keys: list[str], field: str,
          force: bool = False) -> tuple[np.ndarray, list[str]]:
    """Integer label per key, plus the vocabulary. MISSING (-1) where absent.

    Returned as codes rather than strings because every consumer -- AMI, purity,
    probes -- wants codes, and building the vocab at each call site is how two
    callers end up disagreeing about what class 7 is.
    """
    if field in DECILE_OF:
        return _decile_codes(category, keys, field, force)
    if field not in FIELDS:
        raise KeyError(f"unknown label field {field!r}; "
                       f"have {sorted(list(FIELDS) + list(DECILE_OF))}")
    if not force:
        ok, why = usable(category, keys)[field]
        if not ok:
            raise ValueError(f"label field {field!r} is not usable here: {why}. "
                             f"Pass force=True to override, and say so in the output.")
    lab = load(category)
    vocab: dict[str, int] = {}
    out = np.full(len(keys), MISSING, dtype=np.int64)
    for i, k in enumerate(keys):
        v = lab.get(k, {}).get(field)
        if isinstance(v, str) and v.strip():
            out[i] = vocab.setdefault(v, len(vocab))
    return out, [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def _decile_codes(category: str, keys: list[str], field: str,
                  force: bool) -> tuple[np.ndarray, list[str]]:
    """Deciles of a continuous field, cut on the covered subset's own quantiles.

    Ties are frequent (price clusters on $9.99-style points), so bin edges are
    deduplicated and the actual number of bins is reported in the vocabulary
    rather than assumed to be 10.
    """
    src = DECILE_OF[field]
    if not force:
        ok, why = usable(category, keys)[field]
        if not ok:
            raise ValueError(f"label field {field!r} is not usable here: {why}.")
    lab = load(category)
    raw = np.array([lab.get(k, {}).get(src) if isinstance(
        lab.get(k, {}).get(src), (int, float)) else np.nan for k in keys], dtype=float)
    present = np.isfinite(raw)
    out = np.full(len(keys), MISSING, dtype=np.int64)
    if present.sum() < 10:
        return out, []
    edges = np.unique(np.quantile(raw[present], np.linspace(0, 1, 11)))
    binned = np.clip(np.digitize(raw[present], edges[1:-1], right=False),
                     0, len(edges) - 2)
    out[present] = binned
    return out, [f"{edges[i]:.6g}-{edges[i + 1]:.6g}" for i in range(len(edges) - 1)]


def neighbour_sets(category: str, keys: list[str]) -> dict[str, set[str]]:
    """ASIN -> its co-purchase neighbours that are IN this catalogue.

    `also_buy` points at the whole Amazon catalogue, most of which is not in the
    3105-item subset the SIDs partition. Restricting to in-catalogue neighbours
    is what makes the "do items sharing a prefix also get co-purchased" question
    answerable at all; the fraction surviving that restriction is reported by
    callers so a small overlap is never mistaken for a weak effect.
    """
    lab = load(category)
    inside = set(keys)
    return {k: {n for n in (lab.get(k, {}).get("also_buy") or []) if n in inside}
            for k in keys}
