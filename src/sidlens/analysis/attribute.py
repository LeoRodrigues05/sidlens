"""Do the digits carry attributes -- and does digit d carry any GIVEN 1..d-1?

This is the measurement half of Experiment 1. `atlas.py` describes what each
code cell CONTAINS; this module asks whether the containment lines up with a
real attribute, and whether the alignment survives conditioning on the prefix.

The distinction that carries the experiment
-------------------------------------------
  marginal      AMI(label, digit d) over the whole catalogue. Confounded --
                digit 1 correlates with product type, product type correlates
                with material, so digit 3 can look material-aligned purely
                through digit 1.

  conditional   AMI(label, digit d) computed INSIDE each depth-(d-1) cluster and
                pooled by cluster size. This is the brief's actual question:
                what does digit d add once digits 1..d-1 are known?

For a residual quantizer the conditional value can exceed the marginal -- the
digit resolves an attribute its parent left open. For a parallel quantizer like
MQ, digits re-encode the same embedding, so conditioning removes the shared
component and the value collapses.

Why there is a permutation null
-------------------------------
AMI is chance-corrected in expectation, which is not the same as having low
variance. Pooled over parent cells holding 8-30 items each, the conditional
statistic has a wide null distribution, and a small positive number there means
nothing on its own. So every conditional value is accompanied by a null built
the only way that respects the design: shuffle the labels WITHIN each parent
cell and recompute, which destroys the digit-label association while preserving
cluster sizes, the label marginal inside each parent, and the number of parents
contributing. The reported z and empirical p come from that null, not from an
asymptotic assumption that does not hold at these group sizes.

The labels themselves
---------------------
`labels.py` serves the recovered Amazon-2018 category hierarchy. Two things it
does that matter here: it refuses near-identity columns (brand, at 2.6 items per
value, would score high for any quantizer and mean nothing), and it exposes
`also_buy` -- a co-purchase signal owing nothing to item text. Every text-derived
label is ambiguous between "the digit encodes meaning" and "the digit encodes
the text encoder that produced z_i". Co-purchase is the one column that breaks
that tie, so `copurchase_preservation` is reported alongside the AMI table.

The lexicon path below predates the label recovery. It is kept, but demoted: it
labels a fraction of the catalogue from title strings alone, and its only job
now is to cross-check the recovered labels on the subset it does cover.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from sidlens.data import labels as L

# Lexicons are deliberately small and unambiguous. An item mentioning two
# colours is left unlabelled rather than assigned the first match: "red/yellow
# connector" is not a red item, and guessing would put noise into a signal whose
# whole purpose is to audit another signal.
LEXICONS: dict[str, list[str]] = {
    "colour": ["red", "blue", "yellow", "black", "white", "green", "orange",
               "clear", "chrome", "brass"],
    "material": ["stainless steel", "aluminum", "brass", "nylon", "plastic",
                 "rubber", "ceramic", "copper", "zinc"],
    "form": ["kit", "assortment", "pack", "set", "roll", "spool", "tube",
             "bottle", "cartridge"],
}

MIN_GROUP = 8       # labelled items a parent cell needs to contribute
MIN_CLASSES = 2
N_PERM = 200        # permutations for the conditional null
SUPPORT_ITEMS = 100
SUPPORT_PARENTS = 5


def weak_labels(titles: dict[str, str], lexicon: list[str]) -> dict[str, str]:
    """ASIN -> term, for items whose title states exactly one term."""
    out = {}
    pats = [(t, re.compile(rf"\b{re.escape(t)}\b")) for t in lexicon]
    for asin, title in titles.items():
        low = title.lower()
        hits = [t for t, p in pats if p.search(low)]
        if len(hits) == 1:
            out[asin] = hits[0]
    return out


def _pooled_conditional(prefixes: np.ndarray, digit: np.ndarray,
                        y: np.ndarray) -> tuple[float, int, int]:
    """Size-weighted mean AMI(y, digit) within each distinct prefix group.

    Returns (value, n_items_contributing, n_groups). Groups too small, or with
    no variation in either the digit or the label, contribute nothing -- inside
    such a group there is no question to ask, and scoring it as 0 would drag the
    pooled value toward zero for a reason that has nothing to do with the digit.
    """
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, p in enumerate(prefixes):
        groups[p.tobytes()].append(i)
    num = den = 0.0
    n_groups = 0
    for idx in groups.values():
        if len(idx) < MIN_GROUP:
            continue
        xs, ys = digit[idx], y[idx]
        if len(set(xs.tolist())) < MIN_CLASSES or len(set(ys.tolist())) < MIN_CLASSES:
            continue
        num += float(adjusted_mutual_info_score(ys, xs)) * len(idx)
        den += len(idx)
        n_groups += 1
    return (num / den if den else float("nan")), int(den), n_groups


def _conditional_null(prefixes: np.ndarray, digit: np.ndarray, y: np.ndarray,
                      n_perm: int, seed: int) -> np.ndarray:
    """Null for the pooled conditional AMI: labels shuffled WITHIN each parent.

    Shuffling globally would also destroy the parent-label association, which is
    not the null we want -- that null would call a digit informative merely
    because its parent was. Permuting inside the parent holds everything the
    prefix already determined and randomizes only what digit d could add.
    """
    rng = np.random.default_rng(seed)
    groups: dict[bytes, list[int]] = defaultdict(list)
    for i, p in enumerate(prefixes):
        groups[p.tobytes()].append(i)
    idx_lists = [np.array(v) for v in groups.values() if len(v) >= MIN_GROUP]
    out = np.empty(n_perm)
    for t in range(n_perm):
        yp = y.copy()
        for idx in idx_lists:
            yp[idx] = rng.permutation(yp[idx])
        out[t] = _pooled_conditional(prefixes, digit, yp)[0]
    return out


def alignment(table, y: np.ndarray, keys: list[str], field: str,
              n_perm: int = N_PERM, seed: int = 0) -> list[dict]:
    """Marginal and conditional AMI between each digit and one label field.

    `y` is integer codes row-aligned with `keys`, MISSING (-1) for unlabelled.
    Unlabelled items are dropped once, up front, so every digit in the returned
    table is computed on exactly the same item set and the rows are comparable
    down the column.
    """
    y = np.asarray(y)
    keep = np.flatnonzero(y != L.MISSING)
    if len(keep) < MIN_GROUP * 2:
        return []
    sub_keys = [keys[i] for i in keep]
    yk = y[keep]
    codes = np.array([table.asin2codes[k] for k in sub_keys], dtype=np.int64)

    rows = []
    for d in range(table.variant.n_codebook):
        digit = codes[:, d]
        marginal = float(adjusted_mutual_info_score(yk, digit))

        prefixes = codes[:, :d]
        if d == 0:
            # Depth 0 has one group -- the catalogue -- so the conditional is
            # the marginal by definition. Stated rather than recomputed.
            cond, n_items, n_groups = marginal, len(yk), 1
            null = np.array([])
        else:
            cond, n_items, n_groups = _pooled_conditional(prefixes, digit, yk)
            null = (_conditional_null(prefixes, digit, yk, n_perm, seed + d)
                    if np.isfinite(cond) and n_groups else np.array([]))

        if null.size:
            mu, sd = float(null.mean()), float(null.std(ddof=1))
            z = (cond - mu) / sd if sd > 0 else float("nan")
            p = float((null >= cond).sum() + 1) / (null.size + 1)
        else:
            mu = sd = z = float("nan")
            p = float("nan")

        rows.append({
            "variant": table.variant.name,
            "quantizer": table.variant.quantizer,
            "field": field,
            "digit": d,
            "n_labelled": len(yk),
            "n_classes": int(len(set(yk.tolist()))),
            "marginal_ami": round(marginal, 4),
            "conditional_ami": round(cond, 4) if np.isfinite(cond) else None,
            "cond_null_mean": round(mu, 4) if np.isfinite(mu) else None,
            "cond_null_sd": round(sd, 4) if np.isfinite(sd) else None,
            "cond_z": round(z, 2) if np.isfinite(z) else None,
            "cond_p": round(p, 4) if np.isfinite(p) else None,
            "n_parent_cells": n_groups,
            "n_items_conditioned": n_items,
            # The absolute count hides how thin the conditioning gets: at depth 3
            # it is routinely a few percent of the labelled catalogue, because
            # most parents no longer hold MIN_GROUP labelled items. Carried as a
            # share so the reader sees that without doing the division.
            "conditioned_share": round(n_items / len(yk), 4) if len(yk) else 0.0,
            # Depth-3+ conditioning routinely falls to a few dozen items across a
            # handful of parents. Carried as a column so a reader never has to
            # infer whether a conditional number is supported.
            "conditional_supported": bool(n_items >= SUPPORT_ITEMS
                                          and n_groups >= SUPPORT_PARENTS),
        })
    return rows


def copurchase_preservation(table, keys: list[str],
                            category: str) -> list[dict]:
    """Do items sharing a depth-d prefix also get bought together?

    Every other column in this module is text-derived and therefore cannot
    distinguish a digit that encodes meaning from one that encodes the text
    encoder. `also_buy` is behavioural, so a prefix that concentrates co-purchase
    edges is evidence about the catalogue, not about Qwen3-Embedding.

    Reported as a lift over the chance rate for a partition with these exact
    cluster sizes, sum_C n_C(n_C - 1) / n(n - 1), so the number is comparable
    across depths and across quantizers with different branching.
    """
    nbrs = L.neighbour_sets(category, keys)
    edges = [(a, b) for a, ns in nbrs.items() for b in ns]
    n = len(keys)
    out = []
    for d in range(1, table.variant.n_codebook + 1):
        pfx = {k: table.asin2codes[k][:d] for k in keys}
        sizes = np.array([len(v) for v in table.clusters(d).values()], dtype=float)
        chance = float((sizes * (sizes - 1)).sum() / (n * (n - 1))) if n > 1 else 0.0
        same = sum(1 for a, b in edges if pfx[a] == pfx[b])
        obs = same / len(edges) if edges else float("nan")
        out.append({
            "variant": table.variant.name,
            "quantizer": table.variant.quantizer,
            "depth": d,
            "digit": d - 1,
            "n_edges": len(edges),
            "n_items_with_edges": sum(1 for s in nbrs.values() if s),
            "same_prefix_rate": round(obs, 4),
            "chance_rate": round(chance, 6),
            "lift": round(obs / chance, 2) if chance > 0 else None,
        })
    return out


def lexicon_crosscheck(table, keys: list[str], titles: dict[str, str],
                       category: str) -> list[dict]:
    """Agreement between the title lexicon and the recovered category labels.

    Not a result in itself. It answers one question: on the items the lexicon
    does cover, does it see the same structure the recovered labels see? If the
    two disagree sharply, one of them is measuring something other than what its
    name says, and that has to be known before either is trusted.
    """
    out = []
    for name, terms in LEXICONS.items():
        wl = weak_labels(titles, terms)
        if not wl:
            continue
        vocab: dict[str, int] = {}
        y = np.array([vocab.setdefault(wl[k], len(vocab)) if k in wl else L.MISSING
                      for k in keys], dtype=np.int64)
        both = np.flatnonzero(y != L.MISSING)
        cat, _ = L.codes(category, keys, "cat_l1")
        both = np.array([i for i in both if cat[i] != L.MISSING])
        rows = alignment(table, y, keys, f"lexicon:{name}", n_perm=0)
        out.extend(rows)
        if len(both) >= MIN_GROUP:
            out.append({
                "variant": table.variant.name,
                "quantizer": table.variant.quantizer,
                "field": f"lexicon:{name}",
                "digit": -1,          # -1 marks the label-vs-label row
                "n_labelled": int(len(both)),
                "n_classes": len(vocab),
                "marginal_ami": round(float(
                    adjusted_mutual_info_score(y[both], cat[both])), 4),
                "conditional_ami": None,
                "cond_null_mean": None, "cond_null_sd": None,
                "cond_z": None, "cond_p": None,
                "n_parent_cells": 0, "n_items_conditioned": 0,
                "conditional_supported": False,
            })
    return out
