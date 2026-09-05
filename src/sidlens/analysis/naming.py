"""What does this code SAY? Discriminative terms per SID code cell.

`attribute.py` answers "does digit d align with a known attribute" as a number.
That number cannot say what a digit means when it aligns with nothing in the
label set -- and the label set is 25 categories, while `rqkmeans 3cb x 128` has
128 codes at digit 1 alone. Most codes are finer than any label available, so
the only way to read them is from the item text itself.

The contrast is the method
--------------------------
Naming a cell by its most frequent words returns "pack", "kit", "inch" for every
cell in the catalogue. The question is not what a cell contains but what it
contains MORE OF THAN ITS SIBLINGS -- which is the same conditional question the
AMI table asks, in readable form:

    foreground   items under this code cell
    background   items under the SAME PARENT but a different code

For a depth-1 cell the background is the rest of the catalogue. For deeper cells
it is the sibling set, so the terms that come back describe what digit d added,
not what the prefix already established.

Weighted log-odds, not TF-IDF
-----------------------------
TF-IDF has no variance model, so on cells holding 8-30 items its top terms are
dominated by words appearing once. This uses the Monroe/Colaresi/Quinn weighted
log-odds ratio with an informative Dirichlet prior taken from the full
catalogue:

    delta_w = log( (y_fg + a_w) / (n_fg + a0 - y_fg - a_w) )
            - log( (y_bg + a_w) / (n_bg + a0 - y_bg - a_w) )
    var_w  ~= 1/(y_fg + a_w) + 1/(y_bg + a_w)
    z_w     = delta_w / sqrt(var_w)

The prior shrinks rare words toward the corpus rate, so a word seen twice in a
12-item cell has to beat its own sampling noise before it outranks a word seen
nine times. Ranking by z rather than delta is what makes small cells readable.

Does the signature actually hold?
---------------------------------
A term list always looks plausible, which is exactly why one should not be
trusted on sight. `validate` splits each cell in half, builds the signature from
one half, and scores the held-out half against sibling items. The reported AUC
says whether the signature identifies cell membership on items it never saw. A
cell whose terms are noise lands at 0.5 and is reported as such rather than
quietly printed next to cells that land at 0.9.

Terms containing digits are kept, not stripped. "1/4-inch" and "18-gauge" are
genuine attributes in this catalogue, while "740001201" is a model number and is
not; both carry `has_digit` so a reader can separate them without the module
guessing which is which.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

TOKEN = re.compile(r"[a-z0-9][a-z0-9./\-']*")

# Product titles are dense with packaging and marketing filler that is uniform
# across the catalogue. These carry no contrast and only crowd the output.
STOP = {
    "the", "and", "for", "with", "of", "in", "to", "a", "an", "by", "on", "or",
    "from", "at", "is", "it", "as", "no", "new", "pack", "count", "piece",
    "pieces", "pcs", "inch", "inches", "x", "w", "l", "h", "size", "type",
    "use", "used", "each", "per", "包", "-",
}

ALPHA0 = 500.0     # Dirichlet prior strength, in pseudo-tokens
MIN_FG = 4         # items a cell needs before a signature is attempted
MIN_DF = 2         # times a term must appear in the foreground


@dataclass(frozen=True)
class Term:
    term: str
    z: float
    delta: float
    fg_count: int
    bg_count: int
    fg_docs: int          # items in the cell whose text contains it
    has_digit: bool


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower())
            if t not in STOP and len(t) > 1]


def tokenize_all(texts: dict[str, str]) -> dict[str, list[str]]:
    """Tokenize the catalogue once.

    The naive version re-tokenizes the background for every cell, and at depth 1
    the background is the whole catalogue -- 125 cells x 3105 items per variant,
    times 27 variants. Doing it once up front is the difference between a
    two-hour sweep and a ten-minute one.
    """
    return {k: tokenize(v) for k, v in texts.items()}


def corpus_prior(texts: dict[str, str] | dict[str, list[str]]) -> tuple[Counter, float]:
    """Catalogue-wide term counts, the informative part of the prior.

    Accepts raw strings or pre-tokenized documents, so callers that already
    tokenized do not pay for it twice.
    """
    c = Counter()
    for t in texts.values():
        c.update(t if isinstance(t, list) else tokenize(t))
    return c, float(sum(c.values()))


def doc_counts(docs: dict[str, list[str]], keys: list[str]) -> tuple[Counter, Counter]:
    """(term counts, document frequencies) over `keys`."""
    tf, df = Counter(), Counter()
    for k in keys:
        toks = docs.get(k, ())
        tf.update(toks)
        df.update(set(toks))
    return tf, df


def signature_from_counts(fg: Counter, fg_docs: Counter, bg: Counter,
                          n_fg_items: int, prior: Counter, prior_total: float,
                          top: int = 12, alpha0: float = ALPHA0) -> list[Term]:
    """Weighted log-odds ranking, given counts that the caller already has.

    Split out from `signature` so `describe_cells` can obtain a cell's sibling
    counts by SUBTRACTING the cell from its parent, rather than re-counting the
    siblings for every one of the parent's children.
    """
    if n_fg_items < MIN_FG:
        return []
    n_fg, n_bg = sum(fg.values()), sum(bg.values())
    if n_fg == 0 or n_bg == 0:
        return []

    out: list[Term] = []
    for w, y_fg in fg.items():
        if y_fg < MIN_DF:
            continue
        a_w = alpha0 * (prior.get(w, 0) / prior_total) if prior_total else 0.0
        if a_w <= 0:
            continue
        y_bg = bg.get(w, 0)
        num_fg = y_fg + a_w
        num_bg = y_bg + a_w
        den_fg = n_fg + alpha0 - num_fg
        den_bg = n_bg + alpha0 - num_bg
        if den_fg <= 0 or den_bg <= 0:
            continue
        delta = math.log(num_fg / den_fg) - math.log(num_bg / den_bg)
        var = 1.0 / num_fg + 1.0 / num_bg
        out.append(Term(w, delta / math.sqrt(var), delta, y_fg, y_bg,
                        fg_docs[w], any(ch.isdigit() for ch in w)))
    out.sort(key=lambda t: -t.z)
    return out[:top]


def signature(fg_keys: list[str], bg_keys: list[str],
              texts: dict[str, str] | dict[str, list[str]],
              prior: Counter, prior_total: float, top: int = 12,
              alpha0: float = ALPHA0) -> list[Term]:
    """Terms that distinguish `fg_keys` from `bg_keys`, ranked by z."""
    if len(fg_keys) < MIN_FG or not bg_keys:
        return []
    docs = _as_docs(texts)
    fg, fg_docs = doc_counts(docs, fg_keys)
    bg, _ = doc_counts(docs, bg_keys)
    return signature_from_counts(fg, fg_docs, bg, len(fg_keys), prior,
                                 prior_total, top=top, alpha0=alpha0)


def _as_docs(texts) -> dict[str, list[str]]:
    """Accept either raw text or pre-tokenized documents."""
    if texts and isinstance(next(iter(texts.values())), list):
        return texts
    return tokenize_all(texts)


def _score(toks, weights: dict[str, float]) -> float:
    """Sum of signature weights over an item's terms, length-normalised.

    Without the normalisation a long title outscores a short one for having more
    chances to hit the signature, which would make the AUC below a measure of
    title length rather than of the signature.
    """
    if isinstance(toks, str):
        toks = tokenize(toks)
    if not toks:
        return 0.0
    return sum(weights.get(t, 0.0) for t in toks) / math.sqrt(len(toks))


def validate(fg_keys: list[str], bg_keys: list[str],
             texts: dict[str, str] | dict[str, list[str]],
             prior: Counter, prior_total: float, top: int = 12,
             seed: int = 0, bg_scores: np.ndarray | None = None) -> dict:
    """Held-out AUC for a cell's signature.

    Half the cell trains the signature, the other half is scored against the
    sibling items. AUC 0.5 means the terms carry nothing about membership; the
    number is computed on items the signature never saw, so a cell cannot pass
    by memorising itself.
    """
    if len(fg_keys) < 2 * MIN_FG or len(bg_keys) < MIN_FG:
        return {"auc": None, "n_train": 0, "n_test": 0, "n_bg": len(bg_keys),
                "reason": "too few items to split"}
    docs = _as_docs(texts)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fg_keys))
    half = len(fg_keys) // 2
    train = [fg_keys[i] for i in perm[:half]]
    test = [fg_keys[i] for i in perm[half:]]

    terms = signature(train, bg_keys, docs, prior, prior_total, top=top)
    if not terms:
        return {"auc": None, "n_train": len(train), "n_test": len(test),
                "n_bg": len(bg_keys), "reason": "no signature from train half"}
    weights = {t.term: t.z for t in terms}
    pos = np.array([_score(docs.get(k, ()), weights) for k in test])
    neg = (bg_scores if bg_scores is not None
           else np.array([_score(docs.get(k, ()), weights) for k in bg_keys]))

    # Mann-Whitney U as AUC, ties counted as half -- ties are common here
    # because most background items contain none of the signature terms and
    # score exactly 0, and calling those wins would inflate every cell.
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    auc = float((wins + 0.5 * ties) / (len(pos) * len(neg)))
    return {"auc": round(auc, 4), "n_train": len(train), "n_test": len(test),
            "n_bg": len(bg_keys), "reason": ""}


def describe_cells(table, texts: dict[str, str], max_depth: int | None = None,
                   min_items: int = MIN_FG, top: int = 12,
                   do_validate: bool = True, seed: int = 0) -> list[dict]:
    """One signature row per code cell, with its held-out AUC.

    Cells below `min_items` are skipped rather than described from two items;
    the count of skipped cells is the caller's to report, and the atlas already
    carries it as the singleton share.
    """
    docs = _as_docs(texts)
    prior, prior_total = corpus_prior(docs)
    max_depth = max_depth or table.variant.n_codebook
    rows: list[dict] = []
    for depth in range(1, max_depth + 1):
        clusters = table.clusters(depth)
        parents = table.clusters(depth - 1)
        # One pass per parent, reused by all of its children. The sibling counts
        # of a child are the parent's counts minus the child's, which is exact
        # and costs O(child) instead of O(parent) per child.
        parent_tf: dict[tuple, Counter] = {}
        for pfx, members in sorted(clusters.items()):
            if len(members) < min_items:
                continue
            ppfx = pfx[:-1]
            if ppfx not in parent_tf:
                parent_tf[ppfx] = doc_counts(docs, parents[ppfx])[0]
            fg, fg_docs = doc_counts(docs, members)
            bg = parent_tf[ppfx].copy()
            bg.subtract(fg)
            bg = +bg          # drop zero and negative entries
            sib_n = len(parents[ppfx]) - len(members)
            if sib_n <= 0 or not bg:
                continue
            terms = signature_from_counts(fg, fg_docs, bg, len(members),
                                          prior, prior_total, top=top)
            sibs = ([k for k in parents[ppfx] if k not in set(members)]
                    if do_validate else [])
            val = (validate(members, sibs, docs, prior, prior_total, top=top,
                            seed=seed) if do_validate else {})
            rows.append({
                "variant": table.variant.name,
                "quantizer": table.variant.quantizer,
                "depth": depth,
                "digit": depth - 1,
                "prefix": "-".join(map(str, pfx)),
                "code": pfx[-1],
                "n_items": len(members),
                "n_siblings": sib_n,
                "terms": " ".join(t.term for t in terms),
                "terms_z": " ".join(f"{t.z:.2f}" for t in terms),
                "terms_no_digit": " ".join(
                    t.term for t in terms if not t.has_digit),
                "top_term": terms[0].term if terms else "",
                "top_z": round(terms[0].z, 3) if terms else None,
                "heldout_auc": val.get("auc"),
                "auc_note": val.get("reason", ""),
            })
    return rows


def depth_summary(rows: list[dict]) -> list[dict]:
    """Per-digit rollup of how readable the codes at that digit are.

    `median_auc` is the headline: a digit whose cells validate near 0.5 is not
    carrying text-readable meaning, whatever its terms look like.
    """
    out = []
    for d in sorted({r["digit"] for r in rows}):
        sub = [r for r in rows if r["digit"] == d]
        aucs = np.array([r["heldout_auc"] for r in sub
                         if r["heldout_auc"] is not None], dtype=float)
        out.append({
            "variant": sub[0]["variant"],
            "quantizer": sub[0]["quantizer"],
            "digit": d,
            "depth": d + 1,
            "n_cells_described": len(sub),
            "n_cells_validated": int(aucs.size),
            "median_auc": round(float(np.median(aucs)), 4) if aucs.size else None,
            "q25_auc": round(float(np.quantile(aucs, 0.25)), 4) if aucs.size else None,
            "q75_auc": round(float(np.quantile(aucs, 0.75)), 4) if aucs.size else None,
            "share_auc_above_0.7": (round(float((aucs > 0.7).mean()), 4)
                                    if aucs.size else None),
        })
    return out
