"""The autoregressive recommender: loading it, and finding its SID digits.

Upstream's AR arm is Qwen2.5-1.5B SFT'd on histories written as SID token
strings. Three checkpoints survived the sweep's delete-after-scoring step, and
they are the only AR weights that exist:

    next-item_best        rqkmeans 3cb x 128   -- matched to diff-next1-rqkmeans-3cb-128
    oneoff_rqvae4cb128    rqvae    4cb x 128   -- matched to diff-next1-rqvae-4cb-128
    two-item_best         MQ       4cb x 256   -- NO matched diffusion run exists

So every paradigm comparison in RQ1 runs on the first two cells, and the
two-item comparison in RQ4 cannot run at all without a retrain.

The token-id trap
-----------------
SID tokens were added to the tokenizer as `<a_N>`, `<b_N>`, ... one letter per
digit. Each letter occupies a CONTIGUOUS id block, which invites the obvious
shortcut:

    token_id = block_base + code          # WRONG

The ids inside a block are ordered by a plain string sort of the token text, so
`<a_100>` (id base+1) sorts before `<a_10>` (id base+11) because `'0' < '>'`.
The shortcut is off by an amount that varies per code, produces no error, and
yields logits for a different codeword than the one asked about. Every mapping
here is therefore READ from `added_tokens.json`, never computed, and
`validate_against` re-derives it from the SID table as a cross-check.

The vocabulary is not the codebook
----------------------------------
Only codes that actually occur in the catalogue were added as tokens.
`oneoff_rqvae4cb128` carries 423 SID tokens, not 4 x 128 = 512, because RQ-VAE's
first digit collapsed to 39 used codes. That is not a bookkeeping quirk: the
model's decision at digit 1 is over 39 options, and any per-digit entropy,
calibration, or "how much choice remains" statistic that assumes `codebook_size`
alternatives is wrong by a factor of three for that checkpoint. `n_codes(digit)`
is the number to use.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from sidlens import paths
from sidlens.data.sids import SidTable, SidVariant

RE_SID_TOKEN = re.compile(r"^<(?P<letter>[a-z])_(?P<code>\d+)>$")
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# The three surviving AR checkpoints and the SID each was trained on. Taken from
# frozen/ckpt/ar/best_variant_labels/, except the one-off whose variant is only
# in its directory name -- `validate_against` is what actually proves each one.
AR_CHECKPOINTS: dict[str, str] = {
    "next-item_best": "rqkmeans_3codebook_128",
    "oneoff_rqvae4cb128": "rqvae_4codebook_128",
    "two-item_best": "MQ_4codebook_256",
}

# Cells where an AR and a diffusion checkpoint share a SID table. RQ1's paradigm
# comparison is only defined on these.
MATCHED_CELLS: dict[str, tuple[str, str]] = {
    "rqkmeans_3codebook_128": ("next-item_best", "diff-next1-rqkmeans-3cb-128"),
    "rqvae_4codebook_128": ("oneoff_rqvae4cb128", "diff-next1-rqvae-4cb-128"),
}


@dataclass(frozen=True)
class SidVocab:
    """The bijection between (digit, code) and this model's token ids.

    Built by reading the tokenizer, so it is correct even though the ids are not
    ordered by code. `code_of` and `id_of` are inverses over exactly the codes
    the checkpoint knows about; asking for a code it never saw raises rather
    than returning a plausible neighbour.
    """
    variant: SidVariant
    id_by_digit_code: dict[tuple[int, int], int]
    digit_code_by_id: dict[int, tuple[int, int]]

    @cached_property
    def digits(self) -> list[int]:
        return sorted({d for d, _ in self.id_by_digit_code})

    def codes(self, digit: int) -> list[int]:
        """The codes this checkpoint has tokens for, at one digit position."""
        return sorted(c for d, c in self.id_by_digit_code if d == digit)

    def n_codes(self, digit: int) -> int:
        """Size of the model's ACTUAL decision at this digit.

        Not `codebook_size`. See the module docstring: for rqvae 4cb x 128 the
        first digit offers 39 options, not 128.
        """
        return sum(1 for d, _ in self.id_by_digit_code if d == digit)

    def id_of(self, digit: int, code: int) -> int:
        try:
            return self.id_by_digit_code[(digit, code)]
        except KeyError:
            raise KeyError(
                f"digit {digit} code {code} has no token in this checkpoint. "
                f"It has {self.n_codes(digit)} codes at digit {digit}, which is "
                f"fewer than the nominal {self.variant.codebook_size} because "
                f"unused codes were never added to the vocabulary.") from None

    def code_of(self, token_id: int) -> tuple[int, int]:
        return self.digit_code_by_id[token_id]

    def ids(self, digit: int) -> np.ndarray:
        """Token ids for one digit, ordered by CODE.

        Ordered by code rather than by id so that `logits[:, vocab.ids(d)]`
        yields a vector indexable by codeword. Slicing the raw contiguous id
        block instead would give a vector in string-sort order -- the exact
        mistake this class exists to prevent.
        """
        return np.array([self.id_by_digit_code[(digit, c)]
                         for c in self.codes(digit)], dtype=np.int64)

    def tokens_for(self, code: tuple[int, ...]) -> list[int]:
        """A full SID as the token ids the model consumes."""
        return [self.id_of(d, c) for d, c in enumerate(code)]


def load_vocab(ckpt: str, variant: SidVariant | str | None = None) -> SidVocab:
    """Read the (digit, code) -> token id map out of a checkpoint's tokenizer."""
    d = paths.FROZEN_CKPT / "ar" / ckpt
    if variant is None:
        variant = AR_CHECKPOINTS.get(ckpt)
        if variant is None:
            raise KeyError(f"unknown AR checkpoint {ckpt!r}; "
                           f"have {sorted(AR_CHECKPOINTS)}")
    if isinstance(variant, str):
        variant = SidVariant.parse(variant)

    added = json.loads((d / "added_tokens.json").read_text())
    fwd: dict[tuple[int, int], int] = {}
    rev: dict[int, tuple[int, int]] = {}
    for tok, tid in added.items():
        m = RE_SID_TOKEN.match(tok)
        if not m:
            continue
        digit = LETTERS.index(m["letter"])
        code = int(m["code"])
        fwd[(digit, code)] = tid
        rev[tid] = (digit, code)

    if not fwd:
        raise ValueError(f"{d}: no <letter_code> SID tokens in added_tokens.json")
    n_digits = len({d for d, _ in fwd})
    if n_digits != variant.n_codebook:
        raise ValueError(
            f"{ckpt}: tokenizer spans {n_digits} digit letters but "
            f"{variant.name} has {variant.n_codebook} codebooks. The checkpoint "
            f"and the variant do not agree; one of them is mislabelled.")
    return SidVocab(variant, fwd, rev)


def validate_against(vocab: SidVocab, table: SidTable | None = None) -> dict:
    """Prove the vocabulary matches the SID table it claims to encode.

    The checkpoint's tokenizer and the frozen `.sem_ids` are independent
    artifacts. If they disagree -- a code used in the table with no token, or a
    token for a code the table never assigns -- then either the checkpoint is
    paired with the wrong variant or the SID table changed after training. Both
    are silent failures at inference: generation still produces tokens, they
    just decode to the wrong items.
    """
    table = table or SidTable.load(vocab.variant)
    codes = table.codes
    report = {"variant": vocab.variant.name, "ok": True, "digits": []}
    for d in range(vocab.variant.n_codebook):
        used = set(np.unique(codes[:, d]).tolist())
        have = set(vocab.codes(d))
        missing = sorted(used - have)
        extra = sorted(have - used)
        ok = not missing
        report["digits"].append({
            "digit": d,
            "codes_used_in_table": len(used),
            "codes_in_vocab": len(have),
            "nominal_codebook_size": vocab.variant.codebook_size,
            "missing_from_vocab": missing[:10],
            "n_missing": len(missing),
            # Extra tokens are harmless -- an unused token simply never wins.
            # Missing ones are fatal: an item the model cannot address.
            "n_extra_in_vocab": len(extra),
            "ok": ok,
        })
        report["ok"] &= ok
    return report


def digit_logits(logits, vocab: SidVocab, digit: int):
    """Restrict a full-vocabulary logit vector to one digit's codes, code-ordered.

    Works on the last axis, so it accepts (vocab,), (T, vocab) or (B, T, vocab).
    Returned columns are indexed by position in `vocab.codes(digit)`, not by
    codeword value -- the two differ whenever a codebook has unused codes, and
    `vocab.codes(digit)` is the key to read them back.
    """
    idx = vocab.ids(digit)
    try:
        import torch
        if isinstance(logits, torch.Tensor):
            return logits.index_select(-1, torch.as_tensor(idx, device=logits.device))
    except ImportError:
        pass
    return np.take(np.asarray(logits), idx, axis=-1)


def load_model(ckpt: str, device: str = "cpu", dtype: str = "float32"):
    """Load the AR checkpoint as a HF causal LM.

    Local files only. These weights are the artifact under study; silently
    fetching a same-named model from the hub would substitute a different
    network and every downstream number would describe that one instead.
    """
    import torch
    from transformers import AutoModelForCausalLM

    d = paths.FROZEN_CKPT / "ar" / ckpt
    if not (d / "model.safetensors").exists():
        raise FileNotFoundError(f"{d}/model.safetensors is missing")
    model = AutoModelForCausalLM.from_pretrained(
        str(d), local_files_only=True, dtype=getattr(torch, dtype))
    model.eval()
    return model.to(device)


def available() -> list[dict]:
    """Which AR checkpoints are on disk, and whether each has a diffusion match.

    26 of 28 planned AR runs had their weights deleted after scoring, so this
    list is short by design and the `matched_diffusion` column is the practical
    constraint on RQ1.
    """
    out = []
    for ckpt, variant in sorted(AR_CHECKPOINTS.items()):
        d = paths.FROZEN_CKPT / "ar" / ckpt
        match = MATCHED_CELLS.get(variant)
        out.append({
            "checkpoint": ckpt,
            "variant": variant,
            "on_disk": (d / "model.safetensors").exists(),
            "matched_diffusion": match[1] if match else None,
        })
    return out
