"""The exact token sequence each AR recommender saw, with every position named.

An activation, a logit read or a patch on the AR model is only meaningful if
(a) the input is byte-for-byte the one upstream fed the model, and (b) we know
what each position *is*: which history item, which digit, which target slot.
This module rebuilds both from the frozen SFT CSVs. `tests/data/test_ar_prompts.py`
proves the token ids equal the vendored dataset classes' output row for row,
and that the text equals the archived prediction files.

Traps, each with the guard that closes it
-----------------------------------------
1. **Two prompt wordings, not one.** Training (`sft.py`) mixed five datasets.
   The SID-history -> SID part is `data.SidSFTDataset`: "The user has
   interacted with items {h} in chronological order. Can you predict ...".
   Validation loss used the same class. Test evaluation (`evaluate.py`,
   `evaluate_two_item.py` -> `data.EvalSidDataset`) used *different* words:
   "Can you predict the next possible item the user may expect, given the
   following chronological interaction history: {h}". Every archived AR
   prediction comes from the eval wording. `template` is therefore a required
   argument: "eval" explains archived predictions, "sft" is the training
   distribution, and a capture must say which one it used.
2. **Two encodes, not one.** Upstream tokenizes the instruction and the
   prompt separately and concatenates the ids. Tokenizing the joined string
   can merge tokens across the boundary. `_encode` mirrors
   `vendor/onediffrec/data.py::Tokenizer.encode`, including its BOS/EOS
   stripping.
3. **Off-by-one on logits.** The logits at position t score the token at
   t + 1. Digit 0 is decided at the last prompt token (the ":\n" of
   "### Response:\n"). Digit d > 0 is decided at the position of target digit
   d - 1, under teacher forcing. Use `predict_pos(slot, digit)`, never
   `prompt_len + digit`.
4. **The constrained-decoding key.** `evaluate.py` keys its trie on the last
   three prompt tokens, which must be the tokens of "### Response:\n". If a
   prompt ever ends differently, the archived decoder did not see what we
   think it saw, so `encode` raises.
5. **Padding moves positions.** `evaluate.py` left-pads, and `generate()`
   derives `position_ids` from the attention mask. A plain `model(input_ids,
   attention_mask)` on a left-padded batch does not, so every real token sits
   at a shifted RoPE position. `collate` returns explicit `position_ids` and
   per-row offsets for either padding side.
6. **Row order is the only join key to archived predictions.** They carry no
   user id. On 2026-09-25 the CSV row index matched 3,681 / 3,681 archived
   `input`/`output` strings for `rqkmeans_3codebook_128`. `check_against_archive`
   re-proves it for any variant. Never join AR rows to the diffusion cohort by
   row number: they are different cohorts (see CLAUDE.md).
7. **Python-literal lists.** The CSVs store lists as reprs, and upstream calls
   `eval()` on them. Here they are parsed with `ast.literal_eval`.
8. **SIDs can collide.** A target SID may decode to several items.
   `target_item_ids` holds the true item(s) from the CSV, not a decode of the
   SID.
9. **Two-item layout.** The target is "sid1 ||| sid2\\n". Evaluation pass 2
   appends pred1 + " ||| " and generates slot 1 under the same trie. The
   position map labels slot 0 / slot 1 and the separator separately.
10. **Stale SID strings.** A CSV from another cluster or branch may carry
   SIDs from a different table, e.g. the pre-repair RQ-KMeans codes.
   `check_against_table` compares every history SID with the frozen `.sem_ids`.
   All 54 frozen test CSVs pass.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sidlens import paths
from sidlens.data.sids import SidTable, SidVariant, load_item2id

PRIMARY_CATEGORY = "Industrial_and_Scientific"
TASKS = ("next-item", "two-item")
SPLITS = ("train", "valid", "test")
TEMPLATES = ("eval", "sft")

# Byte-exact copies of the vendored literals (vendor/onediffrec/data.py).
INSTRUCTION = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request. \n\n### Instruction:\nCan you predict the next possible item "
    "that the user may expect?\n\n")
USER_INPUT = {
    "eval": ("Can you predict the next possible item the user may expect, given "
             "the following chronological interaction history: {history}"),
    "sft": ("The user has interacted with items {history} in chronological order. "
            "Can you predict the next possible item that the user may expect?"),
}
PROMPT = "### User Input: \n{input}\n\n### Response:\n"
RESPONSE_HEADER = "### Response:\n"
HISTORY_SEP = ", "
TWO_ITEM_SEP = " ||| "

RE_SID_PART = re.compile(r"<([a-z])_(\d+)>")
RE_CSV_NAME = r"^{stem}(_\d{{4}}-\d{{2}}-\d{{4}}-\d{{2}})?\.csv$"


# ------------------------------------------------------------------- rows --
@dataclass(frozen=True)
class ArExample:
    """One CSV row, parsed. `row` is the 0-based data row; it is the join key
    to the archived predictions for the same (task, split, variant)."""
    example_id: str
    task: str
    split: str
    variant: str
    row: int
    user_id: str
    history_item_ids: tuple[int, ...]
    history_sids: tuple[str, ...]          # e.g. "<a_28><b_33><c_113>"
    target_item_ids: tuple[int, ...]       # one per slot
    target_sids: tuple[str, ...]           # one per slot

    @property
    def target_text(self) -> str:
        """What upstream appends after "### Response:\\n" (before EOS)."""
        return TWO_ITEM_SEP.join(self.target_sids) + "\n"


def parse_sid(s: str, n_digits: int) -> tuple[int, ...]:
    """'<a_28><b_33><c_113>' -> (28, 33, 113), refusing anything malformed."""
    parts = RE_SID_PART.findall(s)
    if "".join(f"<{l}_{c}>" for l, c in parts) != s.strip():
        raise ValueError(f"not a pure SID string: {s!r}")
    letters = "".join(l for l, _ in parts)
    if letters != "abcdefghijklmnopqrstuvwxyz"[:n_digits]:
        raise ValueError(f"expected {n_digits} digits a.. in order, got {s!r}")
    return tuple(int(c) for _, c in parts)


def csv_path(variant: SidVariant | str, task: str = "next-item",
             split: str = "test", category: str = PRIMARY_CATEGORY) -> Path:
    """The frozen SFT CSV for one variant. Two variants carry a date suffix."""
    if task not in TASKS or split not in SPLITS:
        raise ValueError(f"task in {TASKS}, split in {SPLITS}; got {task!r}, {split!r}")
    name = variant.name if isinstance(variant, SidVariant) else SidVariant.parse(variant).name
    rx = re.compile(RE_CSV_NAME.format(stem=re.escape(f"{category}_{name}")))
    root = paths.FROZEN_DATA / "splits" / task / split
    hits = sorted(p for p in root.iterdir() if rx.match(p.name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{root}: expected one CSV for {name}, found {[p.name for p in hits]}")
    return hits[0]


def load_examples(variant: SidVariant | str, task: str = "next-item",
                  split: str = "test", path: Path | None = None) -> list[ArExample]:
    """Every row of one SFT CSV, in file order.

    `path` overrides the frozen location, e.g. to read a CSV exported from
    another cluster. Such a file should then pass `check_against_table`
    before anything uses it.
    """
    import pandas as pd

    v = variant if isinstance(variant, SidVariant) else SidVariant.parse(variant)
    path = path or csv_path(v, task, split)
    # dtype=str: upstream reads with default dtypes then calls str(); keeping the
    # raw text avoids a float round-trip on any numeric-looking column.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"user_id", "history_item_id", "history_item_sid", "item_id", "item_sid"}
    if missing := need - set(df.columns):
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    n_slots = 2 if task == "two-item" else 1
    out = []
    for i, r in enumerate(df.itertuples(index=False)):
        hist_ids = tuple(int(x) for x in ast.literal_eval(r.history_item_id))
        hist_sids = tuple(str(x) for x in ast.literal_eval(r.history_item_sid))
        tgt_sids = tuple(s.strip() for s in r.item_sid.split("|||"))
        tgt_ids = tuple(int(s) for s in r.item_id.split("|||"))
        if len(hist_ids) != len(hist_sids) or not hist_ids:
            raise ValueError(f"{path.name} row {i}: history ids/SIDs disagree or empty")
        if len(tgt_sids) != n_slots or len(tgt_ids) != n_slots:
            raise ValueError(f"{path.name} row {i}: expected {n_slots} target(s), "
                             f"got {r.item_sid!r}")
        for s in (*hist_sids, *tgt_sids):
            parse_sid(s, v.n_codebook)
        out.append(ArExample(f"{task}/{split}/{v.name}/{i}", task, split, v.name, i,
                             r.user_id, hist_ids, hist_sids, tgt_ids, tgt_sids))
    return out


def user_input(ex: ArExample, template: str) -> str:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    return USER_INPUT[template].format(history=HISTORY_SEP.join(ex.history_sids))


# ------------------------------------------------------------- validation --
def archive_path(variant: SidVariant | str, task: str = "next-item",
                 category: str = PRIMARY_CATEGORY) -> Path:
    v = variant if isinstance(variant, SidVariant) else SidVariant.parse(variant)
    stem = {"next-item": "nextitem", "two-item": "twoitem"}[task]
    return (paths.FROZEN_RESULTS / "sweep_metrics" / task / "metrics"
            / f"{stem}__{category}__{v.quantizer}__{v.n_codebook}cb__{v.codebook_size}"
              f".predictions.json")


def check_against_archive(examples: list[ArExample]) -> dict:
    """Row-by-row text equality with the archived AR predictions (test split).

    Next-item rows carry the eval-template `input` and the target `output`, so
    both are compared. Two-item rows carry only `gt1`/`gt2`.
    """
    if not examples:
        raise ValueError("no examples")
    ex0 = examples[0]
    if ex0.split != "test":
        raise ValueError("archived predictions exist for the test split only")
    pred = json.loads(archive_path(ex0.variant, ex0.task).read_text())
    bad = []
    if len(pred) != len(examples):
        return {"ok": False, "n": len(examples), "n_archive": len(pred), "bad_rows": []}
    for ex, p in zip(examples, pred):
        if ex.task == "next-item":
            same = (p["input"] == user_input(ex, "eval") and p["output"] == ex.target_text)
        else:
            same = (p["gt1"], p["gt2"]) == ex.target_sids
        if not same:
            bad.append(ex.row)
    return {"ok": not bad, "n": len(examples), "n_archive": len(pred), "bad_rows": bad[:20],
            "n_bad": len(bad)}


def check_against_table(examples: list[ArExample], table: SidTable | None = None) -> dict:
    """Every history and target SID string equals the frozen `.sem_ids` code."""
    if not examples:
        raise ValueError("no examples")
    v = SidVariant.parse(examples[0].variant)
    table = table or SidTable.load(v)
    id2asin = {i: a for a, i in load_item2id(PRIMARY_CATEGORY).items()}
    n, bad, unknown = 0, [], 0
    for ex in examples:
        pairs = [*zip(ex.history_item_ids, ex.history_sids),
                 *zip(ex.target_item_ids, ex.target_sids)]
        for item, sid in pairs:
            asin = id2asin.get(item)
            if asin is None:
                unknown += 1
                continue
            n += 1
            if parse_sid(sid, v.n_codebook) != table.asin2codes[asin]:
                bad.append((ex.row, item))
    return {"ok": not bad and not unknown, "n_checked": n, "n_bad": len(bad),
            "bad": bad[:20], "n_unknown_items": unknown}


# --------------------------------------------------------------- encoding --
def _encode(tok, s: str, bos: bool, eos: bool) -> list[int]:
    """`vendor/onediffrec/data.py::Tokenizer.encode`, reproduced exactly."""
    t = tok.encode(s)
    while t and t[0] == tok.bos_token_id:
        t = t[1:]
    while t and t[-1] == tok.eos_token_id:
        t = t[:-1]
    if bos and tok.bos_token_id is not None:
        t = [tok.bos_token_id] + t
    if eos and tok.eos_token_id is not None:
        t = t + [tok.eos_token_id]
    return t


@dataclass
class Encoded:
    """Token ids plus a label for every position.

    role    instruction | prompt | hist_sid | hist_sep | response_header |
            target_sid | target_sep | target_end
    item    history index (0 = oldest) for hist_*, target slot for target_*,
            else -1. For hist_sep, the index of the item it follows.
    digit   SID digit for *_sid positions, else -1
    code    SID code for *_sid positions, else -1
    """
    example: ArExample
    template: str
    input_ids: list[int]
    prompt_len: int
    role: list[str]
    item: list[int]
    digit: list[int]
    code: list[int]

    def __len__(self) -> int:
        return len(self.input_ids)

    @property
    def has_target(self) -> bool:
        return len(self.input_ids) > self.prompt_len

    def positions(self, role: str, item: int | None = None,
                  digit: int | None = None) -> list[int]:
        return [i for i, r in enumerate(self.role) if r == role
                and (item is None or self.item[i] == item)
                and (digit is None or self.digit[i] == digit)]

    def target_pos(self, slot: int, digit: int) -> int:
        hits = self.positions("target_sid", slot, digit)
        if len(hits) != 1:
            raise KeyError(f"no target token for slot {slot} digit {digit}"
                           + ("" if self.has_target else " (encoded without target)"))
        return hits[0]

    def predict_pos(self, slot: int, digit: int) -> int:
        """Position whose next-token logits score target (slot, digit).

        Slot 0 digit 0 is the last prompt token and needs no target. Anything
        later is teacher-forced, so it needs `with_target=True`.
        """
        if slot == 0 and digit == 0:
            return self.prompt_len - 1
        return self.target_pos(slot, digit) - 1

    def records(self) -> list[dict]:
        """One dict per position, for an activation-store row table."""
        return [{"example_id": self.example.example_id, "template": self.template,
                 "pos": i, "token_id": t, "role": self.role[i], "item": self.item[i],
                 "digit": self.digit[i], "code": self.code[i]}
                for i, t in enumerate(self.input_ids)]


def _label_sids(ids: list[int], vocab, n_digits: int, expected: tuple[str, ...],
                sid_role: str, sep_role: str, before: str | None, after: str,
                where: str) -> tuple[list[str], list[int], list[int], list[int]]:
    """Label a span that contains `expected` SIDs in order, separated by text.

    `before=None` means no text may precede the first SID (the target span).
    """
    role, item, digit, code = [], [], [], []
    k = -1                       # SID group currently open / last closed
    seen = 0
    for t in ids:
        dc = vocab.digit_code_by_id.get(t)
        if dc is not None:
            d, c = dc
            if d == 0:
                k += 1
            if k >= len(expected) or seen % n_digits != d:
                raise ValueError(f"{where}: SID tokens out of order at group {k}, digit {d}")
            role.append(sid_role); item.append(k); digit.append(d); code.append(c)
            seen += 1
        else:
            if seen and seen % n_digits:
                raise ValueError(f"{where}: text token inside SID group {k}")
            if k < 0 and before is None:
                raise ValueError(f"{where}: text before the first SID")
            r = before if k < 0 else (after if k == len(expected) - 1 else sep_role)
            role.append(r); item.append(k if r == sep_role else -1)
            digit.append(-1); code.append(-1)
    got = seen // n_digits
    if got != len(expected) or seen % n_digits:
        raise ValueError(f"{where}: found {seen} SID tokens, expected "
                         f"{len(expected)} x {n_digits}")
    for j, s in enumerate(expected):
        want = parse_sid(s, n_digits)
        have = tuple(c for c, it, r in zip(code, item, role) if r == sid_role and it == j)
        if have != want:
            raise ValueError(f"{where}: SID {j} tokenised as {have}, expected {want}")
    return role, item, digit, code


def encode(ex: ArExample, tokenizer, vocab, *, template: str,
           with_target: bool = False) -> Encoded:
    """Upstream's exact input ids for one example, with every position labelled.

    `with_target=False` gives what the evaluator fed `generate()`
    (EvalSidDataset/SidSFTDataset with test=True). `with_target=True` appends
    the golden target + EOS, as the SFT loss saw it (test=False). That is the
    teacher-forced input a per-digit logit or activation read needs.
    """
    n = SidVariant.parse(ex.variant).n_codebook
    if vocab.variant.name != ex.variant:
        raise ValueError(f"vocab is for {vocab.variant.name}, example is {ex.variant}")
    instr = _encode(tokenizer, INSTRUCTION, bos=True, eos=False)
    prompt = _encode(tokenizer, PROMPT.format(input=user_input(ex, template)),
                     bos=False, eos=False)
    header = _encode(tokenizer, RESPONSE_HEADER, bos=False, eos=False)
    if prompt[-len(header):] != header:
        raise ValueError(f"{ex.example_id}: prompt does not end in the tokens of "
                         f"{RESPONSE_HEADER!r}; the archived trie key would differ")
    if any(t in vocab.digit_code_by_id for t in instr):
        raise ValueError("SID token inside the instruction")

    role = ["instruction"] * len(instr)
    item = [-1] * len(instr)
    digit = [-1] * len(instr)
    code = [-1] * len(instr)
    r, it, d, c = _label_sids(prompt, vocab, n, ex.history_sids, "hist_sid", "hist_sep",
                              "prompt", "prompt", f"{ex.example_id} prompt")
    r[-len(header):] = ["response_header"] * len(header)
    role += r; item += it; digit += d; code += c
    ids = instr + prompt
    prompt_len = len(ids)
    if with_target:
        tgt = _encode(tokenizer, ex.target_text, bos=False, eos=True)
        r, it, d, c = _label_sids(tgt, vocab, n, ex.target_sids, "target_sid", "target_sep",
                                  None, "target_end", f"{ex.example_id} target")
        role += r; item += it; digit += d; code += c
        ids = ids + tgt
    return Encoded(ex, template, ids, prompt_len, role, item, digit, code)


def collate(batch: list[Encoded], pad_id: int, side: str = "right"):
    """Pad a batch; return tensors plus per-row offsets into the padded axis.

    `side="left"` mirrors evaluate.py's batches, and its `position_ids` are
    the ones `generate()` would compute: cumsum(mask) - 1, with pads set to
    1. Always pass them to the forward. `side="right"` leaves every real token
    at its unpadded position, which is the simpler choice for teacher-forced
    capture. Position p of row i lives at padded index `offset[i] + p`.
    """
    import torch

    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    L = max(len(e) for e in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    offset = []
    for i, e in enumerate(batch):
        n = len(e)
        o = L - n if side == "left" else 0
        ids[i, o:o + n] = torch.tensor(e.input_ids)
        mask[i, o:o + n] = 1
        offset.append(o)
    pos = mask.cumsum(-1) - 1
    pos = pos.masked_fill(mask == 0, 1)
    return {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
            "offset": torch.tensor(offset)}
