"""Recover each diffusion checkpoint's configuration from its run artifacts.

The checkpoints are bare `torch.save(model.state_dict())` -- no config, no
optimizer, nothing. Two artifacts carry the rest:

    <rundir>/logs/.../<stamp>.log   the module repr, printed at construction
    diffgrm*/<category>/<variant>.txt  the full stdout transcript

Between them almost everything is recoverable. The exception that matters:

    n_head IS NOT RECOVERABLE from either, or from the weights.

`qkv` is Linear(256 -> 768) for any head count, so a wrong `n_head` reshapes the
same parameters into a different number of heads, passes load_state_dict(strict=True)
without complaint, and silently computes different attention. It is supplied from
the vendored sbatch and confirmed by a behavioral check against recorded metrics.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---- module repr -----------------------------------------------------------
RE_EMBEDDING = re.compile(r"\(embedding\): Embedding\((\d+), (\d+)\)")
RE_ITEM_MLP = re.compile(r"\(item_mlp\): Sequential\(\s*\n\s*\(0\): Linear\(in_features=(\d+)")
RE_MASK_EMB = re.compile(r"\(mask_emb_table\): Embedding\((\d+), (\d+)\)")
RE_POS_ENC = re.compile(r"\(pos_emb_enc\): Embedding\((\d+), (\d+)\)")
RE_FFN = re.compile(r"\(c_fc\): Linear\(in_features=(\d+), out_features=(\d+)")
RE_DROPOUT = re.compile(r"Dropout\(p=([\d.]+)")
RE_LN_EPS = re.compile(r"LayerNorm\(\((\d+),\), eps=([\deE.+-]+)")
RE_OUTPUT_ADAPTER = re.compile(r"\(output_adapter\): (\w+)\(")

# ---- run metadata ----------------------------------------------------------
RE_INDEX_FACTORY = re.compile(r"Index factory: EXTERNAL(\d+)x(\d+)")
RE_INDEX_FACTORY_ANY = re.compile(r"Index factory: (\S+)")
RE_SEM_IDS = re.compile(r"Loading external semantic IDs from (\S+?)\.\.\.")
RE_MAP_TAG = re.compile(r"Saved mappings with tag: (\S+)")
RE_GUIDED = re.compile(
    r"GUIDED: steps=(\d+), metric=(\w+), select=(\w+), augment_factor=(\d+)")
RE_MULTI_ITEM = re.compile(r"Multi-item target: n_target_items=(\d+), n_target_digits=(\d+)")
RE_PARAMS = re.compile(r"Total parameters:\s*([\d,]+)")
RE_USERS = re.compile(r"Number of users: (\d+)")
RE_ITEMS = re.compile(r"Number of items: (\d+)")
RE_INTER = re.compile(r"Number of interactions: (\d+)")
RE_TEST_RESULTS = re.compile(r"Test Results: (OrderedDict\(.*\))")
RE_BEST = re.compile(r"Best epoch: (\d+), Best val score: ([\d.]+)")


def _count_blocks(text: str, name: str) -> int | None:
    """Count layers in a ModuleList repr.

    torch collapses identical siblings, so a 4-layer stack prints as
    `(0-3): 4 x DecoderBlock` while a 1-layer stack prints as `(0): EncoderBlock`.
    Both spellings have to be handled or layer counts come out wrong.
    """
    m = re.search(rf"\({name}\): ModuleList\(\s*\n\s*\((\d+)-(\d+)\): (\d+) x ", text)
    if m:
        return int(m.group(3))
    m = re.search(rf"\({name}\): ModuleList\(\s*\n\s*\((\d+)\): ", text)
    if m:
        # Distinct siblings each print their own index.
        block = text.split(f"({name}): ModuleList(", 1)[1]
        return len(re.findall(r"\n      \(\d+\): \w+Block\(", block[:20000])) or 1
    return None


def parse_test_results(text: str) -> dict[str, float]:
    """Parse the trailing `Test Results: OrderedDict([...])` line.

    The repr embeds `np.float64(...)` wrappers, which literal_eval rejects, so
    they are stripped before parsing. Only the LAST occurrence is used -- a
    resumed run can emit several.
    """
    hits = RE_TEST_RESULTS.findall(text)
    if not hits:
        return {}
    raw = hits[-1]
    raw = re.sub(r"np\.float64\(([^)]*)\)", r"\1", raw)
    raw = re.sub(r"np\.int64\(([^)]*)\)", r"\1", raw)
    try:
        pairs = ast.literal_eval(raw[len("OrderedDict("):-1])
        return {k: float(v) for k, v in pairs}
    except (ValueError, SyntaxError):
        return {}


@dataclass
class ParsedRun:
    """Everything recoverable from a run's log + transcript."""
    # shape-derived
    vocab_size: int | None = None
    n_embd: int | None = None
    n_inner: int | None = None
    n_target_digits: int | None = None
    max_history_len: int | None = None
    encoder_n_layer: int | None = None
    decoder_n_layer: int | None = None
    dropout: float | None = None
    layer_norm_eps: float | None = None
    output_adapter: str | None = None
    item_mlp_in: int | None = None
    # tag-derived
    n_digit: int | None = None
    n_codebook_bits: int | None = None
    codebook_size: int | None = None
    index_factory: str | None = None
    sem_ids_path: str | None = None
    map_tag: str | None = None
    n_target_items: int = 1
    # training config visible in stdout
    masking_strategy: str | None = None
    guided_steps: int | None = None
    guided_conf_metric: str | None = None
    guided_select: str | None = None
    # dataset
    n_users: int | None = None
    n_items: int | None = None
    n_interactions: int | None = None
    # outcome
    total_parameters: int | None = None
    best_epoch: int | None = None
    best_val_score: float | None = None
    test_results: dict = field(default_factory=dict)
    # bookkeeping
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_head(self):
        """Deliberately absent. See the module docstring."""
        raise AttributeError(
            "n_head is not recoverable from logs or weights -- it must come from "
            "the vendored sbatch and be confirmed by the behavioral check")


def parse_text(text: str, into: ParsedRun | None = None) -> ParsedRun:
    p = into or ParsedRun()

    if m := RE_EMBEDDING.search(text):
        p.vocab_size, p.n_embd = int(m.group(1)), int(m.group(2))
    if m := RE_ITEM_MLP.search(text):
        p.item_mlp_in = int(m.group(1))
    if m := RE_MASK_EMB.search(text):
        p.n_target_digits = int(m.group(1))
    if m := RE_POS_ENC.search(text):
        p.max_history_len = int(m.group(1))
    if m := RE_FFN.search(text):
        p.n_inner = int(m.group(2))
    if m := RE_DROPOUT.search(text):
        p.dropout = float(m.group(1))
    if m := RE_LN_EPS.search(text):
        p.layer_norm_eps = float(m.group(2))
    if m := RE_OUTPUT_ADAPTER.search(text):
        p.output_adapter = m.group(1)

    enc = _count_blocks(text, "encoder_blocks")
    dec = _count_blocks(text, "decoder_blocks")
    if enc:
        p.encoder_n_layer = enc
    if dec:
        p.decoder_n_layer = dec

    if m := RE_INDEX_FACTORY.search(text):
        p.n_digit = int(m.group(1))
        p.n_codebook_bits = int(m.group(2))
        p.codebook_size = 2 ** p.n_codebook_bits
        p.index_factory = f"EXTERNAL{m.group(1)}x{m.group(2)}"
    elif m := RE_INDEX_FACTORY_ANY.search(text):
        p.index_factory = m.group(1)
        p.warnings.append(f"non-external index factory: {m.group(1)}")

    if m := RE_SEM_IDS.search(text):
        p.sem_ids_path = m.group(1)
    if m := RE_MAP_TAG.search(text):
        p.map_tag = m.group(1)
    if m := RE_MULTI_ITEM.search(text):
        p.n_target_items = int(m.group(1))
        p.n_target_digits = int(m.group(2))
    if m := RE_GUIDED.search(text):
        p.masking_strategy = "guided"
        p.guided_steps = int(m.group(1))
        p.guided_conf_metric = m.group(2)
        p.guided_select = m.group(3)
    if m := RE_PARAMS.search(text):
        p.total_parameters = int(m.group(1).replace(",", ""))
    if m := RE_BEST.search(text):
        p.best_epoch, p.best_val_score = int(m.group(1)), float(m.group(2))
    if m := RE_USERS.search(text):
        p.n_users = int(m.group(1))
    if m := RE_ITEMS.search(text):
        p.n_items = int(m.group(1))
    if m := RE_INTER.search(text):
        p.n_interactions = int(m.group(1))

    if results := parse_test_results(text):
        p.test_results = results
    return p


def canonical_metric(results: dict, name: str) -> float | None:
    """Look up a metric across the two naming conventions in play.

    The one-item evaluator emits `ndcg@10` / `recall@10`; the two-item trainer's
    two-pass branch emits `NDCG@10` / `HR@10` with partial pair credit. Callers
    that just want "the ndcg@10 of this run" should not have to know which.
    """
    aliases = {
        "ndcg@10": ("ndcg@10", "NDCG@10"),
        "ndcg@5": ("ndcg@5", "NDCG@5"),
        "ndcg@3": ("ndcg@3", "NDCG@3"),
        "recall@10": ("recall@10", "HR@10"),
        "recall@5": ("recall@5", "HR@5"),
        "recall@3": ("recall@3", "HR@3"),
    }
    for key in aliases.get(name, (name,)):
        if key in results:
            return results[key]
    return None


def parse_run(log_path: Path | None, transcript_path: Path | None) -> ParsedRun:
    """Merge a run's .log and .txt.

    The .log is read first (clean module repr); the transcript then fills in
    what only stdout carries -- the GUIDED banner and the final Test Results.
    """
    p = ParsedRun()
    for path in (log_path, transcript_path):
        if path is None or not Path(path).exists():
            continue
        text = Path(path).read_text(errors="replace")
        parse_text(text, into=p)
        p.sources.append(str(path))
    _check_consistency(p)
    return p


def _check_consistency(p: ParsedRun) -> None:
    """Cross-validate independently-derived fields.

    Each shape below is recoverable two ways; disagreement means the log and the
    transcript describe different runs, which must never be silently merged.
    """
    if p.vocab_size and p.n_digit and p.codebook_size:
        expected = 3 + p.n_digit * p.codebook_size
        if p.vocab_size != expected:
            p.warnings.append(
                f"vocab_size {p.vocab_size} != 3 + n_digit*K = {expected} "
                f"(n_digit={p.n_digit}, K={p.codebook_size})")
    if p.item_mlp_in and p.n_digit and p.n_embd:
        expected = p.n_digit * p.n_embd
        if p.item_mlp_in != expected:
            p.warnings.append(
                f"item_mlp in_features {p.item_mlp_in} != n_digit*n_embd = {expected}")
    if p.n_target_digits and p.n_digit:
        if p.n_target_digits % p.n_digit != 0:
            p.warnings.append(
                f"n_target_digits {p.n_target_digits} not a multiple of n_digit {p.n_digit}")
        elif p.n_target_digits // p.n_digit != p.n_target_items:
            p.warnings.append(
                f"n_target_digits/n_digit = {p.n_target_digits // p.n_digit} "
                f"!= n_target_items {p.n_target_items}")
