"""The grid's AR test decoder (OneDiffRec `evaluate_ar_joint.decode_batch`), with copy edits.

OneDiffRec decodes every grid AR run in two phases: (1) plain beam search,
`num_beams` beams, a catalog grammar (trie) at every step, the checkpoint's
`generation_config.json` ignored; (2) every distinct returned SID is scored
again under teacher forcing with the raw-vocabulary log-prob of all its tokens
(SID tokens, newline, EOS), and the list is that score, sorted. `decode` is
that function statement for statement. With `edit=None` it is the vendored
decoder (`tests/grid/test_decode_ar.py`); the reproduction check (V2) proves it
on each run's saved lists.

Copy edits
----------
`CopyEdit` removes (or scales by alpha) the attention edges exp6 removes: when
a beam predicts digit d, its reads of digit d of the history items whose first
d digits equal the beam's own prefix (and at digit 0, under `C_all`, of every
history item), in the later half of the layers. The SAME edges are applied in
both phases, generation (from each beam's prefix) and rescoring (from each
candidate's codes), because the reported list is ranked by the rescoring: an
edit applied to the search alone would be partly undone by the ranking.

Every condition of an experiment, including its baseline, goes through the
edit route (an empty plan, or alpha = 1), so conditions differ only by the
edges. Whether that route equals the plain decoder is reported (V2b).

Traps
-----
1. **Row to prompt.** Generation rows are prompt-major beams (`row //
   num_beams`); the plan maps through it, never by position.
2. **Padding.** Prompts are left-padded to the batch's widest; history token
   positions are shifted by each prompt's padding. With the evaluator's batch
   of one there is none.
3. **Steps after the SID.** Newline and EOS steps get no edges.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field

from sidlens.data import ar_prompts as P
from sidlens.grid import v3code

CONDITIONS = ("B", "C_later", "C_all", "N_later")


def _ev():
    return v3code.ar_eval()


def catalog_grammar(tokenizer, index_path, horizon: int = 1):
    """OneDiffRec's `CatalogGrammar` over the run's own SID table (`load_catalog`)."""
    ev = _ev()
    groups = ev.load_catalog(index_path)
    return ev.CatalogGrammar(tokenizer, groups, horizon), groups


def encode_prompt(tokenizer, history_sids):
    return _ev().encode_prompt(tokenizer, list(history_sids))


@dataclass
class PromptInfo:
    """What the edit needs about one prompt: each history item's codes and the
    position (in the unpadded prompt) of each of its digit tokens."""
    codes: list[tuple[int, ...]]
    token_pos: dict[tuple[int, int], int]
    length: int

    @classmethod
    def from_encoded(cls, enc: P.Encoded, n_digits: int) -> "PromptInfo":
        ex = enc.example
        codes = [P.parse_sid(s, n_digits) for s in ex.history_sids]
        pos = {(j, d): enc.positions("hist_sid", item=j, digit=d)[0]
               for j in range(len(codes)) for d in range(n_digits)}
        return cls(codes, pos, enc.prompt_len)


@dataclass
class CopyEdit:
    """exp6's copy conditions as edge plans for both decoding phases.

    condition  B | C_later | C_all | N_later
    alpha      None = knockout (boolean masks); a number = scale by alpha
               (float masks; alpha = 0 is the knockout through that route)
    """
    condition: str
    infos: list[PromptInfo]
    vocab: object
    n_digits: int
    num_beams: int
    layers: tuple[int, ...]
    alpha: float | None = None
    counts: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")

    # -- which history items a prediction of digit d reads -------------------
    def keys_for(self, info: PromptInfo, d: int, prefix: tuple[int, ...]) -> list[tuple[int, int]]:
        """(item, digit) pairs whose edge is edited when digit d follows `prefix`."""
        if self.condition == "B":
            return []
        items = range(len(info.codes))
        if d == 0:
            return [(j, 0) for j in items] if self.condition == "C_all" else []
        match = [j for j in items if tuple(info.codes[j][:d]) == prefix]
        if self.condition in ("C_later", "C_all"):
            return [(j, d) for j in match]
        non = [j for j in reversed(items) if j not in match]       # N_later: as many non-matching
        return [(j, d) for j in non[:len(match)]]

    def _count(self, phase, n):
        self.counts[phase] = self.counts.get(phase, 0) + n

    # -- phase 1: generation ---------------------------------------------------
    def generation(self, model, width: int):
        pads = [width - info.length for info in self.infos]

        def plan(seqs, step, q, kv):
            if step >= self.n_digits:
                return []
            gen = seqs[:, width:width + step].tolist()
            edges = []
            for row in range(seqs.shape[0]):
                p = row // self.num_beams
                info, pad = self.infos[p], pads[p]
                if step == 0:
                    if q != width:
                        raise RuntimeError(f"step 0 must be the prefill ({q} != {width})")
                    qi, prefix = width - 1, ()
                else:
                    if q != 1:
                        raise RuntimeError(f"step {step} expected one query, got {q}")
                    dc = [self.vocab.digit_code_by_id.get(t) for t in gen[row]]
                    if any(x is None for x in dc) or [x[0] for x in dc] != list(range(step)):
                        raise RuntimeError(f"row {row} step {step}: the grammar let a non-SID prefix through")
                    qi, prefix = 0, tuple(x[1] for x in dc)
                keys = [pad + info.token_pos[jd] for jd in self.keys_for(info, step, prefix)]
                if keys:
                    edges.append((row, qi, keys))
            self._count("generation_edges", sum(len(k) for _, _, k in edges))
            return edges

        from sidlens.interventions.attn_scale import generation_scale
        from sidlens.interventions.generation import generation_knockout
        if self.alpha is None:
            return generation_knockout(model, self.layers, plan, prompt_len=width)
        return generation_scale(model, self.layers, plan, prompt_len=width, alpha=self.alpha)

    # -- phase 2: rescoring ------------------------------------------------------
    def rescoring_edges(self, prompt_index: int, generated: list[int]) -> list[tuple[int, list[int]]]:
        """(query position, key positions) in the unpadded rescoring sequence."""
        info = self.infos[prompt_index]
        dc = [self.vocab.digit_code_by_id.get(t) for t in generated[:self.n_digits]]
        if any(x is None for x in dc):
            raise RuntimeError("a returned candidate does not start with a whole SID")
        codes = tuple(x[1] for x in dc)
        out = []
        for d in range(self.n_digits):
            keys = [info.token_pos[jd] for jd in self.keys_for(info, d, codes[:d])]
            if keys:
                out.append((info.length - 1 + d, keys))
        self._count("rescoring_edges", sum(len(k) for _, k in out))
        return out


def _score(model, sequences, edit: CopyEdit | None, owners, prompt_lens, generated):
    """Raw-vocabulary logits of right-padded `sequences`, with the edit if any."""
    import torch

    device = next(model.parameters()).device
    pad = generated["pad"]
    size = max(map(len, sequences))
    ids = torch.tensor([s + [pad] * (size - len(s)) for s in sequences], device=device)
    mask2 = torch.tensor([[1] * len(s) + [0] * (size - len(s)) for s in sequences], device=device)
    if edit is None:
        return model(input_ids=ids, attention_mask=mask2).logits
    from sidlens.interventions.attention import Knockout, base_mask, knockout_attention
    from sidlens.interventions.attn_scale import Scale, float_mask, scale_attention
    bmask = base_mask(mask2, model.config.num_attention_heads)
    edges = []
    for r, (p, plen, gen) in enumerate(zip(owners, prompt_lens, generated["tokens"])):
        for q, keys in edit.rescoring_edges(p, gen):
            edges.append((r, q, keys))
    if edit.alpha is None:
        kos = [Knockout(r, (q,), tuple(keys), tuple(edit.layers)) for r, q, keys in edges]
        with knockout_attention(model, bmask, kos):
            return model(input_ids=ids, attention_mask=bmask).logits
    fmask = float_mask(bmask, next(model.parameters()).dtype)
    scs = [Scale(r, (q,), tuple(keys), tuple(edit.layers), edit.alpha) for r, q, keys in edges]
    with scale_attention(model, fmask, scs):
        return model(input_ids=ids, attention_mask=fmask).logits


def decode(model, tokenizer, grammar, prompts, num_beams=50, top_k=20, score_batch_size=1,
           edit: CopyEdit | None = None):
    """`evaluate_ar_joint.decode_batch`, plus `edit` (see module doc)."""
    import torch
    from transformers import GenerationConfig

    if num_beams < 2 or top_k < 1 or top_k > num_beams or score_batch_size < 1:
        raise ValueError("Require num_beams >= 2, 1 <= top_k <= num_beams and positive score_batch_size")
    if edit is not None and len(edit.infos) != len(prompts):
        raise ValueError("the edit needs one PromptInfo per prompt, in batch order")
    width = max(map(len, prompts))
    pad = tokenizer.eos_token_id
    device = next(model.parameters()).device
    input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
    attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
    context = getattr(model.config, "max_position_embeddings", getattr(model.config, "n_positions", None))
    if context and width + grammar.max_new_tokens > context:
        raise ValueError(f"Prompt plus target exceeds model context ({width} + {grammar.max_new_tokens} > {context})")
    config = GenerationConfig(
        do_sample=False, num_beams=num_beams, num_return_sequences=num_beams,
        max_new_tokens=grammar.max_new_tokens, length_penalty=0.0,
        early_stopping=False, renormalize_logits=False, repetition_penalty=1.0,
        eos_token_id=pad, pad_token_id=pad, bos_token_id=tokenizer.bos_token_id,
        forced_bos_token_id=None, forced_eos_token_id=None,
    )
    ctx = edit.generation(model, width) if edit is not None else nullcontext()
    with torch.inference_mode(), ctx:
        output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            generation_config=config, use_model_defaults=False,
            prefix_allowed_tokens_fn=lambda batch_id, ids: grammar.allowed(ids[width:].tolist()),
        )
    candidates = []
    by_example = [[] for _ in prompts]
    for example, prompt in enumerate(prompts):
        seen = set()
        for beam in output[example * num_beams:(example + 1) * num_beams]:
            sids, target_tokens = grammar.parse(beam[width:].tolist())
            key = tuple(sids)
            if key in seen:
                continue
            seen.add(key)
            candidate = {"sids": sids, "generated_token_ids": target_tokens}
            candidates.append((example, prompt, candidate))
            by_example[example].append(candidate)
    for start in range(0, len(candidates), score_batch_size):
        part = candidates[start:start + score_batch_size]
        sequences = [prompt + candidate["generated_token_ids"] for _, prompt, candidate in part]
        with torch.inference_mode():
            logits = _score(model, sequences, edit, [e for e, _, _ in part], [len(p) for _, p, _ in part],
                            {"pad": pad, "tokens": [c["generated_token_ids"] for _, _, c in part]})
        for index, (_, prompt, candidate) in enumerate(part):
            count = len(candidate["generated_token_ids"])
            positions = logits[index, len(prompt) - 1:len(prompt) - 1 + count].float()
            labels = torch.tensor(candidate["generated_token_ids"], device=device)
            values = positions.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1).cpu().tolist()
            candidate["raw_vocabulary_token_logprobs"] = values
            candidate["joint_logprob"] = float(sum(values))
            if not math.isfinite(candidate["joint_logprob"]):
                raise ValueError("Non-finite model score during joint evaluation")
        del logits
    return [sorted(cands, key=lambda c: (-c["joint_logprob"], c["sids"]))[:top_k] for cands in by_example]


def intervention_protocol(proto: dict, rescore_batch: int = 0) -> dict:
    """The run's protocol for an experiment's intervention decodes: its beams, list length and
    batch size, with each row's returned candidates rescored `rescore_batch` at a time (0: all
    of them in one forward, at most `num_beams`; 1: the evaluator's own setting).

    Amendment of 2026-10-23 for exp6, exp9 and N3 on the grid, before any grid result: the
    evaluator rescores one candidate at a time, which is most of a decode at 50 beams (A100:
    1.30 s per row, or 1.63 s with copy reads blocked, against 0.37 / 0.51 s batched).
    Batching right-pads the candidates of one row together; it changes float rounding, not
    which candidates are scored. Every condition of an experiment uses the same setting, so
    contrasts stay clean, and each experiment's reproduction gate (V2a, V-D0) still decodes
    at the evaluator's setting, where the saved lists are reproduced exactly."""
    if rescore_batch < 0:
        raise ValueError("rescore_batch is 0 (all candidates of a row) or a positive batch size")
    return {**proto, "score_batch_size": int(rescore_batch or proto["num_beams"]),
            "evaluator_score_batch_size": proto["score_batch_size"]}


def late_layers(model) -> tuple[int, ...]:
    """The later half of the layers, where exp4 found every copy read (14-27 of 28)."""
    n = model.config.num_hidden_layers
    return tuple(range(n // 2, n))


def decode_rows(model, tokenizer, grammar, examples, proto: dict, *, prompts=None, edit_for=None,
                log=None, log_every: int = 200) -> dict[str, list[dict]]:
    """Decode `examples` the way the run's evaluator did: its beams, list length and
    batch sizes (`grid.predictions.ar_protocol`), batches in the given row order.

    prompts   token ids per example (default: the evaluator's prompt of its history);
              an experiment's changed prompts must keep the example's id.
    edit_for  callable(batch examples, batch prompts) -> CopyEdit | None, per batch.
    Returns example_id -> ranked candidates (sids, joint_logprob, token log-probs).
    """
    if prompts is None:
        prompts = [encode_prompt(tokenizer, ex.history_sids) for ex in examples]
    if len(prompts) != len(examples):
        raise ValueError("one prompt per example")
    out, bs = {}, proto["batch_size"]
    for start in range(0, len(examples), bs):
        exs, ps = examples[start:start + bs], prompts[start:start + bs]
        edit = edit_for(exs, ps) if edit_for is not None else None
        ranked = decode(model, tokenizer, grammar, ps, proto["num_beams"], proto["top_k"],
                        proto["score_batch_size"], edit=edit)
        for ex, r in zip(exs, ranked):
            if ex.example_id in out:
                raise ValueError(f"{ex.example_id} decoded twice")
            out[ex.example_id] = [{"sids": c["sids"], "joint_logprob": c["joint_logprob"],
                                   "raw_vocabulary_token_logprobs": c["raw_vocabulary_token_logprobs"]}
                                  for c in r]
        if log and (start // bs + 1) % log_every == 0:
            log(f"decoded {min(start + bs, len(examples))}/{len(examples)}")
    return out


def run_conditions(model, tokenizer, grammar, examples, encoded, vocab, proto: dict, conditions,
                   *, alpha: float | None = None, log=None) -> tuple[dict, dict]:
    """Decode every condition on the same rows and batches, all through the edit route.

    encoded   `P.encode(..., template="v3")` of each example (prompt only); its ids
              must be the evaluator's prompt, which is checked row by row here.
    Returns ({condition: {example_id: ranked candidates}}, {condition: edge counts}).
    """
    n = vocab.variant.n_codebook
    prompts = [e.input_ids for e in encoded]
    for ex, p in zip(examples, prompts):
        if p != encode_prompt(tokenizer, ex.history_sids):
            raise RuntimeError(f"{ex.example_id}: sidlens prompt differs from the evaluator's")
    infos = {ex.example_id: PromptInfo.from_encoded(e, n) for ex, e in zip(examples, encoded)}
    layers = late_layers(model)
    out, counts = {}, {}
    for cond in conditions:
        edits = []

        def edit_for(exs, ps, cond=cond):
            ed = CopyEdit(cond, [infos[x.example_id] for x in exs], vocab, n, proto["num_beams"], layers,
                          alpha=alpha)
            edits.append(ed)
            return ed

        out[cond] = decode_rows(model, tokenizer, grammar, examples, proto, prompts=prompts,
                                edit_for=edit_for, log=log)
        counts[cond] = {}
        for ed in edits:
            for k, v in ed.counts.items():
                counts[cond][k] = counts[cond].get(k, 0) + v
        if log:
            log(f"[{cond}{'' if alpha is None else f' alpha={alpha}'}] done, edges {counts[cond]}")
    return out, counts


def lists_frame(lists: dict, examples, condition: str):
    """One row per example: `predict` (SID strings, ' ||| '-joined for two targets) and `scores`."""
    import pandas as pd
    return pd.DataFrame([{"example_id": ex.example_id, "row": ex.row, "user_id": ex.user_id,
                          "condition": condition,
                          "predict": [" ||| ".join(c["sids"]) for c in lists[ex.example_id]],
                          "scores": [c["joint_logprob"] for c in lists[ex.example_id]]}
                         for ex in examples if ex.example_id in lists])
