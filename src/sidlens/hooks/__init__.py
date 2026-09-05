"""Reading hidden states out of both decoders without editing either.

Experiment 1's probe half, and all of RQ1, need the same primitive: run a model
and keep what it computed at every layer. The two architectures under study are
not alike --

    AR          Qwen2.5-1.5B, 28 layers x 1536, `model.layers.N`
    DiffGRM     4 decoder layers x 256 over a 1-layer encoder,
                `encoder_blocks.N` / `decoder_blocks.N`, with cross-attention

-- so anything that hardcodes one of them will quietly do nothing on the other.
`residual_points` names the comparable read positions per architecture and
raises on a model it does not recognise, rather than returning an empty list
that would make a capture look successful and produce no data.

Why hooks and not a patch
-------------------------
`vendor/` is byte-frozen and every deviation must be a registered monkeypatch.
Forward hooks are not a deviation: they observe the module graph without
altering it, are removed on exit even if the forward raises, and cannot change
numerics. So nothing here needs a patch id -- and by the same token, nothing
here may be used to CHANGE an activation. Interventions belong in
`sidlens.interventions`, where the numerics flag is a real question.

The tuple problem
-----------------
A transformer block usually returns `(hidden, attn_weights, ...)` rather than a
bare tensor, and which element is the hidden state differs by implementation.
Silently taking `output[0]` works for both models here but would fail somewhere
else without saying so, so `_first_tensor` records WHICH element it took and the
capture carries that alongside the data.

Memory
------
Activations are detached and moved off-device by default. A 28-layer 1536-wide
capture over a batch of 64 histories of length 50 is ~550 MB in float32; the
same run in float16 on GPU without the move will sit on top of the model
weights. `to_cpu=False` is available and is a deliberate choice, not a default.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# The residual read positions per architecture. Keyed by the model class name
# because that is what distinguishes them; a model whose class is absent raises.
RESIDUAL_PATTERNS: dict[str, tuple[str, ...]] = {
    # HF Qwen2 causal LM: one entry per decoder layer, plus the final norm.
    "Qwen2ForCausalLM": (r"^model\.layers\.\d+$", r"^model\.norm$"),
    # Vendored DIFF_GRM: encoder and decoder stacks are both read positions,
    # because the paradigm question is partly about where history is consumed.
    "DIFF_GRM": (r"^encoder_blocks\.\d+$", r"^decoder_blocks\.\d+$"),
}

# Attention submodules, for the head-level work RQ1 and RQ4 need. DiffGRM's
# cross_attn is the module RQ4's "remove the link between the two items" test
# operates on, so it is named separately rather than lumped in with self_attn.
ATTENTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "Qwen2ForCausalLM": (r"^model\.layers\.\d+\.self_attn$",),
    "DIFF_GRM": (r"^encoder_blocks\.\d+\.attn$",
                 r"^decoder_blocks\.\d+\.self_attn$",
                 r"^decoder_blocks\.\d+\.cross_attn$"),
}


@dataclass
class Captured:
    """What one hooked module produced on one forward pass."""
    name: str
    value: Any                      # tensor, detached
    took_index: int | None          # which tuple element, None if bare tensor
    shape: tuple
    dtype: str

    def __repr__(self) -> str:
        where = "" if self.took_index is None else f"[{self.took_index}]"
        return f"Captured({self.name}{where}, {tuple(self.shape)}, {self.dtype})"


@dataclass
class Capture:
    """An ordered record of every hooked module's output.

    Ordered by module name at registration, not by execution order -- execution
    order is not observable from hooks alone and pretending otherwise would
    invite reading a residual stream in the wrong sequence.
    """
    data: "OrderedDict[str, Captured]" = field(default_factory=OrderedDict)
    misses: list[str] = field(default_factory=list)

    def __getitem__(self, name: str) -> Any:
        return self.data[name].value

    def __contains__(self, name: str) -> bool:
        return name in self.data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def names(self) -> list[str]:
        return list(self.data)

    def layers(self, pattern: str) -> list[Any]:
        """Values for every captured name matching `pattern`, in layer order."""
        rx = re.compile(pattern)
        hits = [(n, c) for n, c in self.data.items() if rx.search(n)]
        hits.sort(key=lambda nc: _layer_index(nc[0]))
        return [c.value for _, c in hits]

    def summary(self) -> list[dict]:
        return [{"name": c.name, "took_index": c.took_index,
                 "shape": list(c.shape), "dtype": c.dtype}
                for c in self.data.values()]


def _layer_index(name: str) -> tuple:
    """Sort key ordering `layers.2` before `layers.10`, within one stack.

    Keyed on the name with its digits removed first, so `encoder_blocks.N` and
    `decoder_blocks.N` do not interleave. Sorting on the numbers alone put
    `decoder_blocks.0` ahead of `encoder_blocks.0`, which reads a decoder state
    as if it came before the encoder that produced its cross-attention input.
    """
    template = re.sub(r"\d+", "#", name)
    parts = [int(p) for p in re.findall(r"\d+", name)]
    return (template, tuple(parts), name)


def _first_tensor(out: Any) -> tuple[Any, int | None]:
    """(tensor, index) -- the hidden state and where it was found.

    Blocks return tuples whose layout varies. Rather than assume element 0, this
    finds the first tensor and reports its position, so a capture from an
    unfamiliar module carries the evidence of what was actually read.
    """
    import torch
    if isinstance(out, torch.Tensor):
        return out, None
    if isinstance(out, (tuple, list)):
        for i, item in enumerate(out):
            if isinstance(item, torch.Tensor):
                return item, i
    if isinstance(out, dict):
        for i, (_, item) in enumerate(out.items()):
            if isinstance(item, torch.Tensor):
                return item, i
    raise TypeError(f"no tensor in module output of type {type(out).__name__}")


def match_modules(model, patterns: tuple[str, ...] | list[str]) -> list[str]:
    """Module names matching the patterns, grouped in the order given.

    Pattern order is the declared read order -- `RESIDUAL_PATTERNS` lists the
    encoder stack before the decoder stack because that is the order the model
    computes them. A single global sort cannot express that, and got it wrong.
    Within a pattern, names are ordered numerically.
    """
    all_names = [n for n, _ in model.named_modules() if n]
    out: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        rx = re.compile(p)
        hits = sorted((n for n in all_names if rx.search(n) and n not in seen),
                      key=_layer_index)
        out.extend(hits)
        seen.update(hits)
    return out


def residual_points(model) -> list[str]:
    """Comparable residual read positions for this architecture."""
    cls = type(model).__name__
    if cls not in RESIDUAL_PATTERNS:
        raise KeyError(
            f"no residual read positions registered for {cls!r}. Add them to "
            f"RESIDUAL_PATTERNS rather than passing raw patterns at the call "
            f"site, so every experiment reads the same positions.")
    return match_modules(model, RESIDUAL_PATTERNS[cls])


def attention_points(model) -> list[str]:
    cls = type(model).__name__
    if cls not in ATTENTION_PATTERNS:
        raise KeyError(f"no attention positions registered for {cls!r}")
    return match_modules(model, ATTENTION_PATTERNS[cls])


@contextmanager
def capture(model, names: list[str] | None = None,
            patterns: tuple[str, ...] | list[str] | None = None,
            to_cpu: bool = True, dtype: str | None = None,
            keep_all_calls: bool = False) -> Iterator[Capture]:
    """Hook `names` (or everything matching `patterns`) for the enclosed forward.

    Yields a `Capture` that fills in as the model runs. Hooks are removed on
    exit under every path, including an exception inside the block -- a leaked
    hook would silently contaminate every later forward in the process.

    A module called more than once per forward (a diffusion model unrolled over
    steps, beam search over positions) overwrites its entry by default, so the
    capture holds the LAST call. `keep_all_calls=True` keeps every call in a
    list instead; the two are different questions and neither is a safe default
    for the other, so the flag is explicit.
    """
    import torch

    if names is None:
        names = (match_modules(model, patterns) if patterns
                 else residual_points(model))
    wanted = list(dict.fromkeys(names))
    by_name = dict(model.named_modules())
    cap = Capture()
    cap.misses = [n for n in wanted if n not in by_name]
    if cap.misses:
        raise KeyError(
            f"{len(cap.misses)} requested module(s) do not exist on "
            f"{type(model).__name__}: {cap.misses[:5]}. A capture that silently "
            f"skipped them would look like it worked and return nothing.")

    calls: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            tensor, idx = _first_tensor(out)
            v = tensor.detach()
            if to_cpu:
                v = v.cpu()
            if dtype is not None:
                v = v.to(getattr(torch, dtype))
            rec = Captured(name, v, idx, tuple(v.shape), str(v.dtype))
            calls[name] = calls.get(name, 0) + 1
            if keep_all_calls:
                if name in cap.data:
                    cap.data[name].value.append(v)
                else:
                    cap.data[name] = Captured(name, [v], idx, tuple(v.shape),
                                              str(v.dtype))
            else:
                cap.data[name] = rec
        return hook

    try:
        for n in wanted:
            handles.append(by_name[n].register_forward_hook(make_hook(n)))
        yield cap
    finally:
        for h in handles:
            h.remove()
        cap.call_counts = dict(calls)


def run_capture(model, forward_fn, names: list[str] | None = None,
                patterns: tuple[str, ...] | list[str] | None = None,
                to_cpu: bool = True, dtype: str | None = None,
                keep_all_calls: bool = False):
    """(forward output, Capture) for one no-grad forward.

    `torch.no_grad` is not optional here: keeping the graph alive for a 28-layer
    capture costs more memory than the activations themselves, and nothing that
    reads activations wants gradients.
    """
    import torch
    with capture(model, names=names, patterns=patterns, to_cpu=to_cpu,
                 dtype=dtype, keep_all_calls=keep_all_calls) as cap:
        with torch.no_grad():
            out = forward_fn()
    return out, cap
