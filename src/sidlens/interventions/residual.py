"""Residual-stream patching: overwrite a block's output at (row, position) pairs.

Activation patching asks where information lives: run input X, but at block L
and positions P substitute the activation that input Y produced there, and see
how much of Y's output comes back. This module does the substitution and
nothing else; choosing X, Y, L and P is the experiment's job.

Traps, each with the guard that closes it
-----------------------------------------
1. **A patch that never fires looks like a null result.** DiffGRM's decoder is
   not reached by `forward()`; a hook there is silently skipped and "patching
   changed nothing" would be reported as a finding. Every patched site must
   fire exactly `expect_calls` times inside the context, or exit raises.
2. **Sites that are not residual read points.** Patching an attention
   submodule's output is a different intervention (it bypasses the residual
   add). Only names from `hooks.residual_points(model)` are accepted, so a
   "layer 12 patch" always means the residual stream after block 12.
3. **Ambiguous writes.** Two patches targeting the same (row, position) at one
   site would make the result depend on concatenation order. Duplicates raise.
4. **Out-of-range indices.** Advanced indexing past the batch or sequence
   raises on GPU only asynchronously, far from the cause. Bounds are checked
   in the hook before the write.
5. **Tuple outputs.** Some transformers versions return `(hidden, ...)` from a
   decoder layer and others a bare tensor (4.57 does the latter). The hook
   finds the hidden state the same way `hooks.capture` does and rebuilds the
   container, so an upgrade cannot turn the patch into a no-op.
6. **In-place mutation of a captured source.** The hook writes into a clone of
   the block output, never into the tensor another hook or cache holds.

Values are cast to the block output's dtype and device. When the source
activation came from a forward in the same dtype this is the identity, which is
what makes the self-patch and full-restore controls exact.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Patch:
    """Write `values[i]` at (`rows[i]`, `pos[i]`) of `site`'s output.

    `pos` indexes the padded sequence axis of the batch actually fed to the
    model (with right padding, the unpadded position).
    """
    site: str
    rows: Any       # (k,) long tensor
    pos: Any        # (k,) long tensor
    values: Any     # (k, width) tensor

    def __post_init__(self) -> None:
        if self.rows.ndim != 1 or self.pos.ndim != 1 or len(self.rows) != len(self.pos):
            raise ValueError(f"{self.site}: rows and pos must be 1-d and equally long")
        if self.values.ndim != 2 or self.values.shape[0] != len(self.rows):
            raise ValueError(f"{self.site}: values must be (k, width) with k = {len(self.rows)}, "
                             f"got {tuple(self.values.shape)}")


def _merge(patches: list[Patch]) -> dict[str, tuple]:
    import torch

    by_site: dict[str, list[Patch]] = {}
    for p in patches:
        by_site.setdefault(p.site, []).append(p)
    out = {}
    for site, ps in by_site.items():
        rows = torch.cat([p.rows.long().cpu() for p in ps])
        pos = torch.cat([p.pos.long().cpu() for p in ps])
        widths = {int(p.values.shape[1]) for p in ps}
        if len(widths) != 1:
            raise ValueError(f"{site}: patches disagree on width {sorted(widths)}")
        key = rows * (int(pos.max()) + 1) + pos
        if len(torch.unique(key)) != len(key):
            raise ValueError(f"{site}: two patches write the same (row, position); "
                             f"the result would depend on their order")
        dev = ps[0].values.device
        vals = torch.cat([p.values.to(dev) for p in ps])
        out[site] = (rows, pos, vals)
    return out


@contextmanager
def patch_residual(model, patches: list[Patch], expect_calls: int = 1) -> Iterator[dict]:
    """Apply `patches` to every forward inside the context.

    Yields a dict site -> call count that fills in as the model runs. On a
    clean exit every patched site must have fired exactly `expect_calls` times.
    Hooks are removed under every exit path.
    """
    import torch
    from sidlens.hooks import _first_tensor, residual_points

    known = set(residual_points(model))
    if unknown := sorted({p.site for p in patches} - known):
        raise KeyError(f"not residual read points of {type(model).__name__}: {unknown[:5]}. "
                       f"Patch only the residual stream here; other sites are other "
                       f"interventions.")
    merged = _merge(patches)
    modules = dict(model.named_modules())
    fired = {s: 0 for s in merged}
    handles = []

    def make_hook(site: str):
        rows, pos, vals = merged[site]

        def hook(_mod, _inp, out):
            fired[site] += 1
            t, idx = _first_tensor(out)
            if t.ndim != 3:
                raise ValueError(f"{site}: expected a (batch, seq, width) output, got {tuple(t.shape)}")
            if int(rows.max()) >= t.shape[0] or int(pos.max()) >= t.shape[1] \
                    or int(rows.min()) < 0 or int(pos.min()) < 0:
                raise IndexError(f"{site}: patch index outside output {tuple(t.shape)}")
            if vals.shape[1] != t.shape[2]:
                raise ValueError(f"{site}: patch width {vals.shape[1]} != output width {t.shape[2]}")
            new = t.clone()
            new[rows.to(t.device), pos.to(t.device)] = vals.to(device=t.device, dtype=t.dtype)
            if idx is None:
                return new
            items = list(out)
            items[idx] = new
            return type(out)(items) if isinstance(out, tuple) else items
        return hook

    try:
        for site in merged:
            handles.append(modules[site].register_forward_hook(make_hook(site)))
        yield fired
    finally:
        for h in handles:
            h.remove()
    if bad := {s: n for s, n in fired.items() if n != expect_calls}:
        raise RuntimeError(f"patched sites fired {bad} times, expected {expect_calls} each. "
                           f"A patch that did not fire is a silent no-op, not a null result.")


def run_patched(model, forward_fn, patches: list[Patch], expect_calls: int = 1):
    """forward_fn() under `patches` and `torch.no_grad`; returns its output."""
    import torch
    with patch_residual(model, patches, expect_calls=expect_calls):
        with torch.no_grad():
            return forward_fn()
