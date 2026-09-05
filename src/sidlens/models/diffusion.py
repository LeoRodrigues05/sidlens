"""Reconstruct a trained DIFF_GRM from a bare state_dict.

Upstream saves `torch.save(model.state_dict())` -- no config, no optimizer.
Rebuilding the model therefore means supplying a config from outside, and the
registry is what supplies it.

Two levels of validation, because they catch different things:

  shape     load_state_dict(strict=True) plus explicit assertions on the
            embedding table and item_mlp. Catches a wrong n_digit, codebook
            size, layer count, or width.

  behavior  re-run the recorded evaluation and compare metrics. This is the
            ONLY check that catches a wrong n_head: `qkv` is Linear(256->768)
            for any head count, so an incorrect n_head reshapes the same
            parameters into a different number of heads and loads silently.

The model only needs `vocab_size` and `sid_offset` from its tokenizer, so a stub
built from the frozen `.sem_ids` is enough to construct it -- no dataset
pipeline, no cache directory, no sentence encoder.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from sidlens import paths, vendorpath
from sidlens.registry.diffusion import DiffGRMConfig


class StubTokenizer:
    """Minimal stand-in for DIFF_GRMTokenizer.

    The model reads exactly two attributes off its tokenizer (`vocab_size` at
    construction, `sid_offset` in the three places that map codes to token ids).
    Everything else in the real tokenizer is dataset plumbing that interp work
    does not need and that would drag in a cache directory and a sentence encoder.
    """

    def __init__(self, config: DiffGRMConfig):
        self.vocab_size = config.vocab_size
        self.sid_offset = config.sid_offset
        self.n_digit = config.n_digit
        self.codebook_size = config.codebook_size
        self.mask_token = -1

    def codebooks_to_item_id(self, tokens):
        raise NotImplementedError(
            "StubTokenizer does not decode. Use sidlens.data.sids.SidTable.cb2items, "
            "which is one-to-many and does not drop colliding items.")


class StubDataset:
    """AbstractModel stores `dataset` but DIFF_GRM never reads it."""
    def __init__(self, config: dict):
        self.config = config


def config_to_dict(config: DiffGRMConfig, training: dict | None = None) -> dict:
    """Translate the typed config into the loose dict the vendored model wants."""
    d = {
        "n_digit": config.n_digit,
        "codebook_size": config.codebook_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_inner": config.n_inner,
        "dropout": config.dropout,
        "attn_pdrop": config.dropout,
        "resid_pdrop": config.dropout,
        "encoder_n_layer": config.encoder_n_layer,
        "decoder_n_layer": config.decoder_n_layer,
        "max_history_len": config.max_history_len,
        "norm_type": "layernorm",
        "norm_eps": config.layer_norm_eps,
        "share_decoder_output_embedding": config.share_decoder_output_embedding,
        "n_target_items": config.n_target_items,
    }
    if training:
        for k in ("masking_strategy", "guided_steps", "guided_conf_metric",
                  "guided_select", "guided_refresh_each_step"):
            if training.get(k) is not None:
                d[k] = training[k]
    return d


def build_model(config: DiffGRMConfig, training: dict | None = None,
                vendor: str = "diffgrm"):
    """Construct an untrained DIFF_GRM matching `config`."""
    vendorpath.activate(vendor)
    from genrec.models.DIFF_GRM.model import DIFF_GRM
    vendorpath.assert_frozen(__import__("genrec.models.DIFF_GRM.model",
                                        fromlist=["model"]))
    cfg = config_to_dict(config, training)
    return DIFF_GRM(cfg, StubDataset(cfg), StubTokenizer(config))


def verify_shapes(model, state_dict: dict, config: DiffGRMConfig) -> dict:
    """Assert the weights and the config describe the same architecture."""
    problems: list[str] = []

    emb = state_dict.get("embedding.weight")
    if emb is None:
        problems.append("state_dict has no embedding.weight")
    else:
        want = (3 + config.n_digit * config.codebook_size, config.n_embd)
        if tuple(emb.shape) != want:
            problems.append(f"embedding.weight {tuple(emb.shape)} != {want}")

    mlp = state_dict.get("item_mlp.0.weight")
    if mlp is not None and mlp.shape[1] != config.n_digit * config.n_embd:
        problems.append(
            f"item_mlp.0.weight in_features {mlp.shape[1]} != "
            f"n_digit*n_embd = {config.n_digit * config.n_embd}")

    mask = state_dict.get("mask_emb_table.weight")
    if mask is not None and mask.shape[0] != config.n_target_digits:
        problems.append(
            f"mask_emb_table rows {mask.shape[0]} != n_target_digits "
            f"{config.n_target_digits}")

    for prefix, n in (("encoder_blocks", config.encoder_n_layer),
                      ("decoder_blocks", config.decoder_n_layer)):
        seen = {int(k.split(".")[1]) for k in state_dict if k.startswith(prefix + ".")}
        if seen and max(seen) + 1 != n:
            problems.append(f"{prefix}: weights have {max(seen)+1} layers, config says {n}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        problems.append(f"missing keys: {sorted(missing)[:5]}")
    if unexpected:
        problems.append(f"unexpected keys: {sorted(unexpected)[:5]}")

    return {
        "ok": not problems,
        "problems": problems,
        "n_params": sum(p.numel() for p in model.parameters()),
        # n_head is invisible to every check above. Recorded so a later
        # behavioral check has something to attribute a failure to.
        "n_head_unverified": config.n_head,
    }


def load(entry: dict, device: str = "cpu", strict: bool = True):
    """Load one registry entry into a runnable model.

    Returns (model, report). `report['ok']` covers shapes only -- see the module
    docstring for why that is not sufficient on its own.
    """
    config = DiffGRMConfig(**entry["config"])
    vendor = "diffgrm" if entry["task"] == "next1" else "diffgrm_new"
    model = build_model(config, entry.get("training"), vendor=vendor)
    sd = torch.load(entry["ckpt_path"], map_location="cpu", weights_only=True)
    report = verify_shapes(model, sd, config)
    if strict and not report["ok"]:
        raise ValueError(
            f"{entry['ckpt_id']}: checkpoint does not match config\n  "
            + "\n  ".join(report["problems"]))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, report


def load_by_id(ckpt_id: str, device: str = "cpu", strict: bool = True):
    """Load by registry id instead of by entry dict.

    `load` takes the entry because that is what carries the config; every call
    site was otherwise re-reading and re-indexing the registry JSON, and one of
    them passed the id straight through and got a confusing TypeError.
    """
    reg = json.loads((paths.MANIFESTS / "registry.diffusion.json").read_text())
    if ckpt_id not in reg:
        raise KeyError(f"unknown diffusion checkpoint {ckpt_id!r}; "
                       f"{len(reg)} known, e.g. {sorted(reg)[:3]}")
    return load(reg[ckpt_id], device=device, strict=strict)


# --------------------------------------------------------------------------
# Runtime: the two-stage path, and per-digit logits in CODE order.
#
# `DIFF_GRM.forward(batch, return_loss=False)` returns after the ENCODER -- it
# is an inference short-circuit, not a full forward. Anything hooking the
# decoder blocks through that call captures nothing and looks like it worked.
# The decoder is reached only via `forward_decoder_only`, so the two stages are
# exposed separately here and every interp path goes through them.
#
# Token layout differs from the AR side and the difference is silent:
#
#     DiffGRM   id = sid_offset + digit * K + code      contiguous, CODE order
#     AR        id read from added_tokens.json          STRING-SORT order
#
# So `digit_logits` below and `ar.digit_logits` both return code-ordered
# columns, and no cross-paradigm comparison should index either model's
# vocabulary directly.
# --------------------------------------------------------------------------

def sid_token_id(digit: int, code: int, n_codebook: int, codebook_size: int,
                 sid_offset: int = 3) -> int:
    """Token id for one (digit, code) under the DiffGRM layout."""
    if not 0 <= digit < n_codebook:
        raise ValueError(f"digit {digit} outside [0,{n_codebook})")
    if not 0 <= code < codebook_size:
        raise ValueError(f"code {code} outside [0,{codebook_size})")
    return sid_offset + digit * codebook_size + code


def encode(model, history_sid, history_mask=None):
    """Run the encoder only. `history_sid` is [B, S, n_digit] of codebook ids.

    PAD is -1, matching the vendor's own assertion. Passing offset token ids
    here instead of raw codes trips that assertion, which is the intended
    behaviour -- the two id spaces must not be mixed.
    """
    batch = {"history_sid": history_sid}
    if history_mask is not None:
        batch["history_mask"] = history_mask
    with torch.no_grad():
        return model(batch, return_loss=False).hidden_states


def digit_logits(model, encoder_hidden, decoder_input_ids, mask_positions=None):
    """[B, n_digit, K] logits, column k being codeword k.

    `decoder_input_ids` holds revealed codes (0 where masked) and
    `mask_positions` marks which digits are still to be predicted -- the same
    convention the vendor's guided path uses, so a capture taken here sits on
    the model's real decoding trajectory rather than a reconstruction of it.
    """
    if mask_positions is None:
        mask_positions = torch.ones_like(decoder_input_ids, dtype=torch.float)
    with torch.no_grad():
        out = model.forward_decoder_only(
            {"decoder_input_ids": decoder_input_ids,
             "encoder_hidden": encoder_hidden,
             "mask_positions": mask_positions},
            return_loss=False, digit=None, use_cache=False)
    return out.logits
