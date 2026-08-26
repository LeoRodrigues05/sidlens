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
