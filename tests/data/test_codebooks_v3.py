"""The v3 codebook loader's guards, without the downloaded data."""

from __future__ import annotations

import argparse
import collections
import os
import pickle

import numpy as np
import pytest
import torch

from sidlens.data import codebooks_v3 as V


def test_allowlist_loads_a_plain_checkpoint(tmp_path):
    p = tmp_path / "ok.pth"
    sd = collections.OrderedDict(w=torch.arange(6, dtype=torch.float32).reshape(2, 3))
    torch.save({"state_dict": sd, "args": argparse.Namespace(lr=1e-3), "epoch": 3}, p)
    ck = V.safe_torch_load(p)
    assert ck["args"] == {"lr": 1e-3}
    assert torch.equal(ck["state_dict"]["w"], sd["w"])


def test_allowlist_refuses_other_globals(tmp_path):
    p = tmp_path / "bad.pth"
    torch.save({"state_dict": {}, "hook": os.getcwd}, p)
    with pytest.raises(pickle.UnpicklingError, match="refusing to load global"):
        V.safe_torch_load(p)


def test_index_keys_must_be_contiguous(tmp_path):
    raw = {"0": ["<a_1>", "<b_2>"], "2": ["<a_0>", "<b_0>"]}
    with pytest.raises(ValueError, match="keys are not exactly"):
        V._codes_from_index(raw, tmp_path / "t.json", 2)


def test_index_digit_letters_must_be_in_order(tmp_path):
    raw = {"0": ["<a_1>", "<c_2>"], "1": ["<a_0>", "<b_0>"]}
    with pytest.raises(ValueError, match="digit letters"):
        V._codes_from_index(raw, tmp_path / "t.json", 2)


def test_index_parses_codes():
    raw = {"0": ["<a_10>", "<b_2>"], "1": ["<a_0>", "<b_127>"]}
    c = V._codes_from_index(raw, None, 2)
    assert c.tolist() == [[10, 2], [0, 127]]


def test_init_audit_tells_fresh_from_trained_weights():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 8)
    bn = torch.nn.BatchNorm1d(8)
    sd = {"encoders.0.0.weight": lin.weight.detach(), "encoders.0.0.bias": lin.bias.detach(),
          "encoders.0.1.weight": bn.weight.detach(), "encoders.0.1.bias": bn.bias.detach(),
          "pos.weight": torch.empty(1, 64).uniform_(-1 / 128, 1 / 128)}
    rows = {r["param"]: r for r in V.mq_init_audit(sd, 128)}
    assert all(r["within_bound"] for r in rows.values())
    sd["encoders.0.0.weight"] = sd["encoders.0.0.weight"] * 3       # left its init range
    sd["encoders.0.1.bias"] = sd["encoders.0.1.bias"] + 0.01         # BatchNorm moved off 0
    rows = {r["param"]: r for r in V.mq_init_audit(sd, 128)}
    assert not rows["encoders.0.0.weight"]["within_bound"]
    assert not rows["encoders.0.1.bias"]["within_bound"]
    assert rows["encoders.0.0.bias"]["within_bound"]


def test_rqvae_nearest_matches_brute_force():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(50, 6)).astype(np.float32)
    cbs = [rng.normal(size=(7, 6)).astype(np.float32) for _ in range(3)]
    idx, res = V.rqvae_nearest(z, cbs)
    r = z.astype(np.float64)
    for l, cb in enumerate(cbs):
        np.testing.assert_allclose(res[l], r, atol=1e-5)
        a = ((r[:, None, :] - cb[None].astype(np.float64)) ** 2).sum(-1).argmin(1)
        assert (a == idx[:, l]).all()
        r = r - cb[a]
