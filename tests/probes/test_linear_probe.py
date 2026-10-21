"""Linear probes: a readable label is read, an unreadable one scores near zero information (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from sidlens.probes import linear as LP                               # noqa: E402


def _data(n=3000, d=12, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, k, (n,), generator=g)
    centers = torch.randn(k, d, generator=g) * 3
    X = centers[y] + torch.randn(n, d, generator=g)
    return X, y


def test_a_separable_label_is_read_and_a_shuffled_one_is_not():
    X, y = _data()
    fit, test = slice(0, 2000), slice(2000, None)
    val = torch.arange(2000) % 10 == 0
    probe, info = LP.fit_selected(X[fit], y[fit], val)
    assert info["lam"] in LP.LAMBDAS and set(info["val_acc"]) == set(LP.LAMBDAS)
    good = LP.evaluate(probe, X[test], y[test], y[fit], n_labels=8)
    assert good["acc"] > 0.95 and good["nig"] > 0.8 and good["correct"].shape == (1000,)
    shuffled = y[fit][torch.randperm(2000, generator=torch.Generator().manual_seed(1))]
    bad = LP.evaluate(LP.fit_selected(X[fit], shuffled, val)[0], X[test], y[test], shuffled, n_labels=8)
    assert bad["acc"] < 0.35 and bad["nig"] < 0.05


def test_unseen_labels_have_finite_cross_entropy_and_missing_labels_are_refused():
    X, y = _data(k=3)
    probe = LP.fit_logistic(X[:500], y[:500], 1e-3)
    y_test = y[500:600].clone()
    y_test[:10] = 7                                                  # a label the probe never saw
    out = LP.evaluate(probe, X[500:600], y_test, y[:500], n_labels=8)
    assert torch.isfinite(torch.tensor(out["ce"])) and out["n_fit_classes"] == 3
    with pytest.raises(ValueError, match="negative"):
        LP.fit_logistic(X[:10], torch.full((10,), -1), 1e-3)
    with pytest.raises(ValueError, match="vocabulary"):
        LP.evaluate(probe, X[:10], torch.full((10,), 9), y[:500], n_labels=8)
