"""The shared user bootstrap: point estimates equal the plain formulas, and pairing holds."""

import numpy as np
import pandas as pd

from sidlens.analysis.bootstrap import UserBootstrap


def test_point_estimates_and_pairing():
    rng = np.random.default_rng(0)
    users = np.repeat([f"u{i}" for i in range(50)], 4)
    x = pd.Series(rng.normal(size=200))
    y = x + 1.0
    b = UserBootstrap(users, draws=300, seed=1)
    assert np.isclose(b.mean(users, x)["est"], x.mean())
    assert np.isclose(b.ratio(users, x, np.ones(200))["est"], x.mean())
    d = b.paired_difference((users, y), (users, x))
    assert np.isclose(d["est"], 1.0) and np.isclose(d["lo"], 1.0) and np.isclose(d["hi"], 1.0)
    r = b.ratio_of_means((users, 2 * y), (users, y))
    assert np.isclose(r["est"], 2.0) and np.isclose(r["lo"], 2.0) and np.isclose(r["hi"], 2.0)
    lo, hi = b.mean(users, x)["lo"], b.mean(users, x)["hi"]
    assert lo < x.mean() < hi
