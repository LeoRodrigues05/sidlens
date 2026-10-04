"""Paired cluster bootstrap over users, shared by every estimand of an experiment.

Every SidLens estimand here is a ratio of per-user sums: a mean over rows is
sum(x) / sum(1), a recovery is sum(num) / sum(den), a ratio of means has four
sums. One (draws x users) matrix of resampling counts then serves them all, so
intervals of different estimands (conditions, cells, digits) come from the SAME
resampled cohorts, which is what "all rows, conditions and cells of a sampled
user are kept together" requires, and differences between estimands can be
bootstrapped by `paired_difference` without losing that pairing.

Trap: a user who contributes no rows to one estimand still counts in the
resample; with a per-user weight vector that is automatic (their sums are 0),
but it means n_users in a result is the number with rows, not the cohort size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class UserBootstrap:
    def __init__(self, users, draws: int = 2000, seed: int = 20260927):
        self.users = np.sort(np.unique(np.asarray(users)))
        self.index = pd.Series(np.arange(len(self.users)), index=self.users)
        rng = np.random.default_rng(seed)
        n = len(self.users)
        self.W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]).astype(float)
        self.draws, self.seed = draws, seed

    def _sum(self, user, x) -> np.ndarray:
        u = self.index.loc[np.asarray(user)].to_numpy()
        return np.bincount(u, weights=np.asarray(x, dtype=float), minlength=len(self.users))

    def _count(self, user) -> np.ndarray:
        u = self.index.loc[np.asarray(user)].to_numpy()
        return np.bincount(u, minlength=len(self.users)).astype(float)

    @staticmethod
    def _summ(est, bs, n_rows, n_users) -> dict:
        bs = bs[np.isfinite(bs)]
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan))
        return {"est": float(est), "lo": float(lo), "hi": float(hi), "n_rows": int(n_rows),
                "n_users": int(n_users)}

    def ratio_draws(self, user, num, den):
        nu, de = self._sum(user, num), self._sum(user, den)
        with np.errstate(invalid="ignore", divide="ignore"):
            return nu.sum() / de.sum(), (self.W @ nu) / (self.W @ de), (de != 0).sum()

    def ratio(self, user, num, den) -> dict:
        est, bs, nu = self.ratio_draws(user, num, den)
        return self._summ(est, bs, len(np.asarray(num)), nu)

    def mean_draws(self, user, x):
        s, c = self._sum(user, x), self._count(user)
        with np.errstate(invalid="ignore", divide="ignore"):
            return s.sum() / c.sum(), (self.W @ s) / (self.W @ c), (c != 0).sum()

    def mean(self, user, x) -> dict:
        est, bs, nu = self.mean_draws(user, x)
        return self._summ(est, bs, len(np.asarray(x)), nu)

    def paired_difference(self, a: tuple, b: tuple) -> dict:
        """mean(xa) - mean(xb) over the same resamples; a, b = (user, x)."""
        ea, ba, na = self.mean_draws(*a)
        eb, bb, nb = self.mean_draws(*b)
        return self._summ(ea - eb, ba - bb, len(np.asarray(a[1])) + len(np.asarray(b[1])), max(na, nb))

    def ratio_of_means(self, a: tuple, b: tuple) -> dict:
        """mean(xa) / mean(xb) with both row counts resampled with the users."""
        ea, ba, na = self.mean_draws(*a)
        eb, bb, _ = self.mean_draws(*b)
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._summ(ea / eb, ba / bb, len(np.asarray(a[1])), na)
