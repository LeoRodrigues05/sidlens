"""Multinomial linear probes with a held-out penalty choice and an information score.

A probe here answers "is label y linearly readable from activation x, on rows
the probe never saw". It is the full-batch L-BFGS logistic regression of
representation/exp1, packaged so every experiment fits and scores probes the
same way.

Traps, each with the guard that closes it
-----------------------------------------
1. **Choosing the penalty on the evaluation rows.** `fit_selected` picks the L2
   penalty on a validation subset of the FIT rows (the caller passes it, split
   by user), then refits on all fit rows. Evaluation rows are never seen.
2. **Accuracy that a constant would get.** With 128-512 classes and a skewed
   label, accuracy alone hides the base rate. `evaluate` reports the majority
   accuracy of the fit labels and a normalized information gain
   1 - CE(probe) / CE(prior), with the prior the fit-label frequencies.
3. **Classes the probe never saw.** A test label absent from the fit rows has
   probe probability 0 and infinite cross-entropy. Both the probe and the prior
   are mixed with eps of a uniform over the full label vocabulary, so every
   label has finite CE and the two are scored on equal terms.
4. **Unscaled features.** Inputs are standardized with the fit rows' mean and
   std (floored at 1e-6), stored with the weights, and applied to any later X.
5. **Missing labels.** Rows with a negative label are refused, not dropped:
   the caller filters, so the row count it reports is the one scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

LAMBDAS = (1e-4, 1e-3, 1e-2)
EPS = 1e-3


@dataclass
class LinearProbe:
    mu: Tensor
    sd: Tensor
    W: Tensor
    b: Tensor
    classes: Tensor
    lam: float

    def logits(self, X: Tensor) -> Tensor:
        return ((X.float() - self.mu) / self.sd) @ self.W + self.b

    def predict(self, X: Tensor) -> Tensor:
        return self.classes[self.logits(X).argmax(1)]

    def label_log_probs(self, X: Tensor, n_labels: int) -> Tensor:
        """[n, n_labels] log-probabilities over the full label vocabulary (trap 3)."""
        p = torch.softmax(self.logits(X), dim=1)
        full = torch.zeros((X.shape[0], n_labels), device=p.device, dtype=p.dtype)
        full[:, self.classes] = p
        return torch.log((1 - EPS) * full + EPS / n_labels)


def _check(X: Tensor, y: Tensor) -> None:
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(f"need X [n, d] and y [n], got {tuple(X.shape)} and {tuple(y.shape)}")
    if X.shape[0] == 0:
        raise ValueError("no rows")
    if bool((y < 0).any()):
        raise ValueError("negative labels: filter missing labels before fitting or scoring")


def fit_logistic(X: Tensor, y: Tensor, lam: float, iters: int = 300) -> LinearProbe:
    """Standardized multinomial logistic regression with penalty lam * ||W||^2."""
    _check(X, y)
    classes, yi = torch.unique(y, return_inverse=True)
    X = X.float()
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    A = (X - mu) / sd
    W = torch.zeros(A.shape[1], len(classes), device=A.device, requires_grad=True)
    b = torch.zeros(len(classes), device=A.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-10)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(A @ W + b, yi) + lam * (W ** 2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return LinearProbe(mu, sd, W.detach(), b.detach(), classes, float(lam))


def fit_selected(X: Tensor, y: Tensor, val: Tensor, lams=LAMBDAS, iters: int = 300) -> tuple[LinearProbe, dict]:
    """Pick lam by accuracy on the `val` rows of the fit set, then refit on all of it.

    Ties go to the stronger penalty. Returns the probe and the validation accuracies.
    """
    _check(X, y)
    val = val.to(torch.bool)
    if not bool(val.any()) or bool(val.all()):
        raise ValueError("the validation mask must select some, but not all, fit rows")
    scores = {}
    for lam in lams:
        p = fit_logistic(X[~val], y[~val], lam, iters).predict(X[val])
        scores[float(lam)] = float((p == y[val]).float().mean())
    best = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return fit_logistic(X, y, best, iters), {"lam": best, "val_acc": scores}


def evaluate(probe: LinearProbe, X: Tensor, y: Tensor, y_fit: Tensor, n_labels: int) -> dict:
    """Accuracy, majority accuracy and information gain on evaluation rows.

    ``y_fit`` are the labels the probe was fitted on (for the majority class and
    the prior); ``n_labels`` is the size of the label vocabulary. ``correct`` is
    the per-row hit vector, for a user bootstrap.
    """
    _check(X, y)
    if int(y.max()) >= n_labels or int(y_fit.max()) >= n_labels:
        raise ValueError("a label is outside the vocabulary")
    correct = probe.predict(X) == y
    counts = torch.bincount(y_fit, minlength=n_labels).float()
    prior = (counts + 1) / (counts.sum() + n_labels)
    log_prior = torch.log((1 - EPS) * prior + EPS / n_labels)
    ce = float(-probe.label_log_probs(X, n_labels).gather(1, y[:, None]).mean())
    ce_prior = float(-log_prior[y].mean())
    majority = int(counts.argmax())
    return {"acc": float(correct.float().mean()), "majority_acc": float((y == majority).float().mean()),
            "ce": ce, "ce_prior": ce_prior, "nig": 1 - ce / ce_prior, "n": int(len(y)),
            "n_fit": int(len(y_fit)), "n_fit_classes": int(len(probe.classes)), "correct": correct}
