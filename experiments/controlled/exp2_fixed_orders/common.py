"""Declared design and paired-user statistics for the depth-three order sweep."""

from itertools import permutations

import numpy as np

CELL_INDICES = (0, 1, 6, 7, 12, 13)
ORDERS = tuple(permutations(range(3)))
BOOTSTRAP_SEED = 20260915


def conditions(beams):
    beams = tuple(beams)
    if not beams or len(set(beams)) != len(beams) or any(b not in (64, 256) for b in beams):
        raise ValueError("use distinct declared beams 64 and/or 256")
    return [(b, "confidence", None) for b in beams] + [
        (b, "fixed", order) for b in beams for order in ORDERS]


def label(policy, order):
    return policy if order is None else "fixed_" + "".join(str(x) for x in order)


def paired_effects(values, axes, *, n_boot=2000, seed=BOOTSTRAP_SEED):
    """Input [cell, condition, user, metric]; bootstrap users jointly across cells.

    The fixed comparator is each user's average outcome over all six orders.
    Equal cell averaging happens before averaging users. No ensemble is formed.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4 or min(values.shape) < 1 or values.shape[1] != len(axes):
        raise ValueError("expected nonempty [cell,condition,user,metric] values")
    if not np.isfinite(values).all() or n_boot < 1:
        raise ValueError("nonfinite values or invalid bootstrap count")
    beams = sorted({a[0] for a in axes})
    deltas = []
    for beam in beams:
        fixed = [i for i, (b, p, _) in enumerate(axes) if b == beam and p == "fixed"]
        guided = [i for i, (b, p, _) in enumerate(axes) if b == beam and p == "confidence"]
        if len(guided) != 1 or len(fixed) != 6 or {axes[i][2] for i in fixed} != set(ORDERS):
            raise ValueError("every beam requires confidence and six unique fixed orders")
        deltas.append(values[:, guided[0]] - values[:, fixed].mean(axis=1))
    # [user, beam*cell*metric], with one set of bootstrap weights for all axes.
    delta = np.stack(deltas)  # beam,cell,user,metric
    n = values.shape[2]
    matrix = delta.transpose(2, 0, 1, 3).reshape(n, -1)
    draws = np.empty((n_boot, matrix.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, 64):
        count = min(64, n_boot - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=count)
        draws[start:start + count] = weights @ matrix / n
    draws = draws.reshape(n_boot, len(beams), values.shape[0], values.shape[3])
    return {
        "beams": beams,
        "per_cell": delta.mean(axis=2),
        "per_cell_ci": np.quantile(draws, [0.025, 0.975], axis=0),
        "overall": delta.mean(axis=(1, 2)),
        "overall_ci": np.quantile(draws.mean(axis=2), [0.025, 0.975], axis=0),
    }
