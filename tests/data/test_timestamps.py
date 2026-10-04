"""Interaction days: every event aligned, every ambiguity resolved, every join proven.

The frozen review keys are the only record of time, so these tests pin the
facts the temporal analyses rest on: the counts, the resolution of duplicate
days through the global split order, one position per AR window, the DiffGRM
cohort's identity with `inter.json`, and the same-day ASIN ordering that makes
"most recent" within a day an artefact.
"""

import numpy as np
import pytest

from sidlens import paths
from sidlens.data import ar_prompts as P
from sidlens.data import timestamps as T

pytestmark = pytest.mark.skipif(
    not (paths.FROZEN_DATA / "reviews" / f"{T.CATEGORY}.review.json").exists(),
    reason="frozen reviews absent")


@pytest.fixture(scope="module")
def ev():
    return T.load_event_times()


def test_counts_and_days(ev):
    r = ev.report
    assert (r["n_users"], r["n_events"], r["n_review_keys"]) == (6297, 43102, 40449)
    assert r["n_duplicate_events"] == 2653
    assert r["n_next_item_rows"] == 43102 - 6297
    assert all(np.all(t % T.DAY == 0) for t in ev.times.values())
    assert all(np.all(np.diff(t) >= 0) for t in ev.times.values())


def test_ambiguity_is_resolved_not_guessed(ev):
    # 32 users had several alignments; the global target-time order leaves one each.
    assert ev.report["n_users_ambiguous_before_resolution"] == 32
    assert ev.report["n_positions_ambiguous_before_resolution"] == 72


def test_splits_are_consecutive_periods(ev):
    last = {}
    for split in T.SPLIT_ORDER:
        ts = [ev.times[u][T.window_position(ev.sequences[u], h, t)] for u, h, t in T._read_inter(split)]
        assert np.all(np.diff(ts) >= 0), split
        last[split] = (ts[0], ts[-1])
    assert last["train"][1] <= last["valid"][0] and last["valid"][1] <= last["test"][0]


@pytest.mark.parametrize("variant", ["rqkmeans_3codebook_128", "rqvae_4codebook_128"])
def test_ar_rows_join_once(ev, variant):
    ex = P.load_examples(variant, "next-item", "test")
    rows, hist = T.ar_row_times(ex, ev)
    assert len(rows) == 3681 and rows.example_id.is_unique
    assert len(hist) == sum(len(e.history_item_ids) for e in ex)
    assert (hist.gap_days >= 0).all()
    assert (rows.gap1_days == hist[hist.recency == 1].set_index("example_id")
            .loc[rows.example_id, "gap_days"].to_numpy()).all()
    # A duplicate target is always same-day and always a repeat item.
    assert (rows[rows.dup_target].same_day_as_target & rows[rows.dup_target].repeat_item).all()


def test_ar_join_is_variant_independent(ev):
    a, _ = T.ar_row_times(P.load_examples("rqkmeans_3codebook_128", "next-item", "valid"), ev)
    b, _ = T.ar_row_times(P.load_examples("rqvae_4codebook_128", "next-item", "valid"), ev)
    assert (a[["user", "pos", "t_target"]].to_numpy() == b[["user", "pos", "t_target"]].to_numpy()).all()


def test_same_day_order_is_mostly_asin_order(ev):
    """The fact behind trap 3: within a day, sequence order follows the ASIN."""
    from sidlens.data.sids import load_item2id
    asin = {i: a for a, i in load_item2id(T.CATEGORY).items()}
    n = asc = 0
    for u, seq in ev.sequences.items():
        t = ev.times[u]
        for p in range(1, len(seq)):
            if t[p] == t[p - 1] and seq[p] != seq[p - 1]:
                n += 1
                asc += asin[seq[p - 1]] < asin[seq[p]]
    assert n > 10000 and asc / n > 0.85


def test_gap_bins():
    assert list(T.gap_bin([0, 1, 30, 31, 365, 366])) == [
        "same day", "1-30 d", "1-30 d", "31-365 d", "31-365 d", ">365 d"]
    with pytest.raises(ValueError):
        T.gap_bin([-1])


def test_window_position_refuses_non_windows():
    with pytest.raises(ValueError):
        T.window_position((1, 2, 3), (9,), 3)


def test_diffusion_cohort_is_inter_json(ev):
    from sidlens.data.diffusion_eval import load_eval_cohort
    from sidlens.registry.diffusion import load_runtime
    cohort = load_eval_cohort(load_runtime()["diff-next1-rqkmeans-3cb-128"])
    rows, hist = T.diffusion_row_times(cohort, ev)
    assert len(rows) == 6297 and (rows.pos == rows.hist_len).sum() > 0
    assert (hist.gap_days >= 0).all()
