"""Check pair statistics and lossless prefix lookup against independent oracles."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from sidlens.analysis.semantic_mapping import (
    assignment_rows, coherence, make_database, pair_agreement, query_database,
)
from sidlens.data.sids import SidTable, SidVariant


def test_pair_agreement_matches_explicit_pairs():
    groups = np.array([3, 3, 3, 5, 5, 9])
    labels = np.array([0, 0, 1, 1, 1, 0])
    pairs = [(i, j) for i, j in itertools.combinations(range(6), 2) if groups[i] == groups[j]]
    observed, count = pair_agreement(groups, labels)
    assert count == len(pairs) == 4
    assert observed == sum(labels[i] == labels[j] for i, j in pairs) / len(pairs)


def test_singletons_and_missing_labels_are_not_perfect_semantics():
    codes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    rows = coherence(codes, ["a", "a", "b", ""], n_perm=10, seed=8)
    assert rows[0]["n_labelled"] == 3
    assert rows[0]["n_labelled_pairs"] == 1
    assert rows[0]["pair_category_agreement"] == 1
    assert rows[1]["n_labelled_pairs"] == 0
    assert rows[1]["pair_category_agreement"] is None
    assert rows[1]["null_pair_mean"] is None
    assert rows[1]["singleton_item_share"] == 1
    empty = coherence(codes, ["", "", "", ""], n_perm=3)
    assert all(r["pair_category_agreement"] is None for r in empty)


def test_permutation_null_is_reproducible_and_preserves_pair_count():
    codes = np.array([[0], [0], [0], [1], [1], [1]])
    a = coherence(codes, ["x", "x", "x", "y", "y", "y"], 100, 123)
    assert a == coherence(codes, ["x", "x", "x", "y", "y", "y"], 100, 123)
    assert a[0]["n_labelled_pairs"] == 6
    assert a[0]["pair_category_agreement"] == 1
    assert a[0]["catalogue_pair_agreement"] == .4
    assert .3 < a[0]["null_pair_mean"] < .5


def test_sql_and_export_preserve_full_sid_collisions(tmp_path):
    table = SidTable(SidVariant("rqkmeans", 3, 128), {
        "A": (1, 2, 3), "B": (1, 2, 3), "C": (1, 22, 3), "D": (11, 2, 3)
    }, Path("test.sem_ids"))
    items = [{"asin": a, "item_id": i, "title": "Quoted, item \"name\"", "brand": "", "cat_l1": "test", "cat_l2": "", "cat_l3": ""}
             for i, a in enumerate(table.keys)]
    db = tmp_path / "mapping.sqlite"
    result = make_database(db, items, [table])
    assert result["sqlite_integrity"] == "ok"
    assert result["assignment_rows"] == 4
    assert {r["asin"] for r in query_database(db, table.variant.name, (1, 2))} == {"A", "B"}
    assert {r["asin"] for r in query_database(db, table.variant.name, (1,))} == {"A", "B", "C"}
    assert len(query_database(db, table.variant.name, (1, 2, 3))) == 2
    assert query_database(db, table.variant.name, (99,)) == []
    assert len(query_database(db, table.variant.name, contains='item "name"')) == 4
    assert query_database(db, table.variant.name, contains="' OR 1=1 --") == []
    rows = list(assignment_rows(table, items))
    assert len(rows) == 4
    assert rows[0]["sid_shared_by"] == rows[1]["sid_shared_by"] == 2
    assert rows[0]["prefix2"] == "(1)(2)"
    with pytest.raises(ValueError, match="unknown variant"):
        query_database(db, "missing")
