"""The --source/--catalog options of semantic_mapping.build and the explorer's provenance hooks.

Everything except the last test is CPU-only and needs no `$SIDLENS_WORK`: it
checks the argument gate, the generic examples and the template placeholders on
synthetic tables. The last test builds a real one-variant v3 explorer and skips
when the v3 download or the labels are absent.
"""
import argparse
import json
from pathlib import Path

import pytest

from sidlens import paths
from sidlens.analysis import semantic_mapping as sm
from sidlens.data import codebooks_v3
from sidlens.data.sids import SidTable, SidVariant
from sidlens.viz import semantic_explorer as se


def _args(out, **kw):
    base = dict(variants=["all"], out=str(out), n_perm=1, seed=0, source="frozen", catalog="industrial")
    return argparse.Namespace(**{**base, **kw})


@pytest.mark.parametrize("source,catalog", [("frozen", "industrial"), ("v3", "industrial"), ("v3", "office")])
def test_valid_source_catalog_pairs(source, catalog):
    sm.check_source_args(source, catalog)


@pytest.mark.parametrize("source,catalog,match", [
    ("frozen", "office", "Industrial tables only"),
    ("v4", "industrial", "--source must be one of"),
    ("v3", "books", "--catalog must be one of"),
])
def test_invalid_source_catalog_pairs_raise(source, catalog, match):
    with pytest.raises(ValueError, match=match):
        sm.check_source_args(source, catalog)


def test_build_validates_before_creating_the_output_directory(tmp_path):
    # A bad invocation must not leave a `running` directory that blocks the
    # corrected rerun (existing result directories are never overwritten).
    out = tmp_path / "result"
    with pytest.raises(ValueError, match="Industrial tables only"):
        sm.build(_args(out, source="frozen", catalog="office"))
    assert not out.exists()


def test_cli_rejects_frozen_office(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        sm.main(["build", "--source", "frozen", "--catalog", "office", "--out", str(tmp_path / "r")])
    assert exit_info.value.code == 2
    assert "needs --source v3" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


def test_cli_defaults_stay_frozen_industrial(monkeypatch):
    seen = {}
    monkeypatch.setattr(sm, "build", lambda a: seen.update(vars(a)))
    sm.main(["build", "--out", "unused"])
    assert seen["source"] == "frozen" and seen["catalog"] == "industrial"


def _table():
    codes = {"A": (1, 0), "B": (1, 1), "C": (1, 1), "D": (2, 0), "E": (2, 1), "F": (3, 0), "G": (0, 0)}
    return SidTable(SidVariant("rqkmeans", 2, 128), codes, Path("synthetic.sem_ids"))


def _by_asin(table):
    cat = {"A": "x", "B": "x", "C": "y", "D": "y", "E": "y", "F": "", "G": "z"}
    return {a: {"asin": a, "title": f"item {a}", "cat_l1": "top", "cat_l2": cat[a], "cat_l3": ""}
            for a in table.keys}


def test_generic_examples_are_the_largest_first_digit_groups_ties_by_code():
    table = _table()
    ex = sm.generic_examples([table], _by_asin(table), "cat_l2", variant="rqkmeans_2codebook_128", k=3)
    # sizes: (1,)=3, (2,)=2, then the singletons (0,), (3,) tie and sort by code
    assert [e["prefix"] for e in ex] == [[1], [2], [0]]
    assert all(e["variant"] == "rqkmeans_2codebook_128" for e in ex)
    assert "3 items" in ex[0]["label"] and "most common cat_l2: x" in ex[0]["label"]
    # a group with no label in the field says nothing rather than inventing one
    assert "most common" not in sm.generic_examples([table], _by_asin(table), "cat_l3",
                                                    variant="rqkmeans_2codebook_128", k=1)[0]["label"]
    # a table set without the variant yields no examples, not another variant's
    assert sm.generic_examples([table], _by_asin(table), "cat_l2") == []


def test_group_rows_default_fields_are_the_frozen_ones_and_can_be_overridden():
    table = _table()
    by_asin = _by_asin(table)
    default = next(sm.group_rows(table, by_asin))
    assert "cat_l1_purity" in default and "cat_l2_purity" in default and "cat_l3_purity" not in default
    custom = next(sm.group_rows(table, by_asin, ("cat_l2", "cat_l3")))
    assert "cat_l3_purity" in custom and "cat_l1_purity" not in custom
    assert custom["cat_l3_purity"] is None          # nothing labelled: undefined, not perfect


def test_render_examples_default_text_and_custom_examples():
    table = _table()
    by_asin = _by_asin(table)
    # No example variant is in this table set: the intro is still the original sentence.
    default = sm.render_examples([table], by_asin)
    assert default.startswith("# Concrete SID prefix mappings\n\nThese examples were selected after "
                              "inspecting the catalogue to illustrate useful and imperfect groupings. "
                              "They are exploratory, not a random sample. Every item in each selected "
                              "first-digit group is listed below; the full exports cover every group.")
    examples = sm.generic_examples([table], by_asin, "cat_l2", variant="rqkmeans_2codebook_128", k=1)
    text = sm.render_examples([table], by_asin, examples, "Chosen by size.")
    assert "Chosen by size. Every item in each selected" in text
    assert "**All 3 matching items:**" in text and "| (1)(1) | B | item B | x |" in text


def test_template_has_each_placeholder_exactly_once():
    for placeholder in ("__SIDLENS_PAYLOAD__", "__SIDLENS_STAT_LABEL__", "__SIDLENS_SOURCE_TAG__"):
        assert se._HTML.count(placeholder) == 1, placeholder


def _payload(**extra):
    return {"category": "Office", "items": [{"asin": "A", "title": "t", "brand": "", "item_id": 0,
                                              "cat_l1": "", "cat_l2": "", "cat_l3": ""}],
            "variants": {"MQ_3codebook_128": {"codes": [[1, 2, 3]], "summary": []}}, **extra}


def test_render_explorer_defaults_keep_the_original_text():
    html = se.render_explorer(_payload())
    assert "__SIDLENS_" not in html
    assert "have the same category label." in html
    assert "codebook variants. Follow a prefix to inspect its items.'" in html


def test_render_explorer_names_summary_field_and_source():
    html = se.render_explorer(_payload(summary_field="cat_l2", source_label="Assignments: v3 tables (a/b @ 9733eda7)."))
    assert "have the same category (cat_l2) label." in html
    assert "codebook variants. Assignments: v3 tables (a/b @ 9733eda7). Follow a prefix" in html


@pytest.mark.parametrize("key,bad", [("source_label", "x'; alert(1); '"), ("source_label", "</script>"),
                                      ("summary_field", "cat_l2'")])
def test_render_explorer_refuses_template_text_that_could_break_out(key, bad):
    with pytest.raises(ValueError, match="plain text"):
        se.render_explorer(_payload(**{key: bad}))


def test_render_explorer_leaves_catalogue_text_alone():
    # A title that spells a placeholder or closes the script must come out as data.
    payload = _payload()
    payload["items"][0]["title"] = "__SIDLENS_STAT_LABEL__ </script><b>"
    html = se.render_explorer(payload)
    assert "__SIDLENS_STAT_LABEL__ \\u003c/script\\u003e\\u003cb\\u003e" in html
    assert "</script><b>" not in html


@pytest.mark.skipif(
    not ((codebooks_v3.ROOT / "industrial" / "tokenizers" / "tables.json").exists()
         and (paths.EXTERNAL / "labels" / "Industrial_and_Scientific.labels.json").exists()
         and (paths.FROZEN_DATA / "item_meta" / "Industrial_and_Scientific.item.json").exists()),
    reason="needs the v3 tables, labels and frozen item metadata under $SIDLENS_WORK")
def test_v3_industrial_one_variant_build(tmp_path):
    out = tmp_path / "v3"
    name = "rqkmeans_3codebook_128"
    sm.build(_args(out, source="v3", catalog="industrial", variants=[name], n_perm=2))
    validation = json.loads((out / "validation.json").read_text())
    assert validation["source"] == "v3" and validation["revision"] == codebooks_v3.REVISION
    assert validation["coherence_fields"] == ["cat_l1", "cat_l2"]
    assert validation["all_items_preserved"] and validation["sqlite_integrity"] == "ok"
    data = json.loads((out / "data.json").read_text())
    table = codebooks_v3.load_table("industrial", name)
    # Item-index order, i.e. the embedding row order, and the v3 (not frozen) codes.
    assert [i["asin"] for i in data["items"]] == table.keys
    assert data["variants"][name]["codes"] == [list(table.asin2codes[a]) for a in table.keys]
    assert [i["item_id"] for i in data["items"]] == list(range(len(table.keys)))
    assert data["source"] == "v3" and data["summary_field"] == "cat_l1"
    assert data["examples"] and all(e["variant"] == name for e in data["examples"])
    assert not any(e["label"] == "3D-printer filaments" for e in data["examples"])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["v3"]["revision"] == codebooks_v3.REVISION
    assert any(p.endswith("done.json") for p in manifest["input_hashes"])
    assert any(p.endswith("codebooks_v3.py") for p in manifest["source_hashes"])
    readme = (out / "README.md").read_text()
    assert "v3 final-run tables" in readme and "frozen repaired tables" not in readme
    html = (out / "explorer.html").read_text()
    assert "have the same category (cat_l1) label." in html
    assert "Assignments: v3 final-run tables" in html
