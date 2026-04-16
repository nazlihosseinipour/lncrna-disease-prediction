from __future__ import annotations

from pathlib import Path
import tempfile

from mainfolder.utils.rebuild_disease_mapping import parse_do_obo


def test_parse_do_obo_adds_edge_names():
    obo_text = """
format-version: 1.2

[Term]
id: DOID:1
name: parent disease

[Term]
id: DOID:2
name: child disease
is_a: DOID:1 ! parent disease
""".strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        obo_path = Path(tmpdir) / "mini.obo"
        obo_path.write_text(obo_text, encoding="utf-8")

        terms_df, edges_df = parse_do_obo(obo_path)

    assert list(terms_df.columns) == ["doid", "name", "synonyms", "synonyms_json", "synonym_count"]
    assert list(edges_df.columns) == ["child", "parent", "child_name", "parent_name"]
    assert len(edges_df) == 1
    edge = edges_df.iloc[0]
    assert edge["child"] == "DOID:2"
    assert edge["parent"] == "DOID:1"
    assert edge["child_name"] == "child disease"
    assert edge["parent_name"] == "parent disease"
