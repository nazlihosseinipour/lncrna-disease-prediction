from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit_shared_disease_map import audit, main, write_outputs


def _write(tmp_path, *, v1_extra=None, v2_extra=None, map_rows=None, v2_y_mutate=None):
    v1 = pd.DataFrame({
        "ID": ["a", "b", "c", "d"],
        "seqs": ["AC", "GU", "AA", "CC"],
        "Lung Cancer": [1, 0, 0, 0],
        "lung-cancer": [0, 1, 0, 0],
        "Glioma": [0, 0, 1, 0],
        "V1 only": [0, 0, 0, 1],
    })
    if v1_extra:
        for k, v in v1_extra.items():
            v1[k] = v
    v1.index = range(len(v1))
    v1_path = tmp_path / "v1.csv"
    v1.to_csv(v1_path)  # leading unnamed index column like the real file

    v2 = pd.DataFrame({
        "ID": ["x", "y", "z", "w", "007"],
        "seqs": ["A", "C", "G", "U", "A"],
        "lung cancer": [1, 1, 0, 0, 1],
        "glioma": [0, 0, 1, 0, 0],
        "Gliomas": [0, 0, 0, 1, 0],
    })
    if v2_extra:
        for k, v in v2_extra.items():
            v2[k] = v
    v2_path = tmp_path / "v2.csv"
    v2.to_csv(v2_path, index=False)

    rows = map_rows or [
        {"order": 0, "canonical_name": "glioma", "v1_source_columns": "Glioma",
         "v2_source_columns": "glioma", "v1_positives": 1, "v2_positives": 1},
        {"order": 1, "canonical_name": "lung cancer", "v1_source_columns": "Lung Cancer | lung-cancer",
         "v2_source_columns": "lung cancer", "v1_positives": 2, "v2_positives": 2},
    ]
    map_path = tmp_path / "map.csv"
    pd.DataFrame(rows).to_csv(map_path, index=False)

    v1_y = pd.DataFrame({"glioma": [0, 0, 1, 0], "lung cancer": [1, 1, 0, 0]},
                        index=pd.Index(["a", "b", "c", "d"], name="ID"))
    # V2 Y is a subset (x, z, 007): lung cancer positives = 2 on subset, 3 on full raw
    v2_y = pd.DataFrame({"glioma": [0, 1, 0], "lung cancer": [1, 0, 1]},
                        index=pd.Index(["x", "z", "007"], name="ID"))
    if v2_y_mutate:
        v2_y_mutate(v2_y)
    v1_y_path, v2_y_path = tmp_path / "v1_y.csv", tmp_path / "v2_y.csv"
    v1_y.to_csv(v1_y_path)
    v2_y.to_csv(v2_y_path)
    return v1_path, v2_path, map_path, v1_y_path, v2_y_path


def run(paths):
    return audit(*paths, n_expected=2)


def test_pass_case_and_or_counts(tmp_path):
    r = run(_write(tmp_path))
    assert r["all_pass"], {k: v["pass"] for k, v in r["checks"].items()}
    c6 = r["checks"]["6_positives_match"]
    assert c6["map_v2_positive_population"] == "v2_y_subset"
    row = r["table"].set_index("canonical_name").loc["lung cancer"]
    assert row["v2_positives_full_raw"] == 3 and row["v2_positives_y_subset"] == 2
    orc = r["or_combinations"].set_index("canonical_name")
    assert list(orc.index) == ["lung cancer"]
    assert orc.loc["lung cancer", "v1_full_first_col_positives"] == 1
    assert orc.loc["lung cancer", "v1_full_or_positives"] == 2
    assert orc.loc["lung cancer", "v1_full_rows_changed_by_or"] == 1
    # near miss: V2 "gliomas" vs "glioma"
    nm = r["near_misses"]
    assert ((nm.v2_normalized == "gliomas") & (nm.v1_normalized == "glioma")).any()
    # leading-zero ID preserved as string
    assert r["checks"]["5_y_rebuild_equal"]["v2"]["ids_equal"]


def test_missing_source_column(tmp_path):
    rows = [
        {"order": 0, "canonical_name": "glioma", "v1_source_columns": "Glioma",
         "v2_source_columns": "glioma", "v1_positives": 1, "v2_positives": 1},
        {"order": 1, "canonical_name": "lung cancer", "v1_source_columns": "Lung Cancer | lung-cancer",
         "v2_source_columns": "lung cancer ", "v1_positives": 2, "v2_positives": 2},
    ]
    r = run(_write(tmp_path, map_rows=rows))
    assert not r["all_pass"]
    c1 = r["checks"]["1_source_columns_exist"]
    assert not c1["pass"]
    assert c1["missing"]["v2"] == [{"canonical_name": "lung cancer", "column": "lung cancer "}]


def test_unlisted_raw_column_same_normalized_name(tmp_path):
    r = run(_write(tmp_path, v2_extra={"Lung_Cancer": [0, 0, 0, 1, 0]}))
    c3 = r["checks"]["3_completeness"]
    assert not c3["pass"] and not r["all_pass"]
    mm = c3["column_set_mismatches"]
    assert mm == [{"version": "v2", "canonical_name": "lung cancer",
                   "unlisted_raw_columns": ["Lung_Cancer"], "extra_listed_columns": []}]
    assert r["checks"]["1_source_columns_exist"]["pass"]


def test_y_mismatch(tmp_path):
    def mutate(y):
        y.loc["z", "glioma"] = 0
    r = run(_write(tmp_path, v2_y_mutate=mutate))
    c5 = r["checks"]["5_y_rebuild_equal"]
    assert not c5["pass"] and c5["v1"]["pass"]
    assert c5["v2"]["n_value_differences"] == 1
    assert not r["all_pass"]


def test_non_binary_raw_value_flagged(tmp_path):
    r = run(_write(tmp_path, v2_extra={"Other": [0, 2, 0, 0, 0]}))
    c5 = r["checks"]["5_y_rebuild_equal"]
    assert c5["raw_non_binary_columns"] == [{"version": "v2", "column": "Other", "non_binary_values": ["2"]}]
    assert not c5["pass"]


def test_write_outputs_refuses_nonempty_and_writes(tmp_path):
    paths = _write(tmp_path)
    r = run(paths)
    out = tmp_path / "out"
    payload = write_outputs(r, out, dict(zip("abcde", paths)))
    assert payload["all_pass"]
    names = {p.name for p in out.iterdir()}
    assert names == {"disease_map_audit.csv", "or_combinations.csv", "near_misses.csv", "disease_map_audit.json"}
    data = json.loads((out / "disease_map_audit.json").read_text())
    assert set(data["outputs"]) == names - {"disease_map_audit.json"}
    with pytest.raises(FileExistsError):
        write_outputs(r, out, dict(zip("abcde", paths)))


def test_main_exit_code_nonzero_on_failure(tmp_path):
    v1, v2, m, y1, y2 = _write(tmp_path)
    # real default n_expected=73 fails completeness on the 2-concept synthetic map
    code = main(["--v1-raw", str(v1), "--v2-raw", str(v2), "--label-map", str(m),
                 "--v1-y", str(y1), "--v2-y", str(y2), "--outdir", str(tmp_path / "o")])
    assert code == 1
