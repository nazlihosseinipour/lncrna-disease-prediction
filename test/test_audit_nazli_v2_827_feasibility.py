from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import audit_nazli_v2_827_feasibility as audit


def fixture_frames():
    v2 = pd.DataFrame(
        {
            "ID": ["rna-a", "rna-b", "rna-c"],
            "seqs": ["ACGU", None, "UGCA"],
            "Disease A": [1, 0, 0],
            "Disease-B": [0, 1, 1],
            "disease b": [0, 0, 1],
        }
    )
    v1 = pd.DataFrame(
        {
            "ID": ["old-a"],
            "seqs": ["ACGU"],
            "Disease A": [1],
            "Disease B": [0],
            "Old only": [1],
        }
    )
    x = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=pd.Index(["rna-a", "rna-c"], name="ID"),
        columns=["dcc::a", "kmer2::AA"],
    )
    return v2, v1, x


def test_analysis_counts_population_widths_and_normalized_overlap():
    v2, v1, x = fixture_frames()
    summary, overlap, missing = audit.analyze_frames(v2, v1, x)

    assert summary["v2_population"] == {
        "full_rows": 3,
        "unique_rna_ids": 3,
        "usable_sequence_rows": 2,
        "missing_sequence_rows": 1,
        "sequence_feature_rows": 2,
    }
    assert summary["association_outcomes"]["v2_raw_disease_columns"] == 3
    assert summary["association_outcomes"]["v2_normalized_unique_diseases"] == 2
    assert summary["v1_v2_disease_overlap"] == {
        "exact_name_matches": 1,
        "normalized_name_matches": 2,
        "normalization_rule": "lowercase; replace non-alphanumeric runs with spaces; trim",
    }
    assert summary["width_analysis"]["sequence_plus_all_v2_outcomes_width"] == 5
    assert summary["decision"]["append_v2_association_outcomes"] is False
    assert overlap.loc[overlap.v1_disease == "Disease B", "normalized_match_count"].item() == 2
    assert missing.to_dict(orient="list") == {"ID": ["rna-b"]}


def test_analysis_refuses_row_misalignment_duplicates_and_nonbinary_outcomes():
    v2, v1, x = fixture_frames()
    with pytest.raises(ValueError, match="exactly match"):
        audit.analyze_frames(v2, v1, x.iloc[::-1])

    duplicate = v2.copy()
    duplicate.loc[1, "ID"] = "rna-a"
    with pytest.raises(ValueError, match="duplicate RNA IDs"):
        audit.analyze_frames(duplicate, v1, x)

    nonbinary = v2.copy()
    nonbinary.loc[0, "Disease A"] = 2
    with pytest.raises(ValueError, match="binary 0/1"):
        audit.analyze_frames(nonbinary, v1, x)


def test_written_artifacts_are_deterministic_and_checkable(tmp_path):
    v2, v1, x = fixture_frames()
    summary, overlap, missing = audit.analyze_frames(v2, v1, x)
    summary["sources"] = {"fixture": True}
    outdir = tmp_path / "audit"
    report = tmp_path / "report.md"

    audit.write_artifacts(summary, overlap, missing, outdir, report)

    assert json.loads((outdir / "summary.json").read_text()) == summary
    assert "association-derived" in report.read_text()
    assert "label leakage" in report.read_text()
    with pytest.raises(FileExistsError):
        audit.write_artifacts(summary, overlap, missing, outdir, report)


def test_default_sources_measure_the_canonical_v2_contract():
    summary, overlap, missing = audit.load_and_analyze()

    assert summary["v2_population"]["full_rows"] == 5_338
    assert summary["v2_population"]["usable_sequence_rows"] == 4_130
    assert summary["feature_contract"]["valid_sequence_features"] == 542
    assert summary["association_outcomes"]["v2_raw_disease_columns"] == 436
    assert summary["width_analysis"]["sequence_plus_all_v2_outcomes_width"] == 978
    assert summary["v1_v2_disease_overlap"]["exact_name_matches"] == 20
    assert summary["v1_v2_disease_overlap"]["normalized_name_matches"] == 73
    assert len(overlap) == 285
    assert len(missing) == 1_208
