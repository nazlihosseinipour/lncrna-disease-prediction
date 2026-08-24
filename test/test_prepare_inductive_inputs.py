import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


P = Path(__file__).resolve().parents[1] / "scripts/prepare_inductive_feature_representations.py"
SPEC = importlib.util.spec_from_file_location("prepare_inductive_inputs", P)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_identical_sequences_are_collapsed_and_labels_are_ored():
    x = pd.DataFrame(
        {
            "sample_id": ["alias-a", "alias-b", "unique"],
            "f1": [1.0, 1.0, 2.0],
            "f2": [3.0, 3.0, 4.0],
        }
    )
    y = pd.DataFrame(
        {
            "ID": ["alias-a", "alias-b", "unique"],
            "d1": [1, 0, 0],
            "d2": [0, 1, 1],
        }
    )
    sequences = pd.Series(["ACTT", "ACUU", "GGGG"])

    x_out, y_out, audit = M.collapse_identical_sequences(x, y, sequences)

    assert x_out.sample_id.tolist() == ["alias-a", "unique"]
    assert y_out.ID.tolist() == ["alias-a", "unique"]
    assert y_out.loc[0, ["d1", "d2"]].tolist() == [1, 1]
    assert audit.loc[0, "group_size"] == 2
    assert audit.loc[0, "conflicting_label_count"] == 2


def test_label_filter_uses_final_aligned_counts():
    y = pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "keep": [1, 1, 0],
            "drop_after_alignment": [1, 0, 0],
        }
    )

    filtered, counts, dropped = M.filter_aligned_labels(
        y, min_positives=1, keep_rule="gt"
    )

    assert filtered.columns.tolist() == ["ID", "keep"]
    assert counts.to_dict() == {"keep": 2}
    assert dropped.to_dict("records") == [
        {"label": "drop_after_alignment", "aligned_positive_count": 1}
    ]


def test_exact_duplicate_feature_columns_are_removed():
    x = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "block_a__x": [1.0, 2.0],
            "block_b__same_x": [1.0, 2.0],
            "different": [3.0, 4.0],
        }
    )

    cleaned, audit = M.drop_duplicate_feature_columns(x)

    assert cleaned.columns.tolist() == ["sample_id", "block_a__x", "different"]
    assert audit.to_dict("records") == [
        {
            "retained_feature": "block_a__x",
            "dropped_duplicate_feature": "block_b__same_x",
        }
    ]


def test_target_contamination_audit_rejects_shifted_label_block():
    rng = np.random.default_rng(7)
    labels = (rng.random((20, 6)) < 0.25).astype(int)
    # Mimic a sequence extractor dropping one row and a caller later assigning
    # the original IDs by position: most association rows remain direct copies,
    # while the suffix is shifted and the omitted row is appended.
    leaked = np.vstack([labels[:12], labels[13:], labels[12]])
    x = pd.DataFrame(leaked, columns=[f"feature_{i}" for i in range(6)])
    x.insert(0, "sample_id", [f"rna_{i}" for i in range(20)])
    y = pd.DataFrame(labels, columns=[f"disease_{i}" for i in range(6)])
    y.insert(0, "ID", x["sample_id"])

    audit = M.audit_target_contamination(x, y)

    assert audit["flagged"] is True
    assert audit["best_label_width_block_positive_f1"] >= 0.5
    assert audit["best_label_width_block_exact_row_fraction"] >= 0.5
    with pytest.raises(ValueError, match="target-derived data"):
        M.assert_no_target_contamination(x, y, context="test/paper-artifact")


def test_target_contamination_audit_accepts_clean_continuous_features():
    x = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "gc_content": [0.41, 0.53, 0.38, 0.61],
            "kmer_ac": [0.11, 0.07, 0.14, 0.05],
        }
    )
    y = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "disease_a": [1, 0, 0, 1],
            "disease_b": [0, 1, 0, 0],
        }
    )

    audit = M.assert_no_target_contamination(x, y, context="test/clean")

    assert audit["flagged"] is False
    assert audit["binary_feature_columns"] == 0


def test_target_contamination_audit_ignores_constant_zero_coincidence():
    x = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "always_zero": [0, 0, 0, 0],
            "unrelated_binary": [1, 0, 1, 0],
        }
    )
    y = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "unused_label": [0, 0, 0, 0],
            "disease": [0, 1, 0, 0],
        }
    )

    audit = M.audit_target_contamination(x, y)

    assert audit["flagged"] is False
    assert audit["exact_label_copy_count"] == 0
