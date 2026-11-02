import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/regenerate_nazli_v1_827_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("regenerate_nazli_v1_827_diagnostic", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def test_assemble_matrix_places_outcomes_between_dcc_and_kmers():
    ids = ["rna-a", "rna-b"]
    seq = pd.DataFrame(
        np.arange(2 * 542).reshape(2, 542),
        index=ids,
        columns=[f"seq-{i}" for i in range(542)],
    )
    assoc = pd.DataFrame(
        np.ones((2, 285), dtype=int),
        index=ids,
        columns=[f"association_outcome::d-{i}" for i in range(285)],
    )

    result = M.assemble_matrix(seq, assoc)

    assert result.shape == (2, 827)
    assert result.columns[:462].tolist() == seq.columns[:462].tolist()
    assert result.columns[462:747].tolist() == assoc.columns.tolist()
    assert result.columns[747:].tolist() == seq.columns[462:].tolist()


def test_sequence_row_mapping_distinguishes_identity_shift_and_unmatched():
    rng = np.random.default_rng(42)
    generated = pd.DataFrame(rng.normal(size=(4, 827)), index=["a", "b", "c", "d"])
    released = generated.copy()
    # Row 1 carries row 2's two sequence blocks, while row 0 stays in place.
    released.iloc[1, :462] = generated.iloc[2, :462]
    released.iloc[1, 747:] = generated.iloc[2, 747:]
    # Row 3 has no sequence candidate.
    released.iloc[3, :462] = 1e6
    released.iloc[3, 747:] = 1e6

    mapping = M.sequence_row_mapping(released, generated)

    assert mapping.loc[0, "mapping_class"] == "identity"
    assert mapping.loc[1, "mapping_class"] == "shifted"
    assert mapping.loc[1, "mapped_generated_row"] == 2
    assert mapping.loc[3, "mapping_class"] == "unmatched"
    assert mapping.loc[3, "mapped_generated_row"] == -1


def test_comparison_stats_separates_exact_from_tolerance_matches():
    left = np.array([[0.0, 1.0], [2.0, 3.0]])
    right = np.array([[0.0, 1.0 + 1e-8], [2.0, 3.0]])

    stats = M.comparison_stats(left, right)

    assert stats["exact_cells"] == 3
    assert stats["within_1e6_cells"] == 4
    assert stats["exact_rows"] == 1


def test_checked_artifact_is_explicitly_outcome_derived_and_not_predictive():
    M.check(M.DEFAULT_OUTDIR)
    audit = json.loads((M.DEFAULT_OUTDIR / M.AUDIT_FILE).read_text(encoding="utf-8"))

    assert bool(audit["association_columns_are_outcome_derived"]) is True
    assert bool(audit["valid_sequence_only_predictive_input"]) is False
    assert audit["row_mapping_counts"] == M.EXPECTED_MAPPING_COUNTS
