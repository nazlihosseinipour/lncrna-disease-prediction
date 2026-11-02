from __future__ import annotations

import pandas as pd
import pytest

from scripts.prepare_inductive_feature_representations import preserve_aligned_inputs


def test_preservation_mode_keeps_every_row_and_column():
    x = pd.DataFrame(
        {"sample_id": ["a", "b"], "duplicate_1": [1, 0], "duplicate_2": [1, 0]}
    )
    y = pd.DataFrame({"ID": ["a", "b"], "rare": [1, 0], "empty": [0, 0]})

    x_out, y_out, sequence_audit, feature_audit, counts, dropped = (
        preserve_aligned_inputs(
            x,
            y,
            source_ids=["a", "b"],
            label_cols=["rare", "empty"],
            context="test",
        )
    )

    pd.testing.assert_frame_equal(x_out, x)
    pd.testing.assert_frame_equal(y_out, y)
    assert sequence_audit.empty
    assert feature_audit.empty
    assert dropped.empty
    assert counts.to_dict() == {"rare": 1, "empty": 0}


def test_preservation_mode_rejects_missing_rows():
    x = pd.DataFrame({"sample_id": ["a"], "f": [1]})
    y = pd.DataFrame({"ID": ["a"], "label": [1]})

    with pytest.raises(ValueError, match="would change source rows"):
        preserve_aligned_inputs(
            x,
            y,
            source_ids=["a", "b"],
            label_cols=["label"],
            context="test",
        )
