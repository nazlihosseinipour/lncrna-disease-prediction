from pathlib import Path

import pandas as pd

from mainfolder.utils.disease_override_merge import merge_manual_review_into_overrides


def test_merge_manual_review_into_overrides_updates_and_adds(tmp_path: Path) -> None:
    review_path = tmp_path / "manual.csv"
    overrides_path = tmp_path / "overrides.csv"
    do_terms_path = tmp_path / "do_terms.csv"

    pd.DataFrame(
        [
            {"doid": "DOID:1", "name": "foo"},
            {"doid": "DOID:2", "name": "bar"},
        ]
    ).to_csv(do_terms_path, index=False)

    pd.DataFrame(
        [
            {"disease": "Disease A", "term": "DOID:1", "note": "old"},
        ]
    ).to_csv(overrides_path, index=False)

    pd.DataFrame(
        [
            {"disease": "Disease A", "chosen_term": "DOID:2", "chosen_name": "bar", "note": "new"},
            {"disease": "Disease B", "chosen_term": "DOID:1", "chosen_name": "foo", "note": ""},
        ]
    ).to_csv(review_path, index=False)

    summary = merge_manual_review_into_overrides(review_path, overrides_path, do_terms_path=do_terms_path)

    assert summary == {
        "selected_rows": 2,
        "added": 1,
        "updated": 1,
        "unchanged": 0,
        "total_overrides": 2,
    }

    merged = pd.read_csv(overrides_path).fillna("")
    assert list(merged.columns) == ["disease", "term", "note"]
    assert merged.to_dict("records") == [
        {"disease": "Disease A", "term": "DOID:2", "note": "chosen_name=bar | new"},
        {"disease": "Disease B", "term": "DOID:1", "note": "chosen_name=foo"},
    ]
