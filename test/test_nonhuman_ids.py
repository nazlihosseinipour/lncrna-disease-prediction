import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mainfolder.utils.nonhuman_ids import classify_nonhuman_id, split_nonhuman_unresolved_df


def test_classify_nonhuman_id_obvious_cases_only():
    assert classify_nonhuman_id("NONRATT001894.2") == {
        "species_hint": "rat",
        "non_human_reason": "rat_noncode",
    }
    assert classify_nonhuman_id("ENSRNOT00000078751") == {
        "species_hint": "rat",
        "non_human_reason": "rat_ensembl",
    }
    assert classify_nonhuman_id("1700020I14Rik") == {
        "species_hint": "mouse",
        "non_human_reason": "mouse_rik_symbol",
    }
    assert classify_nonhuman_id("NONHSAT000612.2") is None
    assert classify_nonhuman_id("MALAT1") is None


def test_split_nonhuman_unresolved_df_separates_rows():
    df = pd.DataFrame(
        [
            {"id": "NONRATT001894.2", "type": "symbol", "status": "unresolved", "reason": "x", "resolved_id": ""},
            {"id": "ENSRNOT00000078751", "type": "ensembl", "status": "unresolved", "reason": "y", "resolved_id": ""},
            {"id": "MALAT1", "type": "symbol", "status": "unresolved", "reason": "z", "resolved_id": ""},
        ]
    )
    human_df, nonhuman_df = split_nonhuman_unresolved_df(df)

    assert human_df["id"].tolist() == ["MALAT1"]
    assert nonhuman_df["id"].tolist() == ["NONRATT001894.2", "ENSRNOT00000078751"]
    assert nonhuman_df["species_hint"].tolist() == ["rat", "rat"]
