import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from mainfolder.utils.disease_mapping import build_disease_term_mapping, ensure_disease_override_csv


def toy_do_terms():
    return pd.DataFrame(
        [
            {
                "doid": "DOID:1",
                "name": "rheumatoid arthritis",
                "synonyms": "",
                "synonyms_json": "[]",
            },
            {
                "doid": "DOID:2",
                "name": "breast cancer",
                "synonyms": "breast neoplasm",
                "synonyms_json": '["breast neoplasm"]',
            },
            {
                "doid": "DOID:3",
                "name": "traumatic brain injury",
                "synonyms": "",
                "synonyms_json": "[]",
            },
            {
                "doid": "DOID:4",
                "name": "esophagus adenocarcinoma",
                "synonyms": "",
                "synonyms_json": "[]",
            },
        ]
    )


def test_build_disease_term_mapping_handles_variants_and_synonyms():
    mapping_df, review_df = build_disease_term_mapping(
        [
            "Arthritis, Rheumatoid",
            "Breast Neoplasms",
            "Brain Injuries, Traumatic",
            "Adenocarcinoma Of Esophagus",
        ],
        toy_do_terms(),
    )

    got = dict(zip(mapping_df["disease"], mapping_df["term"]))
    assert got["Arthritis, Rheumatoid"] == "DOID:1"
    assert got["Breast Neoplasms"] == "DOID:2"
    assert got["Brain Injuries, Traumatic"] == "DOID:3"
    assert got["Adenocarcinoma Of Esophagus"] == "DOID:4"

    review = review_df.set_index("disease")
    assert review.loc["Arthritis, Rheumatoid", "match_type"] == "exact_name"
    assert review.loc["Breast Neoplasms", "match_type"] in {"exact_name", "exact_synonym", "token_name", "token_synonym"}
    assert review.loc["Brain Injuries, Traumatic", "match_type"] in {"exact_name", "token_name"}


def test_manual_override_takes_precedence():
    with tempfile.TemporaryDirectory() as tmpdir:
        override_path = Path(tmpdir) / "disease_term_overrides.csv"
        ensure_disease_override_csv(override_path)
        pd.DataFrame(
            [
                {
                    "disease": "Breast Neoplasms",
                    "term": "DOID:999",
                    "note": "manual test override",
                }
            ]
        ).to_csv(override_path, index=False)

        mapping_df, review_df = build_disease_term_mapping(
            ["Breast Neoplasms"],
            toy_do_terms(),
            overrides_path=override_path,
        )

    assert mapping_df.iloc[0]["term"] == "DOID:999"
    assert review_df.iloc[0]["match_type"] == "override"
