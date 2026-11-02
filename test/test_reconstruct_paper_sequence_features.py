from pathlib import Path

import numpy as np
import pandas as pd

from scripts.reconstruct_paper_sequence_features import reconstruct


ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_matches_released_sequence_blocks_before_source_drift():
    rebuilt, metadata = reconstruct(
        ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv",
        ROOT / "config/pse_in_one_rna_dcc_properties.csv",
        id_column="ID",
        sequence_column="seqs",
    )
    released = pd.read_csv(
        ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/features.csv",
        index_col=0,
    )
    released_sequence_only = np.concatenate(
        [released.iloc[:231, :462].to_numpy(), released.iloc[:231, 747:].to_numpy()],
        axis=1,
    )
    np.testing.assert_allclose(
        rebuilt.iloc[:231].to_numpy(), released_sequence_only, atol=1e-8, rtol=0
    )
    assert rebuilt.shape == (355, 542)
    assert metadata["cleaned_sequence_count"] == 2
    assert metadata["missing_sequence_count"] == 0


def test_no_reconstructed_column_is_the_released_association_block():
    rebuilt, _ = reconstruct(
        ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv",
        ROOT / "config/pse_in_one_rna_dcc_properties.csv",
        id_column="ID",
        sequence_column="seqs",
    )
    released = pd.read_csv(
        ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/features.csv",
        index_col=0,
    )
    association = released.iloc[:, 462:747].to_numpy()
    assert rebuilt.shape[1] == 542
    assert association.shape[1] == 285
    assert not np.isin(rebuilt.to_numpy(), [0, 1]).all(axis=0).any()
