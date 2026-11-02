from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import prepare_nazli_v1_inferred_827_input as prep


def test_default_frames_are_code_generated_with_expected_blocks():
    x, y, facts = prep.build_frames()

    assert x.shape == (355, 827)
    assert y.shape == (355, 44)
    assert x.index.tolist() == y.index.tolist()
    assert facts["components"] == {
        "dcc": 462,
        "association": 285,
        "kmer2": 16,
        "kmer3": 64,
    }
    assert facts["felipe_feature_csv_used"] is False
    assert facts["association_block"]["outcome_derived"] is True
    assert facts["association_block"]["label_leakage_risk"] is True
    assert np.isin(x.iloc[:, 462:747].to_numpy(), [0, 1]).all()


def test_generated_artifacts_pass_fresh_check(tmp_path):
    outdir = tmp_path / "inputs"
    prep.generate(outdir)
    prep.check(outdir)

    manifest = pd.read_csv(outdir / prep.MANIFEST_NAME).iloc[0]
    provenance = json.loads((outdir / prep.PROVENANCE_NAME).read_text())
    assert int(manifest["n_features"]) == 827
    assert manifest["feature_key"] == "nazli_inferred_827"
    assert provenance["association_block"]["positive_cells"] == 1132
