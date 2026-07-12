import importlib.util
from pathlib import Path

import pandas as pd


P = Path(__file__).resolve().parents[1] / "scripts/run_inductive_transfer_experiments.py"
SPEC = importlib.util.spec_from_file_location("transfer_runner", P)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_remove_exact_target_overlap_preserves_alignment_and_order():
    xs = pd.DataFrame({"sample_id": ["A", "B"], "x": [1, 2]})
    xt = pd.DataFrame({"sample_id": ["C", "A", "D"], "x": [3, 4, 5]})
    yt = pd.DataFrame({"ID": ["C", "A", "D"], "d": [0, 1, 1]})
    x, y, overlap = M.remove_exact_target_overlap(xs, xt, yt)
    assert overlap == ["A"]
    assert x.sample_id.tolist() == ["C", "D"]
    assert y.ID.tolist() == ["C", "D"]
    assert y.d.tolist() == [0, 1]


def test_large_concatenation_uses_short_stable_key():
    p = Path(__file__).resolve().parents[1] / "scripts/prepare_inductive_feature_representations.py"
    spec = importlib.util.spec_from_file_location("prepare_runner", p)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.sanitize_feature_key("+".join(["long_feature_name"] * 20)) == "all_safe_concatenated"
