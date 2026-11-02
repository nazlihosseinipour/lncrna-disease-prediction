import numpy as np
import pytest

from scripts.audit_827_feature_reconstruction import (
    best_row_matches,
    classify,
    classify_rows,
    column_matches,
    contiguous_runs,
    dac_vector,
    mismatch_counts,
    rel_diff,
    run_audit,
)


def test_classify_tolerances():
    assert classify(0.0, 0.0) == "exact"
    assert classify(1e-8, 0.5) == "within_abs_1e-6"
    assert classify(1e-6, 1.0) == "within_abs_1e-6"
    assert classify(1e-3, 1e-7) == "within_rel_1e-6"
    assert classify(1e-3, 1e-3) == "unmatched"
    assert classify(np.inf, 0.0) == "unmatched"


def test_rel_diff_handles_zeros():
    out = rel_diff(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0]))
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5])


def test_row_mapping_detects_shift_and_unmatched():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(8, 5))
    # released: rows 0-2 identity, raw row 3 skipped (shift +1), last row unknown
    released = np.vstack([raw[0:3], raw[4:8] + 1e-9, rng.normal(size=(1, 5)) + 50])
    best = best_row_matches(released, raw)
    classes = classify_rows(best)
    assert classes[:3] == ["identity"] * 3
    assert classes[3:7] == ["shifted"] * 4
    assert classes[7] == "unmatched"
    assert best.best_row.tolist()[:7] == [0, 1, 2, 4, 5, 6, 7]
    assert (best.best_max_abs.iloc[3:7] <= 1e-6).all()


def test_column_matches_under_mapping():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(6, 4))
    mapping = np.array([0, 1, 3, 4, 5, -1])
    released = np.zeros((6, 3))
    released[:5, 0] = raw[mapping[:5], 2]           # exact copy of candidate 2
    released[:5, 1] = raw[mapping[:5], 0] + 5e-7     # within abs tolerance of candidate 0
    released[:5, 2] = rng.normal(size=5) + 100       # no match
    released[5] = 999.0                              # excluded row must not matter
    best, best_abs, best_rel, _ = column_matches(released, raw, mapping)
    assert best[0] == 2 and best_abs[0] == 0.0
    assert best[1] == 0 and classify(best_abs[1], best_rel[1]) == "within_abs_1e-6"
    assert classify(best_abs[2], best_rel[2]) == "unmatched"
    # identity mapping on shifted data does not match
    _, abs_ident, rel_ident, _ = column_matches(released[:5], raw[:5], np.arange(5))
    assert classify(abs_ident[0], rel_ident[0]) == "unmatched"


def test_contiguous_runs():
    runs = contiguous_runs(["a", "a", "b", "a"])
    assert [(r["start"], r["end_inclusive"], r["family"]) for r in runs] == [
        (0, 1, "a"), (2, 2, "b"), (3, 3, "a")
    ]


def test_mismatch_counts_small():
    # "AAA" has one 3-mer: AAA. Hamming<=1 targets: AAA plus 9 single substitutions.
    out = mismatch_counts("AAA", 3, 1)
    assert out.sum() == 10 and out[0] == 1


def test_dac_constant_property_is_zero():
    props = np.ones((16, 2))
    assert np.allclose(dac_vector("ACGUACGU", props, 1, style="textbook"), 0.0)


def test_run_audit_refuses_overwrite(tmp_path):
    (tmp_path / "column_audit.csv").write_text("x")
    with pytest.raises(FileExistsError):
        run_audit(tmp_path)
