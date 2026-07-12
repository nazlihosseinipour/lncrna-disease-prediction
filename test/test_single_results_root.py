from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOVED = ("results_canonical", "results_reconciliation", "results_backup_2026-06-26", "results-v2")


def test_active_python_scripts_do_not_reference_removed_result_roots():
    active = [
        "run_inductive_within_cv.py",
        "run_inductive_transfer_experiments.py",
        "run_binary_comparison.py",
        "rescore_saved_predictions.py",
        "build_final_deliverables.py",
        "build_generalization_matrix.py",
        "generate_publication_results.py",
    ]
    for name in active:
        text = (ROOT / "scripts" / name).read_text()
        assert all(token not in text for token in REMOVED), name


def test_server_jobs_write_only_to_results_root():
    for path in (ROOT / "server").glob("server_*.sh"):
        text = path.read_text()
        if path.name == "server_run_lib.sh":
            assert 'RESULTS_ROOT="${RESULTS_ROOT:-results}"' in text
            continue
        assert "results/" in text
        assert all(token not in text for token in REMOVED), path.name
