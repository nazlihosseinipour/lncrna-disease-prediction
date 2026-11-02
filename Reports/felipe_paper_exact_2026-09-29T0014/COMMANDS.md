# Exact commands (2026-09-29)

Run from the repo root. `PY=results/runtime/felipe_env/bin/python`. Environment: `ENVIRONMENT.md`.

```bash
# Phase 1 freeze (before any work)
R=results/felipe_paper_exact_2026-09-29T0014
git status --porcelain=v1 > $R/freeze/git_status.txt; git rev-parse HEAD > $R/freeze/git_head.txt
# protected_before.sha256 = shasum -a 256 over: results/felipe_final_2026-09-28, results/felipe_shared_sequence_2026-09-28,
#   results/paper_reproduction_reference_2026-09-20, results/audit/felipe_exact_results_2026-09-27,
#   results/felipe_basics_2026-09-27/inputs, lncRNA_CIBCB2025-main/{create_dataset/datasets,create_dataset/raw_data,RF,IPCARF,RFLDA,utils,parse_results},
#   Data/corrected_inputs/v2_transcript_sequences, the paper PDF, scripts/*.py, config/*.csv, Reports/felipe_audit_2026-09-27
shasum -a 256 -c $R/freeze/protected_before.sha256        # re-check: 584/584 OK

# Phase 2 feature gate (read-only; no fitting)
$PY -m pytest -q test/test_audit_827_feature_reconstruction.py   # 8 passed
$PY scripts/audit_827_feature_reconstruction.py                  # writes $R/feature_gate/

# Phase 3 verification of existing V1 827 results (no rerun)
$PY scripts/verify_v1_827_existing_results.py                    # writes $R/v1_verification/
```

- **Existing V1 runs that were verified (not rerun):** their commands are in
  `results/paper_reproduction_reference_2026-09-20/ipcarf/run_all.sh`, `run_ipcarf_authors.py` and
  `run_eval_authors.sh`, and in `table3_other_rows/run_variant.sh`, `run_authors.py` and
  `run_eval.sh`. The exact code that ran is in each folder's `code_used/`.
- **Saved fold-level probabilities:**
  - `results/paper_reproduction_reference_2026-09-20/ipcarf/predictions_mlc/`
  - `results/paper_reproduction_reference_2026-09-20/table3_other_rows/predictions/rflda_mlc/`
- **Thresholds:** `$R/v1_verification/thresholds_{ipcarf,rflda}_mlc.csv`.

## Added 2026-10-05: V1 plain RF (released 827 MLC construction)

```bash
$PY scripts/run_v1_827_plain_rf.py      # writes results/felipe_paper_exact_2026-09-29T0014/v1_rf_827/ (refuses to overwrite)
$PY scripts/verify_v1_827_plain_rf.py   # independent recomputation -> .../v1_rf_827_verification.json (passed: true)
shasum -a 256 -c results/felipe_paper_exact_2026-09-29T0014/freeze/protected_before.sha256
```

- **RF outputs:**
  - `v1_rf_827/predictions/rf_mlc_predictions_test_fold{1..10}.csv` (ID index, `test_position`, 44
    label scores)
  - `rf_mlc_thresholds.csv`
  - `rf_mlc_per_fold.csv`
  - `rf_mlc_summary.csv`
  - `run_contract.json`
- **Saved fold-level probabilities, all three models:**
  - RF: `results/felipe_paper_exact_2026-09-29T0014/v1_rf_827/predictions/`
  - IPCARF: `results/paper_reproduction_reference_2026-09-20/ipcarf/predictions_mlc/`
  - RFLDA: `results/paper_reproduction_reference_2026-09-20/table3_other_rows/predictions/rflda_mlc/`
- **Thresholds, all three models:**
  - `v1_rf_827/rf_mlc_thresholds.csv`
  - `v1_verification/thresholds_{ipcarf,rflda}_mlc.csv`
