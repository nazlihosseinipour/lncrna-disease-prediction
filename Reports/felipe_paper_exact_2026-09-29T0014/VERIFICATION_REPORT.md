# Verification report (2026-09-29)

Updated 2026-10-05. The only model fitted in this task is the V1 plain RF; IPCARF and RFLDA were
not rerun.

- **V1 IPCARF/RFLDA MLC.** The existing paper-faithful results were verified independently from
  their saved fold-level predictions (`scripts/verify_v1_827_existing_results.py`, read-only).
- **V1 plain RF.** Run on 2026-10-05 with `scripts/run_v1_827_plain_rf.py`, then verified
  independently with `scripts/verify_v1_827_plain_rf.py` (see the RF section below).
- **V2.** Not run (feature gate failed).

## Checks

| # | Check | Result |
|---|---|---|
| 1 | Source and protected-result hashes | PASS. 584/584 hashes in `freeze/protected_before.sha256` are unchanged. They cover: released V1 X/Y/folds (`b264dd75…`, `53c3573b…`, `e9fe2eb5…`), V2 inputs, model and evaluation code, the 542-feature and shared-sequence results, and `paper_reproduction_reference_2026-09-20/`. Git status differs from the freeze only by new files. Re-checked on 2026-10-05 after the RF run: 584/584 OK (`freeze/protected_check_2026-10-05.txt`); the only additions are this report folder and five new scripts or tests. |
| 2 | Feature dimensions and order | V1: 355 × 827, used as-is. Column families: DCC 0–461, binary block 462–746, k-mers 747–826. V2: not built (gate failed). |
| 3 | X/Y alignment | The X index equals the Y index equals raw `ID` by position. **Content offset found:** from row 231 on, X content comes from raw rows +1/+2 (3 rows unmatched), while Y follows the IDs (see `V1_FEATURE_RECONSTRUCTION_AUDIT.md`). This is a property of the released artifact, left as-is. |
| 4 | Fold completeness | PASS. Train and test are disjoint in every fold, and each row is in exactly one test fold (V1). V2 group separation was verified earlier for `v2_all_436_group_splits.csv`, but is unused here. |
| 5 | Model parameters | PASS. 150 trees, `random_state=0`, `binary_mode=False`, `transductive_mode=False` for all three models. Fold counts are in the table below. Taken from `code_used/` and the logs. |
| 6 | Prediction-file completeness | PASS. 10 folds each for IPCARF MLC and RFLDA MLC, each (n_test, 44) in Y column order, with finite values in [0, 1]. The files have no ID column. Their row order was confirmed by an alignment test: aligned pooled AUROC 0.823/0.809, against ≤ 0.67 over 200 permutations. |
| 7 | Threshold calculation | PASS. Per-label Youden thresholds were recomputed with the paper rule, including the authors' Σ(tpr−fpr) ≤ 0 → 0.5 fallback (62 single-class cells; 73 and 77 further fallback cells). Saved in `v1_verification/thresholds_*.csv`. |
| 8 | Metrics from saved probabilities | PASS. All 8 measures per fold match: max \|Δ\| ≤ 1e-16 against the per-fold CSVs, and 0.0 against the authors' `eval/*_performance.csv`. |
| 9 | Aggregation | PASS. Mean and population SD reproduce the reported 4-decimal tables exactly. |
| 10 | No historical results overwritten | PASS (see #1). |
| — | Determinism | PASS. IPCARF MLC predictions are identical (max \|Δ\| = 0) to an earlier same-day run. |
| — | Test labels during fit/predict | PASS. `predict_proba(x_test, y_test)` ignores `y_test` in MLC mode. Test labels enter only through the evaluation-stage Youden threshold (the paper procedure). |

## Fold counts actually used (from `code_used/`)

| Level | RF | IPCARF | RFLDA |
|---|---|---|---|
| Outer evaluation folds | 10 (released `inductive_splits.csv`) | 10 (same) | 10 (same) |
| Inner model selection | none | `GridSearchCV(cv=5)`: 5-fold, unshuffled KFold (sklearn default for a regressor), over components {2, 4, 8, 16, 32, 64, 128}, default R² | Feature-count selection: `MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)`, steps of 50, subset accuracy |

The authors' committed RFLDA code sets 10 feature-selection folds. The reproduction was patched to
the paper's stated 5. The committed IPCARF code already uses 5.

## V1 plain RF (run 2026-10-05)

- **Inputs and settings.** Released 355 × 827 X (all columns), 44 flat labels and the released
  10 folds. The authors' `RF` class (`RandomForestRegressor`, 150 trees, `random_state=0`, default
  `max_features`) and the authors' 73b21ee Evaluator.
- **Saved outputs** under `results/felipe_paper_exact_2026-09-29T0014/v1_rf_827/`:
  - probabilities with RNA ID and test position;
  - thresholds;
  - eight measures per fold;
  - mean and population SD;
  - `run_contract.json` with input, code and output hashes.
- **Independent recomputation** (own sklearn code; labels joined by ID): output hashes OK, every
  row in exactly one test fold, and thresholds and per-fold measures within 1.1e-16.
- **Means match the pre-existing summary to 3 decimals:**

  | | AUROC | AUPRC | P | R | F |
  |---|---|---|---|---|---|
  | Recomputed | 0.8072 | 0.4106 | 0.3243 | 0.7509 | 0.4469 |
  | Pre-existing | 0.807 | 0.411 | 0.324 | 0.751 | 0.447 |

  Also recomputed: Hamming 0.0904, ranking error 0.2418, accuracy 0.1434.
- **Result:** `v1_rf_827_verification.json`, `passed: true`.

## Findings that do not change the frozen results

- **Paper agreement.**
  - IPCARF MLC matches Table III on all 8 measures after rounding.
  - RFLDA MLC matches on 5 of 8. The three differences are ranking error 0.252 vs 0.24,
    precision 0.3346 vs 0.34 (it rounds to 0.33), and accuracy 0.138 vs 0.17.
  - RFLDA's SDs are also smaller than the paper's.
  - The cause has not been tested or demonstrated. The protocol was not altered to improve
    agreement. Possible explanations, all untested:
    - library versions: scikit-learn 1.7.2 here vs the authors' pinned 1.5.1, which can change
      random streams and tree construction;
    - the authors' paper run may have used their committed 10 RFLDA feature-selection folds,
      whereas this reproduction used the paper's stated 5;
    - unknown differences in the authors' original environment or code at the time of the paper.
- **Plain RF.** Rerun and verified on 2026-10-05 (section above). The paper has no plain-RF row,
  so there is no paper value to compare it with.
- **The current working tree would not reproduce these results.** It contains uncommitted edits
  (IPCARF StandardScaler; RFLDA micro-AUPRC selection). The `code_used/` copies are the record of
  what ran, and they are hash-frozen.
- **Wording.** Older files (`FIDELITY.md`, `scripts/reproduce_paper_reference.py`) describe cols
  462–746 as leakage/association-derived. They were left unchanged. The new reports use: "The
  released matrix contains a 285-column binary block that numerically matches the raw association
  matrix, but its provenance conflicts with the paper's description and remains unresolved."
