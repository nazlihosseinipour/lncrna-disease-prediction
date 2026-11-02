# Paper-exact V1→V1 / V2→V2 protocol (2026-09-29)

Sources:
- Paper `Inductive_models_for_structured_output_prediction_of_lncRNA-disease_associations.pdf`
  (text in `tmp/pdfs/inductive_paper.txt`): Sections IV-A to IV-E, Tables I and III.
- The authors' repository (`lncRNA_CIBCB2025-main/`; public `github.com/fkenjinakano/lncRNA_CIBCB2025`,
  5 commits, same layout).
- Evaluation code at the authors' first commit `73b21ee`.
- Freeze taken before any work: `results/felipe_paper_exact_2026-09-29T0014/freeze/`
  (git status, HEAD `8111d81`, 584 file hashes).

## Protocol

| Item | Paper-faithful setting |
|---|---|
| Dataset population | V1: 355 lncRNAs (Table I). V2: not run (feature gate failed; see `V2_INPUT_AND_POPULATION_AUDIT.md`). |
| Feature matrix | V1: the released `create_dataset/datasets/features/features_preprocessed.csv`, 355 × 827, used as-is (sha256 `b264dd75…8887`). No row, disease or feature selection. |
| Label matrix | V1: `flat_labels/labels_flat_preprocessed.csv`, 355 × 44 (MLC; sha256 `53c3573b…7e9e`). |
| Folds | Released `splits/inductive/inductive_splits.csv`, 10 inductive folds (sha256 `e9fe2eb5…1e5a`). Rows are sliced by position (`iterator_cross_validation`). |
| Output mode | Multi-label (MLC): `binary_mode=False`, `transductive_mode=False`. |
| RF | `RandomForestRegressor(n_estimators=150, random_state=0)`, sklearn default `max_features`. **The paper reports no plain-RF row** (Table III has RF + ET only). |
| IPCARF | `IncrementalPCA(n_components)` → RF with 150 trees, seed 0. Components chosen from {2, 4, 8, 16, 32, 64, 128} by 5-fold inner CV on the training fold (GridSearchCV, default R²). No input scaling. |
| RFLDA | RF-importance-based feature count chosen by 5-fold inner CV in steps of 50 ({50, …, 800}; the 827 candidate was added and never selected). Selection by subset accuracy. RF with 150 trees, seed 0. |
| Threshold | Per-disease Youden cut-off from `roc_curve` on the **evaluated test fold's own labels** (paper [47]; authors' `sensivity_specifity_cutoff`). It is argmax(tpr−fpr) if Σ(tpr−fpr) > 0, else 0.5; single-class labels get 0.5. It is applied only in evaluation. |
| Metrics | Hamming loss, ranking error (authors' code: `label_ranking_loss` on the thresholded matrix), micro AUROC and micro AUPRC (on scores), micro precision/recall/F, and subset accuracy. |
| Aggregation | Per-fold values; mean and population SD (`np.std`) over 10 folds. |
| Seed | `random_state=0` for all forests. |

## Documented deviations of the authors' code from the paper text (kept for fidelity)

- **RFLDA top-n features.** Importances are indexed by position and never sorted, so "top-n"
  means the first n columns in file order.
- **Inner CV folds.** The committed RFLDA code uses 10 inner folds. The reproduction used the
  paper's 5, via a recorded patch.
- **Ranking error** is computed on thresholded predictions.
- **The Youden fallback** tests the *sum* of tpr−fpr rather than its maximum.
- **Thresholds are fitted on test labels.** Thresholded measures are therefore optimistic.
  AUROC/AUPRC do not depend on the threshold.
