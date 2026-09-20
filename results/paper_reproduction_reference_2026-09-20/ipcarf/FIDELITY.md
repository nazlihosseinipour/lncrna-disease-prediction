# FIDELITY: IPCARF (MLC / HMC) reference reproduction vs the authors' pipeline

Leakage-inflated reference (not a valid inductive result). `features_preprocessed.csv` carries a 285-column block that copies the association matrix.

Working copy of the authors' code: `scratchpad/ipcarf_fresh/authors_copy` (cp -R of `lncRNA_CIBCB2025-main`, then the files below replaced). Nothing under the project tree was modified. Exact files used are in `code_used/`.

## Authors' HMC pipeline, step by step, and whether this run matched

| # | Authors' step (source) | This run | Match |
|---|---|---|---|
| 1 | Features: `pd.read_csv(features_preprocessed.csv, index_col=0)`, 355 x 827, used as-is (`IPCARF/main.py:15`). No scaling outside the model. | Same file, same call (authors' `main.py` executed verbatim). | exact |
| 2 | Flat labels `labels_flat_preprocessed.csv` (355 x 44, MLC). Built by `preprocess/main.py` (keep labels with >5 lncRNAs, merge cancer into lymphoma, rename to MeSH). | Shipped file used, not regenerated. | exact |
| 3 | Hierarchical labels `mesh_labels_final.csv` (355 x 369). Built by `mesh/main.py`: for each of the 44 diseases take every MeSH tree number, add all dotted-prefix ancestors, add the disease name itself; row = union over the lncRNA's diseases. The 44 flat columns are present in it by name and equal the flat matrix (checked). | Shipped file used, not regenerated (the third step of the README run order, `filter_flat_hierarchical`, is not in the repo). | exact (as shipped) |
| 4 | Splits `inductive_splits.csv` (10 folds, `MultilabelStratifiedKFold(10, shuffle, seed 0)` run on the hierarchical Y); `iterator_cross_validation` slices X and Y with `.iloc[position]`. | Same file and function. | exact |
| 5 | Per fold: `IPCARF(binary_mode=False, transductive_mode=False)`; `fit(x_train, y_train)`; converter is the identity in this mode; `predict_proba(x_test, y_test)` ignores `y_test` in multilabel mode, returns `RandomForestRegressor.predict` scores. The same model class and call are used for flat (MLC) and hierarchical (HMC) labels; the HMC model is trained on all 369 outputs. | Authors' `main.py` executed verbatim except `range(n_folds)` became `FOLDS` (all 0..9). | exact |
| 6 | Model: `IncrementalPCA(n_components)` then `RandomForestRegressor(n_estimators, random_state=0, n_jobs=-1)`; `n_components` chosen per fold by `GridSearchCV(pipe, {pca__n_components: grid})`, default scoring (regressor R^2), on the training fold only. | Same structure. Hyperparameters below. | see below |
| 7 | Predictions saved per fold with column names = label names (`main.py:51`). | Same. | exact |
| 8 | HMC -> original disease space: paper Sec. III-C "remove the prediction probabilities related to the added diseases". Code: `Evaluator.evaluate_flat_mesh` keeps `y_pred_mesh[flat_columns]` by name, evaluated against the flat 44-column test labels (`parse_results/main.py:52`, output `*_hierarchical_to_flat_performance.csv`). So only the original 44 columns are evaluated; Youden thresholds are fitted on those 44. | Authors' `parse_results/main.py` (73b21ee text), unmodified. Table III HMC row = the `hierarchical_to_flat` file. | exact |
| 9 | Evaluation per fold: `sensivity_specifity_cutoff` = per-label `roc_curve(y_test[c], score[c])`, threshold = argmax(tpr - fpr) if `sum(tpr - fpr) > 0` else 0.5 (fitted on the TEST fold); predictions thresholded per label; hamming, `label_ranking_loss` on the THRESHOLDED matrix, micro AUROC, micro AUPRC (raw scores), micro P/R/F, subset accuracy; mean and population std (np.std) over 10 folds. | `evaluate.py` and `parse_results/main.py` taken from authors' first commit 73b21ee (pre-"fixed"-mode). | exact |
| 10 | HMC over all 369 nodes (`*_multilabel_performance.csv` from the same script) is a second output of the authors' code. Paper text says predictions are converted back, so the Table III HMC row is the 44-column version. | Reported as an extra row only, labelled as not the Table III protocol. | as reported |

## Hyperparameters: authors' first commit (73b21ee) vs repo HEAD vs this run

| Item | 73b21ee (debug) | HEAD | This run | Paper |
|---|---|---|---|---|
| RF trees (`n_estimators`) | 10 (150 commented out) | 150 | 150 | 150 |
| Inner-CV folds | `n_folds = 2` defined but never passed to `GridSearchCV`, which used its default `cv=5` (so 5-fold in effect); the 2 was dead code | `cv=self.n_folds = 5` | 5 | 5 |
| Component grid | `[2]` (7-value grid commented out) | `[2,4,8,16,32,64,128]`, filtered to values <= min(n_rows, n_cols) | same as HEAD; on 255-row inner training folds all 7 values survive | {2,...,128} |
| PCA input scaling | none | none in committed HEAD | none | not stated |
| Uncommitted local edit (StandardScaler before PCA) | n/a | absent (working-tree only) | not used; the copy carries the committed file | n/a |
| `GridSearchCV` `n_jobs` | None | -1 | -1 (speed only) | n/a |
| Final PCA fit | `pca.fit` then `pipe.fit` (refit) | one pipeline fit | one fit; identical result | n/a |
| Input sanitising (`_sanitize_x`: coerce numeric, inf/NaN -> 0) | absent | present | present; X has no NaN, so a no-op | n/a |
| PCA-failure fallback (randomized PCA) | absent | present | present; never triggered (log grep for `FALLBACK` is empty) | n/a |
| RF `random_state` | 0 | 0 | 0 | not stated |

Instrumentation added to the copy: print of the mean inner-CV score per component value, print on fallback. No effect on results.

## Remaining differences (cannot be removed or are unverifiable)

- Library versions: this venv has scikit-learn 1.7.2, numpy 2.3.1, pandas 2.3.3, Python 3.11; authors' `requirements.txt` pins scikit-learn 1.5.1, numpy 1.26.4, pandas 2.2.2 (their `.python-version` is 3.12). Tree building order is seed-controlled, but version drift can change random streams and PCA solver details.
- The authors' original released code is not available to us apart from the repo history: 73b21ee is the earliest commit in this repo, which already carries debug settings; the real settings are taken from the paper text (150 trees, 5-fold, grid 2..128).
- `label_ranking_loss` is computed on the thresholded matrix (authors' original behaviour), which is not a ranking metric on scores. Kept on purpose for fidelity.
- Threshold is fitted on the evaluated test fold (authors' original behaviour). Precision, recall, F, hamming, accuracy and ranking error are therefore optimistic. AUROC and AUPRC do not use the threshold.
- The 285-column association-matrix block in the features leaks test-lncRNA labels into X. Kept on purpose (reference reproduction).
- Fold-1 inner-CV scores were compared with an earlier same-day run of the same code (`_prior_attempt_20260920_2004/`): identical, so the run is deterministic.
