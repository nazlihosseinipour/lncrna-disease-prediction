# IPCARF (HMC) and IPCARF (MLC), 5 outer folds: alternative variant of the CIBCB2025 Table III reproduction

**Leakage-inflated reference (not a valid inductive result).** `features_preprocessed.csv` has a 285-column block that copies the lncRNA-disease association matrix, and precision, recall, F, hamming, accuracy and ranking error use a per-label Youden threshold fitted on each test fold's own labels (the authors' original `evaluate.py`, commit 73b21ee).

**Which protocol the paper describes.** The paper's text describes 10 outer folds; 5-fold is the inner CV used to pick the PCA size. This 5-outer-fold run is therefore an alternative variant, not the paper's protocol. The 10-fold run (`../ipcarf/`) is untouched and remains the reference.

Run: 2026-09-20 23:10 to 23:18 (MLC 1 min 40 s, HMC 6 min 29 s, 8 cores). Same code as the 10-fold run: authors' committed HEAD IPCARF (no StandardScaler), 150 trees, 5-fold inner CV over {2,4,8,16,32,64,128} components, seed 0, per-label Youden threshold fitted on the test fold with the authors' 73b21ee `evaluate.py`, HMC evaluated on the 44 original diseases. Only the outer split changed: 5 folds of 284 train / 71 test rows. Sanity checks (`logs/sanity_checks.log`): 37/37 pass.

Splits: `inductive_splits_5fold.csv`, same column format as the authors' file, made by `make_splits_5fold.py` (MultilabelStratifiedKFold, shuffle, seed 0, hierarchical label matrix `mesh_labels_final.csv`). Caveat: the same generator with n_splits=10 does not reproduce the authors' shipped 10-fold file (see FIDELITY.md), so the 5-fold and 10-fold partitions come from different stratifier runs, and the shipped file cannot be verified to come from the shipped label matrix.

## Mean (std) over folds: paper vs 10-fold vs 5-fold

|dev| = absolute difference of the mean from the paper's mean. "Closer" names the run with the smaller |dev|.

**IPCARF (HMC)**

| Metric | Paper | 10-fold | 5-fold | dev 10 | dev 5 | Closer |
|---|---|---|---|---|---|---|
| Hamming | 0.08 (0.02) | 0.083 (0.025) | 0.110 (0.025) | 0.003 | 0.030 | 10-fold |
| RankingErr | 0.24 (0.07) | 0.236 (0.066) | 0.293 (0.019) | 0.004 | 0.053 | 10-fold |
| AUROC | 0.83 (0.07) | 0.825 (0.067) | 0.832 (0.025) | 0.005 | 0.002 | 5-fold |
| AUPRC | 0.44 (0.13) | 0.436 (0.126) | 0.431 (0.081) | 0.004 | 0.009 | 10-fold |
| Precision | 0.36 (0.08) | 0.357 (0.079) | 0.282 (0.040) | 0.003 | 0.078 | 10-fold |
| Recall | 0.78 (0.13) | 0.782 (0.125) | 0.749 (0.055) | 0.002 | 0.031 | 10-fold |
| F | 0.48 (0.08) | 0.484 (0.082) | 0.408 (0.042) | 0.004 | 0.072 | 10-fold |
| Accuracy | 0.18 (0.08) | 0.181 (0.083) | 0.104 (0.067) | 0.001 | 0.076 | 10-fold |

**IPCARF (MLC)**

| Metric | Paper | 10-fold | 5-fold | dev 10 | dev 5 | Closer |
|---|---|---|---|---|---|---|
| Hamming | 0.08 (0.02) | 0.084 (0.025) | 0.122 (0.035) | 0.004 | 0.042 | 10-fold |
| RankingErr | 0.24 (0.05) | 0.245 (0.050) | 0.303 (0.021) | 0.005 | 0.063 | 10-fold |
| AUROC | 0.82 (0.06) | 0.823 (0.061) | 0.821 (0.022) | 0.003 | 0.001 | 5-fold |
| AUPRC | 0.44 (0.12) | 0.440 (0.117) | 0.427 (0.078) | 0.000 | 0.013 | 10-fold |
| Precision | 0.36 (0.09) | 0.356 (0.090) | 0.267 (0.059) | 0.004 | 0.093 | 10-fold |
| Recall | 0.77 (0.11) | 0.773 (0.107) | 0.752 (0.043) | 0.003 | 0.018 | 10-fold |
| F | 0.48 (0.09) | 0.483 (0.093) | 0.389 (0.062) | 0.003 | 0.091 | 10-fold |
| Accuracy | 0.14 (0.07) | 0.141 (0.074) | 0.116 (0.078) | 0.001 | 0.025 | 10-fold |

Reading: the 10-fold run is closer to the paper on 7 of 8 metrics for both variants (mean |dev| 0.003 vs 0.044 for HMC, 0.003 vs 0.043 for MLC). The 5-fold run is closer only on AUROC, where both runs already sit within 0.005 of the paper's 2-decimal value, so that difference is noise. The 5-fold run is worse on every threshold-dependent metric (precision, F, hamming, ranking error, accuracy fall short or overshoot by 0.03 to 0.09) and its stds are smaller than the paper's (e.g. AUROC std 0.02 vs 0.06 to 0.07). This supports 10 outer folds as the protocol behind Table III. With 5 folds each model trains on 284 rows instead of 319, and the Youden thresholds are fitted on 71-row test folds instead of 35 to 36, which changes precision and F most. Extra, not the Table III protocol: evaluating all 369 HMC nodes gives AUROC 0.848, AUPRC 0.497, F 0.466.

Files: `ipcarf_5fold_table3_reproduction.csv` (8 Table III columns, mean and std over 5 folds, both variants, with deviations from the paper), `ipcarf_5fold_table3_per_fold.csv`, `ipcarf_5fold_vs_10fold_vs_paper.csv` (comparison), `inductive_splits_5fold.csv`, `eval/` (authors' parse_results output), `predictions_mlc/`, `predictions_hmc/`, `logs/`, `code_used/` (same files as the 10-fold run), drivers `make_splits_5fold.py`, `run_all.sh`, `run_ipcarf_authors.py`, `run_eval_authors.sh`, `run_eval_authors.py`, `build_table.py`, `compare_table.py`, `sanity_checks.py`, and `FIDELITY.md`.
