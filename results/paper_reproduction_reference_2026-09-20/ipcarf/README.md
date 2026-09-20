# IPCARF (HMC) and IPCARF (MLC): reproduction of CIBCB2025 Table III with the authors' own code

**Leakage-inflated reference (not a valid inductive result).** `features_preprocessed.csv` has a 285-column block that copies the lncRNA-disease association matrix, and precision, recall, F, hamming, accuracy and ranking error use a per-label Youden threshold fitted on each test fold's own labels (the authors' original `evaluate.py`, commit 73b21ee).

Run: 2026-09-20, 10 inductive folds, 150 trees, 5-fold inner CV over {2,4,8,16,32,64,128} components, no PCA scaling. Runtime: MLC 5 min, HMC 18 min (8 cores, n_jobs=-1). Sanity checks (`logs/sanity_checks.log`): 48/48 pass.

Mean (std) over folds, ours vs paper; deviation = ours - paper mean.

| Variant | Hamming | RankingErr | AUROC | AUPRC | Precision | Recall | F | Accuracy |
|---|---|---|---|---|---|---|---|---|
| HMC ours | 0.083 (0.025) | 0.236 (0.066) | 0.825 (0.067) | 0.436 (0.126) | 0.357 (0.079) | 0.782 (0.125) | 0.484 (0.082) | 0.181 (0.083) |
| HMC paper | 0.08 (0.02) | 0.24 (0.07) | 0.83 (0.07) | 0.44 (0.13) | 0.36 (0.08) | 0.78 (0.13) | 0.48 (0.08) | 0.18 (0.08) |
| HMC deviation | +0.003 | -0.004 | -0.005 | -0.004 | -0.003 | +0.002 | +0.004 | +0.001 |
| MLC ours | 0.084 (0.025) | 0.245 (0.050) | 0.823 (0.061) | 0.440 (0.117) | 0.356 (0.090) | 0.773 (0.107) | 0.483 (0.093) | 0.141 (0.074) |
| MLC paper | 0.08 (0.02) | 0.24 (0.05) | 0.82 (0.06) | 0.44 (0.12) | 0.36 (0.09) | 0.77 (0.11) | 0.48 (0.09) | 0.14 (0.07) |
| MLC deviation | +0.004 | +0.005 | +0.003 | -0.000 | -0.004 | +0.003 | +0.003 | +0.001 |

All 16 means agree with the paper at its 2-decimal precision (max |dev| 0.005), and the stds agree too. HMC = predictions of the 369-node MeSH label space restricted to the 44 original diseases (paper Sec. III-C; authors' `evaluate_flat_mesh`). Extra, not the Table III protocol: evaluating all 369 nodes gives AUROC 0.844, AUPRC 0.507, F 0.529.

Files: `ipcarf_table3_reproduction.csv` (table with deviations), `ipcarf_table3_per_fold.csv`, `eval/` (authors' parse_results output), `predictions_mlc/`, `predictions_hmc/`, `logs/`, `code_used/`, drivers `run_all.sh`, `run_ipcarf_authors.py`, `run_eval_authors.sh`, `build_table.py`, `sanity_checks.py`, and `FIDELITY.md` (step-by-step match table). `_prior_attempt_20260920_2004/` holds an earlier same-day run of the same code; its predictions are byte-identical to this run's (the run is deterministic).
