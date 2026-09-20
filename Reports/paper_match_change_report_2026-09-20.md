# Paper match and change report, 2026-09-20

## What changed

Two new scripts, several gitignored results folders, and this report. Nothing existing was edited.

- `scripts/reproduce_paper_reference.py` reproduces the CIBCB2025 Random Forest numbers from the authors' own files.
- `scripts/report_clean_baselines.py` reads saved fold predictions and adds the baselines that were missing beside our micro numbers.
- `results/paper_reproduction_reference_2026-09-20/` holds the reference runs: plain Random Forest, `ipcarf/` (10 outer folds) and `ipcarf_5fold/` (a 5-fold variant). `results/clean_baselines_2026-09-20/` holds the clean baselines.

The clean pipeline, the Evaluator defaults and the manifests are untouched. The paper's leaky feature block and its test-fitted threshold live only inside the reference runs. The IPCARF runs use a scratch copy of the authors' code.

## Reference reproduction, plain Random Forest

Setup: plain Random Forest, 150 trees, seed 0, the paper's own X and Y (355 x 44), its inductive splits, 10 folds. Precision, recall and F1 use a per-label Youden threshold fitted on each test fold, as the authors' `evaluate.py` at 73b21ee does, including its rule that falls back to 0.5 when sum(tpr - fpr) is not positive.

- Prior only (train prevalence, no features): AUROC 0.704, AUPRC 0.112, F1 0.000.
- Paper 542 features (block removed): AUROC 0.615, AUPRC 0.081, precision 0.148, recall 0.359, F1 0.198. The threshold is fitted on the test fold, so P/R/F1 are in-sample.
- Paper full 827 features: AUROC 0.807, AUPRC 0.411, precision 0.324, recall 0.751, F1 0.447, fixed-0.5 F1 0.316. This is a leakage-inflated reference (not a valid inductive result).
- Block only, 285 features: AUROC 0.820, AUPRC 0.436, F1 0.551. This is a leakage-inflated reference (not a valid inductive result).
- Paper Table III ranges are AUROC 0.76-0.83, AUPRC 0.31-0.44, F1 0.38-0.48. The 827 row falls inside all three, which makes it a leakage-inflated reference (not a valid inductive result).

Take out the 285-column block and the paper's own pipeline scores 0.615 AUROC, below the feature-free prior at 0.704.

## IPCARF reproduction with the authors' code

I ran the authors' IPCARF code on their features, labels and shipped `inductive_splits.csv`: 150 trees, 5-fold inner cross-validation over {2, 4, 8, 16, 32, 64, 128} PCA components, seed 0, 10 outer folds, and the authors' Youden evaluation fitted on each test fold. The HMC variant trains on the 369-node MeSH label space and scores the 44 original diseases after dropping the added MeSH nodes, as paper Section III-C describes. These runs are a leakage-inflated reference (not a valid inductive result).

Means over 10 folds, ours then paper (Hamming, ranking error, AUROC, AUPRC, precision, recall, F, accuracy):

- IPCARF (HMC): 0.083 / 0.236 / 0.825 / 0.436 / 0.357 / 0.782 / 0.484 / 0.181 against 0.08 / 0.24 / 0.83 / 0.44 / 0.36 / 0.78 / 0.48 / 0.18.
- IPCARF (MLC): 0.084 / 0.245 / 0.823 / 0.440 / 0.356 / 0.773 / 0.483 / 0.141 against 0.08 / 0.24 / 0.82 / 0.44 / 0.36 / 0.77 / 0.48 / 0.14.

All 16 means agree at the paper's two-decimal precision, with a largest difference of 0.005, and the standard deviations agree as well. Evaluating all 369 HMC nodes instead gives AUROC 0.844, AUPRC 0.507 and F 0.529, which is not the Table III protocol.

Checks. An independent recomputation of every metric from the saved predictions found a maximum difference of 0.00005. Three folds refit from scratch gave identical predictions. All 48 sanity checks pass.

## The 5-fold outer variant

The paper text describes 10 outer folds and uses 5-fold cross-validation only inside them, to pick the PCA size. To test the other reading I ran the same code with 5 outer folds (284 train and 71 test rows per fold) in `ipcarf_5fold/`. This is also a leakage-inflated reference (not a valid inductive result).

AUROC and AUPRC stay close to the paper (HMC AUROC 0.832, AUPRC 0.431; MLC AUROC 0.821, AUPRC 0.427). The threshold-dependent metrics move away from it. HMC precision falls to 0.282 and F to 0.408 (paper 0.36 and 0.48); MLC precision falls to 0.267 and F to 0.389 (paper 0.36 and 0.48). Precision, F and Hamming miss by 0.03 to 0.09. Mean absolute deviation from the paper is 0.044 for HMC and 0.043 for MLC, against 0.003 for both at 10 folds. The 5-fold run sits closer only on AUROC, where both runs are already within 0.005. Larger test folds (71 rows against 35 to 36) probably account for the shift.

Caveat. The authors' split generator (`MultilabelStratifiedKFold`, shuffle, seed 0, hierarchical labels) does not reproduce their shipped `inductive_splits.csv`: 0 of 10 test folds match, including after I tried the flat matrix, sorted and reversed columns, and several column filters. The 10-fold versus 5-fold gap therefore mixes the fold count with a different stratifier run. The repo has one commit holding both files, so history cannot show which label matrix produced the shipped splits.

## Caveats on the IPCARF match

- Agreement to 0.005 is weak evidence about the exact protocol. Reseeding the model with `random_state=1` breaks the 10-fold match: MLC AUPRC drops to 0.424 against the paper's 0.44.
- Library versions differ. I used scikit-learn 1.7.2 and numpy 2.3.1; the authors pin 1.5.1 and 1.26.4.
- The label space, meaning which diseases and MeSH nodes are kept, is defined over all 355 rows before any split. This is a third paper-level leak, after the feature block and the test-fitted threshold.
- Every IPCARF number above is a leakage-inflated reference (not a valid inductive result).

## What was saved

Each run keeps out-of-fold predictions per fold, the PCA components chosen per fold, the exact code used, and the logs. No fitted model objects are stored. Runs are deterministic at seed 0. The results folders are gitignored. The small tables, scripts and logs are force-added to the repository, and the prediction directories stay local.

## Why the 827 row scores high

Three findings explain it.

1. Columns 462-746 of the 827-column file equal the raw association matrix, in the same column order. Rows 0-230 equal each row's own label vector in 231 of 231 cases. Later rows are shifted by 1 or 2 because `AK153778.1` and `BX374997.2 ` were dropped from the 353-row set before the block was pasted into a 355-row frame. The model receives each test lncRNA's own answer as input.
2. No script in the repo builds this file.
3. The F1 threshold is fitted on the test fold's own labels. That rule alone moves clean F1 substantially, and it applies again on top of the leak.

## First run correction

My first run of the plain Random Forest reference script omitted the authors' "sum > 0 else 0.5" guard. It gave F1 0.381 (precision 0.252, recall 0.811), which sat barely inside the Table III range. I found the difference by comparing against the authors' code and fixed it. AUROC and AUPRC did not change. The F1 of 0.447 above comes from the corrected script.

## Clean numbers

All from saved predictions: 24 labels, 353 samples, 10 folds, Random Forest, train-fitted threshold (`train_oob_f1`).

- core583 features: micro AUROC 0.6405, micro AUPRC 0.1411, macro AUROC 0.5531, per-sample AUROC 0.6769, F1 0.1793.
- all2023 features: micro AUROC 0.6113, micro AUPRC 0.1284, macro AUROC 0.5244, per-sample AUROC 0.6640, F1 0.1749.
- Train-prevalence prior with no features, same folds: micro AUROC 0.6636, micro AUPRC 0.1266.

The prior matches or beats the Random Forest on both micro metrics. Micro pooling rewards getting label prevalence right, and that needs no sequence signal. Macro AUROC of 0.55 (core583) and 0.52 (all2023) is the only per-label signal above chance, and it is small.

On the 44 paper labels the prior scores 0.704 and the clean Random Forest scores 0.664. These two figures come from the earlier audits. Saved predictions for 44 labels do not exist, so the new script cannot regenerate them.

Under the train-fitted threshold, F1 is 0.179 for core583. The earlier fixed-0.5 figure was 0.037. The leakage-inflated reference (not a valid inductive result) reports F1 0.447 with a threshold fitted on the answers, so the two F1 values cannot be compared.

## Ceiling

Leak-free inductive prediction on these 353 lncRNAs should give micro AUROC 0.66-0.72, macro AUROC 0.52-0.60, AUPRC 0.10-0.16, and F1 0.12-0.20 under a train-fitted threshold. About 35 test rows per fold set the binding limit. A published figure near 0.81 is out of reach without target-derived features.

## Deliberately not done

- No feeding the leaky block into clean models.
- No Random Forest regularization, fold-safe neighbour features, or reproduction of the 542 feature families. The other 542 families scored 0.049 AUROC and 0.023 AUPRC below our 583 features on identical labels and folds.

## Unverified

I have not confirmed that the published numbers came from exactly this file. The circumstantial evidence is strong. The number 827 equals 542 plus 285, the block sum of 1132 matches the interaction count in Paper Table I, and the plain Random Forest and both IPCARF variants reproduce Table III to its printed precision. The 10-fold IPCARF match does not identify the file on its own, for the seed reason above. The authors can settle it with one question, and I suggest asking them before the leak is written up as a finding.

## Still open

- RFLDA and IPCARF/RFLDA binary rows of Table III are running separately. Results go in `table3_other_rows/` and a later commit will add them.
- Clean-pipeline improvements.
- Contacting the authors.
- The push is on hold. The repo is public, and this report documents leakage in the authors' released data.
