# Paper match and change report, 2026-09-20

## What changed

Two new scripts, two gitignored results folders, and this report. Nothing existing was edited.

- `scripts/reproduce_paper_reference.py` reproduces the CIBCB2025 Random Forest numbers from the authors' own files.
- `scripts/report_clean_baselines.py` reads saved fold predictions and adds the baselines that were missing beside our micro numbers.
- `results/paper_reproduction_reference_2026-09-20/` and `results/clean_baselines_2026-09-20/` hold the outputs.

The clean pipeline, the Evaluator defaults, the manifests, RFLDA and IPCARF are untouched. The paper's leaky feature block and its test-fitted threshold live only inside the reference script.

## Reference reproduction

Setup: plain Random Forest, 150 trees, seed 0, the paper's own X and Y (355 x 44), its inductive splits, 10 folds. Micro metrics are averaged over folds. Precision, recall and F1 use a per-label Youden threshold fitted on each test fold, as the authors' `evaluate.py` at 73b21ee does, including its rule that falls back to 0.5 when sum(tpr - fpr) is not positive.

- Prior only (train prevalence, no features): AUROC 0.704, AUPRC 0.112, F1 0.000.
- Paper 542 features (block removed): AUROC 0.615, AUPRC 0.081, precision 0.148, recall 0.359, F1 0.198. The threshold is fitted on the test fold, so P/R/F1 are in-sample.
- Paper full 827 features: AUROC 0.807, AUPRC 0.411, precision 0.324, recall 0.751, F1 0.447, fixed-0.5 F1 0.316. This is a leakage-inflated reference (not a valid inductive result).
- Block only, 285 features: AUROC 0.820, AUPRC 0.436, F1 0.551. This is a leakage-inflated reference (not a valid inductive result).
- Paper Table III ranges are AUROC 0.76-0.83, AUPRC 0.31-0.44, F1 0.38-0.48. The 827 row falls inside all three, which makes it a leakage-inflated reference (not a valid inductive result).

Take out the 285-column block and the paper's own pipeline scores 0.615 AUROC, below the feature-free prior at 0.704.

## Why the 827 row scores high

Three findings explain it.

1. Columns 462-746 of the 827-column file equal the raw association matrix, in the same column order. Rows 0-230 equal each row's own label vector in 231 of 231 cases. Later rows are shifted by 1 or 2 because `AK153778.1` and `BX374997.2 ` were dropped from the 353-row set before the block was pasted into a 355-row frame. The model receives each test lncRNA's own answer as input.
2. No script in the repo builds this file, so its provenance is undocumented.
3. The F1 threshold is fitted on the test fold's own labels. That rule alone moves clean F1 substantially, and it applies again on top of the leak.

## First run correction

My first run of the reference script omitted the authors' "sum > 0 else 0.5" guard. It gave F1 0.381 (precision 0.252, recall 0.811), which sat barely inside the Table III range. I found the difference by comparing against the authors' code and fixed it. AUROC and AUPRC did not change. The F1 of 0.447 above comes from the corrected script.

## Clean numbers

All from saved predictions: 24 labels, 353 samples, 10 folds, Random Forest, train-fitted threshold (`train_oob_f1`).

- core583 features: micro AUROC 0.6405, micro AUPRC 0.1411, macro AUROC 0.5531, per-sample AUROC 0.6769, F1 0.1793.
- all2023 features: micro AUROC 0.6113, micro AUPRC 0.1284, macro AUROC 0.5244, per-sample AUROC 0.6640, F1 0.1749.
- Train-prevalence prior with no features, same folds: micro AUROC 0.6636, micro AUPRC 0.1266.

The prior matches or beats the Random Forest on both micro metrics. Micro pooling rewards getting label prevalence right, and that needs no sequence signal. Macro AUROC of 0.55 (core583) and 0.52 (all2023) is the only per-label signal above chance, and it is small. The core583 macro std is 0.056 and the all2023 macro std is 0.103 across folds.

On the 44 paper labels the prior scores 0.704 and the clean Random Forest scores 0.664. These two figures come from the earlier audits. Saved predictions for 44 labels do not exist, so the new script cannot regenerate them.

Under the train-fitted threshold, F1 is 0.179 for core583. The earlier fixed-0.5 figure was 0.037. The leakage-inflated reference (not a valid inductive result) reports F1 0.447 with a threshold fitted on the answers, so the two F1 values cannot be compared.

## Deliberately not done

- No RFLDA or IPCARF reproduction. Plain Random Forest already lands inside every Table III range with the same leak, so extra rows would cost hours of nested cross-validation and change no conclusion.
- No feeding the leaky block into clean models.
- No Random Forest regularization, fold-safe neighbour features, or reproduction of the 542 feature families. Earlier tests showed no gain or a loss. The other 542 families scored 0.049 AUROC and 0.023 AUPRC below our 583 features on identical labels and folds.

## Ceiling

Leak-free inductive prediction on these 353 lncRNAs should give micro AUROC 0.66-0.72, macro AUROC 0.52-0.60, AUPRC 0.10-0.16, and F1 0.12-0.20 under a train-fitted threshold. About 35 test rows per fold set the binding limit. A published figure near 0.81 is out of reach without target-derived features.

## Unverified

I have not confirmed that the published numbers came from exactly this file. The circumstantial evidence is strong: 827 equals 542 plus 285, the block sum of 1132 matches the interaction count in Paper Table I, and precision, recall and F1 reproduce to two decimals. The authors can settle it with one question, and I suggest asking them before the leak is written up as a finding.
