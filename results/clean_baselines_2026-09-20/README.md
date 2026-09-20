# Clean baselines (2026-09-20)

Produced by `scripts/report_clean_baselines.py`. Reads saved per-fold predictions only; nothing was fit or refit.
Runs: `v1_core583_full10_oob_2026-08-24` (583 features) and `v1_all2023_full10_oob_2026-08-24`
(all_safe_concatenated). Both: 353 samples, 24 labels, 10 folds, RF, threshold_mode `train_oob_f1`.

`clean_baselines.csv`: one row per fold, then `mean` and `std` (population std, ddof=0) per run.

- micro_auroc / micro_auprc: pooled over the fold's test matrix. Mean matches `within_version_cv_summary.csv` exactly (0.6405 and 0.6113).
- macro_auroc: mean over labels; labels with one class in the test fold are skipped (`macro_labels_skipped`, 0 in every fold of both runs).
- persample_auroc: mean over test samples; samples with no positives (or all positives) are skipped (`persample_skipped`, mean 11.3 per fold).
- prior_micro_auroc / prior_micro_auprc: no-feature baseline, per-label training-fold positive rate as a constant score.
  Its AUROC is not 0.5 because scores differ across labels while micro pooling compares across labels.
- train_oob_f1_mean/std: F1 stored in the run summary under the train-OOB threshold (not recomputed).

Reading: the prior baseline (micro AUROC 0.664, AUPRC 0.127) matches or beats the RF on micro metrics
(0.641 / 0.141 for core583). The RF's macro AUROC of 0.553 is the label-level signal above chance.
