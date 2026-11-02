# Supervisor summary — Felipe V1/V2 (2026-09-28)

## 1. Existing full-population within-version results (unchanged, descriptive)

- V1→V1: 355 RNAs × 44 diseases. V2→V2: 4,130 RNAs × 436 diseases.
- Both use 542 sequence-only features, 10-fold CV and the paper models
  (`WITHIN6_VERIFIED_RESULTS_2026-09-28.md`).
- Micro AUROC: V1 0.63–0.66, V2 0.88. Every model is below a training-prevalence baseline, and
  per-disease AUROC is ≈ 0.5.
- **These are not a controlled comparison.** Different RNAs, different disease sets (44 vs 436)
  and different label frequencies drive the numbers. They were not modified; a hash check of all
  222 protected files confirms it.

## 2. New paired shared-sequence results (controlled)

- **Same RNAs.** The 48 sequences are present in both versions: identical normalized sequence
  (SHA-256) and bit-identical 542-feature vectors. The primary cohort is the 44 with a one-to-one
  ID mapping. The sensitivity cohort is all 48, with V2 labels OR-combined over four duplicated V2
  IDs.
- **Same outputs.** The 73 disease concepts shared by both versions. The map was re-audited against
  the raw columns, and all checks pass.
- **No leakage.** Every row carrying one of those sequences was removed from both training sets
  (V1 307 rows, V2 4,078 rows). There is zero sequence or exact feature-row overlap with the test
  rows.
- **2×2 design.** Each of RF/IPCARF/RFLDA is trained on V1 and on V2. Both trained models predict
  the identical test matrix. Each prediction set is scored against V1 labels and against V2 labels.
  Fixed 0.5 threshold; no Youden.

Result (primary cohort, micro AUROC):

| Train → labels | RF | IPCARF | RFLDA | Training-prevalence baseline |
|---|---|---|---|---|
| V1 → V1 | 0.590 | 0.623 | 0.648 | 0.715 |
| V1 → V2 | 0.636 | 0.669 | 0.666 | 0.720 |
| V2 → V1 | 0.666 | 0.663 | 0.668 | 0.667 |
| V2 → V2 | 0.666 | 0.671 | 0.677 | 0.797 |

- **No model beats the no-model baseline.** At best a model ties it (V2 → V1). Otherwise it is
  lower, and in 7 of 12 cells the 95% CI of the difference excludes 0.
- **Micro AUPRC is 0.04–0.13 for every model.** V2-trained models score higher AUPRC than
  V1-trained ones only against V1 labels: RF +0.056, IPCARF +0.083, CIs just above 0. Against V2
  labels there is no difference.
- **Per-disease AUROC median is 0.37–0.50.**
- **The sensitivity cohort gives the same picture.**

**Conclusion:** holding the RNA sequences, features and disease set fixed, the paper models show
no detectable sequence-based disease signal under this experimental design, beyond what label
frequency alone provides, whichever version they are trained on. The V1/V2 differences are
consistent with differences arising mainly from label annotations and prevalence. Of the
48 × 73 sequence-disease pairs, 107 are labelled differently in V1 and V2.

**Uncertainty:**
- Only 44–48 test sequences and 51–73 positive associations.
- 41–43 of the 73 diseases have no positive in the test set.
- AUROC CIs are ~0.15–0.25 wide.
- Differences under ~0.05 should not be interpreted.

## Approved decisions (Nazli, 2026-09-28)

1. The audited 73-disease map stays unchanged for the primary analysis. Possessive, plural,
   spelling and typo variants are documented as limitations. The label space was not modified
   after observing results.
2. The 44 one-to-one cohort is primary. The all-48 sensitivity cohort OR-combines labels across
   V2 IDs that share a sequence. The complete list of 11 conflicting labels is kept in
   `results/felipe_shared_sequence_2026-09-28/cohort/v2_label_conflicts.csv`.

## Process notes for the record

- **The earlier transfer design leaks by sequence.** It removed shared RNAs by exact ID only
  (1 of 48), leaving 47 identical sequences in the target population. The old `fromserver`
  bundle confirms the same design (`overlap_removed=False`). The paired design replaces it. The
  old transfer runs were not executed.
- **Scorer rounding fix.** A pandas float-parsing issue in reading prediction files was found by
  the independent verifier and fixed in the new scorer. Its effect on the verified within results
  is ≤ 6e-6 AUROC, so no reported value changes.
- **Supervisor error, repaired.** A resume test on the finished run moved and partly refitted the
  verified cells. The original cells were restored, the refits kept aside, and everything
  re-verified. The refits also confirmed determinism: identical predictions, apart from ≤ 1e-16
  threading noise on V2.
- **Artifacts:**
  - `SHARED_SEQUENCE_COHORT_AUDIT.md`
  - `SHARED_SEQUENCE_TRANSFER_RESULTS.md`
  - `shared_sequence_transfer_table.csv`
  - `shared_sequence_correspondence.csv`
  - `shared_sequence_run_manifest.json`
  - `SHARED_SEQUENCE_SPEC_2026-09-28.md` (decisions SS1–SS16)
  - outputs under `results/felipe_shared_sequence_2026-09-28/`
- Nothing was committed or pushed.
