# Summary for Felipe: paper-exact V1→V1 and V2→V2 (updated 2026-10-05)

## Status at a glance

| Analysis | Status |
|---|---|
| V1→V1, released 827-feature MLC construction | **Complete** for RF, IPCARF and RFLDA, each verified from saved fold-level probabilities |
| V2→V2, paper-exact 827 | **Blocked.** The 827-feature construction cannot be recovered. |
| V2→V2, 542 reproducible sequence features | Already complete. **Supplementary.** |
| V1↔V2 cross-version | Already analysed on the 44 one-to-one pairs, with a 48-sequence sensitivity cohort. **Supplementary.** |
| Exact V1/V2 sequence intersection | **48** shared unique sequences, not 44 |

## 1. Paper-faithful V1→V1 (released 355 × 827, 44 diseases, 10 released folds)

Settings:
- Paper model settings (150 trees, `random_state=0`, MLC: `binary_mode=False`,
  `transductive_mode=False`).
- The paper's evaluation code: per-disease Youden threshold on the evaluated fold; mean and
  population SD over 10 folds.

| Model (MLC) | micro AUROC | micro AUPRC | Precision | Recall | F-score | Paper (AUROC/AUPRC/P/R/F) |
|---|---|---|---|---|---|---|
| RF (plain) | 0.807 (0.067) | 0.411 (0.109) | 0.324 | 0.751 | 0.447 | no plain-RF row in the paper |
| IPCARF | 0.823 (0.061) | 0.440 (0.117) | 0.356 | 0.773 | 0.483 | 0.82 / 0.44 / 0.36 / 0.77 / 0.48 |
| RFLDA | 0.811 (0.062) | 0.413 (0.098) | 0.335 | 0.748 | 0.460 | 0.81 / 0.41 / 0.34 / 0.75 / 0.46 |

- **RF** was run on 2026-10-05. It saves probabilities with RNA IDs, the thresholds, all eight
  measures per fold, and a hash contract. An independent recomputation matches within 1e-16 and
  equals the earlier RF summary.
- **IPCARF and RFLDA** are the existing runs, verified without rerunning.
  - IPCARF matches Table III on all 8 measures.
  - RFLDA matches 5 of 8. It differs on ranking error (0.252 vs 0.24), precision (0.3346 vs 0.34)
    and accuracy (0.138 vs 0.17). The cause is untested; possible explanations are in
    `VERIFICATION_REPORT.md`.
- **Thresholds.** Precision, recall and F-score use the paper's test-fold Youden threshold, so
  they are optimistic. AUROC and AUPRC do not depend on the threshold.

## 2. Supplementary analyses (not the complete paper 827-feature construction)

None of these was rerun, deleted or changed.

- **542-feature V1→V1 and V2→V2.** 355 × 44 and 4,130 × 436, using the 542 columns rebuilt from
  sequences (DCC + 2/3-mers). See
  `Reports/felipe_audit_2026-09-27/WITHIN6_VERIFIED_RESULTS_2026-09-28.md`. Every model there was
  below a training-prevalence baseline.
- **Shared-sequence V1↔V2.**
  - Primary cohort: the 44 one-to-one pairs.
  - Sensitivity cohort: all 48 sequences.
  - Report: `Reports/felipe_audit_2026-09-27/SHARED_SEQUENCE_TRANSFER_RESULTS.md`.
- **Exact intersection:**
  - 48 unique sequence hashes occur in both datasets;
  - 44 have a clean one-to-one correspondence;
  - four shared hashes map to multiple V2 identifiers;
  - those four produce eight V2 rows, giving 52 corresponding V2 rows in total.

## 3. Unresolved discrepancy: paper description vs released matrix

- **What the paper says.** All 827 features are Pse-in-One sequence features: "eight" vectors,
  of which seven are named.
- **What the released matrix contains:**
  - DCC with 22 properties at lag 1 (462 columns, reproduced exactly);
  - 2-mer and 3-mer frequencies (80, reproduced);
  - a 285-column binary block (cols 462–746).
- **Absent.** Mismatch, DAC, DACC, PC-PseDNC-General and SC-PseDNC-General were not found.
- **The binary block.** The released matrix contains a 285-column binary block that numerically
  matches the raw association matrix, but its provenance conflicts with the paper's description
  and remains unresolved.
- **Row offset.** From row 231 on, the released feature rows are offset by one or two positions
  relative to the RNA IDs and labels, and 3 rows match no raw sequence. This is kept as-is in the
  V1 runs.

## 4. Missing Pse-in-One configuration (blocks exact V2)

We searched the repository and its history, the server bundle, the authors' public GitHub
repository, and the Pse-in-One documentation. We found no record of:
- the Pse-in-One version;
- the commands;
- the descriptor parameters;
- the eighth vector;
- the column order and normalization;
- the handling of ambiguous bases;
- how cols 462–746 were generated.

So no 827-feature V2 matrix was built:
- no guessed parameters;
- no labels appended to X;
- the 542-feature V2 run is not presented as paper-exact.

The draft request to Felipe is in `EMAIL_DRAFT_TO_FELIPE.md`. It has not been sent.
