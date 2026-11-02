# V2 input and population audit (2026-09-29)

**Status: V2→V2 paper-faithful run BLOCKED.** No V2 model was fitted.

Stop conditions met (from the task specification):
1. The 827-feature V1 construction cannot be reproduced (`V1_FEATURE_RECONSTRUCTION_AUDIT.md`).
2. The exact feature parameters are unknown (`PSE_IN_ONE_FEATURE_CONTRACT.csv`).

We therefore did not do any of the following:
- build an 827-column V2 matrix;
- append labels to V2 features;
- substitute the 542-feature matrix.

## V2 inputs (hash-frozen; not modified)

- **Raw file:** `Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv`
  - 5,338 rows; columns `ID`, `seqs` and 436 disease columns.
  - Hash recorded in `results/felipe_paper_exact_2026-09-29T0014/freeze/protected_before.sha256`.
- **Rows without a transcript sequence: 1,208.** Every one is listed by ID in
  `config/v2_missing_sequence_ledger.csv` (reason `missing_sequence`; 19 rat IDs flagged
  `species_mismatch_suspected`).
- **Rows with a sequence:** 4,130 (population `v2_sequence_available_4130`).
- **Duplicate sequences:** 205 identical-sequence groups covering 486 rows.
- **Group-aware folds:** a 10-fold split exists that keeps identical sequences together
  (`results/felipe_basics_2026-09-27/inputs/v2_all_436_group_splits.csv`). It would be the
  leakage safeguard for any future V2 run.
- **Diseases:** 436, with no filter applied. 10 of them have zero positives in the 4,130
  population.

## Population choice (open item, not decided here)

A paper-faithful V2 run would need two decisions from Nazli/Felipe:
- whether to use the 4,130 RNAs that have a sequence, or to recover the 1,208 missing sequences
  from an authoritative source (missing sequences are never zero-filled or substituted);
- whether to run group-aware folds only, or also the paper's ordinary split as a separately
  labelled sensitivity run.

Neither decision matters until the feature construction is known.
