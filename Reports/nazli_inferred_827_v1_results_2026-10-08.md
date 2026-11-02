# Nazli V1 run with the inferred 827-column construction

## Direct conclusion

Based on the files Felipe released, the 827-column input is not a valid sequence-only feature
construction for inductive RNA–disease prediction. Columns 462–746 reproduce the 285 known
RNA–disease outcomes, and 124 of the 355 released rows do not contain the reconstructed content
of their displayed RNA ID. The paper describes 827 sequence features, but the released artifact
does not match that description.

The numerical experiment is reproducible. A fresh, correctly aligned 827-column table was made
entirely from Nazli's processed V1 source data, without using Felipe's feature CSV as model input.
It scores much higher than both the paper and Felipe's released file. This is evidence that the
285 outcome columns give the models information about the associations they are asked to predict.

## What was generated

The model input has 355 RNAs and 827 columns:

| Block | Columns | Source |
|---|---:|---|
| DCC | 462 | Calculated from Nazli's processed V1 sequences |
| RNA–disease association outcomes | 285 | Nazli's processed V1 association table |
| Normalized dinucleotide frequencies | 16 | Calculated from Nazli's processed V1 sequences |
| Normalized trinucleotide frequencies | 64 | Calculated from Nazli's processed V1 sequences |

The generated matrix is
`results/nazli_inferred_827_v1_2026-10-08/inputs/v1_nazli_inferred_827_X.csv`.
Its SHA-256 is
`53e2da97ae248374fc877a49dc4546e3b044edc2cf2863c94ed522b8fb5c051c`.
The provenance file explicitly records that Felipe's feature CSV was not used.

The 285 association columns are a binary table. A value of 1 means that the source dataset
records an association between that RNA and that disease; 0 means that no association is
recorded in the table. These values come from the RNA–disease dataset rather than from the RNA
sequence. The prediction target is built from the same associations. Passing this block into X
therefore exposes outcome information before training and evaluation.

## Measured V1 results

All results below use the released ten folds and paper-style per-disease Youden evaluation.

| Input | Model | AUROC | AUPRC | F |
|---|---|---:|---:|---:|
| Nazli valid sequence-only 542 | RF | 0.628 | 0.087 | 0.204 |
| Nazli valid sequence-only 542 | IPCARF | 0.662 | 0.100 | 0.244 |
| Nazli valid sequence-only 542 | RFLDA | 0.639 | 0.087 | 0.206 |
| Nazli regenerated aligned 827 | RF | 0.966 | 0.834 | 0.734 |
| Nazli regenerated aligned 827 | IPCARF | **0.953** | **0.773** | **0.740** |
| Nazli regenerated aligned 827 | RFLDA | **0.975** | **0.852** | **0.774** |
| Felipe released 827 | RF | 0.807 | 0.411 | 0.447 |
| Felipe released 827 | IPCARF | 0.823 | 0.440 | 0.483 |
| Felipe released 827 | RFLDA | 0.811 | 0.413 | 0.460 |
| Paper Table 3 | IPCARF | 0.82 | 0.44 | 0.48 |
| Paper Table 3 | RFLDA | 0.81 | 0.41 | 0.46 |

The complete machine-readable comparison is
`results/nazli_inferred_827_v1_2026-10-08/report/v1_827_comparison.csv`.

## Why the corrected 827 result is higher than the paper

The regenerated table aligns every RNA with its own 285 association values. This makes the
outcome information unusually useful to the models and raises IPCARF and RFLDA far above the
paper values.

Felipe's released 827 file has a separate row-alignment defect. Direct reconstruction shows:

| Released row relationship | Rows |
|---|---:|
| Content belongs to the displayed RNA ID | 231 |
| Content belongs to a later source row | 121 |
| No reconstructed row matches within `1e-6` | 3 |

This means 124 displayed IDs are paired with feature content that is not their own reconstructed
row. That mismatch weakens the otherwise very strong association leakage and plausibly explains
why the released result falls near 0.82 instead of the 0.95–0.97 measured with correct alignment.
This explanation is supported by the artifact comparison and the measured run; it is an
inference because the original unpublished feature-generation history is unavailable.

## What this says about Nazli's preprocessing

The raw and processed V1 files contain the same 355 sequence strings and the same 285
association columns after trimming one trailing-space ID. Nazli's code can regenerate the 542
sequence-derived columns to the recovered construction. The low 542-feature scores therefore
do not show a preprocessing corruption. They show that the valid sequence-only input does not
receive the outcome information present in the released 827 artifact.

Reproducing or exceeding the paper number with the diagnostic 827 table confirms the artifact's
behavior. It does not validate the 827 table as a predictive method for unseen associations.
This issue occurs in the feature matrix before RF, IPCARF, RFLDA, or a neural network is fitted.

## V2 boundary

No V2 model was run. V2 has 542 sequence features and 436 disease outcome columns, which would
produce 978 columns. Selecting only 285 diseases to force 827 columns would have no supported
scientific rule, and adding all 436 outcomes as predictors would repeat the same leakage.

A valid V2 experiment should use the 542 sequence-derived features, or compute any
association-derived representation separately inside each training fold while excluding held-out
outcomes from its construction.

## Verification

- Input preparation checker: passed.
- Provenance assertions: passed.
- RF, IPCARF, and RFLDA: 30 of 30 fold cells completed.
- Scoring: three complete groups, 30 folds, zero rejected cells, zero incomplete groups.
- Focused tests cover input construction, the independent 827 reconstruction, experiment
  contracts, and rescoring: 27 passed.
