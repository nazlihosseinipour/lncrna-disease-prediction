# AI handoff: clean and evaluate Nazli's alternative V1 features

## Objective

Determine whether Nazli's handcrafted and learned RNA features add useful, leakage-free predictive signal beyond the reconstructed Felipe V1 sequence baseline. Clean the feature pipeline, build documented V1 candidate matrices, and run a controlled feature-family ablation using identical samples, folds, models, seeds, and metrics.

This task is restricted to V1. Do not start V2 work.

## Repository

Project root:

`/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project`

Inspect the current Git status before editing. The checkout contains extensive existing and untracked work. Preserve it. Do not reset, delete, overwrite, or reformat unrelated files. If a branch is needed, use the `nazli/` prefix.

## Established facts to preserve

1. The historical Nazli RNA OOP run contains 353 RNAs and 729 descriptor dimensions, excluding `sample_id`.
2. The 729 dimensions are:
   - k-mer k=1,2,3,4: 4 + 16 + 64 + 256
   - reverse-complement k-mer k=1,2,3,4: 2 + 10 + 32 + 136
   - ordinary PseDNC: 18
   - DAC: 6
   - DCC: 6
   - DACC: 12
   - mono-, di-, and trinucleotide composition: 4 + 16 + 64
   - Z-curve: 3
   - mono-mono and mono-di K-gap: 16 + 64
3. The historical run used raw counts. k-mer k=1/2/3 duplicate mono-/di-/trinucleotide composition in that concatenation.
4. Historical DCC used two corpus-statistic columns and lags 1-3, yielding 6 values. It is not equivalent to Felipe's DCC block.
5. Felipe's released V1 matrix has 355 rows and 827 columns:
   - columns 0-461: 462 DCC values, matching 22 properties and lag 1
   - columns 462-746: 285 binary values that numerically match the raw RNA-disease association matrix
   - columns 747-762: 16 normalized 2-mer frequencies
   - columns 763-826: 64 normalized 3-mer frequencies
6. The independently reconstructible sequence-only Felipe baseline has 542 columns: 462 + 16 + 64.
7. The 285 binary columns are target association values. They may be retained only as a clearly labelled paper/released-artifact reproduction. Do not include them in a leakage-free predictive feature matrix.
8. Nazli's cross-feature code does exist. It transforms the association matrix into GIP and SVD outputs; those outputs are not the same values as Felipe's raw 285-column block.
9. The saved V1 cross outputs include an unintended numeric `Unnamed: 0` CSV index as a disease, producing 286 rather than 285 disease dimensions.
10. Any RNA-side GIP or SVD representation constructed from the complete target matrix uses test-label information. Do not describe it as leakage-free. For an inductive unseen-RNA experiment, exclude it unless a valid transformation for test RNAs can be computed without their target labels. A separate transductive experiment is allowed only when its information assumptions are stated precisely.

## Key files and evidence

Read these before changing code:

- `mainfolder/features/rna_features.py`
- `mainfolder/features/cross_features.py`
- `mainfolder/features/nn_features.py`
- `pipelines/run_all_rna_features.py`
- `pipelines/run_all_cross_features.py`
- `scripts/prepare_nazli_repaired_v1_paper_selected_input.py`
- `scripts/reconstruct_paper_sequence_features.py`
- `Data/output_data/sequences_for_oop.csv`
- `Final_output/V1/rna/rna_feature_shapes.csv`
- `Final_output/V1/Cross/cross_feature_shapes.csv`
- `Reports/V1_FEATURE_SIDE_BY_SIDE_2026-10-08.xlsx`
- `Reports/nazli_v1_pipeline_diagnostic_2026-10-05.md`
- `Reports/felipe_paper_exact_2026-09-29T0014/V1_FEATURE_RECONSTRUCTION_AUDIT.md`
- `Reports/felipe_paper_exact_2026-09-29T0014/PSE_IN_ONE_FEATURE_CONTRACT.csv`

## Required work

### 1. Audit the candidate features

Create a machine-readable feature manifest with one row per feature family and these fields:

- family name and method ID
- implementation file/function
- source data
- row count and feature dimension
- parameter values
- raw count, normalized frequency, standardized value, or learned embedding
- duplicate or overlapping family
- whether it uses association labels
- whether it can be calculated for a completely unseen RNA without knowing its labels
- leakage status and permitted experiment type
- output column names/order

Verify all statements from code and numeric artifacts. Do not infer dimensions from filenames alone.

### 2. Repair the bounded pipeline issues

- Drop CSV artifact columns such as `Unnamed: 0` before selecting numeric disease columns.
- Require exactly 285 ordered V1 disease columns when a V1 association matrix is expected.
- Preserve RNA identifiers explicitly and assert one-to-one alignment before concatenation.
- Prevent duplicated k-mer/composition families from silently entering the same candidate matrix.
- Normalize count-based features using a documented denominator appropriate to each descriptor.
- Replace the historical two-column corpus statistics used for DCC/DAC/DACC with a scientifically meaningful, documented physicochemical property source, or exclude those families from the proposed alternative set.
- Keep the independently verified 462-column Felipe DCC reconstruction as the reference DCC baseline.
- Do not alter the historical output files in place. Write repaired candidates to a new, clearly named V1 output directory.

Add focused tests for identifier alignment, expected dimensions, duplicate detection, normalization, removal of index columns, and leakage-gate behavior.

### 3. Build candidate feature sets

At minimum, build these separate V1 matrices:

1. `felipe_sequence_542`: verified 462 DCC + normalized k=2 + normalized k=3.
2. `nazli_kmer`: unique normalized k-mer families; do not duplicate composition families.
3. `nazli_handcrafted_core`: selected normalized k-mers plus individually justified Z-curve, K-gap, reverse-complement, PseDNC, DAC/DCC/DACC families.
4. `nazli_embedding`: each neural embedding family evaluated separately.
5. `nazli_handcrafted_plus_embedding`: only after the individual families have been evaluated.

Keep the released Felipe 827 matrix as a separate reference artifact. Never merge its 285 association columns into a clean candidate matrix.

### 4. Run a controlled ablation

Use exactly the same V1 RNA cohort, disease columns, folds, model settings, seed, threshold rule, and metric implementation for every comparison. Reuse existing verified experiment tooling and saved fold definitions where possible.

Evaluate feature families individually before testing combinations. Do not select features based on the held-out test folds. If hyperparameters or feature families are selected, use training-only or nested validation.

Report at least:

- micro AUPRC as the primary metric because the association matrix is sparse
- micro AUROC
- micro F1 with the threshold source stated
- Hamming loss
- fold-level values, mean, standard deviation, and paired differences versus `felipe_sequence_542`
- the number of labels with both classes in each evaluated fold
- runtime and final feature dimension

Use multiple seeds only after the fixed-seed reproduction is stable. Preserve predictions and exact configuration metadata.

### 5. Handle cross features correctly

Create an explicit decision record for GIP/SVD before running them:

- If the task is inductive prediction for unseen RNAs, explain how a test RNA receives an RNA-side GIP/SVD vector without using its disease labels. If that cannot be done, exclude the representation.
- If the task is transductive link prediction, define which associations are observed and masked, construct all transforms using only permitted observations, and label the experiment transductive.
- Never compute GIP or SVD once from the complete target matrix and then treat ordinary row-wise cross-validation as leakage-free.

## Acceptance criteria

The task is complete only when all of the following are true:

- Every candidate matrix has a stable manifest, explicit RNA IDs, deterministic column order, and recorded source hashes.
- Row alignment against the V1 target matrix is exact and asserted.
- Matrices contain no unintended index columns, duplicate columns, NaN, infinity, or constant artifact columns unless explicitly justified.
- The raw 285 association block is absent from all clean feature matrices.
- Association-derived transformations pass the declared information-boundary checks or are excluded.
- The 542-feature reference reproduces its previously verified values before alternative comparisons begin.
- Ablation results use identical folds and scoring code and include fold-level evidence.
- The final report distinguishes: released 827 reference, clean 542 sequence baseline, Nazli handcrafted alternatives, neural alternatives, and any separately defined transductive experiment.
- Claims about a feature being useful are supported by consistent validation improvement, especially AUPRC, rather than feature count or training performance.

## Expected deliverables

Write new outputs under a dedicated directory such as:

`results/nazli_v1_feature_ablation_<date>/`

Include:

- `feature_manifest.csv`
- `matrix_inventory.csv`
- repaired candidate matrices or immutable paths plus hashes
- `experiment_config.json`
- saved fold assignments
- fold-level predictions
- `ablation_results.csv`
- `paired_differences.csv`
- `verification_report.md`
- `felipe_handoff_summary.md`

The final Felipe-facing summary should state which Nazli feature families added measurable signal, which were removed as duplicates or unstable, and why the 827-feature released reference produces a different result. Use the exact wording that the 285-column block "numerically matches the raw association matrix" and that its upstream provenance remains unresolved.

## Boundaries

- Work on V1 only.
- Do not overwrite the released 827 artifact or historical `Final_output/V1` evidence.
- Do not treat the paper's full 827 score as the clean performance target.
- Do not add the raw target vector to `X` for a scientific prediction result.
- Do not claim Nazli's features are better or worse until the controlled ablation is complete.
- Keep every result reproducible and clearly distinguish verified facts from interpretations.
