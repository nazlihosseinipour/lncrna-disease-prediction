# Felipe V1/V2 target — audit of Nazli's processed-data pipeline

## Requested deliverable

Felipe's 2026-09-28 email asks for:

1. V1→V1 and V2→V2 using the same features and the paper's analysis;
2. V1→V2 on the intersection, which may be skipped if the intersection is only 44 RNA sequences.

The requested model lane is multi-label classification (MLC). The evaluation suite is the paper's
eight measures: Hamming loss, ranking error, micro AUROC, micro AUPRC, micro precision, micro
recall, micro F-score and subset accuracy. RF, IPCARF and RFLDA are evaluated with 150 trees and
seed 0. Threshold-dependent measures in the paper-comparison lane use the paper's per-disease
Youden threshold fitted on the evaluated fold.

## Which data are Nazli's processed data

- V1 source: `Data/raw/sequences.csv`. It is byte-identical (SHA-256
  `56c0e35f...efd9bcd`) to the V1 raw file copied into the authors' repository.
- V2 source: `Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv`. It contains
  5,338 label rows; 4,130 have a usable transcript sequence and 1,208 do not.
- The old V1 RNA feature folder has 353 rows because the feature pipeline dropped
  `AK153778.1` (contains `N`) and `BX374997.2 ` (contains `D`, `K` and `Y`).
- The corrected V2 RNA feature folder has 4,130 rows.

The completed recovered-feature experiment therefore does use Nazli's data content. Its V1 input
path points at the authors' mirror, but that file and `Data/raw/sequences.csv` have the same hash.
Its V2 input is Nazli's corrected transcript matrix.

## What the old feature pipeline produced

The V1 folder `Final_output/V1/rna/` contains 18 files and 729 dimensions before duplicate-column
handling. The generation command can be reproduced with:

```bash
PYTHONPATH=. <python-with-project-dependencies> pipelines/run_all_rna_features.py \
  --seqs_csv Data/output_data/sequences_for_oop.csv \
  --outdir <new-output-root> --version_name V1 \
  --k_values 1 2 3 4 --lam 2 --weight 0.5 --L 3 --k_gap 1 \
  --no_normalize --props_csv Data/output_data/dinuc_props.csv
```

The reproduction preserved all 353 IDs and column names. Count-based families are byte-identical;
PseDNC and covariance families agree to approximately 1.2e-15 relative error, consistent with
runtime floating-point differences.

This proves that the old feature files are reproducible. It also reveals why they are unsuitable
for Felipe's requested comparison:

1. `--no_normalize` produced raw k-mer counts, while the released paper k-mer columns are
   normalized frequencies. Dividing Nazli's stored k=2 and k=3 counts by their sequence-window
   counts reproduces the corresponding 80 recovered paper columns within 5e-9.
2. `Data/output_data/dinuc_props.csv` has only two columns named `count` and `freq`. These are
   corpus statistics, not Pse-in-One physicochemical properties. Consequently the stored PseDNC,
   DAC, DCC and DACC files are not the paper descriptors.
3. The old DCC file has 6 dimensions (2 properties × 3 lags). The recovered paper DCC block has
   462 dimensions (22 Pse-in-One properties, lag 1).
4. The corrected V2 folder contains only 603 dimensions and omits PseDNC/DAC/DCC/DACC. Its schema
   is therefore different from the 729-dimension V1 folder.
5. The later model matrices changed the experiment again: corrected V1 used 2,023 features and
   only 24 filtered labels; corrected V2 used 583 features, 3,849 sequence-deduplicated rows and
   86 filtered labels. Those old results cannot answer Felipe's requested analysis.

The classification of every historical candidate is recorded in
`results/codex_user_pipeline_audit_2026-10-05/existing_results_classification.csv`.

## Same constructible paper features on both processed datasets

The reproducible common schema is:

| Component | Dimensions |
|---|---:|
| Pse-in-One 1.0.6 DCC, all 22 RNA properties, lag 1 | 462 |
| Normalized overlapping k-mers, k=2 | 16 |
| Normalized overlapping k-mers, k=3 | 64 |
| **Total** | **542** |

These 542 values are generated from sequence alone with the same column definitions and order for
V1 and V2. The input matrices are:

- `results/felipe_basics_2026-09-27/inputs/v1_paper_sequence_542_X.csv`
- `results/felipe_basics_2026-09-27/inputs/v2_paper_sequence_542_X.csv`

The V1 analysis uses all 355 rows and all 44 paper labels. Ambiguous symbols are removed from two
sequences rather than dropping their rows. The V2 analysis uses all 4,130 rows with sequences and
all 436 disease columns. Its ten folds keep identical sequences in the same fold.

## Verified MLC results for the recovered 542-feature construction

All 60 prediction cells (2 versions × 3 models × 10 folds) pass the frozen-input, hash, shape,
label-order and model-contract verifier. Values below are mean over ten folds using the paper's
Youden evaluation lane.

| Version | Model | Hamming | Ranking error | micro AUROC | micro AUPRC | Precision | Recall | F-score | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | RF | 0.142 | 0.276 | 0.628 | 0.087 | 0.143 | 0.396 | 0.204 | 0.014 |
| V1 | IPCARF | 0.144 | 0.283 | 0.662 | 0.100 | 0.174 | 0.482 | 0.244 | 0.023 |
| V1 | RFLDA | 0.147 | 0.270 | 0.639 | 0.087 | 0.143 | 0.404 | 0.206 | 0.031 |
| V2 | RF | 0.037 | 0.125 | 0.878 | 0.080 | 0.062 | 0.458 | 0.109 | 0.001 |
| V2 | IPCARF | 0.038 | 0.122 | 0.881 | 0.081 | 0.058 | 0.441 | 0.102 | 0.000 |
| V2 | RFLDA | 0.041 | 0.120 | 0.882 | 0.083 | 0.057 | 0.471 | 0.101 | 0.000 |

The exact unrounded table is
`results/codex_user_pipeline_audit_2026-10-05/verified_recovered_542_mlc_results.csv`.

V2's larger micro AUROC does not by itself establish that V2 has stronger sequence signal. V1 and
V2 have different RNA populations, disease columns and prevalence. The values are valid
within-version measurements under the stated contract, but their absolute numbers are not a paired
comparison.

## Why this is not a full 827-feature V2 claim

The released V1 artifact has 827 columns, but only 542 can be reproduced from documented sequence
operations. Its remaining 285 columns have no available sequence-generation command or parameter
set. Nazli's old 729-dimensional pipeline does not generate those columns and is not numerically
equivalent to the released artifact.

Therefore the defensible wording is **"same recovered 542-feature paper construction"**, rather
than "paper-exact 827-feature V2." A complete 827-feature V2 run requires Felipe/the authors to
provide the original Pse-in-One commands or feature-generation script for all 827 columns.

## Dataset intersection

The sequence audit found 48 shared sequence hashes: 48 V1 rows and 52 V2 rows. Four hashes map to
two V2 IDs, leaving **44 strictly one-to-one RNA pairs**. Thus Felipe's stated condition for
skipping V1→V2 is satisfied when “44” means the unambiguous one-to-one cohort. The complete count
should still be disclosed: 48 shared sequences in total, 44 one-to-one.

## Current decision

- Do not report the old `Final_output`/`all_safe` model results as Felipe's experiment.
- Keep `Data/` and the corrected V2 transcript processing as the population source.
- Use the verified 542-feature V1/V2 results if Felipe accepts the reproducible portion of the
  paper construction.
- If Felipe requires all 827 columns, send the feature-generation question before presenting a V2
  result as paper-exact.
- The V1→V2 analysis may be skipped under Felipe's 44-one-to-one exception; retain the 48/44 audit
  evidence in case he asks how the count was obtained.

No historical result, `Final_output` file, source file, commit or branch was overwritten by this
audit.
