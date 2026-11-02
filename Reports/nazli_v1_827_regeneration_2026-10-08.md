# Nazli V1 diagnostic 827-feature regeneration — 2026-10-08

## Answer

The 827-column construction can be regenerated from Nazli's processed V1 file as:

1. 462 Pse-in-One-compatible DCC sequence features;
2. 285 raw RNA-disease association outcomes;
3. 16 normalized dinucleotide plus 64 normalized trinucleotide frequencies.

The generated matrix has 355 rows and 827 columns. It is available at
`results/nazli_827_regeneration_audit_2026-10-08/v1/v1_nazli_diagnostic_827.csv`.

This answers the reconstruction question, but it also exposes two problems in the released
827-feature artifact: its rows are not consistently aligned to its displayed RNA IDs, and its
middle 285 columns are target outcomes rather than sequence features.

## Sources and construction

- Nazli processed V1 data: `Data/output_data/sequences_for_oop.csv`
- Sequence property table: `config/pse_in_one_rna_dcc_properties.csv`
- Felipe reference: `lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv`
- Generator and checker: `scripts/regenerate_nazli_v1_827_diagnostic.py`

The script repairs the known trailing-space form of `BX374997.2`, aligns all 355 IDs to
Felipe's displayed index order, reconstructs DCC and normalized k-mers from Nazli's processed
sequences, and inserts Nazli's 285 raw association columns at positions 462–746.

The raw-to-processed boundary was also checked directly. All 355 sequence strings and all
285 association columns are unchanged between `Data/raw/sequences.csv` and
`Data/output_data/sequences_for_oop.csv`. The only ID difference is whitespace normalization
for `BX374997.2`. Therefore, Nazli's preprocessing did not alter the V1 biological values used
in this regeneration.

## Comparison with Felipe's released file

The mapping was recomputed directly from the DCC and k-mer values. The script does not merely
copy the earlier audit's result.

| Released row relationship | Rows |
|---|---:|
| Sequence content belongs to the same-position RNA | 231 |
| Sequence content belongs to a later raw row | 121 |
| No reconstructed sequence row matches within `1e-6` | 3 |

The 121 shifted rows consist of 46 rows at offset +1 and 75 rows at offset +2. This is why a
correct same-ID regeneration is byte different from Felipe's file after row 230.

| Block | Same-ID exact rows | Same-ID exact cells | Recovered-mapping agreement |
|---|---:|---:|---:|
| 462 DCC | 231 / 355 | 106,722 / 164,010 | 162,624 / 162,624 exact |
| 285 association outcomes | 231 / 355 | 100,503 / 101,175 | 100,320 / 100,320 exact |
| 80 normalized k-mers | 231 / 355 | 18,484 / 28,400 | 28,160 / 28,160 within `1e-6` |
| Full 827 | 231 / 355 | 225,709 / 293,585 | 291,104 / 291,104 within `1e-6` |

For the mapped sequence content, every DCC and association value is exact. Fifteen k-mer cells
differ only by `1e-8`, consistent with decimal serialization. The full evidence is in
`results/nazli_827_regeneration_audit_2026-10-08/v1/block_comparison.csv` and
`results/nazli_827_regeneration_audit_2026-10-08/v1/row_mapping.csv`.

## Critical interpretation

The 285 middle columns contain 1,132 positive RNA-disease outcomes copied from the same matrix
that defines the prediction target. They are outcome-derived and create target leakage when
used to predict those associations. The generated 827 matrix is therefore a diagnostic artifact,
not a valid sequence-only feature matrix and not a defensible predictive input.

The reconstruction shows that Nazli's corrected sequence processing is capable of reproducing
Felipe's 542 sequence-derived columns when rows are compared by their actual sequence content.
Trying to force same-ID byte equality would reproduce the released file's row-alignment defect.

## Consequence for V2

Do not transfer the V1 285-column outcome block to V2. V2 has a different outcome space, and
adding its own labels as predictors would repeat the leakage. A valid V2 experiment should use:

- the clean 542 sequence-derived construction; or
- association-derived transforms fitted only from the permitted training-fold associations,
  with held-out associations excluded from construction.

This V1 diagnostic establishes the preprocessing boundary before a V2 run. It does not run V2.

## Verification

The checker freshly reconstructs the matrix and validates its values, ID order, schema, hashes,
block comparisons, leakage flags, and the independently recovered 231/121/3 row mapping. Its
success message is:

```text
V1 diagnostic 827 regeneration passed
```

The generated feature CSV SHA-256 is
`53e2da97ae248374fc877a49dc4546e3b044edc2cf2863c94ed522b8fb5c051c`.
