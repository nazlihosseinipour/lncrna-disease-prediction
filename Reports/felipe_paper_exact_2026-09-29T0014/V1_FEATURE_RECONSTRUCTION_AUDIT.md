# V1 feature reconstruction audit — feature gate (2026-09-29)

**Gate result: FAILED.** The released 355 × 827 matrix cannot be rebuilt from documented sequence
operations. 542 of its 827 columns can be reproduced from the V1 sequences. The remaining 285 match
no sequence descriptor, and the paper's descriptor parameters are not recorded anywhere we could
find. V2 model fitting was therefore not started.

Evidence:
- `results/felipe_paper_exact_2026-09-29T0014/feature_gate/`: `column_audit.csv`, `row_mapping.csv`,
  `binary_block_audit.csv`, `negative_controls.csv`, `feature_gate_summary.json`.
  - Produced by `scripts/audit_827_feature_reconstruction.py`, sha256 `f60832ee…`; 8 tests pass.
- `evidence/provenance_search.md`.
- `PSE_IN_ONE_FEATURE_CONTRACT.csv`.
- Supervisor spot-checks, reproduced independently: the row offsets, the binary-block row equality,
  and that Y follows the ID/raw order.

## What the paper specifies

- **Families.** Section IV-B says it "extracted eight RNA feature vectors using the package
  Pse-in-One [46]". It then names seven: K-mers, Mismatch, DAC, DCC (misprinted as a second
  "DAC"), DACC, PC-PseDNC-General and SC-PseDNC-General. It states that "in total, it sums to 827
  features".
- **Parameters.** No k, mismatch m, lag, λ, weight, property set, normalization or order is given.
- **Version.** Reference [46] is Pse-in-One 2.0. Mismatch exists only in 2.0.
- **No generation record found.** There is no command, configuration or generation script in:
  - the repository or its 65 commits across all refs;
  - `fromserver/`;
  - the authors' public repository;
  - the local slides (they read "K-mers with K = ?").

## Column-by-column result (tolerance 1e-6; rows aligned as below)

| Columns | Family (by value) | Exact | Within tolerance | Unmatched |
|---|---|---|---|---|
| 0–461 | DCC, 22 RNA properties, lag 1 | 462 | 0 | 0 |
| 462–746 | 285-column binary block | — (matches no sequence descriptor) | — | 285 vs sequence families |
| 747–762 | 2-mer frequencies | 65 (2-mer + 3-mer combined) | 15 (1e-8 rounding, one row) | 0 |
| 763–826 | 3-mer frequencies | (above) | (above) | 0 |

- **Absent families.** The remaining named families were tested with standard settings, as
  negative controls only (not parameter fitting). None matches any released column:
  - DAC lag 1;
  - DACC (beyond its DCC half);
  - Mismatch k=3 m=1;
  - PC-PseDNC-General λ 1–3, w 0.05/0.1;
  - SC-PseDNC-General λ 1–2.

  So only two of the seven named families (K-mers and DCC) are present.
- **Arithmetic cannot recover the parameters.** Pse-in-One defaults for the named families do not
  sum to 827. Other combinations that happen to total 827 are arithmetically possible but
  undocumented, and were not used.
- **The 285-column binary block.** The released matrix contains a 285-column binary block that
  numerically matches the raw association matrix, but its provenance conflicts with the paper's
  description and remains unresolved.
  - Values are {0, 1}, with 1,132 positives, equal to the raw 355 × 285 matrix.
  - It equals the raw association row in 352 of 352 aligned rows.
  - No Pse-in-One RNA descriptor produces binary values: the DCC values range from −7.79 to 12.50,
    and the k-mer frequencies from 0 to 0.204.

## Row-order checks

- The released row labels (index) equal the raw `ID` column by position for all 355 rows, and Y
  has the same index.
- **Row content is shifted.** The content of the sequence blocks (and the binary block) matches
  the raw row with the same position only for rows 0–230. After that:

  | Released rows | Offset (content comes from raw row i+k) |
  |---|---|
  | 231–275 | +1 |
  | 276–292 | +2 |
  | 293 | no match |
  | 294 | +1 |
  | 295–352 | +2 |
  | 353–354 | no match |

  - Raw rows 231, 277 and 296 appear nowhere.
  - Released rows 293, 353 and 354 (`NR_033205`, `NR_015370`, `NR_003202`) match no raw row.
- **Y follows the IDs, not the shifted content.** Row positive counts correlate 0.96 with the
  same-position raw rows and −0.06 with the shifted ones. So for ~124 rows the X content and the Y
  labels describe different RNAs.
- **The earlier results show the same break.** The existing IPCARF/RFLDA predictions reach
  per-fold AUROC 0.89–0.98 on rows < 231 and 0.50–0.77 on rows ≥ 231.
- **Consistent but unverified explanation.** Raw rows 231 and 277 are the only two sequences with
  ambiguous bases (N; K/Y/D), and Pse-in-One rejects non-ACGU input. Skipping them would produce
  the +1/+2 offsets. This does not explain raw row 296, the three unmatched released rows, or the
  binary block.

## Can the released matrix be reproduced?

- **The V1 reference runs can use it.** The released file itself is available, so the V1
  paper-faithful runs use it unchanged, including all 827 columns.
- **It cannot be regenerated from sequences.** There is no documented construction for 5 of the 7
  named families, the eighth vector is unknown, and the 285-column block has no sequence
  derivation.
- **No 827-feature V2 matrix was made.** It was not fabricated, no labels were appended, and the
  542-feature matrix was not used in its place.

## What would unblock V2

- The original feature-generation script or configuration from the authors: Pse-in-One version,
  command lines, the parameters for every family, and the generation of cols 462–746.
- Their handling of T/U and ambiguous bases.
- An explanation of the row offsets.
