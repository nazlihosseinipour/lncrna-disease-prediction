# Shared-sequence cohort audit (2026-09-28)

Scope: Phases 1–3 of the shared-sequence plan (`HANDOFF_SHARED_SEQUENCE_2026-09-28.md` §B).
Decisions: `SHARED_SEQUENCE_SPEC_2026-09-28.md` (SS1–SS16). Machine-readable:
`shared_sequence_correspondence.csv`, `shared_sequence_run_manifest.json`.
Outputs: `results/felipe_shared_sequence_2026-09-28/{cohort,disease_map_audit}/`.

## 1. Sequence normalization and hashing

- Normalization is the frozen `reconstruct_paper_sequence_features.normalize_sequence`, the
  function that built the 542 features: uppercase, `T→U`, remove every character outside
  `ACGU`. An empty result counts as missing and is not hashed.
- Hash: SHA-256 of the UTF-8 normalized string.
- Inputs:
  - V1 raw `lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv`: 355 rows, 0 missing.
  - V2 raw `Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv`: 5,338 rows,
    1,208 missing (these are the rows in the ledger `config/v2_missing_sequence_ledger.csv`).
- None of the 48 shared rows had ambiguity symbols removed.

## 2. Intersection: reproduced independently three times

The same result came out of three separate derivations: the supervisor's pre-check (before any
code existed), the cohort builder (S1), and the independent verifier (S5, which does not import
the builder).

| Quantity | Value |
|---|---|
| Unique normalized-sequence hashes shared by V1 and V2 | **48** |
| V1 rows among them | 48 (one per hash; no V1 duplicates) |
| V2 rows among them | 52 |
| Hashes mapping to >1 V2 ID | 4 → strict one-to-one cohort = **44** |
| Pairs with the same ID in both versions | 1 (`NR_024015`) |
| Sequence length (min / median / max) | 200 / 1,978.5 / 37,027 nt |

| V1 ID | V2 IDs (duplicates) | V2 label conflicts | Conflicting diseases |
|---|---|---|---|
| NR_002765 | `ASAP1-IT1` / `ASAP1‑IT1` (U+2011 non-breaking hyphen) | 1 | esophageal squamous cell carcinoma |
| NR_045406 | `HIF1A-AS2` / `LINLK-A` | 3 | coronary artery disease, endometriosis, osteosarcoma |
| NR_026597 | `DIRC3` / `DIRC3-204` | 1 | melanoma |
| NR_131221 | `SPRY4-IT1` / `SPRY4‑IT1` (U+2011) | 6 | ESCC, glioma, multiple sclerosis, nasopharyngeal carcinoma, osteosarcoma, PDAC |

Two of the four duplicates are the same gene name, once with an ordinary hyphen and once with a
non-breaking hyphen. In V2 these became separate "RNAs" with split annotations.

## 3. Feature agreement

- The V1 and V2 542-feature vectors are **bit-identical** for every one of the 48 hashes.
- The frozen reconstruction (DCC lag 1 + 2-mers + 3-mers), recomputed from each normalized
  sequence, equals every V1 and V2 X row within `atol 1e-12`.
- Parsing note (SS12): the X files are read with `float_precision="round_trip"`. The default
  pandas parser returns identical values on the original 8-digit X files. It differs by ≤1.8e-15
  on the 17-digit `test_X.csv`, because it is not correctly rounded.

## 4. Correspondence table

`shared_sequence_correspondence.csv` (byte-identical to `cohort/correspondence.csv`) has one row
per hash. Each row gives:
- the hash, sequence length, V1 ID and all V2 IDs;
- the X row indices;
- `one_to_one` or `ambiguous_multi_v2_id`;
- whether the features agree between V1 and V2 and with the reconstruction;
- the removed ambiguity symbols;
- the V2 duplicate-ID label conflicts;
- the number of V1-vs-V2 label disagreements;
- cohort membership.

Labels also differ between the versions for the same sequence: 107 disease-level disagreements
over 32 of the 48 hashes. The largest are `NR_131012`/`NEAT1` (30), `NR_015379`/`UCA1a` (9) and
SPRY4-IT1 (6). This is why each prediction is scored against both label versions.

## 5. `fromserver/` (read-only)

- 3,441 files were scanned (3,112 tabular headers read, 230 JSON files parsed).
- **No file maps V1 IDs to V2 IDs or records shared sequences.** The 48-hash correspondence
  therefore cannot be cross-checked pair by pair against the server bundle.
- The only related evidence:
  - `results/audit/reconciliation/transfer_alignment_audit.csv` has counts only:
    `exact_rna_overlap=1`, `normalized_mapped_overlap=14`, `overlap_removed=False`.
  - 114 protocol files list `removed_exact_target_ids=["NR_024015"]`. That agrees with the one
    same-ID pair here: it covers 1 of the 48.
- **Consequence:** earlier transfer designs removed target overlap by exact ID only. This includes
  the frozen `scripts/run_felipe_strict_transfer.py`. They therefore left **47 identical
  sequences** in the target population, so they leak by sequence. The paired design here replaces
  them. The old transfer commands were not run.

## 6. Disease/output contract (Phase 2)

The independent audit (`scripts/audit_shared_disease_map.py`) passed all six checks:
1. All listed source columns exist byte-exactly in the raw headers.
2. Every listed column normalizes (`normalized_label`) to its canonical name.
3. The normalized V1 ∩ V2 names are exactly the 73 concepts. For every concept, the listed
   columns are all raw columns with that name.
4. No raw column is used twice.
5. Y rebuilt from raw equals `v1_shared_73_Y.csv` (355×73) and `v2_shared_73_Y.csv` (4,130×73),
   with 0 differences.
6. The map's positive counts match V1 (355 rows) and V2 (the 4,130-row population).

- **OR combinations:** 3, all in V2 only (choriocarcinoma, glioblastoma, pneumonia; upper vs lower
  case). Each adds exactly one positive row.
- **No minimum-positive filter.** All 73 columns are in every prediction and label file.
- **Not merged (SS11; approved by Nazli 2026-09-28, SS17), reported as a limitation.** The primary
  label space was not modified after observing results. Near-identical names that the map's
  normalization does not unify:
  - possessive vs none: Alzheimer's/Alzheimer, Parkinson's, Huntington's, Hirschsprung's,
    Klinefelter's disease;
  - singular vs plural: cardiovascular disease(s), GIST(s), myelodysplastic syndrome(s), diabetic
    nephropathy/nephropathies, cardiomyopathy/cardiomyopathies;
  - spelling variants: oesophageal/esophageal SCC, the V1 typo "unveal" melanoma, vulvar/vulva SCC.

  Merging them would change the prespecified 73-label space, so the shared concept count is
  conservative.

## 7. Disease definedness in the small test cohorts

| Cohort | Label version | Diseases with both classes | Zero-positive diseases | Positive associations |
|---|---|---|---|---|
| primary (44) | V1 | 30 | 43 | 51 |
| primary (44) | V2 | 30 | 43 | 61 |
| sensitivity (48) | V1 | 32 | 41 | 56 |
| sensitivity (48) | V2 (OR) | 30 | 43 | 73 |

- Only 12 of the 73 diseases have both classes in all four cohort×label combinations, and 24 have
  none.
- Per-disease AUROC/AUPRC are therefore computed only where they are defined (NaN otherwise). Most
  defined diseases have 1–3 positives among 44 sequences, so per-disease values are very noisy.
- Micro metrics pool all 73 columns and are always defined here: no bootstrap replicate was
  single-class.

## 8. Leakage prevention (Phase 3)

Both training populations exclude every row whose hash is among the 48:

| | V1 | V2 |
|---|---|---|
| Rows before exclusion | 355 | 4,130 |
| Excluded (shared hash) | 48 | 52 |
| Training rows | **307** | **4,078** |
| Hash overlap with test | 0 | 0 |
| Exact 542-row (float64 bytes) overlap with test | 0 | 0 |

These checks were asserted three times: by the cohort builder, by the runner before fitting
(recomputed from the raw files), and by the independent verifier.

- **Fitting uses source rows only.** Scaling/PCA (IPCARF), feature selection (RFLDA) and model
  fitting all happen inside `model.fit(X_train, Y_train)` on training rows. `predict_proba` is
  called with the test X only.
- **No thresholds are fitted.** The fixed 0.5 threshold is used; no Youden threshold is computed.
- **Test labels are read only by the scorer.**
- **Primary and sensitivity cohorts share the same trained models (SS3).** The training exclusion
  (all 48 hashes) is identical for both cohorts, so each (train version, model) was fitted once.
  It predicted all 48 rows, and the cohorts are row subsets at scoring time.

## 9. Cohort rules (approved by Nazli 2026-09-28, SS17)

- **Primary (44):** strictly one-to-one hashes. V1 labels come from the single V1 row; V2 labels
  from the single V2 row.
- **Sensitivity (48):** all hashes. V1 labels are as in the primary cohort. V2 labels are the OR
  over the hash's V2 IDs. Duplicate IDs are one RNA, never counted as independent. The 11
  conflicts are listed in `cohort/v2_label_conflicts.csv`.
