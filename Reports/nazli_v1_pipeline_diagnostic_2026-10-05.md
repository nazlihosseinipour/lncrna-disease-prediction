# Nazli V1 pipeline diagnostic — repaired 355-row run

## Answer

The repaired V1 run shows that the missing two RNAs were a real data-processing defect, but they were not the main reason the results differed from the paper. Restoring `AK153778.1` and `BX374997.2 `, retaining all 355 rows, and reusing the released ten folds changed the 542-feature results only modestly. The large paper gap remains.

### Released 827 positive control

The released 355×827 positive control reproduces the paper closely: IPCARF AUROC/AUPRC are **0.823/0.440**, and RFLDA AUROC/AUPRC are **0.811/0.413**. This confirms that Felipe's RF/IPCARF/RFLDA model and evaluation code can recover the paper result when given the released 827-column input. The principal unresolved cause is therefore the input feature construction.

The repaired Nazli run is a verified sequence-only comparison. It is **not paper exact** because only 542 of the released artifact's 827 columns are defensibly reconstructible from sequence.

## What was rerun

- Population: all 355 V1 RNAs and all 44 paper disease labels.
- Row repair: invalid symbols were removed from the two previously dropped sequences; no row was dropped. The trailing-space paper ID `BX374997.2 ` was aligned explicitly.
- Features: 462 Pse-in-One DCC columns using 22 RNA properties at lag 1, plus 16 normalized 2-mers and 64 normalized 3-mers; total 542.
- Splits: the released ten-fold file, copied byte for byte. Every row occurs in one test fold per model.
- Models: RF, IPCARF and RFLDA under the saved `paper` profile, with `binary_mode=false`, 150 trees, `random_state=0` and seed 0.
- Evaluation: the paper-comparison lane, using per-disease Youden thresholds fitted on each evaluated fold, with all eight paper metrics.

## Repaired 355×542 result

Values are ten-fold mean (population SD).

| Model | Hamming | Ranking error | AUROC | AUPRC | Precision | Recall | F | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | 0.142 (0.025) | 0.276 (0.029) | 0.628 (0.050) | 0.087 (0.025) | 0.143 (0.020) | 0.396 (0.151) | 0.204 (0.039) | 0.014 (0.014) |
| IPCARF | 0.144 (0.036) | 0.283 (0.042) | 0.662 (0.033) | 0.100 (0.022) | 0.174 (0.038) | 0.482 (0.134) | 0.244 (0.036) | 0.023 (0.025) |
| RFLDA | 0.147 (0.040) | 0.270 (0.031) | 0.639 (0.043) | 0.087 (0.014) | 0.143 (0.018) | 0.404 (0.146) | 0.206 (0.029) | 0.031 (0.036) |

The exact values are in `results/nazli_repaired_v1_felipe_2026-10-05/report/repaired_v1_metrics.csv`.

## Three-way evidence

| Input and purpose | Model | AUROC | AUPRC | F | Interpretation |
|---|---|---:|---:|---:|---|
| Paper Table III | IPCARF | 0.82 | 0.44 | 0.48 | Reported paper value |
| Released 355×827 positive control | IPCARF | 0.823 | 0.440 | 0.483 | Reproduces the paper |
| Accepted Nazli 353×542 | IPCARF | 0.644 | 0.103 | 0.234 | Two rows absent |
| Repaired Nazli 355×542 | IPCARF | 0.662 | 0.100 | 0.244 | Full population; gap remains |
| Paper Table III | RFLDA | 0.81 | 0.41 | 0.46 | Reported paper value |
| Released 355×827 positive control | RFLDA | 0.811 | 0.413 | 0.460 | Reproduces the paper |
| Accepted Nazli 353×542 | RFLDA | 0.636 | 0.086 | 0.196 | Two rows absent |
| Repaired Nazli 355×542 | RFLDA | 0.639 | 0.087 | 0.206 | Full population; gap remains |

For IPCARF, the 353→355 repair changes AUROC by +0.018 and AUPRC by −0.004. For RFLDA, it changes AUROC by +0.003 and AUPRC by +0.001. These shifts do not approach the approximately 0.82 AUROC and 0.41–0.44 AUPRC paper values. The complete all-metric comparison, including RF and population SD, is in `results/nazli_repaired_v1_felipe_2026-10-05/report/v1_cohort_comparison.csv`.

The paper contains no plain RF row, so the released RF result is a useful pipeline control but has no paper value to compare against.

## Where the old feature pipeline differs

1. `pipelines/run_all_rna_features.py` lines 85–96 map `--no_normalize` to `normalize_features=False`; the historical V1 invocation used that flag. Its k-mer outputs are raw counts, while the identifiable released k=2 and k=3 columns are normalized frequencies.
2. `Data/output_data/dinuc_props.csv` has only `count` and `freq` after `dinuc`. These are corpus statistics, not the 22 Pse-in-One physicochemical properties used by the identifiable paper DCC block. Passing this file into methods 3–6 therefore constructs different PseDNC/DAC/DCC/DACC descriptors.
3. The old DCC output has 6 columns (2 supplied values × 3 lags). The released, identifiable DCC block has 462 columns (22 properties × 21 directed cross-property pairs, lag 1).
4. Concatenating every historical output produced 729 columns, but that is not the paper's construction. It mixes k=1–4, reverse-complement k-mers, ordinary PseDNC, covariance outputs made from the wrong property table, duplicate composition families, Z-curve and K-gap descriptors.

The released 827-column matrix contains a 285-column binary, association-like block that numerically matches the V1 RNA-disease association matrix. No documented sequence-generation procedure or parameters for that block were found. The other 542 columns are the only sequence columns that can currently be reconstructed and matched column by column. Guessing the missing construction or appending labels to `X` would not be a valid reproduction.

## Model-code scope

The working tree contains model edits relative to the repository commit, including selectable IPCARF scaling/fallback behavior and RFLDA selection metrics. They do not make this run ambiguous: every prediction contract records frozen source hashes and effective classes, and `scripts/felipe_paper_models.py` explicitly constructs IPCARF with `standardize=False` and `allow_pca_fallback=False`, and RFLDA with `selection_metric="subset_accuracy"`. The verifier checks those saved paper-profile parameters for all 30 cells. Results from a general current/default profile must not be mixed with this verified run contract.

## Verification and decision

- `python3 scripts/verify_nazli_repaired_v1_run.py --check cells` passed all 30 prediction/contract/done cells, input hashes, released fold membership, label order, score ranges, effective classes and model parameters.
- `python3 scripts/verify_nazli_repaired_v1_run.py --check scores` passed three complete groups and 30 fold rows, with zero rejected cells and zero incomplete groups. All eight means and population SDs were independently recomputed within `1e-12`.
- Structured evidence: `results/nazli_repaired_v1_felipe_2026-10-05/report/verification.json`.

V1 now establishes that the row-processing repair is correct and insufficient to reproduce Table III. V2 should remain deferred until the full V1 feature construction is supplied or the 542-column sequence-only contract is explicitly accepted as the intended comparison.
