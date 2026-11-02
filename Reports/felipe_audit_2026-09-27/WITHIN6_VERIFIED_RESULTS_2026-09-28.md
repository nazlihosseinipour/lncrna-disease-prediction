# Within-version results — V1→V1 and V2→V2 (verified 2026-09-28 09:55 CEST)

Status: all six cells complete. `verify_felipe_exact_results.py --check matrix --scope within6`
accepted 60/60 prediction cells. Rescored from saved predictions with no refitting
(`results/felipe_final_2026-09-28/scored_within6/`, 0 rejected, 0 incomplete). Every pooled
AUROC, AUPRC and fixed-threshold F1 was independently recomputed with sklearn from the raw
prediction files and matched exactly. Transfer (V1→V2, V2→V1) and the 73-label baselines
have not been run.

## Protocol

- Features: 542 sequence-only columns (Pse-in-One DCC, 22 properties, lag 1 + k-mer 2 and 3),
  rebuilt from raw sequences, identical definitions and order in V1 and V2, joined by RNA ID.
  No association labels in X.
- V1: 355 RNAs × 44 diseases, released paper ten folds.
- V2: population `v2_sequence_available_4130` (4,130 RNAs × 436 diseases, no disease filter;
  1,208 raw rows without a transcript sequence excluded and listed by ID; 10 diseases have zero
  positives in this population and are kept). Group-aware ten folds: RNAs with identical
  sequences never cross train/test.
- Models: paper profile (150 trees, seed 0; IPCARF unscaled, components {2…128} by inner
  5-fold; RFLDA inner 5-fold, steps of 50, subset-accuracy selection).
- The V1/V2 exact overlap `NR_024015` matters only for transfer (not run).
- Two scoring lanes from the same saved scores:
  - `primary_clean` (`fixed`): threshold 0.5, no test labels used.
  - `paper_comparison` (`paper_youden`): per-disease Youden threshold fitted on the evaluated
    fold's own labels, as in the paper. It uses test labels, so its thresholded metrics
    are optimistic.

AUROC/AUPRC are micro-averaged, mean ± SD over ten folds, and identical in both lanes.

## Verified table

| Direction | Model | RNAs × labels | Micro AUROC | Micro AUPRC | F1 (fixed 0.5) | F1 (paper Youden) |
|---|---|---|---|---|---|---|
| V1→V1 | RF | 355 × 44 | 0.628 ± 0.050 | 0.087 ± 0.025 | 0.013 ± 0.022 | 0.204 ± 0.039 |
| V1→V1 | IPCARF | 355 × 44 | 0.662 ± 0.033 | 0.100 ± 0.022 | 0.015 ± 0.020 | 0.244 ± 0.036 |
| V1→V1 | RFLDA | 355 × 44 | 0.639 ± 0.043 | 0.087 ± 0.014 | 0.008 ± 0.013 | 0.206 ± 0.029 |
| V2→V2 | RF | 4130 × 436 | 0.878 ± 0.020 | 0.080 ± 0.010 | 0.010 ± 0.004 | 0.109 ± 0.010 |
| V2→V2 | IPCARF | 4130 × 436 | 0.881 ± 0.019 | 0.081 ± 0.009 | 0.004 ± 0.004 | 0.102 ± 0.014 |
| V2→V2 | RFLDA | 4130 × 436 | 0.882 ± 0.017 | 0.083 ± 0.009 | 0.008 ± 0.005 | 0.101 ± 0.013 |

Machine-readable: `within6_verified_table.csv` (both lanes, with precision/recall and pooled
out-of-fold values).

## Interpretation checks

A no-model baseline scores every test RNA with the training-fold disease prevalence:

| Population | Baseline micro AUROC | Baseline micro AUPRC |
|---|---|---|
| V1, 44 labels | 0.704 ± 0.025 | 0.112 ± 0.019 |
| V2, 436 labels | 0.920 ± 0.014 | 0.095 ± 0.004 |

All six models are **below** this baseline on micro AUROC and AUPRC. Per-disease AUROC
(which removes the prevalence effect) is near chance: median 0.48–0.53 for every model. The
only weak sign of signal is in V2 diseases with ≥10 positives (n = 89): median 0.521–0.524,
67–73 % above 0.5.

Conclusions:

1. The high V2 micro AUROC reflects the spread of disease frequencies over 436 labels, not
   sequence signal. Micro AUROC must not be compared across V1 and V2; V2 is not "better".
2. With clean sequence-only features, the paper models carry little disease-specific signal.
   This is consistent with the finding that the released 827-column matrix owes its high
   scores to 285 association-label columns inside X.
3. The paper-lane F1 values are well above the fixed-threshold F1 because the Youden threshold
   is fitted on the test fold's own labels.

## Limitations

- V2 population is the 4,130 sequence-available RNAs, not all 5,338.
- RF uses sklearn's default `max_features` (the paper does not state it); IPCARF selects
  components by the regressor's default R² score.
- V1 RF reproduces the earlier clean control exactly (AUROC 0.628469, AUPRC 0.087408, Youden
  F1 0.204490), from an independent code path.

## Draft message to Felipe

> Dear Felipe,
>
> I have verified results for the within-version comparison with RF, IPCARF, and RFLDA.
>
> Setup: I rebuilt the paper's 542 sequence-only features (Pse-in-One DCC + 2/3-mers) from raw
> sequences for both versions. The released 827-column V1 matrix also contains 285
> lncRNA–disease association columns, and its rows stop matching the RNA IDs after row 230, so I
> kept it only as a reproduction reference. V1 uses 355 RNAs × 44 diseases with the released
> folds. V2 uses the 4,130 RNAs that have a transcript sequence (1,208 of 5,338 have none) ×
> all 436 diseases, with ten folds that keep identical sequences on one side of each split.
>
> | | RF | IPCARF | RFLDA |
> |---|---|---|---|
> | V1→V1 AUROC / AUPRC | 0.628 / 0.087 | 0.662 / 0.100 | 0.639 / 0.087 |
> | V2→V2 AUROC / AUPRC | 0.878 / 0.080 | 0.881 / 0.081 | 0.882 / 0.083 |
>
> (Micro-averaged, mean over ten folds; full table with SDs and F1 under a fixed 0.5 threshold
> and under the paper's test-fold Youden threshold attached.)
>
> The main finding is that the V2 AUROC is not evidence of better prediction. A baseline that
> only predicts each disease's training frequency gets 0.920 AUROC on V2 and 0.704 on V1, above
> all models, and per-disease AUROC is close to 0.5. With label-free sequence features, these
> models carry little disease-specific signal; the high scores in the paper come from the
> association columns inside the released feature matrix.
>
> Transfer runs (V1→V2, V2→V1) on the 73 shared diseases are implemented and tested but not yet
> run. Next, I can run them together with the matching 73-label within baselines if you think
> that is still useful given this result.
>
> Best regards,
> Nazli
