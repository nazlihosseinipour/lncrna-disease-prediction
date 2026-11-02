# Shared-sequence paired V1/V2 comparison — results (verified 2026-09-28)

Status: complete and verified.
- Six fits: {RF, IPCARF, RFLDA} × {train V1, train V2}.
- 32 scored rows: 4 scorers (3 models + training-prevalence baseline) × 2 training versions × 2
  cohorts × 2 label versions.
- The independent verifier (`scripts/verify_felipe_shared_sequence.py`) passes every check:
  frozen, cohort, cells, scores, protected.
- The supervisor separately recomputed every micro AUROC, AUPRC and F1 from the saved prediction
  files with plain sklearn. The largest difference was 1.1e-16.

Machine-readable: `shared_sequence_transfer_table.csv` (all 32 rows with bootstrap CIs) and
`shared_sequence_run_manifest.json`. Cohort construction: `SHARED_SEQUENCE_COHORT_AUDIT.md`.
Decisions: `SHARED_SEQUENCE_SPEC_2026-09-28.md`.

## Design

| Element | Setting |
|---|---|
| Test rows | Shared RNA sequences (normalized-sequence SHA-256). Primary: 44 one-to-one. Sensitivity: all 48 (V2 labels OR-combined over duplicate V2 IDs). |
| Test X | One 542-feature row per sequence, identical for every model (`test_X.csv` sha256 `2f244bf5…`). V1 and V2 vectors bit-identical. |
| Outputs | 73 shared disease concepts, all kept. No minimum-positive filter. |
| Training | V1: 307 rows. V2: 4,078 rows. Every row with a shared sequence removed. Zero hash overlap and zero exact 542-row overlap with the test rows. |
| Models | Paper profile (150 trees, seed 0; IPCARF components by inner 5-fold → 8 (V1) / 32 (V2); RFLDA inner 5-fold, steps of 50 → 250 (V1) / 350 (V2) features). |
| 2×2 | Each trained model's scores on the same 48 rows are scored against V1 labels and against V2 labels. |
| Threshold | Fixed 0.5 (`primary_clean`, `threshold_source=constant_0.5`). No Youden threshold of any kind was computed. |
| Baseline | Per-disease prevalence in the **training** population of the same train version, as a constant score. |
| Uncertainty | 2,000 bootstrap resamples of sequences. Indices are shared across all scorers so paired differences are valid. Percentile 95% CI. |

## Primary cohort (44 one-to-one sequences × 73 diseases)

- Positive associations: 51 (V1 labels), 61 (V2 labels).
- Diseases with both classes: 30 of 73 under each label version.

| Train → labels | Model | Micro AUROC [95% CI] | Micro AUPRC [95% CI] | F1 @0.5 | Hamming | Per-disease AUROC median (n=30) |
|---|---|---|---|---|---|---|
| V1 → V1 | RF | 0.590 [0.509, 0.666] | 0.047 [0.014, 0.110] | 0.038 | 0.016 | 0.407 |
| V1 → V1 | IPCARF | 0.623 [0.549, 0.715] | 0.044 [0.014, 0.105] | 0.000 | 0.016 | 0.453 |
| V1 → V1 | RFLDA | 0.648 [0.564, 0.727] | 0.053 [0.016, 0.127] | 0.038 | 0.016 | 0.498 |
| V1 → V1 | Baseline | **0.715** [0.625, 0.802] | 0.054 [0.027, 0.089] | 0.000 | 0.016 | 0.500 |
| V1 → V2 | RF | 0.636 [0.581, 0.755] | 0.048 [0.021, 0.115] | 0.032 | 0.019 | 0.407 |
| V1 → V2 | IPCARF | 0.669 [0.625, 0.760] | 0.061 [0.029, 0.132] | 0.000 | 0.019 | 0.476 |
| V1 → V2 | RFLDA | 0.666 [0.621, 0.769] | 0.051 [0.023, 0.107] | 0.032 | 0.019 | 0.437 |
| V1 → V2 | Baseline | **0.720** [0.662, 0.847] | 0.073 [0.041, 0.120] | 0.000 | 0.019 | 0.500 |
| V2 → V1 | RF | 0.666 [0.564, 0.798] | 0.103 [0.044, 0.202] | 0.071 | 0.016 | 0.477 |
| V2 → V1 | IPCARF | 0.663 [0.572, 0.771] | **0.128** [0.049, 0.255] | 0.038 | 0.016 | 0.488 |
| V2 → V1 | RFLDA | 0.668 [0.574, 0.783] | 0.097 [0.040, 0.197] | 0.000 | 0.016 | 0.489 |
| V2 → V1 | Baseline | 0.667 [0.575, 0.802] | 0.073 [0.035, 0.127] | 0.000 | 0.016 | 0.500 |
| V2 → V2 | RF | 0.666 [0.592, 0.827] | 0.051 [0.031, 0.131] | 0.000 | 0.021 | 0.371 |
| V2 → V2 | IPCARF | 0.671 [0.588, 0.848] | 0.078 [0.044, 0.206] | 0.000 | 0.019 | 0.384 |
| V2 → V2 | RFLDA | 0.677 [0.609, 0.826] | 0.053 [0.032, 0.123] | 0.000 | 0.019 | 0.384 |
| V2 → V2 | Baseline | **0.797** [0.752, 0.890] | 0.091 [0.051, 0.156] | 0.000 | 0.019 | 0.500 |

## Sensitivity cohort (all 48 sequences; V2 labels = OR over duplicate IDs)

- Positive associations: 56 (V1), 73 (V2).
- Diseases with both classes: 32 (V1), 30 (V2).

| Train → labels | RF AUROC / AUPRC | IPCARF AUROC / AUPRC | RFLDA AUROC / AUPRC | Baseline AUROC / AUPRC |
|---|---|---|---|---|
| V1 → V1 | 0.621 / 0.053 | 0.638 / 0.052 | 0.668 / 0.055 | 0.731 / 0.057 |
| V1 → V2 | 0.677 / 0.062 | 0.705 / 0.084 | 0.709 / 0.075 | 0.741 / 0.085 |
| V2 → V1 | 0.688 / 0.104 | 0.687 / 0.126 | 0.682 / 0.096 | 0.687 / 0.077 |
| V2 → V2 | 0.675 / 0.059 | 0.679 / 0.081 | 0.685 / 0.061 | 0.806 / 0.103 |

Every conclusion below holds in both cohorts. Adding the four ambiguous sequences moves point
estimates by at most 0.043 AUROC. A few intervals cross 0 in one cohort and not the other: for
model − baseline AUROC, 7 of 12 cells exclude 0 in the primary cohort and 6 of 12 in the
sensitivity cohort. No conclusion changes.

## Paired comparisons (bootstrap by sequence; CIs from `paired_differences.csv`)

1. **Model vs training-prevalence baseline, micro AUROC.**
   - No model exceeds the baseline by more than 0.0012, in any of the 12 model × direction cells
     of either cohort.
   - The CI excludes 0 (model worse) in 7 of 12 primary cells and 6 of 12 sensitivity cells, for
     example primary V2 → V2:
     - RF −0.131 [−0.173, −0.045]
     - IPCARF −0.126 [−0.176, −0.025]
     - RFLDA −0.120 [−0.154, −0.047]
   - V2 → V1 is the only direction where models tie the baseline: differences −0.004 to +0.001,
     with every CI spanning 0.
2. **Model vs baseline, micro AUPRC.**
   - Mostly indistinguishable from 0.
   - The one exception is IPCARF V2 → V1: +0.055 [0.000, 0.158] (primary) and +0.049
     [0.003, 0.148] (sensitivity). The interval borders 0, and this is one of dozens of
     comparisons (no multiplicity correction), so it is not evidence of signal.
3. **Training-version effect (train V2 − train V1, same labels).**
   - AUROC: CIs include 0 for every model under both label versions.
   - AUPRC with V1 labels: positive for RF (+0.056 [0.006, 0.148]) and IPCARF (+0.083
     [0.014, 0.201]); RFLDA +0.044 [−0.017, 0.135]. The same pattern holds in the sensitivity
     cohort.
   - AUPRC with V2 labels: no effect (+0.002 to +0.017, CIs include 0).
   - So the V2-trained models rank V1's (sparser) annotations slightly better. They gain nothing
     on V2's own annotations, where the V2 prevalence baseline is strongest.
4. **Label-version effect (V2 labels − V1 labels, same predictions).**
   - For the models, CIs include 0 everywhere.
   - For the V2-prevalence baseline it is +0.130 [0.030, 0.222] AUROC. V2-training prevalence
     matches V2 labels better than V1 labels. This is consistent with a label-prevalence
     effect rather than sequence signal.
5. **Thresholded lane.**
   - At the fixed 0.5 threshold the models predict almost no positives: micro F1 0.000–0.071,
     with many cells at exactly 0.
   - Hamming loss (0.016–0.021) is at or above the all-negative baseline's (0.016–0.019).
   - The baseline predicts nothing positive (no prevalence reaches 0.5), so its F1 = 0.

## Per-disease view

- Per-disease AUROC is defined for 30–32 diseases per cell. Most of them have only 1–3 positive
  sequences.
- Medians are 0.37–0.50 for every model and direction, and ≤ 0.50 in all 24 model cells.
- The fraction of defined diseases with AUROC > 0.5 is 13–43 %.
- No detectable disease-specific sequence signal under this experimental design (small, noisy
  test set).

## Interpretation (gate G8 respected)

1. On the same RNA sequences, the same 542 features and the same 73 disease concepts, **no model
   beats a no-model training-prevalence baseline** on micro AUROC, in either training version or
   label version: at best they tie it (V2 → V1, within ±0.005), and otherwise they fall below it.
   On AUPRC the models are indistinguishable from it, apart from one borderline interval. There
   is no detectable sequence-based disease signal under this experimental design. This agrees
   with the full-population within-version result
   (`WITHIN6_VERIFIED_RESULTS_2026-09-28.md`), where all six models were below the same kind of
   baseline.
2. The train-V2 AUPRC gain on V1 labels is small and comes with wide intervals. It does not mean
   V2 is "better". The same models show no gain on V2 labels, per-disease AUROC stays at or below
   0.5, and neither "V2 higher" nor "V2 lower" is an acceptance criterion.
3. The results are consistent with differences arising mainly from **label annotations and
   prevalence**: there are 107 disease-level V1/V2 disagreements on the same 48 sequences, and
   the prevalence baseline tracks the label version. The design does not prove this cause.

## Small-cohort caveats (why these numbers are unstable)

- 44–48 test sequences × 73 diseases, with only 51–73 positive associations in total.
- 41–43 of the 73 diseases have no positive at all in the test cohort. Only 12 diseases have both
  classes in every cohort × label combination.
- 95% CIs on micro AUROC are ~0.15–0.25 wide, and wider on AUPRC. Differences below ~0.05 are not
  interpretable.
- Per-disease metrics rest on 1–3 positives each. Medians are reported as descriptive only.
- No correction for multiple comparisons. The paired table contains 100+ intervals.

## Verification and provenance

| Check | Result |
|---|---|
| Both trained models per model type predicted the identical test X | all 6 cells `test_x_sha256=2f244bf5…`, identical row/column hashes |
| Source/test overlap | 0 hashes, 0 exact 542-rows (builder, runner, verifier independently) |
| Dimensions/order | 48 × 73; rows = `test_X` hash order; columns = label-map order |
| Metrics recomputed independently | verifier: 32 rows within 1e-12; supervisor sklearn recomputation: max Δ 1.1e-16; bootstrap CIs recomputed for two combinations |
| Frozen code | fit-time table `config/felipe_shared_sequence_frozen_sources.csv` (15 base rows unchanged + 4 new scripts) recorded in every cell; scoring table differs only in the scorer row (SS15) |
| Resume | probe: resume reused 6/6 cells with no refit; refusal without `--resume`; changed training subset → quarantine + refit (no mixing); scorer refuses mixed subsets/probe cells |
| Protected results | all 222 snapshot entries (`results/felipe_final_2026-09-28/`, WITHIN6 report/table, frozen sources) unchanged |

Two issues were found and fixed during verification:

- **SS15, scorer parser fix.** S5 found that pandas' default CSV parser merged distinct 17-digit
  scores into ties, shifting micro AUROC by up to 2e-5. The scorer now re-reads each verified
  cell with the exact (`round_trip`) parser. The frozen within-version rescorer has the same
  parser effect. Measured read-only on the verified within results, it moves V2 per-fold AUROC by
  ≤ 6e-6 and V1 by 0. The reported 3-decimal within values are unaffected, and the within results
  were not changed.
- **SS16, incident.** The supervisor tested `--resume` on the completed run against the new
  scoring table. That triggered quarantine and refitting of the verified cells. The run was
  stopped, the original cells were moved back, and the accidental refits were kept under
  `_rejected/20260928T0925_accidental_refit_supervisor_error/`. The verifier then passed again in
  full.
  - The incident also measured determinism: V1 refits were byte-identical. V2 RF/IPCARF refits
    differed in 16 and 9 values by ≤ 1.1e-16 (multithreaded prediction accumulation), with
    identical AUROC.

## Limitations

- Cohort rule and disease map approved by Nazli (2026-09-28, SS17); not changed after observing
  results.

- The V2 population is the 4,130 RNAs with a transcript sequence. The shared cohort is only 48
  sequences.
- The disease map was not extended with possessive/plural/spelling variants (SS11). The shared
  concept count is conservative.
- Only the fixed-0.5 thresholded lane was computed. No source-only Youden or OOB threshold is
  included.
- Model settings are those of the paper profile. RF uses sklearn's default `max_features`.
