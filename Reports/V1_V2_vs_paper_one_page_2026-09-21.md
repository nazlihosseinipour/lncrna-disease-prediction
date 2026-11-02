# V1, V2 and the paper's Table III: one-page summary

## 1. Paper protocol reproduced on the paper's files
Authors' feature file, labels (355 x 44), 10 inductive folds, 150 trees, seed 0, micro metrics per fold then mean, Youden threshold per label on the test fold.

| Model | AUROC | AUPRC | F | Paper |
|---|---:|---:|---:|---|
| Plain RF | 0.807 | 0.411 | 0.447 | RFLDA (MLC) 0.81 / 0.41 / 0.46 |
| RFLDA (MLC) | 0.811 | 0.413 | 0.460 | 0.81 / 0.41 / 0.46 |
| IPCARF (MLC) | 0.823 | 0.440 | 0.483 | 0.82 / 0.44 / 0.48 |
| IPCARF (HMC) | 0.825 | 0.436 | 0.484 | 0.83 / 0.44 / 0.48 |

Table III is reproduced within 0.005 on nearly every column (RFLDA accuracy is 0.03 off).

## 2. What drives it
Columns 462-746 of `features_preprocessed.csv` (285 binary columns, sum 1132) equal the raw lncRNA-disease matrix. For rows 0-230 the block is that row's own label vector (231 of 231 rows; `results/paper_reproduction_reference_2026-09-20/leak_check.py`). Section IV-B of the paper says such features are excluded. Same pipeline on the other 542 columns: AUROC 0.615, AUPRC 0.081, F 0.198. A no-feature baseline (train prevalence per label) scores 0.704 / 0.112.

## 3. V2 under the paper's rules
Same measure and folds as the paper (10-fold inductive, micro metrics per fold, Youden on the test fold, 150 trees). V2 = 3,849 transcripts, 86 labels, our 583 sequence features (length-normalised).

| V2 model | AUROC | AUPRC | Youden F1 | Macro AUROC (shuffle-test p) |
|---|---:|---:|---:|---:|
| Default RF (fully grown trees) | 0.785 | 0.099 | 0.119 | 0.531 (0.008) |
| RF, leaf size chosen by out-of-bag score | 0.827 | 0.142 | 0.108 | 0.577 (0.002) |
| No-feature baseline | 0.818 | 0.110 | 0 | 0.500 |
| Paper (with the 285-column block) | 0.81 | 0.41 | 0.46 | - |

## 4. Why V2 is lower than the paper
- The paper's AUPRC and F include the association block. Without it the paper's own file scores AUPRC 0.081, below its 0.112 baseline.
- V2 is sparser: 2.4% positive cells against 5% in the paper's 44 labels. AUPRC and F fall with sparsity even when the model is no worse.
- V2's micro AUROC (0.827) exceeds the paper's 0.81, but its no-feature baseline already scores 0.818. Micro AUROC mostly measures label frequency.
- Against its own baseline, V2's clean features do carry signal: AUPRC 0.142 against 0.110, macro AUROC 0.577 (p = 0.002 by shuffle test). The paper's 542 non-block columns do not beat their baseline.

## 5. Why V2 differs from our V1
Our V1 used raw k-mer counts, so its above-chance score comes from transcript length. On length-normalised frequencies (as in V2) its macro AUROC falls from 0.553 to 0.441. V1 and V2 were not on the same scale. No pipeline bug was found in V2 (alignment tests, label and fold audit, feature audit).

## 6. What is not the same as the paper
- Features: the paper's eight Pse-in-One families cannot be rebuilt (the released file has numeric column names and no configuration). V2 uses k-mer, reverse-complement k-mer, k-gap and Z-curve features.
- The 285-column block is not built for V2. Doing so would mean copying each transcript's own labels into its features.
- IPCARF and RFLDA were not run on V2; the V2 numbers are plain RF.
- The regularised V2 result chose hyperparameters by out-of-bag score inside each training fold. A stricter nested-CV run is not finished.

## 7. Questions for the supervisor
1. How should columns 462-746 of `features_preprocessed.csv` be read, given Section IV-B? Could you share the script that built the file?
2. Can I have the Pse-in-One configuration for the eight feature families?
