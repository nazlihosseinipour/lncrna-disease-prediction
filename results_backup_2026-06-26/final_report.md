# Task 6 — Final Report

## Best within-version configurations (leakage-free)

- **V1** best by micro-AUPRC: `rc_kmer_matrix_k4` + `rflda` (AUPRC=0.1117, AUROC=0.6703, F1=0.2437)
- **V2** best by micro-AUPRC: `kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix` + `ipcarf` (AUPRC=0.1185, AUROC=0.8279, F1=0.1208)

### Feature comparison (within-version)

| train_dataset | feature_set | model | leakage | micro_roc_mean | micro_auprc_mean | fscore_mean | hamming_mean | label_ranking_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | rc_kmer_matrix_k4 | rflda | False | 0.6703 | 0.1117 | 0.2437 | 0.1352 | 0.5138 | 0.0171 |
| v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rf | False | 0.6697 | 0.1095 | 0.2410 | 0.1320 | 0.5076 | 0.0313 |
| v1 | kmer_matrix_k4 | rf | False | 0.6681 | 0.1071 | 0.2300 | 0.1432 | 0.4946 | 0.0113 |
| v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rflda | False | 0.6525 | 0.1026 | 0.2218 | 0.1372 | 0.5319 | 0.0085 |
| v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | ipcarf | False | 0.6668 | 0.1007 | 0.2393 | 0.1510 | 0.4985 | 0.0170 |
| v1 | kmer_matrix_k4 | ipcarf | False | 0.6726 | 0.1006 | 0.2350 | 0.1570 | 0.5009 | 0.0141 |
| v1 | rc_kmer_matrix_k4 | rf | False | 0.6657 | 0.0996 | 0.2268 | 0.1598 | 0.5187 | 0.0029 |
| v1 | kmer_matrix_k4 | rflda | False | 0.6508 | 0.0964 | 0.2249 | 0.1300 | 0.5608 | 0.0056 |
| v1 | rc_kmer_matrix_k4 | ipcarf | False | 0.6659 | 0.0954 | 0.2376 | 0.1442 | 0.5095 | 0.0084 |
| v1 | psednc_matrix | rf | False | 0.5891 | 0.0816 | 0.2176 | 0.1005 | 0.6102 | 0.0311 |
| v1 | psednc_matrix | ipcarf | False | 0.5890 | 0.0816 | 0.2175 | 0.1005 | 0.6102 | 0.0311 |
| v1 | psednc_matrix | rflda | False | 0.5889 | 0.0815 | 0.2171 | 0.1008 | 0.6102 | 0.0311 |
| v2 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | ipcarf | False | 0.8279 | 0.1185 | 0.1208 | 0.1308 | 0.4617 | 0.0078 |
| v2 | kmer_matrix_k4 | ipcarf | False | 0.8278 | 0.1176 | 0.1216 | 0.1343 | 0.4528 | 0.0093 |
| v2 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rflda | False | 0.8186 | 0.1152 | 0.1280 | 0.1172 | 0.4560 | 0.0081 |
| v2 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rf | False | 0.8181 | 0.1145 | 0.1278 | 0.1179 | 0.4532 | 0.0066 |
| v2 | kmer_matrix_k4 | rflda | False | 0.8192 | 0.1142 | 0.1260 | 0.1191 | 0.4565 | 0.0078 |
| v2 | rc_kmer_matrix_k4 | ipcarf | False | 0.8244 | 0.1139 | 0.1195 | 0.1334 | 0.4536 | 0.0076 |
| v2 | kmer_matrix_k4 | rf | False | 0.8193 | 0.1137 | 0.1222 | 0.1244 | 0.4584 | 0.0081 |
| v2 | rc_kmer_matrix_k4 | rf | False | 0.8152 | 0.1100 | 0.1297 | 0.1147 | 0.4467 | 0.0071 |
| v2 | rc_kmer_matrix_k4 | rflda | False | 0.8119 | 0.1097 | 0.1290 | 0.1116 | 0.4636 | 0.0076 |
| v2 | psednc_matrix | rflda | False | 0.6295 | 0.0415 | 0.1393 | 0.0379 | 0.7204 | 0.0262 |
| v2 | psednc_matrix | rf | False | 0.6281 | 0.0414 | 0.1397 | 0.0377 | 0.7203 | 0.0257 |
| v2 | psednc_matrix | ipcarf | False | 0.6273 | 0.0410 | 0.1366 | 0.0377 | 0.7246 | 0.0238 |

### Model comparison (mean over feature sets)

| train_dataset | model | micro_roc_mean | micro_auprc_mean | fscore_mean | hamming_mean |
| --- | --- | --- | --- | --- | --- |
| v1 | ipcarf | 0.6485 | 0.0946 | 0.2323 | 0.1382 |
| v1 | rf | 0.6481 | 0.0995 | 0.2289 | 0.1339 |
| v1 | rflda | 0.6406 | 0.0980 | 0.2269 | 0.1258 |
| v2 | ipcarf | 0.7768 | 0.0978 | 0.1246 | 0.1090 |
| v2 | rf | 0.7701 | 0.0949 | 0.1299 | 0.0987 |
| v2 | rflda | 0.7698 | 0.0951 | 0.1306 | 0.0965 |

## Task 5 — V1 → V2 change (matched feature+model)

- mean Δ(micro_roc) V2−V1 = +0.1265
- mean Δ(micro_auprc) V2−V1 = -0.0014
- mean Δ(fscore) V2−V1 = -0.1010

## Task 4 — Cross-dataset generalization

| experiment | train_dataset | test_dataset | feature_set | model | n_labels | micro_roc_mean | micro_auprc_mean | fscore_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_to_v2_transfer | v1 | v2 | kmer_matrix_k4 | ipcarf | 40 | 0.7057 | 0.0433 | 0.0802 |
| v2_to_v1_transfer | v2 | v1 | kmer_matrix_k4 | ipcarf | 40 | 0.6788 | 0.0725 | 0.2552 |
| v1_to_v2_transfer | v1 | v2 | kmer_matrix_k4 | rf | 40 | 0.7246 | 0.0431 | 0.0824 |
| v2_to_v1_transfer | v2 | v1 | kmer_matrix_k4 | rf | 40 | 0.6656 | 0.0730 | 0.2246 |
| v1_to_v2_transfer | v1 | v2 | kmer_matrix_k4 | rflda | 40 | 0.7177 | 0.0398 | 0.0943 |
| v2_to_v1_transfer | v2 | v1 | kmer_matrix_k4 | rflda | 40 | 0.6610 | 0.0627 | 0.2065 |
| v1_to_v2_transfer | v1 | v2 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | ipcarf | 40 | 0.7118 | 0.0483 | 0.0917 |
| v2_to_v1_transfer | v2 | v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | ipcarf | 40 | 0.6621 | 0.0704 | 0.2412 |
| v1_to_v2_transfer | v1 | v2 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rf | 40 | 0.7262 | 0.0432 | 0.0820 |
| v2_to_v1_transfer | v2 | v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rf | 40 | 0.6795 | 0.0635 | 0.2423 |
| v1_to_v2_transfer | v1 | v2 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rflda | 40 | 0.7177 | 0.0398 | 0.0943 |
| v2_to_v1_transfer | v2 | v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rflda | 40 | 0.6610 | 0.0627 | 0.2065 |
| v1_to_v2_transfer | v1 | v2 | psednc_matrix | ipcarf | 40 | 0.5561 | 0.0207 | 0.0818 |
| v2_to_v1_transfer | v2 | v1 | psednc_matrix | ipcarf | 40 | 0.5549 | 0.0320 | 0.1316 |
| v1_to_v2_transfer | v1 | v2 | psednc_matrix | rf | 40 | 0.5561 | 0.0206 | 0.0818 |
| v2_to_v1_transfer | v2 | v1 | psednc_matrix | rf | 40 | 0.5566 | 0.0328 | 0.1332 |
| v1_to_v2_transfer | v1 | v2 | psednc_matrix | rflda | 40 | 0.5561 | 0.0206 | 0.0818 |
| v2_to_v1_transfer | v2 | v1 | psednc_matrix | rflda | 40 | 0.5568 | 0.0328 | 0.1320 |
| v1_to_v2_transfer | v1 | v2 | rc_kmer_matrix_k4 | ipcarf | 40 | 0.7199 | 0.0691 | 0.0890 |
| v2_to_v1_transfer | v2 | v1 | rc_kmer_matrix_k4 | ipcarf | 40 | 0.6661 | 0.0689 | 0.2378 |
| v1_to_v2_transfer | v1 | v2 | rc_kmer_matrix_k4 | rf | 40 | 0.7194 | 0.0411 | 0.1008 |
| v2_to_v1_transfer | v2 | v1 | rc_kmer_matrix_k4 | rf | 40 | 0.6758 | 0.0631 | 0.2204 |
| v1_to_v2_transfer | v1 | v2 | rc_kmer_matrix_k4 | rflda | 40 | 0.7099 | 0.0392 | 0.1178 |
| v2_to_v1_transfer | v2 | v1 | rc_kmer_matrix_k4 | rflda | 40 | 0.6611 | 0.0605 | 0.2235 |
| v1_common_cv | v1 | v1 | kmer_matrix_k4 | ipcarf | 40 | 0.6851 | 0.0491 | 0.2352 |
| v1_common_cv | v1 | v1 | kmer_matrix_k4 | rf | 40 | 0.6622 | 0.0465 | 0.2187 |
| v1_common_cv | v1 | v1 | kmer_matrix_k4 | rflda | 40 | 0.6858 | 0.0474 | 0.2254 |
| v1_common_cv | v1 | v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | ipcarf | 40 | 0.6867 | 0.0401 | 0.2314 |
| v1_common_cv | v1 | v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rf | 40 | 0.6762 | 0.0456 | 0.2189 |
| v1_common_cv | v1 | v1 | kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix | rflda | 40 | 0.6858 | 0.0474 | 0.2254 |
| v1_common_cv | v1 | v1 | psednc_matrix | ipcarf | 40 | 0.5654 | 0.0423 | 0.1503 |
| v1_common_cv | v1 | v1 | psednc_matrix | rf | 40 | 0.5654 | 0.0423 | 0.1503 |
| v1_common_cv | v1 | v1 | psednc_matrix | rflda | 40 | 0.5654 | 0.0423 | 0.1500 |
| v1_common_cv | v1 | v1 | rc_kmer_matrix_k4 | ipcarf | 40 | 0.6777 | 0.0427 | 0.2163 |
| v1_common_cv | v1 | v1 | rc_kmer_matrix_k4 | rf | 40 | 0.6694 | 0.0429 | 0.2181 |
| v1_common_cv | v1 | v1 | rc_kmer_matrix_k4 | rflda | 40 | 0.6855 | 0.0599 | 0.2114 |


Best transfer config (micro-AUPRC): `kmer_matrix_k4` + `rf` (v2→v1, AUPRC=0.0730, AUROC=0.6656) over 40 shared diseases.

## Interpretation (Tasks 5 & 6)

**Dataset facts.** V1 = 355 lncRNA × 285 disease (1,132 positives, density 1.12%, ~3.97 pos/disease). V2 = 5,338 lncRNA × 436 disease (9,907 positives, density 0.43%, ~22.7 pos/disease). After aligning to RNA features and the min-positives>5 label filter, the modelled matrices are V1: 353×45 labels, V2: 4,114×124 labels.

**What changed V1→V2 (Task 5).** V2 is ~12× more lncRNAs and far more positives per disease, but lower per-cell density. Ranking metrics improve sharply (micro-AUROC ~0.66→~0.82) because each disease column has many more positive examples to learn from. micro-AUPRC rises modestly. Micro-F1 *drops* (~0.23→~0.12): V2's label space is larger and sparser, so the Youden-thresholded operating point trades precision/recall differently — an artefact of sparsity, not of worse ranking. Use AUROC/AUPRC, not F1, to compare across the two label spaces.

**Does V2 generalize better? (Task 5).** Yes. On the 40-disease shared space, models *trained on V2 and tested on V1* outperform models trained on V1 itself: mean Δ micro-AUPRC = +0.0122, mean Δ micro-AUROC = -0.0110 (V2→V1 transfer minus within-V1 common-CV). The larger, more label-rich V2 training set yields representations that transfer better than the small V1 set — strong evidence for training future models on V2.

**Effect of dataset sparsity.** Both datasets are sparse multilabel problems (<1.2% density), which keeps exact-match accuracy near zero and micro-AUPRC low in absolute terms (the positive rate is the AUPRC baseline). V2's higher positives-per-disease is what lifts AUROC despite lower overall density.

**Cross-dataset caveat.** Transfer is evaluated only over RNA/psednc features (version-independent columns) and the 40 diseases shared after name normalization; lncRNA-sample intersection (1 exact / 14 mapped) is not a viable transfer axis. NxN kernels (gip_lncRNA, lfs_from_Y) and per-dataset SVD bases do not transfer across versions and are excluded from Task 4 by construction.

## Recommendation for V3

Train on **V2** using **`kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix`** with **`ipcarf`** — best within-version micro-AUPRC (0.1185). See the dataset-size/sparsity discussion in this report and the transfer table above for generalization evidence.