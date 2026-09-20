# LEAKAGE-INFLATED reference reproduction of the CIBCB2025 RF numbers

Not a valid inductive result. See scripts/reproduce_paper_reference.py docstring.
* 285 of 827 features (cols 462-746) are the raw association matrix (own labels for rows 0-230).
* F1/P/R use a per-label Youden threshold fitted on each test fold's own labels.

```
                                    AUROC  AUPRC  F1_fixed0.5  P_youden  R_youden  F1_youden
prior_only_no_features              0.704  0.112        0.000     0.000     0.000      0.000
paper_542_block_removed             0.615  0.081        0.022     0.148     0.359      0.198
paper_827_full__PAPER_REPRODUCTION  0.807  0.411        0.316     0.324     0.751      0.447
block_only_285                      0.820  0.436        0.390     0.444     0.744      0.551
```
