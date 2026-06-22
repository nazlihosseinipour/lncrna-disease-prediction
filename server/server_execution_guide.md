# Server Execution Guide

How to run the full experiment suite on the server. All long jobs are wrapped in the
four scripts in this directory. Nothing here was run at full scale locally — only small
validations.

## 0. Environment

```bash
cd <repo-root>                       # the directory containing this `server/` folder
source .venv/bin/activate            # or: export PYTHON=/path/to/python
# (the scripts default to .venv/bin/python; override with: export PYTHON=python3)
python -c "import sklearn, pandas, numpy, iterstrat; print('env OK')"
```

Dependencies: `scikit-learn`, `pandas`, `numpy`, `iterative-stratification`
(`pip install -r requirements.txt`). No GPU needed.

## 1. Recommended execution order

| Step | Script | Produces | Depends on |
|------|--------|----------|------------|
| 1 | `server/run_feature_comparison.sh` | within-version CV (V1→V1, V2→V2) | — |
| 2 | `server/run_v1_v2_generalization.sh` | transfer V1→V2, V2→V1 | manifest from step 1 |
| 3 | `server/run_binary_comparison.sh` | binary pairwise comparison | manifest from step 1 |
| 4 | `server/run_final_aggregation.sh` | all comparison tables + reports | steps 1–3 |

Steps 1–3 are independent after step 1 has built the manifest; on a multi-core box they
can run concurrently. Step 4 must run last.

## 2. Exact commands

```bash
# Step 1 — feature comparison (LONGEST; background it)
nohup bash server/run_feature_comparison.sh      > logs/feature_comparison.log 2>&1 &

# Step 2 — cross-dataset generalization
nohup bash server/run_v1_v2_generalization.sh    > logs/generalization.log 2>&1 &

# Step 3 — binary comparison (validate first with 1 fold)
MAX_FOLDS=1 bash server/run_binary_comparison.sh                  # quick smoke
nohup bash server/run_binary_comparison.sh       > logs/binary.log 2>&1 &

# Step 4 — aggregation (after 1–3 finish)
bash server/run_final_aggregation.sh
```

Create the log dir first: `mkdir -p logs`.

Optional leakage-free thresholded metrics (any step): append
`--threshold-mode fixed` (already wired into the within-CV and binary runners).

## 3. Estimated runtime (single multi-core node, `n_jobs=-1`)

| Step | Scope | Estimate |
|------|-------|----------|
| 1 | V1 all features × 3 models | ~30–60 min |
| 1 | V2 RFLDA per feature (nested feature search) | ~6–8 h **each** |
| 1 | V2 IPCARF/RF per feature | minutes each |
| 2 | V1→V2 transfer (cheap source fit) | minutes |
| 2 | V2→V1 transfer (5k-sample source, fit-once) | ~tens of min/feature |
| 3 | Binary V1 | ~1–2 h; **Binary V2** | many hours (n_lncRNA×n_disease rows) |
| 4 | Aggregation | < 1 min |

The V2 structured-output RFLDA runs and binary V2 are the long poles. Budget overnight
for a full sweep. To shrink: keep the 4 default feature sets (don't use
`--all-compatible`), and/or run V2 with `--models ipcarf rf` first, adding RFLDA later.

## 4. Output directories

```
inductive_inputs/feature_representations/   prepared X / Y / fold-splits + manifest (reproducible inputs)
results/within_version_cv/                  within-version CV: predictions/, performance/, summary, run_config.json
results/transfer/                           V1→V2 transfer perf + summary
results/transfer_v2_to_v1/                  V2→V1 transfer + V1 common-CV baseline
results/binary_comparison/                  binary pairwise: predictions/, performance/, summary, run_config.json
results/                                    final_comparison.csv, feature_comparison.csv, model_comparison.csv,
                                            cross_dataset_generalization.csv, generalization_matrix.csv,
                                            final_report.md, results_summary.md
results/audit/                              leakage_audit_report.md, disease_preprocessing_and_intersection.md,
                                            feature_catalogue.csv, feature_statistics.csv
```

## 5. Reproducibility

Every runner writes a `run_config.json` (seed=0, n_splits=10, models, threshold mode).
Fold definitions live in `inductive_inputs/feature_representations/*_X_splits.csv`.
Because X/Y/splits are persisted, all experiments rerun without rebuilding features.

## 6. Verification after the run

```bash
python - <<'PY'
import pandas as pd
d = pd.read_csv("results/final_comparison.csv")
assert d["leakage"].sum() == 0, "leaky rows present!"
print("rows:", len(d), "| leakage rows:", int(d["leakage"].sum()))
print(d.groupby("evaluation").size())
PY
cat results/results_summary.md
```
