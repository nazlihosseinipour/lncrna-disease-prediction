#!/usr/bin/env bash
# Aggregate everything into the final deliverables. Cheap (no training): reads the
# per-experiment summaries and writes the comparison tables + reports.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

echo "[1/4] Merging transfer summaries (V1->V2 and V2->V1) ..."
"$PYTHON" - <<'PY'
import pandas as pd, os
a = "results/transfer/transfer_feature_representation_summary.csv"
b = "results/transfer_v2_to_v1/transfer_feature_representation_summary.csv"
frames = [pd.read_csv(p) for p in (a, b) if os.path.exists(p)]
if frames:
    pd.concat(frames, ignore_index=True).drop_duplicates().to_csv(a, index=False)
    print(f"  merged -> {a} ({sum(len(f) for f in frames)} rows in)")
else:
    print("  no transfer summaries found; skipping merge")
PY

echo "[2/4] Final comparison tables + report ..."
"$PYTHON" scripts/build_final_deliverables.py

echo "[3/4] Generalization matrix (V1->V1 / V2->V2 / V1->V2 / V2->V1) ..."
"$PYTHON" scripts/build_generalization_matrix.py

echo "[4/4] Feature-analysis outputs ..."
"$PYTHON" scripts/build_feature_analysis.py

echo "DONE: see results/final_report.md, results/results_summary.md, results/audit/"
