#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/server_run_lib.sh"
python - <<'PY' | while IFS=, read -r version feature model; do
import pandas as pd
d=pd.read_csv('results/audit/reconciliation/binary_experiment_audit.csv')
for r in d[d.completion.ne('VALID COMPLETE')].itertuples(): print(r.version,r.feature,r.model,sep=',')
PY
  cell="binary_${version}_${feature}_${model}"
  output="results/binary_comparison/performance/${version}_${feature}_${model}_performance.csv"
  run_cell "$cell" "$output" python scripts/run_binary_comparison.py --versions "$version" --models "$model" --feature-keys "$feature" --threshold-mode fixed --outdir results/binary_comparison
done
