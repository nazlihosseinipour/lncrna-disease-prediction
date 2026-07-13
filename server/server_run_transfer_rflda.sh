#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/server_run_lib.sh"
python -c 'import iterstrat' 2>/dev/null || {
  echo "Missing iterative-stratification. Install with: python -m pip install iterative-stratification" >&2
  exit 2
}
mapfile -t features < <(python - <<'PY'
import pandas as pd
d=pd.read_csv('inductive_inputs/feature_representations/feature_representation_manifest.csv')
print('\n'.join(sorted(set(d[d.version.eq('v1')].feature_key)&set(d[d.version.eq('v2')].feature_key))))
PY
)
for direction in v1:v2 v2:v1; do
  IFS=: read -r source target <<<"$direction"; outdir="results/transfer_${source}_to_${target}"
  for feature in "${features[@]}"; do
    cell="transfer_${source}_${target}_${feature}_rflda"
    output="$outdir/${feature}_rflda_${source}_to_${target}_transfer_performance.csv"
    run_transfer_cell "$cell" "$output" python scripts/run_inductive_transfer_experiments.py --source-version "$source" --target-version "$target" --models rflda --feature-keys "$feature" --label-space both --shared-disease-list config/canonical_shared_disease_list.csv --threshold-mode fixed --skip-target-cv --outdir "$outdir"
  done
done
