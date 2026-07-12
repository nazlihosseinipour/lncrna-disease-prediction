#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/server_run_lib.sh"
for version in v1 v2; do
  manifest="inductive_inputs/canonical_all_safe/$version/feature_representation_manifest.csv"
  for model in rf ipcarf rflda; do
    cell="concat_${version}_${model}"
    output="results/within_version_cv/performance/${version}_all_safe_concatenated_${model}_performance.csv"
    run_cell "$cell" "$output" python scripts/run_inductive_within_cv.py --manifest "$manifest" --versions "$version" --models "$model" --feature-keys all_safe_concatenated --threshold-mode fixed --outdir results/within_version_cv
  done
done
