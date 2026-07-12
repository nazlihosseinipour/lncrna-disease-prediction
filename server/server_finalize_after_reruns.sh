#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/server_run_lib.sh"
python scripts/build_final_deliverables.py --within-dir results/within_version_cv --transfer-summary results/transfer_v1_to_v2/transfer_summary.csv --outdir results
python scripts/build_generalization_matrix.py --final-comparison results/final_comparison.csv --outdir results
python -m pytest -q test/test_transfer_protocol.py
echo "Finalization complete; review results/audit/completion/."
