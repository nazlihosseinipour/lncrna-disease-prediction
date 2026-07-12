#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RESULTS_ROOT="${RESULTS_ROOT:-results}"
[[ "$RESULTS_ROOT" == "results" ]] || { echo "RESULTS_ROOT must be results" >&2; exit 2; }
case "$RESULTS_ROOT" in *results-v2*|*results_canonical*|*results_reconciliation*|*results_backup*) exit 2;; esac
mkdir -p results/logs results/logs/locks
LEDGER="results/logs/server_rerun_ledger.tsv"
[[ -f "$LEDGER" ]] || printf 'timestamp\tcell\tstatus\tcommand\n' > "$LEDGER"

valid_performance() {
  local file="$1"
  [[ -f "$file" ]] && python - "$file" <<'PY'
import sys, pandas as pd
d=pd.read_csv(sys.argv[1])
assert 'folds' in d and d.folds.astype(str).str.match(r'fold\d+').sum()==10
PY
}

run_cell() {
  local cell="$1" output="$2"; shift 2
  local lock="results/logs/locks/${cell}.lock" log="results/logs/${cell}.log"
  if valid_performance "$output"; then
    printf '%s\t%s\tSKIPPED_VALID\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"; return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then echo "Locked: $cell"; return 0; fi
  trap 'rmdir "$lock" 2>/dev/null || true' RETURN
  printf '%s\t%s\tSTARTED\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"
  if "$@" >"$log" 2>&1 && valid_performance "$output"; then
    printf '%s\t%s\tCOMPLETE\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"
  else
    printf '%s\t%s\tFAILED\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"; return 1
  fi
}
