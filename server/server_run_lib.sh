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
  [[ -f "$file" ]] && python scripts/validate_performance.py "$file"
}

valid_transfer_performance() {
  local file="$1" protocol="${1%.csv}.protocol.json"
  valid_performance "$file" && [[ -f "$protocol" ]] && python - "$protocol" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
assert data.get("threshold_mode") == "fixed"
assert data.get("strict_target_overlap_removal") is True
assert data.get("n_splits") == 10
assert data.get("canonical_disease_names")
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

run_transfer_cell() {
  local cell="$1" output="$2"; shift 2
  local lock="results/logs/locks/${cell}.lock" log="results/logs/${cell}.log"
  if valid_transfer_performance "$output"; then
    printf '%s\t%s\tSKIPPED_VALID_STRICT\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"; return 0
  fi
  if ! mkdir "$lock" 2>/dev/null; then echo "Locked: $cell"; return 0; fi
  trap 'rmdir "$lock" 2>/dev/null || true' RETURN
  printf '%s\t%s\tSTARTED\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"
  if "$@" >"$log" 2>&1 && valid_transfer_performance "$output"; then
    printf '%s\t%s\tCOMPLETE_STRICT\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"
  else
    printf '%s\t%s\tFAILED\t%s\n' "$(date -u +%FT%TZ)" "$cell" "$*" >> "$LEDGER"; return 1
  fi
}
