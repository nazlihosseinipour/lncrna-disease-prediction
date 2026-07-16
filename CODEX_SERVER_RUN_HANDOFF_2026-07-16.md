# lncRNA server-run handoff — 2026-07-16

## Purpose

This document is the starting point for a new Codex chat/CLI session. The local audit and server-run code repairs were made, but the final two server phases described below have **not** been completed. One earlier supervised attempt failed. Codex CLI must be made available again on the server before it can babysit the resumable runs.

Project locations:

- Local Mac repository: `/Users/nazlihosseinipour/Library/CloudStorage/OneDrive-KULeuven/Projects/Internship/lncrna-gip-project`
- Server repository: `~/data/lncrna-disease-prediction`
- Server account/host previously shown as: `r0865750@kor-s-dali`
- Corrected Git commit currently checked out locally: `0539a73 Fix server validation and strict transfer protocol`

## Current completion state

As last verified locally on 2026-07-16:

- `results/logs/NONBINARY_REQUESTED_EXPERIMENTS_DONE` is missing.
- `results/logs/ALL_REQUESTED_EXPERIMENTS_DONE` is missing.
- `results/logs/CODEX_NONBINARY_COMPLETION_STATUS.md` is missing.
- `results/logs/CODEX_ALL_EXPERIMENTS_COMPLETION_STATUS.md` is missing.
- Therefore, neither the repaired non-binary phase nor the remaining binary phase may be declared complete.
- The two final Codex phase prompts were discussed but were not successfully executed.

## Work already completed and committed

The corrected repository contains:

- `scripts/validate_performance.py`
  - Requires exactly one each of folds 1–10.
  - Allows a separate mean/std summary row.
- `scripts/repair_completed_concat_metadata.py`
  - Already ran successfully on the server and printed: `Validated and labelled 4 completed all-safe concatenation rows.`
- `config/canonical_shared_disease_list.csv`
  - Contains the corrected symmetric 11-disease strict transfer list.
- Updated `server/server_run_lib.sh`
  - Uses the corrected validator.
  - Transfer completion requires a strict protocol sidecar.
- Updated `server/server_run_concat.sh`.
- Updated `server/server_run_transfer_fast.sh`.
- Updated `server/server_run_transfer_rflda.sh`.
- Updated transfer code regenerates deterministic target folds after exact target-RNA overlap removal.
- Corrected transfer executions write `.protocol.json` sidecars.
- Old transfer CSVs without strict protocol evidence are not accepted as corrected completed cells; stale files are preserved under a legacy/stale location before replacement.
- Fixed threshold metadata is recorded for within-version, transfer and binary summaries.
- Local lightweight validation previously passed: shell syntax, Python compilation, seven focused tests and `git diff --check`.

## What succeeded on the server

The downloaded logs showed that four all-safe concatenation experiments actually completed all ten folds and produced predictions:

- V1 RF
- V1 IPCARF
- V2 RF
- V2 IPCARF

These are valid results. They must be preserved and skipped by resumed execution.

## Earlier failure causes

### 1. False concatenation failures

The old supervisor rejected valid performance files with:

```text
AssertionError: not exactly ten folds
```

It incorrectly counted the mean/std row. This is fixed by `scripts/validate_performance.py` and the updated server library.

### 2. RFLDA dependency missing in the Python used inside tmux

The server stopped with:

```text
Missing iterative-stratification for RFLDA.
```

The experiment command was retried without first successfully installing the package in the exact Python environment used by tmux. The next session must use `.venv/bin/python`, install `iterative-stratification==0.1.9` there, verify `import iterstrat`, and ensure the stage scripts inherit that environment.

### 3. Strict transfer disease-list incompatibility

All earlier transfer jobs failed with the old 12-disease list because `heart failure` was incompatible after aligned-feature and strict overlap filtering. The committed canonical list now contains the defensible symmetric 11-disease set.

### 4. Transfer split-index defect

The old code removed exact target-RNA overlaps but reused target split indices created before removal. The committed code regenerates deterministic target folds after removal.

### 5. Broken old supervisor/final status

The downloaded `results/logs/server_supervised_run.console.log` ends with symptoms including:

```text
KeyError: 'feature_set'
server/server_supervised_run.sh: line 148: 0: command not found
server/server_supervised_run.sh: line 148: results/logs/agent_repairs/repair_ledger.csv: Permission denied
```

The old supervisor also reported `remaining=120` because it did not recognize the valid outputs. `server/server_supervised_run.sh` is not present in the corrected local tracked repository. Do not use the old server-only copy as the experiment orchestrator.

### 6. Automatic repair agent unavailable

Earlier repair attempts exited with code 127 because `codex` was unavailable inside the tmux environment/PATH. Before beginning, verify that Codex CLI works in the server shell and inside tmux.

## Server quota warning

Last observed server quota:

```text
/dev/mapper/system-home   996M / 1024M hard limit
/dev/mapper/data-data   48351M / 51200M hard limit
```

The data filesystem was over its 46080M soft quota and had roughly 2.8 GB left before the hard limit. The home filesystem was nearly full. Use a project-local temporary directory and disable the pip cache. Monitor quota throughout. Never delete valid scientific results or audit evidence merely to make space.

Useful checks:

```bash
quota -s
du -sh results inductive_inputs final_output .tmp 2>/dev/null
du -sh results/* 2>/dev/null | sort -h | tail -20
```

## Local working-tree warning

The local working tree was not clean when this handoff was created:

```text
 M mainfolder/.DS_Store
?? .results_original_premerge/
?? Inductive_models_for_structured_output_prediction_of_lncRNA-disease_associations.pdf
?? results-v2/
```

These items are not part of commit `0539a73`. Treat them as user data. Do not delete, merge, commit, or upload them until their provenance is checked. In particular, do not assume the reappeared `results-v2/` or `.results_original_premerge/` is redundant.

## Mandatory preflight in the new server session

Run from `~/data/lncrna-disease-prediction`:

```bash
git status --short
git log -1 --oneline

test -f scripts/validate_performance.py && \
test -f scripts/repair_completed_concat_metadata.py && \
test -f config/canonical_shared_disease_list.csv && \
grep -q 'run_transfer_cell' server/server_run_transfer_fast.sh && \
echo 'CORRECTED FILES PRESENT'

command -v codex || echo 'CODEX CLI NOT AVAILABLE'
test -x .venv/bin/python && .venv/bin/python --version
```

Expected corrected commit is at least `0539a73`. Do not overwrite server-local files blindly if `git status` reports modifications. Back up or stash only the conflicting tracked scripts, inspect the diff, and pull with `git pull --ff-only`.

If Codex CLI is not available, restore/reinstall it using the server's established installation method, then verify both outside and inside tmux:

```bash
command -v codex
codex --version
tmux new-session -d -s codex-check 'command -v codex; codex --version; sleep 30'
tmux capture-pane -pt codex-check
```

Do not start training until `codex --version` and the corrected-file check succeed.

## Phase 1 — finish the failed `RUN_BINARY=0` work

This phase includes only all-safe concatenation, strict transfer RF/IPCARF, strict transfer RFLDA, and partial finalization. It must not launch binary experiments.

The new Codex session should autonomously:

1. Create `.tmp` under the project root.
2. Install into the exact project environment:

   ```bash
   PIP_NO_CACHE_DIR=1 TMPDIR="$PWD/.tmp" \
     .venv/bin/python -m pip install iterative-stratification==0.1.9
   .venv/bin/python -c "import sys, iterstrat; print(sys.executable); print('iterstrat OK')"
   ```

3. Ensure `PATH="$PWD/.venv/bin:$PATH"` or activate `.venv` for every stage.
4. Inspect current logs and results before changing code.
5. Run, sequentially and never concurrently:

   ```bash
   bash server/server_run_concat.sh
   bash server/server_run_transfer_fast.sh
   bash server/server_run_transfer_rflda.sh
   bash server/server_finalize_after_reruns.sh
   ```

6. For each failure, inspect the full traceback, make only evidence-supported environment/implementation/validation/orchestration fixes, run focused lightweight tests, and resume cell-by-cell.
7. Preserve and skip valid cells. Do not overwrite valid outputs or change scientific methodology to suppress an error.
8. Do not run `server/server_run_binary.sh` or the old `server/server_supervised_run.sh` in Phase 1.
9. Create `results/logs/NONBINARY_REQUESTED_EXPERIMENTS_DONE` only after all requested concatenation and strict-transfer cells validate and finalization succeeds.
10. Write `results/logs/CODEX_NONBINARY_COMPLETION_STATUS.md` with completed, skipped, repaired, failed and unfinished cells.

Phase 1 completion check:

```bash
test -f results/logs/NONBINARY_REQUESTED_EXPERIMENTS_DONE \
  && echo 'NON-BINARY COMPLETE' \
  || echo 'NON-BINARY NOT COMPLETE'
```

Do not begin Phase 2 unless this marker is present and its claim has been independently checked against the ledger and performance/protocol files.

## Phase 2 — run the not-yet-executed `RUN_BINARY=1` work

Binary experiments have not yet been completed through the corrected workflow. Phase 2 should begin only after Phase 1 is verified.

The new Codex session should autonomously:

1. Require and validate `results/logs/NONBINARY_REQUESTED_EXPERIMENTS_DONE`.
2. Activate the exact project `.venv`.
3. Inspect `server/server_run_binary.sh`, its audit input, validators and existing binary files.
4. Run syntax checks and lightweight tests before training.
5. Run sequentially:

   ```bash
   bash server/server_run_binary.sh
   bash server/server_finalize_after_reruns.sh
   ```

6. Complete all requested remaining binary cells (previous planning expected up to 96), skip valid completed cells, use fixed thresholds, and never accept legacy test-fitted Youden metrics as clean fixed-threshold results.
7. Diagnose, safely fix, test and resume failures cell-by-cell. Do not restart valid non-binary work and do not execute cells concurrently.
8. Regenerate and verify final aggregation, tables, figures, `results/final_report.md`, and `results/results_summary.md`.
9. Create `results/logs/ALL_REQUESTED_EXPERIMENTS_DONE` only after every requested non-binary and binary result is genuinely complete and valid.
10. Write `results/logs/CODEX_ALL_EXPERIMENTS_COMPLETION_STATUS.md`.

Final check:

```bash
test -f results/logs/ALL_REQUESTED_EXPERIMENTS_DONE \
  && echo 'ALL EXPERIMENTS COMPLETE' \
  || echo 'EXPERIMENTS STILL INCOMPLETE'
```

## Scientific and data-safety rules

- Save all active results only under `results/` and logs under `results/logs/`.
- Never overwrite a valid completed result.
- Never delete valid Mac/server results, audit evidence, `results/`, `inductive_inputs/`, `final_output/`, or safety archives.
- Require folds 1–10 exactly once; a separate mean/std row is permitted.
- Require strict `.protocol.json` evidence for corrected transfer completion.
- Use `config/canonical_shared_disease_list.csv`.
- Preserve old transfer outputs without strict sidecars as stale historical evidence; do not count them as corrected results.
- Use fixed thresholds. Do not present legacy test-fitted Youden metrics as clean threshold-dependent results.
- Do not silently change datasets, disease lists, folds, models, feature definitions or evaluation protocol to make a run pass.
- Run expensive cells only on the server, sequentially and resumably.
- A missing marker means incomplete, regardless of how many log files exist.

## Suggested opening message for the new Codex chat

> Read `CODEX_SERVER_RUN_HANDOFF_2026-07-16.md` fully. Verify the repository, server environment, quota, logs, result validators and completion markers rather than assuming the handoff is current. Codex CLI previously disappeared from the tmux PATH, Phase 1 (`RUN_BINARY=0` repair/completion) is not complete, and Phase 2 (remaining binary work) has not been run. Restore/verify the Codex CLI first, then execute and babysit Phase 1 exactly as scoped. Diagnose, safely fix, test and resume failures until the non-binary completion marker is genuinely justified. Do not start Phase 2 until I explicitly ask or Phase 1 is independently verified. Preserve every valid result and all audit evidence.

