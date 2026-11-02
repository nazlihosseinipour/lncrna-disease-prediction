# Opus decisions and implementation spec — Felipe V1/V2 (2026-09-27)

Authoritative contract: `for claude/FELIPE_IMPLEMENTATION_AND_AUDIT_PLAN_2026-09-27.md` and
`for claude/GATES.md`. This file records supervisor decisions made at the first-wave checkpoint
and the shared spec every implementation task must follow. First-wave evidence:
`agentA_v2_population_audit.md`, `agentB_model_fidelity_audit.md`,
`agentC_execution_rescoring_design.md` (this folder).

## Decision log

| # | Decision | Rejected alternative and reason | Changes scientific meaning? |
|---|---|---|---|
| D1 | Freeze the current working-tree `IPCARF/ipcarf.py`, `RFLDA/rflda.py`, `RF/rf.py` and runner sources by SHA-256 recorded in every contract; runs refuse to start if hashes differ from the frozen table. | Commit the edits (forbidden: no commits); revert to HEAD (HEAD cannot express the paper profile). | No |
| D2 | Model profile and threshold mode are independent. Fit once, save raw scores once; Youden is only a rescoring mode. | Force `--model-profile paper` ⇒ Youden (Agent B): would reintroduce overwrite/mislabel risk and fitting-time thresholds. | No |
| D3 | Transfer resume unit = one (direction, model) cell: one fit, one full-target prediction, atomic write + contract + `.done`. Interrupted cell is redone whole. No persisted model, no prediction shards. The deterministic target folds are kept only as an evaluation partition recorded in the contract (for paper-style mean(std)). | Persist fitted model with joblib (Agent C): unnecessary when predictions of one fit are written atomically in one file; fits are never mixed. | No |
| D4 | Transfer default `--threshold-mode` becomes `fixed`; transfer writes raw predictions; `predict_proba` is called with `y=None`. No training-only thresholds for transfer this pass. | Keep target-label Youden default: leaks target labels into the clean lane. | Yes — transfer clean lane never uses target labels |
| D5 | 73-label within baselines reuse each version's own split (V1 released folds; V2 new group split) and pass the same manifest verification. | Separate provenance process: unnecessary, same X/rows. | No |
| D6 | One-class / zero-positive labels are kept (no filtering). Per-label AUROC/AUPRC = NaN with `one_class=true`; micro metrics pool all labels; per-population positive-count tables are written. | Drop zero-positive labels: violates "never silently filter disease columns". | No |
| D7 | V2 within split regenerated group-aware (user approved 2026-09-27): all 4,130 rows and 436 labels kept; groups = normalized-sequence SHA-256; deterministic group-level multilabel stratification; zero identical-sequence train/test overlap verified. Old `v2_all_436_splits.csv` kept unchanged as historical evidence. Used by V2 RF/IPCARF/RFLDA and the V2 73-label baseline. | Keep old split (identical X across train/test; 166 of 205 groups have conflicting labels); collapse duplicates (changes population). | Yes — V2→V2 fold composition changes |
| D8 | **Final (2026-09-27):** V2 population = `v2_sequence_available_4130`; ID-level ledger `config/v2_missing_sequence_ledger.csv` (1,208 rows, reason `missing_sequence`; 19 rat IDs flagged `species_mismatch_suspected`). Read-only `../fromserver/` check found no sequences; its only overlap with the 1,208 (`ABALON`) came from a gene-level sequence the corrected transcript pipeline rejected (`missing_canonical_transcript`). Evidence: `WAVE1_IMPLEMENTATION_ADDENDUM.md`. | Zero-fill / synthesize: forbidden. Reinstate ABALON from the gene-level sequence: not a transcript sequence; breaks the corrected contract. | Yes — declared population |
| D9 | `verify_felipe_exact_results.py --check protocol` is rescoped as a released-artifact reference check, clearly labelled, not part of the four-direction acceptance. | Retire outright: loses reproduction evidence. | No |
| D10 | RF uses `RandomForestRegressor` with sklearn default `max_features` (all features). Paper does not state it; recorded as open fidelity note, not changed. | — | No |

## Shared spec — prediction files (all tasks)

A *cell* is: within = (version, label_space, model, fold); transfer = (source, target, model).

Files per cell, in the cell's output directory:

- `<stem>.predictions.csv` — first column `ID` (RNA ID, string), then label columns in the
  exact manifest Y column order; values are raw model scores (not thresholded).
- `<stem>.contract.json` — JSON object with at least:
  `schema_version` (=1), `kind` (`within`|`transfer`), `version` or `source`/`target`,
  `label_space` (`all44`|`all436`|`shared73`), `feature_key`, `model`, `model_profile`,
  `effective_params` (dict), `seed`, `fold` (int or null), `row_ids_sha256` (SHA-256 of
  `"\n".join(IDs)`), `label_names_sha256` (SHA-256 of `"\n".join(label names)`),
  `n_rows`, `n_labels`, `input_hashes` (x, y, split, manifest, label map where applicable),
  `code_hashes` (runner + model sources), `library_versions`, `created_at_utc`,
  `prediction_file_sha256`. Transfer adds `removed_exact_target_ids` and
  `evaluation_partition` (list of target-ID lists per evaluation fold, plus its SHA-256).
- `<stem>.done` — written last; presence means complete. A cell without `.done`, or whose
  CSV hash ≠ `prediction_file_sha256`, or whose contract fields ≠ the current run's expected
  values, is rejected (resume redoes it; verifier fails it). Rejected files are never
  edited in place: resume moves them to `<outdir>/_rejected/<timestamp>/` before redoing.

Writes: temp file in same directory → flush → `os.fsync` → `os.replace`. Order: CSV,
contract, `.done`.

Nothing in the fit/predict path writes thresholded metrics. Metrics come only from
`scripts/rescore_felipe_predictions.py`, whose output rows and file names always contain
`evaluation_mode` (`fixed` | `paper_youden`). `paper_youden` rows also carry
`threshold_source = evaluated_fold_labels` and `lane = paper_comparison`; `fixed` rows carry
`threshold_source = constant_0.5` and `lane = primary_clean`.

## Shared API — `scripts/prediction_io.py`

```python
sha256_file(path) -> str
sha256_lines(items: Sequence[str]) -> str
atomic_write_bytes(path, data: bytes) -> None
atomic_write_text(path, text: str) -> None
write_prediction_cell(outdir, stem, predictions: pd.DataFrame, contract: dict) -> Path
    # predictions: index = RNA IDs (str), columns = ordered labels. Fills row_ids_sha256,
    # label_names_sha256, n_rows, n_labels, prediction_file_sha256, created_at_utc,
    # schema_version; writes CSV, contract, .done atomically in that order.
read_prediction_cell(outdir, stem, expected: dict | None = None) -> tuple[pd.DataFrame, dict]
    # Raises PredictionCellError if .done missing, hash mismatch, row/label hash mismatch,
    # duplicate IDs, non-finite values, or any key in `expected` differs from the contract.
cell_status(outdir, stem, expected: dict | None = None) -> Literal["missing","complete","invalid"]
quarantine_cell(outdir, stem) -> Path | None   # moves existing cell files to _rejected/<ts>/
class PredictionCellError(ValueError)
```

Imports: tests use `from scripts.prediction_io import ...` from the repo root (existing
convention). Runners add `ROOT / "scripts"` handling consistent with their current imports.

## Implementation ownership (exclusive)

| Task | Owns (may edit/create) | Depends on |
|---|---|---|
| T1 prediction I/O | `scripts/prediction_io.py`, `test/test_prediction_io.py` | — |
| T2 V2 group split | `scripts/make_v2_group_splits.py`, `test/test_v2_group_splits.py`, new `results/felipe_basics_2026-09-27/inputs/v2_all_436_group_splits.csv` (+ `.metadata.json`) | — |
| T3 transfer runner | `scripts/run_felipe_strict_transfer.py`, `test/test_transfer_resume.py` | T1 |
| T4 within runner | `scripts/run_inductive_within_cv.py`, `test/test_within_cv_resume.py` | T1 |
| T5 rescoring | `scripts/rescore_felipe_predictions.py`, `test/test_rescore_felipe_predictions.py` | T1 |
| T6 verifier + manifests + ledger | `scripts/verify_felipe_exact_results.py`, `test/test_verify_felipe_exact_results.py`, `config/felipe_basics_*.csv` (edits/new rows/new files), `config/felipe_frozen_sources.csv` (new), `config/v2_missing_sequence_ledger.csv` (new) | T2, D8 |

Global rules for every task: no git add/commit/stash/checkout/clean/push; no deletion or in-place
edits of existing results; no full model runs (probes ≤ a few minutes on tiny/1-fold inputs,
written under `results/felipe_probes_2026-09-27/<task>/`); no network; never append labels to
X; never align X/Y by position without an ID equality assertion; never filter disease columns.
