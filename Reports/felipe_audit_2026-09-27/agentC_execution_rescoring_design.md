> NOTE: repo paths referenced below (e.g. `scripts/...`, `config/...`,
> `lncRNA_CIBCB2025-main/...`) are relative to the project root
> `/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project`,
> which this audit read directly (not through this worktree).

# Agent C — Execution & Rescoring Design (Felipe V1/V2 audit, 2026-09-27)

Read-only audit. No files edited, no models run. All line numbers refer to the
working-tree (uncommitted) state at time of review.

## Q1 — Current formats

**Within (`scripts/run_inductive_within_cv.py`)**
- Per-fold raw probabilities ARE saved: `pred_dir / f"test_fold{fold+1}.csv"`
  (line 357), `pred = pd.DataFrame(model.predict_proba(...), columns=y_test.columns,
  index=y_test.index)` (lines 352–356) — RNA IDs as the row index, disease-label
  names as columns, values are probabilities. Good baseline shape.
- Write is **not atomic**: `pred.to_csv(pred_dir / ...)` (line 357) writes
  directly to the final path — no temp file, no `fsync`, no `os.replace`. A
  crash mid-write leaves a truncated/corrupt CSV at the real filename.
- "Completion marker": none at the fold level. The only provenance artifact is
  `pred_dir / "prediction_contract.json"` (lines 304–319), which records
  `x_sha256`/`y_sha256`/`split_sha256`/`model_source_sha256`/config — but this
  is a **directory-level** contract written once, not a per-file
  hash/checksum of the prediction CSV itself, and not a "done" flag.
- `--resume` (lines 329–340, `load_existing_prediction`): reuses an existing
  `test_fold{N}.csv` only if (a) not OOB-threshold mode, (b) `--resume` passed,
  (c) file exists, and (d) `list(pred.columns) == list(y_test.columns)` and
  `set(pred.index) == set(y_test.index)`. **It does not hash-check the file
  contents** — a truncated file with the right header/index but wrong values,
  or a partially-written file that happens to still parse, would be silently
  accepted and reused. It also does not re-validate against the directory
  contract's recorded hashes at reuse time (only at initial contract creation).
- Aggregation: `within_version_cv_summary.csv` merges cumulatively by key
  `["version", "feature_key", "model", "model_profile"]` (line 496) — see Q2.

**Transfer (`scripts/run_felipe_strict_transfer.py`)**
- **No raw prediction files are ever written.** `predictions` (line 138–142) is
  computed per fold and passed straight into `evaluator.evaluate(...)`
  (line 143); it is never persisted with `to_csv`. Only aggregate
  per-fold-metric CSVs (`{model}_{source}_to_{target}.csv`, line 146–147) and a
  `.protocol.json` sidecar (lines 164–184) are saved. This means the transfer
  lane cannot currently be rescored later (Q1's "raw probabilities saved"
  requirement fails today) and full model refits are the only way to get new
  metrics.
- No `--resume` flag exists at all. If interrupted, a rerun re-fits every
  model in `args.models` from scratch (`model.fit(x_source, y_source)`,
  line 132, happens unconditionally at the top of the per-model loop).
- Writes are not atomic (`performance_frame.to_csv(output, ...)`, line 147;
  `output.with_suffix(".protocol.json").write_text(...)`, line 182–184).
- Completion marker: none. `summary.csv` (lines 186–196) is a cumulative,
  dedup-by-key append, so a half-finished run leaves no per-model/per-fold
  marker distinguishing "done" from "not yet run."

## Q2 — Summary-key bug (confirmed)

`scripts/run_inductive_within_cv.py:496`:
```python
key = ["version", "feature_key", "model", "model_profile"]
```
This dedup/merge key for `within_version_cv_summary.csv` **omits
`threshold_mode`**. Two invocations that differ only by
`--threshold-mode fixed` vs `--threshold-mode youden` produce summary rows
with identical `(version, feature_key, model, model_profile)` tuples, so the
second run's row replaces the first's (lines 497–500:
`existing = existing[~existing[key].apply(tuple, axis=1).isin(done)]`). The
per-run `performance_csv` files on disk are distinct problems too — the
filename `f"{experiment_key}_{model_key}_performance.csv"` (line 417–420) does
not encode `threshold_mode` either, so a fixed run and a Youden run for the
same experiment_key/model **also overwrite the same performance CSV file on
disk**, not just the summary row. This is a real bug: confirmed at both the
summary-merge key (line 496) and the performance-file path (lines 417–420).

## Q3 — Transfer runner correctness

- **Target Y used only for evaluation, with one caveat.** `model.fit(x_source,
  y_source)` (line 132) is the only fit call — target labels never enter
  training. `y_target` is however used to build the `MultilabelStratifiedKFold`
  split (line 119: `splitter.split(np.zeros(...), y_target.to_numpy())`) —
  this only decides which target rows land in which evaluation shard, it does
  not touch any fitted parameter, so this is benign (fold membership is not a
  fitted statistic) but should be called out explicitly in the sidecar
  contract since it is target-label information used before evaluation.
- **Source-ID removal is generic, not ID-specific.** Lines 109–114:
  `overlap = sorted(set(x_source.index) & set(x_target.index))`; if non-empty,
  those rows are dropped from `x_target`/`y_target`. The plan doc
  (`for claude/FELIPE_IMPLEMENTATION_AND_AUDIT_PLAN_2026-09-27.md:114`) names
  the current known overlap as `NR_024015`; the code does not hardcode that ID
  — it recomputes the overlap from the index sets every run, so it will still
  catch `NR_024015` (and anything else) as long as the manifests are the
  frozen ones. No assertion confirms the overlap set actually contains the
  expected ID, so a manifest edit that silently changed IDs would not be
  flagged.
- **Scaler/PCA/selector/threshold fit on source only:** `build_model` always
  constructs a fresh model (lines 63–78) and `model.fit(x_source, y_source)`
  is the only fit; per-model internal preprocessing (RF/IPCARF/RFLDA scalers,
  PCA, feature selection) is therefore fit only on source. Fixed-threshold
  scoring is fine; **Youden threshold is not fit on source** — see Q4, it is
  computed from target/test labels at evaluation time, which is the leak.
- **73 labels in identical order both sides:** enforced by
  `if y_source.columns.tolist() != y_target.columns.tolist(): raise` (line
  106–107) and by the transfer manifest's shared `y_path`s
  (`v1_shared_73_Y.csv`, `v2_shared_73_Y.csv`) both built from
  `felipe_basics_shared_73_diseases.csv`'s `canonical_name` ordering.
- **Failure on interruption today:** no checkpointing exists at all. If the
  process dies mid-loop (e.g., after fitting `rf` and `ipcarf` but before
  `rflda`), rerunning the script re-fits **all** models named in `--models`
  from scratch — including the already-completed ones — because there is no
  persisted model state, no per-model completion marker, and no raw
  prediction file to reuse. This directly violates the plan's "fit once, score
  twice" and "resumable" goals (plan doc lines 116–118, 189–191).

## Q4 — Youden threshold source (confirmed test-label leakage)

`lncRNA_CIBCB2025-main/parse_results/evaluate.py`:
- `_evaluate_inductive` (line 77): when `threshold_mode != "fixed"`,
  `thresholds = self.sensivity_specifity_cutoff(y_true, y_pred)` (line 85),
  where `y_true` is the **same `y_test`/target-fold labels being evaluated**
  in that call.
- `sensivity_specifity_cutoff` (lines 153–200) computes an ROC curve per label
  from `y_true` (the evaluation labels) and `y_score` and picks the
  Youden-optimal cut per label (lines 191–199).
- Both runners call this with the evaluation split's own truth labels:
  within-CV's `evaluator.evaluate(y_test, pred)` (run_inductive_within_cv.py,
  e.g. line 347/376) and transfer's `evaluator.evaluate(y_test, predictions)`
  (run_felipe_strict_transfer.py:143), where `y_test`/`predictions` in the
  transfer case are **target-labels** for the very shard being scored.
- **There is no "clean" path today that avoids this.** `threshold_mode="fixed"`
  (0.5) is the only mode that doesn't touch test/target labels. The
  `train_oob_f1` mode added in `run_inductive_within_cv.py`
  (`select_global_f1_threshold`, lines 164–202) is training-only and correct,
  but it is within-CV-only (RF model only, gated at lines 280–283) and has no
  transfer equivalent. The evaluate.py docstring itself flags Youden as
  "legacy... not valid as a primary estimate" (lines 42–44), confirming this
  is a known, not hidden, leakage — but it is still the transfer runner's
  **default** (`--threshold-mode` default is `"youden"`, line 89).

## Q5 — `verify_felipe_exact_results.py`: what it checks, what's obsolete

Four `--check` modes exist today: `protocol`, `matrix`, `manifests`, `report`
(lines 108, 137). None of these names need renaming — GATES.md's G2/G6/G7
already invoke this same script with `--check manifests` / `--check matrix` /
`--check report` respectively, and those match (G2→manifests line 17,
G6→matrix line 30, G7→report line 34) — so the CLI surface is already aligned
with the gates; only the assertion content needs to change.

- `check_protocol` (lines 45–81) is **hardcoded to the old V1-only, 44-label,
  827-feature, 285-association paper-reproduction artifact**
  (`results/audit/felipe_exact_results_2026-09-27/protocol.md`), asserting
  literal strings including `"missing Pse-in-One generation configuration"`
  (line 54) — i.e. it still asserts the **conclusion that V2's feature recipe
  is missing/unreconstructable**. That conclusion is now obsolete: the project
  has since built `paper_reconstructed_sequence_542` features for **both** V1
  and V2 (`config/felipe_basics_sequence_542_manifest.csv` has rows for both
  versions with `preprocessing_mode=reconstructed_from_sequence_no_label_features`)
  and a 73-label shared-disease transfer manifest
  (`config/felipe_basics_transfer_73_manifest.csv`). `check_protocol` and
  `check_report`'s asserted phrases ("did not build the same paper feature
  construction for V2", "Do not use the old V2 RF values", lines 126–128) are
  stale narratives from before that reconstruction existed.
- `check_matrix` (lines 84–105) hardcodes exactly 6 rows
  (`V1_to_V1`/`V2_to_V2` × `RF`/`IPCARF`/`RFLDA`) and asserts `V2_to_V2` has
  `requested_protocol_status == "MISSING_EXACT_PROTOCOL"` with an empty
  `source` (lines 103–104) — again encoding the "V2 can't be reproduced"
  conclusion. This needs to become 4-direction × 3-model (12 cells, or more
  once 73-label within baselines are added) once G6 is implemented.
- `check_manifests` (lines 108–118) is generic/still valid, but only checks
  `felipe_basics_{v1,v2}_manifest.csv`, not the newer
  `felipe_basics_sequence_542_manifest.csv` / `felipe_basics_transfer_73_manifest.csv`
  / `felipe_basics_shared_73_diseases.csv` — needs extension for G2's "frozen
  manifests" scope.

## Q6 — 73-label within-baseline path

**No such path exists today.** `felipe_basics_sequence_542_manifest.csv` has
exactly two rows: V1 with `y_path=v1_paper_44_Y.csv` (44 labels) and V2 with
`y_path=v2_all_436_Y.csv` (436 labels) — neither is the 73-shared-disease
label set. To produce 73-label within baselines (needed per plan doc line 118
and GATES.md G6 "matching 73-label within baselines"), two new manifest rows
are needed, reusing existing split files:
- V1, 73-label: `x_path=v1_paper_sequence_542_X.csv`,
  `y_path=v1_shared_73_Y.csv` (already exists, built for transfer),
  `split_path=lncRNA_CIBCB2025-main/.../inductive_splits.csv` (**V1's own
  10-fold split** — same one V1 within-CV already uses).
- V2, 73-label: `x_path=v2_paper_sequence_542_X.csv`,
  `y_path=v2_shared_73_Y.csv` (already exists), `split_path=v2_all_436_splits.csv`
  (**V2's own split** — same one V2's 436-label within-CV uses).
No new split generation is required — only new manifest rows (with fresh
`feature_key`, e.g. `paper_sequence_shared73`) pointing at already-frozen X and
the already-built shared-73 Y files, each still hash-validated by
`experiment_contract.validate_manifest_row`.

---

# Proposal

## 1. Single prediction file format

Per cell (within: version×feature_key×model×fold; transfer: direction×model×
shard):

**Prediction file** `predictions.csv` (or `test_fold{N}.csv` for within, kept
for backward compatibility):
- Row index: RNA ID (string), one row per evaluated sample.
- Columns: ordered label names exactly matching the manifest's Y column order
  (no re-sorting) — probabilities, one column per label.

**Sidecar contract** `<same-stem>.contract.json`:
```json
{
  "schema_version": 1,
  "direction": "within|v1_to_v2|v2_to_v1",
  "version_or_source_target": {"source": "v1", "target": "v2"},
  "feature_key": "paper_sequence_542",
  "model": "rf|ipcarf|rflda",
  "model_profile": "current|paper",
  "effective_params": {"n_estimators": 150, "random_state": 0, "...": "..."},
  "seed": 0,
  "fold_or_shard_id": "fold3",
  "input_hashes": {"x_sha256": "...", "y_sha256": "...", "split_sha256": "...",
                    "label_map_sha256": "..."},
  "code_hashes": {"model_source_sha256": "...", "runner_source_sha256": "..."},
  "row_count": 355,
  "created_at_utc": "2026-09-27T20:47:13.795331Z",
  "prediction_file_sha256": "..."
}
```
`label_map_sha256` is the hash of the ordered label-name list actually used
(guards against manifest label reordering being silently accepted). The
contract's own `prediction_file_sha256` is computed **after** the prediction
file is finalized and is what resume/verify checks against — not just column
names/index sets as `run_inductive_within_cv.py` does today.

## 2. Atomic write + completion marker

For every prediction file and its contract:
1. Write to `*.tmp-<pid>-<random>` in the same directory.
2. `f.flush(); os.fsync(f.fileno())`.
3. `os.replace(tmp_path, final_path)` (atomic on POSIX, same filesystem).
4. Compute `prediction_file_sha256` from the now-final file, write the
   contract JSON the same way (temp + fsync + replace).
5. Only after both replaces succeed, write a zero-byte `*.done` marker
   (also via temp+replace) whose presence is the single source of truth for
   "this shard is complete and verified." A resume scan looks for `*.done`
   files, not for the CSV's mere existence.

This guarantees a reader never observes a partially-written prediction file,
and a missing `.done` marker unambiguously means "redo this shard," regardless
of what partial bytes happen to be sitting in the CSV.

## 3. Transfer `--resume` with deterministic shards and single-fit guarantee

- **Deterministic target shards**: derive `MultilabelStratifiedKFold` splits
  from a fixed `random_state` (already done, line 116–118) and additionally
  hash the resulting fold assignment (row IDs per fold) into the contract, so
  a resumed run can verify shard membership hasn't drifted (e.g. because the
  manifest or overlap set changed) before reusing any file.
- **One model fit per (direction, model), recommendation: persist the fitted
  model, not deterministic refit.** Rationale: even with a fixed seed,
  `model.fit` reruns are not guaranteed bit-identical across library/BLAS
  versions or thread counts (RF/IPCARF/RFLDA all use randomized components),
  so "refit deterministically" risks silently producing different
  predictions on resume than the interrupted run would have — exactly the
  kind of drift this audit is trying to eliminate. Persist the fitted
  estimator (`joblib.dump`) alongside a contract recording
  `source_x_sha256`/`source_y_sha256`/`model_source_sha256`/`seed`/library
  versions immediately after `model.fit` succeeds (atomic write, as above).
  On resume: if a valid persisted model + matching contract exists, load it
  and only re-run the *scoring* loop over shards missing a `.done` marker;
  if no valid persisted model exists, fit exactly once. This satisfies "ONE
  model fit per (direction, model)" literally, and never re-derives training
  results from target data.
- **Corrupted/partial shard rejection**: on resume, for each target shard,
  require: contract exists, `prediction_file_sha256` matches a freshly
  computed hash of the CSV on disk, `.done` marker exists, and the contract's
  `input_hashes`/`code_hashes`/`seed`/label order match the current run's
  computed values. Any mismatch → delete the stale shard's files and refit
  that shard's *scoring* (not the persisted model) from scratch. Never
  silently accept a shard whose recorded hash doesn't match its bytes.

## 4. `scripts/rescore_felipe_predictions.py` (new script)

- **Inputs**: `--predictions-dir` (root to scan for `*.contract.json` +
  `.done` shards), `--mode {fixed,paper_youden}` (reserve
  `train_oob_f1`-style training-only calibration as a documented future mode,
  not implemented now since it needs training-fold OOB scores that transfer
  doesn't currently produce), `--outdir`.
- For each accepted shard: load prediction CSV + contract, verify
  `prediction_file_sha256`, load the matching Y (validated by manifest hash),
  compute metrics.
- **`fixed`**: threshold 0.5 per label, no test/target label fitting.
- **`paper_youden`**: reproduces `sensivity_specifity_cutoff` for
  paper-comparison purposes only, but the output must be clearly labeled
  non-primary (mirroring `evaluate.py`'s own docstring warning).
- **Output paths/keys**: `<outdir>/<direction>_<model>_<mode>_metrics.csv`,
  with an explicit `evaluation_mode` column (`fixed` / `paper_youden` /
  future `train_oob_f1`) included in every row and in the filename — this is
  the direct fix for the Q2 bug: **threshold_mode must always be part of both
  the summary dedup key and the output filename**, never omitted.
- **No overwrite**: refuse to write over an existing metrics file unless
  `--force`; on collision, write `<...>_<timestamp>.csv` and warn.
- **Metrics**: per-label AUROC/AUPRC + micro AUROC/AUPRC (rank-based, no
  threshold needed) plus per-label and micro F1/precision/recall/specificity/
  accuracy (threshold-dependent, per `--mode`).
- **One-class labels**: if a label's Y column in the shard has a single class,
  skip AUROC/AUPRC for that label (record `NaN` + a `one_class=true` flag
  rather than raising, mirroring `evaluate.py`'s existing 0.5-threshold
  fallback at lines 187–188) and exclude it from the micro-average
  denominator's per-label loop but keep it in micro pooled counts only if
  both classes exist somewhere in the pooled set.

## 5. Verifier changes (`verify_felipe_exact_results.py`)

- `--check manifests`: extend to also validate
  `felipe_basics_sequence_542_manifest.csv`,
  `felipe_basics_transfer_73_manifest.csv`, and
  `felipe_basics_shared_73_diseases.csv` (row counts, required hash columns,
  and — for the transfer manifest — that `n_labels == 73` and V1/V2 label
  columns match the shared-diseases file's canonical order).
- `--check matrix`: replace the hardcoded 6-row V1/V2-only expectation with a
  derived expectation set: 4 directions (`V1_to_V1`, `V2_to_V2`, `V1_to_V2`,
  `V2_to_V1`) × 3 models, **plus** the new 73-label within baselines (2
  more version rows × 3 models) — all with `requested_protocol_status ==
  "ACCEPTED"` once G4/G5 land, sourced from `.done`-marked, hash-verified
  shards only (no more `MISSING_EXACT_PROTOCOL` placeholder rows).
- `--check report`: replace the stale "did not build the same paper feature
  construction for V2" / "Do not use the old V2 RF values" assertions (lines
  126–128) with assertions that the report states population/feature/label
  counts, model profile, and threshold mode per GATES.md G7, and does not
  contain a stale "V2 unreproducible" claim now that the 542-feature
  reconstruction covers both versions.
- `--check protocol`: either retire this check (it documents a historical,
  now-superseded V1-only artifact) or rename/rescope it explicitly as a
  historical V1-paper-fidelity reference check so it's not confused with the
  current four-direction scope — recommend retiring per G6's language ("all
  four directions... complete, hash-accepted"), since keeping an assertion
  that hardcodes "missing Pse-in-One generation configuration" would
  actively contradict the new reconstructed-542 manifests.

## 6. Task split — exclusive file ownership

| Task | Files owned (exclusive) | Must add tests | Probe command |
|---|---|---|---|
| A. Prediction/contract I/O library | `scripts/prediction_io.py` (new: atomic write, sha256, `.done` marker, contract schema) | `test/test_prediction_io.py`: atomic write under simulated crash (kill before replace → no partial final file); hash mismatch detection | `python -c "from scripts.prediction_io import atomic_write_csv; ..."` on a scratch dir |
| B. Within-CV runner fixes | `scripts/run_inductive_within_cv.py` only | `test/test_within_cv_resume.py`: threshold_mode included in summary key and filename; corrupted fold CSV rejected on resume | `results/runtime/felipe_env/bin/python scripts/run_inductive_within_cv.py --manifest config/felipe_basics_sequence_542_manifest.csv --models rf --versions v1 --max-folds 1 --resume` |
| C. Transfer runner resume + raw predictions | `scripts/run_felipe_strict_transfer.py` only | `test/test_transfer_resume.py`: interrupt-then-resume produces identical predictions to an uninterrupted run; one fit per (direction, model) verified via fit-call counter/mock | `results/runtime/felipe_env/bin/python scripts/run_felipe_strict_transfer.py --source v1 --target v2 --models rf --n-splits 2 --resume` |
| D. Rescoring script | `scripts/rescore_felipe_predictions.py` (new) | `test/test_rescore_felipe_predictions.py`: fixed vs paper_youden produce distinct, non-overwriting outputs; one-class label handled without raising | `results/runtime/felipe_env/bin/python scripts/rescore_felipe_predictions.py --predictions-dir results/felipe_basics_2026-09-27/transfer --mode fixed --outdir /tmp/rescore_probe` |
| E. Verifier + manifests | `scripts/verify_felipe_exact_results.py`, `config/felipe_basics_sequence_542_manifest.csv` (add 73-label rows), new `config/felipe_basics_within73_manifest.csv` | `test/test_verify_felipe_exact_results.py`: matrix check reflects 4×3(+2×3) cells; manifests check covers new files | `results/runtime/felipe_env/bin/python scripts/verify_felipe_exact_results.py --check manifests` |

No two tasks write the same file. Task A is a dependency for B/C/D (import
only, no shared mutable state) and should land first or be stubbed
concurrently with an agreed function signature.

## Open questions for the supervisor

1. Confirm the recommendation in §3 (persist fitted model vs. deterministic
   refit) — persisting means `joblib`/pickle artifacts of RF/IPCARF/RFLDA
   models on disk; is that acceptable storage-wise, or is deterministic refit
   with strict environment/version pinning preferred despite the drift risk?
2. Should `check_protocol` be retired outright or renamed/rescoped as a
   historical V1-only reference check (see §5)? Retiring changes G-numbering
   expectations if anything currently depends on `--check protocol`.
3. `train_oob_f1`-style training-only threshold calibration has no transfer
   equivalent today (transfer never sees OOB scores, and IPCARF/RFLDA don't
   expose OOB at all). Is a training-only clean threshold for transfer
   in scope for this pass, or explicitly deferred (as proposed in §4)?
4. Do the 73-label within-baseline manifest rows (Q6/§1 Task E) need their own
   frozen `feature_key`/provenance review before G2 will accept them, or can
   Task E add and hash-validate them directly?
