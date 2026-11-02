# Shared-sequence paired comparison — supervisor decisions and interface spec (2026-09-28)

Owner: Opus supervisor. Task plan: `HANDOFF_SHARED_SEQUENCE_2026-09-28.md` Section B.
Every sub-agent implements against this file. Where it is silent, ask the supervisor via your
final report rather than inventing a contract.

Repo root (`ROOT`): `/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project`
Python: `results/runtime/felipe_env/bin/python` (3.14; sklearn 1.7.2; pandas 2.3.3; numpy 2.3.5;
pytest 9; `tabulate` NOT installed).

## Supervisor pre-check (reproduced independently before any code was written)

Normalizing with `scripts/reconstruct_paper_sequence_features.normalize_sequence` and hashing
with SHA-256 of the UTF-8 normalized string:

- 48 shared hashes; 48 V1 rows (one per hash); 52 V2 rows; 4 hashes map to 2 V2 IDs each:
  `ASAP1-IT1`/`ASAP1‑IT1` (U+2011), `SPRY4-IT1`/`SPRY4‑IT1` (U+2011), `DIRC3`/`DIRC3-204`,
  `HIF1A-AS2`/`LINLK-A` → 44 one-to-one hashes. Only `NR_024015` has the same ID in both.
- V1 and V2 542-vectors for every shared pair are bit-identical (max |diff| = 0).
- V1 training = 307 rows, V2 training = 4,078 rows; zero exact 542-row overlap with the test rows.
- V1 X index equals raw V1 `ID` order; V2 identical-sequence groups == identical-X groups (3,849).

## Decisions

| # | Decision |
|---|---|
| SS1 | Output root `results/felipe_shared_sequence_2026-09-28/` (fresh). Probes under `results/felipe_probes_2026-09-28/shared_sequence/`. Nothing existing is edited. |
| SS2 | Test row identifier = `sequence_sha256` (64 lowercase hex). Test X = one row per shared hash (all 48), sorted ascending by hash, 542 columns in the existing X column order, values taken from the V1 X row and asserted bit-identical to every V2 X row of that hash. |
| SS3 | Training exclusion is identical for both cohorts (every row whose hash is among the 48 is removed from both versions). Therefore each (train version, model) is fitted **once** and predicts all 48 test rows; the primary (44) and sensitivity (48) cohorts are row subsets at scoring time. 6 fits total. No refit per cohort. |
| SS4 | Training data = transfer manifest `config/felipe_basics_transfer_73_manifest.csv` X/Y (hash-checked) restricted to the training IDs, in the manifest's row order. Labels = 73 shared concepts. |
| SS5 | Label versions for scoring: `v1` = the hash's single V1 row (shared73 Y); `v2` primary = the single V2 row; `v2` sensitivity = element-wise OR over all V2 IDs of the hash. Conflicts (disease where duplicate V2 IDs disagree) are listed, never hidden. V1 labels are identical in both cohorts. |
| SS6 | Primary thresholded lane = fixed 0.5 (`threshold_source=constant_0.5`). No Youden lane is implemented in this task (no source-only Youden, no test-label Youden). |
| SS7 | Prevalence baseline = per-disease prevalence of the **training** population of that train version (307 V1 or 4,078 V2 rows), as a constant score for every test row. Never test-label prevalence. |
| SS8 | Bootstrap: resample `sequence_sha256` rows with replacement within a cohort, B = 2000, `numpy.random.default_rng(20260928)`; one index matrix per cohort shared by every model/train version/label version/baseline, so paired differences are valid. Percentile 2.5/97.5. Replicates where a metric is undefined are dropped and counted. |
| SS9 | No minimum-positive disease filter. All 73 columns in every prediction and label file. Per-disease AUROC/AUPRC only where the evaluated labels have both classes; otherwise NaN + `defined=false`. |
| SS10 | New frozen table `config/felipe_shared_sequence_frozen_sources.csv` (created by the supervisor after review) = the 15 rows of `config/felipe_frozen_sources.csv` with identical hashes + the four new scripts. New runner refuses on any mismatch; no bypass flag. |

## Hard rules for every agent

- Never edit any file listed in `config/felipe_frozen_sources.csv`, nor that table. Import them.
- Never touch `results/felipe_final_2026-09-28/` or any existing result/report file.
- No git add/commit/stash/checkout/clean/push; no deletions; no network; no worktrees.
- Edit/create only the files your task owns. Tests use `tmp_path` and synthetic data where possible
  and import with `from scripts.<module> import ...` from the repo root.
- Do not run full model fits. The only allowed fits are tiny synthetic ones in tests, or probes the
  supervisor asks for.

## Frozen APIs to reuse

- `scripts.reconstruct_paper_sequence_features.normalize_sequence(value) -> (seq, removed_counter)`;
  empty seq = missing. Also `reconstruct(...)`, `feature_columns`, `load_properties`, `dcc_vector`,
  `kmer_vector` (read the file for signatures).
- `scripts.prepare_felipe_basics_inputs.normalized_label(name)`.
- `scripts.felipe_paper_models`: `MODELS`, `MODEL_SOURCES`, `build_paper_model(key, n_jobs=)`,
  `effective_params(key, model)`, `fitted_selection(key, model)`; put `ROOT/lncRNA_CIBCB2025-main`
  on `sys.path` before `build_paper_model`.
- `scripts.prediction_io`: `write_prediction_cell`, `read_prediction_cell`, `cell_status`,
  `cell_paths`, `quarantine_cell`, `verify_frozen_sources`, `library_versions`, `sha256_file`,
  `sha256_lines`, `atomic_write_text`, `PredictionCellError`, `FrozenSourceError`.
- `scripts.rescore_felipe_predictions.load_evaluator_class(ROOT)` → `(Evaluator, path)`; use
  `Evaluator(threshold_mode="fixed", threshold=0.5)._evaluate_inductive(y_true_df, y_pred_df)` →
  `(hamming, label_ranking, micro_roc, micro_auprc, precision, recall, fscore, accuracy)`.

Raw inputs: V1 `lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv` (columns `Unnamed: 0`,
`ID`, `seqs`, 285 diseases); V2 `Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv`
(`ID`, `seqs`, 436 diseases). Read IDs with `dtype=str`. Existing inputs (read-only):
`results/felipe_basics_2026-09-27/inputs/{v1,v2}_paper_sequence_542_X.csv`,
`{v1,v2}_shared_73_Y.csv`; label map `config/felipe_basics_shared_73_diseases.csv`.

## Interface 1 — cohort directory (written by S1, read by S3/S4/S5)

`<root>/cohort/` (default `<root>` = `results/felipe_shared_sequence_2026-09-28`):

| File | Content |
|---|---|
| `correspondence.csv` | one row per shared hash, sorted by hash. Columns: `sequence_sha256, sequence_length, v1_id, v1_x_row_index` (0-based position in V1 X), `v2_ids` (sorted, joined `" \| "`), `n_v2_ids, v2_x_row_indices` (joined `" \| "`, same order as `v2_ids`), `status` (`one_to_one` \| `ambiguous_multi_v2_id`), `same_id_v1_v2` (bool), `features_v1_equal_v2` (bool, bit-identical), `features_equal_reconstruction` (bool: 542 vector recomputed from the normalized sequence with the frozen reconstruction code equals the X row within `np.allclose(rtol=0, atol=1e-12)`), `v1_ambiguity_removed` / `v2_ambiguity_removed` (removed-symbol counts as `"N:1;Y:3"` or empty), `n_v2_label_conflicts, v2_conflict_diseases` (`" \| "`), `n_v1_v2_label_disagreements_primary` (V1 vs V2 single/OR label disagreements, informational), `in_primary, in_sensitivity` (bool) |
| `test_X.csv` | index column `sequence_sha256`, 48 rows sorted by hash, 542 feature columns (exact X column order) |
| `labels_v1.csv` | index `sequence_sha256`, 48 rows, 73 columns in label-map `order` |
| `labels_v2_primary.csv` | index `sequence_sha256`, the 44 one-to-one rows only, 73 columns |
| `labels_v2_or.csv` | index `sequence_sha256`, 48 rows, OR over V2 IDs, 73 columns |
| `cohorts.csv` | `sequence_sha256, in_primary, in_sensitivity` (48 rows) |
| `train_ids_v1.txt`, `train_ids_v2.txt` | one ID per line, manifest X row order (307 / 4,078) |
| `excluded_ids_v1.txt`, `excluded_ids_v2.txt` | IDs removed because their hash is shared (48 / 52) |
| `v2_label_conflicts.csv` | long form: `sequence_sha256, disease, v2_id, label` for every hash with >1 V2 ID and a disagreeing disease |
| `fromserver_check.json` | read-only findings about `../fromserver/` mappings (see S1) |
| `cohort_manifest.json` | `schema: "shared_sequence_cohort/1"`, counts, `inputs` {path: sha256} for raw files, X, Y, label map, transfer manifest; `outputs` {filename: sha256} for every file above except itself; `assertions` {name: true}; `created_at_utc`; `code` {builder path: sha256} |

All CSV floats written with `float_format="%.17g"`; labels as ints 0/1. Written via
`prediction_io.atomic_write_text`; the builder refuses to write into an existing non-empty cohort
dir.

## Interface 2 — prediction cells (written by S3, read by S4/S5)

`<outdir>/paired/train_<v>__shared73__paper_sequence_542__paper/<model>/shared48.{predictions.csv,contract.json,done}`
via `prediction_io.write_prediction_cell`. Prediction index = `sequence_sha256` in `test_X.csv`
order; columns = 73 labels in label-map order; values = raw `predict_proba` scores.

Caller-supplied contract fields:
`kind="shared_sequence_paired"`, `train_version`, `label_space="shared73"`,
`feature_key="paper_sequence_542"`, `model`, `model_profile="paper"`, `effective_params`, `seed`,
`test_population="shared_sequence_all48"`, `test_x_sha256` (file hash of `cohort/test_X.csv`),
`test_row_ids_sha256` (`sha256_lines` of test hashes), `train_ids_sha256` (`sha256_lines` of the
training IDs actually fitted), `n_train`, `cohort_manifest_sha256`, `input_hashes`
{`train_x`, `train_y`, `transfer_manifest`, `label_map`, `test_x`}, `code_hashes`
(from `verify_frozen_sources`), `library_versions`, `probe` (`null` for real runs, else
`{"max_train_rows": N}`), `fitted_selection`, `execution` {`n_jobs`, `hostname`, `fit_seconds`}.
Resume expected keys = all fields except `fitted_selection`, `execution`.

## Interface 3 — scores (written by S4)

`<outdir>/` (must not exist or be empty): `summary.csv` (one row per model incl. `prevalence_baseline`
× train_version × cohort × label_version), `per_disease.csv`, `bootstrap_ci.csv`,
`paired_differences.csv`, `disease_definedness.csv`, `score_run.json` (input/output hashes).

## Decisions added during review

| # | Decision |
|---|---|
| SS11 | S2 audit (`results/felipe_shared_sequence_2026-09-28/disease_map_audit/`) passes all six checks; the 73-concept map is used unchanged. Near-miss spellings it did not merge (possessive "'s" vs none, e.g. Alzheimer/Parkinson/Huntington; singular/plural; oesophageal/esophageal; "unveal" typo) are reported as a limitation, not merged: merging would change the prespecified 73-label space. Map `v2_positives` refer to the 4,130-row population. |
| SS12 | Float parsing: every reader of `cohort/test_X.csv` and the 542 X files that compares values bit-for-bit must use `pd.read_csv(..., float_precision="round_trip")`. Under it `test_X` equals the V1/V2 X rows exactly; the default pandas parser differs by ≤1.8e-15 on the 17-digit `test_X` text (it is not correctly rounded). On the original 8-digit X files both parsers give identical values (checked), so the fitted training data equals what the within runs used. |
| SS13 | S1 cohort accepted (supervisor re-derived hashes, test_X rows, V1/V2-primary/V2-OR labels, cohorts, train IDs independently). `fromserver/` holds no RNA-level V1↔V2 mapping; its only relevant record is `removed_exact_target_ids=["NR_024015"]` (1 of 48) and a count `normalized_mapped_overlap=14`. |
| SS14 | Probe found the runner rejected the real V1 ID `BX374997.2 ` (trailing space; present identically in raw, X, Y and cohort files; used by the verified within runs). Supervisor edit: IDs are read byte-exact; only `\r` is rejected. Scorer and builder already preserve it. The new frozen table was regenerated for the runner before any fit used it. |
| SS15 | S5's independent verifier found the scorer read predictions via `read_prediction_cell`, whose default pandas parser is not correctly rounded on 17-digit text (≤1.1e-16 per value) and merged distinct scores into ties, shifting micro AUROC by up to 2.1e-5 in the probe. Supervisor fix: after `read_prediction_cell` verifies the cell, the scorer re-reads the same bytes with `float_precision="round_trip"` (asserting equal IDs/labels and |Δ| ≤ 1e-15) and scores the exact values. The fit-time frozen table `config/felipe_shared_sequence_frozen_sources.csv` (recorded in every full-run cell) is left unchanged; `config/felipe_shared_sequence_scoring_frozen_sources.csv` differs only in the scorer row. The verifier checks cell `code_hashes` against the fit-time table and current files against the scoring table. Note: the frozen within-version rescorer uses the same default parser; the verified within6 values are therefore subject to the same ≤~1e-5 AUROC tie effect (not modified here; flagged). |
| SS16 | Incident (supervisor error, 09:19–09:25 UTC): to test real-run resume the supervisor ran the runner with `--resume` against the scoring table. Because `code_hashes` are resume keys and the scorer row changed, every verified cell was judged invalid, quarantined (moved, not deleted) and refitted; the run was killed during V2 RFLDA (no run record: SIGTERM). Recovery: the six original cells were moved back; the five accidental refits were moved to `<cell>/_rejected/20260928T0925_accidental_refit_supervisor_error/`. Evidence gained: V1 refits byte-identical; V2 RF/IPCARF refits differ in 16/9 values by ≤1.1e-16 (multithreaded prediction accumulation), AUROC identical. Verifier re-run after restore: cells/scores/protected PASS. Lesson recorded: `--resume` is safe only with unchanged code/inputs; after any code change it refits (by design, same as the transfer runner) — never "test" resume on a completed run. With the fit-time table the runner refuses (FrozenSourceError), which is the intended guard. |
| SS17 | Nazli approved (2026-09-28): (1) the audited 73-disease map stays unchanged for the primary analysis; possessive/plural/spelling/typo variants are documented as limitations and the primary label space is not modified after observing results; (2) the 44 one-to-one cohort is primary; the all-48 sensitivity cohort OR-combines labels across V2 IDs sharing a sequence, retaining the full list of 11 conflicting labels (`cohort/v2_label_conflicts.csv`). Wording corrected (documentation only, no refit): "no detectable sequence-based disease signal under this experimental design"; results "consistent with differences arising mainly from label annotations and prevalence". |
