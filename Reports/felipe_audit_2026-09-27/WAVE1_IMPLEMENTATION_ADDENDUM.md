# Wave 1 implementation addendum (Opus, 2026-09-27)

Supplements `OPUS_DECISIONS_AND_SPEC.md`. Where the two differ, this file is the more
specific rule for wave T1–T6. Repo root: the main checkout
`/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project`.
Python: `results/runtime/felipe_env/bin/python` (3.14, sklearn 1.7.2, pandas 2.3.3).

## D8 final — V2 population

Declared population: **`v2_sequence_available_4130`**. The 1,208 raw rows without a usable
transcript sequence are excluded with reason `missing_sequence`; 19 rat IDs
(18 `NONRATT…`, 1 `ENSRNOT…`) additionally flagged `species_mismatch_suspected`.

Evidence from the read-only `../fromserver/` check:

- `fromserver/` holds 3,441 files (CSV/JSON/MD/figures): performance tables, protocol
  sidecars, and fold predictions. It contains **no sequence file and no CSV with a
  sequence column**. Its run configs point to a different, pre-correction input tree
  (`inductive_inputs/canonical_all_safe/v2/…`) with different features and disease
  handling; these are historical results, not Felipe results.
- Its V2 prediction files cover 4,114 RNA IDs. Of the 1,208 missing IDs, exactly one
  (`ABALON`) appears there. The historical pipeline had a 1,903-nt **gene-level** sequence
  for it (`Data/output_data/website_sequences.csv`, resolved `ABALON → ENSG00000281376`).
  The corrected transcript-level pipeline rejected it:
  `Data/corrected_inputs/v2_transcript_sequences/transcript_refetch_report.csv` records
  `ENSG00000281376, missing_canonical_transcript`. A gene-level sequence is not an
  authoritative transcript sequence under the current contract, so ABALON stays excluded
  and its ledger row cites this.
- The 17 IDs present now but absent from the server population were recovered by the
  corrected pipeline; they are not relevant to the 1,208.

## Output layout (paths are informational; tools discover cells by contract)

```
<outdir>/within/<version>__<label_space>__<feature_key>__<model_profile>/<model>/fold01 … fold10
<outdir>/transfer/<source>_to_<target>__<label_space>__<feature_key>__paper/<model>/full_target
```
each as `<stem>.predictions.csv`, `<stem>.contract.json`, `<stem>.done` via
`scripts/prediction_io.write_prediction_cell`.

## Contract fields (both kinds) — callers supply these; prediction_io adds the derived ones

| field | within | transfer |
|---|---|---|
| `kind` | `"within"` | `"transfer"` |
| `version` | `"v1"`/`"v2"` | absent |
| `source`, `target` | absent | `"v1"`/`"v2"` |
| `label_space` | manifest `label_space` column (`all44`/`all436`/`shared73`) | `"shared73"` |
| `feature_key` | manifest `feature_key` | `"paper_sequence_542"` (from manifest) |
| `model` | `rf`/`ipcarf`/`rflda` | same |
| `model_profile` | `"paper"` | `"paper"` |
| `effective_params` | `felipe_paper_models.effective_params(key, model)` (pre-fit) | same |
| `fitted_selection` | `felipe_paper_models.fitted_selection(key, model)` (post-fit; never an acceptance key) | same |
| `seed` | `0` | `0` |
| `fold` | 1-based int | `null` |
| `n_folds_total` | `10` (run's `--n-splits`) | `null` |
| `truth_y_path` | manifest `y_path` (repo-relative POSIX) | target `y_path` |
| `truth_y_sha256` | SHA-256 of that file | same |
| `input_hashes` | `{x, y, split, manifest}` SHA-256s | `{source_x, source_y, target_x, target_y, manifest, label_map}` |
| `code_hashes` | dict returned by `prediction_io.verify_frozen_sources` | same |
| `library_versions` | `prediction_io.library_versions()` | same |
| `execution` | `{n_jobs, fold_workers, hostname}` (never an acceptance key) | `{n_jobs, hostname}` |
| `removed_exact_target_ids` | absent | sorted list |
| `evaluation_partition` | absent | `{method, n_splits, random_state, uses_target_labels: true, folds: [[IDs]…], sha256}` |

`evaluation_partition.sha256` = `sha256_lines` of the fold lists, each fold joined by
`","` and prefixed by its 1-based index (`"1:ID,ID…"`). It only groups target rows for
paper-style mean(std) reporting; it touches no fitted parameter.

Resume "expected" keys = every supplied field **except** `fitted_selection`, `execution`
(and the derived fields prediction_io fills).

## Frozen sources (D1)

`config/felipe_frozen_sources.csv` (T6 creates it after T3–T5 land): columns
`path,sha256,role`. Runners call `prediction_io.verify_frozen_sources(table, ROOT)` at start
and refuse to run on any mismatch. CLI flag `--frozen-sources` (default
`config/felipe_frozen_sources.csv`). **No bypass flag.** Tests build their own table in
`tmp_path`.

## Manifests (T6 edits; runners read)

- `config/felipe_basics_sequence_542_manifest.csv` gains a `label_space` column and
  four rows: `v1/all44`, `v2/all436`, `v1/shared73`, `v2/shared73`. Both V2 rows use
  `results/felipe_basics_2026-09-27/inputs/v2_all_436_group_splits.csv` (D7). V1 rows use the
  released V1 split.
- `config/felipe_basics_transfer_73_manifest.csv` gains `label_space=shared73`,
  `feature_key=paper_sequence_542`, and `label_map_path/label_map_sha256`
  (`config/felipe_basics_shared_73_diseases.csv`).

## Threshold policy (D2/D4 as implemented)

Paper-profile runs write raw scores only. They compute and write **no** thresholded metric.
All metrics come from `scripts/rescore_felipe_predictions.py`
(`fixed` → lane `primary_clean`, `threshold_source=constant_0.5`;
`paper_youden` → lane `paper_comparison`, `threshold_source=evaluated_fold_labels`).
Models are always called as `predict_proba(x)` with no labels.

## Global rules

No git add/commit/stash/checkout/clean/push. Do not delete or edit existing results. No full
model runs; probes only, written under `results/felipe_probes_2026-09-27/<task>/`. No network.
Never append labels to X, never align X/Y positionally without an ID assertion, never filter
disease columns. Edit only the files your task owns.
