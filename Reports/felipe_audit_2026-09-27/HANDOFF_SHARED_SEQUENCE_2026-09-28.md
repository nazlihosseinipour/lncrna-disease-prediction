# Handoff — shared-sequence paired comparison (for the next Opus supervisor chat)

Written 2026-09-28 ~10:00 CEST by the Opus session that implemented wave T1–T6 and ran the six
within-version cells. Read this file fully before acting. Section A is context and constraints
learned in that session; Section B is Nazli's plan, verbatim, which is the task.

---

## A. Context and constraints you must know

### A1. Where and how to work

- Local main checkout only: `/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project`
  (branch `pipeline-correction-2026-08-24`). Use absolute paths. Do not `cd` into
  `.claude/worktrees/...`; do not create worktrees.
- Python: `results/runtime/felipe_env/bin/python` (3.14, sklearn 1.7.2, pandas 2.3.3, numpy 2.3.5,
  iterstrat installed; `tabulate` is NOT installed — use `DataFrame.to_string`).
- Machine: 10-core Mac. Every model uses all cores; run heavy fits sequentially.
- No git add/commit/stash/checkout/clean/push. Never delete or edit existing results. Use a fresh
  output root, e.g. `results/felipe_shared_sequence_2026-09-28/`, and probes under
  `results/felipe_probes_2026-09-28/shared_sequence/`.
- Also read `Reports/felipe_audit_2026-09-27/HANDOFF_2026-09-28.md` (state of the completed work).

### A2. Completed work that must stay unchanged

- Verified full-population within results: `results/felipe_final_2026-09-28/` (60 cells),
  scores `results/felipe_final_2026-09-28/scored_within6/`, report
  `Reports/felipe_audit_2026-09-27/WITHIN6_VERIFIED_RESULTS_2026-09-28.md`,
  table `Reports/felipe_audit_2026-09-27/within6_verified_table.csv`.
- Key finding there: every model is below a training-prevalence-only baseline on micro AUROC and
  AUPRC; per-disease AUROC median 0.48–0.53. Expect the paired comparison to be close to chance as
  well, and report it honestly; do not treat "V2 higher" as success (gate G8).

### A3. Frozen sources — the most important operational rule

`config/felipe_frozen_sources.csv` pins 15 files (model code, `utils/*.py`, `evaluate.py`,
`scripts/prediction_io.py`, `scripts/felipe_paper_models.py`, `scripts/experiment_contract.py`,
both runners, `scripts/rescore_felipe_predictions.py`, and three input-provenance scripts).
**Do not edit any of them and do not edit that table**: the verified results' contracts record
those hashes. For this task:

- Write NEW scripts only (suggested names below). Import and reuse the frozen modules:
  - `scripts/reconstruct_paper_sequence_features.py::normalize_sequence` — the exact normalization
    the 542 features used (uppercase, `T→U`, remove every non-`ACGU` character; empty result =
    missing). Returns `(sequence, removed_counter)`.
  - `scripts/felipe_paper_models.py` — `build_paper_model`, `effective_params`, `fitted_selection`,
    `MODEL_SOURCES` (put `lncRNA_CIBCB2025-main` on `sys.path` first).
  - `scripts/prediction_io.py` — atomic cells, `read_prediction_cell`, `cell_status`,
    `quarantine_cell`, `verify_frozen_sources`, `library_versions`, `sha256_lines`.
  - The paper `Evaluator` in `lncRNA_CIBCB2025-main/parse_results/evaluate.py` for the 8 measures.
- Create a NEW frozen table for this task, e.g. `config/felipe_shared_sequence_frozen_sources.csv`,
  listing the 15 existing files (same hashes) plus the new scripts, after they pass review.
  New runners must refuse to run on any mismatch (same pattern as the existing runners).
- `scripts/rescore_felipe_predictions.py` only understands `kind` = `within`/`transfer`. It cannot
  score the paired 2×2 design and must not be edited. Write a new scorer.
- `scripts/verify_felipe_exact_results.py` is not frozen but is the verified checker for the
  existing results; prefer a new verifier for this task rather than modifying it.

### A4. Facts already measured (reproduce independently — do not trust)

A quick check in the previous session (normalize with `normalize_sequence`, SHA-256 of the
normalized string, raw files named in Phase 1) gave:

- 48 unique normalized-sequence hashes shared by V1 and V2;
- 48 V1 rows (one per hash; no V1 duplicates among them);
- 52 V2 rows; 4 shared hashes map to more than one V2 ID → strict one-to-one cohort = 44 hashes;
- only 1 of the 48 pairs has the same ID in both versions (`NR_024015`).

Consequence (flag this in the audit report): the existing
`scripts/run_felipe_strict_transfer.py` removes target overlap **by exact ID only**, so it would
leave 47 identical sequences in the transfer target. The earlier transfer design is therefore
leaky by sequence; this new paired design supersedes it. Do not run the old transfer commands.

Other input facts:

- V1 raw `sequences.csv`: 355 rows, columns `"", ID, seqs, <285 disease columns>`; 2 V1 sequences
  had ambiguity symbols removed (D1, K3, N1, Y3). V1 X built from it:
  `results/felipe_basics_2026-09-27/inputs/v1_paper_sequence_542_X.csv` (sha256 `4a2863…8604`).
- V2 raw `website_full_matrix.csv`: 5,338 rows, `ID, seqs, <436 disease columns>`; 1,208 without
  sequence (ledger `config/v2_missing_sequence_ledger.csv`). V2 X:
  `results/felipe_basics_2026-09-27/inputs/v2_paper_sequence_542_X.csv` (4,130 rows). V2 has 205
  internal identical-sequence groups (486 rows), identical to identical-X-row groups.
- Shared disease map: `config/felipe_basics_shared_73_diseases.csv` (columns `order, canonical_name,
  v1_source_columns, v2_source_columns, v1_positives, v2_positives`; source columns joined by
  `" | "`). It was built by `prepare_felipe_basics_inputs.normalized_label` (lowercase, non
  alphanumerics → space) on the raw V1 285 and V2 436 columns, OR-combining same-normalized
  columns. Existing 73-label Y files: `v1_shared_73_Y.csv` (355 rows), `v2_shared_73_Y.csv`
  (4,130 rows). Note V1's 44-label paper Y is a different (preprocessed) label set; the paired
  design should use the 73 shared concepts.
- A V1 and a V2 row with the same normalized sequence should have identical 542-feature vectors
  (both built by the same frozen code). Assert it rather than assume it; if the vectors differ,
  that is a blocker.

### A5. Runtime expectations (probe before launching)

From the within runs: V2 RF ≈ 5.6 min, IPCARF ≈ 8.5 min, RFLDA ≈ 43 min per fit on ~3,700 rows ×
436 labels. A single V2 fit on ~4,080 rows × 73 labels should be at or below that (fewer outputs;
RFLDA's inner search dominates). V1 fits on ~307 rows take seconds. So the primary cohort needs 6
fits total (3 models × 2 training versions): roughly 1–1.5 h. The sensitivity cohort uses the same
trained models if its training exclusions are identical (all 48 hashes are removed in both
cohorts) — if so, predict once and score twice rather than refit; record that decision.

### A6. Suggested task split for sub-agents (exclusive ownership)

| Task | Owns | Depends |
|---|---|---|
| S1 cohort + correspondence | `scripts/build_shared_sequence_cohort.py`, `test/test_shared_sequence_cohort.py`, outputs under the new results root, `Reports/.../shared_sequence_correspondence.csv` | — |
| S2 disease-map revalidation | `scripts/audit_shared_disease_map.py`, `test/test_shared_disease_map.py` | — |
| S3 paired runner | `scripts/run_felipe_shared_sequence_paired.py`, `test/test_shared_sequence_paired.py` | S1, S2 |
| S4 paired scorer (2×2, prevalence baseline, per-disease, bootstrap by hash) | `scripts/score_felipe_shared_sequence.py`, `test/test_score_shared_sequence.py` | S3 output format |
| S5 verifier + frozen table | `scripts/verify_felipe_shared_sequence.py`, `test/test_verify_shared_sequence.py`, `config/felipe_shared_sequence_frozen_sources.csv` | S1–S4 |

Opus reviews every diff, reruns the decisive tests, and independently recomputes metrics from
saved prediction files. Give sub-agents self-contained prompts (paths, API, rules above).

### A7. Design points to settle as supervisor (record as decisions)

- Test X: one row per shared hash, ordered by hash; take the vector from reconstruction and assert
  V1 and V2 vectors match. Both trained models predict this identical matrix (hash it).
- Training populations: V1 = V1 rows (355) minus all 48 shared hashes; V2 = V2 sequence-available
  rows (4,130) minus every row whose hash is among the 48 (52 rows). Assert zero hash overlap and
  zero exact 542-row overlap with the test matrix. Labels: the 73 shared concepts on both sides.
- Label versions for scoring: V1 labels for the hash (from its single V1 row); V2 labels = the
  single V2 row (primary) or OR over the V2 IDs of the hash (sensitivity), with conflicts listed.
- Metrics on a ~44 × 73 test matrix will be unstable and many diseases will be single-class:
  report counts of defined diseases, keep all 73 columns in prediction files, bootstrap by hash.
- Primary lane: fixed 0.5 threshold. Any test-label Youden number is descriptive only and kept
  separate. Prevalence baseline must use source-training prevalence, not test labels.

---

## B. The plan to implement (from Nazli, verbatim)

Continue the Felipe V1/V2 project in the LOCAL main checkout:

/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project

Objective:
Produce a controlled shared-sequence comparison for RF, IPCARF, and RFLDA. Use the same reconstructed 542 sequence features and common disease outputs. Preserve the completed and verified full-population V1→V1 and V2→V2 results.

Important:
- Work in the main checkout, not a worktree or cloud repository copy.
- Do not commit, push, delete, clean, or overwrite existing results.
- Use a fresh timestamped output directory.
- Opus should supervise, inspect agent work, make methodological decisions, review diffs, and verify results.
- Lower-level agents may handle independent audits or implementation tasks with exclusive file ownership.
- Run computationally heavy models sequentially because each model already uses the available cores.

Read first:

1. for claude/FELIPE_IMPLEMENTATION_AND_AUDIT_PLAN_2026-09-27.md
2. for claude/GATES.md
3. Reports/felipe_audit_2026-09-27/OPUS_DECISIONS_AND_SPEC.md
4. Reports/felipe_audit_2026-09-27/WAVE1_IMPLEMENTATION_ADDENDUM.md
5. Reports/felipe_audit_2026-09-27/WITHIN6_VERIFIED_RESULTS_2026-09-28.md
6. scripts/run_felipe_strict_transfer.py
7. scripts/verify_felipe_exact_results.py
8. scripts/rescore_felipe_predictions.py

PHASE 1 — Construct and audit the shared-sequence cohort

1. Load:
   - V1: lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv
   - V2: Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv

2. Normalize sequences using exactly the same normalization as the 542-feature reconstruction:
   - uppercase;
   - normalize U/T consistently;
   - remove only characters that the feature builder removes;
   - reject missing or invalid sequences.

3. Calculate a SHA-256 hash for every normalized sequence.

4. Construct the V1/V2 sequence-hash intersection.

5. Independently verify the currently observed result:
   - 48 unique sequence hashes shared between V1 and V2;
   - V1 has one row per shared hash;
   - identify all V2 hashes associated with multiple IDs.

6. Write an audited correspondence table containing:
   - sequence hash;
   - sequence length;
   - V1 ID;
   - all V2 IDs;
   - V1 and V2 feature-row identifiers;
   - one-to-one versus ambiguous status;
   - whether the 542 feature vectors agree with reconstruction;
   - label-conflict information for duplicate V2 IDs.

7. Do not trust an old fromserver mapping automatically. Inspect fromserver read-only and record whether any mapping there agrees with the newly computed hashes.

8. Create:
   - Primary cohort: strictly one-to-one shared sequence groups.
   - Sensitivity cohort: all 48 sequence groups, with a documented rule for duplicate V2 IDs.
   Prefer OR-combined V2 labels for the sensitivity cohort, but report label conflicts and do not silently treat duplicated IDs as independent RNAs.

Stop and report a blocker if the 48-sequence result cannot be independently reproduced.

PHASE 2 — Establish the disease/output contract

1. Use disease concepts present in both V1 and V2.
2. Begin from config/felipe_basics_shared_73_diseases.csv.
3. Revalidate every mapping against the raw V1 and V2 columns.
4. OR-combine spelling/case duplicates only according to the audited disease map.
5. Do not apply a minimum-positive disease filter.
6. Record diseases that cannot support AUROC or AUPRC in the small paired test cohort.
7. Keep all mapped diseases in prediction files, but calculate per-disease metrics only where mathematically defined.

Shared RNA sequences control the sample rows. Shared disease concepts control the prediction columns. Both intersections are required.

PHASE 3 — Prevent sequence leakage

For every shared sequence hash, remove every row with that hash from both potential training populations before fitting any model.

- V1 training population: V1 rows excluding every shared sequence hash.
- V2 training population: V2 rows excluding every shared sequence hash.
- Test population: the same shared-sequence cohort for both trained models.

Assert:
- zero sequence-hash overlap between training and test;
- zero exact 542-feature-row overlap between training and test;
- source-only fitting for scaling, PCA, feature selection, thresholds, and models;
- target labels are never passed to fit, transform, predict_proba, or threshold selection.

PHASE 4 — Run a paired 2×2 evaluation matrix

For each of RF, IPCARF, and RFLDA:

A. Fit on the leakage-free V1 training population.
B. Fit on the leakage-free V2 training population.
C. Generate probabilities from both fitted models for exactly the same ordered shared-sequence test matrix.
D. Score both probability matrices against both label versions:

- train V1 → evaluate with V1 labels;
- train V1 → evaluate with V2 labels;
- train V2 → evaluate with V1 labels;
- train V2 → evaluate with V2 labels.

This 2×2 matrix separates the effect of the training version from the effect of the label version while holding the RNA sequences fixed.

Run the primary one-to-one cohort first. Run the all-48 sensitivity cohort afterward if its duplicate-label rule passes review.

PHASE 5 — Evaluation

Primary metrics:
- micro AUROC;
- micro AUPRC;
- fixed-threshold micro F1;
- Hamming loss;
- number of evaluated sequences;
- number of mapped diseases;
- number of positive associations;
- number of diseases with both classes.

Also calculate:
- label-prevalence-only baseline;
- per-disease AUROC/AUPRC where defined;
- median per-disease performance;
- bootstrap confidence intervals resampled by sequence hash.

Use fixed threshold 0.5 as the primary thresholded lane.

If source-only Youden thresholds are implemented, estimate them only from source-training data or source out-of-fold predictions. Never select thresholds using the paired target labels.

Do not use target-label Youden values as primary results. If computed solely to reproduce the paper's measure, label them explicitly as optimistic/descriptive and keep them separate.

PHASE 6 — Verification

Before reporting:

1. Save probabilities and labels with explicit sequence-hash and disease-column ordering.
2. Save input and output hashes.
3. Verify that both trained models used the identical test X matrix.
4. Independently recompute metrics from saved prediction files.
5. Assert zero source/test sequence overlap.
6. Assert prediction dimensions and ordering.
7. Verify resume behavior does not refit models or mix cohorts.
8. Compare the primary and sensitivity cohorts.
9. Explain undefined and unstable metrics caused by the small cohort.
10. Preserve the completed full-population V1→V1 and V2→V2 results unchanged.

PHASE 7 — Deliverables

Create under Reports/felipe_audit_2026-09-27/:

- SHARED_SEQUENCE_COHORT_AUDIT.md
- shared_sequence_correspondence.csv
- SHARED_SEQUENCE_TRANSFER_RESULTS.md
- shared_sequence_transfer_table.csv
- a machine-readable run manifest;
- a concise supervisor-ready summary.

The supervisor summary must distinguish:

1. Existing full-population within-version results:
   - V1→V1: 355 RNAs × 44 diseases.
   - V2→V2: 4,130 RNAs × 436 diseases.
   These are descriptive and not a controlled direct comparison.

2. New paired shared-sequence results:
   - same RNA-sequence cohort;
   - common disease concepts;
   - identical feature schema;
   - shared sequences excluded from training;
   - small-cohort uncertainty stated clearly.

Execution order:
- Complete the mapping and leakage tests first.
- Run one small end-to-end probe for each direction.
- Review outputs and hashes.
- Then launch the full three-model runs sequentially.
- Continue monitoring until verification and reports finish.
- Do not stop after implementation or probes unless a real gate fails.
