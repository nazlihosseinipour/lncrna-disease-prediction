# Agent A — V2 Population Audit (2026-09-27)

Scope: read-only investigation of why 1,208 of 5,338 raw V2 RNA rows lack a sequence,
whether any of them are locally recoverable, V1/V2 ID overlap, X/Y/split integrity for the
4,130-row population, and label-count impact of dropping the 1,208 rows.

All commands were run with:
`"$REPO/results/runtime/felipe_env/bin/python"` (env: `results/runtime/felipe_env`).

---

## Q1 — Provenance of the raw V2 table and "missing sequence" rule

- **Raw V2 source file:** `Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv`
  - SHA-256: `52c57c9896304d489c8a47dbb7f008ffc4d72c81b205365df671e45f687ca3c7`
  - Size: 25,921,794 bytes; 5,339 lines total = 1 header + **5,338 data rows** (matches the
    briefed count). Columns: `ID`, `seqs`, then 436 disease columns.
  - No duplicate `ID` values (`src['ID'].duplicated().sum() == 0`).
- **Consumer:** `scripts/prepare_felipe_basics_inputs.py` reads this exact file at line 92-93
  (`v2_source = pd.read_csv(ROOT / "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv")`)
  and uses it twice — once (after dropping the `seqs` column) to build the within-version
  label matrix `v2_all_436_Y.csv` aligned to whatever IDs already have sequence features
  (`v2_paper_sequence_542_X.csv`), and again to build the 73-shared-disease transfer labels.
  It never re-derives which rows have a sequence itself; that determination happens one
  script earlier, in `scripts/reconstruct_paper_sequence_features.py`.
- **"Missing sequence" rule** (`scripts/reconstruct_paper_sequence_features.py`,
  `normalize_sequence()` lines 42-53 and `reconstruct()` lines 132-148): for each row, the
  raw `seqs` cell is upper-cased, `T`→`U`, and every character outside `ACGU` is stripped.
  If the *resulting cleaned string is empty* (this happens for `NaN`/blank `seqs` cells, and
  would also happen for a pure-IUPAC-ambiguity-code string), the row is recorded in
  `missing_sequence_ids` and **excluded** from the 542-feature output — it never becomes a
  row in `v2_paper_sequence_542_X.csv`. This is a length-after-cleaning test, not a
  presence/absence test on the raw cell, but in practice every one of the 1,208 dropped rows
  has `seqs` = `NaN`/blank in the raw file. A separate 13 rows had ambiguity codes stripped
  but still kept ≥3 nt and were **not** dropped (`cleaned_sequence_count: 13`,
  `removed_symbols: {'K':1,'M':1,'N':16494,'R':5,'W':1,'Y':5}` — i.e. those stray
  `N`/`K`/`M`/`R`/`W`/`Y` characters were found inside otherwise-valid kept sequences, not
  inside the 1,208 dropped ones).
  - Metadata sidecar confirming all this:
    `results/felipe_basics_2026-09-27/inputs/v2_paper_sequence_542_X.csv.metadata.json`
    (`output_rows: 4130`, `missing_sequence_count: 1208`).
  - `input_summary.json`'s `v2_raw_rows_without_sequence: 1208` is computed independently in
    `prepare_felipe_basics_inputs.py` line 201 as `len(v2_source) - len(v2_x)` (5338 − 4130),
    which is consistent with, but not literally sourced from, the reconstruction script's own
    count — both agree at 1,208.

### ID types among the 1,208 missing-sequence rows

(classification by regex on `rna_id`; see ledger CSV for the per-row assignment)

| id_type | count |
|---|---|
| gene_symbol_or_other (plain HGNC-style symbols, no recognized prefix) | 1,085 |
| LOC_symbol (`LOC\d+`) | 64 |
| **NONCODE_rat** (`NONRATT\d+…`) | 18 |
| antisense_symbol (`*-AS\d*`) | 12 |
| Ensembl_ENST | 11 |
| clone_based_symbol (e.g. `AC010745.4`, `AL138720.1`) | 7 |
| Ensembl_ENSG | 6 |
| LINC_symbol (`LINC\d+`, not already resolved) | 2 |
| NONCODE_human (`NONHSAT\d+`, no version suffix) | 2 |
| **Ensembl_rat_ENSRNOT** | 1 |
| **Total** | **1,208** |

No RefSeq `NM_/NR_/XR_/XM_` accessions and no bare/plain NONCODE-with-version IDs appear
among the 1,208 — those ID families are apparently all successfully sequenced already inside
the 4,130 kept rows.

**Data-quality flag:** 19 of the 1,208 IDs are explicitly rat identifiers (18 `NONRATT…`,
1 `ENSRNOT…`) embedded in what is documented as a human V2 lncRNA-disease dataset. These are
not a sequence-fetch failure so much as a species-mismatch in the source ID list; recovering
a "sequence" for them would mean fetching a rat transcript into a human feature matrix, which
is out of scope for any of Q1-Q5's "no synthesis" instruction and should be surfaced to
whoever curated `website_full_matrix.csv`, not silently patched.

---

## Q2 — Do authoritative local sequences exist for any of the 1,208?

**No network calls, no downloads were performed. Only files already present in the repo were
inspected.**

Candidate local sources checked and their coverage of the 1,208 missing IDs:

| Candidate source | Provenance | ID matching | Coverage of the 1,208 |
|---|---|---|---|
| `Data/raw/sequence_id_overrides.csv` (1,831 rows; `query_id, resolved_id, sequence, source, notes`) | Output of a **prior network-fetch attempt** (`mainfolder/utils/sequence_fetch.py`, which calls Ensembl REST, NCBI E-utilities, RNAcentral, and LNCipedia over HTTP) | exact string match on `query_id` | 1,187 of the 1,208 IDs are present as `query_id` rows in this file, but **all 1,187 have an empty `sequence`/`resolved_id`** — i.e. the earlier network attempt already tried and failed for exactly these IDs. Of the 217 rows in this file that *do* have a resolved sequence (sources: `noncode_fasta` 115, `rnacentral` 58, `hgnc_ensembl_cdna` 18, `lncipedia_fasta` 12, `lncipedia_conversion_gene` 6, `ncbi_nucleotide` 5, `ensembl_name_cdna` 2, `lncipedia_conversion_transcript` 1), **0 overlap** with the still-missing 1,208 — those successes are presumably already folded into the 4,130 kept rows. 21 of the 1,208 IDs (mostly rat `NONRATT…` ones) don't even appear in this override log at all. |
| `Data/corrected_inputs/v2_transcript_sequences/transcript_sequences.checkpoint.csv` (2,619 rows, keyed by `transcript_id`, all `sequence_status == resolved`) | Intermediate cache from the same fetch pipeline, keyed by Ensembl transcript ID | exact match on `transcript_id` | **0 overlap** — this cache only holds transcript IDs that already resolved (again presumably already merged into the kept 4,130). |
| `Data/raw/lncipedia_5_2_hc.fasta` (155 MB, 107,039 records, header = `>SYMBOL:isoform_count`) | Bulk LNCipedia 5.2 high-confidence download | extracted symbol before `:`, matched case-sensitively and case-insensitively against the 1,208 | **0 matches**, either direction. |
| `Data/raw/NONCODEv6_human.fa` (293 MB, 173,112 records, header = `>NONHSAT#######.version`) | Bulk NONCODE v6 human transcript download | exact grep for the 2 bare `NONHSAT…` IDs among the 1,208 | **0 matches** (the 2 IDs have no version suffix and don't appear verbatim; this local NONCODE dump may be a different sub-release than whatever originally produced those 2 IDs). |
| `lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv` (V1 raw sequence+label table, 355 rows) | V1 project's own raw data | exact match on `ID` | **0 overlap** (expected — V1 and V2 share only `NR_024015`, see Q3). |
| `Data/output_data/website_sequences.csv`, `Data/corrected_inputs/v2_transcript_sequences/website_sequences.csv` | Both are copies of the same `ID,seqs` pairs as the raw source, **with the same 1,208 blanks** — not an independent source, just an extract. | n/a | Not usable — same blanks. |
| `Data/output_data/sequence_fetch_report.csv` | Per-ID fetch status log for the same 1,208-class of IDs (`status: unresolved`, plus the specific NCBI/Ensembl endpoints that returned empty) | n/a | Confirms failure, not a source of sequence. |

**Conclusion for Q2: no authoritative local sequence exists for any of the 1,208 rows.**
Every plausible local cache in the repo is either (a) an unfetched extract of the same raw
file, or (b) the output of an already-exhausted network-fetch attempt whose failures
correspond almost 1:1 to the still-missing 1,208 (1,187/1,208 explicitly logged as
unresolved; the remaining 21 — mostly rat IDs — were never even attempted). I did not
attempt any new fetch, per instructions.

---

## Q3 — V1/V2 ID normalization and duplicates

- **V1 within-version table:** `v1_paper_sequence_542_X.csv`, 355 rows, 0 duplicate IDs.
- **V2 within-version table:** `v2_paper_sequence_542_X.csv`, 4,130 rows, 0 duplicate IDs
  (`align_labels()` in `prepare_felipe_basics_inputs.py` line 34 would raise on any label-source
  duplicate `ID`, and the raw `website_full_matrix.csv` itself has 0 duplicate `ID` values).
- **Exact V1∩V2 overlap:** `{'NR_024015'}` — exactly one ID, confirming the brief.
- **Near-overlap checks performed** (none found beyond the one exact hit):
  - Stripping a trailing `.<version>` suffix and re-intersecting: still only `NR_024015`
    (0 additional pairs where only the version differs).
  - Case-insensitive intersection: still only `NR_024015` (0 case-only collisions).
- **Duplicate *sequences* within the raw V2 source** (distinct from duplicate IDs): among the
  4,130 rows that do have a sequence, 484 rows' sequences are byte-identical to at least one
  other row's sequence, forming 204 duplicate-sequence groups documented in
  `Data/corrected_inputs/v2_transcript_sequences/duplicate_sequence_groups.csv` (e.g.
  `17A|LncRNA-17A`, `AC004938.5|DDX56-208`, three-way group
  `AC084816.1-201|AC084816.1-205|AC084816.1-207`). **These groups are not collapsed/aggregated
  anywhere in the current pipeline** — `prepare_felipe_basics_inputs.py` keeps each ID as its
  own row in `v2_paper_sequence_542_X.csv`/`v2_all_436_Y.csv`, so a 542-feature X row derived
  from an identical sequence is duplicated verbatim under multiple IDs, each with its own
  (possibly different) disease-label row. This is a modeling-integrity concern (near-duplicate
  train/test leakage across the 10-fold split) but is reported here as a finding, not
  something I changed.

---

## Q4 — X/Y/split integrity for the V2 4,130 population

Verified by ID-based join, not positional assumption:

- `v2_paper_sequence_542_X.csv` and `v2_all_436_Y.csv`: both 4,130 rows;
  `x.index.equals(y.index) == True` (same IDs, same order) and
  `set(x.index) == set(y.index) == True` (same IDs regardless of order).
- `v2_all_436_splits.csv` (10-fold `MultilabelStratifiedKFold`, `write_splits()` in
  `prepare_felipe_basics_inputs.py` lines 41-48) stores **positional row indices** (0..4129)
  into `y`, one `train_fold*`/`test_fold*` column pair per fold. Concatenating all 10
  `test_fold*` columns yields exactly 4,130 entries with 4,130 unique values covering the
  full range `0..4129` — i.e. every row is a test-fold member in **exactly one** fold. No
  gaps, no double-coverage.

---

## Q5 — Effect of dropping the 1,208 rows on per-disease positive counts

Compared raw positive counts (`website_full_matrix.csv`, all 5,338 rows, 436 disease columns)
against the kept-4,130 counts (`v2_all_436_Y.csv`), joined by disease-column name (all 436
names match exactly between the two tables):

- **82 of 436 diseases (18.8%) lose at least one positive** when the 1,208 sequence-missing
  rows are dropped. Largest absolute drops:

  | Disease | raw positives | kept-4,130 positives | lost |
  |---|---|---|---|
  | Esophageal Squamous Cell Carcinoma | 1,260 | 852 | 408 |
  | Depression | 389 | 104 | 285 |
  | Atrial Fibrillation | 579 | 507 | 72 |
  | Stomach Neoplasms | 494 | 426 | 68 |
  | Carcinoma, Renal Cell | 150 | 120 | 30 |
  | Pterygium | 277 | 256 | 21 |
  | Diabetic Macroangiopathy | 20 | 1 | 19 |
  | acute asthma | 19 | 0 | 19 |
  | Adenocarcinoma of Lung | 197 | 180 | 17 |
  | Carcinoma, Hepatocellular | 430 | 413 | 17 |

  (full 82-row table is reproducible from the commands in this audit; not persisted as a
  separate file since it wasn't part of the required deliverables — happy to add if wanted.)

- **10 diseases drop to exactly 0 positives** in the kept-4,130 population despite having
  ≥1 positive in the raw 5,338-row table — i.e. these 436-label columns become
  **all-negative dead columns** once the 1,208 rows are removed:
  `Cardiac Fibrosis` (1→0), `Cryptosporidium Parvum Infection` (1→0),
  `Fusarium graminearum` (1→0), `Hearing Loss` (1→0), `Hepatitis, Alcoholic` (1→0),
  `Leukemia, Biphenotypic, Acute` (1→0), `Lupus Nephritis` (1→0),
  `Mammary Neoplasms, Animal` (2→0), `acute asthma` (19→0), `head and neck carcinoma` (1→0).

  **This is a modeling-relevant finding**: any classifier trained per-label on the 4,130
  population for these 10 diseases has zero positive examples to learn from or evaluate
  against (AUPRC/recall undefined), even though the 436-column label schema still nominally
  includes them.

---

## Recommendation

**Declare `v2_sequence_available_4130` as the analysis population; do not attempt recovery
of the 1,208 rows.**

Reasons:
1. Every locally available candidate source (bulk LNCipedia/NONCODE FASTA, override log,
   transcript-sequence checkpoint cache, alternate CSV extracts) either duplicates the same
   blanks or is the *output* of an already-run network-fetch attempt that failed for 1,187 of
   the 1,208 IDs and never even attempted the remaining 21 (18 of which are rat IDs that
   shouldn't be in a human dataset in the first place).
2. Recovering the rest would require new network calls (explicitly out of scope here) against
   IDs that a prior, more thorough automated fetch already could not resolve — largely
   free-text gene symbols/clone names/LOC IDs without a clean external database key, plus a
   genuine species mismatch for ~19 rows.
3. Given (1) and (2), silently imputing or zero-filling sequences for these rows (which I did
   not do) would not be scientifically defensible, and the honest label is "sequence
   permanently unavailable given current inputs," not "temporarily missing."

Caveats/residual risk for whoever owns this decision:
- The 10 diseases that go to 0 positives, and the 82 that lose positives, should be
  re-examined before final label-set selection (drop those 10 columns from the 436, or keep
  them and explicitly flag them as unlearnable/unevaluable for the 4,130-row population).
- The 204 duplicate-sequence groups (Q3) are a separate correctness issue independent of the
  1,208-row question and should be resolved (dedup or explicit leakage-aware fold
  construction) before trusting within-version V2 cross-validation numbers.
- The 19 rat-ID rows are a likely upstream data-curation bug in
  `Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv` and are worth a
  targeted follow-up with whoever built that file, independent of the sequence-recovery
  question.

---

## Files produced by this audit

- `Reports/felipe_audit_2026-09-27/agentA_v2_population_audit.md` (this file)
- `Reports/felipe_audit_2026-09-27/agentA_v2_missing_sequence_ledger_PROPOSED.csv`
  (1,208 rows, one per raw V2 row lacking a sequence; columns: `raw_row_index`, `rna_id`,
  `id_type`, `n_positive_labels`, `candidate_local_source`, `resolution_proposed`, `notes`).
  All 1,208 rows are proposed as `resolution_proposed = missing_sequence`
  (no `recoverable_from:<path>` rows were found — see Q2).
