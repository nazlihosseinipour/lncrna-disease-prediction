# Provenance search: the 827 lncRNA features (sub-agent A, read-only)

Date: 2026-09-29. Repo HEAD 8111d81 (branch pipeline-correction-2026-08-24). No model was run. No existing file was modified. Web access was limited to Pse-in-One documentation, source code and papers. Descriptor definitions are in `pse_in_one_descriptor_reference.csv` (same folder).

Required wording for columns 462–746: *a 285-column binary block that numerically matches the raw association matrix; its provenance conflicts with the paper's description and remains unresolved.*

## 1. What the paper claims

- `tmp/pdfs/inductive_paper.txt:287-305` (Section IV-B) and `pdf/IEEE_CIBCB_2025___Livia-6.pdf` (same wording): "eight RNA feature vectors using the package Pse-in-One [46]". The list names seven distinguishable families: K-mers, Mismatch, DAC, a cross covariance also printed as "(DAC)", DACC, PC-PseDNC-General and SC-PseDNC-General. It says "In total, it sums to 827 features". It gives only one K-mer example ("if K = 2").
- The paper gives no Pse-in-One version, command, k, m, lag, λ, w, property selection, column order or per-family sizes.
- Ref [46] (`inductive_paper.txt:588-590`) is Liu, Wu, Chou et al., "Pse-in-One 2.0", Natural Science 9(4):67 (2017).
- Code link in the paper (footnote, `inductive_paper.txt:290`): https://github.com/fkenjinakano/lncRNA_CIBCB2025. **Not fetched**, because the web scope was limited to Pse-in-One documentation. The local copy is `lncRNA_CIBCB2025-main/`.
- `pdf/LabMeeting_Felipe_140325-1.pdf` (slide 26, text extracted with a stdlib zlib parse): "LncRNAs: K-mers with K = ?", "Total number of features = 827". The author's own slide leaves K unspecified and names no other family.

## 2. Searches performed

| # | Scope / command | Hits | Result |
|---|---|---|---|
| S1 | `grep -rIil -E "pse-?in-?one\|psedna\|pseinone\|psednc\|mismatch\|\bDACC\b\|features_preprocessed"` over the repo (excluding `.git`, `felipe_env`, `.claude/worktrees`) | 68 files | All hits are project-authored code or reports, the paper text, or data CSVs where the words appear in prose. None is an upstream extraction script, config or log (details in §3). |
| S2 | Same grep restricted to `lncRNA_CIBCB2025-main/` (non-CSV) for `pse\|kmer\|k-mer\|mismatch\|dacc\|827` | 14 files | Only the `concatenate_kmers_with_svd` flag and `preprocess/main.py` comments. No Pse-in-One call. |
| S3 | `lncRNA_CIBCB2025-main/.git` | absent | No upstream history is available locally. |
| S4 | `git log --all -i -S<term>` for pse-in-one, pseinone, psedna, Pse-in-One, repDNA, repRNA / psednc / mismatch / dacc / lambda / features_preprocessed | 0 / 15 / 5 / 5 / 21 / 3 commits | Commits for psednc, dacc, mismatch and lambda belong to the project's own feature module (`mainfolder/features/rna_features.py`, 2026). features_preprocessed commits are 8c6e6cd, 31967ba and 8111d81 (the project's own reproduction scripts). |
| S5 | `git grep -I -i -l -F <term> $(git rev-list --all)` over all 65 commits and 5 refs (main, pipeline-correction-2026-08-24, claude/lncrna-gip-report-b727f4, recovery-big-files, origin/main), with data files dropped | pse-in-one 0; pseinone 0; psedna 0; pc-psednc 0; sc-psednc 0; psednc 13 paths; mismatch 4; dacc 7; features_preprocessed 12; kmers_preprocessed 1; dirna 16 (all false positives on "dirname") | No historical file contains a Pse-in-One command or config. The "dacc" hit in the IPCARF PDF is a match inside compressed bytes. |
| S6 | `git log --all --diff-filter=D --name-only` filtered by `pse\|feature\|kmer\|dacc\|mismatch` | 1 (a `.pyc` test cache) | No deleted feature-extraction script. |
| S7 | `git log -- lncRNA_CIBCB2025-main/create_dataset/raw_data/*` | 1 commit (73b21ee, 2026-05-11, "intergating correct version") | `features.csv` and `sequences.csv` were added once and never changed. There is no earlier version. |
| S8 | `fromserver/` (read-only): file listing (3,441 files) plus grep for `pse-?in-?one\|features_preprocessed\|pc-psednc\|sc-psednc` | 0 | "827", "mismatch" and "dacc" hits are digits inside floats and SHA hashes (false positives). Only the project's own 2026 results are there. |
| S9 | Filename search for `*pse*`, `*dacc*`, `*mismatch*`, `*kmers_preprocessed*`, fasta | 29 | All are project-generated `psednc_matrix` artifacts, `config/pse_in_one_rna_dcc_properties.csv` (project export), and NONCODE/LNCipedia fasta files. |
| S10 | `pdf/*.pdf` text (stdlib zlib parse) for Pse/K-mer/DACC/827 | LabMeeting: 2 relevant; CIBCB draft: same text as the paper; IPCARF paper: none (it uses GIP features) | See §1. |
| S11 | Web: github.com/bioinformatics-hitsz/Pse-in-One (tree, NEWS.md, kmer.py, acc.py, pse.py, const.py, util.py, data/index_list.py via `gh api`) | read | Defines the 1.0.x descriptors (CSV). The latest commit is bd4f39e, labelled "1.0.3, 2015-08-21 dev". |
| S12 | Web: github.com/banshanren/Pse-in-One-2.0 (nac.py, acc.py, pse.py, const.py, data/index_list.py, README) | read | Defines the 2.0 descriptors, including Mismatch. HEAD is 62d31af (2017-03-31). |
| S13 | Web: NAR 2015 article page; scirp.org Pse-in-One 2.0 HTML; hitsz download page; WebSearch | NAR and scirp read; hitsz refused the connection | See §4. The "Pse-in-One-1.0.6.rar" label comes only from a search-result snippet (mybiosoftware.com). |

## 3. Every relevant local hit and what it shows

| Path / commit | What it shows |
|---|---|
| `lncRNA_CIBCB2025-main/create_dataset/raw_data/features.csv` (355×827, columns `0`–`826`) | The features arrive already extracted. The file has no names, no family boundaries and no generator. |
| `lncRNA_CIBCB2025-main/create_dataset/preprocess/main.py:8,33` | The upstream author labels the features input `# raw_data KMERS` and writes the output as `kmers_preprocessed.csv`. This is upstream naming only. It does not mention Pse-in-One or any parameter. Features are row-aligned to `sequences.csv` and not transformed. |
| `lncRNA_CIBCB2025-main/create_dataset/README.md` | The run order starts at `preprocess/main.py`. There is no feature-extraction step. |
| `lncRNA_CIBCB2025-main/utils/transductive_converter.py:18-27` | Upstream comments separate "kmers" from "rna_features". This is context only and does not describe how the 827 columns were built. |
| `lncRNA_CIBCB2025-main/requirements.txt` | No Pse-in-One, repDNA or repRNA package is listed. |
| `THIRD_PARTY_CODE.md:33-39` (uncommitted edit) | Project note: the 22 RNA property vectors were exported from `Pse-in-One/data/dirna.data` at upstream commit bd4f39e. It calls this "Pse-in-One 1.0.6", but that commit's NEWS.md says **1.0.3**. The version label is therefore unverified (see §6). |
| `config/pse_in_one_rna_dcc_properties.csv` (untracked) | Project-exported 16×22 property table. |
| `scripts/reconstruct_paper_sequence_features.py` (untracked) | A project reconstruction, not upstream provenance. Its docstring says the 285 block was "copied from the V1 lncRNA-disease association matrix". That is a prior local interpretation, not evidence of how the block was generated. It also says the paper artifact "skipped several sequences containing ambiguous symbols", with no upstream source cited. |
| `scripts/audit_827_feature_reconstruction.py` (untracked; sibling task in this run) | Tests the families as negative controls. Its results were not used here. |
| `.unlazy/felipe-basics-audit/agent-reports/paper-protocol.md:28,36,126`; `SYNTHESIS.md:12,50,63` | Earlier audits reached the same conclusion: no command, parameters or boundaries were supplied, and the configuration must be requested from the authors. |
| `Reports/V1_V2_vs_paper_one_page_2026-09-21.md:44-45` | Open questions to the supervisor: the script that built the file, and the Pse-in-One configuration. No answer is recorded in the repo. |
| `mainfolder/features/rna_features.py`, `pipelines/run_all_rna_features.py`, `inductive_inputs/**/psednc*` | The project's own 2026 re-implementations. Irrelevant to the paper's provenance. |

## 4. Pse-in-One documentation findings (from primary source code)

- **1.0.x** (GitHub bd4f39e, "1.0.3"). The RNA modes are exactly `['Kmer', 'PC-PseDNC-General', 'SC-PseDNC-General', 'DAC', 'DCC', 'DACC']` (`const.py`). **Mismatch is not an RNA mode in 1.0.x.** The NAR 2015 paper also lists six RNA modes and "22 built-in properties".
- **2.0** (banshanren mirror). The RNA modes are `['Kmer', 'Mismatch', 'Subsequence', 'PC-PseDNC-General', 'SC-PseDNC-General', 'DAC', 'DCC', 'DACC', 'MAC', 'GAC', 'NMBAC']` (`const.py:18`). The paper's list, including Mismatch, therefore requires 2.0 (consistent with ref [46]).
- **Defaults.** kmer.py (1.0.3) `-k default=2`. In 2.0, `nac.py -k` has no default and `-m default=1`. acc.py `-lag default=2` in both. pse.py `-lamada default=2` and `-w default=0.1` in both. The default RNA properties are 6 (Rise, Roll, Shift, Slide, Tilt, Twist). `-all_index` selects all 22 (`dirna_list`, identical in 1.0.3 and 2.0).
- **Dimensions** (read from the loop structure):
  - Kmer = 4^k
  - Mismatch = 4^k
  - DAC = N·lag
  - DCC = N(N−1)·lag
  - DACC = N²·lag (DAC then DCC)
  - PC-PseDNC-General = 16+λ
  - SC-PseDNC-General = 16+λ·N
- **Normalization.** Kmer returns frequencies (count/total, 8 decimals). The 2.0 Mismatch returns raw counts. ACC uses the full length L when averaging and L−lag−1 when dividing. **The DAC/DCC arithmetic is identical in 1.0.3 and 2.0, so matching DCC values cannot tell the two versions apart.**
- **Ambiguous nucleotides.** `util.is_under_alphabet` rejects any character outside ACGU (including T) and calls `sys.exit`. Pse-in-One has no documented way to handle ambiguous bases. Raw `sequences.csv` uses T, and 2 of 355 rows contain N, K, Y or D (1, 3, 3 and 1 occurrences). How the authors converted or cleaned these sequences is undocumented.

## 5. Arithmetic test (no fitting)

**Documented defaults.** With N=6, lag=2, λ=2, Kmer k=2 (the 1.0.3 default), and Mismatch k undocumented:

- Kmer 16 + DAC 12 + DCC 60 + DACC 72 + PC 18 + SC 28 = **206**.
- Adding Mismatch 4^k gives 222, 270, 462 or 1230 for k = 2, 3, 4 or 5.
- **No documented-default combination equals 827.**

**All 22 properties at default lag 2.** DCC alone is 924, which is already more than 827. This setting is impossible.

**Observed released structure: 462 + 285 + 80.**

- 462 = N(N−1)·lag with N=22, lag=1. That is DCC with non-default parameters, not the default lag of 2.
- 80 = 16 + 64. That is two Kmer runs (k=2 and k=3). The paper lists K-mers once.
- The 285 block has no Pse-in-One dimension formula that yields a binary matrix. Kmer, DAC, DCC, DACC, PC/SC and Mismatch all produce real-valued outputs; Mismatch produces counts.
- The paper lists both DCC and DACC. If they shared parameters, DACC's DCC half would duplicate the DCC columns. Direct measurement: all 90 duplicate column vectors in the 827 matrix lie inside 462–746. There are none in 0–461 or 747–826, and none across blocks.
- No contiguous DAC, DACC, PC-PseDNC-General, SC-PseDNC-General or Mismatch block is visible in the released block structure.

**Conclusion.** The observed structure is **inconsistent** with the paper's family list. It holds 2 of the 7 named families (DCC, and K-mers as two k values) plus a block that is not a Pse-in-One output type.

Many parameter combinations of the seven families could sum to 827. They are **arithmetically possible, undocumented**. They are not listed as candidates, and none is inferred.

## 6. Per-family verdict

| Descriptor | Parameters found? | Source | Status |
|---|---|---|---|
| K-mers | No. The paper and the author's slide say only "e.g. K=2" and "K = ?". Released cols 747–826 are consistent with k=2 plus k=3 frequencies (prior audits) | paper IV-B; LabMeeting slide 26; Pse-in-One kmer.py | **undocumented** (the family is documented; k is not) |
| Mismatch | No. k and m are not given; 2.0 has m default 1 and no k default | paper; 2.0 nac.py | **undocumented**; absent from the released block structure (**contradicted** by the artifact) |
| DAC | No | paper; acc.py | **undocumented**; no DAC block in the released structure (**contradicted** by the artifact) |
| DCC ("DAC" typo) | No, for the paper. The released cols 0–461 match N=22, lag=1 (prior audits); that is not the Pse-in-One default (6 properties, lag 2) | paper; acc.py; prior audits | **undocumented**; artifact parameters inferred only by reproduction, not documented |
| DACC | No | paper; acc.py | **undocumented**; not present. It would create duplicate DCC columns, and none exist (**contradicted** by the artifact) |
| PC-PseDNC-General | No (λ, w, properties) | paper; pse.py | **undocumented**; not present (**contradicted** by the artifact) |
| SC-PseDNC-General | No (λ, w, properties) | paper; pse.py | **undocumented**; not present (**contradicted** by the artifact) |
| "Eighth" vector | None named | paper says eight, names seven | **undocumented** |
| Pse-in-One version | The paper cites 2.0. Local notes say "1.0.6", but the property source commit bd4f39e is labelled 1.0.3; "1.0.6.rar" appears only in a third-party search snippet | ref [46]; THIRD_PARTY_CODE.md; GitHub NEWS.md | **undocumented / internally inconsistent** |
| Ambiguous-nucleotide / T→U handling | None | Pse-in-One util.py exits on non-ACGU | **undocumented** |
| Feature ordering / boundaries | None (numeric column names) | features.csv header | **undocumented** |

## 7. Still unknown

- The exact Pse-in-One version, commands and parameters: k (Kmer, Mismatch), m, lag, λ, w, property subset, and output format.
- What the "eighth" feature vector is.
- Why only DCC (22 properties, lag 1) and 2-/3-mer frequencies appear in the released matrix, when the paper lists five other families.
- How T→U conversion and the 2 sequences with IUPAC codes were handled.
- Who generated `features.csv`, and when. Upstream history is not available locally; the local commit 73b21ee only imported the file.
- **Columns 462–746.** No evidence found in the repo, its git history, `fromserver/`, the upstream code copy, the slides or the Pse-in-One documentation explains how this block was generated. No Pse-in-One RNA descriptor produces a 285-column binary output. It remains *a 285-column binary block that numerically matches the raw association matrix; its provenance conflicts with the paper's description and remains unresolved.* The only local statements about its origin are the project's own interpretations in `scripts/reconstruct_paper_sequence_features.py:6` and `for claude/FELIPE_IMPLEMENTATION_AND_AUDIT_PLAN_2026-09-27.md:50`. Neither is primary evidence.

## 8. Not searched (out of scope) and suggested next steps

- The authors' public GitHub repo (paper footnote): not fetched, because of the scope limit. It is the most likely place for an extraction script and should be checked with the user's approval.
- The hitsz download page and the Pse-in-One 2.0 manual PDF (the connection was refused, or the file is in the mirror's `docs/`): not read. They may give the parameter ranges shown in the web UI.
- Asking the authors for the exact command and configuration remains the only way to settle the questions in §7.
