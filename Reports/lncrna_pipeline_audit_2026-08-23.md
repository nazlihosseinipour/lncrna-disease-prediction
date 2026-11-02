# lncRNA–GIP Pipeline Audit — Root Cause Report
**Date:** 2026-08-23
**Process:** 5 parallel Sonnet code audits (one per pipeline stage, each with live shell access to the actual code on your Mac) → 1 Fable synthesis pass reviewing all 5 reports against your original diagnosis.

---

## Headline diagnosis

Your low AUROC is mostly **real, not a bug**: the paper's 0.814 is driven by association-derived leakage features that your guarded pipeline correctly excludes, and honest sequence-only signal sits around 0.63–0.67. Your low F1 is an **evaluation artifact** — a hardcoded 0.5 cutoff applied to an uncalibrated regressor. Your **inconsistency across reruns is fully deterministic** — not folds, not randomness (both refuted below) — caused by which of ~6 entry-point scripts you ran and which of 5 different data trees the default manifest happened to point at.

There is no single dramatic bug. There is one legitimate scientific finding (the paper leaks), one evaluation artifact (thresholding), and a serious engineering-hygiene problem (fragmented defaults, fragmented metric code, and your critical fixes currently living only in an uncommitted working tree).

**Note on timing:** during Audit 4, the codebase was observed changing live — uncommitted edits were appearing in `evaluate.py`, `run_inductive_within_cv.py`, and the model files while the audit was running, including what looks like an in-progress leakage-free threshold fix. If that's you or a Codex session actively working the same files, some items below may already be mid-fix — re-check before acting.

---

## Ranked root causes

**RC1 — The paper's benchmark rests on target-leakage features. The gap to the paper is expected, not an error.**
The paper's 285-column binary block is a near-exact copy of the raw disease-association matrix; that block *alone* scores AUROC 0.814 — higher than the paper's full 827-feature model. Your production pipeline has a working, empirically-verified three-layer guard that excludes this. *Status: not a bug — the guard works. One follow-up open (see below).*

**RC2 — Evaluation-protocol fragmentation.** The Youden-on-test-labels threshold is the *default* of the shared `Evaluator` class and of several run scripts; defaults disagree across ~6 entry points; 4 separate, independently-coded metric implementations exist. Concrete traced consequence: `build_generalization_matrix.py` mixes youden-scored and fixed-scored numbers in one table, so the "generalization drop" it shows is partly a measurement artifact. *Status: partially mid-fix (a leakage-free calibration mode is being built live); the leaky defaults themselves are unaddressed.*

**RC3 — Data-lineage fragmentation.** Five different `inductive_inputs/*` trees exist with different sample counts for "the same" V2 dataset (4,114 / 3,788 / 3,849). Run scripts default to the oldest, non-deduplicated one. You've been manually passing the corrected manifest — nothing enforces it. *Status: open.*

**RC4 — Your fixes exist only in an uncommitted working tree.** ~396 uncommitted lines in `prepare_inductive_feature_representations.py`; `rebuild_v2_transcript_sequences.py` and its test file are entirely untracked by git. A stash, checkout, or fresh clone destroys everything above. *Status: open — highest priority, see below.*

**RC5 — Uncalibrated RandomForestRegressor + hardcoded 0.5 cutoff.** Confirmed: raw `.predict()` output vs. a flat 0.5 threshold, no class weighting anywhere in the repo. RFLDA and IPCARF share the identical pattern — they are not a clean control group as you assumed. *Status: mid-fix (in-progress `train_oob_f1` calibration is the right shape of solution — finish it).*

**RC6 — Label-taxonomy problems.** Zero disease-synonym merging happens anywhere in the current label-matrix pivot (39 unmerged cancer-related columns survive in V1 alone — your "cancer+lymphoma" finding understated this). A live case-sensitivity bug splits real labels (`"Glioblastoma"` = 88 positives vs. `"glioblastoma"` = 1 positive, ≥7 terms affected). A purpose-built disease-ontology merge module already exists in the repo but is dead code w.r.t. the label matrix. *Status: open.*

**RC7 — Zero = "unknown," not "confirmed negative."** Uncontested, caps all metrics for you *and* the paper. Framing issue for the write-up, not an engineering bug.

**Explicitly refuted as causes:** fold construction/seeding (`MultilabelStratifiedKFold`, `random_state=0` everywhere — fully deterministic; this also directly answers "I did more folds and it still doesn't add up" — folds were never the source of variance). Duplicate handling itself (even the fully deduplicated V1 data scores AUROC 0.611 / F1 0.030 — correct hygiene, not the explanation).

---

## Your original diagnosis vs. what the audits found

**Held up essentially as stated:**
- The 285-column leakage finding — independently re-derived from raw numbers, not trusted from prose. Your strongest result; it survived hostile verification.
- All headline counts: 353×44/766 positives/4.93%; V2 9,907→7,806; 205 groups/281 rows dedup — reproduced exactly.
- The 0.5-threshold F1-collapse mechanism, and that test-fitted Youden is leaky and shouldn't be primary. Your own recommendation (fixed-0.5 for comparability + training-only calibrated threshold as the operating result) is exactly what the in-progress fix implements.
- "20 duplicate columns" — confirmed to be reverse-complement feature columns (mathematically expected), not sample rows.

**Confirmed but bigger than you thought:**
- Label merging: not one missed pair, but *no* synonym merging at all in the current pivot, plus a live case bug, plus a DOID module that was never wired in.
- Youden isn't just something you experimented with rescoring — it's the *default* of the shared evaluator and the main production runner.
- RFLDA and IPCARF are equally affected by the regression-to-threshold pattern — not a clean baseline.
- Your fixes only exist uncommitted; 5 different data trees are live as defaults.

**Refuted or corrected:**
- `analyze_residual.py` has nothing to do with model residuals — it's an ID-resolution data-quality script. The "zeros dominate the objective" theory itself is still plausible on first principles, but this citation for it is void.
- "583 vs. 2,023 features give identical results → count doesn't matter" overreaches: 583 is a strict *subset* of 2,023, so this only shows the extra MP-RNA embeddings add ~no marginal signal.
- Fold construction/randomness as a source of inconsistency — refuted.

---

## The single most important, cheap fix

**Commit the working tree to git right now, on a branch — including the untracked `rebuild_v2_transcript_sequences.py` and its test file — after coordinating with whoever else may be editing the repo live.** Minutes of work. Every "corrected" result you have, including today's `rerun_corrected_v1_rf_2026-08-23` numbers, was produced by code that exists nowhere except an uncommitted diff on one machine. Until that's committed, nothing else is reproducible or safe from being silently lost, and no one (including a future you, or Felipe) can verify any of it.

---

## Prioritized fix plan

1. **Commit everything** (RC4). New branch, include untracked files, tag the commit matching today's corrected V1 rerun. Check who else is editing the tree first.
2. **Flip the leaky defaults to safe** (RC2). `threshold_mode` default `"youden"` → `"fixed"` in the `Evaluator` class, `run_inductive_within_cv.py`, and `run_binary_comparison.py`; set the mode explicitly at the two bare `Evaluator()` call sites (`evaluate_inductive_experiments.py`, `parse_results/main.py`). A handful of lines; kills every silent leaky-scoring path.
3. **Kill the manifest ambiguity** (RC3). Make `--manifest` required with no default, or repoint all four run scripts to the corrected deduplicated manifest; archive/delete the stale `inductive_inputs/*` trees; record the manifest path + content hash in every `run_config.json`.
4. **Finish the in-progress `train_oob_f1` calibration** (RC5) instead of writing a new one. Then adopt: fixed-0.5 F1 for strict comparability, train-calibrated F1 as the operating result, test-Youden only ever as an explicitly labeled paper-reproduction column.
5. **Consolidate metrics into one implementation** (RC2). Route `plot_corrected_validation.py`, `run_v1_source_ablation.py`, and RFLDA's internal nested-CV scoring through the now-safe `Evaluator`. Then regenerate `build_generalization_matrix.py` under one protocol — don't cite the current version, it mixes threshold modes.
6. **Fix the label matrix** (RC6). Lowercase-normalize disease names before pivoting (fixes Glioblastoma/glioblastoma immediately); wire the existing DOID module into the actual Y-matrix build (groupby resolved term, OR-merge), replacing the ad hoc min-count filter; for paper comparisons use the paper's own precomputed 44-label matrix already sitting in the repo instead of your ad hoc 24-label subset. Re-run and see how much this moves the numbers — nobody has measured it yet.
7. **Close one open follow-up from Audit 2:** grep `run_inductive_transfer_experiments.py`, `rescore_saved_predictions.py`, and the RFLDA/IPCARF/LDA-SABC reproduction scripts for any place that might concatenate the persisted `Cross/*.csv` GIP/SVD output into X *outside* the guarded path. The leakage guard is verified only for the within-CV path — this is the one place a real feature-leak into *your own* model could still be hiding, given GIP features are computed and saved to disk even though excluded from the manifests you checked.
8. **Reframe the paper comparison in your write-up** (RC1, RC7). Report three rows: paper features under the paper's own protocol (0.814, leakage-inflated), paper features minus the 285-column block (~0.63), your clean pipeline (~0.63–0.67). That table *is* the finding. Add the positive-unlabeled caveat once, prominently, for both you and the paper.

---

## Open questions worth resolving next

- **Provenance of your historical numbers:** your *recent* run_configs already pass the safe flags, so RC2's leaky defaults may not explain your latest specific numbers — but nobody built a full table of manifest-path × threshold-mode for every past run. A quick sweep of every `results/*/run_config.json` would retroactively explain (or clear) each inconsistent historical result.
- **The unguarded GIP path** (item 7 above) — settle with a grep sweep, should take minutes.
- **Re-check `git status`/`git diff` before acting on items 2–5** — the audits' snapshots are already a few hours old given the live editing observed.
- No apparent tension between "use the paper's labels for comparability" (RC6) and "the paper's features are leaky" (RC1) — these are compatible: adopt the paper's Y, keep your guarded X. Keep the two decisions distinct in the write-up.
- Nobody has yet quantified how much RC6 (label splitting) moves AUROC/F1 on its own — that's the experiment in fix-plan item 6.

---

## Bottom line

Your instincts were substantially right, especially the leakage finding — it survived adversarial re-derivation and is publication-grade. The pipeline's real problem isn't a hidden bug turning good results into bad ones; it's that your honest results (~0.63–0.67 AUROC) were obscured by inconsistent defaults, fragmented metric code, and fixes that were never committed. Commit the code, unify the defaults, rebuild the comparison table under one protocol, and you have a clean, defensible story: the paper's number is leakage-inflated, and yours is the real baseline.

---

## Appendix: full per-domain audit reports

### Audit 1 — Label matrix + ID alignment (Sonnet)

**Verdict:** Real contributor — not via a silent alignment bug, but a fundamental label-taxonomy mismatch against the paper. The X/Y alignment code itself is well engineered (strict equality assertions, `validate="one_to_one"` merges).

Claim verdicts: all your headline counts (355×285/1,132 positives → 353/1,130 after sequence validation; 353×44/766 positives/4.93%; 353×24/613 positives/7.24%; V2 9,907→8,692→8,570→7,806 chain) were independently recomputed from the raw CSVs and matched exactly. The "cancer+lymphoma merge" issue was reclassified from a 3-positive rounding error to a systemic finding: **zero** disease-synonym merging happens anywhere in the current label-matrix pivot (`Data/processed/procces_website_alldata.ipynb`), leaving 39 distinct cancer/lymphoma-related columns unmerged in V1 alone, even though a full DOID-ontology merge module already exists in the repo (`mainfolder/utils/disease_mapping.py`, `rebuild_disease_mapping.py`) — it's just never wired into the Y-matrix build, only into a side review file and an X-feature.

New finding: a live case-sensitivity bug in V2's pivot (no `.lower()` before pivoting) silently splits real labels — `"Glioblastoma"` (88 positives) vs. `"glioblastoma"` (1 positive), at least 7 disease terms affected this way.

Suggested fixes: lowercase-normalize disease names before pivoting; wire the DOID module into the actual Y-build (groupby resolved term, OR-merge) instead of the ad hoc min-positive-count filter; use the paper's own precomputed 44-label matrix (already in the repo, currently unused by the pipeline) for paper comparisons.

### Audit 2 — Feature signal & leakage (Sonnet)

**Verdict:** Prior leakage finding CONFIRMED and independently re-derived from raw numbers. The paper's 285-column block is a near-exact/shifted copy of the raw disease-association matrix (164/285 columns bit-for-bit identical; most of the rest off by exactly the two shifted invalid-sequence rows). That block alone scores AUROC 0.814 — higher than the full 827-feature paper model, a textbook leakage signature.

The production pipeline (`prepare_inductive_feature_representations.py` → `run_inductive_within_cv.py`) has a real, wired-in three-layer leakage guard: a named-exclusion list (`svd_*`, `gip_*`, `lfs_from_Y`), wholesale exclusion of the `"cross"` and `"disease"` feature domains, and a structural `audit_target_contamination()` check that gates the file write. Verified empirically: none of the 9 manifests on disk contain cross/disease domains.

Standout finding: the project's own name ("lncrna-**gip**-project") is not a coincidence — GIP/SVD association-derived similarity features (computed directly from the label matrix, via `calculate_gip_lncRNA_and_dis()` in `cross_features.py`) genuinely exist and are computed and saved to disk, but as far as this auditor could verify, are excluded from the manifests that actually train the CV runner. Flagged as an open follow-up (not fully checked): whether `run_inductive_transfer_experiments.py`, `rescore_saved_predictions.py`, or the paper-baseline reproduction scripts (which *legitimately* use GIP as the paper's own method) might independently concatenate the persisted `Cross/*.csv` files into an X matrix outside the guarded path.

Also clarified: the "583 vs. 2,023 features give near-identical results" claim only shows the extra features (MP-RNA embeddings) add ~no marginal signal — 583 is a strict subset of 2,023, not an independent alternative, so this doesn't generalize to "feature count never matters."

### Audit 3 — Duplicate handling (Sonnet)

**Verdict:** Correctly implemented, consistently applied to V1 and V2, and confirmed NOT the cause of the low scores — even the fully deduplicated V1 data scores AUROC 0.611/AUPRC 0.128/F1 0.030 (from today's `rerun_corrected_v1_rf_2026-08-23/within_version_cv_summary.csv`). "20 duplicate columns" confirmed to be reverse-complement palindromic k-mer feature columns (mathematically expected to be identical), not sample rows. The 205-groups/281-rows V2 dedup number reproduced exactly; dedup happens once, globally, before fold splitting, on the correctly aligned ID space.

Two serious new findings: (1) the entire dedup/leakage-guard fix exists only as an uncommitted local git diff — `git status` shows ~396 uncommitted lines in `prepare_inductive_feature_representations.py` with none of the relevant functions in the last commit; `rebuild_v2_transcript_sequences.py` and its test are entirely untracked. (2) The actual run scripts (`run_inductive_within_cv.py`, `run_inductive_transfer_experiments.py`, `print_inductive_run_commands.py`, `evaluate_inductive_experiments.py`) all hardcode their default `--manifest` to an older, non-deduplicated tree. At least 5 different `inductive_inputs/*` directories exist on disk with different sample counts (4,114 / 3,788 / 3,849) for "the same" V2 dataset — called out as "a direct, concrete explanation for inconsistent results across reruns."

### Audit 4 — Fold construction + threshold protocol (Sonnet)

**Verdict:** Real, confirmed cause of internally-inconsistent numbers, with a nuance. The Youden threshold is genuinely fit on the test fold's own true labels (leakage in the evaluation protocol, mechanism traced exactly in `evaluate.py`), and this is the **default** of both the shared `Evaluator` class and the "self-contained production runner" `run_inductive_within_cv.py`. Two bare `Evaluator()` call sites inherit this leaky default silently. Nuance: your own recent `run_config.json` files (2026-08-17 through today) show you've been explicitly passing `--threshold-mode fixed` — so this specific default may not explain your very latest numbers, but it's an unfixed landmine.

Refuted: fold construction (`MultilabelStratifiedKFold`, genuinely stratified) and seeding (`random_state=0` everywhere — splitter, RF, RFLDA, IPCARF) are both solid; fully deterministic, not a source of run-to-run variance.

Best single explanation for "internally inconsistent" numbers found: threshold-mode default differs across ~6 entry points (`run_inductive_within_cv.py`→youden, `run_binary_comparison.py`→youden, `run_inductive_transfer_experiments.py`→fixed, `rescore_saved_predictions.py`→fixed, two bare `Evaluator()` calls→youden) — meaning identical saved predictions scored by different scripts produce non-comparable numbers with zero randomness involved.

Notable observation: the codebase was changing live during this audit — uncommitted edits appearing in real time to `evaluate.py`, `run_inductive_within_cv.py`, and the model files, including what looks like an in-progress, leakage-free `train_oob_f1` threshold-calibration mode already being test-run (`results/audit/rf_probe_v1_core583_oob_2026-08-24/`).

### Audit 5 — Model objective + metric definition consistency (Sonnet)

**Verdict:** Confirmed the RF-regressor claim exactly (unweighted `RandomForestRegressor`, zero `class_weight`/`RandomForestClassifier` usage anywhere in the repo). Refuted the assumption that RFLDA/IPCARF are a clean control group — both share the identical regression-to-threshold pattern (three separate `RandomForestRegressor` fits each).

Bigger finding, independently corroborating Audit 4: four separate, non-shared AUROC/AUPRC/F1 implementations exist in the repo (the canonical `Evaluator`; a fourth internal call site inside RFLDA's own nested-CV feature search; `plot_corrected_validation.py`, which bypasses `Evaluator` entirely with its own hardcoded 0.5 cutoff; `run_v1_source_ablation.py`, which fits a bare `RandomForestRegressor` directly, skipping even the production RF wrapper class). Traced a concrete downstream consequence: `build_generalization_matrix.py` pivots F1/accuracy/hamming numbers from a youden-default script and a fixed-default script into the same table, so the "generalization drop" it shows is partly a measurement artifact of mismatched protocols, not necessarily real.

Refuted: `analyze_residual.py`, cited by the prior analysis as corroborating "zeros dominate the regression objective," actually has nothing to do with model residuals — it's an ID-resolution data-quality script (checks which lncRNA IDs failed to match during harmonization). The prior analysis appears to have inferred its role from the filename alone.

---

*Generated by 5 parallel Sonnet audits with live shell access to the repository on your Mac, synthesized by a Fable review pass. One note on process: file staging to this session was blocked by a stale device sign-in ("untrusted_device") for this run, so the audits read your code via direct shell access instead — if you want faster file transfer in future sessions, re-signing in to the Claude desktop app should clear that.*
