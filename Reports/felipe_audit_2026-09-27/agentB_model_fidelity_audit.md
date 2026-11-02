# Agent B — Model Fidelity Audit (2026-09-27)

REPO: `/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project`
Scope: read-only review of uncommitted edits to IPCARF/RFLDA/runner + THIRD_PARTY_CODE.md.
No source files were modified by this audit.

## 1. Diff summary

| File | Hunk | What it changes | Alters paper-profile behavior? | Category |
|---|---|---|---|---|
| `IPCARF/ipcarf.py` | `+from sklearn.preprocessing import StandardScaler`, new `standardize`/`allow_pca_fallback` ctor args (L17-29) | Adds two opt-in flags; whitespace re-indent of ctor signature | No (defaults `standardize=True`, `allow_pca_fallback=True` preserve prior behavior) | New option |
| `IPCARF/ipcarf.py` | `_build_pipeline` helper (L126-131), used in `_optimize_n_components` (L71-94) and `_fit_pipeline_with_pca_fallback` (L105-124) | Inserts `StandardScaler` before PCA when `standardize=True`; gates the randomized-PCA fallback behind `allow_pca_fallback`, re-raising instead of silently swapping solvers when `False` | Yes, but only when `--model-profile paper` is selected (runner sets `standardize=False, allow_pca_fallback=False`) — paper lane runs unscaled IPCARF with no silent PCA solver substitution | Behavior change (gated, new option) |
| `RFLDA/rflda.py` | `+from sklearn.metrics import average_precision_score`; new `n_folds`, `selection_metric` ctor args (L21-39) with validation (`ValueError` on `<2` folds or invalid metric string) | Adds configurable inner-CV fold count and selection objective; class default `n_folds=10` (L15) is untouched, so `current` profile still runs a 10-fold inner search | Yes for `paper` profile (runner passes `n_folds=5, selection_metric="subset_accuracy"`) | New option, behavior change gated behind explicit args |
| `RFLDA/rflda.py` | `_find_best_n_features` / `_selection_score` (L86-145) | Replaces hardcoded 0.5-threshold + `accuracy_score` selection with a pluggable `_selection_score` (`micro_auprc` via `average_precision_score`, or `subset_accuracy` reproducing the old thresholded-accuracy logic) | Default (`micro_auprc`) changes the **current**-profile feature-count selection criterion from thresholded accuracy to micro-AUPRC — this is a real behavior change even outside the paper profile, since `selection_metric` defaults to `"micro_auprc"`, not the old accuracy behavior | **Bug risk / behavior change**, not purely additive |
| `RFLDA/rflda.py` | `_find_intervals` (L146-153) | Appends the exact feature count as a final interval when it isn't already a multiple of 50 (e.g. 237 features → `[50,100,150,200,237]` instead of stopping at 200) | Yes — changes candidate feature counts for any dataset whose width isn't a multiple of 50 | Bug fix (paper says "steps of 50" but silently truncating the tail features looks unintended) / behavior change |
| `scripts/run_inductive_within_cv.py` | new `--model-profile {current,paper}` CLI flag (L104-111), `build_model_factory` branch (L209-231) | Wires the two model options above into a switch; `current` path is byte-for-byte the old factory call | No change to `current` lane for the IPCARF/RFLDA construction call itself | New option |
| `scripts/run_inductive_within_cv.py` | `experiment_contract` import + `validate_manifest_row`, `sha256_file`, `ensure_prediction_contract`, `write_run_configs` (L13-17, L122-163, L264-273, L296-317) | Adds SHA-256 provenance hashing of x/y/split files and model source, writes an immutable `run_configs/<ts>_<digest>.json` in addition to `run_config.json`, and refuses to reuse a `predictions/<key>/<model>/` directory whose `prediction_contract.json` doesn't match (protects against silently mixing runs from different code/data) | No effect on model math; purely a run-integrity/provenance guard | New option / hardening (matches "strict transfer protocol" commit message) |
| `scripts/run_inductive_within_cv.py` | `experiment_key` gains a `_{model_profile}` suffix when profile != `current` (L294-295), summary key/sort now includes `model_profile` (L496-504) | Prevents `paper` and `current` runs from overwriting each other's predictions/performance CSVs in the same `--outdir` | No | Bug fix / hardening |
| `THIRD_PARTY_CODE.md` | New "Pse-in-One RNA property values" section inserted at L29-36 | Documents provenance of `config/pse_in_one_rna_dcc_properties.csv` (GPL-3.0, upstream commit `bd4f39e...`) | N/A (docs only) | **Formatting bug**: the new section is spliced into the middle of the existing RFLDA bullet list — `- \`RFLDA/rflda.py\`` is immediately followed by the new `##` heading, and the two bullets that actually describe RFLDA's behavior ("supports feature sets with fewer than 50 columns", "guarantees max_features ≥ 1") now render as if they belong to the new section, not to RFLDA. Content is accurate but placement needs fixing. |

## 2. Paper profile: HEAD vs working tree

Working-tree edits are **not** behaviorally identical to HEAD when the paper profile is selected — the paper profile only exists in the working tree (the `standardize`/`allow_pca_fallback`/`n_folds`/`selection_metric` parameters and the `--model-profile` flag do not exist at HEAD at all). At HEAD, `IPCARF` always standardizes-then-PCA-falls-back-silently, and `RFLDA` always runs a 10-fold inner search scored by thresholded accuracy — there is no way to reproduce the paper's documented 5-fold/`{2,4,...,128}`/no-scaler setup on HEAD. All paper-profile behavior is new, uncommitted code.

Within the **working tree itself**, `current` profile ≈ HEAD behavior for IPCARF (defaults preserve old flow) but **not** for RFLDA, because RFLDA's default `selection_metric` changed from implicit thresholded-accuracy to `"micro_auprc"` even under `current`/HEAD-equivalent construction (no explicit `selection_metric` is ever passed by the `current` factory call in the runner). This is the one place where an "additive" refactor silently changes non-paper-profile numeric results too.

SHA-256 hashes (also in `agentB_source_hashes.csv`):

| path | HEAD blob | working tree | identical |
|---|---|---|---|
| `IPCARF/ipcarf.py` | `ce5409a8...b8a9fc29`→`ce5409a8bc42751e3e96533a1bdec77461aee56e97579217300e268edb4b230b` | `13c00e59...8b8a9fc29` | false |
| `RFLDA/rflda.py` | `597189e1...2c0236426` | `2bc43722...77d6cbc2` | false |
| `scripts/run_inductive_within_cv.py` | `a40c0dd7...98eacd0987` | `c50ecc90...778fb8ac697` | false |

## 3. Paper-profile parameter table

| Param | Paper spec | Effective (`--model-profile paper`) | File:line | Match? |
|---|---|---|---|---|
| RF n_estimators | 150 | 150 (`RF.n_estimators`, `IPCARF.n_estimators`, `RFLDA.n_estimators` all class-level `= 150`) | `RF/rf.py:18`, `IPCARF/ipcarf.py:11`, `RFLDA/rflda.py:11` | Match |
| random_state | fixed | `0` everywhere | `RF/rf.py:19`, `ipcarf.py:12`, `rflda.py:13` | Match |
| max_features | — (paper: default/`sqrt`-style) | RF: sklearn `RandomForestRegressor` default (`1.0`, i.e. all features, since sklearn ≥1.1) — **not overridden**; RFLDA: `max(1, n_features // 3)` (`_rf_max_features`, `rflda.py:155-157`) | `RF/rf.py:43-48`; `rflda.py:66,99,164` | RF: **mismatch/unspecified** — plain RF never sets `max_features`, so it silently uses sklearn's regressor default, not the paper's presumed `sqrt`/`log2` fraction. Worth flagging even though it's outside the diff (pre-existing). |
| class_weight | n/a (regressor, not classifier) | n/a — all three use `RandomForestRegressor`, not `RandomForestClassifier`, so `class_weight` doesn't apply; class imbalance is handled only via threshold selection | `rf.py:43`, `ipcarf.py:49/70`, `rflda.py:64/97/162` | Consistent with paper's regression-then-threshold design |
| Standardization/scaler | (paper profile ⇒ unscaled per runner's own doc-comment) | `standardize=False` under paper profile ⇒ no `StandardScaler` step | `run_inductive_within_cv.py:222-227` (factory), `ipcarf.py:126-131` (`_build_pipeline`) | Match (by construction/definition — there's no independent paper-text confirmation in repo, but it's internally consistent with the runner's own doc-string) |
| PCA component grid | `{2,4,8,16,32,64,128}` | Same list, `IPCARF.n_components_optimize` (class attr, profile-independent) | `ipcarf.py:15` | Match |
| PCA fallback | not in paper (paper uses IncrementalPCA only) | `allow_pca_fallback=False` under paper profile ⇒ exceptions from `IncrementalPCA` propagate instead of silently swapping to randomized `PCA` | `run_inductive_within_cv.py:222-227`; `ipcarf.py:87-88,113-114` | Match to stated intent |
| Inner CV scheme (IPCARF component search) | five-fold | `GridSearchCV(..., cv=self.n_folds)` with `IPCARF.n_folds = 5` (class attr, **not** parameterized by the new ctor args — always 5 regardless of profile) | `ipcarf.py:13,79,98` | Match |
| Inner CV metric (IPCARF) | not specified beyond component search | `GridSearchCV` default scorer = regressor `.score()` (R²) since no `scoring=` is passed | `ipcarf.py:76-82,95-101` | **Unverified/likely mismatch** — paper likely optimizes a classification metric (AUPRC/accuracy on binarized output), not R²; this predates the diff and is not touched by it, but it undermines "IPCARF inner five-fold component search" fidelity |
| RFLDA inner CV / step | five-fold, steps of 50 | `n_folds=5` (paper profile only; class default is 10) via `MultilabelStratifiedKFold`/`StratifiedKFold`; `_find_intervals` steps of 50, now also appending the exact final width | `run_inductive_within_cv.py:213-218`; `rflda.py:32-34,46-48,146-153` | Match under paper profile; `current` profile still 10-fold |
| RFLDA selection metric | paper: (implied) subset/thresholded accuracy | `selection_metric="subset_accuracy"` under paper profile; **default is `"micro_auprc"` for both `current` and any caller that doesn't pass the arg explicitly** | `run_inductive_within_cv.py:213-218`; `rflda.py:29,141-145` | Paper profile: match. Default/`current`: silently changed from the old implicit accuracy-threshold logic — flag as unintended scope creep in a supposedly non-behavior-changing refactor |
| n_jobs | not specified | `-1` (all cores) uniformly, profile-independent | `rf.py:17`, `ipcarf.py:10`, `rflda.py:9` | Consistent |
| Outer CV / ten outer folds | 10 | Controlled by `--n-splits` CLI arg to the runner (default not paper-specific), independent of `--model-profile`; must be passed as `--n-splits 10` by the operator | `run_inductive_within_cv.py` (`parse_args`, not itself part of the reviewed diff) | Operator-dependent, not enforced by `--model-profile paper` |
| Per-disease Youden threshold (paper lane only) | expected coupled to paper profile | **Not coupled** — `--threshold-mode` is a fully independent CLI flag (`choices=["youden","fixed","train_oob_f1"]`, default `"fixed"`); selecting `--model-profile paper` does **not** auto-select `--threshold-mode youden` | `run_inductive_within_cv.py:100-101` vs `104-113` | **Mismatch / operator risk**: running `--model-profile paper` alone silently evaluates with the fixed 0.5 threshold, not Youden. Must remember `--threshold-mode youden` explicitly. |

## 4. One-class labels / predict_proba shape

All three wrappers use `RandomForestRegressor` (not a classifier), so a constant-label training column never raises a scikit-learn "one class" error — the regressor just fits to a constant target and predicts a near-constant value. Verified in the synthetic probe (Section 5): a fold with an all-zero label column runs to completion for RF, IPCARF (current + paper flavor), and RFLDA (current + paper flavor) with no exception. Downstream, `evaluate.py`'s `sensivity_specifity_cutoff` (`parse_results/evaluate.py:187-189`) explicitly guards `y_true[column].nunique() < 2` and falls back to a fixed 0.5 threshold instead of computing an undefined ROC/Youden curve — this guard is unaffected by the reviewed diff.

`predict_proba` in all three wrappers returns the raw `RandomForestRegressor.predict()` output — a continuous score in roughly `[0,1]` for a 0/1-encoded target, not a calibrated probability — with shape `(n_samples, n_labels)` confirmed by the probe below (label dimension preserved; single unflattening step only triggers in transductive/binary_mode, not exercised here).

## 5. Test / probe results

```
$ results/runtime/felipe_env/bin/python -m pytest -q test/test_model_edge_cases.py
....                                                                     [100%]
4 passed, 1 warning in 0.96s   (total wall incl. interpreter startup ~1.2s)
```
(One `DataConversionWarning` from sklearn about a column-vector `y` in `test_ipcarf_scales_features_before_pca` — pre-existing, harmless, not part of the reviewed diff.)

Synthetic probe (40 samples × 20 features × 3 labels, label #3 forced constant/all-zero in the training split), `Reports/felipe_audit_2026-09-27/agentB_probe_models.py`:

```
RF: predict_proba shape=(10, 3) expected=(10, 3)  time=0.07s
IPCARF: predict_proba shape=(10, 3) expected=(10, 3)  time=0.06s
IPCARF_paper: predict_proba shape=(10, 3) expected=(10, 3)  time=0.06s
RFLDA: predict_proba shape=(10, 3) expected=(10, 3)  time=0.90s
RFLDA_paper: predict_proba shape=(10, 3) expected=(10, 3)  time=0.51s
OK - no exceptions on constant-label training fold
```
Total probe wall time ≈2.3s. No repo files were modified to run this; the script only imports `RF`/`IPCARF`/`RFLDA` from `lncRNA_CIBCB2025-main`.

## 6. Recommendation

- **Freeze source**: the working tree is the only place the documented paper profile exists at all — HEAD cannot reproduce the paper's spec (no scaler toggle, no 5-fold/subset-accuracy RFLDA search, no PCA-fallback gate). The dirty edits should be committed, not reverted, once the two issues below are addressed — reverting to HEAD would be a regression relative to the stated goal.
- **Must fix before trusting `current`-profile numbers against pre-diff baselines**: `RFLDA`'s default `selection_metric` silently switched from thresholded-accuracy to `micro_auprc` for every caller that doesn't pass the new argument explicitly, including the `current` factory branch in the runner. This is a real behavior change hiding inside what reads like an additive refactor — either make `current` explicitly pass the old-equivalent `selection_metric="subset_accuracy"` to preserve backward comparability, or explicitly document/accept the metric change and re-baseline `current`-profile results.
- **Must fix / gate**: `--model-profile paper` does not imply `--threshold-mode youden`; an operator who runs only `--model-profile paper` silently gets fixed-0.5 thresholding, not the paper's per-disease Youden cut. Recommend either erroring when `model_profile=="paper"` and `threshold_mode!="youden"`, or defaulting `threshold_mode` to `"youden"` whenever `model_profile=="paper"` unless explicitly overridden.
- **Should fix**: `THIRD_PARTY_CODE.md`'s new Pse-in-One section is spliced into the middle of the RFLDA bullet list, orphaning two RFLDA-specific bullets under an unrelated heading — move the new section to the end of the file or before the `## ` list it currently interrupts.
- **Worth a follow-up, not blocking**: plain `RF` never sets `max_features` (uses sklearn's regressor default of using all features), and IPCARF's inner `GridSearchCV` component search uses the regressor's default R² scorer rather than a classification-relevant metric — both predate this diff but reduce confidence that "IPCARF inner five-fold component search over {2,4,...,128}" is truly selecting components the way the paper intends.
- No git operations, test edits, or model-code edits were performed; only new files were written under `Reports/felipe_audit_2026-09-27/`.
