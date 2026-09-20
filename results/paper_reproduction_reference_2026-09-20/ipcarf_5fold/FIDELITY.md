# FIDELITY: 5-outer-fold variant vs the 10-fold reference run

Leakage-inflated reference (not a valid inductive result). `features_preprocessed.csv` carries a 285-column block that copies the association matrix. The paper describes 10 outer folds; this 5-outer-fold run is an alternative variant. Everything in `../ipcarf/FIDELITY.md` (step-by-step match to the authors' pipeline, hyperparameter table, library-version caveats) applies here unchanged except the items below.

Working copy of the authors' code: `scratchpad/ipcarf_5fold/authors_copy`, a `cp -R` of the 10-fold run's copy (identical files: `diff -rq` showed none). Nothing under the project tree outside this folder was written. Exact files used are in `code_used/` (identical to the 10-fold run's).

## Differences from the 10-fold run

| Item | 10-fold run | 5-fold run |
|---|---|---|
| Outer folds | 10 (`inductive_splits.csv` shipped by the authors, 319 train / 35-36 test) | 5 (`inductive_splits_5fold.csv`, 284 train / 71 test each) |
| Split generation | authors' shipped file | `make_splits_5fold.py`: authors' `split_inductive.py` logic (`MultilabelStratifiedKFold(n_splits, shuffle=True, random_state=0)`, split on `mesh_labels_final.csv`, same CSV layout, float row indices, NaN padding) with n_splits=5. Written only into this folder. |
| Fold count in authors' code | `n_folds = 10`, `iterator_cross_validation(..., n_folds=10)`, `Evaluator.columns_to_add_df` fold1..fold10 | Patched at run time by the drivers, without editing any authors' file: `iterator_cross_validation.__defaults__` 10 -> 5 (asserted first), `n_folds = 10` -> 5 in the `parse_results/main.py` source text (asserted), `columns_to_add_df` = fold1..fold5 + mean(std). |
| Drivers | `run_ipcarf_authors.py`, `run_all.sh`, `run_eval_authors.sh`, `build_table.py`, `sanity_checks.py` | Same scripts, parameterized: splits path, 5 folds, output dirs. New: `make_splits_5fold.py`, `run_eval_authors.py` (the exec wrapper that applies the patches above), `compare_table.py`. |
| Inner CV, trees, grid, PCA scaling, seed, threshold rule, HMC 44-disease evaluation | 5-fold inner, 150, {2..128}, none, 0, per-label Youden on the test fold, yes | identical (same code). Inner training folds have 227 rows, so all 7 grid values stay valid. Best components per outer fold: MLC 64/64/128/64/128, HMC 64/128/128/64/64. |
| Std | population std (np.std) over 10 folds | population std over 5 folds |
| Runtime | MLC 5 min, HMC 18 min | MLC 1 min 40 s, HMC 6 min 29 s (half the folds, run in the same 8-core environment) |
| Sanity checks | 48 | 37 (fold counts 5 instead of 10; added checks: 5 column pairs, test folds partition all 355 rows, splits file equals a fresh run of the generator, fold sizes 71) |

## Split provenance caveat (cannot be resolved)

Running `make_splits_5fold.py 10` (same generator, same settings, hierarchical label matrix from the repo) does not reproduce the authors' shipped `inductive_splits.csv`: 0 of 10 test folds match, and the partitions differ as sets. Tried without success: the shipped flat matrix, the 369-column hierarchical matrix, hierarchical minus flat columns, sorted and reversed column order, columns filtered by positive count (>0,1,2,3,5,10), `.values` and float input. The repo has a single commit that adds both the label file and the splits file, so history cannot say what matrix was stratified. The likely cause is that the splits were produced from an earlier or differently ordered label matrix (the README's third step, `filter_flat_hierarchical`, is not in the repo), or a different iterative-stratification release (0.1.9 installed; not tested with another version, since that needs a download). Consequences: the 5-fold partition is a valid stratified 5-fold split with the authors' generator settings on the shipped hierarchical labels, but it is not guaranteed to match what the authors' pipeline would have produced. The 10-vs-5 difference in the results comes from both fold count and a different stratifier run; 5 seeds of this is not explored.

## Not changed

Authors' committed HEAD IPCARF (the uncommitted StandardScaler edit in the project tree is not used), leaky 285-column feature block, Youden threshold fitted on the evaluated test fold, `label_ranking_loss` on the thresholded matrix (authors' original behaviour), library versions (scikit-learn 1.7.2, numpy 2.3.1, pandas 2.3.3, Python 3.11 vs the authors' pins 1.5.1, 1.26.4, 2.2.2).
