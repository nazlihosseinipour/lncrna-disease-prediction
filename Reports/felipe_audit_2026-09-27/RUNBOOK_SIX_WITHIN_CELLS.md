# Runbook — six within-version cells (V1→V1, V2→V2 × RF/IPCARF/RFLDA)

Prepared 2026-09-28 ~00:05 CEST after wave T1–T6 passed review. Transfer runs are deferred.
Run from the repo root. Do **not** edit any file listed in `config/felipe_frozen_sources.csv`
while runs are in progress: the runners refuse to start, and resume rejects cells, if a hash changes.

```bash
cd "/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project"
PY=results/runtime/felipe_env/bin/python
OUT=results/felipe_final_2026-09-28
```

## 1. Pre-flight (seconds)

```bash
$PY scripts/verify_felipe_exact_results.py --check manifests
```

## 2. V1→V1, all three models (~8 min)

```bash
caffeinate -i $PY scripts/run_inductive_within_cv.py --manifest config/felipe_basics_sequence_542_manifest.csv --model-profile paper --versions v1 --label-spaces all44 --models rf ipcarf rflda --resume --outdir $OUT 2>&1 | tee -a $OUT.v1.log
```

## 3. V2→V2 RF then IPCARF (~2.5 h)

```bash
caffeinate -i $PY scripts/run_inductive_within_cv.py --manifest config/felipe_basics_sequence_542_manifest.csv --model-profile paper --versions v2 --label-spaces all436 --models rf ipcarf --resume --outdir $OUT 2>&1 | tee -a $OUT.v2_rf_ipcarf.log
```

## 4. V2→V2 RFLDA (~7.7 h; critical path)

```bash
caffeinate -i $PY scripts/run_inductive_within_cv.py --manifest config/felipe_basics_sequence_542_manifest.csv --model-profile paper --versions v2 --label-spaces all436 --models rflda --resume --outdir $OUT 2>&1 | tee -a $OUT.v2_rflda.log
```

Run steps 2–4 one after another, not concurrently: each already uses all 10 cores.
If interrupted, rerun the same command unchanged; `--resume` keeps complete folds and
quarantines anything incomplete into `_rejected/`.

## 5. Verify, then score (minutes)

```bash
$PY scripts/verify_felipe_exact_results.py --check matrix --scope within6 --results-root $OUT
$PY scripts/rescore_felipe_predictions.py --input-root $OUT/within --modes fixed paper_youden --outdir $OUT/scored_within6
```

For a checkpoint before RFLDA V2 finishes, score the complete groups only (the rescorer
excludes incomplete groups and lists them in `incomplete_groups.csv`):

```bash
$PY scripts/rescore_felipe_predictions.py --input-root $OUT/within --modes fixed paper_youden --outdir $OUT/scored_checkpoint_$(date +%H%M)
```

## Runtime basis (probe-derived, this Mac, 10 cores)

| Cell | Evidence | Per fold | 10 folds |
|---|---|---|---|
| V1 RF+IPCARF+RFLDA | real runner, full settings, fold 1: 47 s for all three | 47 s | ~8 min |
| V2 RF | fold 1, 30 trees: 71.8 s × 5 | ~6 min | ~60 min |
| V2 IPCARF | fold 1, 30 trees: 107 s × 5 (PCA part overcounted) | ≤9 min | ≤90 min |
| V2 RFLDA | fold 1, 30 trees: 554 s × 5 | ~46 min | ~7.7 h |

The ×5 scaling assumes forest cost is linear in tree count (30→150 trees = 3→15 parallel
rounds on 10 cores). Treat these as ±25 %. Sequential total ≈ 10.3 h.
