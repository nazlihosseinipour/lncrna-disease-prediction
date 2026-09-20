"""Clean-pipeline baselines from SAVED per-fold predictions only (no model is fit or refit).

Reads results/v1_core583_full10_oob_2026-08-24 and results/v1_all2023_full10_oob_2026-08-24
(predictions/<cell>/rf/test_fold*.csv, the run's manifest, and within_version_cv_summary.csv), recovers
y_train/y_test from the manifest Y + split file with the repo's iterator_cross_validation, and reports
per fold and mean: micro AUROC, micro AUPRC, macro AUROC (labels with one class in the test fold are
skipped and counted), per-sample AUROC (samples with no positives or all positives are skipped and
counted), and a no-feature prior baseline (constant per-label score = training-fold positive rate) for
micro AUROC/AUPRC on the same folds. Also carries through the stored train_oob_f1 F1 from the summary.
Writes results/clean_baselines_2026-09-20/clean_baselines.csv. Stops if the micro AUROC mean
does not match the summary value to 3 decimals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (ROOT / "lncRNA_CIBCB2025-main").as_posix())
from utils.utils import iterator_cross_validation  # noqa: E402

RUNS = ["v1_core583_full10_oob_2026-08-24", "v1_all2023_full10_oob_2026-08-24"]
OUT = ROOT / "results" / "clean_baselines_2026-09-20"


def macro_auroc(y, s):
    aucs = [roc_auc_score(y[:, j], s[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(aucs)), y.shape[1] - len(aucs)


def sample_auroc(y, s):
    ok = [i for i in range(y.shape[0]) if 0 < y[i].sum() < y.shape[1]]
    return float(np.mean([roc_auc_score(y[i], s[i]) for i in ok])), y.shape[0] - len(ok)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in RUNS:
        rdir = ROOT / "results" / run
        summ = pd.read_csv(rdir / "within_version_cv_summary.csv")
        assert len(summ) == 1, f"{run}: expected one summary row"
        s = summ.iloc[0]
        cfg = json.loads((rdir / "run_config.json").read_text())
        man = pd.read_csv(cfg["manifest"])
        m = man[man["feature_key"] == s["feature_key"]].iloc[0]
        y = pd.read_csv(m["y_path"], index_col=0)
        splits = pd.read_csv(m["split_path"])
        train, test = iterator_cross_validation(splits, pd.DataFrame(index=y.index), y, n_folds=10)
        pdir = Path(s["prediction_dir"])
        run_rows = []
        for f in range(1, 11):
            ytr, yte = train[f - 1][1], test[f - 1][1]
            pred = pd.read_csv(pdir / f"test_fold{f}.csv", index_col=0)
            Y, S = yte.values.astype(int), pred.values
            assert S.shape == Y.shape, (run, f, S.shape, Y.shape)
            prior = np.tile(ytr.values.mean(axis=0), (Y.shape[0], 1))
            mac, skip_l = macro_auroc(Y, S)
            per, skip_s = sample_auroc(Y, S)
            run_rows.append(dict(
                run=run, fold=str(f), n_test=Y.shape[0], n_labels=Y.shape[1],
                micro_auroc=roc_auc_score(Y, S, average="micro"),
                micro_auprc=average_precision_score(Y, S, average="micro"),
                macro_auroc=mac, macro_labels_skipped=skip_l,
                persample_auroc=per, persample_skipped=skip_s,
                prior_micro_auroc=roc_auc_score(Y, prior, average="micro"),
                prior_micro_auprc=average_precision_score(Y, prior, average="micro"),
            ))
        df = pd.DataFrame(run_rows)
        num = [c for c in df.columns if c not in ("run", "fold")]
        mean = df[num].mean().to_dict()
        mean.update(run=run, fold="mean")
        std = df[num].std(ddof=0).to_dict()  # population std, matches summary's micro_roc_std
        std.update(run=run, fold="std")
        for r in (mean, std):
            r["n_test"], r["n_labels"] = df["n_test"].sum() if r["fold"] == "mean" else np.nan, df["n_labels"].iloc[0]
            if r["fold"] == "mean":  # stored in summary, not recomputed
                r["train_oob_f1_mean"], r["train_oob_f1_std"] = s["fscore_mean"], s["fscore_std"]
                r["summary_micro_roc_mean"] = s["micro_roc_mean"]
        rows += run_rows + [mean, std]
        diff = abs(mean["micro_auroc"] - s["micro_roc_mean"])
        print(f"{run}: micro AUROC mean {mean['micro_auroc']:.6f} vs summary {s['micro_roc_mean']:.6f} (|d|={diff:.2e})")
        if round(mean["micro_auroc"], 3) != round(s["micro_roc_mean"], 3):
            sys.exit(f"MISMATCH in {run}; not writing outputs")
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "clean_baselines.csv", index=False)
    show = out[out.fold.isin(["mean", "std"])]
    print(show.drop(columns=["n_test"]).round(4).T.to_string())


if __name__ == "__main__":
    main()
