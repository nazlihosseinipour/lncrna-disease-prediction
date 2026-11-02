"""Independent verification of the EXISTING V1 (355 x 827) paper reproductions.

Scope: IPCARF (MLC) and RFLDA (MLC) saved per-fold predictions under
results/paper_reproduction_reference_2026-09-20/, plus a protocol check of the
plain-RF summary. No model is fitted. Nothing outside OUT is written.
The authors' evaluate.py is NOT imported; metrics are recomputed here with sklearn/numpy.

Wording note: the 285-column binary block (cols 462-746) numerically matches the raw
association matrix; its provenance conflicts with the paper's description and remains
unresolved. This script does not assert anything further about it.

usage: <python> scripts/verify_v1_827_existing_results.py
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, average_precision_score, hamming_loss,
                             label_ranking_loss, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/paper_reproduction_reference_2026-09-20"
DS = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets"
OUT = ROOT / "results/felipe_paper_exact_2026-09-29T0014/v1_verification"
FREEZE = ROOT / "results/felipe_paper_exact_2026-09-29T0014/freeze/protected_before.sha256"

if OUT.exists():
    sys.exit(f"refusing to overwrite existing output dir: {OUT}")

INPUTS = {
    "X": (DS / "features/features_preprocessed.csv", "b264dd75"),
    "Y": (DS / "flat_labels/labels_flat_preprocessed.csv", "53c3573b"),
    "folds": (DS / "splits/inductive/inductive_splits.csv", "e9fe2eb5"),
}
MODELS = {
    "ipcarf_mlc": {
        "pred": REF / "ipcarf/predictions_mlc/labels_flat_preprocessed_IPCARF_inductive_multilabel_predictions_test_fold{k}.csv",
        "prior": REF / "ipcarf/_prior_attempt_20260920_2004/predictions_mlc/labels_flat_preprocessed_IPCARF_inductive_multilabel_predictions_test_fold{k}.csv",
        "per_fold": REF / "ipcarf/ipcarf_table3_per_fold.csv",
        "repro": REF / "ipcarf/ipcarf_table3_reproduction.csv",
        "variant": "IPCARF (MLC)",
        "eval": REF / "ipcarf/eval/labels_flat_preprocessed_IPCARF_inductive_multilabel_performance.csv",
    },
    "rflda_mlc": {
        "pred": REF / "table3_other_rows/predictions/rflda_mlc/labels_flat_preprocessed_RFLDA_inductive_multilabel_predictions_test_fold{k}.csv",
        "prior": None,
        "per_fold": REF / "table3_other_rows/table3_other_rows_per_fold.csv",
        "repro": REF / "table3_other_rows/table3_other_rows_reproduction.csv",
        "variant": "RFLDA (MLC)",
        "eval": REF / "table3_other_rows/eval/rflda_mlc/labels_flat_preprocessed_RFLDA_inductive_multilabel_performance.csv",
    },
}
METRICS = ["Hamming", "RankingErr", "AUROC", "AUPRC", "Precision", "Recall", "F", "Accuracy"]
EVAL_COLS = ["hamming", "label_ranking", "micro_roc", "micro_auprc", "precision", "recall", "fscore", "accuracy"]
# Paper Table III (tmp/pdfs/inductive_paper.txt), mean (std)
PAPER = {
    "IPCARF (MLC)": dict(Hamming=(0.08, 0.02), RankingErr=(0.24, 0.05), AUROC=(0.82, 0.06), AUPRC=(0.44, 0.12),
                         Precision=(0.36, 0.09), Recall=(0.77, 0.11), F=(0.48, 0.09), Accuracy=(0.14, 0.07)),
    "RFLDA (MLC)": dict(Hamming=(0.08, 0.01), RankingErr=(0.24, 0.05), AUROC=(0.81, 0.07), AUPRC=(0.41, 0.11),
                        Precision=(0.34, 0.06), Recall=(0.75, 0.12), F=(0.46, 0.07), Accuracy=(0.17, 0.07)),
}
TOL_EXACT = 1e-9


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def youden_thresholds(yt, ys):
    """Paper rule: per label roc_curve on the evaluated fold; argmax(tpr-fpr) if sum>0 else 0.5.
    Single-class labels: roc_curve yields NaN tpr or fpr -> sum is NaN -> 0.5."""
    out = []
    for c in range(yt.shape[1]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fpr, tpr, thr = roc_curve(yt[:, c], ys[:, c])
        s = tpr - fpr
        if np.nansum(s) > 0 and not np.isnan(s.sum()):
            out.append((float(thr[np.nanargmax(s)]), False))
        else:
            out.append((0.5, True))
    return out


def apply_threshold(ys, th):
    # authors' sequential rule: set < t to 0, then >= t to 1 (for scores in [0,1] and finite t
    # this equals ys >= t); reproduced literally for safety
    y = ys.astype(float).copy()
    for c, t in enumerate(th):
        col = y[:, c]
        col[col < t] = 0
        col[col >= t] = 1
        y[:, c] = col
    return y.astype(int)


def fold_metrics(yt, ys):
    th = youden_thresholds(yt, ys)
    yb = apply_threshold(ys, [t for t, _ in th])
    p, r, f, _ = precision_recall_fscore_support(yt, yb, average="micro", zero_division=0)
    m = dict(Hamming=hamming_loss(yt, yb), RankingErr=label_ranking_loss(yt, yb),
             AUROC=roc_auc_score(yt, ys, average="micro"),
             AUPRC=average_precision_score(yt, ys, average="micro"),
             Precision=p, Recall=r, F=f, Accuracy=accuracy_score(yt, yb))
    return m, th


def _changed_lines(a, b):
    import difflib
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))]


def code_audit(summary, checks, Y, Yv, tests):
    """Static inspection of code_used vs git HEAD and the current working tree (no execution)."""
    import re
    import subprocess
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    git = lambda rel: subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{rel}"], capture_output=True,
                                     text=True).stdout
    L = "lncRNA_CIBCB2025-main/"
    pairs = {
        "ipcarf_mlc_model": (REF / "ipcarf/code_used/ipcarf_HEAD_plus_instrumentation.py", L + "IPCARF/ipcarf.py"),
        "rflda_mlc_model": (REF / "table3_other_rows/code_used/rflda_HEAD_patched.py", L + "RFLDA/rflda.py"),
        "ipcarf_b_model_(not_audited_run)": (REF / "table3_other_rows/code_used/ipcarf_HEAD_patched.py", L + "IPCARF/ipcarf.py"),
        "IPCARF_main": (REF / "ipcarf/code_used/IPCARF_main.py", L + "IPCARF/main.py"),
        "RFLDA_main": (REF / "table3_other_rows/code_used/RFLDA_main.py", L + "RFLDA/main.py"),
        "evaluate": (REF / "ipcarf/code_used/evaluate_73b21ee.py", L + "parse_results/evaluate.py"),
        "parse_main": (REF / "ipcarf/code_used/parse_results_main_73b21ee.py", L + "parse_results/main.py"),
        "converter": (REF / "table3_other_rows/code_used/transductive_converter.py", L + "utils/transductive_converter.py"),
    }
    audit = {"git_head": head}
    for k, (used, rel) in pairs.items():
        u = used.read_text()
        audit[k] = {"changed_lines_vs_git_HEAD": _changed_lines(git(rel), u),
                    "n_changed_lines_vs_working_tree": len(_changed_lines((ROOT / rel).read_text(), u))}
    ev73 = subprocess.run(["git", "-C", str(ROOT), "show", "73b21ee:" + L + "parse_results/evaluate.py"],
                          capture_output=True, text=True).stdout
    audit["evaluate_used_equals_73b21ee"] = ev73 == (REF / "ipcarf/code_used/evaluate_73b21ee.py").read_text()
    ip = (REF / "ipcarf/code_used/ipcarf_HEAD_plus_instrumentation.py").read_text()
    rl = (REF / "table3_other_rows/code_used/rflda_HEAD_patched.py").read_text()
    ipc_instr_only = all(("[INSTR]" in l) or re.fullmatch(r"[+-]\s*except Exception( as e)?:", l)
                         for l in audit["ipcarf_mlc_model"]["changed_lines_vs_git_HEAD"])
    settings = {
        "ipcarf_n_estimators_150": "n_estimators = 150" in ip, "ipcarf_random_state_0": "random_state = 0" in ip,
        "ipcarf_inner_cv_5": "n_folds = 5" in ip and "cv=self.n_folds" in ip,
        "ipcarf_grid_2_to_128": "[2, 4, 8, 16, 32, 64, 128]" in ip,
        "rflda_n_estimators_150": "n_estimators = 150" in rl, "rflda_random_state_0": "random_state = 0" in rl,
        "rflda_inner_cv_5": re.search(r"^\s*n_folds = 5", rl, re.M) is not None,
        "rflda_step_50_incl_827": "range(50, x.shape[1] + 1, 50)" in rl and "intervals.append(x.shape[1])" in rl,
        "main_transductive_mode_False": "transductive_mode = False" in (REF / "ipcarf/code_used/IPCARF_main.py").read_text()
                                        and "transductive_mode = False" in (REF / "table3_other_rows/code_used/RFLDA_main.py").read_text(),
        "binary_mode_false_ipcarf_run": '"false"]' in (REF / "ipcarf/run_ipcarf_authors.py").read_text(),
        "binary_mode_false_rflda_run": 'rflda_mlc) M=RFLDA; L="$D/flat_labels/labels_flat_preprocessed.csv"; MODE=mlc'
                                       in (REF / "table3_other_rows/run_variant.sh").read_text(),
        "no_pca_fallback_in_logs": all("FALLBACK" not in q.read_text() for q in (REF / "ipcarf/logs").glob("*.log")),
    }
    # RFLDA inner selection results
    sel = []
    for k in range(1, 11):
        f = pd.read_csv(REF / f"table3_other_rows/predictions/rflda_mlc/labels_flat_preprocessed_accuracy_inductive_multilabel_find_best_n_{k}.csv")
        v = f.iloc[1, 1:].astype(float)
        sel.append({"fold": k, "best_n": int(v.index[int(np.argmax(v.values))]), "n_candidates": len(v),
                    "val_subset_acc_best": float(v.max()), "val_subset_acc_827": float(v["827"])})
    audit["rflda_inner_selection"] = sel
    audit["rflda_827_ever_selected"] = any(r["best_n"] == 827 for r in sel)
    audit["rflda_importance_order_is_identity"] = (
        "pd.Series(rf.feature_importances_, index = range(x.shape[1]))" in rl
        and "self.feature_importance.index.tolist()" in rl and re.search(r"sort_values|argsort|sorted\(|\.sort\(", rl) is None)
    # Youden sum-rule fallbacks on two-class labels
    fb = {}
    for name, cfg in MODELS.items():
        n2 = nmax = 0
        for k in range(1, 11):
            te = tests[k]; P = pd.read_csv(str(cfg["pred"]).format(k=k)).values; yt = Yv[te]
            for c in range(44):
                if 0 < yt[:, c].sum() < len(te):
                    fpr, tpr, _ = roc_curve(yt[:, c], P[:, c]); sc = tpr - fpr
                    if not sc.sum() > 0:
                        n2 += 1; nmax += int(sc.max() > 0)
        fb[name] = {"two_class_labels_fallback_0.5": n2, "of_which_max_youden_gt_0": nmax}
    audit["youden_sum_rule_fallbacks"] = fb
    audit["settings"] = settings
    summary["code_audit"] = audit
    checks["ipcarf_mlc_code_used_is_HEAD_plus_print_instrumentation_only"] = {"pass": bool(ipc_instr_only)}
    checks["settings_confirmed_from_code_used_and_run_scripts"] = {"pass": all(settings.values()), **settings}
    checks["evaluate_code_used_equals_commit_73b21ee"] = {"pass": audit["evaluate_used_equals_73b21ee"]}
    checks["working_tree_model_files_equal_code_used"] = {
        "pass": audit["ipcarf_mlc_model"]["n_changed_lines_vs_working_tree"] == 0
        and audit["rflda_mlc_model"]["n_changed_lines_vs_working_tree"] == 0,
        "ipcarf_changed_lines": audit["ipcarf_mlc_model"]["n_changed_lines_vs_working_tree"],
        "rflda_changed_lines": audit["rflda_mlc_model"]["n_changed_lines_vs_working_tree"],
        "note": "informational: a rerun from the current working tree would not use the code that produced these predictions"}


def region_auc(yt, ys, te, cut=231):
    out = {}
    for tag, mask in (("pos_lt231", te < cut), ("pos_ge231", te >= cut)):
        idx = np.where(mask)[0]
        out[f"n_{tag}"] = int(len(idx))
        for sh in (0, 1):
            key = f"auroc_{tag}_" + ("aligned" if sh == 0 else "shift_plus1")
            try:
                out[key] = float(roc_auc_score(yt[idx], np.roll(ys[idx], sh, axis=0), average="micro"))
            except ValueError:
                out[key] = None
    return out


def main():
    OUT.mkdir(parents=True)
    summary = {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "hashes": {}, "checks": {}, "max_abs_diff": {}}
    checks = summary["checks"]

    # ---------- 1. hashes ----------
    for k, (p, pref) in INPUTS.items():
        h = sha(p)
        summary["hashes"][str(p.relative_to(ROOT))] = h
        checks[f"input_hash_{k}"] = {"pass": h.startswith(pref), "sha256": h, "expected_prefix": pref}
    hash_dirs = [REF / "ipcarf/predictions_mlc", REF / "ipcarf/eval", REF / "ipcarf/code_used", REF / "ipcarf/logs",
                 REF / "ipcarf/_prior_attempt_20260920_2004/predictions_mlc",
                 REF / "table3_other_rows/predictions/rflda_mlc", REF / "table3_other_rows/eval/rflda_mlc",
                 REF / "table3_other_rows/code_used", REF / "table3_other_rows/logs"]
    hash_files = [REF / "ipcarf/ipcarf_table3_per_fold.csv", REF / "ipcarf/ipcarf_table3_reproduction.csv",
                  REF / "ipcarf/FIDELITY.md", REF / "ipcarf/run_ipcarf_authors.py", REF / "ipcarf/run_all.sh",
                  REF / "ipcarf/run_eval_authors.sh", REF / "ipcarf/sanity_checks.py",
                  REF / "table3_other_rows/table3_other_rows_per_fold.csv",
                  REF / "table3_other_rows/table3_other_rows_reproduction.csv",
                  REF / "table3_other_rows/run_authors.py", REF / "table3_other_rows/run_variant.sh",
                  REF / "table3_other_rows/run_eval.sh", REF / "paper_reproduction_reference.csv",
                  ROOT / "scripts/reproduce_paper_reference.py",
                  ROOT / "lncRNA_CIBCB2025-main/IPCARF/ipcarf.py", ROOT / "lncRNA_CIBCB2025-main/RFLDA/rflda.py",
                  ROOT / "lncRNA_CIBCB2025-main/IPCARF/main.py", ROOT / "lncRNA_CIBCB2025-main/RFLDA/main.py",
                  ROOT / "lncRNA_CIBCB2025-main/parse_results/evaluate.py",
                  ROOT / "lncRNA_CIBCB2025-main/parse_results/main.py",
                  ROOT / "lncRNA_CIBCB2025-main/utils/transductive_converter.py",
                  ROOT / "lncRNA_CIBCB2025-main/utils/utils.py"]
    for d in hash_dirs:
        hash_files += sorted(q for q in d.iterdir() if q.is_file() and q.name != ".DS_Store")
    for p in hash_files:
        summary["hashes"][str(p.relative_to(ROOT))] = sha(p)
    # compare with freeze manifest where listed
    frozen = {}
    if FREEZE.exists():
        for line in FREEZE.read_text().splitlines():
            h, _, path = line.partition("  ")
            frozen[path.strip()] = h
    listed = {k: v for k, v in summary["hashes"].items() if k in frozen}
    mism = [k for k, v in listed.items() if frozen[k] != v]
    checks["hashes_vs_freeze_manifest"] = {"pass": not mism, "n_compared": len(listed), "mismatches": mism}

    # ---------- 2. data / fold integrity ----------
    X = pd.read_csv(INPUTS["X"][0], index_col=0)
    Y = pd.read_csv(INPUTS["Y"][0], index_col=0)
    S = pd.read_csv(INPUTS["folds"][0])
    Yv = Y.values.astype(int)
    checks["X_Y_shapes_and_row_order"] = {
        "pass": bool(X.shape == (355, 827) and Y.shape == (355, 44) and (X.index == Y.index).all()
                     and np.isfinite(X.values.astype(float)).all() and set(np.unique(Yv)) <= {0, 1}),
        "X_shape": list(X.shape), "Y_shape": list(Y.shape), "same_index_order": bool((X.index == Y.index).all())}
    tests, fold_ok, fold_info = {}, True, []
    for k in range(1, 11):
        tr_raw, te_raw = S[f"train_fold{k}"].dropna().values, S[f"test_fold{k}"].dropna().values
        integer = bool(np.all(tr_raw == np.round(tr_raw)) and np.all(te_raw == np.round(te_raw)))
        tr, te = tr_raw.astype(int), te_raw.astype(int)
        ok = (integer and len(set(tr)) == len(tr) and len(set(te)) == len(te) and not set(tr) & set(te)
              and set(tr) | set(te) == set(range(355)))
        fold_ok &= ok
        tests[k] = te
        fold_info.append({"fold": k, "n_train": len(tr), "n_test": len(te), "disjoint_and_cover_all": ok,
                          "test_sorted": bool(np.all(np.diff(te) > 0))})
    allte = np.concatenate(list(tests.values()))
    once = bool(len(allte) == 355 and set(allte) == set(range(355)))
    checks["folds_train_test_disjoint_cover"] = {"pass": bool(fold_ok), "per_fold": fold_info}
    checks["each_row_in_exactly_one_test_fold"] = {"pass": once, "n_test_total": int(len(allte))}

    # ---------- 3. predictions + recomputation ----------
    paper_rows, rng = [], np.random.default_rng(0)
    for name, cfg in MODELS.items():
        per_fold_rows, th_rows, integ, pooled_t, pooled_s = [], [], [], [], []
        for k in range(1, 11):
            P = pd.read_csv(str(cfg["pred"]).format(k=k))
            te = tests[k]
            ys = P.values.astype(float)
            yt = Yv[te]
            cols_ok = list(P.columns) == list(Y.columns)
            shape_ok = P.shape == (len(te), 44)
            finite = bool(np.isfinite(ys).all())
            in01 = bool((ys >= 0).all() and (ys <= 1).all())
            m, th = fold_metrics(yt, ys)
            # alignment evidence (predictions carry no IDs; rows are positional = order of test_fold<k>)
            perm_auc = np.array([roc_auc_score(yt, ys[rng.permutation(len(te))], average="micro")
                                 for _ in range(200)])
            pooled_t.append(yt); pooled_s.append(ys)
            integ.append({"fold": k, "rows": P.shape[0], "n_test": len(te), "shape_ok": shape_ok,
                          "columns_equal_Y_order": cols_ok, "finite": finite, "in_0_1": in01,
                          "has_id_column": False,
                          "auroc_aligned": m["AUROC"], "auroc_row_permuted_mean": float(perm_auc.mean()),
                          "auroc_row_permuted_max_of_200": float(perm_auc.max()),
                          "perm_p_value": float((1 + (perm_auc >= m["AUROC"]).sum()) / 201),
                          "auroc_shift_plus1": roc_auc_score(yt, np.roll(ys, 1, axis=0), average="micro"),
                          "auroc_shift_minus1": roc_auc_score(yt, np.roll(ys, -1, axis=0), average="micro"),
                          # diagnostic: aligned vs shifted AUROC for test rows at file positions <231 and >=231
                          **region_auc(yt, ys, te)})
            per_fold_rows.append({"model": name, "fold": f"fold{k}", **m})
            for j, (t, fb) in enumerate(th):
                th_rows.append({"model": name, "fold": k, "label_idx": j, "label": Y.columns[j],
                                "threshold": t, "fallback_0.5": fb, "n_pos_test": int(yt[:, j].sum()),
                                "n_test": len(te)})
        rec = pd.DataFrame(per_fold_rows)
        rec.to_csv(OUT / f"recomputed_{name}_per_fold.csv", index=False)
        pd.DataFrame(th_rows).to_csv(OUT / f"thresholds_{name}.csv", index=False)
        ok_int = all(r["shape_ok"] and r["columns_equal_Y_order"] and r["finite"] for r in integ)
        # pooled positional-alignment test over all 355 rows (rows of fold k = order of test_fold<k>)
        PT, PS = np.vstack(pooled_t), np.vstack(pooled_s)
        a0 = roc_auc_score(PT, PS, average="micro")
        pp = np.array([roc_auc_score(PT, PS[rng.permutation(len(PT))], average="micro") for _ in range(200)])
        sh = {d: roc_auc_score(PT, np.roll(PS, d, axis=0), average="micro") for d in (-2, -1, 1, 2)}
        align = bool(a0 > pp.max() and a0 > max(sh.values()))
        checks[f"{name}_prediction_integrity"] = {"pass": ok_int, "per_fold": integ}
        checks[f"{name}_alignment_pooled_positional"] = {
            "pass": align, "pooled_auroc_aligned": a0, "pooled_perm_max_of_200": float(pp.max()),
            "pooled_perm_mean": float(pp.mean()), "pooled_cyclic_shift_auroc": {str(k): v for k, v in sh.items()},
            "folds_with_per_fold_perm_p_gt_0.05": [r["fold"] for r in integ if r["perm_p_value"] > 0.05],
            "note": "predictions carry no IDs; identity is positional. Per-fold row-permutation tests are weak "
                    "(36 rows); the pooled test and exact metric agreement with the authors' eval are the evidence."}

        # compare with reported per-fold CSV
        pf = pd.read_csv(cfg["per_fold"])
        pf = pf[pf["variant"] == cfg["variant"]].set_index("fold").loc[[f"fold{k}" for k in range(1, 11)]]
        d_pf = {mt: float(np.max(np.abs(pf[mt].values - rec[mt].values))) for mt in METRICS}
        # compare with authors' eval output
        ev = pd.read_csv(cfg["eval"]).set_index("folds")
        evf = ev.loc[[f"fold{k}" for k in range(1, 11)], EVAL_COLS].astype(float)
        d_ev = {mt: float(np.max(np.abs(evf[c].values - rec[mt].values))) for mt, c in zip(METRICS, EVAL_COLS)}
        mean = rec[METRICS].mean()
        std = rec[METRICS].apply(lambda s: np.std(s.values))  # population std
        # eval "mean(std)" row
        ms = ev.loc["mean(std)", EVAL_COLS]
        ev_mean = {mt: float(str(ms[c]).split("(")[0]) for mt, c in zip(METRICS, EVAL_COLS)}
        ev_std = {mt: float(str(ms[c]).split("(")[1].rstrip(")")) for mt, c in zip(METRICS, EVAL_COLS)}
        rp = pd.read_csv(cfg["repro"])
        rp = rp[rp["variant"] == cfg["variant"]].set_index("metric")
        d_rep_mean = {mt: float(abs(rp.loc[mt, "ours_mean"] - round(mean[mt], 4))) for mt in METRICS}
        d_rep_std = {mt: float(abs(rp.loc[mt, "ours_std"] - round(std[mt], 4))) for mt in METRICS}
        summary["max_abs_diff"][name] = {
            "vs_per_fold_csv": d_pf, "vs_authors_eval_per_fold": d_ev,
            "mean_vs_authors_eval_mean": {mt: float(abs(ev_mean[mt] - mean[mt])) for mt in METRICS},
            "std_vs_authors_eval_std": {mt: float(abs(ev_std[mt] - std[mt])) for mt in METRICS},
            "rounded4_mean_vs_reproduction_table": d_rep_mean, "rounded4_std_vs_reproduction_table": d_rep_std}
        checks[f"{name}_recompute_matches_per_fold_csv"] = {"pass": max(d_pf.values()) < TOL_EXACT,
                                                            "max": max(d_pf.values())}
        checks[f"{name}_recompute_matches_authors_eval"] = {"pass": max(d_ev.values()) < TOL_EXACT,
                                                            "max": max(d_ev.values())}
        checks[f"{name}_recompute_matches_reproduction_table_4dp"] = {
            "pass": max(list(d_rep_mean.values()) + list(d_rep_std.values())) < 1e-9}
        within = {}
        for mt in METRICS:
            pm, ps = PAPER[cfg["variant"]][mt]
            within[mt] = abs(round(mean[mt], 2) - pm) < 1e-9
            paper_rows.append({"model": cfg["variant"], "metric": mt, "recomputed_mean": mean[mt],
                               "recomputed_std": std[mt], "reported_mean": rp.loc[mt, "ours_mean"],
                               "reported_std": rp.loc[mt, "ours_std"], "paper_mean": pm, "paper_std": ps,
                               "diff_vs_paper_mean": mean[mt] - pm, "diff_vs_paper_std": std[mt] - ps,
                               "rounds_to_paper_mean_2dp": within[mt],
                               "rounds_to_paper_std_2dp": abs(round(std[mt], 2) - ps) < 1e-9})
        checks[f"{name}_means_round_to_paper_2dp"] = {"pass": all(within.values()),
                                                      "not_matching": [m for m, v in within.items() if not v]}

        # ---------- 5. determinism ----------
        if cfg["prior"] is not None:
            diffs, present = [], []
            for k in range(1, 11):
                pp = Path(str(cfg["prior"]).format(k=k))
                if pp.exists():
                    a = pd.read_csv(str(cfg["pred"]).format(k=k)).values
                    b = pd.read_csv(pp).values
                    present.append(k)
                    diffs.append(float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf"))
            bn_a = pd.read_csv(REF / "ipcarf/predictions_mlc/labels_flat_preprocessed_inductive_multilabel_best_n_components.csv").values.ravel().tolist()
            bn_p = pd.read_csv(REF / "ipcarf/_prior_attempt_20260920_2004/predictions_mlc/labels_flat_preprocessed_inductive_multilabel_best_n_components.csv").values.ravel().tolist()
            checks[f"{name}_determinism_vs_prior_attempt"] = {
                "pass": bool(present) and max(diffs) == 0.0, "folds_compared": present,
                "max_abs_diff_per_fold": diffs, "best_n_components_current": bn_a,
                "best_n_components_prior": bn_p, "best_n_equal": bn_a == bn_p}

    pd.DataFrame(paper_rows).to_csv(OUT / "v1_paper_comparison_draft.csv", index=False)

    # ---------- 6. plain RF (summary only) ----------
    rf_csv = pd.read_csv(REF / "paper_reproduction_reference.csv", index_col=0)
    checks["plain_rf_summary_only"] = {
        "pass": True, "note": "summary CSV only; no per-fold predictions saved; paper Table III has no plain-RF row",
        "row_paper_827_full": rf_csv.loc["paper_827_full__PAPER_REPRODUCTION"].to_dict()}

    code_audit(summary, checks, Y, Yv, tests)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    fails = [k for k, v in checks.items() if not v["pass"]]
    print("checks:", len(checks), "failed:", fails)
    print(pd.DataFrame(paper_rows)[["model", "metric", "recomputed_mean", "recomputed_std", "paper_mean",
                                    "diff_vs_paper_mean", "rounds_to_paper_mean_2dp"]].to_string())
    for name in MODELS:
        print(name, "max|diff| per-fold csv:", max(summary["max_abs_diff"][name]["vs_per_fold_csv"].values()),
              "authors eval:", max(summary["max_abs_diff"][name]["vs_authors_eval_per_fold"].values()))


if __name__ == "__main__":
    main()
