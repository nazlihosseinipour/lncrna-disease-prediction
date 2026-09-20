"""Aggregate the authors' parse_results output into the Table III comparison.
Reads eval/*_performance.csv (per-fold rows fold1..fold10 + authors' mean(std) row), recomputes mean and
population std (np.std, ddof=0, exactly as the authors' create_df_inductive does), compares with the paper.
LEAKAGE-INFLATED REFERENCE (not a valid inductive result).
usage: python build_table.py [eval_dir] [out_prefix]
"""
import sys, pandas as pd, numpy as np
R = "/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/results/paper_reproduction_reference_2026-09-20/ipcarf"
ev = sys.argv[1] if len(sys.argv) > 1 else R + "/eval"
pre = sys.argv[2] if len(sys.argv) > 2 else R + "/ipcarf_table3"
cols = ["hamming", "label_ranking", "micro_roc", "micro_auprc", "precision", "recall", "fscore", "accuracy"]
names = ["Hamming", "RankingErr", "AUROC", "AUPRC", "Precision", "Recall", "F", "Accuracy"]
paper = {  # Table III (mean, std)
 "IPCARF (HMC)": [(.08,.02),(.24,.07),(.83,.07),(.44,.13),(.36,.08),(.78,.13),(.48,.08),(.18,.08)],
 "IPCARF (MLC)": [(.08,.02),(.24,.05),(.82,.06),(.44,.12),(.36,.09),(.77,.11),(.48,.09),(.14,.07)]}
src = {  # variant -> authors' output file. HMC primary = hierarchical predictions filtered to the 44 original diseases
 "IPCARF (HMC)": "mesh_labels_final_IPCARF_inductive_multilabel_hierarchical_to_flat_performance.csv",
 "IPCARF (MLC)": "labels_flat_preprocessed_IPCARF_inductive_multilabel_performance.csv"}
extra = {"IPCARF (HMC, all 369 MeSH nodes; not the Table III protocol)": "mesh_labels_final_IPCARF_inductive_multilabel_performance.csv"}
rows, perfold = [], []
for v, f in {**src, **extra}.items():
    d = pd.read_csv(f"{ev}/{f}"); folds = d[d.folds.str.startswith("fold")]; assert len(folds) == 10
    m, s = folds[cols].astype(float).mean().values, folds[cols].astype(float).std(ddof=0).values
    for _, r in folds.iterrows(): perfold.append({"variant": v, "fold": r.folds, **{n: float(r[c]) for n, c in zip(names, cols)}})
    for i, n in enumerate(names):
        row = {"variant": v, "metric": n, "ours_mean": round(m[i], 4), "ours_std": round(s[i], 4)}
        if v in paper:
            pm, ps = paper[v][i]; row.update(paper_mean=pm, paper_std=ps, dev_mean=round(m[i]-pm, 4), dev_std=round(s[i]-ps, 4))
        rows.append(row)
t = pd.DataFrame(rows); t.to_csv(pre + "_reproduction.csv", index=False); pd.DataFrame(perfold).to_csv(pre + "_per_fold.csv", index=False)
for v in list(src) + list(extra):
    x = t[t.variant == v].set_index("metric")
    print(f"\n{v}\n" + x.drop(columns="variant").to_string())
