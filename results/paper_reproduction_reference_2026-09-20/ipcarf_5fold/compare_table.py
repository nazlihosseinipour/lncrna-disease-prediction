"""10-fold vs 5-fold vs paper Table III. Reads both reproduction CSVs, writes ipcarf_5fold_vs_10fold_vs_paper.csv and prints a summary."""
import pandas as pd
B = "/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/results/paper_reproduction_reference_2026-09-20/"
a = pd.read_csv(B + "ipcarf/ipcarf_table3_reproduction.csv"); b = pd.read_csv(B + "ipcarf_5fold/ipcarf_5fold_table3_reproduction.csv")
a = a[a.paper_mean.notna()].set_index(["variant", "metric"]); b = b[b.paper_mean.notna()].set_index(["variant", "metric"])
t = pd.DataFrame({"paper_mean": a.paper_mean, "paper_std": a.paper_std,
                  "tenfold_mean": a.ours_mean, "tenfold_std": a.ours_std, "fivefold_mean": b.ours_mean, "fivefold_std": b.ours_std})
t["tenfold_absdev"] = (t.tenfold_mean - t.paper_mean).abs().round(4); t["fivefold_absdev"] = (t.fivefold_mean - t.paper_mean).abs().round(4)
t["closer_to_paper"] = ["5-fold" if x < y - 1e-9 else "10-fold" if y < x - 1e-9 else "tie" for x, y in zip(t.fivefold_absdev, t.tenfold_absdev)]
t.reset_index().to_csv(B + "ipcarf_5fold/ipcarf_5fold_vs_10fold_vs_paper.csv", index=False)
print(t.round(3).to_string())
for v, g in t.groupby(level=0): print(v, "mean |dev|: 10-fold %.4f, 5-fold %.4f" % (g.tenfold_absdev.mean(), g.fivefold_absdev.mean()), g.closer_to_paper.value_counts().to_dict())
