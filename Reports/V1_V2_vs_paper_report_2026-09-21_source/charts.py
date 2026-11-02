import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT = Path(__file__).parent / "fig"
PAPER, OURS, V1C, V2C, BASE, GREY = "#8a94a6", "#2f6db5", "#e08a2c", "#2a9d8f", "#b9bdc4", "#444"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "figure.dpi": 100})

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

def labels(ax, bars, fmt="{:.2f}", dy=0.01):
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8, color=GREY)

# Fig 1: reproduction vs Table III
models = ["RFLDA\n(MLC)", "IPCARF\n(MLC)", "IPCARF\n(HMC)"]
paper = {"AUROC": [.81, .82, .83], "AUPRC": [.41, .44, .44], "F-score": [.46, .48, .48]}
ours = {"AUROC": [.811, .823, .825], "AUPRC": [.413, .440, .436], "F-score": [.460, .483, .484]}
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.1), sharey=True)
x = np.arange(3); w = .36
for ax, m in zip(axs, paper):
    b1 = ax.bar(x - w/2, paper[m], w, color=PAPER, label="Paper, Table III")
    b2 = ax.bar(x + w/2, ours[m], w, color=OURS, label="Our reproduction")
    labels(ax, b1); labels(ax, b2); ax.set_title(m); ax.set_xticks(x, models); ax.set_ylim(0, .95)
axs[0].set_ylabel("Mean over 10 folds")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(); save(fig, "fig1_reproduction")

# Fig 2: column layout + effect of the 285-column block
fig = plt.figure(figsize=(9.2, 4.6))
ax0 = fig.add_axes([0.03, 0.80, 0.94, 0.14])
segs = [(462, "#c9d3e3", "columns 0-461\nsequence-type features"),
        (285, "#e6a1a1", "columns 462-746: 285 binary columns\n= raw lncRNA-disease association matrix"),
        (80, "#c9d3e3", "747-826")]
left = 0
for n, c, t in segs:
    ax0.barh(0, n, left=left, color=c, edgecolor="white"); ax0.text(left + n/2, 0, t, ha="center", va="center", fontsize=8.5); left += n
ax0.set_xlim(0, 827); ax0.axis("off"); ax0.set_title("Layout of the released feature file (355 x 827)", loc="left", fontsize=10)
ax = fig.add_axes([0.07, 0.10, 0.90, 0.56])
cfg = ["Full file\n(827 cols)", "Block only\n(285 cols)", "Without block\n(542 cols)", "No-feature\nbaseline"]
vals = {"AUROC": [.807, .820, .615, .704], "AUPRC": [.411, .436, .081, .112], "F-score (Youden)": [.447, .551, .198, 0]}
cols = [OURS, "#c0504d", V1C, BASE]
x = np.arange(4); w = .26
for i, (m, v) in enumerate(vals.items()):
    bars = ax.bar(x + (i-1)*w, v, w, color=cols, alpha=[1, .75, .5][i], edgecolor="white")
    labels(ax, bars, "{:.2f}", .01)
ax.set_xticks(x, cfg); ax.set_ylim(0, .95); ax.set_ylabel("Mean over 10 folds")
from matplotlib.patches import Patch
ax.legend([Patch(color="#777", alpha=a) for a in (1, .75, .5)], list(vals), frameon=False, ncol=3, loc="upper right", fontsize=8)
save(fig, "fig2_block")

# Fig 3: V2 vs baseline vs paper
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.1))
names = ["V2\ndefault", "V2\nleaf by\nOOB", "No-feature\nbaseline", "Paper\n(with\nblock)"]
data = {"AUROC (micro)": [.785, .827, .818, .81], "AUPRC (micro)": [.099, .142, .110, .41], "F-score (Youden)": [.119, .108, 0, .46]}
cols = [V2C, V2C, BASE, PAPER]
for ax, (m, v) in zip(axs, data.items()):
    bars = ax.bar(range(4), v, color=cols, edgecolor="white")
    bars[0].set_alpha(.55); labels(ax, bars, "{:.3f}" if m != "AUROC (micro)" else "{:.3f}", .01)
    ax.set_title(m); ax.set_xticks(range(4), names, fontsize=8); ax.set_ylim(0, max(v)*1.22 if max(v) > .5 else max(v)*1.3)
    ax.set_ylim(0, {"AUROC (micro)": 1.0, "AUPRC (micro)": .5, "F-score (Youden)": .55}[m])
fig.tight_layout(); save(fig, "fig3_v2")

# Fig 4: label density vs no-feature AUROC
fig, axs = plt.subplots(1, 2, figsize=(9.2, 2.9))
ds = ["V1\n(24 labels)", "Paper\n(44 labels)", "V2\n(86 labels)"]; cs = [V1C, PAPER, V2C]
b = axs[0].bar(range(3), [7.2, 4.9, 2.4], color=cs); labels(axs[0], b, "{:.1f}%", .1)
axs[0].set_title("Share of positive label cells"); axs[0].set_xticks(range(3), ds); axs[0].set_ylim(0, 9)
b = axs[1].bar(range(3), [.664, .704, .818], color=cs); labels(axs[1], b, "{:.3f}", .01)
axs[1].set_title("Micro AUROC of a no-feature baseline"); axs[1].set_xticks(range(3), ds); axs[1].set_ylim(0, 1.0)
fig.tight_layout(); save(fig, "fig4_density")

# Fig 5: macro AUROC, V1 vs V2 by scaling
fig, ax = plt.subplots(figsize=(7.6, 3.4))
arms = ["V1, raw counts\n(original V1)", "V1, length-\nnormalised", "V2, length-\nnormalised", "V2, default RF\n(fully grown)"]
m = [.556, .441, .577, .531]; s = [.059, .069, .058, .034]; p = ["p = 0.040", "p = 0.98", "p = 0.002", "p = 0.008"]
bars = ax.bar(range(4), m, yerr=s, color=[V1C, V1C, V2C, V2C], capsize=4, edgecolor="white")
bars[1].set_alpha(.55); bars[3].set_alpha(.55)
ax.axhline(.5, color=GREY, ls="--", lw=1); ax.set_xlim(-0.6, 3.85); ax.text(3.82, .503, "chance", ha="right", va="bottom", fontsize=8, color=GREY)
for i, (mm, ss, pp) in enumerate(zip(m, s, p)): ax.text(i, mm + ss + .008, f"{mm:.3f}\n{pp}", ha="center", fontsize=8, color=GREY)
ax.set_xticks(range(4), arms); ax.set_ylim(.3, .72); ax.set_ylabel("Macro AUROC, mean +/- SD over 10 folds")
ax.set_title("Per-label signal: V1 vs V2 (p = 500-draw shuffled-label test)")
fig.tight_layout(); save(fig, "fig5_macro")

# Fig 6: V2 learning curve (fold 1)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
n = [353, 800, 1600, 2400, 3467]; mod = [.776, .794, .792, .802, .802]; pri = [.816, .821, .822, .823, .824]
ax.plot(n, pri, "o--", color=BASE, label="No-feature baseline"); ax.plot(n, mod, "o-", color=V2C, label="RF on V2 features")
ax.set_xlabel("Training rows"); ax.set_ylabel("Micro AUROC on fold 1"); ax.set_ylim(.75, .85); ax.legend(frameon=False)
ax.set_title("More rows narrow the gap to the baseline but do not close it")
for a, b_ in zip(n, mod): ax.text(a, b_ - .009, f"{b_:.3f}", ha="center", fontsize=8, color=GREY)
fig.tight_layout(); save(fig, "fig6_learning")
print("ok", sorted(p.name for p in OUT.glob("*.png")))
