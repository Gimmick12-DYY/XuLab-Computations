# m55 Amendment 3: bar chart of the median number of genes per TF at each |score| cutoff (from the m55 table).
import sys, os; sys.path.insert(0, "/work/users/d/y/dyy12/envs/pptx_pydeps")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
B = "/work/users/d/y/dyy12/XuLab/regulatome/m55_genes_per_tf_by_score"; PL = f"{B}/plots"
T = pd.read_excel(f"{B}/out/genes_left_per_TF_by_score_cutoff.xlsx", sheet_name="summary over 63 TFs", dtype={"|score| cutoff": str})
cut = T["|score| cutoff"].tolist(); med = T["median"].values
assert cut == [f"{0.05 * k:.2f}" for k in range(1, 17)] and med.tolist() == [8402, 5838, 3749, 2194, 1195, 627, 320, 147, 70, 29, 10, 4, 1, 0, 0, 0], med
for mode in ("slide", "paper"):
    P = rp.use(); s = mode == "slide"
    if s: plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    fig, ax = plt.subplots(figsize=(12.9, 5.9) if s else (7.2, 2.6)); fig.subplots_adjust(left=0.08 if s else 0.08, right=0.99, bottom=0.14 if s else 0.18, top=0.95)
    x = np.arange(len(cut)); ax.bar(x, med, width=0.7, color=P.arms.raw)
    for xi, v in zip(x, med): ax.text(xi, v + med.max() * 0.012, f"{int(v):,}", ha="center", va="bottom", fontsize=12 if s else 5.5, color=P.neutral.ink)
    ax.set_xticks(x); ax.set_xticklabels(cut); ax.set_xlim(-0.6, len(cut) - 0.4); ax.set_ylim(0, med.max() * 1.08)
    ax.set_xlabel("gene score cutoff (|score| ≥)"); ax.set_ylabel("median number of genes per TF"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/median_genes_per_TF_by_score_{mode}.{ext}", dpi=300 if s else 600)
    plt.close(fig)
print("colour", rp.use().arms.raw); print("done")
