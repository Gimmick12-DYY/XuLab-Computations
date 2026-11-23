# m53 figures: score spread vs cells (the target histograms were removed at the user's request, 2026-11-23).
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
B = "/work/users/d/y/dyy12/XuLab/regulatome/m53_stage1_percentile"; D = np.load(f"{B}/out/plot_data.npz", allow_pickle=True); PL = f"{B}/plots"; os.makedirs(PL, exist_ok=True)
def style(mode):
    P = rp.use()
    if mode == "slide": plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "axes.titlesize": 14, "axes.linewidth": 0.9, "lines.linewidth": 2.0, "xtick.major.width": 0.9, "ytick.major.width": 0.9})
    return P
def save(fig, name, mode):
    for ext in ("png", "pdf", "svg"): fig.savefig(f"{PL}/{name}_{mode}.{ext}", dpi=300 if mode == "slide" else 600)
    plt.close(fig); print("saved", name, mode, flush=True)
def spread(mode):
    P = style(mode); s = mode == "slide"; fig, ax = plt.subplots(figsize=(12.9, 5.9) if s else (3.5, 2.6), gridspec_kw=None); fig.subplots_adjust(left=0.08 if s else 0.17, right=0.98, bottom=0.14 if s else 0.18, top=0.95)
    ax.scatter(D["cells"], D["p95"], s=60 if s else 9, color=P.arms.raw, edgecolor="none"); ax.set_xlabel("top-25% cells per TF"); ax.set_ylabel("95th percentile of |score|"); ax.set_ylim(0, None); ax.set_xlim(0, None)
    save(fig, "P1_score_spread_vs_cells", mode)
for mode in ("slide", "paper"):
    spread(mode)
print("done")
