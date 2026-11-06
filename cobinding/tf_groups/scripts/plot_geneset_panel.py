"""Slide figure: functional gene sets among multi-TF clique genes (>=3 TFs), before vs after peak-density matching
(palette C, linear axes). args: panel_dir outpng"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
from pathlib import Path
P = rp.use(); D = Path(sys.argv[1]); plt.rcParams.update({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.labelsize": 9, "legend.fontsize": 8})
B = pd.read_csv(D / "panel_pooled_geK.tsv", sep="\t"); B = B[B.k == 3].set_index("set")
C = pd.read_csv(D / "panel_pooled_density_matched.tsv", sep="\t").set_index("set")
order = C.sort_values("OE_ratio").index.tolist(); y = np.arange(len(order))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 5.2), sharey=True)
orr = B.OR.reindex(order); sig = B.q.reindex(order) < 0.05
a1.barh(y, orr - 1, left=1, color=[P.neutral.ink if s else P.neutral.light for s in sig], height=.62)
a1.axvline(1, color=P.neutral.dark, lw=.6); a1.set_yticks(y); a1.set_yticklabels(order); a1.set_xlabel("Odds ratio, clique genes (≥3 TFs) vs peak genes")
a1.set_title("Before density matching", loc="left", fontsize=9)
a1.legend(handles=[matplotlib.patches.Patch(color=P.neutral.ink, label="q < 0.05"), matplotlib.patches.Patch(color=P.neutral.light, label="n.s.")], frameon=False, loc="lower right")
c = C.reindex(order)
a2.barh(y, c.null_hi - c.null_lo, left=c.null_lo, color=P.neutral.pale, height=.62, label="95% range, random gene sets")
a2.scatter(c.OE_ratio, y, color=P.neutral.ink, s=22, zorder=3, label="Observed / expected")
a2.axvline(1, color=P.neutral.dark, lw=.6); a2.set_xlabel("Clique TFs observed / expected (density-matched)")
a2.set_title("After density matching: all q ≥ 0.18", loc="left", fontsize=9); a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=300)
