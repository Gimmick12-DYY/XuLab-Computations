#!/usr/bin/env python
"""Standalone 6x6 sc-vs-bulk correlation matrix (the round-3 summit-group Pearson, ZNF777 depth, 3 seeds averaged;
r2_05_pearson.py output r2p_corr_preferred_top5000_znf777.tsv). Plain Pearson r, no row/column normalisation,
own-reference cells outlined, references labelled by cell line. Outputs OUT/fig2_draft/corr_matrix_6x6.{png,pdf,svg}."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Rectangle

from fig2_common import OUT, ROOT, TFS

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
CELL = {"CTCF": "HEK293/HEK293T", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
B = pd.read_csv(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0).values[6:, :6]
fig, ax = plt.subplots(figsize=(3.0, 2.7))
lim = 0.5
im = ax.imshow(B, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{B[i, j]:+.2f}", ha="center", va="center", fontsize=6, color="white" if abs(B[i, j]) > 0.3 else P.neutral.ink)
    ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90, fontsize=6)
ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=7)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Pearson correlation", fontsize=7); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / "fig2_draft" / f"corr_matrix_6x6.{ext}", dpi=600, bbox_inches="tight")
print(pd.DataFrame(B, TFS, TFS).round(2))
