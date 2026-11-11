#!/usr/bin/env python
"""Figure 2 correlation matrix, Paired-Tag style: HEK293T ChIP (CTCF, ZNF282) and sc raw tracks (ZNF777 depth),
Pearson over summit groups (values from hek293t_2x2/corr4x4_all.tsv), average-linkage clustering with dendrograms
on top and left, labels on the right and bottom, no cell text. Outputs OUT/fig2_draft/corr_matrix.{png,pdf,svg}."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from fig2_common import OUT, ROOT

for f in (ROOT.parent / "fonts/arial").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
df = pd.read_csv(OUT / "hek293t_2x2/corr4x4_all.tsv", sep="\t", index_col=0)
lab = {"HEK293T ChIP CTCF": "CTCF ChIP", "HEK293T ChIP ZNF282": "ZNF282 ChIP", "sc CTCF": "CTCF sc", "sc ZNF282": "ZNF282 sc"}
R = df.values
Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
fig = plt.figure(figsize=(2.6, 2.2))
x0, y0, w = 0.2, 0.08, 0.5
axt = fig.add_axes([x0, y0 + w, w, 0.14])
axl = fig.add_axes([x0 - 0.15, y0, 0.14, w])
dt = dendrogram(Z, ax=axt, no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
dendrogram(Z, ax=axl, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
axl.invert_yaxis()
for a in (axt, axl):
    a.axis("off")
for l in axt.collections + axl.collections:
    l.set_linewidth(0.6)
o = dt["leaves"]
ax = fig.add_axes([x0, y0, w, w])
im = ax.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1, vmax=1, aspect="auto")
names = [lab[df.index[i]] for i in o]
ax.set_xticks(range(4), names, rotation=90)
ax.yaxis.tick_right()
ax.set_yticks(range(4), names)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    if t.get_text().endswith("sc"):
        t.set_fontweight("bold")
ax.tick_params(length=0, labelsize=7)
for s in ax.spines.values():
    s.set_visible(False)
cax = fig.add_axes([x0 + w + 0.27, y0 + 0.05, 0.03, w - 0.05])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Pearson correlation", fontsize=7)
cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_linewidth(0.4)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / "fig2_draft" / f"corr_matrix.{ext}", dpi=600, bbox_inches="tight")
print(pd.DataFrame(R[np.ix_(o, o)], names, names).round(2))
