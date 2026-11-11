#!/usr/bin/env python
"""Clustered 12x12 (6 bulk ChIP + 6 sc raw) Pearson matrix, Paired-Tag style, from the round-3 summit-group analysis
(r2p_corr_preferred_top5000_znf777.tsv; ZNF777 depth, 3 seeds averaged). Average-linkage clustering on 1 - r,
dendrograms top and left, labels right and bottom, references labelled by cell line, no cell text.
Outputs OUT/fig2_draft/corr_matrix_12x12.{png,pdf,svg}."""
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
CELL = {"CTCF": "HEK293/HEK293T", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
df = pd.read_csv(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0)
tf_of = [l.replace("_", " ").split()[-1] for l in df.index]
lab = [(f"{t} ChIP ({CELL[t]})" if l.lower().startswith("bulk") else f"{t} sc") for l, t in zip(df.index, tf_of)]
R = df.values
Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
fig = plt.figure(figsize=(4.0, 3.6))
x0, y0, w = 0.18, 0.1, 0.55
axt = fig.add_axes([x0, y0 + w, w, 0.12]); axl = fig.add_axes([x0 - 0.13, y0, 0.12, w])
dt = dendrogram(Z, ax=axt, no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
dendrogram(Z, ax=axl, orientation="left", no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
axl.invert_yaxis()
for a in (axt, axl):
    a.axis("off")
    for l in a.collections:
        l.set_linewidth(0.6)
o = dt["leaves"]
ax = fig.add_axes([x0, y0, w, w])
im = ax.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1, vmax=1, aspect="auto")
nm = [lab[i] for i in o]
ax.set_xticks(range(12), nm, rotation=90); ax.yaxis.tick_right(); ax.set_yticks(range(12), nm)
for t in ax.get_xticklabels() + ax.get_yticklabels():
    if t.get_text().endswith(" sc"):
        t.set_fontweight("bold")
ax.tick_params(length=0, labelsize=6)
for s in ax.spines.values():
    s.set_visible(False)
cax = fig.add_axes([x0 + w + 0.36, y0 + 0.1, 0.025, w - 0.2])
cb = fig.colorbar(im, cax=cax); cb.set_label("Pearson correlation", fontsize=7)
cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / "fig2_draft" / f"corr_matrix_12x12.{ext}", dpi=600, bbox_inches="tight")
print(nm)
