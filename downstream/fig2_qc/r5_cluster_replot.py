#!/usr/bin/env python
"""Re-draw the clustered 4x4 HEK293T ChIP / sc heatmap from corr4x4_all.tsv with a layout that fits the labels."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from fig2_common import OUT, ROOT

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
O5 = OUT / "hek293t_2x2"
df = pd.read_csv(O5 / "corr4x4_all.tsv", sep="\t", index_col=0)
R, labs = df.values, list(df.index)
Z = linkage(squareform(np.clip(1 - R, 0, None), checks=False), "average")
fig = plt.figure(figsize=(3.2, 3.4))
axd = fig.add_axes([0.38, 0.77, 0.40, 0.12])
d = dendrogram(Z, ax=axd, no_labels=True, color_threshold=0, above_threshold_color=P.neutral.ink)
axd.axis("off")
o = d["leaves"]
axh = fig.add_axes([0.38, 0.37, 0.40, 0.40])
im = axh.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1, vmax=1, aspect="auto")
axh.set_xticks(range(4), [labs[k] for k in o], rotation=90)
axh.set_yticks(range(4), [labs[k] for k in o])
for tl in axh.get_xticklabels() + axh.get_yticklabels():
    tl.set_color(P.neutral.ink if tl.get_text().startswith("HEK") else P.arms.raw)
axh.tick_params(length=0)
for a in range(4):
    for b in range(4):
        v = R[o[a], o[b]]
        axh.text(b, a, f"{v:+.2f}", ha="center", va="center", fontsize=5, color="white" if abs(v) > 0.6 else P.neutral.ink)
fig.colorbar(im, cax=fig.add_axes([0.83, 0.37, 0.03, 0.22]), label="Pearson r")
fig.suptitle("HEK293T ChIP and sc tracks, Pearson over summit groups,\nZNF777 depth", fontsize=6, y=0.98)
for ext in ("png", "pdf"):
    fig.savefig(O5 / f"cluster_4x4.{ext}", dpi=600 if ext == "png" else None)
