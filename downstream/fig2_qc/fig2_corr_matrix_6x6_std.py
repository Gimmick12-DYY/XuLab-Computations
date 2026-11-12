#!/usr/bin/env python
"""Display-only standardisations of the 6x6 sc-vs-bulk correlation matrix (same data as corr_matrix_6x6:
round-3 summit-group Pearson, ZNF777 depth, 3 seeds). The transform is written in each title and colour-bar label.
  S1 raw r, colour range clipped at +-0.2      S2 row z-score (each sc track across the 6 references)
  S3 column z-score (each ChIP across the 6 sc tracks)      S4 double-centred (row and column means removed), / max|value|
Outputs OUT/fig2_draft/corr_matrix_6x6_std_S{1..4}.{png,pdf,svg}, corr_matrix_6x6_std_contact.{png,pdf}, _rows.tsv."""
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
OD = OUT / "fig2_draft"
CELL = {"CTCF": "HEK293/HEK293T", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
B = pd.read_csv(OUT / "round2/pearson/r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0).values[6:, :6]
rz = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
cz = (B - B.mean(0, keepdims=True)) / B.std(0, keepdims=True)
dc = B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean()
dc = dc / np.abs(dc).max()
V = {"S1": (B, 0.2, "Pearson r (colour clipped at ±0.2)", "Pearson r, colour clipped at ±0.2", "{:+.2f}"),
     "S2": (rz, 2.0, "row z-score", "Row z-score (each sc track across the 6 references)", "{:+.1f}"),
     "S3": (cz, 2.0, "column z-score", "Column z-score (each ChIP across the 6 sc tracks)", "{:+.1f}"),
     "S4": (dc, 1.0, "double-centred r / max |value|", "Double-centred (row and column means removed)", "{:+.2f}")}


def draw(ax, M, lim, fmt):
    im = ax.imshow(np.clip(M, -lim, lim), cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=5.5,
                    color="white" if abs(M[i, j]) > 0.6 * lim else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90, fontsize=5.5)
    ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


rows = []
for k, (M, lim, cbl, title, fmt) in V.items():
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    im = draw(ax, M, lim, fmt)
    ax.set_title(title, fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbl, fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.4)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OD / f"corr_matrix_6x6_std_{k}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    own_row = [t for i, t in enumerate(TFS) if np.argmax(M[i]) == i]
    own_col = [t for j, t in enumerate(TFS) if np.argmax(M[:, j]) == j]
    rows.append(dict(version=k, transform=title, own_brightest_in_row=",".join(own_row), n_rows=len(own_row),
                     own_brightest_in_column=",".join(own_col), n_cols=len(own_col)))
fig, axs = plt.subplots(1, 4, figsize=(12.5, 3.3), layout="constrained")
for ax, (k, (M, lim, cbl, title, fmt)) in zip(axs, V.items()):
    im = draw(ax, M, lim, fmt)
    ax.set_title(f"{k}: {title}", fontsize=6.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbl, fontsize=6); cb.ax.tick_params(labelsize=5.5, length=2)
for ext in ("png", "pdf"):
    fig.savefig(OD / f"corr_matrix_6x6_std_contact.{ext}", dpi=400)
pd.DataFrame(rows).to_csv(OD / "corr_matrix_6x6_std_rows.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string())
