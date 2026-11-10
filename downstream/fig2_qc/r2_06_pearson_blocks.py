#!/usr/bin/env python
"""6x6 sc-vs-bulk Pearson blocks (diagonal outlined) from the summit-group Pearson matrices
(OUT/round2/pearson/r2p_corr_<set>_<depth>.tsv, made by r2_05_pearson.py), with the stats from r2p_summary/r2p_loo.
Writes OUT/round2/pearson/figs/r2p_block_<set>_<depth>.png/.pdf for depth znf777 and full."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from fig2_common import OUT, ROOT, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
OP = OUT / "round2" / "pearson"
NT = len(TFS)
summ = pd.read_csv(OP / "r2p_summary.tsv", sep="\t")
loo = pd.read_csv(OP / "r2p_loo.tsv", sep="\t")
for r in summ[summ.depth.isin(["znf777", "full"])].itertuples():
    R = pd.read_csv(OP / f"r2p_corr_{r.set}_{r.depth}.tsv", sep="\t", index_col=0).values
    blk = R[NT:, :NT]
    pc = loo[(loo.set == r.set) & (loo.depth == r.depth) & (loo.dropped == "CTCF")].perm_p_5x5.iloc[0]
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    lim = max(0.1, np.abs(blk).max())
    im = ax.imshow(blk, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for a in range(NT):
        for b in range(NT):
            ax.text(b, a, f"{blk[a, b]:+.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(blk[a, b]) > 0.6 * lim else INK)
        ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.8))
    ax.set_xticks(range(NT), [f"Bulk {t}" for t in TFS], rotation=90)
    ax.set_yticks(range(NT), [f"sc {t}" for t in TFS])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Pearson r")
    dep = "ZNF777 depth" if r.depth == "znf777" else "full depth"
    ax.set_title(f"{r.set.replace('_', ' ')} summit groups, {dep}\nrows {r.own_best_row}/6, cols {r.own_best_col}/6, "
                 f"p = {r.perm_p:.3f} (no CTCF p = {pc:.3f})", fontsize=6)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OP / "figs" / f"r2p_block_{r.set}_{r.depth}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
print("done")
