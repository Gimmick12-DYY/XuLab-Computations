#!/usr/bin/env python
"""Re-render the recommended Fig 2 QC Pearson panels with neutral titles (no 'diagonal' wording, no
row/column normalisation). Inputs: existing matrices from r2_05_pearson.py, r3_heldout.py and r2_02_stats.py.
Own-reference cells are outlined; values are plain Pearson r (or plain double-centred log2 for s2).
Outputs OUT/fig2_recommended/*.png/.pdf."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram

from fig2_common import OUT, ROOT, TFS

sys.path.insert(0, str(ROOT / "figures_paletteC"))
import regulome_palette as rp  # noqa: E402

P = rp.use()
INK = P.neutral.ink
NT = len(TFS)
FD = OUT / "fig2_recommended"
FD.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FD / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def block(M, title, name, cbar="Pearson r", xl="Bulk", yl="sc"):
    fig, ax = plt.subplots(figsize=(2.7, 2.5))
    lim = max(0.1, np.abs(M).max())
    im = ax.imshow(M, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
    for a in range(NT):
        for b in range(NT):
            ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(M[a, b]) > 0.6 * lim else INK)
        ax.add_patch(Rectangle((a - .5, a - .5), 1, 1, fill=False, ec=INK, lw=0.6))
    ax.set_xticks(range(NT), [f"{xl} {t}" for t in TFS], rotation=90)
    ax.set_yticks(range(NT), [f"{yl} {t}" for t in TFS])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cbar)
    ax.set_title(title, fontsize=6)
    fig.tight_layout()
    save(fig, name)


pe = OUT / "round2" / "pearson"
R = pd.read_csv(pe / "r2p_corr_preferred_top5000_znf777.tsv", sep="\t", index_col=0).values
block(R[NT:, :NT], "sc vs bulk Pearson, summit groups, ZNF777 depth\n(own-reference cells outlined)",
      "b1_pearson_6x6_ZNF777depth")
z = np.load(pe / "r2p_link_preferred_top5000_znf777.npz", allow_pickle=True)
labs = [l.replace("bulk_", "Bulk ").replace("sc_", "sc ") for l in z["labs"]]
fig = plt.figure(figsize=(3.4, 3.4))
axd = fig.add_axes([0.27, 0.78, 0.53, 0.11])
d = dendrogram(z["Z"], ax=axd, no_labels=True, color_threshold=0, above_threshold_color=INK)
axd.axis("off")
o = d["leaves"]
axh = fig.add_axes([0.27, 0.22, 0.53, 0.56])
im = axh.imshow(R[np.ix_(o, o)], cmap=P.div_cmap(), vmin=-1, vmax=1, aspect="auto")
axh.set_xticks(range(len(o)), [labs[k] for k in o], rotation=90)
axh.set_yticks(range(len(o)), [labs[k] for k in o])
for tl in axh.get_xticklabels() + axh.get_yticklabels():
    tl.set_color(INK if tl.get_text().startswith("Bulk") else P.arms.raw)
axh.tick_params(length=0)
fig.colorbar(im, cax=fig.add_axes([0.83, 0.22, 0.025, 0.25]), label="Pearson r")
fig.suptitle("Bulk and sc tracks, Pearson over summit groups, ZNF777 depth", fontsize=6, y=0.98)
save(fig, "b2_pearson_12x12_clustered_ZNF777depth")
H = pd.read_csv(OUT / "round3_pearson/heldout/test_corr_published.tsv", sep="\t", index_col=0).values
block(H[NT:, :NT], "sc vs bulk Pearson, held-out chromosomes\n(even autosomes + chrX), ZNF777 depth",
      "b3_heldout_test_6x6")
sm = pd.read_csv(pe / "r2p_summary.tsv", sep="\t")
v = sm[sm.set == "preferred_top5000"].set_index("depth")
fig, ax = plt.subplots(figsize=(3.0, 2.2))
x = [.25, .5, 1.0]
for t_i, t in enumerate(TFS):
    ys = []
    for dname in ("znf777_25pct", "znf777_50pct", "znf777"):
        M = pd.read_csv(pe / f"r2p_corr_preferred_top5000_{dname}.tsv", sep="\t", index_col=0).values[NT:, :NT] \
            if (pe / f"r2p_corr_preferred_top5000_{dname}.tsv").exists() else None
        ys.append(np.nan if M is None else M[t_i, t_i] - np.delete(M[t_i], t_i).mean())
    ax.plot(x, ys, marker="o", ms=3, lw=0.8, color=INK if t == "CTCF" else P.neutral.mid,
            label=t if t == "CTCF" else ("other 5 TFs" if t_i == 1 else None))
ax.axhline(0, color=P.neutral.light, lw=0.5)
ax.set_xticks(x, ["25%", "50%", "100%"])
ax.set_xlabel("Fraction of ZNF777 depth (420,339 reads)")
ax.set_ylabel("own-reference r − mean other r")
ax.set_title("Per TF, summit groups", fontsize=6)
ax.legend(fontsize=5)
fig.tight_layout()
save(fig, "b4_own_minus_other_vs_depth")
Dm = pd.read_csv(OUT / "round2/r2_D_unique_top5000_znf777.tsv", sep="\t", index_col=0).values
block(Dm, "Summit centre/flank vs 79-TF pool, unique top-5000 summits,\nZNF777 depth (double-centred log2)",
      "s2_summit_centre_flank_6x6", cbar="double-centred log2 ratio", xl="", yl="sc")
print("done")
