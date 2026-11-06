"""Basic TF x TF tendency heatmaps from ALL cliques (no loop filter), blacklist removed.
A: raw sharing = shared clique regions / smaller TF's clique regions (overlap coefficient).
B: sharing vs accessibility-matched control, log2 (obs + 1)/(exp + 1).
Both in the same TF order (average-linkage clustering of A). Side bar = clique regions per TF."""
import sys
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
D = Path(sys.argv[1]); OUT = Path(sys.argv[2])
pt = pd.read_csv(D / "tf_pair_shared_regions.tsv", sep="\t")
hr = pd.read_csv(D / "hub_regions.tsv", sep="\t")
size = Counter(t for s in hr.tfs for t in s.split(","))
tfs = sorted(size)
ix = {t: i for i, t in enumerate(tfs)}
n = len(tfs)
OC = np.full((n, n), np.nan); ENR = np.full((n, n), np.nan); SH = np.zeros((n, n))
for r in pt.itertuples():
    if r.tf1 not in ix or r.tf2 not in ix:
        continue
    i, j = ix[r.tf1], ix[r.tf2]
    oc = r.shared_clique_regions / min(size[r.tf1], size[r.tf2])
    OC[i, j] = OC[j, i] = oc
    ENR[i, j] = ENR[j, i] = np.log2(r.obs_over_expected)
    SH[i, j] = SH[j, i] = r.shared_clique_regions
np.fill_diagonal(OC, 1.0)
X = np.nan_to_num(OC)
Dm = 1 - X
np.fill_diagonal(Dm, 0)
order = leaves_list(linkage(squareform(np.clip(Dm, 0, None), checks=False), "average"))
names = [tfs[i] for i in order]
pd.DataFrame(OC[np.ix_(order, order)], index=names, columns=names).to_csv(OUT / "tf_tendency_overlap_coefficient.tsv", sep="\t")
pd.DataFrame(ENR[np.ix_(order, order)], index=names, columns=names).to_csv(OUT / "tf_tendency_log2_obs_over_exp.tsv", sep="\t")
COMPLEX = [("EZH2", "MTF2"), ("MAZ", "ZNF143"), ("WIZ", "ZNF644"), ("RBBP4", "MBD3"), ("RBBP7", "MBD3"), ("RBBP4", "RBBP7")]


def draw(M, fname, title, cmap, vmin, vmax, cbl, mask_low=True):
    Mo = M[np.ix_(order, order)].copy()
    if mask_low:
        Mo[SH[np.ix_(order, order)] < 3] = np.nan
    np.fill_diagonal(Mo, np.nan)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(figsize=(16, 15))
    im = ax.imshow(Mo, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    div = make_axes_locatable(ax)
    axs = div.append_axes("right", size="7%", pad=0.08, sharey=ax)
    cax = div.append_axes("right", size="2%", pad=0.7)
    ax.set_facecolor("#EEEEEE")
    ax.set_xticks(range(n), names, rotation=90, fontsize=9)
    ax.set_yticks(range(n), names, fontsize=9)
    pos = {t: k for k, t in enumerate(names)}
    for a, b in COMPLEX:
        if a in pos and b in pos:
            for i, j in ((pos[a], pos[b]), (pos[b], pos[a])):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=2,
                                       ec="#2E7D32" if "MBD3" in (a, b) or {a, b} == {"RBBP4", "RBBP7"} else "#111"))
    axs.barh(range(n), [size[t] for t in names], color="#7C8693")
    axs.set_xscale("log"); axs.tick_params(axis="y", left=False, labelleft=False)
    axs.set_xlabel("clique\nregions", fontsize=10)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbl)
    ax.set_title(title, loc="left", fontsize=14)
    fig.savefig(OUT / f"{fname}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{fname}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", fname)


draw(OC, "figA_tf_tendency_raw_overlap", "TF × TF tendency to share clique regions (all cliques, blacklist removed)\n"
     "shared clique regions ÷ the smaller TF's clique regions · grey: <3 shared · boxes: known complexes (green = NuRD)",
     "Reds", 0, np.nanpercentile(OC[SH >= 3], 98), "overlap coefficient")
v = np.nanpercentile(np.abs(ENR[SH >= 3]), 98)
draw(ENR, "figB_tf_tendency_vs_control", "TF × TF sharing vs accessibility-matched control (all cliques, blacklist removed)\n"
     "same TF order as the raw-overlap heatmap · grey: <3 shared · boxes: known complexes (green = NuRD)",
     "RdBu_r", -v, v, "log2 (observed + 1) / (expected + 1)")
