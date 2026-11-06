import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
O = Path(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 6
tfs = O.joinpath("tfs.txt").read_text().split("\n")
shr = np.load(O / "shrunk.npy"); Q = np.load(O / "q.npy"); E = np.load(O / "expected.npy")
size = pd.read_csv(O / "tf_clique_region_counts.tsv", sep="\t", index_col=0).clique_regions
keep = [i for i, t in enumerate(tfs) if size[t] >= 50 or t == "CTCF"]
small = [tfs[i] for i, t in enumerate(tfs) if size[t] < 50 and t != "CTCF"]
sig = (Q < 0.05) & (E >= 5) & np.isfinite(shr)
np.fill_diagonal(sig, False)
act = [i for i in keep if sig[i, keep].any()]          # TFs with >=1 significant pair
idle = [i for i in keep if i not in act]                # placed at the end, unclustered
S = np.where(sig, shr, 0.0)[np.ix_(act, act)]
np.fill_diagonal(S, np.nanmax(S))
Z = linkage(pdist(S, "correlation"), "average")
order = leaves_list(Z)
from scipy.cluster.hierarchy import fcluster
lab = fcluster(Z, t=K, criterion="maxclust")[order]
idx = [act[k] for k in order] + idle; names = [tfs[i] for i in idx]
show = shr[np.ix_(idx, idx)].copy()
mask = ~sig[np.ix_(idx, idx)]
show[mask] = np.nan
np.fill_diagonal(show, np.nan)
v = np.nanpercentile(np.abs(show), 98)
fig, ax = plt.subplots(figsize=(16, 15))
im = ax.imshow(show, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
ax.set_facecolor("#E6E6E6")
n = len(names)
ax.set_xticks(range(n), names, rotation=90, fontsize=9); ax.set_yticks(range(n), names, fontsize=9)
# outline the data-driven modules (contiguous blocks of the dendrogram cut) along the diagonal
mods = []
k0 = 0
for k in range(1, len(lab) + 1):
    if k == len(lab) or lab[k] != lab[k0]:
        if k - k0 >= 3:
            mods.append((k0, k))
        k0 = k
for m, (a0, a1) in enumerate(mods, 1):
    ax.add_patch(Rectangle((a0 - .5, a0 - .5), a1 - a0, a1 - a0, fill=False, lw=2.2, ec="#111"))
    ax.text(a1 - .5 + 0.3, a0 - .5, f"M{m}", fontsize=12, fontweight="bold", va="top", ha="left")
if idle:
    ax.axhline(len(act) - .5, color="#555", lw=1, ls=":"); ax.axvline(len(act) - .5, color="#555", lw=1, ls=":")
mod_txt = "; ".join(f"M{m}: " + ", ".join(names[a0:a1]) for m, (a0, a1) in enumerate(mods, 1))
div = make_axes_locatable(ax)
axs = div.append_axes("right", size="7%", pad=0.08, sharey=ax)
axs.barh(range(n), [size[t] for t in names], color="#7C8693"); axs.set_xscale("log"); axs.set_xticks([1e2, 1e4]); axs.set_xticklabels(["100", "10k"], fontsize=9); axs.minorticks_off()
axs.tick_params(axis="y", left=False, labelleft=False); axs.set_xlabel("clique\nregions", fontsize=10)
cax = div.append_axes("right", size="2%", pad=0.7)
fig.colorbar(im, cax=cax).set_label("shrunken log odds ratio (Mantel–Haenszel, stratified)")
open(O / "figC_hidden_tfs.txt", "w").write(", ".join(small) + "\n")
fig.savefig(O / "figC_tf_cooccupancy_stratified_OR.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(O / "figC_tf_cooccupancy_stratified_OR.svg", bbox_inches="tight", facecolor="white")
pd.DataFrame(show, index=names, columns=names).to_csv(O / "tf_cooccupancy_stratified_OR_matrix.tsv", sep="\t")
print("wrote", n, "TFs; hidden small:", small); print(mod_txt); open(O / "cooccupancy_modules.txt", "w").write(mod_txt.replace("; ", "\n") + "\n")
