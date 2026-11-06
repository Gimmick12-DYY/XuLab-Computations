"""Fig 3d (palette C): Jaccard overlap of clique regions between TFs (shared regions / union), with clique-region and cell-count side bars."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
E = Path(__file__).resolve().parent
OUT = ROOT / "figures_paletteC"; OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
R = pd.read_csv(E / "tf_x_tf_shared_clique_regions.tsv", sep="\t", index_col=0)
cells = pd.read_csv(E / "tf_cell_counts.tsv", sep="\t", header=None, index_col=0)[1]
n_reg = pd.Series(np.diag(R.values), index=R.index)
keep = [t for t in R.index if n_reg[t] >= 50 or t == "CTCF"]
R = R.loc[keep, keep]; n_reg = n_reg[keep]
J = R.values / (n_reg.values[:, None] + n_reg.values[None, :] - R.values)
np.fill_diagonal(J, 1.0)
J = pd.DataFrame(J, index=keep, columns=keep)
F = J.values.copy(); np.fill_diagonal(F, np.nan); F = np.where(np.isnan(F), np.nanmax(F), F)
Z = linkage(F, "ward")                       # Ward clustering of the TFs' Jaccard profiles
order = leaves_list(Z)
lab = fcluster(Z, 4, "maxclust")[order]
names = [keep[i] for i in order]
blocks, s0 = [], 0
for k in range(1, len(lab) + 1):
    if k == len(lab) or lab[k] != lab[s0]:
        blocks.append((s0, k)); s0 = k
show = J.loc[names, names].values.copy(); np.fill_diagonal(show, np.nan)
show[R.loc[names, names].values == 0] = np.nan
vmax = np.nanpercentile(show, 99)
fig, ax = plt.subplots(figsize=(120 * MM, 112 * MM))
im = ax.imshow(np.ma.masked_invalid(show), cmap=P.seq_cmap(), vmin=0, vmax=vmax, interpolation="nearest"); ax.set_facecolor(P.na)
n = len(names)
ax.set_xticks(range(n), names, rotation=90, fontsize=5); ax.set_yticks(range(n), names, fontsize=5)
ax.tick_params(length=1.5)
Jn = J.loc[names, names].values.copy(); np.fill_diagonal(Jn, np.nan)
glob = np.nanmedian(Jn)
mods = [(a0, a1) for a0, a1 in blocks if a1 - a0 >= 3 and np.nanmean(Jn[a0:a1, a0:a1]) > 2 * glob]
for m, (a0, a1) in enumerate(mods, 1):
    ax.add_patch(Rectangle((a0 - .5, a0 - .5), a1 - a0, a1 - a0, fill=False, lw=1.0, ec=P.neutral.ink))
    ax.text(a1 - .5 + 0.3, a0 - .5, f"B{m}", fontsize=7, fontweight="bold", va="top", ha="left")
blk = "\n".join(f"B{m} (mean J {np.nanmean(Jn[a0:a1, a0:a1]):.3f}): " + ", ".join(names[a0:a1]) for m, (a0, a1) in enumerate(mods, 1)) + f"\nall pairs median J {glob:.4f}\n"
div = make_axes_locatable(ax)
a1 = div.append_axes("right", size="7%", pad=0.05, sharey=ax)
a1.barh(range(n), [n_reg[t] for t in names], color=P.neutral.mid); a1.set_xscale("log")
a1.set_xticks([1e2, 1e4]); a1.set_xticklabels(["100", "10k"], fontsize=5); a1.minorticks_off()
a1.tick_params(axis="y", left=False, labelleft=False); a1.set_xlabel("clique\nregions")
a2 = div.append_axes("right", size="7%", pad=0.14, sharey=ax)
a2.barh(range(n), [cells.get(t, np.nan) for t in names], color=P.neutral.light); a2.set_xscale("log")
a2.set_xticks([1e3, 1e5]); a2.set_xticklabels(["1k", "100k"], fontsize=5); a2.minorticks_off()
a2.tick_params(axis="y", left=False, labelleft=False); a2.set_xlabel("cells")
cax = div.append_axes("right", size="2%", pad=0.45)
cb = fig.colorbar(im, cax=cax); cb.set_label("Jaccard index of clique regions (shared / union)"); cb.outline.set_linewidth(0.5)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"fig_tf_x_tf_jaccard_regions_paletteC.{ext}", bbox_inches="tight")
old = pd.read_csv(E / "tf_x_tf_jaccard_regions.tsv", sep="\t", index_col=0)
print(len(names), "TFs; order identical to existing:", list(old.index) == names, "; values identical:", np.allclose(old.loc[names, names].values, J.loc[names, names].values))
print("blocks identical:", blk == open(E / "jaccard_blocks.txt").read())
