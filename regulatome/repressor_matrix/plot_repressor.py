# Repressor TF x TF matrices (palette C): fixed order grouped by complex, module side strip, library-group strip, RdBu_r
# centred at 0. Individual SVGs per unit x (Pearson, SpQN) and one combined 4-panel SpQN figure (SVG + 300 dpi PNG).
#   usage: python3 plot_repressor.py <dir with pearson_*.tsv / spqn_*.tsv / tf_meta.tsv>
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list
D = sys.argv[1]
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.5, "svg.fonttype": "none", "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6, "savefig.dpi": 600, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A"})
ORDER = [("PRC2", ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"]), ("NuRD", ["MBD3"]), ("HP1/ChAHP", ["ADNP", "HP1", "CHAMP1"]),
         ("G9a/WIZ", ["DR1", "THAP6", "WIZ", "ZNF644", "SRCAP"]), ("other", ["SETDB1"]),
         ("other", ["NEUROD4", "SOX12", "ZNF79", "ZNF606", "ZNF777", "ZNF581"]), ("reference", ["CTCF", "NFYA"])]
MOD_COL = {"PRC2": "#1F3F8F", "NuRD": "#EB6834", "HP1/ChAHP": "#FF9DA7", "G9a/WIZ": "#B07AA1", "other": "#BDBDBD", "reference": "#FFFFFF"}
LIB_COL = {"high": "#3F3F3F", "low": "#BDBDBD"}
UNITS = [("active_elements", "Active elements"), ("reprpc_domains", "ReprPC domains"), ("bins_50kb", "50 kb bins"),
         ("h3k9me3_peaks", "H3K9me3 peaks (HEK293)")]
meta = pd.read_csv(f"{D}/tf_meta.tsv", sep="\t").set_index("tf")
order = [t for _, ts in ORDER for t in ts]; modof = {t: m for m, ts in ORDER for t in ts}
NURD_ALSO = {"RBBP4", "RBBP7"}; WIZ_ALSO = {"CHAMP1"}  # shared members: second strip colour

def panel(ax, R, title, cax=None, ordr=None, seps=True):
    global order
    base = order; order = ordr or base
    A = R.loc[order, order].values.astype(float); np.fill_diagonal(A, np.nan)
    lim = np.nanmax(np.abs(A)); norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#F2F2F2")
    im = ax.imshow(A, cmap=cmap, norm=norm, interpolation="none")
    n = len(order)
    ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(order, rotation=90); ax.set_yticklabels(order)
    ax.tick_params(length=1.5, width=0.4, pad=1)
    for i, t in enumerate(order):  # module strip (left of rows, top of columns) and library strip
        ax.add_patch(Rectangle((-2.6, i - 0.5), 0.9, 1, color=MOD_COL[modof[t]], ec="#1A1A1A" if modof[t] == "reference" else "none", lw=0.3, clip_on=False))
        if t in NURD_ALSO: ax.add_patch(Rectangle((-2.6, i - 0.5), 0.45, 1, color=MOD_COL["NuRD"], clip_on=False))
        if t in WIZ_ALSO: ax.add_patch(Rectangle((-2.6, i - 0.5), 0.45, 1, color=MOD_COL["G9a/WIZ"], clip_on=False))
        ax.add_patch(Rectangle((-1.6, i - 0.5), 0.9, 1, color=LIB_COL[meta.loc[t, "library"]], clip_on=False))
        ax.add_patch(Rectangle((i - 0.5, -2.6), 1, 0.9, color=MOD_COL[modof[t]], ec="#1A1A1A" if modof[t] == "reference" else "none", lw=0.3, clip_on=False))
        ax.add_patch(Rectangle((i - 0.5, -1.6), 1, 0.9, color=LIB_COL[meta.loc[t, "library"]], clip_on=False))
    k = 0
    for _, ts in (ORDER if seps else []):  # block separators (fixed order only)
        k += len(ts)
        if k < n: ax.axhline(k - 0.5, color="#1A1A1A", lw=0.4); ax.axvline(k - 0.5, color="#1A1A1A", lw=0.4)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, pad=24)
    for s_ in ax.spines.values(): s_.set_linewidth(0.5)
    order = base
    return im, lim

def legend(fig, x, y):
    """one horizontal row of legend swatches starting at figure coords (x, y)"""
    items = [(MOD_COL[m], m) for m in ("PRC2", "NuRD", "HP1/ChAHP", "G9a/WIZ", "other")] + [(LIB_COL["high"], "library high"), (LIB_COL["low"], "library low")]
    xx = x
    for c, l in items:
        fig.patches.append(Rectangle((xx, y), 0.012, 0.012, color=c, transform=fig.transFigure, figure=fig))
        fig.text(xx + 0.016, y + 0.001, l, fontsize=6); xx += 0.03 + 0.0075 * len(l)
    fig.text(xx + 0.01, y + 0.001, "G9a/WIZ = broad-domain module (WIZ/ZNF644 group); split strip = shared member", fontsize=5)

for u, lab in UNITS:
    for kind in ("pearson", "spqn"):
        R = pd.read_csv(f"{D}/{kind}_{u}.tsv", sep="\t", index_col=0)
        fig = plt.figure(figsize=(4.2, 3.7)); ax = fig.add_axes([0.2, 0.2, 0.62, 0.66])
        im, lim = panel(ax, R, f"{lab}: {'SpQN' if kind == 'spqn' else 'Pearson'}")
        cax = fig.add_axes([0.86, 0.5, 0.015, 0.25]); cb = fig.colorbar(im, cax=cax); cb.set_label("r" if kind == "pearson" else "SpQN r", fontsize=6)
        cb.ax.tick_params(labelsize=5, width=0.3, length=1.5); cb.outline.set_linewidth(0.3)
        fig.savefig(f"{D}/repressor_{kind}_{u}.svg"); plt.close(fig)
        sub = R.loc[order, order].values.astype(float)
        cl = [order[i] for i in leaves_list(linkage(1 - sub[np.triu_indices(len(order), 1)], "average"))]
        fig = plt.figure(figsize=(4.2, 3.7)); ax = fig.add_axes([0.2, 0.2, 0.62, 0.66])
        im, lim = panel(ax, R, f"{lab}: {'SpQN' if kind == 'spqn' else 'Pearson'}, clustered", ordr=cl, seps=False)
        cax = fig.add_axes([0.86, 0.5, 0.015, 0.25]); cb = fig.colorbar(im, cax=cax); cb.set_label("r" if kind == "pearson" else "SpQN r", fontsize=6)
        cb.ax.tick_params(labelsize=5, width=0.3, length=1.5); cb.outline.set_linewidth(0.3)
        fig.savefig(f"{D}/repressor_{kind}_{u}_clustered.svg"); plt.close(fig)
fig = plt.figure(figsize=(7.6, 7.6))
pos = [[0.09, 0.53, 0.34, 0.34], [0.53, 0.53, 0.34, 0.34], [0.09, 0.07, 0.34, 0.34], [0.53, 0.07, 0.34, 0.34]]
for (u, lab), p, letter in zip(UNITS, pos, "abcd"):
    R = pd.read_csv(f"{D}/spqn_{u}.tsv", sep="\t", index_col=0)
    ax = fig.add_axes(p); im, lim = panel(ax, R, f"{lab} (SpQN)")
    cax = fig.add_axes([p[0] + p[2] + 0.008, p[1] + 0.12, 0.008, 0.12]); cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5, width=0.3, length=1.5); cb.outline.set_linewidth(0.3)
    fig.text(p[0] - 0.08, p[1] + p[3] + 0.04, letter, fontsize=9, fontweight="bold")
legend(fig, 0.03, 0.975)
fig.savefig(f"{D}/repressor_matrix_combined_spqn.svg"); fig.savefig(f"{D}/repressor_matrix_combined_spqn.png", dpi=300); plt.close(fig)
# within-block vs between-block mean r (among the plotted TFs)
BL = {"PRC2": ["EZH2", "MTF2", "PCGF6", "RBBP4", "RBBP7"], "NuRD": ["MBD3", "RBBP4", "RBBP7"], "ChAHP": ["ADNP", "HP1", "CHAMP1"],
      "broad-domain (WIZ/ZNF644)": ["DR1", "THAP6", "CHAMP1", "WIZ", "ZNF644", "SRCAP"], "low-depth set": ["NEUROD4", "SOX12", "ZNF79", "ZNF606", "ZNF777", "ZNF581"]}
rows = []
for u, lab in UNITS:
    for kind in ("pearson", "spqn"):
        R = pd.read_csv(f"{D}/{kind}_{u}.tsv", sep="\t", index_col=0).loc[order, order]
        for b, mem in BL.items():
            w = [R.loc[a, c] for i, a in enumerate(mem) for c in mem[i + 1:]]
            bt = [R.loc[a, c] for a in mem for c in order if c not in mem]
            rows.append(dict(unit=u, matrix=kind, block=b, within_mean_r=np.mean(w), between_mean_r=np.mean(bt), difference=np.mean(w) - np.mean(bt)))
pd.DataFrame(rows).to_csv(f"{D}/within_vs_between.tsv", sep="\t", index=False, float_format="%.3f")
print("written")
