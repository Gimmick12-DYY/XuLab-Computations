#!/usr/bin/env python
"""Round 2 figures (palette C, Arial 7 pt) -> OUT/round2/figs/.
  r2_profile_grid_<set>      6x6 pool-normalised +-1 kb profiles (rows sc TF, cols reference summits)
  r2_D_heatmap_<set>         double-centred L (full depth and ZNF777 depth), diverging at 0
  r2_panel_rank_<set>        L of all 79 TFs per reference (own TF in ink)
Usage: r2_04_plots.py [set ...]   (default: unique_top2000 preferred_top2000 all_top2000)
"""
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
INK, GREY = P.neutral.ink, P.neutral.light
R2 = OUT / "round2"
fd = R2 / "figs"
fd.mkdir(exist_ok=True)
NT = len(TFS)
sets = sys.argv[1:] or ["unique_top2000", "preferred_top2000", "all_top2000"]
pr = np.load(R2 / "r2_profiles.npz", allow_pickle=True)
H, tf79, tot, psets, edges = pr["H"], list(pr["tf79"]), pr["tot79"], list(pr["sets"]), pr["edges"]
x = (edges[:-1] + edges[1:]) / 2
summ = pd.read_csv(R2 / "r2_summary.tsv", sep="\t")
ranks = pd.read_csv(R2 / "r2_panel_rank.tsv", sep="\t")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(fd / f"{name}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def smooth(v, k=15):
    return np.convolve(v, np.ones(k) / k, mode="same")


for s in sets:
    typ = s.split("_")[0]
    a = psets.index(typ)
    Hp = H[:, a].sum(0)
    fig, axs = plt.subplots(NT, NT, figsize=(7.0, 6.2), sharex=True, sharey=True)
    for i, t in enumerate(TFS):
        k = tf79.index(t)
        for j, r in enumerate(TFS):
            own = H[k, a, j] / tot[k]
            pool = (Hp[j] - H[k, a, j]) / (tot.sum() - tot[k])
            v = np.log2(smooth(own + 1e-12) / smooth(pool + 1e-12))
            ax = axs[i, j]
            ax.axhline(0, color=GREY, lw=0.4)
            ax.plot(x, v, color=INK if i == j else P.neutral.mid, lw=0.9 if i == j else 0.6)
            if i == j:
                ax.set_facecolor(P.highlight)
            if i == 0:
                ax.set_title(f"{r} summits", fontsize=6)
            if j == 0:
                ax.set_ylabel(f"sc {t}", fontsize=6)
            if i == NT - 1:
                ax.set_xticks([-1000, 0, 1000], ["-1 kb", "0", "+1 kb"])
    fig.supylabel("log2 (sc TF / 79-TF pool), per-bin share", fontsize=7)
    fig.suptitle(f"{typ} top-2000 summits per reference, 10 bp bins, 150 bp running mean", fontsize=7)
    fig.tight_layout()
    save(fig, f"r2_profile_grid_{typ}_top2000")

    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.6), layout="constrained")
    for ax, depth in zip(axs, ("full", "znf777")):
        D = pd.read_csv(R2 / f"r2_D_{s}_{depth}.tsv", sep="\t", index_col=0).values
        lim = max(0.05, np.abs(D).max())
        im = ax.imshow(D, cmap=P.div_cmap(), vmin=-lim, vmax=lim)
        for ii in range(NT):
            for jj in range(NT):
                ax.text(jj, ii, f"{D[ii, jj]:+.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(D[ii, jj]) > 0.6 * lim else INK)
            ax.add_patch(Rectangle((ii - .5, ii - .5), 1, 1, fill=False, ec=INK, lw=0.8))
        ax.set_xticks(range(NT), TFS, rotation=90)
        ax.set_yticks(range(NT), TFS)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        r = summ[(summ.set == s) & (summ.depth == depth)].iloc[0]
        ax.set_title(f"{'full depth' if depth == 'full' else 'ZNF777 depth'}\n"
                     f"diag−off {r.diag_minus_off:+.3f}, p = {r.perm_p:.3f}\n"
                     f"own best: rows {r.own_best_row}/6, cols {r.own_best_col}/6", fontsize=6)
        ax.set_xlabel("Reference summits")
        ax.set_ylabel("sc track")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="double-centred log2 ratio")
    save(fig, f"r2_D_heatmap_{s}")

    rk = ranks[ranks.set == s].set_index("reference")
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for j, t in enumerate(TFS):
        ax.scatter([j], [rk.loc[t, "rank_full_of79"]], s=16, color=INK, label="full depth" if j == 0 else None)
        ax.scatter([j + 0.15], [rk.loc[t, "rank_dm_of79"]], s=16, facecolor="none", edgecolor=INK, lw=0.6,
                   label="ZNF777 depth" if j == 0 else None)
    ax.axhline(4, color=GREY, lw=0.5, ls=":")
    ax.set_ylim(80, 0)
    ax.set_xticks(range(NT), TFS, rotation=45, ha="right")
    ax.set_ylabel("Rank of own TF among 79")
    ax.set_title(f"Panel rank at own summits ({s.replace('_', ' ')})")
    ax.legend(fontsize=5, loc="lower right")
    fig.tight_layout()
    save(fig, f"r2_panel_rank_{s}")
print("done")
