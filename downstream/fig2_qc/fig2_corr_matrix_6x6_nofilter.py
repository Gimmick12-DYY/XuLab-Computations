#!/usr/bin/env python
"""6x6 sc-vs-bulk Pearson without the 'preferred' summit filter: top 5,000 summits per reference by signalValue
(all summits), otherwise identical to corr_matrix_6x6 (30 groups, ZNF777 depth, 3 seeds). Values reused from
r2_05_pearson.py (r2p_corr_all_top5000_znf777.tsv). Plain r in cells, colour clipped at +-0.2; CTCF column labelled
HEK293 (ENCODE summits and BAM). Outputs OUT/fig2_draft/corr_matrix_6x6_nofilter.* and corr_matrix_6x6_filter_vs_nofilter.*"""
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
CELL = {"CTCF": "HEK293", "MAZ": "HEK293", "ZBTB7A": "HEK293", "ZIC2": "HEK293", "ZNF282": "HEK293T", "ZNF777": "HEK293"}
pe = OUT / "round2/pearson"
summ = pd.read_csv(pe / "r2p_summary.tsv", sep="\t")
loo = pd.read_csv(pe / "r2p_loo.tsv", sep="\t")
M = {"filter": ("preferred_top5000", "Preferred summits (own ChIP > 2x others), top 5,000"),
     "nofilter": ("all_top5000", "All summits, top 5,000 (no preferred filter)")}
LIM = 0.2


def get(setname):
    B = pd.read_csv(pe / f"r2p_corr_{setname}_znf777.tsv", sep="\t", index_col=0).values[6:, :6]
    r = summ[(summ.set == setname) & (summ.depth == "znf777")].iloc[0]
    pc = loo[(loo.set == setname) & (loo.depth == "znf777") & (loo.dropped == "CTCF")].perm_p_5x5.iloc[0]
    return B, r, pc


def draw(ax, B, title):
    im = ax.imshow(np.clip(B, -LIM, LIM), cmap=P.div_cmap(), vmin=-LIM, vmax=LIM)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{B[i, j]:+.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(B[i, j]) > 0.15 else P.neutral.ink)
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec=P.neutral.ink, lw=0.8))
    ax.set_xticks(range(6), [f"{t} ChIP\n({CELL[t]})" for t in TFS], rotation=90, fontsize=5.5)
    ax.set_yticks(range(6), [f"{t} sc" for t in TFS], fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=6)
    return im


B, r, pc = get(M["nofilter"][0])
fig, ax = plt.subplots(figsize=(3.1, 2.9))
im = draw(ax, B, f"{M['nofilter'][1]}\nown best {r.own_best_row}/6, p = {r.perm_p:.3f} (no CTCF {pc:.3f})")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Pearson r (colour clipped at ±0.2)", fontsize=6.5)
cb.ax.tick_params(labelsize=6, length=2)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OD / f"corr_matrix_6x6_nofilter.{ext}", dpi=600, bbox_inches="tight")
plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.1), layout="constrained")
for ax, k in zip(axs, ("filter", "nofilter")):
    Bk, rk, pk = get(M[k][0])
    im = draw(ax, Bk, f"{M[k][1]}\nown best {rk.own_best_row}/6, p = {rk.perm_p:.3f} (no CTCF {pk:.3f})")
    print(k, "own_best", rk.own_best_row, "p", rk.perm_p, "noCTCF", pk); print(pd.DataFrame(Bk, TFS, TFS).round(2))
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Pearson r (colour clipped at ±0.2)", fontsize=6.5)
for ext in ("png", "pdf", "svg"):
    fig.savefig(OD / f"corr_matrix_6x6_filter_vs_nofilter.{ext}", dpi=600)
