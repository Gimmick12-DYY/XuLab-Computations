# m21_tf_tiers: Regulome overview of all 79 TFs by tier (evidence dot matrix, cells, target genes, TF x TF target-list sharing). Diagnostic. No new fits, no tier changes.
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 6, "axes.linewidth": .4, "xtick.major.width": .4, "ytick.major.width": .4, "xtick.major.size": 1.5, "ytick.major.size": 1.5})
RG = "/work/users/d/y/dyy12/XuLab/regulatome"; O = f"{RG}/m21_tf_tiers/out"; T = pd.read_csv(f"{O}/tf_evidence_table.tsv", sep="\t")
ORDER = ["Tier 1", "Tier 2", "Tier 3 (tested, none met)", "Tier 3 (not testable: noise floor only, not met)"]
NAME = {ORDER[0]: "Tier 1", ORDER[1]: "Tier 2", ORDER[2]: "Tier 3, tested", ORDER[3]: "Tier 3, not testable"}
TCOL = {ORDER[0]: "#6A51A3", ORDER[1]: "#2A9D8F", ORDER[2]: "#C9A227", ORDER[3]: "#BDBDBD"}; PROM, ENH = "#2C6FB7", "#E08214"
T["tk"] = T.tier.map({t: i for i, t in enumerate(ORDER)}); T = T.sort_values(["tk", "cells"], ascending=[True, False]).reset_index(drop=True); n = len(T); y = np.arange(n)
L = pd.read_csv(f"{RG}/m11_peak_regulome/out/long_q0.05.tsv.gz", sep="\t", usecols=["tf", "gene", "any_peak", "promoter_peak"])
prom = L[L.promoter_peak].groupby("tf").size().reindex(T.tf).fillna(0).values; dist = L[L.any_peak & ~L.promoter_peak].groupby("tf").size().reindex(T.tf).fillna(0).values
S = {t: set(g.gene) for t, g in L[L.any_peak].groupby("tf")}; J = np.zeros((n, n))
for i, a in enumerate(T.tf):
    for j, b in enumerate(T.tf):
        A_, B_ = S.get(a, set()), S.get(b, set()); J[i, j] = len(A_ & B_) / len(A_ | B_) if (A_ or B_) else np.nan
pd.DataFrame(J, index=T.tf, columns=T.tf).to_csv(f"{O}/jaccard_B1_tf_x_tf.tsv", sep="\t", float_format="%.4f")
MM = 1 / 25.4; W, H = 180, 205; fig = plt.figure(figsize=(W * MM, H * MM))
top, bot = 12, 34; rowsH = H - top - bot; Y0 = bot / H; HH = rowsH / H
def ax_at(x0, w): return fig.add_axes([x0 / W, Y0, w / W, HH])
axT = ax_at(14, 2); axA = ax_at(17, 24); axB = ax_at(46, 17); axC = ax_at(68, 22); axD = ax_at(96, 70)
for a in (axT, axA, axB, axC, axD): a.set_ylim(n - .5, -.5)
# tier bar + TF names
for i, t in enumerate(T.tier): axT.add_patch(plt.Rectangle((0, i - .5), 1, 1, color=TCOL[t], lw=0))
axT.set_xlim(0, 1); axT.set_xticks([]); axT.set_yticks(y); axT.set_yticklabels(T.tf, fontsize=5); axT.tick_params(axis="y", length=0, pad=1.5); [axT.spines[s].set_visible(False) for s in axT.spines]
bounds = [np.flatnonzero(T.tier.values == t) for t in ORDER]
for t, ix in zip(ORDER, bounds):
    fig.text(1 / W, Y0 + HH * (1 - (ix.mean() + .5) / n), f"{NAME[t]}\nn = {len(ix)}", rotation=90, ha="left", va="center", fontsize=6, color=TCOL[t] if t != ORDER[3] else "#777777", fontweight="bold")
# a: evidence dot matrix
cols = ["noise\nfloor", "own\nmotif", "bulk\nChIP", "KD\nfunction"]
def mark(ax, xc, i, state, size):
    if state == "na": ax.plot(xc, i, marker="x", ms=2.2, mew=.5, color="#9E9E9E")
    elif state == "met": ax.scatter(xc, i, s=size, c="k", lw=0)
    else: ax.scatter(xc, i, s=size, facecolors="white", edgecolors="#555555", lw=.5)
for i, r in T.iterrows():
    dsz = 3 + 40 * min(max(r.noise_floor_d, 0), .5) / .5 if pd.notna(r.noise_floor_d) else 0
    mark(axA, 0, i, "na" if pd.isna(r.noise_floor_d) else ("met" if r["line1_d_ge_0.04"] else "no"), dsz)
    msz = 3 + 40 * min(max((r.G1_enrich_vs_pooled78 if pd.notna(r.G1_enrich_vs_pooled78) else 1.2) - .5, 0), 1.5) / 1.5
    mark(axA, 1, i, "na" if not r.line2_testable else ("met" if r.line2_motif else "no"), msz)
    mark(axA, 2, i, "na" if not r.line3_testable else ("met" if r.line3_chip else "no"), 14)
    ft = pd.notna(r.B2xP1_OR) or pd.notna(r.B1xP4_OR); mark(axA, 3, i, "na" if not ft else ("met" if r.function_supported else "no"), 14)
axA.set_xlim(-.6, 3.6); axA.set_xticks(range(4)); axA.set_xticklabels(cols, fontsize=5.2); axA.xaxis.tick_top(); axA.set_yticks([]); [axA.spines[s].set_visible(False) for s in axA.spines]; axA.tick_params(axis="x", length=0)
for a in (axA, axB, axC):
    for ix in bounds[1:]:
        if len(ix): a.axhline(ix[0] - .5, color="#888888", lw=.4)
# b: cells (log)
axB.barh(y, T.cells, height=.7, color=[TCOL[t] for t in T.tier], lw=0); axB.set_xscale("log"); axB.set_yticks([]); axB.set_xlabel("HITAG cells per TF", labelpad=1); axB.spines[["top", "right", "left"]].set_visible(False)
# c: target genes, promoter vs distal-only
axC.barh(y, prom, height=.7, color=PROM, lw=0, label="promoter peak"); axC.barh(y, dist, left=prom, height=.7, color=ENH, lw=0, label="distal only"); axC.set_yticks([])
axC.set_xlabel("genes with a linked\nq005 peak", labelpad=1); axC.spines[["top", "right", "left"]].set_visible(False); axC.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
# d: Jaccard heatmap
im = axD.imshow(J, cmap="Greys", vmin=0, vmax=np.nanmax(J[~np.eye(n, dtype=bool)]), aspect="auto", interpolation="nearest"); axD.set_xticks([]); axD.set_yticks([])
for ix in bounds[1:]:
    if len(ix): axD.axhline(ix[0] - .5, color=TCOL[ORDER[0]], lw=.5); axD.axvline(ix[0] - .5, color=TCOL[ORDER[0]], lw=.5)
axD.set_xlabel("same TFs, same order as rows", labelpad=2)
cax = fig.add_axes([168 / W, Y0 + HH * .6, 2.5 / W, HH * .35]); cb = fig.colorbar(im, cax=cax); cb.set_label("Jaccard of target-gene lists", fontsize=5.5); cb.ax.tick_params(labelsize=5)
# panel letters
for lab, xmm in (("a", 17), ("b", 46), ("c", 68), ("d", 96)): fig.text(xmm / W - .012, (H - 3) / H, lab, fontsize=8, fontweight="bold", va="top")
# legend (bottom)
lx = 14 / W; ly = 26 / H
h1 = [Line2D([], [], marker="o", ls="", ms=3.5, color="k", label="line met"), Line2D([], [], marker="o", ls="", ms=3.5, mfc="white", mec="#555555", mew=.5, label="tested, not met"), Line2D([], [], marker="x", ls="", ms=3, mew=.5, color="#9E9E9E", label="could not be tested")]
lg1 = fig.legend(handles=h1, loc="upper left", bbox_to_anchor=(lx, ly), frameon=False, fontsize=5.5, handletextpad=.3, title="a: evidence", title_fontsize=5.5, alignment="left")
h2 = [plt.Rectangle((0, 0), 1, 1, color=TCOL[t]) for t in ORDER]; lg2 = fig.legend(h2, [NAME[t] for t in ORDER], loc="upper left", bbox_to_anchor=(46 / W, ly), frameon=False, fontsize=5.5, title="tier (bar, b)", title_fontsize=5.5, alignment="left")
h3 = [plt.Rectangle((0, 0), 1, 1, color=PROM), plt.Rectangle((0, 0), 1, 1, color=ENH)]; fig.legend(h3, ["promoter peak", "distal-only peak"], loc="upper left", bbox_to_anchor=(68 / W, ly), frameon=False, fontsize=5.5, title="c: target genes", title_fontsize=5.5, alignment="left")
fig.text(96 / W, ly - .002, "a: dot size = noise-floor specificity d (column 1) or own-motif read\nenrichment over the other 78 TFs (column 2). Noise floor: d >= 0.04.\nMotif: held-out top 10, or enrichment interval above 1. Bulk ChIP: own TF\nin the top 10 of 79 inside open chromatin (CTCF HEK293T and HEK293,\nNFYA K562, others HEK293). KD function: target list overlaps knockdown\nresponders (OR interval > 1 and above the swapped TF).\nTier 1 = >= 2 binding lines; Tier 2 = 1; Tier 3 = none.\nRows: by tier, then cells. d: Jaccard of q005 linked-peak gene lists. HEK293T; diagnostic.", fontsize=5, va="top")
for ext in ("png", "svg", "pdf"): fig.savefig(f"{O}/regulome_tier_overview.{ext}", dpi=300)
off = J[~np.eye(n, dtype=bool)]; print("Jaccard off-diagonal median", np.nanmedian(off), "max", np.nanmax(off))

pairs = [(J[i, j], T.tf[i], T.tf[j]) for i in range(n) for j in range(i + 1, n) if np.isfinite(J[i, j])]; pairs.sort(reverse=True); print("top pairs", [(a, b, round(v, 3)) for v, a, b in pairs[:12]])
t1 = [i for i in range(n) if T.tier[i] == ORDER[0]]; print("Tier1 within", [(T.tf[i], T.tf[j], round(J[i, j], 3)) for i in t1 for j in t1 if i < j])
