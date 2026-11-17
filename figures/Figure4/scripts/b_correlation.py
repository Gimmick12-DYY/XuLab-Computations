"""Fig 4b: the five ChromHMM element-set TF x TF percentile matrices that the candidate list was built from (candidates_frozen_v1 per-set pct:
all elements, non-quiescent, TSS, enhancer incl. EnhBiv, repressive). Style reused from findings_slides/jie/fig_corr_ezh2_box.py (EZH2-MTF2 deck
heatmap): cells coloured by percentile of TF-TF Pearson r, RdYlBu_r 0-1, TF order = average linkage on 1 - percentile of the all-elements matrix
(optimal leaf order), pair TFs labelled on both axes with leader lines, open black squares on the pair cells (both triangles), rounded callout with
arrow, one colourbar. Here: five matrices in one row (Yuyang 2026-11-17), one callout per pair on its best set, squares 3 cells wide so they read at
183 mm, pair TF labels in one colour per pair. Percentiles floored to 2 decimals as in the original. Under each matrix: each pair's percentile in
that set, bold with * on the pair's best set.
Inputs: pct_all / pct_noquies / pct_tss (regulatome/chromhmm_corrected_sets) and pct_enhancer_biv / pct_repressive
(regulatome/repressive_chromhmm_corrected/data_min20); identical to candidates_frozen_v1 pct columns (max diff 5e-5, rounding).
Palette C, Arial (>= 5 pt), editable SVG text, 183 mm. usage: b_correlation.py pct_dir candidates_all3081.tsv palette_dir out_prefix
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

pd_, tsv, paldir, out = sys.argv[1:5]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use()
plt.rcParams.update({"savefig.dpi": 300, "axes.linewidth": 0.5})
SETS = [("all", "pct_all.tsv", "pct_all", "All elements"), ("no quiescent", "pct_noquies.tsv", "pct_noquies", "Non-quiescent"),
        ("TSS", "pct_tss.tsv", "pct_tss", "TSS / promoter"), ("enhancer", "pct_enhancer_biv.tsv", "pct_enh", "Enhancer"),
        ("repressive", "pct_repressive.tsv", "pct_repr", "Repressive")]
PAIRS = [("RBBP4", "ZNF644"), ("ADNP", "MTF2"), ("DR1", "NFYA")]
ACC = [P.arms.raw, P.arms.imputed, P.arms.atac]
f2 = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"          # floor, as fig_corr_ezh2_box.py

def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
Ms = {k: rd(f"{pd_}/{f}") for k, f, _, _ in SETS}
tfs = list(Ms["all"].index)
for k in Ms: Ms[k] = Ms[k].loc[tfs, tfs]
Dm = 1 - (Ms["all"].values + Ms["all"].values.T) / 2; np.fill_diagonal(Dm, 0); dv = squareform(Dm, checks=False)
order = [tfs[i] for i in leaves_list(optimal_leaf_ordering(linkage(dv, "average"), dv))]; pos = {t: i for i, t in enumerate(order)}; n_ = len(order)
A = pd.read_csv(tsv, sep="\t"); REC = {tuple(sorted(x)): r for x, (_, r) in zip(zip(A.tf_a, A.tf_b), A.iterrows())}
TCOL = {t: c for (a, b), c in zip(PAIRS, ACC) for t in (a, b)}

mm = 1 / 25.4; W, H = 183, 60
fig = plt.figure(figsize=(W * mm, H * mm))
def box(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
LAB, GAP, CBW = 10.5, 2.4, 13.0
L = (W - LAB - CBW - 4 * GAP) / 5; Y = 14; cell = L / n_
xs = [LAB + i * (L + GAP) for i in range(5)]
six = sorted(TCOL, key=pos.get)
def spread(ps, gap):                                              # as fig_corr_ezh2_box.py
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
sp = spread([pos[t] for t in six], 2.1 / cell)                    # label slots in cell units, >= 2.1 mm apart
CHK = []
for n_s, ((k, f, pc, title), x) in enumerate(zip(SETS, xs)):
    ax = box(x, Y, L, L); Mx = Ms[k].loc[order, order].values
    im = ax.imshow(Mx, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-.5, n_ - .5); ax.set_ylim(n_ - .5, -.5)
    ax.text(0.5, 1 + 9.0 / L, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    for t, y in zip(six, sp):                                     # top axis on every matrix, left axis on the first (shared order)
        k_ = pos[t]; c = TCOL[t]
        ax.annotate(t, xy=(k_, -.5), xytext=(y, -.5 - 2.2 / cell), ha="center", va="bottom", rotation=90, fontsize=5, fontweight="bold", color=c,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.4, color=c, shrinkA=0, shrinkB=0))
        if n_s == 0:
            ax.annotate(t, xy=(-.5, k_), xytext=(-.5 - 2.2 / cell, y), ha="right", va="center", fontsize=5, fontweight="bold", color=c,
                        annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.4, color=c, shrinkA=0, shrinkB=0))
    for a, b in PAIRS:                                            # open squares (3 cells wide) on the pair cells, both triangles
        i, j = pos[a], pos[b]
        for r_, q_ in ((i, j), (j, i)):
            for lw_, ec_ in ((1.6, "white"), (0.8, P.neutral.ink)): ax.add_patch(Rectangle((q_ - 1.5, r_ - 1.5), 3, 3, fill=False, ec=ec_, lw=lw_, zorder=5))
            CHK.append(dict(pair=f"{a}-{b}", set=k, row=order[r_], col=order[q_], pct_plot=float(Mx[r_, q_]), pct_candidates=float(REC[tuple(sorted((a, b)))][pc])))
    for (a, b), c in zip(PAIRS, ACC):                             # one callout per pair, on its best set
        r = REC[tuple(sorted((a, b)))]
        if r.best_matrix != k: continue
        i, j = sorted((pos[a], pos[b])); tgt = (j, i)               # upper-triangle cell
        ax.annotate(f"{a}–{b}\npercentile {f2(r[pc])}", xy=tgt, xytext=(n_ * 0.27, n_ * 0.80), fontsize=5.5, fontweight="bold", color=P.neutral.ink, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=P.neutral.ink, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=2),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, lw=0.8), zorder=8)
    for n, ((a, b), c) in enumerate(zip(PAIRS, ACC)):              # percentile row
        r = REC[tuple(sorted((a, b)))]; best = r.best_matrix == k; cx = x + 0.6 + n * (L / 3)
        fig.add_artist(Rectangle((cx / W, (Y - 3.9) / H), 1.5 / W, 1.5 / H, color=c, lw=0, transform=fig.transFigure))
        fig.text((cx + 2.1) / W, (Y - 3.15) / H, f2(r[pc]) + ("*" if best else ""), ha="left", va="center", fontsize=5.5, fontweight="bold" if best else "normal")
cax = box(xs[-1] + L + 2.2, Y + 3, 1.5, L - 6)
cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0, .5, 1]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("Percentile of TF–TF Pearson r", fontsize=5.5, labelpad=1.5)
kx = LAB
for (a, b), c in zip(PAIRS, ACC):
    fig.add_artist(Rectangle((kx / W, 4.6 / H), 1.8 / W, 1.8 / H, color=c, lw=0, transform=fig.transFigure))
    fig.text((kx + 2.6) / W, 5.5 / H, f"{a}–{b}", ha="left", va="center", fontsize=6, fontweight="bold", color=c); kx += 2.6 + 0.66 * 6 * 0.3528 * len(f"{a}–{b}") + 4
fig.text(kx / W, 5.5 / H, "Numbers: the pair's percentile in that element set (floored to 2 decimals); * = best set, which placed the pair on the candidate list (≥ 0.60).",
         ha="left", va="center", fontsize=5.5, color=P.neutral.dark)
fig.text(LAB / W, 1.4 / H, f"Cell colour: percentile of the TF–TF Pearson r among all {n_ * (n_ - 1) // 2:,} TF pairs in that set. TF order: clustering of the all-elements matrix, "
         f"applied to all five ({n_} TFs); squares mark the three pairs.", ha="left", va="center", fontsize=5.5, color=P.neutral.dark)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
C = pd.DataFrame(CHK); C["diff"] = (C.pct_plot - C.pct_candidates).abs(); C.to_csv(out + "_placement_check.tsv", sep="\t", index=False)
print("placement check: max |pct_plot - pct_candidates| =", C["diff"].max(), "| rows/cols correct:", all(set((r.row, r.col)) == set(r.pair.split("-")) for r in C.itertuples()))
rows = []
for a, b in PAIRS:
    r = REC[tuple(sorted((a, b)))]
    rows.append(dict(pair=f"{a}-{b}", **{pc: r[pc] for _, _, pc, _ in SETS}, **{pc + "_shown": f2(r[pc]) for _, _, pc, _ in SETS}, best_pct=r.best_pct, best_matrix=r.best_matrix))
pd.DataFrame(rows).to_csv(out + "_values.tsv", sep="\t", index=False); print(pd.DataFrame(rows).to_string(index=False))
