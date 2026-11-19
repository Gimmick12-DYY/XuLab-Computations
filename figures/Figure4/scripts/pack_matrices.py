"""Figure 4 PDF pack, correlation matrices: one PDF per ChromHMM element set (all elements, non-quiescent, TSS/promoter, enhancer incl. EnhBiv,
repressive), each with all 79 TF names on both axes. Approved panel b style (b_correlation.py 8030840, from findings_slides/jie/fig_corr_ezh2_box.py):
cell = percentile of TF-TF Pearson r in that set, RdYlBu_r 0-1; one TF order for all five (average linkage on 1 - percentile of the all-elements
matrix, optimal leaf order); the three Fig 4 pairs boxed (both triangles), their six TFs in bold pair colour; callout only on each pair's best set;
percentiles floored to 2 decimals. Illustrator-editable PDF (pdf.fonttype 42, Arial), 183 mm wide.
Pair percentiles and best set are read from the same matrices (identical to candidates_frozen_v1 pct columns, max diff 5e-5).
usage: pack_matrices.py pct_dir palette_dir out_dir
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

pdir, paldir, out = sys.argv[1:4]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use()
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.5})
SETS = [("all", "pct_all.tsv", "pct_all", "All elements", "1_all_elements"), ("no quiescent", "pct_noquies.tsv", "pct_noquies", "Non-quiescent", "2_non_quiescent"),
        ("TSS", "pct_tss.tsv", "pct_tss", "TSS / promoter", "3_tss_promoter"), ("enhancer", "pct_enhancer_biv.tsv", "pct_enh", "Enhancer (incl. bivalent)", "4_enhancer"),
        ("repressive", "pct_repressive.tsv", "pct_repr", "Repressive", "5_repressive")]
PAIRS = [("RBBP4", "ZNF644"), ("ADNP", "MTF2"), ("DR1", "NFYA")]
ACC = [P.arms.raw, P.arms.imputed, P.arms.atac]
f2 = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"

def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
Ms = {k: rd(f"{pdir}/{f}") for k, f, *_ in SETS}
tfs = list(Ms["all"].index)
for k in Ms: Ms[k] = Ms[k].loc[tfs, tfs]
Dm = 1 - (Ms["all"].values + Ms["all"].values.T) / 2; np.fill_diagonal(Dm, 0); dv = squareform(Dm, checks=False)
order = [tfs[i] for i in leaves_list(optimal_leaf_ordering(linkage(dv, "average"), dv))]; pos = {t: i for i, t in enumerate(order)}; n_ = len(order)
REC = {}
for a, b in PAIRS:
    v = {pc: float(Ms[k].loc[a, b]) for k, f, pc, *_ in SETS}; bk = [k for k, f, pc, *_ in SETS][int(np.argmax([v[pc] for k, f, pc, *_ in SETS]))]
    REC[tuple(sorted((a, b)))] = pd.Series(dict(v, best_matrix=bk))
TCOL = {t: c for (a, b), c in zip(PAIRS, ACC) for t in (a, b)}
mm = 1 / 25.4; W, H = 183, 196; X0, Y0, S = 16, 28, 150          # matrix square S mm; cell ~1.9 mm
rows = []
for k, f, pc, title, stem in SETS:
    fig = plt.figure(figsize=(W * mm, H * mm)); ax = fig.add_axes([X0 / W, Y0 / H, S / W, S / H]); Mx = Ms[k].loc[order, order].values
    im = ax.imshow(Mx, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_)); ax.set_yticks(range(n_)); ax.set_xticklabels(order, rotation=90, fontsize=5); ax.set_yticklabels(order, fontsize=5)
    ax.xaxis.tick_top(); ax.tick_params(length=1.2, width=0.3, pad=1)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        t = lab.get_text()
        if t in TCOL: lab.set_color(TCOL[t]); lab.set_fontweight("bold")
    ax.set_xlim(-.5, n_ - .5); ax.set_ylim(n_ - .5, -.5)
    fig.text((X0 + S / 2) / W, (H - 3) / H, title, ha="center", va="top", fontsize=8, fontweight="bold")
    for (a, b), c in zip(PAIRS, ACC):
        i, j = pos[a], pos[b]
        for r_, q_ in ((i, j), (j, i)):
            for lw_, ec_ in ((1.8, "white"), (0.9, P.neutral.ink)): ax.add_patch(Rectangle((q_ - .5, r_ - .5), 1, 1, fill=False, ec=ec_, lw=lw_, zorder=5))
            assert {order[r_], order[q_]} == {a, b}
        r = REC[tuple(sorted((a, b)))]
        rows.append(dict(file=f"{stem}.pdf", pair=f"{a}-{b}", pct=r[pc], shown=f2(r[pc]), best=r.best_matrix == k, plotted=float(Mx[i, j])))
        if r.best_matrix == k:
            lo, hi = sorted((i, j))
            ax.annotate(f"{a}–{b}\npercentile {f2(r[pc])}", xy=(hi, lo), xytext=(max(hi - 14, 8), max(lo - 9, 4)), fontsize=7, fontweight="bold", color=P.neutral.ink,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=P.neutral.ink, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=3),
                        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=c, lw=1.0), zorder=8)
    cax = fig.add_axes([(X0 + S + 3) / W, (Y0 + S * 0.3) / H, 2.2 / W, S * 0.4 / H])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1]); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4); cb.outline.set_linewidth(0.4)
    cb.set_label("Percentile of TF–TF Pearson r", fontsize=6.5)
    yy = Y0 - 9; xx = X0                                           # pair key with this set's percentiles
    for (a, b), c in zip(PAIRS, ACC):
        r = REC[tuple(sorted((a, b)))]; best = r.best_matrix == k
        fig.add_artist(Rectangle((xx / W, (yy - 1) / H), 2 / W, 2 / H, color=c, lw=0, transform=fig.transFigure))
        fig.text((xx + 3) / W, yy / H, f"{a}–{b}  {f2(r[pc])}" + ("*" if best else ""), ha="left", va="center", fontsize=6.5, fontweight="bold" if best else "normal")
        xx += 46
    fig.text(X0 / W, (yy - 6) / H, f"Cell colour: percentile of the TF–TF Pearson r among all {n_ * (n_ - 1) // 2:,} TF pairs in this element set (floored to 2 decimals). "
             "Boxes: the three Fig. 4 pairs.\n* = the pair's best element set, which placed it on the candidate list (≥ 0.60). TF order: clustering of the all-elements matrix, same in all five.",
             ha="left", va="top", fontsize=5.5, color=P.neutral.dark, linespacing=1.4)
    fig.savefig(f"{out}/{stem}.pdf"); fig.savefig(f"{out}/{stem}.png", dpi=200); plt.close(fig)
R = pd.DataFrame(rows); assert (abs(R.pct - R.plotted) < 1e-4).all(); print(R.to_string(index=False))
R.to_csv(f"{out}/matrices_check.tsv", sep="\t", index=False)
