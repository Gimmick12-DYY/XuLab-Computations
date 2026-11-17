"""Fig 4b: the five ChromHMM element-set TF x TF correlation matrices that the candidate list was built from (accessibility-corrected,
Pearson r, SpQN-normalised by cell count): all elements, non-quiescent, TSS, enhancer, repressive.
One TF order for all five (average linkage on 1 - r of the all-elements matrix, optimal leaf order); shared diverging scale.
The three Fig 4 example pairs are outlined on every matrix; only their six TFs are labelled. Under each matrix: each pair's percentile in
that set (candidates_frozen_v1), bold with an asterisk where it is the pair's best set (the value that put it on the candidate list).
Inputs: spqn_all / spqn_noquies / spqn_tss (b_export_spqn.py, exact code of r2_targets/chromhmm_corrected_sets.py) and
repressive_chromhmm_corrected/data_min20/spqn_enhancer_biv / spqn_repressive; percentiles from candidates_all3081.tsv.
Palette C, Arial 7 pt (5-6 pt ticks), editable SVG text, 183 mm. usage: b_correlation.py spqn_dir candidates_all3081.tsv palette_dir out_prefix
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

sd, tsv, paldir, out = sys.argv[1:5]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use()
plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True, "savefig.dpi": 300})
SETS = [("all", "spqn_all.tsv", "pct_all", "All elements"), ("no quiescent", "spqn_noquies.tsv", "pct_noquies", "Non-quiescent"),
        ("TSS", "spqn_tss.tsv", "pct_tss", "TSS / promoter"), ("enhancer", "spqn_enhancer_biv.tsv", "pct_enh", "Enhancer"),
        ("repressive", "spqn_repressive.tsv", "pct_repr", "Repressive")]
PAIRS = [("EPAS1", "HES1"), ("ADNP", "MTF2"), ("DR1", "NFYA")]
ACC = [P.arms.raw, P.arms.imputed, P.arms.atac]
VL = 0.06

def rd(f):
    m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.upper(); m.columns = m.columns.str.strip().str.upper(); return m
Ms = {k: rd(f"{sd}/{f}") for k, f, _, _ in SETS}
tfs = list(Ms["all"].index)
for k in Ms: Ms[k] = Ms[k].loc[tfs, tfs]
R = Ms["all"].values.copy(); D = 1 - R; np.fill_diagonal(D, 0); D = (D + D.T) / 2
Z = optimal_leaf_ordering(linkage(squareform(D, checks=False), "average"), squareform(D, checks=False))
order = [tfs[i] for i in leaves_list(Z)]; pos = {t: i for i, t in enumerate(order)}
A = pd.read_csv(tsv, sep="\t"); REC = {tuple(sorted(x)): r for x, (_, r) in zip(zip(A.tf_a, A.tf_b), A.iterrows())}
cmap = LinearSegmentedColormap.from_list("div", P.div); cmap.set_bad(P.na)

mm = 1 / 25.4; W, H = 183, 128
fig = plt.figure(figsize=(W * mm, H * mm))
def box(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
L, LAB, GAP = 46, 11.5, 2.5
xs = [LAB, LAB + L + LAB + GAP, LAB + 2 * (L + LAB + GAP)]
cells = [(xs[0], 72), (xs[1], 72), (xs[2], 72), (xs[0], 9), (xs[1], 9)]
trio = sorted({t for p in PAIRS for t in p}, key=pos.get); cell = L / len(order)
for (k, f, pc, title), (x, y) in zip(SETS, cells):
    ax = box(x, y, L, L); Mo = Ms[k].loc[order, order].values.copy(); np.fill_diagonal(Mo, np.nan)
    im = ax.imshow(Mo, cmap=cmap, vmin=-VL, vmax=VL, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
    ax.set_title(title, fontsize=7, pad=2.5)
    for (a, b), c in zip(PAIRS, ACC):
        for i, j in ((pos[a], pos[b]), (pos[b], pos[a])): ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c, lw=0.9))
    ys = [pos[t] * cell for t in trio]
    for q in range(1, len(ys)): ys[q] = max(ys[q], ys[q - 1] + 2.4)
    sh = min(0, (L - cell) - ys[-1]); ys = [v + sh for v in ys]
    for t, yy in zip(trio, ys):
        yr = (pos[t] + 0.5) * cell
        fig.add_artist(plt.Line2D([(x - 0.3) / W, (x - 1.8) / W, (x - 2.8) / W], [(y + L - yr) / H, (y + L - yr) / H, (y + L - yy - cell / 2) / H], lw=0.4, color=P.neutral.dark))
        fig.text((x - 3.2) / W, (y + L - yy - cell / 2) / H, t, ha="right", va="center", fontsize=5.5)
    # percentile row: one coloured square per pair
    for n, ((a, b), c) in enumerate(zip(PAIRS, ACC)):
        r = REC[tuple(sorted((a, b)))]; best = r.best_matrix == k; cx = x + 2 + n * (L / 3)
        fig.add_artist(Rectangle((cx / W, (y - 4.6) / H), 1.8 / W, 1.8 / H, color=c, lw=0, transform=fig.transFigure))
        fig.text((cx + 2.6) / W, (y - 3.7) / H, f"{r[pc]:.2f}" + ("*" if best else ""), ha="left", va="center", fontsize=6, fontweight="bold" if best else "normal")
# legend + colourbar in the free slot (bottom right)
lx, ly = xs[2], 9
cax = box(lx, ly + 36, 34, 1.5)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-VL, 0, VL])
cb.ax.set_xticklabels([f"≤ −{VL:.2f}", "0", f"≥ {VL:.2f}"]); cb.ax.tick_params(labelsize=6, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("Pearson r (SpQN-normalised)", fontsize=6, labelpad=1.5)
for n, ((a, b), c) in enumerate(zip(PAIRS, ACC)):
    yy = ly + 27 - n * 4.2
    fig.add_artist(Rectangle((lx / W, yy / H), 2.6 / W, 2.6 / H, fill=False, ec=c, lw=1.0, transform=fig.transFigure))
    fig.text((lx + 4) / W, (yy + 1.3) / H, f"{a}–{b}", ha="left", va="center", fontsize=6)
fig.text(lx / W, (ly + 11.5) / H, "Numbers under each matrix: the pair's percentile\nin that element set; * = best set, which placed\nthe pair on the candidate list (≥ 0.60).",
         ha="left", va="top", fontsize=5.5, color=P.neutral.dark, linespacing=1.3)
fig.text(lx / W, (ly + 1) / H, f"TF order: clustering of the all-elements matrix,\napplied to all five; {len(order)} panel TFs.", ha="left", va="bottom", fontsize=5.5, color=P.neutral.dark, linespacing=1.3)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
rows = []
for a, b in PAIRS:
    r = REC[tuple(sorted((a, b)))]
    rows.append(dict(pair=f"{a}-{b}", **{f"r_{k}": round(float(Ms[k].loc[a, b]), 4) for k, *_ in SETS}, **{pc: r[pc] for _, _, pc, _ in SETS}, best_pct=r.best_pct, best_matrix=r.best_matrix))
pd.DataFrame(rows).to_csv(out + "_values.tsv", sep="\t", index=False); print(pd.DataFrame(rows).to_string(index=False))
