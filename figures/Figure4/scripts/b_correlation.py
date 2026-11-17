"""Fig 4b: the five ChromHMM element-set TF x TF correlation matrices that the candidate list was built from (accessibility-corrected,
Pearson r, SpQN-normalised by cell count): all elements, non-quiescent, TSS, enhancer, repressive.
One TF order for all five (average linkage on 1 - r of the all-elements matrix, optimal leaf order); shared diverging scale.
All five matrices in one row (Yuyang 2026-11-17). The three Fig 4 example pairs are outlined on every matrix; their six TFs are labelled on the first. Under each matrix: each pair's percentile in
that set (candidates_frozen_v1), bold with an asterisk where it is the pair's best set (the value that put it on the candidate list).
Inputs: spqn_all / spqn_noquies / spqn_tss (b_export_spqn.py, exact code of r2_targets/chromhmm_corrected_sets.py) and
repressive_chromhmm_corrected/data_min20/spqn_enhancer_biv / spqn_repressive; percentiles from candidates_all3081.tsv.
Palette C, Arial 7 pt (5-6 pt ticks), editable SVG text, 183 mm. usage: b_correlation.py spqn_dir candidates_all3081.tsv palette_dir out_prefix
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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
PAIRS = [("RBBP4", "ZNF644"), ("ADNP", "MTF2"), ("DR1", "NFYA")]   # 2026-11-17: RBBP4-ZNF644 replaced EPAS1-HES1
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
CHK = []

mm = 1 / 25.4; W, H = 183, 53
fig = plt.figure(figsize=(W * mm, H * mm))
def box(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])
LAB, GAP, CBW = 10.5, 1.8, 13.5                      # TF-label band (first matrix only), gap between matrices, colourbar block
L = (W - LAB - CBW - 4 * GAP) / 5; Y = 16             # matrix side (mm), bottom of the matrices
xs = [LAB + i * (L + GAP) for i in range(5)]
trio = sorted({t for p in PAIRS for t in p}, key=pos.get); cell = L / len(order)
for n_s, ((k, f, pc, title), x) in enumerate(zip(SETS, xs)):
    ax = box(x, Y, L, L); Mo = Ms[k].loc[order, order].values.copy(); np.fill_diagonal(Mo, np.nan)
    im = ax.imshow(Mo, cmap=cmap, vmin=-VL, vmax=VL, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_linewidth(0.4)
    ax.set_title(title, fontsize=7, pad=2.5)
    for (a, b), c in zip(PAIRS, ACC):
        i, j = sorted((pos[a], pos[b]))                # upper-triangle cell only: hollow ring (white halo under pair colour), cell colour visible inside
        for lw_, ec_ in ((2.2, "white"), (1.1, c)): ax.add_patch(Circle((j, i), 1.6, fill=False, ec=ec_, lw=lw_, zorder=3))
        CHK.append(dict(pair=f"{a}-{b}", set=k, row=order[i], col=order[j], r_plot=float(Mo[i, j]), r_src=float(Ms[k].loc[a, b]), pct=float(REC[tuple(sorted((a, b)))][pc])))
    if n_s == 0:                                      # the six TF labels once, on the first matrix (shared order)
        ys = [pos[t] * cell for t in trio]
        for q in range(1, len(ys)): ys[q] = max(ys[q], ys[q - 1] + 2.0)
        sh = min(0, (L - cell) - ys[-1]); ys = [v + sh for v in ys]
        for t, yy in zip(trio, ys):
            yr = (pos[t] + 0.5) * cell
            fig.add_artist(plt.Line2D([(x - 0.3) / W, (x - 1.5) / W, (x - 2.4) / W], [(Y + L - yr) / H, (Y + L - yr) / H, (Y + L - yy - cell / 2) / H], lw=0.4, color=P.neutral.dark))
            fig.text((x - 2.7) / W, (Y + L - yy - cell / 2) / H, t, ha="right", va="center", fontsize=5)
    for n, ((a, b), c) in enumerate(zip(PAIRS, ACC)):  # percentile row: one coloured square per pair
        r = REC[tuple(sorted((a, b)))]; best = r.best_matrix == k; cx = x + 0.6 + n * (L / 3)
        fig.add_artist(Rectangle((cx / W, (Y - 3.9) / H), 1.5 / W, 1.5 / H, color=c, lw=0, transform=fig.transFigure))
        fig.text((cx + 2.1) / W, (Y - 3.15) / H, f"{r[pc]:.2f}" + ("*" if best else ""), ha="left", va="center", fontsize=5.5, fontweight="bold" if best else "normal")
        fig.text((cx + 2.1) / W, (Y - 5.4) / H, f"r {Ms[k].loc[a, b]:.3f}", ha="left", va="center", fontsize=5, color=P.neutral.dark)
cx0 = xs[-1] + L + 2.2
cax = box(cx0, Y + 3, 1.5, L - 6)
cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[-VL, 0, VL])
cb.ax.set_yticklabels([f"≤ −{VL:.2f}", "0", f"≥ {VL:.2f}"]); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("Pearson r (SpQN-normalised)", fontsize=5.5, labelpad=1.5)
# pair key and notes in one line under the row
kx = LAB
for n, ((a, b), c) in enumerate(zip(PAIRS, ACC)):
    kax = box(kx, 4.3, 2.2, 2.2); kax.set_xlim(-1.1, 1.1); kax.set_ylim(-1.1, 1.1); kax.axis("off")    # key ring = the matrix marker
    kax.add_patch(Circle((0, 0), 0.8, fill=False, ec=c, lw=1.1))
    fig.text((kx + 3.0) / W, 5.4 / H, f"{a}–{b}", ha="left", va="center", fontsize=6); kx += 3.0 + 0.62 * 6 * 0.3528 * len(f"{a}–{b}") + 4
fig.text(kx / W, 5.4 / H, "Under each matrix: the pair's percentile in that element set and its r; * = best set, which placed the pair on the candidate list (≥ 0.60).",
         ha="left", va="center", fontsize=5.5, color=P.neutral.dark)
fig.text(LAB / W, 1.2 / H, f"TF order: clustering of the all-elements matrix, applied to all five ({len(order)} panel TFs); the six TFs of the three pairs are labelled on the first matrix.",
         ha="left", va="center", fontsize=5.5, color=P.neutral.dark)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
rows = []
for a, b in PAIRS:
    r = REC[tuple(sorted((a, b)))]
    rows.append(dict(pair=f"{a}-{b}", **{f"r_{k}": round(float(Ms[k].loc[a, b]), 4) for k, *_ in SETS}, **{pc: r[pc] for _, _, pc, _ in SETS}, best_pct=r.best_pct, best_matrix=r.best_matrix))
C = pd.DataFrame(CHK); C.to_csv(out + "_placement_check.tsv", sep="\t", index=False); print(C.to_string(index=False))
pd.DataFrame(rows).to_csv(out + "_values.tsv", sep="\t", index=False); print(pd.DataFrame(rows).to_string(index=False))
