"""Fig 4b: TF x TF coordinated-binding matrix from the frozen candidate list (candidates_frozen_v1: per pair, best percentile over the
five accessibility-corrected, SpQN-normalised ChromHMM element-set matrices), with the three Fig 4 example pairs.
Left: clustered 79 x 79 heatmap (average linkage on 1 - percentile, optimal leaf order), trio TFs labelled, pair cells outlined.
Right: one zoom per pair (the two TFs and their nearest neighbours in the clustering), pair cell outlined, percentile and best element set.
Palette C (palette/regulome_palette.py), Arial 7 pt (ticks 6 pt), editable SVG text, 183 mm wide.
usage: b_correlation.py candidates_all3081.tsv palette_dir out_prefix
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

tsv, paldir, out = sys.argv[1:4]
sys.path.insert(0, paldir)
import regulome_palette as rp
P = rp.use()
plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True, "savefig.dpi": 300})
PAIRS = [("EPAS1", "HES1"), ("ADNP", "MTF2"), ("DR1", "NFYA")]
ACC = [P.arms.raw, P.arms.imputed, P.arms.atac]          # palette C accents, one per pair
SET = {"all": "all elements", "no quiescent": "non-quiescent", "TSS": "TSS", "enhancer": "enhancer", "repressive": "repressive"}

A = pd.read_csv(tsv, sep="\t")
tfs = sorted(set(A.tf_a) | set(A.tf_b)); ix = {t: i for i, t in enumerate(tfs)}
M = np.eye(len(tfs))
for a, b, v in zip(A.tf_a, A.tf_b, A.best_pct): M[ix[a], ix[b]] = M[ix[b], ix[a]] = v
D = 1 - M; np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), "average"); Z = optimal_leaf_ordering(Z, squareform(D, checks=False))
order = [tfs[i] for i in leaves_list(Z)]; pos = {t: i for i, t in enumerate(order)}
MO = M[np.ix_([ix[t] for t in order], [ix[t] for t in order])]
row = {tuple(sorted(p)): r for p, r in zip(zip(A.tf_a, A.tf_b), A.itertuples())}
cmap = rp.seq_cmap(); vmin, vmax = 0, 1

mm = 1 / 25.4; W, H = 183, 86
fig = plt.figure(figsize=(W * mm, H * mm))
def box(x, y, w, h): return fig.add_axes([x / W, y / H, w / W, h / H])   # mm -> figure fraction
L = 64; Y0 = 17; axL = box(14, Y0, L, L)
im = axL.imshow(MO, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
axL.set_xticks([]); axL.set_yticks([])
for s in axL.spines.values(): s.set_linewidth(0.4)
for (a, b), c in zip(PAIRS, ACC):
    for i, j in ((pos[a], pos[b]), (pos[b], pos[a])):
        axL.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=c, lw=1.0))
# trio labels on the left, spread to >= 2.6 mm apart, with thin leaders to their rows
trio = sorted({t for p in PAIRS for t in p}, key=pos.get); cell = L / len(order); ys = [pos[t] * cell for t in trio]; gapmm = 2.6
for k in range(1, len(ys)): ys[k] = max(ys[k], ys[k - 1] + gapmm)
shift = min(0, (L - cell) - ys[-1]); ys = [y + shift for y in ys]
for t, y in zip(trio, ys):
    yr = (pos[t] + 0.5) * cell
    fig.add_artist(plt.Line2D([(14 - 0.4) / W, (14 - 2.2) / W, (14 - 3.2) / W], [(Y0 + L - yr) / H, (Y0 + L - yr) / H, (Y0 + L - y - cell / 2) / H], lw=0.4, color=P.neutral.dark))
    fig.text((14 - 3.6) / W, (Y0 + L - y - cell / 2) / H, t, ha="right", va="center", fontsize=6)
fig.text((14 + L / 2) / W, (Y0 + L + 1.5) / H, f"{len(order)} panel TFs (clustered)", ha="center", va="bottom", fontsize=7)
cax = box(14 + L / 2 - 16, Y0 - 6.5, 32, 1.5)
cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
cb.ax.tick_params(labelsize=6, length=1.5, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
cb.set_label("coordinated binding (percentile)", fontsize=6, labelpad=1.5)

def neighbours(a, b, k=2):
    s = set()
    for t in (a, b):
        p = pos[t]; s |= {order[i] for i in range(max(0, p - k), min(len(order), p + k + 1))}
    return sorted(s, key=pos.get)

Z0, ZW, LAB = 96, 22, 8.5
for n, ((a, b), c) in enumerate(zip(PAIRS, ACC)):
    sub = neighbours(a, b); Sm = M[np.ix_([ix[t] for t in sub], [ix[t] for t in sub])]
    ax = box(Z0 + n * (ZW + LAB), 42, ZW, ZW)
    ax.imshow(Sm, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub, rotation=90, fontsize=5)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub, fontsize=5)
    ax.tick_params(length=1.0, pad=0.8, width=0.4)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        if lab.get_text() in (a, b): lab.set_fontweight("bold")
    for s in ax.spines.values(): s.set_linewidth(0.4)
    i, j = sub.index(a), sub.index(b)
    for (r, q) in ((i, j), (j, i)): ax.add_patch(Rectangle((q - .5, r - .5), 1, 1, fill=False, ec=c, lw=1.1))
    rr = row[tuple(sorted((a, b)))]
    ax.text(0.5, 1.10, f"{a}–{b}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
    ax.add_patch(Rectangle((0.0, 1.035), 1, 0.035, transform=ax.transAxes, color=c, clip_on=False, lw=0))
    fig.text((Z0 + n * (ZW + LAB) + ZW / 2) / W, 32 / H, f"percentile {rr.best_pct:.2f}\n{SET[rr.best_matrix]} elements", ha="center", va="top", fontsize=6)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02); fig.savefig(out + ".svg", bbox_inches="tight", pad_inches=0.02)
pd.DataFrame([dict(pair=f"{a}-{b}", **{k: getattr(row[tuple(sorted((a, b)))], k) for k in ("pct_all", "pct_noquies", "pct_tss", "pct_enh", "pct_repr", "best_pct", "best_matrix")}) for a, b in PAIRS]).to_csv(out + "_values.tsv", sep="\t", index=False)
print("order saved; pairs:", [(a, b, round(row[tuple(sorted((a, b)))].best_pct, 4), row[tuple(sorted((a, b)))].best_matrix) for a, b in PAIRS])
