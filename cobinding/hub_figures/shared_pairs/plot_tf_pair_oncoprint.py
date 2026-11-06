"""TF x shared-pair matrix sorted like an oncoprint (rows by no. of shared pairs; columns sorted by row membership),
plus an UpSet-style plot of the most frequent TF combinations."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from collections import Counter
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
SUF = sys.argv[1] if len(sys.argv) > 1 else ""; MINS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
S = pd.read_csv(f"shared_pairs{SUF}.tsv", sep="\t")
cells = pd.read_csv("../goodex/tf_cell_counts.tsv", sep="\t", header=None, index_col=0)[1]
MIN_TF, MIN_ROW = MINS, 10
S3 = S[S.n_tfs >= MIN_TF].reset_index(drop=True)
cnt = Counter(t for x in S3.tfs for t in x.split(","))
rows = [t for t, n in cnt.most_common() if n >= MIN_ROW]
M = np.array([[t in set(x.split(",")) for x in S3.tfs] for t in rows], dtype=np.uint8)
keep = M.sum(0) >= 2; M = M[:, keep]; nt = S3.n_tfs.values[keep]
w = 2 ** np.arange(len(rows))[::-1].astype(object)          # oncoprint sort: by first row, then second ...
key = [sum(int(b) * int(v) for b, v in zip(M[:, j], w)) for j in range(M.shape[1])]
co = np.argsort(key)[::-1]; M = M[:, co]; nt = nt[co]
fig = plt.figure(figsize=(16, 0.3 * len(rows) + 2.4))
gs = fig.add_gridspec(2, 3, height_ratios=[0.7, 0.3 * len(rows)], width_ratios=[1, 0.07, 0.07], hspace=0.04, wspace=0.03)
ax = fig.add_subplot(gs[1, 0]); top = fig.add_subplot(gs[0, 0], sharex=ax)
ax.imshow(M, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#F4F4F4", "#1F4E8C"]), interpolation="nearest")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=10); ax.set_xticks([])
ax.set_xlabel(f"Cicero pairs >25 kb shared by ≥{MIN_TF} TFs (n = {M.shape[1]:,}), sorted by which TFs share them")
top.bar(range(len(nt)), nt, width=1.0, color="#555"); top.set_ylabel("TFs per\npair", fontsize=9); top.tick_params(labelbottom=False)
top.set_xlim(-0.5, len(nt) - 0.5)
for sp in ("top", "right"): top.spines[sp].set_visible(False)
a1 = fig.add_subplot(gs[1, 1], sharey=ax); a1.barh(range(len(rows)), M.sum(1), color="#7C8693", height=0.8)
a1.set_xlabel("shared\npairs", fontsize=9); a1.tick_params(axis="y", left=False, labelleft=False)
a2 = fig.add_subplot(gs[1, 2], sharey=ax); a2.barh(range(len(rows)), [cells.get(t, np.nan) for t in rows], color="#C9A227", height=0.8)
a2.set_xscale("log"); a2.set_xticks([1e3, 1e5]); a2.set_xticklabels(["1k", "100k"], fontsize=8); a2.minorticks_off()
a2.set_xlabel("cells", fontsize=9); a2.tick_params(axis="y", left=False, labelleft=False)
for a in (a1, a2):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig(f"fig_tf_x_shared_pairs_sorted{SUF}.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(f"fig_tf_x_shared_pairs_sorted{SUF}.svg", bbox_inches="tight", facecolor="white"); plt.close(fig)

# UpSet: most frequent exact TF combinations among pairs shared by >=2 TFs
combo = S[S.n_tfs >= 3].tfs.value_counts().head(25)
tfsU = sorted({t for c in combo.index for t in c.split(",")}, key=lambda t: -sum(t in c.split(",") for c in combo.index))
fig = plt.figure(figsize=(14, 0.28 * len(tfsU) + 4))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 0.28 * len(tfsU)], width_ratios=[0.18, 1], hspace=0.05, wspace=0.02)
ab = fig.add_subplot(gs[0, 1]); am = fig.add_subplot(gs[1, 1], sharex=ab); al = fig.add_subplot(gs[1, 0], sharey=am)
x = np.arange(len(combo))
ab.bar(x, combo.values, color="#1F4E8C"); ab.set_ylabel("pairs shared by exactly\nthis set of ≥3 TFs")
for i, v in enumerate(combo.values): ab.text(i, v, str(v), ha="center", va="bottom", fontsize=8.5)
ab.tick_params(labelbottom=False)
for sp in ("top", "right"): ab.spines[sp].set_visible(False)
for i, c in enumerate(combo.index):
    ys = [tfsU.index(t) for t in c.split(",")]
    am.scatter([i] * len(tfsU), range(len(tfsU)), s=40, color="#DDDDDD", zorder=1)
    am.scatter([i] * len(ys), ys, s=48, color="#1F4E8C", zorder=2)
    am.plot([i, i], [min(ys), max(ys)], color="#1F4E8C", lw=2, zorder=2)
am.set_yticks(range(len(tfsU))); am.set_yticklabels([]); am.invert_yaxis(); am.set_xticks([])
for sp in ("top", "right", "left", "bottom"): am.spines[sp].set_visible(False)
tot = Counter(t for x in S.tfs for t in x.split(","))
al.barh(range(len(tfsU)), [tot[t] for t in tfsU], color="#7C8693"); al.invert_xaxis()
al.set_yticks(range(len(tfsU))); al.set_yticklabels(tfsU, fontsize=10); al.yaxis.tick_right()
al.set_xlabel("shared pairs\nper TF", fontsize=9)
for sp in ("top", "left"): al.spines[sp].set_visible(False)
fig.savefig(f"fig_shared_pairs_upset{SUF}.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(f"fig_shared_pairs_upset{SUF}.svg", bbox_inches="tight", facecolor="white")
print(len(rows), "rows;", M.shape[1], "cols"); print(combo.head(10).to_string())
