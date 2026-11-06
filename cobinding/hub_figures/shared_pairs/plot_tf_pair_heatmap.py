"""TF x shared Cicero pair heatmap (rows TFs, columns pairs >25 kb shared by >= MIN TFs)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
MIN = int(sys.argv[1]); out = sys.argv[2]
S = pd.read_csv("shared_pairs.tsv", sep="\t"); S = S[S.n_tfs >= MIN].reset_index(drop=True)
cells = pd.read_csv("../goodex/tf_cell_counts.tsv", sep="\t", header=None, index_col=0)[1]
tfs = sorted({t for x in S.tfs for t in x.split(",")})
M = np.zeros((len(tfs), len(S)), dtype=np.uint8); ix = {t: i for i, t in enumerate(tfs)}
for j, x in enumerate(S.tfs):
    for t in x.split(","): M[ix[t], j] = 1
ro = leaves_list(linkage(pdist(M.astype(bool), "jaccard"), "average"))
co = leaves_list(linkage(pdist(M.T.astype(bool), "jaccard"), "average")) if len(S) <= 8000 else np.argsort(-S.n_tfs.values)
M = M[np.ix_(ro, co)]; names = [tfs[i] for i in ro]; nt = S.n_tfs.values[co]
fig = plt.figure(figsize=(16, 0.24 * len(names) + 2.2))
gs = fig.add_gridspec(2, 3, height_ratios=[0.6, 0.24 * len(names)], width_ratios=[1, 0.06, 0.06], hspace=0.03, wspace=0.03)
ax = fig.add_subplot(gs[1, 0]); top = fig.add_subplot(gs[0, 0], sharex=ax)
ax.imshow(M, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#F4F4F4", "#1F4E8C"]), interpolation="nearest")
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9); ax.set_xticks([])
ax.set_xlabel(f"Cicero co-binding pairs >25 kb shared by ≥{MIN} TFs (n = {len(S):,}; clustered)")
top.bar(range(len(nt)), nt, width=1.0, color="#555"); top.set_ylabel("TFs\nper pair", fontsize=9)
top.tick_params(labelbottom=False); top.set_xlim(-0.5, len(nt) - 0.5)
for sp in ("top", "right"): top.spines[sp].set_visible(False)
a1 = fig.add_subplot(gs[1, 1], sharey=ax); a1.barh(range(len(names)), M.sum(1), color="#7C8693", height=0.8)
a1.set_xlabel("shared\npairs", fontsize=9); a1.tick_params(axis="y", left=False, labelleft=False)
a2 = fig.add_subplot(gs[1, 2], sharey=ax); a2.barh(range(len(names)), [cells.get(t, np.nan) for t in names], color="#C9A227", height=0.8)
a2.set_xscale("log"); a2.set_xticks([1e3, 1e5]); a2.set_xticklabels(["1k", "100k"], fontsize=8); a2.minorticks_off()
a2.set_xlabel("cells", fontsize=9); a2.tick_params(axis="y", left=False, labelleft=False)
for a in (a1, a2):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white"); fig.savefig(f"{out}.svg", bbox_inches="tight", facecolor="white")
pd.DataFrame(M, index=names, columns=[f"{S.region1[c]}|{S.region2[c]}" for c in co]).to_csv(f"{out}_matrix.tsv.gz", sep="\t", compression="gzip")
print(MIN, len(names), "TFs", len(S), "pairs")
