"""Fig 3 panel j layout examples (2026-11-13): our binned GO results (j_gene_set_dotplot_by_tf_count.tsv) drawn in the two layouts
most used in papers to compare GO enrichment across gene lists:
 A. clusterProfiler compareCluster dot plot: groups on x, terms on y (one shared axis), dot size = gene ratio (genes in term / genes
    in list), colour = -log10 adjusted p; only enriched (q < 0.05) cells drawn, others left empty.
 B. Metascape-style multi-list heatmap: terms x lists, colour = -log10 adjusted p, blank = not enriched; terms ordered by the list
    where they peak.
Near-duplicate terms are collapsed (one per parent/child pair), as REVIGO / clusterProfiler::simplify do.
  python scripts/j_layout_examples.py   (Mac, reads ../j_gene_set_dotplot_by_tf_count.tsv)"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "axes.labelsize": 7, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False})
H = Path(__file__).resolve().parents[1]; T = pd.read_csv(H / "j_gene_set_dotplot_by_tf_count.tsv", sep="\t", index_col=0); O = H / "j_options"
BINS = ["1 TF", "2 TFs", "3-4 TFs", "≥5 TFs"]; NANN = {"1 TF": 3259, "2 TFs": 1148, "3-4 TFs": 761, "≥5 TFs": 274}
DROP = ["GO:1903047", "GO:0032008", "GO:1903432", "GO:0031570", "GO:0006469", "GO:0042326"]   # child/parent duplicates collapsed
T = T.drop([t for t in DROP if t in T.index])
SHORT = {"regulation of cellular response to stress": "Regulation of response to stress", "signal transduction in response to DNA damage": "DNA damage signal transduction",
         "positive regulation of triglyceride biosynthetic process": "Pos. reg. of triglyceride biosynthesis", "regulation of monoatomic ion transmembrane transport": "Reg. of ion transmembrane transport",
         "monoatomic ion transmembrane transport": "Ion transmembrane transport", "dolichol-linked oligosaccharide biosynthetic process": "Dolichol oligosaccharide synthesis"}
cap = lambda s: SHORT.get(s, s[0].upper() + s[1:])
Q = T[[f"{b}_q" for b in BINS]].values; LQ = -np.log10(np.clip(Q, 1e-10, 1)); best = np.argmin(Q, 1)
keep = (Q < 0.25).any(1)                                    # terms with at least a hint in one list
T, Q, LQ, best = T[keep], Q[keep], LQ[keep], best[keep]
order = np.lexsort((-LQ.max(1), best)); T, Q, LQ = T.iloc[order], Q[order], LQ[order]
cm = LinearSegmentedColormap.from_list("t", ["#C9E3DC", "#2C7A6B", "#0E3B33"]); MM = 1 / 25.4
# A: compareCluster dot plot
fig, ax = plt.subplots(figsize=(120 * MM, (22 + 4.2 * len(T)) * MM)); fig.subplots_adjust(left=0.42, right=0.70, top=0.97, bottom=0.16)
for j, b in enumerate(BINS):
    for i, t in enumerate(T.index):
        q = T.loc[t, f"{b}_q"]
        if q < 0.05:
            gr = T.loc[t, f"{b}_genes"] / NANN[b]; ax.scatter(j, i, s=8 + 900 * gr, c=[[-np.log10(q)]], cmap=cm, vmin=1.3, vmax=2.5, lw=0)
ax.set_xticks(range(4)); ax.set_xticklabels([f"{b}\n({NANN[b]:,})" for b in BINS]); ax.set_yticks(range(len(T))); ax.set_yticklabels([cap(s) for s in T.name])
ax.set_xlim(-0.6, 3.6); ax.set_ylim(len(T) - 0.4, -0.6); ax.grid(color="#EEEEEE", lw=0.5); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_xlabel("Promoter in modules of"); sc = ax.scatter([], [], c=[], cmap=cm, vmin=1.3, vmax=2.5)
cax = fig.add_axes([0.76, 0.62, 0.02, 0.3]); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(1.3, 2.5)), cax=cax); cb.set_label("−log10 q", fontsize=6); cb.ax.tick_params(labelsize=5)
for v in (0.02, 0.05, 0.1): ax.scatter([], [], s=8 + 900 * v, color="#555555", label=f"{v:.0%}")
ax.legend(title="Gene ratio", loc="upper left", bbox_to_anchor=(1.12, 0.48), fontsize=5.5, title_fontsize=6, labelspacing=1.2, borderaxespad=0)
fig.savefig(O / "A_comparecluster_dotplot.png", dpi=300); fig.savefig(O / "A_comparecluster_dotplot.svg"); plt.close(fig)
# B: Metascape-style heatmap
fig, ax = plt.subplots(figsize=(110 * MM, (24 + 4.2 * len(T)) * MM)); fig.subplots_adjust(left=0.42, right=0.80, top=0.97, bottom=0.2)
M = np.where(Q < 0.25, LQ, np.nan); im = ax.imshow(np.where(Q < 0.05, LQ, np.nan), cmap=cm, vmin=1.3, vmax=2.5, aspect="auto")
ax.imshow(np.where((Q >= 0.05) & (Q < 0.25), 1, np.nan), cmap=LinearSegmentedColormap.from_list("g", ["#E6E6E6", "#E6E6E6"]), aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([f"{b}\n({NANN[b]:,})" for b in BINS]); ax.set_yticks(range(len(T))); ax.set_yticklabels([cap(s) for s in T.name]); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-0.5, 4), minor=True); ax.set_yticks(np.arange(-0.5, len(T)), minor=True); ax.grid(which="minor", color="white", lw=1)
cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04); cb.set_label("−log10 q", fontsize=6); cb.ax.tick_params(labelsize=5)
ax.set_xlabel("Promoter in modules of"); fig.text(0.42, 0.02, "grey: 0.05 ≤ q < 0.25; blank: q ≥ 0.25", fontsize=5, color="#555555")
fig.savefig(O / "B_metascape_heatmap.png", dpi=300); fig.savefig(O / "B_metascape_heatmap.svg"); plt.close(fig); print(len(T), "terms")
