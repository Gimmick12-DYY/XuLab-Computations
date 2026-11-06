"""Fig 3a (78 x 45 mm): clique regions and clique edges by the number of TFs whose cliques contain them.
Data: inputs/edges_and_regions_by_n_tfs.tsv (copy of hub_figures/basic/clique_edge_overlap/, all cliques, blacklist removed).
Linear y axis; TF counts binned 1 / 2 / 3-4 / 5-9 / >=10 as in panels b and d; the 1-TF bar is cut at the top of the axis
(break marks) and labelled with its value."""
import numpy as np, pandas as pd
from fig3common import P, INP, fig_mm, mm_axes, save

W, H = 78, 45
k = pd.read_csv(INP / "edges_and_regions_by_n_tfs.tsv", sep="\t")
BINS = ["1", "2", "3–4", "5–9", "≥10"]
k["bin"] = pd.cut(k.n_tfs, [0, 1, 2, 4, 9, 1000], labels=BINS)
g = k.groupby("bin", observed=False)[["regions", "edges"]].sum()
print(g.to_string())
fig = fig_mm(W, H)
spec = [("regions", "Clique regions", 4000, 1000, 11), ("edges", "Clique edges", 800, 200, 49)]
for col, ylab, top, step, left in spec:
    ax = mm_axes(fig, W, H, left, 9.5, 26, 31)
    v = g[col].values; x = np.arange(len(BINS))
    ax.bar(x, np.minimum(v, top), width=0.7, color=[P.neutral.ink] + [P.neutral.ink] * 4)
    for i, val in enumerate(v):
        if val > top:   # cut bar: white break band + label above the axis
            ax.plot([i - 0.45, i + 0.45], [top * 0.86, top * 0.90], color="white", lw=2.2, clip_on=False, zorder=3)
            ax.text(i, top * 1.02, f"{val:,}", ha="center", va="bottom", fontsize=6)
        else:
            ax.text(i, val + top * 0.02, f"{val:,}", ha="center", va="bottom", fontsize=6)
    ax.set_ylim(0, top); ax.set_yticks(np.arange(0, top + 1, step))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
    ax.set_xticks(x, BINS); ax.set_xlim(-0.6, len(BINS) - 0.4)
    ax.set_ylabel(ylab)
    tot, sh = int(v.sum()), int(v[1:].sum())
    ax.text(0.98, 0.98, f"{sh:,} of {tot:,}\n({100 * sh / tot:.1f}%) in ≥2 TFs", transform=ax.transAxes, ha="right", va="top",
            fontsize=6, color=P.neutral.dark)
fig.text((11 + 13 + 49 + 13) / 2 / W, 1.2 / H, "Number of TFs whose cliques contain it", ha="center", va="bottom")
save(fig, "fig3a_edges_regions_by_n_tfs")
