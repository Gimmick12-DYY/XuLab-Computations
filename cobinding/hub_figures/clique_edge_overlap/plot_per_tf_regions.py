"""Per TF: share of its clique regions / cliques / edges that another TF's cliques also contain (same TF order)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({"font.size": 11, "svg.fonttype": "none"})
s = pd.read_csv("edges/per_tf_edge_overlap_summary.tsv", sep="\t")
s = s[(s.cliques >= 20) | (s.tf == "CTCF")].sort_values("pct_regions_shared")
y = np.arange(len(s)); lab = [f"{t} ({c:,})" for t, c in zip(s.tf, s.cliques)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 0.24 * len(s) + 1.8), sharey=True, gridspec_kw={"wspace": 0.08})
a1.barh(y, s.pct_regions_shared, color="#2E7D6B")
a1.set_yticks(y); a1.set_yticklabels(lab, fontsize=9.5)
a1.set_xlabel("% of the TF's clique regions also in another TF's cliques")
a1.set_title("Regions (one shared end is enough)", loc="left", fontsize=12)
for k, v in enumerate(s.pct_regions_shared):
    a1.text(v + 0.6, k, f"{v:.0f}", va="center", fontsize=8, color="#444")
a2.barh(y + 0.2, s.pct_cliques_with_shared_edge, height=0.4, color="#1F4E8C", label="cliques with ≥1 edge another TF also has")
a2.barh(y - 0.2, s.pct_edges_shared, height=0.4, color="#D95F02", label="clique edges another TF also has")
a2.set_xlabel("% of the TF's cliques / edges")
a2.set_title("Edges (both ends must match)", loc="left", fontsize=12)
a2.legend(loc="lower right", frameon=False, fontsize=9.5)
for ax in (a1, a2):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_ylim(-0.7, len(s) - 0.3)
fig.suptitle("Per TF: how much of its clique network other TFs share (TFs with ≥20 cliques, plus CTCF; cliques in brackets)",
             x=0.01, ha="left", y=0.935, fontsize=13)
fig.savefig("edges/fig_per_tf_regions_and_edges_shared.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("edges/fig_per_tf_regions_and_edges_shared.svg", bbox_inches="tight", facecolor="white")
print(s.pct_regions_shared.describe().round(1).to_dict()); print(s.sort_values("pct_regions_shared",ascending=False)[["tf","pct_regions_shared"]].head(6).round(1).values.tolist())
