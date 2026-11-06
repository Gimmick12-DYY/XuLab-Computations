"""Fig 3c (48 x 45 mm): % of SEdb HEK293T super-enhancers (SE_02_1951, n = 338) vs random same-size genome windows that
contain any TF network peak / a clique region of >=2 TFs / of >=5 TFs. Permutation p from 1,000 draws (p = 1/1001 for all three).
Data: cobinding/tf_groups/results_se/se_capture_vs_genome_null.tsv (same table as tf_groups/scripts/plot_se_simple.py)."""
import numpy as np, pandas as pd
from fig3common import P, XULAB, fig_mm, mm_axes, save

W, H = 48, 45
G = pd.read_csv(XULAB / "cobinding/tf_groups/results_se/se_capture_vs_genome_null.tsv", sep="\t", index_col=0)
keys = ["any_network_peak", "clique_ge2TF", "clique_ge5TF"]; lab = ["Any TF\npeak", "Clique\n≥2 TFs", "Clique\n≥5 TFs"]
o = G.loc[keys, "observed"].values * 100; n = G.loc[keys, "null_mean"].values * 100; p = G.loc[keys, "p"].values
print(G.loc[keys].round(4).to_string())
fig = fig_mm(W, H); ax = mm_axes(fig, W, H, 10.5, 9.5, 36, 23)
x = np.arange(3); w = 0.38
b1 = ax.bar(x - w / 2, o, w, color=P.neutral.ink, label="Super-enhancers (n = 338)")
b2 = ax.bar(x + w / 2, n, w, color=P.neutral.light, label="Random windows, same size")
for b, v in list(zip(b1, o)) + list(zip(b2, n)):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(x, lab); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xlim(-0.55, 2.55)
ax.set_ylabel("% of regions")
ax.legend(loc="lower left", bbox_to_anchor=(-0.27, 1.13), handlelength=1, handletextpad=0.4, labelspacing=0.25, borderaxespad=0)
ax.text(1.0, 0.98, f"p = {p.max():.3f} each", transform=ax.transAxes, ha="right", va="top", fontsize=6, color=P.neutral.dark)
save(fig, "fig3c_se_capture")
