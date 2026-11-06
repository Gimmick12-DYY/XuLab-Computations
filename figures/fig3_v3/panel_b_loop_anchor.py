"""Fig 3b (45 x 45 mm): fraction of clique regions on a Hi-C loop anchor (Peakachu 0.7, WT) by the number of TFs whose cliques
include the region; observed (ink dots) vs the null draws (grey violins); ENCODE blacklist removed.
Data: cobinding/results/ourrun_k50_clique_hubs_blacklist_removed/{obs,null}_loop_by_bin.tsv
(same tables as figures_paletteC/plot_loop_anchor_by_tf_count_paletteC.py). Region counts per bin go to the legend."""
import pandas as pd
from fig3common import P, XULAB, fig_mm, mm_axes, save

W, H = 45, 45
D = XULAB / "cobinding/results/ourrun_k50_clique_hubs_blacklist_removed"
BINS = ["1", "2", "3-4", "5-9", ">=10"]
ob = pd.read_csv(D / "obs_loop_by_bin.tsv", sep="\t").set_index("bin")
nl = pd.read_csv(D / "null_loop_by_bin.tsv", sep="\t")
print(ob.round(3).to_string()); print("null draws", len(nl))
fig = fig_mm(W, H); ax = mm_axes(fig, W, H, 10, 9.5, 33, 33)
data = [nl[b].dropna().values for b in BINS]
vp = ax.violinplot(data, positions=range(len(BINS)), showextrema=False, widths=0.7)
for b in vp["bodies"]:
    b.set_facecolor(P.neutral.light); b.set_edgecolor("none"); b.set_alpha(1)
ax.scatter(range(len(BINS)), ob.loc[BINS, "frac"], color=P.neutral.ink, s=9, zorder=3, label="Observed", lw=0)
ax.scatter([], [], marker="s", s=12, color=P.neutral.light, label="Null")
ax.set_xticks(range(len(BINS)), ["1", "2", "3–4", "5–9", "≥10"]); ax.set_xlim(-0.6, 4.6)
ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.50", "0.75", "1"])
ax.set_xlabel("Number of TFs (cliques)")
ax.set_ylabel("Fraction on a loop anchor")
ax.legend(loc="upper left", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.3)
save(fig, "fig3b_loop_anchor_by_tf_count")
