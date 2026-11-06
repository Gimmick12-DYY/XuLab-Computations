"""Fig 3b (palette C): fraction of hub regions on a Hi-C loop anchor by number of TFs, observed vs null (blacklist removed).
Plotting copy of fig4() in cobinding/scripts/plot_clique_hub_figures.py, reading its result tables
(results/ourrun_k50_clique_hubs_blacklist_removed/{obs,null}_loop_by_bin.tsv, copied to fig4_inputs_blacklist_removed/)."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use()
D = Path(__file__).resolve().parent / "fig4_inputs_blacklist_removed"
OUT = ROOT / "figures_paletteC"; OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
BINS = ["1", "2", "3-4", "5-9", ">=10"]
ob = pd.read_csv(D / "obs_loop_by_bin.tsv", sep="\t").set_index("bin")
nl = pd.read_csv(D / "null_loop_by_bin.tsv", sep="\t")
fig, ax = plt.subplots(figsize=(62 * MM, 50 * MM))
data = [nl[b].dropna().values for b in BINS]
keep = [i for i, x in enumerate(data) if len(x)]
vp = ax.violinplot([data[i] for i in keep], positions=keep, showextrema=False, widths=0.7)
for b in vp["bodies"]:
    b.set_facecolor(P.neutral.light); b.set_edgecolor("none"); b.set_alpha(1)
ax.scatter(range(len(BINS)), ob.loc[BINS, "frac"], color=P.neutral.ink, s=12, zorder=3, label="observed")
for i, b in enumerate(BINS):
    ax.text(i, ob.loc[b, "frac"] + 0.04, f"n={int(ob.loc[b, 'n_regions']):,}", ha="center", fontsize=5.5)
ax.set_xticks(range(len(BINS)), ["1", "2", "3–4", "5–9", "≥10"])
ax.set_xlabel("Number of TFs whose cliques include the region")
ax.set_ylabel("Fraction of regions on a Hi-C loop anchor")
ax.set_title("ENCODE blacklist removed", loc="left")
ax.scatter([], [], marker="s", s=12, color=P.neutral.light, label="null")
ax.legend(loc="upper left")
for ext in ("svg", "png"):
    fig.savefig(OUT / f"fig4_loop_anchor_by_tf_count_paletteC.{ext}", bbox_inches="tight")
print(ob.round(3).to_string())
