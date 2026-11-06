"""Simple slide figure: % of SEdb HEK293T SEs vs random same-size genome windows captured by our data (palette C,
linear axis, value labels). args: se_capture_vs_genome_null.tsv out.png"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
P = rp.use(); plt.rcParams.update({"font.size": 11, "xtick.labelsize": 11, "ytick.labelsize": 10, "axes.labelsize": 11, "legend.fontsize": 10})
G = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
keys = ["any_network_peak", "clique_ge2TF", "clique_ge5TF"]; lab = ["Any TF peak", "Clique of ≥2 TFs", "Clique of ≥5 TFs"]
o = G.loc[keys, "observed"].values * 100; n = G.loc[keys, "null_mean"].values * 100; x = np.arange(len(keys)); w = .36
fig, ax = plt.subplots(figsize=(6.4, 4.0))
b1 = ax.bar(x - w / 2, o, w, color=P.neutral.ink, label="Super-enhancers (n = 338)")
b2 = ax.bar(x + w / 2, n, w, color=P.neutral.light, label="Random windows, same size")
for b, v in list(zip(b1, o)) + list(zip(b2, n)): ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 110); ax.set_ylabel("% of regions"); ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=300)
