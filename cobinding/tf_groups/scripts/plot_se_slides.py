"""Slide figures (palette C): (1) SE capture by cliques, observed vs active-chromatin null vs peak-density-matched null;
(2) why density matters: clique rate by local peak-density decile, and per-TF chromatin-class log2 OR before vs after
density matching. args: tf_groups_dir outdir"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/work/users/d/y/dyy12/XuLab/figures_paletteC"); import regulome_palette as rp
from pathlib import Path
P = rp.use(); D, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 9, "axes.labelsize": 9, "legend.fontsize": 8})
ink, grey, band, blue = P.neutral.ink, P.neutral.mid, P.neutral.pale, P.evidence.cicero_only
# (1) SE capture: observed vs random genome windows (primary) and active-chromatin windows
gn = pd.read_csv(D / "results_se/se_capture_vs_genome_null.tsv", sep="\t", index_col=0)
ac = pd.read_csv(D / "results_se/se_capture_vs_active_null.tsv", sep="\t", index_col=0)
ac = ac.rename(index={"clique_ge1TF":"clique_ge1TF"})
keys = ["any_network_peak", "clique_ge1TF", "clique_ge2TF", "clique_ge5TF", "clique_ge10TF"]
lab = ["Any network peak", "Clique, ≥1 TF", "Clique, ≥2 TFs", "Clique, ≥5 TFs", "Clique, ≥10 TFs"]
fig, ax = plt.subplots(figsize=(6.6, 3.6))
for i, k in enumerate(keys):
    r = gn.loc[k]; y = len(keys) - 1 - i
    ax.plot([r.null_mean, r.observed], [y, y], color=P.neutral.light, lw=1.2, zorder=1)
    ax.scatter(r.null_mean, y, s=50, facecolor="white", edgecolor=grey, lw=1.6, zorder=3)
    if k in ac.index: ax.scatter(ac.loc[k].null_mean, y, s=34, marker="D", facecolor="white", edgecolor=blue, lw=1.4, zorder=2)
    ax.scatter(r.observed, y, s=50, color=ink, zorder=4)
    ax.text(1.02, y, f"{r.fold:.1f}×", va="center", fontsize=8, color=P.neutral.dark, transform=ax.get_yaxis_transform())
ax.text(1.02, len(keys) - 0.45, "fold vs\nrandom", fontsize=7, color=P.neutral.mid, transform=ax.get_yaxis_transform(), va="bottom")
ax.set_yticks(range(len(keys))); ax.set_yticklabels(lab[::-1]); ax.set_xlim(0, 1); ax.set_xlabel("Fraction of super-enhancers captured")
h = [plt.Line2D([], [], marker="o", ls="", color=ink, label="SEdb HEK293T SEs (n = 338)"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=grey, mew=1.6, label="Random genome windows, same size"),
     plt.Line2D([], [], marker="D", ls="", mfc="white", mec=blue, mew=1.4, label="Windows on active chromatin")]
ax.legend(handles=h, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig_se_capture_nulls.png", dpi=300); plt.close(fig)
# (2) density
N = pd.read_csv(D / "results_location/nodes_annotated.tsv.gz", sep="\t", usecols=["tf", "chrom", "start", "end", "clique"])
N["mid"] = (N.start + N.end) // 2; dens = np.zeros(len(N))
for (tf, c), idx in N.groupby(["tf", "chrom"]).groups.items():
    m = np.sort(N.mid.values[idx]); x = N.mid.values[idx]; dens[idx] = np.searchsorted(m, x + 5e5, side="right") - np.searchsorted(m, x - 5e5)
N["dens"] = dens; tfc = N.groupby("tf").clique.agg(["sum", "size"]); tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)]
N = N[N.tf.isin(tfs)]; N["dec"] = N.groupby("tf").dens.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False)) + 1
rate = N.groupby(["tf", "dec"]).clique.mean().unstack()
crude = pd.read_csv(D / "results_location/class_enrichment_per_tf.tsv", sep="\t").groupby("category").log2OR.median()
matched = pd.read_csv(D / "results_location/class_enrichment_density_matched.tsv", sep="\t").set_index("category").median_tf_log2OR
order = ["A compartment", "Enh-active", "TSS-flank", "Transcribed", "Bivalent", "Promoter", "Enh-weak", "Polycomb", "Het/ZNF", "Quiescent"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
q = rate.quantile([.25, .5, .75])
a1.fill_between(rate.columns, q.loc[.25], q.loc[.75], color=band, lw=0, label="TF interquartile range")
a1.plot(rate.columns, q.loc[.5], color=ink, lw=2, marker="o", ms=4, label=f"Median of {len(tfs)} TFs")
a1.set_xticks(range(1, 11)); a1.set_xlabel("Local peak density decile (same-TF network peaks within ±500 kb)"); a1.set_ylabel("Fraction of peaks in a clique")
a1.set_ylim(0, None); a1.legend(frameon=False, loc="upper left"); a1.set_title("Cliques form where peaks are dense", loc="left", fontsize=9)
y = np.arange(len(order))[::-1]
a2.barh(y + .19, crude.reindex(order).values, height=.36, color=P.neutral.light, label="Before matching")
a2.barh(y - .19, matched.reindex(order).values, height=.36, color=ink, label="After density matching")
a2.axvline(0, color=P.neutral.dark, lw=.6); a2.set_yticks(y); a2.set_yticklabels(order); a2.set_xlabel("Clique vs non-clique peak, median per-TF log2 odds ratio")
a2.legend(frameon=False, loc="lower right"); a2.set_title("Chromatin preferences shrink after matching", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_clique_density_effect.png", dpi=300); plt.close(fig)
print(q.round(3).to_string())
