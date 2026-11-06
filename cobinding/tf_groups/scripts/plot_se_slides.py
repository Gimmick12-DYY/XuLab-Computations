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
# (1) SE capture
dm = pd.read_csv(D / "results_se/se_density_matched_null.tsv", sep="\t", index_col=0)
ac = pd.read_csv(D / "results_se/se_capture_vs_active_null.tsv", sep="\t", index_col=0)
ks = [1, 2, 5, 10]; lab = ["≥1 TF", "≥2 TFs", "≥5 TFs", "≥10 TFs"]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for i, k in enumerate(ks):
    r = dm.loc[f"frac_SE_clique_ge{k}TF"]; a = ac.loc[f"clique_ge{k}TF"]; y = len(ks) - 1 - i
    ax.barh(y, r.null_hi95 - r.null_lo95, left=r.null_lo95, height=0.42, color=band, lw=0)
    ax.plot([r.null_mean] * 2, [y - .25, y + .25], color=grey, lw=2)
    ax.scatter(a.null_mean, y, s=46, facecolor="white", edgecolor=blue, lw=1.6, zorder=3)
    ax.scatter(r.observed, y, s=46, color=ink, zorder=4)
    ax.text(1.02, y, f"p = {r.p_one_sided:.2f}", va="center", fontsize=8, color=P.neutral.dark, transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(ks))); ax.set_yticklabels(lab[::-1]); ax.set_xlim(0, 1); ax.set_xlabel("Fraction of super-enhancers with clique regions")
ax.text(1.02, len(ks) - 0.35, "density-matched\np (enrichment)", fontsize=7, color=P.neutral.mid, transform=ax.get_yaxis_transform(), va="bottom")
h = [plt.Line2D([], [], marker="o", ls="", color=ink, label="Observed SEs (SEdb, n = 338)"),
     plt.Line2D([], [], marker="o", ls="", mfc="white", mec=blue, mew=1.6, label="Active-chromatin windows"),
     plt.Line2D([], [], color=grey, lw=2, label="Peak-density-matched windows (mean)"),
     matplotlib.patches.Patch(color=band, label="Density-matched 95% range")]
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
