"""Fig 3e (90 x 52 mm): (left) fraction of a TF's network peaks that are in a clique, by local peak-density decile
(same-TF network peaks within +-500 kb; median and IQR over TFs with >=30 clique and >=30 non-clique peaks);
(right) clique vs non-clique peak chromatin log2 odds ratio (median per TF) before vs after density matching.
Data: cobinding/tf_groups/results_location/{nodes_annotated.tsv.gz, class_enrichment_per_tf.tsv, class_enrichment_density_matched.tsv};
computation as tf_groups/scripts/plot_se_slides.py (fig_clique_density_effect.png)."""
import numpy as np, pandas as pd
from fig3common import P, XULAB, fig_mm, mm_axes, save

W, H = 90, 52
D = XULAB / "cobinding/tf_groups/results_location"
N = pd.read_csv(D / "nodes_annotated.tsv.gz", sep="\t", usecols=["tf", "chrom", "start", "end", "clique"])
N["mid"] = (N.start + N.end) // 2; dens = np.zeros(len(N))
for (tf, c), idx in N.groupby(["tf", "chrom"]).groups.items():
    m = np.sort(N.mid.values[idx]); x = N.mid.values[idx]
    dens[idx] = np.searchsorted(m, x + 5e5, side="right") - np.searchsorted(m, x - 5e5)
N["dens"] = dens; tfc = N.groupby("tf").clique.agg(["sum", "size"])
tfs = tfc.index[(tfc["sum"] >= 30) & (tfc["size"] - tfc["sum"] >= 30)]
N = N[N.tf.isin(tfs)]; N["dec"] = N.groupby("tf").dens.transform(lambda v: pd.qcut(v.rank(method="first"), 10, labels=False)) + 1
rate = N.groupby(["tf", "dec"]).clique.mean().unstack(); q = rate.quantile([.25, .5, .75])
crude = pd.read_csv(D / "class_enrichment_per_tf.tsv", sep="\t").groupby("category").log2OR.median()
matched = pd.read_csv(D / "class_enrichment_density_matched.tsv", sep="\t").set_index("category").median_tf_log2OR
order = ["A compartment", "Enh-active", "TSS-flank", "Transcribed", "Bivalent", "Promoter", "Enh-weak", "Polycomb", "Het/ZNF", "Quiescent"]
print(q.round(3).to_string()); print(pd.DataFrame({"before": crude.reindex(order), "after": matched.reindex(order)}).round(3).to_string())
fig = fig_mm(W, H)
a1 = mm_axes(fig, W, H, 9, 9.5, 31, 40)
a1.fill_between(rate.columns, q.loc[.25], q.loc[.75], color=P.neutral.pale, lw=0, label="TF IQR")
a1.plot(rate.columns, q.loc[.5], color=P.neutral.ink, lw=1, marker="o", ms=2.2, label=f"Median, {len(tfs)} TFs")
a1.set_xticks(range(1, 11), ["1", "", "", "", "5", "", "", "", "", "10"]); a1.set_xlim(0.5, 10.5)
a1.set_ylim(0, 0.5); a1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], ["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
a1.set_xlabel("Local peak density decile", labelpad=1.5); a1.set_ylabel("Fraction of peaks in a clique")
a1.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3)
a2 = mm_axes(fig, W, H, 61, 9.5, 27, 40)
y = np.arange(len(order))[::-1]
a2.barh(y + .19, crude.reindex(order).values, height=.36, color=P.neutral.light, label="Before", lw=0)
a2.barh(y - .19, matched.reindex(order).values, height=.36, color=P.neutral.ink, label="Matched", lw=0)
a2.axvline(0, color=P.neutral.dark, lw=.5); a2.set_yticks(y, order); a2.tick_params(axis="y", length=0)
a2.set_xlim(-0.7, 0.7); a2.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "0.5"]); a2.set_ylim(-0.6, len(order) - 0.4)
a2.set_xlabel("Clique log2 odds ratio", labelpad=1.5)
a2.spines["left"].set_visible(False)
a2.legend(loc="upper left", handlelength=1, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.3)
save(fig, "fig3e_density_effect")
