# m24 Part A figure: 79 TFs ordered by fragments; FRiP, fraction of q005 summits in union ATAC (70-80% red-flag band), height-ATAC Spearman. Diagnostic.
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6, "axes.linewidth": .4})
O = "out"; A = pd.read_csv(f"{O}/partA_per_tf.tsv", sep="\t").sort_values("fragments", ascending=False).reset_index(drop=True); R = json.load(open(f"{O}/partA_reference.json")); x = np.arange(len(A))
fig, ax = plt.subplots(3, 1, figsize=(180 / 25.4, 110 / 25.4), sharex=True)
ax[0].bar(x, A.FRiP_q005, color="#555555", width=.75); ax[0].set_ylabel("FRiP\n(reads in own q005 peaks)")
ax[1].axhspan(.7, .8, color="#D55E00", alpha=.18, lw=0); ax[1].text(len(A) - .5, .75, "70-80% red-flag band", ha="right", va="center", fontsize=5, color="#A04000")
ax[1].bar(x, A.frac_summits_in_atac_q005, color="#2C6FB7", width=.75, label="q005 peaks"); ax[1].scatter(x, A.frac_summits_in_atac_ascalled_p01, s=4, c="k", zorder=3, label="as-called p < 0.01")
ax[1].set_ylim(0, 1); ax[1].set_ylabel("fraction of summits\nin union ATAC peaks"); ax[1].legend(frameon=False, fontsize=5, loc="upper left")
ax[2].bar(x, A.spearman_tfcpm_vs_atac, color="#E08214", width=.75, label="TF CPM vs ATAC"); ax[2].scatter(x, A.spearman_sig_vs_atac, s=4, c="k", zorder=3, label="signalValue vs ATAC")
ax[2].axhline(R["spearman_H3K27ac_vs_ATAC_union_centres"], color="#2A9D8F", lw=.6, ls="--"); ax[2].text(len(A) - .5, R["spearman_H3K27ac_vs_ATAC_union_centres"] + .02, f"H3K27ac vs ATAC reference = {R['spearman_H3K27ac_vs_ATAC_union_centres']:.2f}", ha="right", fontsize=5, color="#2A9D8F")
ax[2].axhline(0, color="k", lw=.3); ax[2].set_ylabel("Spearman, peak height\nvs ATAC (peaks in ATAC)"); ax[2].legend(frameon=False, fontsize=5, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=2)
ax[2].set_xticks(x); ax[2].set_xticklabels(A.tf, rotation=90, fontsize=4.8); ax[2].set_xlim(-.7, len(A) - .3)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.text(.01, .005, "TFs ordered by HITAG fragments (left = deepest). HEK293T; diagnostic.", fontsize=5); fig.tight_layout(rect=(0, .02, 1, 1))
[fig.savefig(f"{O}/partA_overview.{e}", dpi=300) for e in ("png", "svg")]
