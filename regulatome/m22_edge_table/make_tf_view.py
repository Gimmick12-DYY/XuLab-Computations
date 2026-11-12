# m22 TF view: top 30 target genes of a TF by predicted impact (ours) with promoter/distal and shared_by, and the external knockdown z beside it. usage: make_tf_view.py TF [TF ...]
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import glob as _g, matplotlib.font_manager as _fm
for _f in _g.glob("/work/users/d/y/dyy12/fonts/arial/*.ttf"): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "font.size": 6, "axes.linewidth": .4})
O = "/work/users/d/y/dyy12/XuLab/regulatome/m22_edge_table/out"; os.makedirs(f"{O}/views", exist_ok=True); ZC = "observed_kd_z [external: X-Atlas/Orion HEK293T]"; RP = "observed_kd_reproducible [external]"
E = pd.read_csv(f"{O}/regulome_edges.tsv.gz", sep="\t", low_memory=False)
for t in sys.argv[1:]:
    d = E[E.tf == t].sort_values("predicted_impact", ascending=False).head(30).iloc[::-1].reset_index(drop=True); n = len(E[E.tf == t]); y = np.arange(len(d))
    fig, ax = plt.subplots(1, 3, figsize=(120 / 25.4, 100 / 25.4), gridspec_kw=dict(width_ratios=[3, 1.2, 2], wspace=.08), sharey=True)
    ax[0].barh(y, d.predicted_impact, color=np.where(d.promoter_peak, "#2C6FB7", "#E08214"), height=.7); ax[0].set_yticks(y); ax[0].set_yticklabels(d.gene, fontsize=5.5); ax[0].set_xlabel("predicted impact\n(signalValue x ABC, ours)")
    ax[1].barh(y, d.shared_by, color="#BDBDBD", height=.7); ax[1].set_xlabel("TFs bound\nat gene (of 79)")
    z = d[ZC].values; col = np.where(z < 0, "#009E73", "#CC79A7"); ax[2].barh(y, np.nan_to_num(z), color=col, height=.7, alpha=.85); ax[2].axvline(0, color="k", lw=.4)
    for i, r in enumerate(d[RP].fillna(False).values):
        if r: ax[2].text(np.nan_to_num(z[i]), i, " *", va="center", fontsize=6)
    ax[2].set_xlabel("knockdown z, all cells\n(external: X-Atlas/Orion)"); [a.spines[s].set_visible(False) for a in ax for s in ("top", "right")]
    fig.suptitle(f"{t}: top 30 of {n:,} target genes (linked q005 peak, our data)\nblue = promoter peak, orange = distal only; * = reproducible in both halves of cells (external)", fontsize=6)
    fig.subplots_adjust(left=.19, right=.98, bottom=.13, top=.91); [fig.savefig(f"{O}/views/tf_{t}.{e}", dpi=300) for e in ("png", "svg")]; plt.close(fig); print(t, n)
