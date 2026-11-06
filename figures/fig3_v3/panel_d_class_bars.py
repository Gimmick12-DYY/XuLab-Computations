"""Fig 3d (85 x 52 mm): genomic class of regions (all Cicero peaks, all clique regions, clique regions by number of TFs).
Class rule (as figC1): promoter = RefSeq Select TSS +-1 kb, else HEK293T 18-state ChromHMM at the region midpoint, grouped.
Data: inputs/hub_regions_chromhmm.tsv.gz and inputs/annotation_counts_chromhmm.tsv (copies of hub_figures/basic/descriptive/,
rebuilt by cobinding/hub_figures/descriptive/rebuild_hub_regions_chromhmm.py); plotting as plot_figC_split_paletteC.py."""
import numpy as np, pandas as pd
from fig3common import P, INP, fig_mm, mm_axes, save

W, H = 85, 52
CLASSES = ["promoter", "enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
LABELS = ["Promoter", "Enhancer", "Other TSS/flank", "Transcribed", "Repressed/het", "Quiescent"]
MERGE = {"active enhancer": "enhancer", "weak/poised enhancer": "enhancer"}
CH = P.chromhmm
COL = [CH.promoter, CH.active_enh, CH.tss_flank, CH.transcribed, CH.repressed, CH.quiescent]
A = pd.read_csv(INP / "annotation_counts_chromhmm.tsv", sep="\t"); A["class"] = A["class"].replace(MERGE)
A = A.groupby(["tf", "class"], as_index=False)[["peaks", "clique_regions"]].sum()
Hh = pd.read_csv(INP / "hub_regions_chromhmm.tsv.gz", sep="\t"); Hh["cls"] = Hh.cls.replace(MERGE)
Hh["bin"] = pd.cut(Hh.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3–4", "5–9", "≥10"])
bars = {"All Cicero peaks": A.groupby("class").peaks.sum(), "All clique regions": A.groupby("class").clique_regions.sum()}
ns = {"All Cicero peaks": A.peaks.sum(), "All clique regions": A.clique_regions.sum()}
for b, d in Hh.groupby("bin", observed=True):
    nm = f"{b} TF" + ("" if b == "1" else "s"); bars[nm] = d.cls.value_counts(); ns[nm] = len(d)
B = pd.DataFrame(bars).reindex(CLASSES).fillna(0); B = B / B.sum()
print(B.round(3).to_string()); print(ns)
fig = fig_mm(W, H); ax = mm_axes(fig, W, H, 31, 16, 51, 34)
left = np.zeros(B.shape[1]); y = np.arange(B.shape[1])
for k, lab_, c in zip(CLASSES, LABELS, COL):
    ax.barh(y, B.loc[k], left=left, color=c, label=lab_, height=0.72, lw=0)
    for i, (l, v) in enumerate(zip(left, B.loc[k])):
        if v >= 0.09:
            ax.text(l + v / 2, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if k in ("promoter", "repressed/het") else P.neutral.ink)
    left += B.loc[k].values
ax.set_yticks(y, [f"{c} ({ns[c]:,})" for c in B.columns]); ax.tick_params(axis="y", length=0)
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
ax.set_xlabel("% of regions", labelpad=1.5)
ax.axhline(1.5, color=P.neutral.mid, lw=0.4, xmin=-0.5, clip_on=False)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.32, -0.2), columnspacing=0.8, handlelength=0.9, handletextpad=0.4, borderaxespad=0, labelspacing=0.3)
for s in ("left",):
    ax.spines[s].set_visible(False)
save(fig, "fig3d_class_bars")
