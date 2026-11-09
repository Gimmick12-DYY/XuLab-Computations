"""Fig 3e v5 (84 x 45 mm): genomic class of regions by number of TFs (data of figC1 / fig3_v3 panel_d_class_bars.py).
Class rule (figure legend, not in the panel): promoter = RefSeq Select TSS +-1 kb, else HEK293T 18-state ChromHMM at the
region midpoint, grouped (active + weak/poised enhancer = enhancer). Bars: all accessible peaks, all module regions, then
regions in higher-order co-binding modules by the number of TFs bound. Inputs (inputs/, copies of fig3_v3/inputs):
annotation_counts_chromhmm.tsv, hub_regions_chromhmm.tsv.gz. Colours follow the v5 code (red/blue reserved for loop/Cicero)."""
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
from fig3common import INP, INK, PALE, FS, ENH, fig_mm, mm_axes, save, plt

CLASSES = ["promoter", "enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
LABELS = ["Promoter", "Enhancer", "TSS/flank", "Transcribed", "Repressed", "Quiescent"]
COL = [INK, ENH, "#F2C28F", "#A9B388", "#5E5E5E", PALE]
WHITE = {"promoter", "enhancer", "repressed/het"}
MERGE = {"active enhancer": "enhancer", "weak/poised enhancer": "enhancer"}
A = pd.read_csv(INP / "annotation_counts_chromhmm.tsv", sep="\t"); A["class"] = A["class"].replace(MERGE)
A = A.groupby(["tf", "class"], as_index=False)[["peaks", "clique_regions"]].sum()
Hh = pd.read_csv(INP / "hub_regions_chromhmm.tsv.gz", sep="\t"); Hh["cls"] = Hh.cls.replace(MERGE)
Hh["bin"] = pd.cut(Hh.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1 TF", "2 TFs", "3–4 TFs", "5–9 TFs", "≥10 TFs"])
bars = {"all": A.groupby("class").peaks.sum(), "mod": A.groupby("class").clique_regions.sum()}
ns = {"all": int(A.peaks.sum()), "mod": int(A.clique_regions.sum())}
for b, d in Hh.groupby("bin", observed=True): bars[b] = d.cls.value_counts(); ns[b] = len(d)
B = pd.DataFrame(bars).reindex(CLASSES).fillna(0); B = B / B.sum() * 100
print(B.round(1).to_string()); print(ns)
rows = ["all", "mod", None] + list(B.columns[2:])          # None = small header row
names = {"all": f"All accessible peaks (n = {ns['all']:,})", "mod": f"All module regions (n = {ns['mod']:,})"}
names.update({b: f"{b} ({ns[b]:,})" for b in B.columns[2:]})
ypos, y = {}, 0.0
for r in rows:
    ypos[r] = y; y += 1.0 + (0.35 if r == "mod" else 0)
W, H = 84.0, 45.0; X0, PW = 37.0, 44.3
fig = fig_mm(W, H); ax = mm_axes(fig, W, H, X0, 11.5, PW, 32.8)
for r in rows:
    if r is None: continue
    left = 0.0
    for k, c in zip(CLASSES, COL):
        v = B.at[k, r]; ax.barh(ypos[r], v, left=left, color=c, height=0.74, lw=0)
        if v >= 10: ax.text(left + v / 2, ypos[r], f"{v:.0f}", ha="center", va="center", fontsize=FS, color="white" if k in WHITE else INK)
        left += v
ax.set_yticks([ypos[r] for r in rows if r], [names[r] for r in rows if r]); ax.tick_params(axis="y", length=0, pad=1.5)
ax.text(-(X0 - 1.0) / PW, ypos[None], "Regions in higher-order co-binding modules of", transform=ax.get_yaxis_transform(), ha="left", va="center",
        fontsize=FS, color=INK)
ax.set_ylim(y - 0.5, -0.6); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("% of regions", labelpad=1, fontsize=FS); ax.tick_params(axis="x", length=1.5, pad=1)
ax.spines["left"].set_visible(False)
lg = fig.add_axes([0, 0, 1, 1]); lg.set_xlim(0, W); lg.set_ylim(0, H); lg.set_axis_off(); lg.patch.set_alpha(0)
x = 1.0
for lab_, c in zip(LABELS, COL):
    lg.add_patch(Rectangle((x, 1.3), 2.0, 1.8, color=c, lw=0)); t = lg.text(x + 2.6, 2.2, lab_, fontsize=FS, va="center")
    fig.canvas.draw(); x = lg.transData.inverted().transform(t.get_window_extent().get_points())[1, 0] + 1.8
print("legend right edge (mm)", round(x - 1.8, 1))
save(fig, "fig3e_class_bars")
