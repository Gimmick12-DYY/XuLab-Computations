"""Fig 3 v7 panel b ("module distribution"; v7: final-size paper render, no title, 2-column frameless legend, thin dark bar edges,
value labels only on segments >= 10%). From Fig 3 v6 panel e: the approved genomic-class bars (figC1_class_bars_paletteC, the bar part of Fig 3c on deck slide 11 / S1C slide 60),
re-run from hub_figures/basic/descriptive/plot_figC_split_paletteC.py (bar part only), same data, colours and layout.
v6 changes: wording only ("in N TFs' cliques" -> "in modules of N TFs", title names higher-order co-binding modules), figure size set
to the panel size, PDF written.
  python e_class_bars_v6.py <outdir> [w h mm]"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ROOT = Path("/Users/dyy/Desktop/HEK293T-ATGTAG")
sys.path.insert(0, str(ROOT / "palette"))
import regulome_palette as rp
P = rp.use(); plt.rcParams.update({"legend.frameon": False, "axes.linewidth": 0.6})
HERE = ROOT / "hub_figures/basic/descriptive"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SZ = [float(v) for v in sys.argv[2:4]] if len(sys.argv) >= 4 else [62, 58]
MM = 1 / 25.4
CLASSES = ["promoter", "enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
MERGE = {"active enhancer": "enhancer", "weak/poised enhancer": "enhancer"}
CH = P.chromhmm
COL = [CH.promoter, CH.active_enh, CH.tss_flank, CH.transcribed, CH.repressed, CH.quiescent]
A = pd.read_csv(HERE / "annotation_counts_chromhmm.tsv", sep="\t"); A["class"] = A["class"].replace(MERGE); A = A.groupby(["tf", "class"], as_index=False)[["peaks", "clique_regions"]].sum()
if (HERE / "hub_regions_chromhmm.tsv.gz").exists():
    H = pd.read_csv(HERE / "hub_regions_chromhmm.tsv.gz", sep="\t"); H["cls"] = H.cls.replace(MERGE)
    H["bin"] = pd.cut(H.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3–4", "5–9", "≥10"])
    lab = lambda b: f"in modules of {b} TF{'s' if b != '1' else ''}"
    bars = {"all Cicero peaks": A.groupby("class").peaks.sum(), "all module regions": A.groupby("class").clique_regions.sum()}
    ns = {"all Cicero peaks": A.peaks.sum(), "all module regions": A.clique_regions.sum()}
    for b, d in H.groupby("bin", observed=True):
        bars[lab(b)] = d.cls.value_counts(); ns[lab(b)] = len(d)
    B = pd.DataFrame(bars).reindex(CLASSES).fillna(0); B = B / B.sum()
    fig, ax = plt.subplots(figsize=(SZ[0] * MM, SZ[1] * MM)); left = np.zeros(B.shape[1])
    for k, c in zip(CLASSES, COL):
        ax.barh(range(B.shape[1]), B.loc[k], left=left, color=c, label=k[0].upper() + k[1:], height=0.72, edgecolor="#4D4D4D", linewidth=0.3)
        for i, (l, v) in enumerate(zip(left, B.loc[k])):
            if v >= 0.10: ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=5, color="white" if k in ("promoter", "enhancer", "repressed/het") else P.neutral.ink)
        left += B.loc[k].values
    ax.set_yticks(range(B.shape[1])); ax.set_yticklabels([f"{c[0].upper() + c[1:]}\n(n = {ns[c]:,})" for c in B.columns], fontsize=5.5, linespacing=0.95); ax.tick_params(axis="y", length=0, pad=1.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Fraction of regions")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.2), fontsize=6, handlelength=1.0, handleheight=0.8, columnspacing=0.8, labelspacing=0.25)
    for ext in ("svg", "pdf", "png"): fig.savefig(OUT / f"b_class_bars.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print(B.round(3).to_string())
