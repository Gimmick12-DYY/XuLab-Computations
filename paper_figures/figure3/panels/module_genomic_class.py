"""Figure 3e: genomic class (HEK293T ChromHMM 18-state, merged to six classes) of all Cicero peaks, all module regions, and module
regions split by how many TFs share the region (union of all TFs' higher-order coordinated binding modules).
Stacked 100 % bars; values printed on segments >= 10 %.

Inputs (source_data/):
  class_annotation_counts_chromhmm.tsv   per TF and class: Cicero peaks, module regions
  class_module_regions_chromhmm.tsv.gz   union module regions: class (cls) and number of TFs (n_tfs)
Output: module_genomic_class.{pdf,svg,png}, axes 62 x 58 mm (print size).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

CLASSES = ["promoter", "enhancer", "other TSS/flank", "transcribed", "repressed/het", "quiescent"]
MERGE = {"active enhancer": "enhancer", "weak/poised enhancer": "enhancer"}


def main(src, out):
    P = style.use()
    import matplotlib.pyplot as plt
    plt.rcParams["axes.linewidth"] = 0.6
    CH = P.chromhmm
    col = [CH.promoter, CH.active_enh, CH.tss_flank, CH.transcribed, CH.repressed, CH.quiescent]
    A = pd.read_csv(src / "class_annotation_counts_chromhmm.tsv", sep="\t")
    A["class"] = A["class"].replace(MERGE)
    A = A.groupby(["tf", "class"], as_index=False)[["peaks", "clique_regions"]].sum()
    H = pd.read_csv(src / "class_module_regions_chromhmm.tsv.gz", sep="\t")
    H["cls"] = H.cls.replace(MERGE)
    H["bin"] = pd.cut(H.n_tfs, [0, 1, 2, 4, 9, 100], labels=["1", "2", "3–4", "5–9", "≥10"])
    bars = {"all Cicero peaks": A.groupby("class").peaks.sum(), "all Multi-region Coordinated\nCluster (MCC) regions": A.groupby("class").clique_regions.sum()}
    ns = {"all Cicero peaks": A.peaks.sum(), "all Multi-region Coordinated\nCluster (MCC) regions": A.clique_regions.sum()}
    for b, d in H.groupby("bin", observed=True):
        lab = f"in MCCs of {b} TF{'s' if b != '1' else ''}"
        bars[lab], ns[lab] = d.cls.value_counts(), len(d)
    B = pd.DataFrame(bars).reindex(CLASSES).fillna(0)
    B = B / B.sum()
    B.to_csv(Path(out) / "module_genomic_class_values.tsv", sep="\t", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(62 * style.MM, 58 * style.MM))
    left = np.zeros(B.shape[1])
    for k, c in zip(CLASSES, col):
        ax.barh(range(B.shape[1]), B.loc[k], left=left, color=c, label=k[0].upper() + k[1:], height=0.72,
                edgecolor=P.neutral.dark, linewidth=0.3)
        for i, (l, v) in enumerate(zip(left, B.loc[k])):
            if v >= 0.10:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=5,
                        color="white" if k in ("promoter", "enhancer", "repressed/het") else P.neutral.ink)
        left += B.loc[k].values
    ax.set_yticks(range(B.shape[1]))
    ax.set_yticklabels([f"{c[0].upper() + c[1:]}\n(n = {ns[c]:,})" for c in B.columns], fontsize=5.5, linespacing=0.95)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of regions")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.2), fontsize=6, handlelength=1.0, handleheight=0.8,
              columnspacing=0.8, labelspacing=0.25)
    style.save(fig, out, "module_genomic_class", bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
