"""Supplementary to Figure 3g-i: edge counts of the chr19 super-enhancer hub by partner class (on a Hi-C loop vs Cicero only) and the
direct Hi-C test (05_hic_test.py): observed/expected contact of the 18 SE loop-edges and of all other hub pairs vs 1,000 sets matched
for distance and TF class.

Inputs (source_data/): se_hub_edges.tsv, se_hic_test_10kb.json.   Output: supplementary/S_se_edge_table.{pdf,svg,png}, 3.1 x 5.75 in.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

EN = dict(se_promoter="promoter", se_enhancer="enhancer", se_other="other", pp_promoter="promoter–promoter",
          pp_enhancer="involving an enhancer", pp_other="other")


def main(src, out):
    style.use(base_pt=12, small_pt=12)
    import matplotlib.pyplot as plt
    EC = style.SE_EDGE
    HE = pd.read_csv(src / "se_hub_edges.tsv", sep="\t")
    HE["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(HE.cls_a, HE.cls_b, HE.from_se, HE.type)]
    Rj = json.load(open(src / "se_hic_test_10kb.json"))
    l_, nn_ = Rj["edges_loop_edges_18"], Rj["hub_nonedge_tf"]
    W, H = 3.1, 5.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    ax.text(0.05, H - 0.16, "Edges in the hub", fontsize=13, fontweight="bold", va="center")
    y = H - 0.5
    ax.text(2.25, y, "loop", ha="center", va="center", fontsize=11)
    ax.text(2.78, y, "Cicero", ha="center", va="center", fontsize=11)
    for hd, keys, lw in (("From the super-enhancer", ("se_promoter", "se_enhancer", "se_other"), 3.5),
                         ("Between partners", ("pp_promoter", "pp_enhancer", "pp_other"), 2.2)):
        y -= 0.27
        ax.text(0.05, y, hd, fontweight="bold", va="center", fontsize=11.5)
        for k in keys:
            y -= 0.26
            m = HE.ek == k
            ax.plot([0.07, 0.3], [y, y], color=EC[k], lw=lw, solid_capstyle="butt")
            ax.text(0.37, y, EN[k], va="center", fontsize=10.5)
            ax.text(2.25, y, str(int((m & (HE.on_loop == 1)).sum())), ha="center", va="center")
            ax.text(2.78, y, str(int((m & (HE.on_loop == 0)).sum())), ha="center", va="center")
    y -= 0.28
    ax.text(0.05, y, "solid = on a Hi-C loop; dashed = Cicero only", va="center", fontsize=9.5, color="#444444")
    y -= 0.42
    ax.text(0.05, y, "Direct Hi-C test", fontsize=13, fontweight="bold", va="center")
    y -= 0.4
    ax.text(0.05, y, f"SE loop-edges (18) vs matched:\n{l_['ratio_tf']:.2f}× (p = {l_['p_tf']:.3f})", va="center", linespacing=1.1)
    y -= 0.54
    ax.text(0.05, y, f"Other pairs in the hub, as a set:\n{nn_['ratio']:.2f}× (p = {nn_['p']:.2f}, n.s.)", va="center", linespacing=1.1)
    y -= 0.36
    ax.text(0.05, y, "Matched for distance and TF class,\n1,000 sets. Edges were selected on\nHi-C loops, so enrichment is partly\nexpected.",
            va="top", fontsize=10, color="#444444", linespacing=1.1)
    style.save(fig, Path(out) / "supplementary", "S_se_edge_table")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
