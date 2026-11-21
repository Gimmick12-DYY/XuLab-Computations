"""Figure 3g-i: one shared legend strip (edge colours and line styles, element classes, 3D-model symbols, TF peak colours).
No data input. Output: ghi_shared_legend.{pdf,svg,png}, 183 x 14 mm, Arial 6 pt.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

W, H = 183.0, 14.0
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "DR1", "MTF2", "E4F1"]


def main(src, out):
    style.use(base_pt=6, small_pt=6)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon
    EC, NODE, TC = style.SE_EDGE, style.SE_NODE, style.SE_TF
    fig = style.figure_mm(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    R = fig.canvas.get_renderer()
    INK = "#1A1A1A"
    y1, y2, y3 = 12.0, 7.0, 2.0

    def seg(x, y, c, lw, ls="-"):
        ax.plot([x, x + 4], [y, y], color=c, lw=lw, ls=ls, solid_capstyle="butt")

    def txt(x, y, t, **k):   # draws the text and returns its measured right edge (mm)
        o = ax.text(x, y, t, va="center", ha="left", color=INK, **k)
        return x + o.get_window_extent(R).width / fig.dpi / style.MM

    x = txt(0.5, y1, "Edges from the SE:", fontweight="bold") + 2
    for k, l in (("se_promoter", "Promoter"), ("se_enhancer", "Enhancer"), ("se_other", "Other")):
        seg(x, y1, EC[k], 1.4)
        x = txt(x + 5, y1, l) + 3
    x = txt(x + 2, y1, "Between partners:", fontweight="bold") + 2
    for k, l in (("pp_promoter", "Promoter–promoter"), ("pp_enhancer", "Involving an enhancer"), ("pp_other", "Other")):
        seg(x, y1, EC[k], 0.8)
        x = txt(x + 5, y1, l) + 3

    x = 0.5
    seg(x, y2, "#555555", 0.8)
    x = txt(x + 5, y2, "On a Hi-C loop") + 3
    seg(x, y2, "#555555", 0.8, (0, (2.5, 2)))
    x = txt(x + 5, y2, "Cicero only") + 5
    x = txt(x, y2, "Elements:", fontweight="bold") + 2
    ax.add_patch(RegularPolygon((x + 1.2, y2), 6, radius=1.25, fc=NODE.se_fill, ec=NODE.se_edge, lw=0.5))
    x = txt(x + 3, y2, "Super-enhancer") + 3
    for c, l in ((NODE.promoter, "Promoter"), (NODE.enhancer, "Enhancer"), (NODE.other, "Other site")):
        ax.add_patch(Circle((x + 1, y2), 0.85, color=c, lw=0))
        x = txt(x + 2.6, y2, l) + 3
    x = txt(x + 2, y2, "3D model (h):", fontweight="bold") + 2
    ax.plot([x, x + 4], [y2, y2], color="#A9A9A9", lw=2.4, solid_capstyle="round")
    x = txt(x + 5, y2, "Chromatin") + 3
    ax.add_patch(Circle((x + 1, y2), 0.85, color="#222222", lw=0))
    txt(x + 2.6, y2, "Promoter / other site")

    x = txt(0.5, y3, "TF peaks (small spheres in h):", fontweight="bold") + 2
    for t in TFS + ["other"]:
        ax.add_patch(Circle((x + 0.8, y3), 0.75, color=TC[t], lw=0))
        x = txt(x + 2.1, y3, t if t != "other" else "Other TFs", fontstyle="italic" if t != "other" else "normal") + 2.5
    style.save(fig, out, "ghi_shared_legend", bbox_inches=style.content_bbox(fig, 0.02))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
