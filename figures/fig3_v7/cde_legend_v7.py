"""Fig 3 v7: one shared legend for panels c-e (edge colours/line styles as in the approved browser, flower and rosette; element
classes; 3D rosette symbols and TF peak colours from se_sedb/scripts/47_rosette_own_beads.py). Final size 183 x 14 mm, Arial 6 pt.
  python cde_legend_v7.py <outdir>"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 6, "svg.fonttype": "none", "pdf.fonttype": 42})
OUT = Path(sys.argv[1]); MM = 1 / 25.4; W, H = 183.0, 14.0
EC = dict(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999", pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#D0D0D0")
TFS = ["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "DR1", "MTF2", "E4F1"]
TC = dict(zip(["ZNF507", "SOX4", "NFYA", "GLYR1", "NME2", "ZNF282", "ADNP", "DR1", "MTF2", "E4F1"], ["#E69F00", "#009E73", "#F0E442", "#CC79A7", "#117733", "#DDCC77", "#44AA99", "#999933", "#AA4499", "#A6761D"]))
fig = plt.figure(figsize=(W * MM, H * MM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
INK = "#1A1A1A"; y1, y2, y3 = 12.0, 7.0, 2.0; R = fig.canvas.get_renderer()
def seg(x, y, c, lw, ls="-"): ax.plot([x, x + 4], [y, y], color=c, lw=lw, ls=ls, solid_capstyle="butt")
def txt(x, y, t, **k):   # draws the text and returns its right edge in mm (measured, not estimated)
    o = ax.text(x, y, t, va="center", ha="left", color=INK, **k); return x + o.get_window_extent(R).width / fig.dpi / MM
x = 0.5; x = txt(x, y1, "Edges from the SE:", fontweight="bold") + 2
for k, l in (("se_promoter", "Promoter"), ("se_enhancer", "Enhancer"), ("se_other", "Other")): seg(x, y1, EC[k], 1.4); x = txt(x + 5, y1, l) + 3
x += 2; x = txt(x, y1, "Between partners:", fontweight="bold") + 2
for k, l in (("pp_promoter", "Promoter–promoter"), ("pp_enhancer", "Involving an enhancer"), ("pp_other", "Other")): seg(x, y1, EC[k], 0.8); x = txt(x + 5, y1, l) + 3
xr1 = x
x = 0.5; seg(x, y2, "#555555", 0.8); x = txt(x + 5, y2, "On a Hi-C loop") + 3; seg(x, y2, "#555555", 0.8, (0, (2.5, 2))); x = txt(x + 5, y2, "Cicero only") + 5
x = txt(x, y2, "Elements:", fontweight="bold") + 2
ax.add_patch(RegularPolygon((x + 1.2, y2), 6, radius=1.25, fc="#FFC20A", ec="#5A4500", lw=0.5)); x = txt(x + 3, y2, "Super-enhancer") + 3
for c, l in (("#0072B2", "Promoter"), ("#D55E00", "Enhancer"), ("#8C8C8C", "Other site")): ax.add_patch(Circle((x + 1, y2), 0.85, color=c, lw=0)); x = txt(x + 2.6, y2, l) + 3
x += 2; x = txt(x, y2, "3D model (d):", fontweight="bold") + 2; ax.plot([x, x + 4], [y2, y2], color="#A9A9A9", lw=2.4, solid_capstyle="round"); x = txt(x + 5, y2, "Chromatin") + 3
ax.add_patch(Circle((x + 1, y2), 0.85, color="#222222", lw=0)); xr2 = txt(x + 2.6, y2, "Promoter / other site")
x = 0.5; x = txt(x, y3, "TF peaks (small spheres in d):", fontweight="bold") + 2
for t in TFS + ["other"]:
    ax.add_patch(Circle((x + 0.8, y3), 0.75, color=TC.get(t, "#BDBDBD"), lw=0)); x = txt(x + 2.1, y3, t if t != "other" else "Other TFs", fontstyle="italic" if t != "other" else "normal") + 2.5
print("legend right edges (mm):", round(xr1, 1), round(xr2, 1), round(x, 1))
for e in ("svg", "pdf"): fig.savefig(OUT / f"cde_legend.{e}")
fig.savefig(OUT / "cde_legend.png", dpi=300)
