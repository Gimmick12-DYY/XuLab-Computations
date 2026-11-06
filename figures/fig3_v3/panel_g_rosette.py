"""Fig 3g (63 x 72 mm): Hi-C-constrained 3D model of the chr16p11.2 promoter hub (conformation 3338 of the Hi-C-first arm),
re-laid out from the PyMOL render of results_3d_model/tf_polymer_hicfirst/scripts/15_simple_figure.py (--figure step) at the
panel size, without its title and footnote; promoter labels and the legend kept. The render is embedded at ~900 dpi.
Inputs: inputs/rosette_render.png (2600 px PyMOL render), inputs/rosette_promoters.tsv (pixel positions + red-edge degree)."""
import numpy as np, pandas as pd
from PIL import Image
from matplotlib.patches import Circle
from fig3common import P, INP, fig_mm, save

W, H = 63.0, 72.0
im = np.asarray(Image.open(INP / "rosette_render.png").convert("RGBA")); ys, xs = np.where(im[..., 3] > 5)
cx_, cy_ = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; half = int(max(np.ptp(xs), np.ptp(ys)) / 2 * 1.04)
x0, y0 = max(0, cx_ - half), max(0, cy_ - half); cr = im[y0:cy_ + half, x0:cx_ + half]; ch_, cw_ = cr.shape[:2]
IMW = 41.0; IMH = IMW * ch_ / cw_; IX = 11.0
rows = np.where(cr[..., 3].max(1) > 5)[0]; cbot = (ch_ - rows.max()) / ch_ * IMH   # transparent margin under the content
IY = 26.0 - cbot   # content starts 26 mm above the panel bottom, legend below it
T = pd.read_csv(INP / "rosette_promoters.tsv", sep="\t")
T["x"] = IX + (T.px - x0) / cw_ * IMW; T["y"] = IY + IMH - (T.py - y0) / ch_ * IMH
lab = T.sort_values(["degree", "bead"], ascending=[False, True]).head(10)
fig = fig_mm(W, H); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off()
ax.imshow(cr, extent=(IX, IX + IMW, IY, IY + IMH), interpolation="none", zorder=0)
mid = IX + IMW / 2
for side, sub in ((-1, lab[lab.x < mid]), (1, lab[lab.x >= mid])):
    sub = sub.sort_values("y"); yy = list(sub.y.values)
    for k in range(1, len(yy)): yy[k] = max(yy[k], yy[k - 1] + 3.0)
    if yy and yy[-1] > IY + IMH: yy = [v - (yy[-1] - (IY + IMH)) for v in yy]
    lx = IX - 0.5 if side < 0 else IX + IMW + 0.5
    for r, ty in zip(sub.itertuples(), yy):
        ax.plot([r.x, lx], [r.y, ty], color=P.neutral.dark, lw=0.35, zorder=3)
        ax.text(lx - 0.4 * (side < 0) + 0.4 * (side > 0), ty, r.label, ha="right" if side < 0 else "left", va="center",
                fontstyle="italic", fontsize=6, zorder=4)
ly = 21.0
ax.plot([4, 8], [ly, ly], color="#9AB8D8", lw=2.5, solid_capstyle="round"); ax.text(9.5, ly, "Chromatin loop", va="center", fontsize=6)
ax.add_patch(Circle((34, ly), 1.0, color="#222222", lw=0)); ax.text(35.8, ly, "Hub promoter", va="center", fontsize=6)
ax.plot([4, 8], [ly - 4, ly - 4], color="#C0392B", lw=1); ax.text(9.5, ly - 4, "Shared edge on a Hi-C loop", va="center", fontsize=6)
save(fig, "fig3g_rosette")
