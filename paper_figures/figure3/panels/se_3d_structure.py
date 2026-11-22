"""Figure 3h: 3D structure of the chr19 super-enhancer hub (one Hi-C-constrained polymer model; no TF input to the model).

The 3D render (se_rosette_render.png) is made with PyMOL by results_3d_model/se_sedb/scripts/47_rosette_own_beads.py (render step; ray-traced,
3000 x 3000 px) from the selected structure; se_rosette_nodes.tsv gives each labelled element's pixel position in that render.
This script crops the render to the square frame shared with the earlier rosette renders, places it, and adds leader lines
and labels for the SE, promoter and enhancer partners (left/right of the structure, spread to avoid overlap).

Inputs (source_data/): se_rosette_render.png, se_rosette_nodes.tsv
Output: se_3d_structure.{pdf,svg,png}; drawn at print size (design canvas x 0.421), labels 6 pt. The render stays raster.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

W, H = 12.95, 5.75                 # design canvas (data units = design inches)
S = 0.421                          # print scale
IX, IY, IW = 1.8, 0.02, 5.3        # where the square render sits on the design canvas
CROP = (147, 101, 2652)            # x0, y0, side (px): square frame of the earlier rosette renders (same camera), so the
                                   # structure keeps the size and position it had in the approved slide figure
TC = dict(se="#8A6A00", promoter="#0072B2", enhancer="#D55E00", other="#666666")


def main(src, out):
    style.use(base_pt=6, small_pt=6)
    import matplotlib.pyplot as plt
    im = Image.open(src / "se_rosette_render.png").convert("RGBA")
    x0, y0, side = CROP
    im = im.crop((x0, y0, x0 + side, y0 + side))
    bg = Image.new("RGB", im.size, "white")          # flatten onto white (transparent PyMOL background)
    bg.paste(im, mask=im.split()[-1])
    im = bg
    T = pd.read_csv(src / "se_rosette_nodes.tsv", sep="\t")
    T["x"] = IX + (T.px - x0) / side * IW
    T["y"] = IY + IW - (T.py - y0) / side * IW

    fig = plt.figure(figsize=(W * S, H * S))
    ax = fig.add_axes([0, 0, 1, 1])
    ys, xs = np.where(np.asarray(im.convert("L")) < 250)      # draw only the structure, not the white frame around it
    px0, px1, py0, py1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    k = IW / side
    ax.imshow(np.asarray(im)[py0:py1, px0:px1], extent=(IX + px0 * k, IX + px1 * k, IY + IW - py1 * k, IY + IW - py0 * k),
              interpolation="lanczos", zorder=0)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    midx = IX + IW / 2
    TL = T[T.cls != "other"]
    for side_, sub in ((-1, TL[TL.x < midx]), (1, TL[TL.x >= midx])):
        sub = sub.sort_values("y")
        yy = list(sub.y.values)
        for i in range(1, len(yy)):
            yy[i] = max(yy[i], yy[i - 1] + 0.3)
        if yy and yy[-1] > H - 0.45:
            yy = [v - (yy[-1] - (H - 0.45)) for v in yy]
        for i in range(len(yy) - 2, -1, -1):
            yy[i] = min(yy[i], yy[i + 1] - 0.3)
        lx = IX - 0.03 if side_ < 0 else IX + IW + 0.03
        for r, ty in zip(sub.itertuples(), yy):
            ax.plot([r.x, lx], [r.y, ty], color="#444444", lw=0.6 * S, zorder=3)
            ax.text(lx + (-0.04 if side_ < 0 else 0.04), ty, r.label, ha="right" if side_ < 0 else "left", va="center",
                    fontstyle="normal" if r.cls == "se" else "italic", fontweight="bold" if r.cls == "se" else "normal", color=TC[r.cls])
    style.save(fig, out, "se_3d_structure", bbox_inches=style.content_bbox(fig, 0.01))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
