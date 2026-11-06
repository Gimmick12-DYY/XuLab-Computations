# Enhancer percentile heatmap for the MLX-SREBF2 correlation slide, 7.0 x 6.25 in, 600 dpi, labels >= 12 pt. TFs ordered by average linkage on 1 - percentile. Writes the order next to the figures.
import json
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A", "axes.linewidth": 0.6, "savefig.dpi": 600})
M = pd.read_csv("data/corr_sets_extra/pct_enhancer_biv.tsv", sep="\t", index_col=0); M.index = M.index.str.strip().str.upper(); M.columns = M.columns.str.strip().str.upper(); M = M.loc[M.index, M.index]; assert np.allclose(M.values, M.values.T, atol=1e-3); Dm = 1 - (M.values + M.values.T) / 2; np.fill_diagonal(Dm, 0); dv = squareform(Dm, checks=False); Z = optimal_leaf_ordering(linkage(dv, "average"), dv); OT = list(M.index[leaves_list(Z)]); Mx = M.loc[OT, OT].values; n_ = len(OT); json.dump(OT, open("fig/heatmap_enh_order.json", "w"))
def spread(ps, gap):
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
f2 = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"
def fig_(pair, block, name, xyt, cmap="RdYlBu_r", vmin=0, vmax=1):
    fig = plt.figure(figsize=(7.0, 6.25)); ax = fig.add_axes([0.13, 0.03, 0.70, 0.70 * 7.0 / 6.25]); im = ax.imshow(Mx, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"); ax.set_xticks(range(n_)); ax.set_yticks(range(n_)); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.xaxis.tick_top(); ax.tick_params(length=1.5, width=.25); ax.set_xlim(-.5, n_ - .5); ax.set_ylim(n_ - .5, -.5)
    idx = sorted(OT.index(t) for t in block); pi = [OT.index(t) for t in pair]; sp = spread(idx, 13 / 72 / (0.70 * 7.0) * n_)
    for k, y in zip(idx, sp):
        c = "#B2182B" if k in pi else "#1A1A1A"; ax.annotate(OT[k], xy=(-.5, k), xytext=(-3, y), ha="right", va="center", fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0)); ax.annotate(OT[k], xy=(k, -.5), xytext=(y, -3), ha="center", va="bottom", rotation=90, fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0))
    for a in idx:
        for b in idx:
            if a != b and not (a in pi and b in pi): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#000000", lw=.8))
    a, b = pi
    for x, y in ((a, b), (b, a)): ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, fill=False, ec="#000000", lw=1.2, zorder=6))
    v = M.loc[pair[0], pair[1]]; print(pair, "rows", pi, "value", v, "shown", f2(v), "block rows", {OT[k]: k for k in idx}); tgt = (max(a, b), min(a, b))
    ax.annotate(f"{pair[0]}–{pair[1]}\npercentile {f2(v)}", xy=tgt, xytext=xyt, fontsize=12, fontweight="bold", color="#000000", ha="center", arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.2), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1A1A1A", lw=.6), zorder=8)
    cax = fig.add_axes([0.86, 0.2, 0.022, 0.36]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=12, width=.3, length=2); cb.outline.set_linewidth(.3); cb.set_label("Percentile of TF–TF Pearson r", fontsize=12); fig.savefig(f"fig/{name}.png", facecolor="white"); plt.close(fig)
fig_(("MLX", "SREBF2"), ["MLX", "SREBF2"], "heatmap_enh_MLX_SREBF2", (60, 74))
# Check: box centres on the MLX-SREBF2 cells, colour at the cell against the table, label order
from PIL import Image
Image.MAX_IMAGE_PIXELS = None; W, H = 7.0, 6.25; rect = [0.13, 0.03, 0.70, 0.70 * 7.0 / 6.25]; img = np.asarray(Image.open("fig/heatmap_enh_MLX_SREBF2.png").convert("RGB")); assert img.shape[:2] == (3750, 4200); ax_w, ax_h = rect[2] * W, rect[3] * H; sd = min(ax_w, ax_h); x0 = rect[0] * W + (ax_w - sd) / 2; yt = (1 - rect[1] - rect[3]) * H + (ax_h - sd) / 2; c = sd / n_; cm = matplotlib.colormaps["RdYlBu_r"]; i, j = OT.index("MLX"), OT.index("SREBF2"); out = {}
for r, q in ((i, j), (j, i)):
    cx, cy = (x0 + (q + .5) * c) * 600, (yt + (r + .5) * c) * 600; hw = c * 300; px = img[int(cy), int(cx)] / 255; d = float(np.abs(px - np.array(cm(float(Mx[r, q]))[:3])).max()); edges = [img[int(cy), int(cx - hw)], img[int(cy), int(cx + hw)], img[int(cy - hw), int(cx)], img[int(cy + hw), int(cx)]]; dark = all(e.max() < 90 for e in edges); print("cell", OT[r], OT[q], "value", round(float(Mx[r, q]), 3), "colour diff", round(d, 3), "box on cell edges", dark, "centre px", int(cx), int(cy)); out[(r, q)] = (int(cx), int(cy), int(hw))
ims = []
for (cx, cy, hw) in out.values(): w = 24 * hw; ims.append(Image.fromarray(img[max(cy - w // 2, 0):cy + w // 2, max(cx - w // 2, 0):cx + w // 2]))
ch = min(x.size[1] for x in ims); cr = Image.new("RGB", (sum(x.size[0] for x in ims) + 40, ch), "white"); xo = 0
for x in ims: cr.paste(x.crop((0, 0, x.size[0], ch)), (xo, 0)); xo += x.size[0] + 40
cr.save("fig/heatmap_enh_MLX_SREBF2_zoom.png")
