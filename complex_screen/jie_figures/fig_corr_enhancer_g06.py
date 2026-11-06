# Enhancer percentile heatmap for the MLX-SREBF2 correlation slide with PowerNorm gamma 0.6 and a two-tone highlight on the MLX-SREBF2 cells (with and without crosshair), 7.0 x 6.25 in, 600 dpi, labels >= 12 pt. TFs ordered by average linkage on 1 - percentile. Writes the order next to the figures.
import json
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import PowerNorm
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
def fig_(pair, block, name, xyt, gamma, cross, cmap="RdYlBu_r"):
    fig = plt.figure(figsize=(7.0, 6.25)); ax = fig.add_axes([0.13, 0.03, 0.70, 0.70 * 7.0 / 6.25]); im = ax.imshow(Mx, cmap=cmap, norm=PowerNorm(gamma, vmin=0, vmax=1), interpolation="nearest"); ax.set_xticks(range(n_)); ax.set_yticks(range(n_)); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.xaxis.tick_top(); ax.tick_params(length=1.5, width=.25); ax.set_xlim(-.5, n_ - .5); ax.set_ylim(n_ - .5, -.5)
    idx = sorted(OT.index(t) for t in block); pi = [OT.index(t) for t in pair]; sp = spread(idx, 13 / 72 / (0.70 * 7.0) * n_)
    for k, y in zip(idx, sp):
        c = "#B2182B" if k in pi else "#1A1A1A"; ax.annotate(OT[k], xy=(-.5, k), xytext=(-3, y), ha="right", va="center", fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0)); ax.annotate(OT[k], xy=(k, -.5), xytext=(y, -3), ha="center", va="bottom", rotation=90, fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0))
    for a in idx:
        for b in idx:
            if a != b and not (a in pi and b in pi): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#000000", lw=.8))
    a, b = pi
    cpt = 0.70 * 7.0 * 72 / n_; pw = .4 / cpt; pk = 1.6 / cpt; po = 2.4 / cpt
    for x, y in ((a, b), (b, a)):
        if cross:
            ax.plot([-.5, x - .5 - po], [y, y], ls=(0, (3, 2)), lw=.7, color="#404040", zorder=5); ax.plot([x, x], [-.5, y - .5 - po], ls=(0, (3, 2)), lw=.7, color="#404040", zorder=5)
        ax.add_patch(Rectangle((x - .5 - pk, y - .5 - pk), 1 + 2 * pk, 1 + 2 * pk, fill=False, ec="#000000", lw=1.6, zorder=6, joinstyle="miter")); ax.add_patch(Rectangle((x - .5 - pw, y - .5 - pw), 1 + 2 * pw, 1 + 2 * pw, fill=False, ec="#FFFFFF", lw=.8, zorder=7, joinstyle="miter"))
    v = M.loc[pair[0], pair[1]]; print(pair, "rows", pi, "value", v, "shown", f2(v), "block rows", {OT[k]: k for k in idx}); tgt = (max(a, b), min(a, b) + .5 + po)
    ax.annotate(f"{pair[0]}–{pair[1]}\npercentile {f2(v)}", xy=tgt, xytext=xyt, fontsize=12, fontweight="bold", color="#000000", ha="center", arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.2), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1A1A1A", lw=.6), zorder=8)
    cax = fig.add_axes([0.86, 0.2, 0.022, 0.36]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, .2, .4, .6, .8, 1]); cb.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"]); cb.ax.tick_params(labelsize=12, width=.3, length=2); cb.outline.set_linewidth(.3); cb.set_label("Percentile of TF–TF Pearson r", fontsize=12); fig.savefig(f"fig/{name}.png", facecolor="white"); plt.close(fig)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
W, H = 7.0, 6.25; rect = [0.13, 0.03, 0.70, 0.70 * 7.0 / 6.25]; cm = matplotlib.colormaps["RdYlBu_r"]; i, j = OT.index("MLX"), OT.index("SREBF2"); hx = lambda c: "#%02X%02X%02X" % tuple(int(round(float(x) * 255)) for x in c[:3]); g = .6; n = PowerNorm(g, vmin=0, vmax=1)
for tag, cross in (("g06", True), ("g06_nocross", False)):
    nm = f"heatmap_enh_MLX_SREBF2_{tag}"; fig_(("MLX", "SREBF2"), ["MLX", "SREBF2"], nm, (60, 74), g, cross); img = np.asarray(Image.open(f"fig/{nm}.png").convert("RGB")); assert img.shape[:2] == (3750, 4200); ax_w, ax_h = rect[2] * W, rect[3] * H; sd = min(ax_w, ax_h); x0 = rect[0] * W + (ax_w - sd) / 2; yt = (1 - rect[1] - rect[3]) * H + (ax_h - sd) / 2; c = sd / n_; out = {}
    for r, q in ((i, j), (j, i)):
        cx, cy = (x0 + (q + .5) * c) * 600, (yt + (r + .5) * c) * 600; hw = c * 300; px = img[int(cy), int(cx)] / 255; want = np.array(cm(float(n(Mx[r, q])))[:3]); d = float(np.abs(px - want).max()); print(tag, "cell", OT[r], OT[q], "value", round(float(Mx[r, q]), 4), "norm", round(float(n(Mx[r, q])), 3), "colour diff at centre", round(d, 3), "hex", hx(want), "cell px", round(c * 600)); out[(r, q)] = (int(cx), int(cy), int(hw))
    ims = []
    for (cx, cy, hw) in out.values(): w = 24 * hw; ims.append(Image.fromarray(img[max(cy - w // 2, 0):cy + w // 2, max(cx - w // 2, 0):cx + w // 2]))
    ch = min(x.size[1] for x in ims); cr = Image.new("RGB", (sum(x.size[0] for x in ims) + 40, ch), "white"); xo = 0
    for x in ims: cr.paste(x.crop((0, 0, x.size[0], ch)), (xo, 0)); xo += x.size[0] + 40
    cr.save(f"fig/{nm}_zoom.png")
v = float(M.loc["MLX", "SREBF2"]); print("before (linear)", round(v, 4), hx(cm(v)))
