# One MLX-SREBF2 structure figure: the full-length annotated AF3 model with its domain labels, two close-ups, AF3 scores. Slide frame size, 600 dpi, labels >= 12 pt.
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, ConnectionPatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "savefig.dpi": 600}); FW, FH = 12.9, 3.45; BL, RD, PU = "#2166AC", "#B2182B", "#6A51A3"
A = json.load(open("super/af3only.json")); DOM = json.load(open("jie_domains.json"))["MLX_SREBF2"]["chains"]
def spread(ps, gap):
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
def img(ax, f):
    a = np.asarray(Image.open(f).convert("RGB")); ax.imshow(a, interpolation="none"); ax.axis("off"); ax.set_xlim(0, a.shape[1]); ax.set_ylim(a.shape[0], 0); return a.shape
fig = plt.figure(figsize=(FW, FH)); ww = 6.5; hh = FH; ax = fig.add_axes([0, 0, ww / FW, 1]); sh = img(ax, "img/MLX_SREBF2_sites_whole.png"); H, W = sh[:2]; _g = np.asarray(Image.open("img/MLX_SREBF2_sites_whole.png").convert("L")); _r = np.flatnonzero((_g < 245).any(axis=1)); CH = W * hh / ww; Y0 = max(0, (_r[0] + _r[-1]) / 2 - CH / 2); ax.set_ylim(Y0 + CH, Y0); ax.set_zorder(1); top = fig.add_axes(ax.get_position()); top.set_xlim(0, W); top.set_ylim(Y0 + CH, Y0); top.axis("off"); top.set_zorder(4)
# Leader tip: nearest pixel to the domain's projected centroid well inside the domain and not covered by a side-chain stick (super/render_mlx_labels.py).
from scipy.ndimage import binary_erosion
A2 = json.load(open("super/mlxfix/anchors.json")); flat = np.asarray(Image.open("super/mlxfix/flat.png").convert("RGB")).astype(int); im0 = np.asarray(Image.open("img/MLX_SREBF2_sites_whole.png").convert("RGB")); sm = im0[::2, ::2][:flat.shape[0], :flat.shape[1]].astype(float); ids = {}; i = 0
for c, D_ in DOM.items():
    for d in D_["dom"]: i += 1; ids[f"{c}:{d[0]}"] = (i * 15, 255 - i * 15, 40 + i * 9)
pts = []
for k, a_ in A2.items():
    c, nm = k.split(":"); hx = next(d[3] for d in DOM[c]["dom"] if d[0] == nm); rgb = np.array([int(hx[j:j + 2], 16) for j in (1, 3, 5)]); m = np.abs(flat - np.array(ids[k])).sum(axis=2) <= 6; hue = np.abs(sm / np.maximum(sm.sum(axis=2, keepdims=True), 1) - rgb / max(rgb.sum(), 1)).sum(axis=2) < .10; bright = sm.sum(axis=2) > .45 * rgb.sum()
    for er in (6, 4, 3, 2, 1, 0):
        ok = (binary_erosion(m, iterations=er) if er else m) & hue & bright
        if ok.sum() >= 5: break
    ys, xs = np.nonzero(ok); j = np.argmin((xs - a_["cen"][0]) ** 2 + (ys - a_["cen"][1]) ** 2); pts.append((float(xs[j]) * 2, float(ys[j]) * 2, f"{DOM[c]['name']} {nm}", hx)); print(pts[-1][2], pts[-1][:2], "inset", er, "pixel", tuple(int(x) for x in im0[int(pts[-1][1]), int(pts[-1][0])]), "domain", hx, "visible frac %.2f" % (a_["px"] / max(a_["px_solo"], 1)))
pts.sort(key=lambda p: p[1]); cx = np.mean([p[0] for p in pts]); gap = 0.2 / hh * CH
for side in (-1, 1):
    G = [p for p in pts if (p[0] < cx) == (side == -1)]
    if not G: continue
    ys = list(np.clip(spread([p[1] for p in G], gap), Y0 + .05 * CH, Y0 + .95 * CH))
    for p, y in zip(G, ys): top.annotate(p[2], xy=p[:2], xytext=((.015 if side == -1 else .985) * W, y), ha="left" if side == -1 else "right", va="center", fontsize=12, fontweight="bold", color=p[3], arrowprops=dict(arrowstyle="-", lw=.6, color="#4D4D4D", shrinkA=1, shrinkB=0), bbox=dict(fc="white", ec="none", pad=.6, alpha=.85))
s_ = lambda k: tuple(2 * x for x in A2[k]["cen"]); c1 = tuple(np.mean([s_("A:HLH"), s_("A:ZIP"), s_("B:HLH")], axis=0)); c2 = tuple(np.mean([s_("A:basic"), s_("B:basic")], axis=0))
ins = []
for rect, f, title in (([9.75 / FW, .20 / FH, 3.1 / FW, 2.325 / FH], "super/AF3_c1.png", "b  Dimer contact"), ([6.55 / FW, .20 / FH, 3.1 / FW, 2.325 / FH], "super/AF3_c2.png", "a  DNA-binding basic regions")):
    a = fig.add_axes(rect); a.set_zorder(3); s2 = img(a, f); a.add_patch(plt.Rectangle((0, 0), s2[1], s2[0], fill=False, ec="#8C8C8C", lw=.6, clip_on=False)); a.text(0, -.04 * s2[0], title, fontsize=13, fontweight="bold", va="bottom"); ins.append((a, s2))
a, s2 = ins[0]; P = A["p1"]; mx, my = (P["y"][0] + P["q"][0]) / 2, (P["y"][1] + P["q"][1]) / 2; a.text(mx + .16 * s2[1], my + .02 * s2[0], f"{A['d']} Å", fontsize=14, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9))
for k, t, c, off in (("y", "MLX Y183", BL, (-.2, .2)), ("q", "SREBF2 Q381", RD, (.18, -.2))): a.annotate(t, xy=P[k], xytext=(np.clip(P[k][0] + off[0] * s2[1], .16 * s2[1], .84 * s2[1]), np.clip(P[k][1] + off[1] * s2[0], .08 * s2[0], .92 * s2[0])), color=c, fontsize=13, fontweight="bold", ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=.6, color=c, shrinkA=0, shrinkB=2), bbox=dict(fc="white", ec="none", pad=.5, alpha=.9))
a, s2 = ins[1]; a.text(.5 * s2[1], .07 * s2[0], "Free; model run without DNA", color=PU, fontsize=13, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9))
for xy, r, l in ((c2, 200, "a"), (c1, 240, "b")):
    ax.add_patch(Circle(xy, r, fc="#BDBDBD", ec="#8C8C8C", alpha=.35, lw=.6, zorder=5)); top.text(xy[0] + r * .72, xy[1] + r * .72, l, fontsize=13, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.5, alpha=.9))
fig.savefig("super/af3_MLX_merged2.png", facecolor="white"); print(Image.open("super/af3_MLX_merged2.png").size)
