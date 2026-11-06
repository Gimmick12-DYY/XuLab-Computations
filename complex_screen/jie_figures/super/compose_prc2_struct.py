# EZH2-SUZ12-MTF2 structure panel: each label in its domain's colour, each leader ending on a visible pixel of that domain (super/render_prc2_labels.py). Slide frame size, 600 dpi.
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "savefig.dpi": 600})
Ch = json.load(open("jie_domains.json"))["EZH2_SUZ12_MTF2"]["chains"]; A = json.load(open("super/prc2fix/anchors.json")); im = np.asarray(Image.open("super/prc2fix/whole.png").convert("RGB")); H, W = im.shape[:2]
def spread(ps, gap):
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
# Leader tip: nearest pixel to the domain's projected centroid that is well inside the domain and not covered by a side-chain stick.
from scipy.ndimage import binary_erosion
flat = np.asarray(Image.open("super/prc2fix/flat.png").convert("RGB")).astype(int); ids = {}; i = 0; sm = im[::2, ::2][:flat.shape[0], :flat.shape[1]]
for c, D_ in Ch.items():
    for d in D_["dom"]: i += 1; ids[f"{c}:{d[0]}"] = (i * 15, 255 - i * 15, 40 + i * 9)
for k, a in A.items():
    c, nm = k.split(":"); col = np.array(next(d[3] for d in Ch[c]["dom"] if d[0] == nm).lstrip("#")); rgb = np.array([int(str(col)[j:j + 2], 16) for j in (0, 2, 4)]); m = np.abs(flat - np.array(ids[k])).sum(axis=2) <= 6; r = sm.astype(float); hue = np.abs(r / np.maximum(r.sum(axis=2, keepdims=True), 1) - rgb / max(rgb.sum(), 1)).sum(axis=2) < .10; bright = r.sum(axis=2) > .45 * rgb.sum()
    for er in (6, 4, 3, 2, 1, 0):
        ok = (binary_erosion(m, iterations=er) if er else m) & hue & bright
        if ok.sum() >= 5: break
    ys, xs = np.nonzero(ok); j = np.argmin((xs - a["cen"][0]) ** 2 + (ys - a["cen"][1]) ** 2); a["anchor"] = (float(xs[j]) * 2, float(ys[j]) * 2); a["er"] = er
fig = plt.figure(figsize=(6.6, 4.85)); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im, interpolation="none"); ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(H, 0); pts = []
for k, a in A.items():
    c, nm = k.split(":"); frac = a["px"] / max(a["px_solo"], 1); pts.append((*a["anchor"], f"{Ch[c]['name']} {nm}" + (" (behind)" if frac < .25 else ""), next(d[3] for d in Ch[c]["dom"] if d[0] == nm)))
A_er = {f"{Ch[k.split(':')[0]]['name']} {k.split(':')[1]}": a["er"] for k, a in A.items()}; pts.sort(key=lambda p: p[1]); cx = np.mean([p[0] for p in pts]); gap = 0.2 / 4.85 * H
for side in (-1, 1):
    G = [p for p in pts if (p[0] < cx) == (side == -1)]
    ys = list(np.clip(spread([p[1] for p in G], gap), .03 * H, .97 * H))
    for p, y in zip(G, ys): ax.annotate(p[2], xy=p[:2], xytext=((.02 if side == -1 else .98) * W, y), ha="left" if side == -1 else "right", va="center", fontsize=12, fontweight="bold", color=p[3], arrowprops=dict(arrowstyle="-", lw=.6, color="#4D4D4D", shrinkA=1, shrinkB=0), bbox=dict(fc="white", ec="none", pad=.6, alpha=.85)); print(p[2], "left" if side == -1 else "right", p[:2], "pixel at anchor", tuple(int(x) for x in im[int(p[1]), int(p[0])]), "domain", p[3], "inset", A_er[p[2]])
fig.savefig("fig/struct_PRC2.png", facecolor="white"); print(Image.open("fig/struct_PRC2.png").size)
