# AF3-only figure for the MLX-SREBF2 slide, at slide frame size, 600 dpi, labels >= 12 pt.
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, ConnectionPatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "savefig.dpi": 600}); J = json.load(open("af3only.json")); FW, FH = 12.9, 5.6; BL, RD, PU = "#2166AC", "#B2182B", "#6A51A3"
def img(ax, f):
    a = np.asarray(Image.open(f).convert("RGB")); ax.imshow(a, interpolation="none"); ax.axis("off"); ax.set_xlim(0, a.shape[1]); ax.set_ylim(a.shape[0], 0); return a.shape
fig = plt.figure(figsize=(FW, FH)); ax = fig.add_axes([0, 0, FH / FW, 1]); sh = img(ax, "AF3_over.png"); O = J["over"]
for k, t, c, dx, dy in (("MLX", "MLX", BL, -.05, .24), ("SREBF2", "SREBF2", RD, -.05, -.24)): ax.annotate(t, xy=O[k], xytext=(O[k][0] + dx * sh[1], O[k][1] + dy * sh[0]), color=c, fontsize=14, fontweight="bold", ha="center", arrowprops=dict(arrowstyle="-", lw=.6, color=c, shrinkA=1, shrinkB=0))
ins = []
for rect, f, title in (([0.47, 0.50, 0.25, 0.39], "AF3_c1.png", "Dimer contact"), ([0.47, 0.03, 0.25, 0.39], "AF3_c2.png", "DNA-binding basic regions")):
    a = fig.add_axes(rect); s2 = img(a, f); a.add_patch(plt.Rectangle((0, 0), s2[1], s2[0], fill=False, ec="#8C8C8C", lw=.6, clip_on=False)); a.text(0, -.04 * s2[0], title, fontsize=13, fontweight="bold", va="bottom"); ins.append((a, s2))
a, s2 = ins[0]; P = J["p1"]; mx, my = (P["y"][0] + P["q"][0]) / 2, (P["y"][1] + P["q"][1]) / 2; a.text(mx + .16 * s2[1], my + .02 * s2[0], f"{J['d']} Å", fontsize=14, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9))
for k, t, c, off in (("y", "MLX Y183", BL, (-.2, .2)), ("q", "SREBF2 Q381", RD, (.18, -.2))): a.annotate(t, xy=P[k], xytext=(np.clip(P[k][0] + off[0] * s2[1], .16 * s2[1], .84 * s2[1]), np.clip(P[k][1] + off[1] * s2[0], .08 * s2[0], .92 * s2[0])), color=c, fontsize=13, fontweight="bold", ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=.6, color=c, shrinkA=0, shrinkB=2), bbox=dict(fc="white", ec="none", pad=.5, alpha=.9))
a, s2 = ins[1]; a.text(.5 * s2[1], .07 * s2[0], "Free; model run without DNA", color=PU, fontsize=13, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9))
def fan(xy, r, a, s2):
    ax.add_patch(Circle(xy, r, fc="#BDBDBD", ec="#8C8C8C", alpha=.35, lw=.6, zorder=5))
    for ty, sg in ((0, -1), (s2[0], 1)): fig.add_artist(ConnectionPatch(xyA=(xy[0], xy[1] + sg * r), coordsA=ax.transData, xyB=(0, ty), coordsB=a.transData, color="#1A1A1A", lw=.7))
fan(O["c1"], 240, *ins[0]); fan(O["bas"], 260, *ins[1])
fig.text(.75, .85, "AlphaFold 3", fontsize=15, fontweight="bold")
for k, t in enumerate(("ipTM 0.45", "ipSAE 0.57", "5 of 5 seeds strong")): fig.text(.75, .77 - k * .075, t, fontsize=14)
fig.savefig("af3_MLX.png", facecolor="white"); print(Image.open("af3_MLX.png").size)
