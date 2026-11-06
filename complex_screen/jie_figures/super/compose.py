# Compose overview + close-ups at slide frame size, 600 dpi, labels >= 12 pt: reference grey, model in one colour, translucent circle over the contact, lines fanning to the close-up, dashed distances in Angstrom.
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, ConnectionPatch
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "savefig.dpi": 600}); J = json.load(open("super.json")); FW, FH = 12.9, 5.6
def img(ax, f):
    a = np.asarray(Image.open(f).convert("RGB")); ax.imshow(a, interpolation="none"); ax.axis("off"); ax.set_xlim(0, a.shape[1]); ax.set_ylim(a.shape[0], 0); return a.shape
def inset(fig, rect, f, title, P, dist, labs, col):
    ax = fig.add_axes(rect); sh = img(ax, f); ax.add_patch(plt.Rectangle((0, 0), sh[1], sh[0], fill=False, ec="#8C8C8C", lw=.6, clip_on=False))
    for (a, b), t, c, off in dist:
        mx, my = (P[a][0] + P[b][0]) / 2 + off[0] * sh[1], (P[a][1] + P[b][1]) / 2 + off[1] * sh[0]; ax.text(np.clip(mx, .1 * sh[1], .9 * sh[1]), np.clip(my, .08 * sh[0], .92 * sh[0]), t, color=c, fontsize=13, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9))
    for k, t, c, off in labs:
        x, y = P[k]; ax.annotate(t, xy=(x, y), xytext=(np.clip(x + off[0] * sh[1], .14 * sh[1], .86 * sh[1]), np.clip(y + off[1] * sh[0], .06 * sh[0], .94 * sh[0])), color=c, fontsize=12, fontweight="bold", ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=1), bbox=dict(fc="white", ec="none", pad=.5, alpha=.9))
    ax.text(0, -.04 * sh[0], title, fontsize=13, fontweight="bold", va="bottom"); return ax
def over(fig, f, O, rw, labels):
    ax = fig.add_axes([0.0, 0.0, FH / FW, 1.0]); sh = img(ax, f); pts = np.array([O[k] for k in O]); 
    for k, t, c, dx, dy in labels:
        x, y = O[k]; ax.annotate(t, xy=(x, y), xytext=(x + dx * sh[1], y + dy * sh[0]), color=c, fontsize=13, fontweight="bold", ha="center", annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.6, color=c, shrinkA=1, shrinkB=0), bbox=dict(fc="white", ec="none", pad=.5, alpha=.85))
    return ax, sh
def fan(fig, ax, xy, r, ins, col="#8C8C8C"):
    ax.add_patch(Circle(xy, r, fc="#BDBDBD", ec="#8C8C8C", alpha=.35, lw=.6, zorder=5)); sh = ins.get_xlim()[1], ins.get_ylim()[0]
    for ty in (0, sh[1]): fig.add_artist(ConnectionPatch(xyA=(xy[0], xy[1] + (-r if ty == 0 else r)), coordsA=ax.transData, xyB=(0, ty), coordsB=ins.transData, color="#1A1A1A", lw=.7))
m = J["MLX"]; fig = plt.figure(figsize=(FW, FH)); ax, sh = over(fig, "MLX_over.png", m["over"], 0, [("MLX", "MLX", "#B2182B", -.08, .22), ("SREBF2", "SREBF2", "#B2182B", -.14, -.16), ("MAX", "MAX", "#6E6E6E", .05, .28), ("MYC", "MYC", "#6E6E6E", .12, -.2), ("dna", "DNA", "#6E6E6E", .22, .28)]); i1 = inset(fig, [0.47, 0.50, 0.25, 0.39], "MLX_c1.png", "Dimer contact: Tyr–Gln", m["p1"], [(("ra", "rb"), f"{m['c1']['ref']} Å", "#1A1A1A", (.13, 0)), (("ma", "mb"), f"{m['c1']['model']} Å", "#B2182B", (-.13, 0))], [("ra", "MAX Y70", "#4D4D4D", (.2, -.16)), ("rb", "MYC Q422", "#4D4D4D", (.22, .16)), ("ma", "MLX Y183", "#B2182B", (-.22, .1)), ("mb", "SREBF2 Q381", "#B2182B", (-.2, -.2))], "#B2182B"); i2 = inset(fig, [0.47, 0.03, 0.25, 0.39], "MLX_c2.png", "Basic region to E-box: Glu–adenine", m["p2"], [(("ra", "rb"), f"{m['c2']['ref']} Å", "#1A1A1A", (.13, -.02)), (("ma", "mb"), f"{m['c2']['model']} Å", "#B2182B", (-.1, .12))], [("ra", "MYC E378", "#4D4D4D", (.12, -.22)), ("rb", "Adenine", "#4D4D4D", (.18, .2)), ("ma", "SREBF2 E339", "#B2182B", (-.1, -.22))], "#B2182B"); i2.text(.5 * i2.get_xlim()[1], .94 * i2.get_ylim()[0], "AF3 model run without DNA", fontsize=12, fontweight="bold", color="#B2182B", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9)); fan(fig, ax, m["over"]["c1"], 230, i1); fan(fig, ax, m["over"]["c2"], 230, i2)
fig.text(.75, .9, "PDB 1NKP", fontsize=14, fontweight="bold"); fig.text(.75, .84, f"Cα RMSD {m['rmsd']} Å ({m['n']} residues)", fontsize=13)
fig.savefig("super_MLX.png", facecolor="white"); plt.close(fig)
m = J["PRC2"]; fig = plt.figure(figsize=(FW, FH)); ax, sh = over(fig, "PRC2_over.png", m["over"], 0, [("MTF2", "MTF2 C-term", "#2166AC", .17, -.14), ("PHF19", "PHF19", "#6E6E6E", -.16, -.12), ("SUZ12", "SUZ12", "#6E6E6E", -.1, .3)]); i1 = inset(fig, [0.46, 0.2, 0.27, 0.55], "PRC2_c1.png", "PCL contact on SUZ12: Gln–Tyr", m["p1"], [(("ra", "rb"), f"{m['c1']['ref']} Å", "#1A1A1A", (-.14, .03)), (("ma", "mb"), f"{m['c1']['model']} Å", "#2166AC", (.1, -.08))], [("ra", "SUZ12 Q95", "#4D4D4D", (-.1, -.25)), ("rb", "PHF19 Y542", "#4D4D4D", (-.12, .28)), ("mb", "MTF2 Y555", "#2166AC", (-.03, -.25))], "#2166AC"); fan(fig, ax, m["over"]["c1"], 260, i1)
fig.text(.75, .9, "PDB 6NQ3", fontsize=14, fontweight="bold"); fig.text(.75, .84, f"Cα RMSD {m['rmsd']} Å ({m['n']} residues)", fontsize=13)
fig.savefig("super_PRC2.png", facecolor="white"); plt.close(fig); print(Image.open("super_MLX.png").size, Image.open("super_PRC2.png").size)
