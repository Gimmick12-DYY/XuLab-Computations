# MLX-SREBF2 bHLH-ZIP on E-box DNA: main view with domain labels, two close-ups, bar map with the modelled fragment boxed, PAE with chains on the axes. Frame sizes, 600 dpi, text >= 12 pt.
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "savefig.dpi": 600}); FW, FH = 12.9, 3.45; BL, RD, PU = "#2166AC", "#B2182B", "#6A51A3"; O = "super/dnafix"; A = json.load(open(f"{O}/pos.json"))
def img(ax, f):
    a = np.asarray(Image.open(f).convert("RGB")); ax.imshow(a, interpolation="none"); ax.axis("off"); ax.set_xlim(0, a.shape[1]); ax.set_ylim(a.shape[0], 0); return a.shape
def spread(ps, gap):
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
fig = plt.figure(figsize=(6.3, 5.4)); ax = fig.add_axes([0, 0, 1, 1]); H, W = img(ax, f"{O}/over.png")[:2]; im0 = np.asarray(Image.open(f"{O}/over.png").convert("RGB"))
COL = {"MLX basic": PU, "MLX HLH": BL, "MLX ZIP": "#4393C3", "SREBF2 basic": PU, "SREBF2 HLH": RD, "SREBF2 ZIP": "#D6604D", "E-box CACGTG": "#D98C1F"}; sc = W / 6.3
ys_ = sorted(A["over"], key=lambda n: A["over"][n][1]); print({n: A["over"][n][:2] for n in A["over"]})
def col_(names, x, ha):
    nm = sorted(names, key=lambda n: A["over"][n][1]); ys = spread([A["over"][n][1] for n in nm], .3 * sc)
    for n, y in zip(nm, ys): p = A["over"][n]; ax.annotate(n, xy=p[:2], xytext=(x, np.clip(y, .15 * sc, H - .15 * sc)), ha=ha, va="center", fontsize=12, fontweight="bold", color=COL[n], arrowprops=dict(arrowstyle="-", lw=.6, color="#4D4D4D", shrinkA=1, shrinkB=0), bbox=dict(fc="white", ec="none", pad=.6, alpha=.85))
MID = np.mean([A["over"][n][0] for n in A["over"]]); L_ = [n for n in A["over"] if A["over"][n][0] < MID]; R_ = [n for n in A["over"] if A["over"][n][0] >= MID]; col_(R_, W - .03 * sc, "right")
def row_(names, y):
    nm = sorted(names, key=lambda n: A["over"][n][0]); xs = spread([A["over"][n][0] for n in nm], 1.25 * sc)
    for n, x in zip(nm, xs): p = A["over"][n]; ax.annotate(n, xy=p[:2], xytext=(np.clip(x, .7 * sc, W - .7 * sc), y), ha="center", va="center", fontsize=12, fontweight="bold", color=COL[n], arrowprops=dict(arrowstyle="-", lw=.6, color="#4D4D4D", shrinkA=1, shrinkB=0), bbox=dict(fc="white", ec="none", pad=.6, alpha=.85))
row_([n for n in L_ if n.startswith("SREBF2")], .27 * H); row_([n for n in L_ if n.startswith("MLX")], .70 * H)
fig.savefig("super/af3_MLX_dna_main.png", facecolor="white"); plt.close(fig)
fig = plt.figure(figsize=(6.6, 2.8))
def inset(x, f, title):
    a = fig.add_axes([x / 6.6, .05 / 2.8, 3.2 / 6.6, 2.4 / 2.8]); s2 = img(a, f); a.add_patch(Rectangle((0, 0), s2[1], s2[0], fill=False, ec="#8C8C8C", lw=.6, clip_on=False)); a.text(0, -.04 * s2[0], title, fontsize=13, fontweight="bold", va="bottom"); return a, s2
a, s2 = inset(.05, f"{O}/c1.png", "a  Basic regions in the major groove")
def lab(a, s2, xy, t, c, off):
    a.annotate(t, xy=xy, xytext=(np.clip(xy[0] + off[0] * s2[1], .22 * s2[1], .78 * s2[1]), np.clip(xy[1] + off[1] * s2[0], .07 * s2[0], .93 * s2[0])), color=c, fontsize=12, fontweight="bold", ha="center", va="center", arrowprops=dict(arrowstyle="-", lw=.6, color=c, shrinkA=0, shrinkB=2), bbox=dict(fc="white", ec="none", pad=.5, alpha=.9))
P = A["c1"]; lab(a, s2, P["e1"], "MLX E138–C 3.0 Å", BL, (-.1, .3)); lab(a, s2, P["e2"], "SREBF2 E339–C 2.8 Å", RD, (.1, -.35))
a, s2 = inset(3.35, f"{O}/c2.png", "b  Dimer contact"); P = A["c2"]; lab(a, s2, P["y"], "MLX Y183", BL, (-.2, .2)); lab(a, s2, P["q"], "SREBF2 Q381", RD, (.18, -.2)); mid = np.mean([P["y"], P["q"]], axis=0); a.text(mid[0] + .16 * s2[1], mid[1] + .02 * s2[0], "2.8 Å", fontsize=14, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", pad=.8, alpha=.9))
fig.savefig("super/af3_MLX_dna_close.png", facecolor="white"); plt.close(fig)
# Bar map: interface ticks from the DNA model, modelled fragment boxed.
cols, rows, inl = [], [], False
for l in open(f"{O}/model.cif"):
    if l.startswith("_atom_site."): cols.append(l.strip().split(".")[1]); inl = True; continue
    if inl and l.startswith(("ATOM", "HETATM")): rows.append(l.split())
    elif inl and rows and l.startswith("#"): break
import pandas as pd
D = pd.DataFrame(rows, columns=cols); D = D[D.type_symbol != "H"]; xa = D[D.label_asym_id == "A"]; xb = D[D.label_asym_id == "B"]; ca = xa[["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values; cb = xb[["Cartn_x", "Cartn_y", "Cartn_z"]].astype(float).values; dd = np.linalg.norm(ca[:, None] - cb[None], axis=2) <= 5; Sa = sorted(set(xa.label_seq_id.astype(int).values[dd.any(1)] + 128)); Sb = sorted(set(xb.label_seq_id.astype(int).values[dd.any(0)] + 329)); print("interface residues", len(Sa), len(Sb), Sa[:3], Sa[-3:], Sb[:3], Sb[-3:])
Ch = json.load(open("jie_domains.json"))["MLX_SREBF2"]["chains"]; fig = plt.figure(figsize=(6.6, 1.95)); ax = fig.add_axes([0.11, 0.02, 0.87, 0.96]); mx = 1141; ax.set_xlim(0, mx); ax.set_ylim(-.47, 1.65); FR = {"A": (129, 220, Sa), "B": (330, 401, Sb)}; GR = {"SREBF2": {"basic": None, "HLH": "bHLH-ZIP", "ZIP": None}}
for r, (ch, c) in enumerate(Ch.items()):
    y = 1 - r; ax.add_patch(Rectangle((0, y - .1), c["len"], .2, fc=c["base"], ec="#8C8C8C", lw=.3)); ax.text(-mx * .012, y, c["name"], ha="right", va="center", fontsize=12, fontweight="bold"); tx = []
    for nm, a_, b_, col, kind in c["dom"]:
        ax.add_patch(Rectangle((a_, y - .1), b_ - a_ + 1, .2, fc=col, ec="none")); l = GR.get(c["name"], {}).get(nm, nm)
        if l: tx.append(((a_ + b_) / 2 if l != "bHLH-ZIP" else 365, l))
    xs = spread([t[0] for t in sorted(tx)], mx * .095)
    for x_, (x0, l) in zip(xs, sorted(tx)): ax.annotate(l, xy=(x0, y - .1), xytext=(x_, y - .22), ha="center", va="top", fontsize=12, arrowprops=dict(arrowstyle="-", lw=.4, color="#8C8C8C", shrinkA=0, shrinkB=0))
    a_, b_, St = FR[ch]; ax.add_patch(Rectangle((a_ - 3, y - .14), b_ - a_ + 7, .28, fill=False, ec="#1A1A1A", lw=1.1, zorder=5))
    for v in St: ax.plot([v, v], [y + .17, y + .3], color="#1A1A1A", lw=.8)
    ax.text(b_ + mx * .018, y + .22, f"modelled {a_}–{b_}", fontsize=12, va="center")
ax.axis("off"); fig.savefig("fig/sitemap_MLX_dna.png", facecolor="white"); plt.close(fig)
