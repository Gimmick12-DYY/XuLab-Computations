# MLX and SREBF2 motifs on a shared column, shared bases shaded. Slide frame size, 600 dpi, text >= 12 pt.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A", "axes.linewidth": 0.6, "ytick.labelsize": 12, "axes.labelsize": 13, "savefig.dpi": 600})
txt = open("img/jaspar.meme").read(); blk = {b.split()[0]: b for b in txt.split("MOTIF ")[1:]}; COL = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}; FP = FontProperties(family="DejaVu Sans", weight="bold")
def pwm(m):
    L = blk[m].strip().split("\n"); i = [k for k, l in enumerate(L) if l.startswith("letter-prob")][0]; return np.array([[float(x) for x in l.split()] for l in L[i + 1:] if l.strip() and not l.startswith("URL")])
R = [("MA0663.1", "MLX · E-box (MA0663.1)", False, (2, 6), "CACGTG"), ("MA0828.3", "SREBF2 · E-box (MA0828.3)", False, (2, 6), "CACGTG")]; fig = plt.figure(figsize=(8.4, 5.35))
for r, (m, t, rc, (st, n), lab) in enumerate(R):
    P = pwm(m); P = P[::-1, ::-1] if rc else P; ax = fig.add_axes([0.07, 0.535 - r * 0.49, 0.90, 0.40]); ax.add_patch(Rectangle((st - .5, 0), n, 2, fc="#E3E3E3", ec="none", zorder=0)); ic = 2 + (P * np.log2(P + 1e-9)).sum(1)
    for x, (p, h) in enumerate(zip(P, ic)):
        y0 = 0.0
        for k in np.argsort(p):
            ht = p[k] * h
            if ht < 0.02: continue
            ch = "ACGT"[k]; tp = TextPath((0, 0), ch, size=1, prop=FP); b = tp.get_extents(); tr = Affine2D().translate(-b.x0, -b.y0).scale(0.9 / b.width, ht / b.height).translate(x - 0.45, y0); ax.add_patch(PathPatch(tp, transform=tr + ax.transData, fc=COL[ch], lw=0, zorder=2)); y0 += ht
    ax.set_xlim(-.6, 9.6); ax.set_ylim(0, 2); ax.set_yticks([0, 2]); ax.tick_params(length=2, width=.4, pad=1); ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False); ax.set_ylabel("bits", labelpad=-6); ax.text(-.6, 2.08, t, fontsize=14, fontweight="bold", va="bottom"); ax.text(st - .5 + n / 2, 2.08, lab, fontsize=13, color="#4D4D4D", fontweight="bold", ha="center", va="bottom")
    print(m, "".join("ACGT"[k] for k in P.argmax(1)))
fig.savefig("fig/motif_shared2.png", facecolor="white")
