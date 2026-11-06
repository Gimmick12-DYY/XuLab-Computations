# All figures for the rebuilt Jie slides, each at its exact slide frame, 600 dpi, text >= 12 pt. No bbox trimming, so inches here = inches on the slide.
import json, os, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A", "axes.linewidth": 0.6, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelsize": 12, "savefig.dpi": 600})
FR = {}
def done(fig, name): fig.savefig(f"fig/{name}.png", facecolor="white"); FR[name] = [float(x) for x in fig.get_size_inches()]; plt.close(fig)
os.makedirs("fig", exist_ok=True); CC = "../../scripts/complex_catalog"; J = json.load(open(f"{CC}/sp_mats.json")); T = J["tfs"]; W = np.array(J["mats"]["wang18"]["pearson"])
HIS = open(f"{CC}/wang_his_order.py").read().split('"pearson": ("')[1].split('"')[0].lower().split(); o = [T.index(t) for t in HIS]; R = W[np.ix_(o, o)]; n = len(HIS); off = W[~np.eye(len(T), dtype=bool)]; pctW = lambda a, b: (off <= W[T.index(a), T.index(b)]).mean()
def spread(ps, gap):
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
def matrix(pair, block, name):
    fig = plt.figure(figsize=(6.7, 6.2)); ax = fig.add_axes([0.14, 0.03, 0.70, 0.80]); im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.65, vmax=1, interpolation="nearest"); ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.xaxis.tick_top(); ax.tick_params(length=1.5, width=0.25); ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5)
    idx = sorted(HIS.index(t) for t in block); pi = [HIS.index(t) for t in pair]; gap = 13 / 72 / (0.80 * 6.2) * n; sp = spread(idx, gap)
    for k, y in zip(idx, sp):
        c = "#B2182B" if k in pi else "#1A1A1A"; ax.annotate(HIS[k].upper(), xy=(-.5, k), xycoords="data", xytext=(-3.0, y), textcoords="data", ha="right", va="center", fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0))
        ax.annotate(HIS[k].upper(), xy=(k, -.5), xycoords="data", xytext=(y, -3.0), textcoords="data", ha="center", va="bottom", rotation=90, fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0))
    for a in idx:
        for b in idx:
            if a != b and not (a in pi and b in pi): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#1A1A1A", lw=0.7))
    a, b = pi
    for x, y in ((a, b), (b, a)): ax.add_patch(Rectangle((x - .9, y - .9), 1.8, 1.8, fill=False, ec="#000000", lw=1.6, zorder=6))
    ax.annotate(f"{pair[0].upper()}–{pair[1].upper()}\nr = {W[T.index(pair[0]), T.index(pair[1])]:.3f}\npercentile {np.floor(pctW(*pair) * 100) / 100:.2f}", xy=(max(a, b) + .9, min(a, b) - .2), xytext=(55, 28), fontsize=13, fontweight="bold", color="#000000", ha="center", arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.3), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1A1A1A", lw=0.6), zorder=8)
    cax = fig.add_axes([0.87, 0.3, 0.025, 0.34]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=12, width=0.3, length=2); cb.outline.set_linewidth(0.3); cb.set_label("Pearson r", fontsize=12); done(fig, name)
matrix(("ezh2", "mtf2"), ["ezh2", "mtf2", "pcgf6", "rbbp4", "rbbp7"], "matrix_EZH2_MTF2"); matrix(("mlx", "srebf2"), ["mlx", "srebf2"], "matrix_MLX_SREBF2")
P = pd.read_csv("data/r1v2_pairs_provisional.tsv", sep="\t"); lab = {"pearson_domain_pct": "Corrected Pearson", "spqn_noquies_pct": "ChromHMM SpQN", "compartment_spqn_pct": "Compartment SpQN", "cicero_excess_z_pct": "Cicero shared edges"}
def dots(a, b):
    r = P[(P.a == a) & (P.b == b)].iloc[0]; v = [float(r[c]) for c in lab]; y = np.arange(4)[::-1]; fig = plt.figure(figsize=(5.9, 2.7)); ax = fig.add_axes([0.33, 0.24, 0.63, 0.72]); ax.axvline(.9, color="#8C8C8C", ls="--", lw=.9); ax.hlines(y, 0, v, color="#BDBDBD", lw=1); ax.scatter(v, y, s=50, color=["#B2182B" if x >= .9 else "#1A1A1A" for x in v], zorder=3)
    for yy, x in zip(y, v): ax.text(x + .03, yy, f"{np.floor(x * 100 + 1e-9) / 100:.2f}", va="center", fontsize=12)
    ax.set_yticks(y); ax.set_yticklabels(list(lab.values())); ax.set_xlim(0, 1.18); ax.set_ylim(-.5, 3.5); ax.set_xticks([0, .5, .9]); ax.set_xticklabels(["0", "0.5", "0.90"]); ax.set_xlabel("Library-stratified percentile"); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=.4); done(fig, f"dots_{a}_{b}")
dots("EZH2", "MTF2"); dots("MLX", "SREBF2")
C = pd.read_csv("data/cpg_pair_results.tsv", sep="\t").iloc[0]; fig, ax = plt.subplots(1, 3, figsize=(8.8, 3.6)); Xl = ["Both", "One", "Matched"]; Cols = ["#B2182B", "#BDBDBD", "#8C8C8C"]
for a, (k, t) in zip(ax, [("cpg_oe", "CpG obs/exp"), ("gcg_per_kb", "GCG + CGC per kb"), ("cgi_frac", "CpG-island-like")]):
    v = [C[f"{k}_co"], C[f"{k}_single"], C[f"{k}_gcmatched"]]; a.bar(range(3), v, color=Cols, width=.7); a.set_xticks(range(3)); a.set_xticklabels(Xl); a.set_title(t, fontsize=13, loc="left"); a.spines[["top", "right"]].set_visible(False); a.tick_params(length=2, width=.4)
    for x, y in enumerate(v): a.text(x, y, f"{y:.2f}" if y < 5 else f"{y:.1f}", ha="center", va="bottom", fontsize=12)
    a.margins(y=.16)
fig.subplots_adjust(left=.06, right=.99, top=.9, bottom=.1, wspace=.27); done(fig, "cpg_bars")
txt = open("img/jaspar.meme").read(); blk = {b.split()[0]: b for b in txt.split("MOTIF ")[1:]}; COL = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}; FP = FontProperties(family="DejaVu Sans", weight="bold")
def logo(m, title, name):
    L = blk[m].strip().split("\n"); i = [k for k, l in enumerate(L) if l.startswith("letter-prob")][0]; Pm = np.array([[float(x) for x in l.split()] for l in L[i + 1:] if l.strip() and not l.startswith("URL")]); ic = 2 + (Pm * np.log2(Pm + 1e-9)).sum(1); fig = plt.figure(figsize=(4.1, 1.75)); ax = fig.add_axes([0.1, 0.05, 0.88, 0.72])
    for x, (p, h) in enumerate(zip(Pm, ic)):
        y0 = 0.0
        for k in np.argsort(p):
            ht = p[k] * h
            if ht < 0.02: continue
            ch = "ACGT"[k]; tp = TextPath((0, 0), ch, size=1, prop=FP); b = tp.get_extents(); tr = Affine2D().translate(-b.x0, -b.y0).scale(0.9 / b.width, ht / b.height).translate(x - 0.45, y0); ax.add_patch(PathPatch(tp, transform=tr + ax.transData, fc=COL[ch], lw=0)); y0 += ht
    ax.set_xlim(-.6, len(Pm) - .4); ax.set_ylim(0, 2); ax.set_yticks([0, 2]); ax.tick_params(length=2, width=.4, pad=1); ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False); ax.set_ylabel("bits", labelpad=-6); fig.text(0.1, 0.84, title, fontsize=13, fontweight="bold"); done(fig, name)
logo("MA0663.1", "MLX · E-box (MA0663.1)", "logo_mlx"); logo("MA0828.3", "SREBF2 · E-box (MA0828.3)", "logo_srebf2_ebox"); logo("MA0596.1", "SREBF2 · SRE (MA0596.1)", "logo_srebf2_sre")
fig = plt.figure(figsize=(7.4, 1.75)); ax = fig.add_axes([0, 0, 1, 1]); ax.plot([.2, 9.8], [0, 0], color="#1A1A1A", lw=1.2); ax.add_patch(Rectangle((1.5, -.3), 2.2, .6, fc="#2166AC", ec="none")); ax.add_patch(Rectangle((6.3, -.3), 2.2, .6, fc="#B2182B", ec="none")); ax.text(2.6, 0, "CACGTG", ha="center", va="center", color="white", fontsize=13, fontweight="bold"); ax.text(7.4, 0, "TCACNCCAC", ha="center", va="center", color="white", fontsize=12, fontweight="bold"); ax.text(2.6, .45, "MLX E-box", ha="center", fontsize=13); ax.text(7.4, .45, "SREBF2 SRE", ha="center", fontsize=13); ax.annotate("", xy=(3.75, -.5), xytext=(6.25, -.5), arrowprops=dict(arrowstyle="<->", lw=.9, color="#4D4D4D")); ax.text(5, -.95, "≤ 50 bp apart, non-overlapping", ha="center", fontsize=12, color="#4D4D4D"); ax.set_xlim(0, 10); ax.set_ylim(-1.15, .85); ax.axis("off"); done(fig, "halfsite")
DOM = json.load(open("jie_domains.json")); GROUP = {"SREBF2": {"basic": None, "HLH": "bHLH-ZIP", "ZIP": None}}
def sitemap(job, name, sites=None, size=(6.2, 2.75)):
    Ch = DOM[job]["chains"]; S = json.load(open(sites)) if sites else None; fig = plt.figure(figsize=size); ax = fig.add_axes([0.13, 0.04, 0.85, 0.92]); N = len(Ch); mx = max(c["len"] for c in Ch.values()); ax.set_xlim(0, mx); ax.set_ylim(-.4, N - .2)
    for r, (ch, c) in enumerate(Ch.items()):
        y = N - 1 - r; ax.add_patch(Rectangle((0, y - .1), c["len"], .2, fc=c["base"], ec="#8C8C8C", lw=.3)); ax.text(-mx * .012, y, c["name"], ha="right", va="center", fontsize=12, fontweight="bold")
        tx = []
        for nm, a, b, col, kind in c["dom"]:
            ax.add_patch(Rectangle((a, y - .1), b - a + 1, .2, fc=col, ec="none")); l = GROUP.get(c["name"], {}).get(nm, nm)
            if l: tx.append(((a + b) / 2 if l != "bHLH-ZIP" else 365, l))
        xs = spread([t[0] for t in sorted(tx)], mx * .125)
        for x, (x0, l) in zip(xs, sorted(tx)): ax.annotate(l, xy=(x0, y - .1), xytext=(x, y - .3), ha="center", va="top", fontsize=12, arrowprops=dict(arrowstyle="-", lw=.4, color="#8C8C8C", shrinkA=0, shrinkB=0))
        if S:
            for v in S[ch]: ax.plot([v, v], [y + .12, y + .27], color="#1A1A1A", lw=.8)
    ax.axis("off"); done(fig, name)
sitemap("MLX_SREBF2", "sitemap_MLX_SREBF2", "img/MLX_SREBF2.sites.json"); sitemap("EZH2_SUZ12_MTF2", "sitemap_PRC2_placeholder", None, (12.5, 3.9))
def pae(job, name):
    Z = np.load(f"img/{job}.pae.npz"); M = np.array(Z["pae"], float); ch = Z["chains"]; fig = plt.figure(figsize=(2.75, 2.35)); ax = fig.add_axes([0.05, 0.05, 0.68, 0.8]); im = ax.imshow(M, cmap="Greys_r", vmin=0, vmax=30, interpolation="nearest"); b = np.flatnonzero(ch[1:] != ch[:-1]) + .5
    for x in b: ax.axhline(x, color="#B2182B", lw=.8); ax.axvline(x, color="#B2182B", lw=.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Predicted error (PAE)", fontsize=12, loc="left"); cax = fig.add_axes([0.76, 0.2, 0.05, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 15, 30]); cb.ax.tick_params(labelsize=12, width=.3, length=2); cb.outline.set_linewidth(.3); cb.set_label("Å", fontsize=12); done(fig, name)
pae("MLX_SREBF2", "pae_MLX_SREBF2")
def struct(job, name, pre=""):
    im = np.asarray(Image.open(f"img/{job}_sites_whole.png").convert("RGB")); H, Wd = im.shape[:2]; L = json.load(open(f"img/{job}.labels.json")); Ch = DOM[job]["chains"]; fig = plt.figure(figsize=(6.6, 4.85)); ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(im, interpolation="none"); ax.axis("off"); ax.set_xlim(0, Wd); ax.set_ylim(H, 0); sx = Wd / 1700 if max(v[0] for v in L.values()) <= 1700 else 1; pts = []
    for k, (x, y) in L.items():
        c, nm = k.split(":"); pts.append((x * sx, y * sx, f"{Ch[c]['name']} {nm}", next(d[3] for d in Ch[c]["dom"] if d[0] == nm)))
    pts.sort(key=lambda p: p[1]); cx = np.mean([p[0] for p in pts]); gap = 0.2 / 4.85 * H
    for side in (-1, 1):
        G = [p for p in pts if (p[0] < cx) == (side == -1)]
        if not G: continue
        ys = spread([p[1] for p in G], gap); ys = list(np.clip(ys, .03 * H, .97 * H))
        for p, y in zip(G, ys): ax.annotate(p[2], xy=(p[0], p[1]), xytext=((.02 if side == -1 else .98) * Wd, y), ha="left" if side == -1 else "right", va="center", fontsize=12, fontweight="bold", color=p[3] if p[3] not in ("#8C8C8C", "#BDBDBD", "#F4A582", "#92C5DE") else "#4D4D4D", arrowprops=dict(arrowstyle="-", lw=.6, color="#4D4D4D", shrinkA=1, shrinkB=0), bbox=dict(fc="white", ec="none", pad=.6, alpha=.85))
    done(fig, name)
struct("MLX_SREBF2", "struct_MLX_SREBF2")

def dimer(job, name):
    im = np.asarray(Image.open(f"img/{job}_dimer.png").convert("RGB")); Hh, Wd = im.shape[:2]; L = json.load(open(f"img/{job}.dimer_labels.json")); nz = np.argwhere(np.any(im < 245, axis=2)); y0, x0 = nz.min(0); y1, x1 = nz.max(0); pad = 40; y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0); y1, x1 = min(y1 + pad, Hh), min(x1 + pad, Wd); Cr = im[y0:y1, x0:x1]; fig = plt.figure(figsize=(6.6, 3.9)); fw, fh = 6.6, 3.9; sc = min(fw * .98 / (x1 - x0), fh * .62 / (y1 - y0)); aw, ah = (x1 - x0) * sc, (y1 - y0) * sc; ax = fig.add_axes([(fw - aw) / 2 / fw, (fh - ah) / 2 / fh, aw / fw, ah / fh]); ax.imshow(Cr, interpolation="none"); ax.axis("off"); ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0); pt = lambda n: (L[n][0] - x0, L[n][1] - y0); Ht = y1 - y0
    st = dict(fontsize=13, fontweight="bold", ha="center", annotation_clip=False)
    for n, col, off in (("MLX", "#2166AC", (0, 1.2)), ("SREBF2", "#B2182B", (-.3, -.13)), ("Zipper", "#4D4D4D", (-.1, -.13)), ("HLH", "#4D4D4D", (0, 1.2)), ("Interface", "#8C8C8C", (0, 1.2))):
        x, y = pt(n); ax.annotate(n, xy=(x, y), xytext=(x + off[0] * Ht, off[1] * Ht), color=col, arrowprops=dict(arrowstyle="-", lw=.7, color=col, shrinkA=1, shrinkB=0), **st)
    a, b = pt("Basic region: DNA contact"), pt("Basic region "); tx = (max(a[0], b[0]) - .1 * Ht, -.13 * Ht); ax.annotate("Basic: DNA contact", xy=a, xytext=tx, color="#6A51A3", arrowprops=dict(arrowstyle="-", lw=.7, color="#6A51A3", shrinkA=1, shrinkB=0), **st); ax.annotate("", xy=b, xytext=(tx[0], tx[1] + .04 * Ht), arrowprops=dict(arrowstyle="-", lw=.7, color="#6A51A3", shrinkA=1, shrinkB=0), annotation_clip=False); done(fig, name)
dimer("MLX_SREBF2", "dimer_MLX_SREBF2")
for a in sys.argv[1:]:
    if a == "prc2": sitemap("EZH2_SUZ12_MTF2", "sitemap_PRC2", "img/EZH2_SUZ12_MTF2.sites.json", (6.2, 2.8)); pae("EZH2_SUZ12_MTF2", "pae_PRC2"); struct("EZH2_SUZ12_MTF2", "struct_PRC2")

# two-panel correlation: raw state-level matrix and the corrected element-level percentile matrix, same order, same annotations, library strip on both
Fr = pd.read_csv("data/frac_reads_by_class.tsv", sep="\t", index_col=0); Fr.index = Fr.index.str.lower(); HIGH = (Fr.all_elements > .22).to_dict(); Pn = pd.read_csv(f"{CC}/mats/pct_noquies.tsv", sep="\t", index_col=0); Pn.index = Pn.index.str.strip().str.lower(); Pn.columns = Pn.columns.str.strip().str.lower(); Pm = Pn.loc[HIS, HIS].values; WH = R; ey = ~np.eye(n, dtype=bool); fl2 = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"
def within(Mx, a, b):
    gv = np.array([HIGH[t] for t in HIS]); i, j = HIS.index(a), HIS.index(b); assert gv[i] == gv[j]; sm = (gv[:, None] == gv[i]) & (gv[None, :] == gv[i]) & ey; return (Mx[sm] <= Mx[i, j]).mean(), (Mx[ey] <= Mx[i, j]).mean()
NUM = {}
def panel(fig, rect, Mx, cmap, vmin, vmax, cl, pair, block, text, tx):
    ax = fig.add_axes(rect); im = ax.imshow(Mx, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5); idx = sorted(HIS.index(t) for t in block); pi = [HIS.index(t) for t in pair]; hgt = rect[3] * fig.get_size_inches()[1]; sp = spread(idx, 13 / 72 / hgt * n)
    for k in range(n): ax.add_patch(Rectangle((-2.1, k - .5), 1.1, 1, fc="#1A1A1A" if HIGH[HIS[k]] else "#BDBDBD", ec="none", clip_on=False)); ax.add_patch(Rectangle((k - .5, -2.1), 1, 1.1, fc="#1A1A1A" if HIGH[HIS[k]] else "#BDBDBD", ec="none", clip_on=False))
    for k, y in zip(idx, sp):
        c = "#B2182B" if k in pi else "#1A1A1A"; ax.annotate(HIS[k].upper(), xy=(-2.1, k), xytext=(-5, y), ha="right", va="center", fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0)); ax.annotate(HIS[k].upper(), xy=(k, -2.1), xytext=(y, -5), ha="center", va="bottom", rotation=90, fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0))
    for a in idx:
        for b in idx:
            if a != b and not (a in pi and b in pi): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#1A1A1A", lw=.7))
    a, b = pi
    for x, y in ((a, b), (b, a)): ax.add_patch(Rectangle((x - .9, y - .9), 1.8, 1.8, fill=False, ec="#000000", lw=1.6, zorder=6))
    ax.annotate(text, xy=(max(a, b) + .9, min(a, b) - .2), xytext=tx, fontsize=12, fontweight="bold", color="#000000", ha="center", arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.2), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1A1A1A", lw=.6), zorder=8)
    cax = fig.add_axes([rect[0] + rect[2] + .008, rect[1] + .2, .012, .38]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=12, width=.3, length=2); cb.outline.set_linewidth(.3); cb.set_label(cl, fontsize=12)
def matrix2(pair, block, name, tx1, tx2):
    a, b = pair; wr, ar = within(WH, a, b); wc, ac = within(Pm, a, b); r = WH[HIS.index(a), HIS.index(b)]; NUM[f"{a}-{b}"] = dict(raw_r=round(float(r), 3), raw_all=round(float(ar), 3), raw_within=round(float(wr), 3), corrected_all=round(float(ac), 3), corrected_within=round(float(wc), 3))
    fig = plt.figure(figsize=(12.9, 5.05)); Hgt = 0.70
    panel(fig, [0.085, 0.03, 0.3505 * 5.05 / 5.05 * Hgt * 5.05 / (0.3505 * 12.9) * 0.3505, Hgt], WH, "RdYlBu_r", .65, 1, "Pearson r", pair, block, f"Within library block {fl2(wr)}\nAll pairs {fl2(ar)}\n(r = {r:.3f})", tx1)
    panel(fig, [0.59, 0.03, Hgt * 5.05 / 12.9, Hgt], Pm, "RdYlBu_r", 0, 1, "Percentile (SpQN r)", pair, block, f"Corrected {fl2(ac)}\nWithin library block {fl2(wc)}", tx2)
    fig.text(0.085, 0.93, "Raw: 18 ChromHMM states (Dr. Wang)", fontsize=13, fontweight="bold"); fig.text(0.59, 0.93, "Corrected: ChromHMM elements, SpQN", fontsize=13, fontweight="bold"); done(fig, name)
matrix2(("ezh2", "mtf2"), ["ezh2", "mtf2", "pcgf6", "rbbp4", "rbbp7"], "matrix2_EZH2_MTF2", (58, 26), (58, 26)); matrix2(("mlx", "srebf2"), ["mlx", "srebf2"], "matrix2_MLX_SREBF2", (56, 30), (56, 30)); json.dump(NUM, open("fig/corr_numbers.json", "w"), indent=1); print(NUM)

# our normalized ChromHMM matrix (deck slide 184): SpQN percentile on all 398,576 elements, deck dendrogram order and colour scale
Fr2 = pd.read_csv("data/frac_reads_by_class.tsv", sep="\t", index_col=0); Fr2.index = Fr2.index.str.lower(); HI2 = (Fr2.all_elements > .22).to_dict()
def rd(k):
    m = pd.read_csv(f"{CC}/mats/pct_{k}.tsv", sep="\t", index_col=0); m.index = m.index.str.strip().str.lower(); m.columns = m.columns.str.strip().str.lower(); return m
PA = rd("all"); OT = list(PA.index); ALLM = {k: rd(k).loc[OT, OT].values for k in ("all", "noquies", "tss", "enhancer")}
def rd2(k):
    m = pd.read_csv(f"../../repressive_chromhmm_matrix/data/pct_{k}.tsv", sep="\t", index_col=0); m.index = m.index.str.strip().str.lower(); m.columns = m.columns.str.strip().str.lower(); return m.loc[OT, OT].values
ALLM["repressive"] = rd2("repressive"); ALLM["polycomb"] = rd2("polycomb"); ey2 = ~np.eye(len(OT), dtype=bool); gv2 = np.array([HI2[t] for t in OT]); NB = {}
def nums(a, b):
    i, j = OT.index(a), OT.index(b); assert gv2[i] == gv2[j]; sm = (gv2[:, None] == gv2[i]) & (gv2[None, :] == gv2[i]) & ey2; return {k: (float(M[i, j]), float((M[sm] <= M[i, j]).mean())) for k, M in ALLM.items()}
LC_ = {"ezh2": ("0.93–0.96", "0.76–0.90"), "mlx": ("0.44–0.81", "0.44–0.73")}
def norm_matrix(pair, block, name):
    Mx = ALLM["all"]; n_ = len(OT); fig = plt.figure(figsize=(7.0, 6.25)); ax = fig.add_axes([0.13, 0.03, 0.70, 0.70 * 7.0 / 6.25]); im = ax.imshow(Mx, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest"); ax.set_xticks(range(n_)); ax.set_yticks(range(n_)); ax.set_xticklabels([]); ax.set_yticklabels([]); ax.xaxis.tick_top(); ax.tick_params(length=1.5, width=.25); ax.set_xlim(-.5, n_ - .5); ax.set_ylim(n_ - .5, -.5)
    idx = sorted(OT.index(t) for t in block); pi = [OT.index(t) for t in pair]; sp = spread(idx, 13 / 72 / (0.70 * 7.0) * n_)
    for k, y in zip(idx, sp):
        c = "#B2182B" if k in pi else "#1A1A1A"; ax.annotate(OT[k].upper(), xy=(-.5, k), xytext=(-3, y), ha="right", va="center", fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0)); ax.annotate(OT[k].upper(), xy=(k, -.5), xytext=(y, -3), ha="center", va="bottom", rotation=90, fontsize=12, fontweight="bold", color=c, annotation_clip=False, arrowprops=dict(arrowstyle="-", lw=.5, color=c, shrinkA=0, shrinkB=0))
    for a in idx:
        for b in idx:
            if a != b and not (a in pi and b in pi): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#000000", lw=.8))
    a, b = pi
    for x, y in ((a, b), (b, a)): ax.add_patch(Rectangle((x - 1, y - 1), 2, 2, fill=False, ec="#000000", lw=1.7, zorder=6))
    N = nums(*pair); NB["-".join(pair)] = N; f2 = lambda v: f"{np.floor(v * 100 + 1e-9) / 100:.2f}"
    ax.annotate(f"{pair[0].upper()}–{pair[1].upper()}\nThis matrix {f2(N['all'][0])}\nCorrected {LC_[pair[0]][0]}\nLocal cluster {LC_[pair[0]][1]}", xy=(max(a, b) + 1, min(a, b)), xytext=(38, 76) if pair[0] == "ezh2" else (60, 68), fontsize=12, fontweight="bold", color="#000000", ha="center", arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.2), bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1A1A1A", lw=.6), zorder=8)
    cax = fig.add_axes([0.86, 0.2, 0.022, 0.36]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=12, width=.3, length=2); cb.outline.set_linewidth(.3); cb.set_label("Percentile of TF–TF Pearson r", fontsize=12); done(fig, name)
norm_matrix(("ezh2", "mtf2"), ["ezh2", "mtf2", "pcgf6", "rbbp4", "rbbp7"], "normmat_EZH2_MTF2"); norm_matrix(("mlx", "srebf2"), ["mlx", "srebf2"], "normmat_MLX_SREBF2"); 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
CSET = {"all": "data/corr_sets/pct_all.tsv", "noquies": "data/corr_sets/pct_noquies.tsv", "tss": "data/corr_sets/pct_tss.tsv", "enhancer": "../../repressive_chromhmm_corrected/data_min20/pct_enhancer_biv.tsv", "repressive": "../../repressive_chromhmm_corrected/data_min20/pct_repressive.tsv", "polycomb": "../../repressive_chromhmm_corrected/data_min20/pct_polycomb.tsv"}
def corrected(a, b, K=4):
    """corrected percentile, and its rank among pairs in the pair's dendrogram cluster on that same matrix, tree cut at K clusters"""
    out = {}
    for k, f in CSET.items():
        m = pd.read_csv(f, sep="\t", index_col=0); m.index = m.index.str.strip().str.lower(); m.columns = m.columns.str.strip().str.lower(); Mx = m.loc[OT, OT].values; D = 1 - Mx; D = (D + D.T) / 2; np.fill_diagonal(D, 0); c = fcluster(linkage(squareform(D, checks=False), "average"), K, "maxclust"); i, j = OT.index(a), OT.index(b); assert c[i] == c[j], (a, b, k); ids = np.flatnonzero(c == c[i]); sub = Mx[np.ix_(ids, ids)][~np.eye(len(ids), dtype=bool)]; out[k] = (float(Mx[i, j]), float((sub <= Mx[i, j]).mean()), int(len(ids)))
    return out
for k_, pr_ in (("ezh2-mtf2", ("ezh2", "mtf2")), ("mlx-srebf2", ("mlx", "srebf2"))): NB[k_ + "_corrected"] = corrected(*pr_)
json.dump(NB, open("fig/norm_numbers.json", "w"), indent=1); print(NB, [OT.index(t) for t in ("ezh2", "mtf2", "mlx", "srebf2")])
json.dump(FR, open("fig/frames.json", "w")); print({k: Image.open(f"fig/{k}.png").size for k in FR})
