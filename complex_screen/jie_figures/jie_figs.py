# Figures for the 11/2 Jie slides: correlation crop + percentile dots, motif logos. Paper palette C, Cambria.
import re, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 11, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A", "axes.linewidth": 0.6, "savefig.dpi": 300})
M = pd.read_csv("../../scripts/complex_catalog/mats/enrich.tsv", sep="\t", index_col=0); M.index = M.index.str.strip().str.upper(); M.columns = M.columns.str.strip().str.upper()
def crop(tfs, pair, out, ttl):
    A = M.loc[tfs, tfs].values.copy(); np.fill_diagonal(A, np.nan); cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#F2F2F2"); n = len(tfs)
    fig, ax = plt.subplots(figsize=(3.1, 2.9)); im = ax.imshow(A, cmap=cm, vmin=-0.8, vmax=0.8, interpolation="nearest"); ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(tfs, rotation=90, fontsize=9); ax.set_yticklabels(tfs, fontsize=9); ax.tick_params(length=1.5, width=0.4)
    i, j = tfs.index(pair[0]), tfs.index(pair[1])
    for a, b in ((i, j), (j, i)): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#1A1A1A", lw=1.8))
    for k in (i, j): ax.get_xticklabels()[k].set_fontweight("bold"); ax.get_yticklabels()[k].set_fontweight("bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8, width=0.4, length=2); cb.outline.set_linewidth(0.4); cb.set_label("Corrected Pearson r", fontsize=9); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig); return M.loc[pair[0], pair[1]]
def dots(vals, out):
    fig, ax = plt.subplots(figsize=(3.1, 1.55)); names = list(vals); y = np.arange(len(names))[::-1]
    ax.axvline(0.90, color="#8C8C8C", ls="--", lw=0.9); ax.hlines(y, 0, list(vals.values()), color="#BDBDBD", lw=1)
    ax.scatter(list(vals.values()), y, s=38, color=["#B2182B" if v >= 0.90 else "#1A1A1A" for v in vals.values()], zorder=3)
    for yy, v in zip(y, vals.values()): ax.text(min(v + .04, .98), yy, f"{np.floor(v * 100 + 1e-9) / 100:.2f}", va="center", fontsize=9, ha="left")  # floored, as in the deck tables, so 0.897 does not read as 0.90
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.set_xlim(0, 1.12); ax.set_xticks([0, .5, .9]); ax.set_xticklabels(["0", "0.5", "0.90"], fontsize=9); ax.set_xlabel("Percentile among TF pairs", fontsize=9); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=0.4); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
P = pd.read_csv("pairs_values.tsv", sep="\t")
lab = {"pearson_domain_pct": "Corrected Pearson", "spqn_noquies_pct": "ChromHMM SpQN", "compartment_spqn_pct": "Compartment SpQN", "cicero_excess_z_pct": "Cicero shared edges"}
for a, b in (("MLX", "SREBF2"), ("HOXC4", "RBBP4")):
    r = P[(P.a == a) & (P.b == b)].iloc[0]; dots({lab[c]: float(r[c]) for c in lab}, f"img/dots_{a}_{b}.png")
print("r MLX-SREBF2", crop(["EZH2", "MTF2", "MAZ", "ZNF143", "NFYA", "MLX", "SREBF2"], ("MLX", "SREBF2"), "img/crop_MLX_SREBF2.png", ""))
# logos
txt = open("img/jaspar.meme").read(); blk = {b.split()[0]: b for b in txt.split("MOTIF ")[1:]}
COL = {"A": "#2C7A6B", "C": "#2166AC", "G": "#D98C1F", "T": "#B2182B"}; FP = FontProperties(family="DejaVu Sans", weight="bold")
def ppm(m):
    L = blk[m].strip().split("\n"); i = [k for k, l in enumerate(L) if l.startswith("letter-prob")][0]; return np.array([[float(x) for x in l.split()] for l in L[i + 1:] if l.strip() and not l.startswith("URL")])
def logo(m, name, out, rc=False):
    P_ = ppm(m); P_ = P_[::-1, ::-1] if rc else P_; ic = 2 + (P_ * np.log2(P_ + 1e-9)).sum(1); fig, ax = plt.subplots(figsize=(max(2.2, 0.27 * len(P_)), 1.05))
    for x, (p, h) in enumerate(zip(P_, ic)):
        y0 = 0.0
        for k in np.argsort(p):
            ht = p[k] * h
            if ht < 0.02: continue
            ch = "ACGT"[k]; tp = TextPath((0, 0), ch, size=1, prop=FP); b = tp.get_extents(); tr = Affine2D().translate(-b.x0, -b.y0).scale(0.9 / b.width, ht / b.height).translate(x - 0.45, y0); ax.add_patch(PathPatch(tp, transform=tr + ax.transData, fc=COL[ch], lw=0)); y0 += ht
    ax.set_xlim(-.6, len(P_) - .4); ax.set_ylim(0, 2); ax.set_yticks([0, 2]); ax.tick_params(labelsize=8, length=2, width=.4, pad=1); ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False); ax.set_ylabel("bits", fontsize=8, labelpad=-4); ax.set_title(name, fontsize=11, loc="left", pad=2); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
logo("MA0663.1", "MLX · E-box (MA0663.1)", "img/logo_mlx.png"); logo("MA0828.3", "SREBF2 · E-box (MA0828.3)", "img/logo_srebf2_ebox.png"); logo("MA0596.1", "SREBF2 · SRE (MA0596.1)", "img/logo_srebf2_sre.png")
