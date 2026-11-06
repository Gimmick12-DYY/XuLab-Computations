# Figures for the rebuilt Jie slides: annotated ChromHMM-18 matrix (Dr. Wang's TF order and scale), percentile dots, CpG bars, motif logos and half-site schematic.
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 11, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A", "axes.linewidth": 0.6, "savefig.dpi": 300})
CC = "../../scripts/complex_catalog"; J = json.load(open(f"{CC}/sp_mats.json")); T = J["tfs"]; W = np.array(J["mats"]["wang18"]["pearson"])
HIS = open(f"{CC}/wang_his_order.py").read().split('"pearson": ("')[1].split('"')[0].lower().split(); o = [T.index(t) for t in HIS]; R = W[np.ix_(o, o)]; n = len(HIS); off = W[~np.eye(len(T), dtype=bool)]; pctW = lambda a, b: (off <= W[T.index(a), T.index(b)]).mean()
def matrix(pair, block, out):
    fig = plt.figure(figsize=(7.1, 6.6)); ax = fig.add_axes([0.09, 0.03, 0.8, 0.86]); im = ax.imshow(R, cmap="RdYlBu_r", vmin=0.65, vmax=1, interpolation="nearest"); ax.set_xticks(range(n)); ax.set_yticks(range(n)); U = [t.upper() for t in HIS]; ax.set_xticklabels(U, rotation=90, fontsize=4.4); ax.set_yticklabels(U, fontsize=4.4); ax.xaxis.tick_top(); ax.tick_params(length=1.2, width=0.3, pad=1)
    idx = [HIS.index(t) for t in block]; pi = [HIS.index(t) for t in pair]
    for ll in (ax.get_xticklabels(), ax.get_yticklabels()):
        for k, lb in enumerate(ll):
            if k in idx: lb.set_fontweight("bold"); lb.set_fontsize(5.6); lb.set_color("#B2182B" if k in pi else "#1A1A1A")
    for a in idx:
        for b in idx:
            if a != b and not (a in pi and b in pi): ax.add_patch(Rectangle((b - .5, a - .5), 1, 1, fill=False, ec="#1A1A1A", lw=0.7))
    a, b = pi
    for x, y in ((a, b), (b, a)): ax.add_patch(Rectangle((x - .9, y - .9), 1.8, 1.8, fill=False, ec="#000000", lw=1.6, zorder=6))
    ax.annotate(f"{pair[0].upper()}–{pair[1].upper()}\nr = {W[T.index(pair[0]), T.index(pair[1])]:.3f}, percentile {np.floor(pctW(*pair) * 100) / 100:.2f}", xy=(max(a, b) + .9, min(a, b) - .2), xytext=(min(n - 26, max(a, b) + 9), min(a, b) + 16), fontsize=10, fontweight="bold", color="#000000", arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.3), bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#1A1A1A", lw=0.6), zorder=8)
    cax = fig.add_axes([0.905, 0.3, 0.018, 0.34]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=7, width=0.3, length=2); cb.outline.set_linewidth(0.3); cb.set_label("Pearson r", fontsize=8); fig.savefig(out, facecolor="white", bbox_inches="tight", pad_inches=0.03); plt.close(fig); return round(W[T.index(pair[0]), T.index(pair[1])], 3), round(pctW(*pair), 3)
print(matrix(("ezh2", "mtf2"), ["ezh2", "mtf2", "pcgf6", "rbbp4", "rbbp7"], "img/matrix_EZH2_MTF2.png")); print(matrix(("mlx", "srebf2"), ["mlx", "srebf2"], "img/matrix_MLX_SREBF2.png"))
P = pd.read_csv("data/r1v2_pairs_provisional.tsv", sep="\t"); lab = {"pearson_domain_pct": "Corrected Pearson", "spqn_noquies_pct": "ChromHMM SpQN", "compartment_spqn_pct": "Compartment SpQN", "cicero_excess_z_pct": "Cicero shared edges"}
def dots(a, b, out):
    r = P[(P.a == a) & (P.b == b)].iloc[0]; v = [float(r[c]) for c in lab]; y = np.arange(4)[::-1]; fig, ax = plt.subplots(figsize=(3.9, 1.8)); ax.axvline(.9, color="#8C8C8C", ls="--", lw=.9); ax.hlines(y, 0, v, color="#BDBDBD", lw=1); ax.scatter(v, y, s=44, color=["#B2182B" if x >= .9 else "#1A1A1A" for x in v], zorder=3)
    for yy, x in zip(y, v): ax.text(min(x + .03, 1.03), yy, f"{np.floor(x * 100 + 1e-9) / 100:.2f}", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels(list(lab.values()), fontsize=10); ax.set_xlim(0, 1.17); ax.set_xticks([0, .5, .9]); ax.set_xticklabels(["0", "0.5", "0.90"], fontsize=9); ax.set_xlabel("Library-stratified percentile", fontsize=10); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=.4); fig.savefig(out, facecolor="white", bbox_inches="tight"); plt.close(fig); return r.view1, r.view1_n_hi
print(dots("EZH2", "MTF2", "img/dots2_EZH2_MTF2.png"), dots("MLX", "SREBF2", "img/dots2_MLX_SREBF2.png"))
# CpG bars
C = pd.read_csv("data/cpg_pair_results.tsv", sep="\t").iloc[0]; Ct = pd.read_csv("data/cpg_control_pairs.tsv", sep="\t"); Ct = Ct[Ct.control_for == "EZH2-MTF2"]
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.75)); M_ = [("cpg_oe", "CpG obs/exp"), ("gcg_per_kb", "GCG + CGC per kb"), ("cgi_frac", "CpG-island-like fraction")]; Xl = ["Co-bound\nEZH2 + MTF2", "One only", "GC + ATAC\nmatched"]; Cols = ["#B2182B", "#BDBDBD", "#8C8C8C"]
for a, (k, t) in zip(ax, M_):
    v = [C[f"{k}_co"], C[f"{k}_single"], C[f"{k}_gcmatched"]]; a.bar(range(3), v, color=Cols, width=.7); a.set_xticks(range(3)); a.set_xticklabels(Xl, fontsize=8.5); a.set_title(t, fontsize=11, loc="left"); a.spines[["top", "right"]].set_visible(False); a.tick_params(length=2, width=.4, labelsize=9)
    for x, y in enumerate(v): a.text(x, y, f"{y:.2f}" if y < 5 else f"{y:.1f}", ha="center", va="bottom", fontsize=9)
    a.margins(y=.15)
fig.tight_layout(w_pad=1.2); fig.savefig("img/cpg_bars.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
print({k: (C[f"{k}_ratio"], C[f"{k}_lo"], C[f"{k}_hi"], round(Ct[f"{k}_ratio"].quantile(.05), 3), round(Ct[f"{k}_ratio"].quantile(.95), 3)) for k, _ in M_}, int(C.n_co), int(C.n_single))
# half-site schematic
fig, ax = plt.subplots(figsize=(6.4, 1.25)); ax.plot([0, 10], [0, 0], color="#1A1A1A", lw=1.2); ax.add_patch(Rectangle((1.6, -.28), 2.1, .56, fc="#2166AC", ec="none")); ax.add_patch(Rectangle((6.3, -.28), 2.1, .56, fc="#B2182B", ec="none")); ax.text(2.65, 0, "CACGTG", ha="center", va="center", color="white", fontsize=11, fontweight="bold"); ax.text(7.35, 0, "TCACNCCAC", ha="center", va="center", color="white", fontsize=10, fontweight="bold"); ax.text(2.65, .42, "MLX E-box", ha="center", fontsize=10.5); ax.text(7.35, .42, "SREBF2 SRE", ha="center", fontsize=10.5); ax.annotate("", xy=(3.75, -.5), xytext=(6.25, -.5), arrowprops=dict(arrowstyle="<->", lw=.8, color="#8C8C8C")); ax.text(5, -.92, "≤ 50 bp, non-overlapping: 15 elements at FIMO p ≤ 1e-5; no preferred spacing", ha="center", fontsize=9.5, color="#4D4D4D"); ax.set_xlim(0, 10); ax.set_ylim(-1.1, .8); ax.axis("off"); fig.savefig("img/halfsite.png", facecolor="white", bbox_inches="tight"); plt.close(fig)
