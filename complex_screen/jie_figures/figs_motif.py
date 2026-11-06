# Motif-slide panels and the MLX-SREBF2 bar map and PAE, each at its exact slide frame, 600 dpi, text >= 12 pt. Tables come from complex_screen/composite_motif and jie_motif; group labels reuse the jie_cpg.py definitions.
import json, sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D
FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
for f in ("Cambria.ttc", "Cambriab.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams.update({"font.family": "Cambria", "font.size": 12, "text.color": "#1A1A1A", "axes.edgecolor": "#1A1A1A", "axes.linewidth": 0.6, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.labelsize": 12, "savefig.dpi": 600})
M = sys.argv[1]; C_ = f"{M}/complex_screen"; BL, RD, G1, G2, G3 = "#2166AC", "#B2182B", "#4D4D4D", "#8C8C8C", "#BDBDBD"
def done(fig, name): fig.savefig(f"fig/{name}.png", facecolor="white"); plt.close(fig)
def spread(ps, gap):
    v = list(ps)
    for _ in range(200):
        for i in range(len(v) - 1):
            d = v[i + 1] - v[i]
            if d < gap: m = (gap - d) / 2; v[i] -= m; v[i + 1] += m
    return v
# --- slide 7, logos on a shared CAC column; the SRE is drawn on the strand that reads TCAC
txt = open("img/jaspar.meme").read(); blk = {b.split()[0]: b for b in txt.split("MOTIF ")[1:]}; COL = {"A": "#2C7A6B", "C": BL, "G": "#D98C1F", "T": RD}; FP = FontProperties(family="DejaVu Sans", weight="bold")
def pwm(m):
    L = blk[m].strip().split("\n"); i = [k for k, l in enumerate(L) if l.startswith("letter-prob")][0]; return np.array([[float(x) for x in l.split()] for l in L[i + 1:] if l.strip() and not l.startswith("URL")])
def cac(P):
    s = [P[i, 1] * P[i + 1, 0] * P[i + 2, 1] for i in range(len(P) - 2)]; return int(np.argmax(s))
Ms = [("MA0663.1", "MLX · E-box (MA0663.1)", False), ("MA0828.3", "SREBF2 · E-box (MA0828.3)", False), ("MA0596.1", "SREBF2 · SRE (MA0596.1)", True)]; PP = []
for m, t, rc in Ms:
    P = pwm(m); P = P[::-1, ::-1] if rc else P; PP.append((P, cac(P), t)); print(m, "CAC at", cac(P), "consensus", "".join("ACGT"[k] for k in P.argmax(1)))
lo = max(c for _, c, _ in PP); hi = max(len(P) - c for P, c, _ in PP); fig = plt.figure(figsize=(4.5, 3.6))
for r, (P, c, t) in enumerate(PP):
    ax = fig.add_axes([0.11, 0.70 - r * 0.325, 0.87, 0.22]); ax.add_patch(Rectangle((-.5, 0), 3, 2, fc="#E3E3E3", ec="none", zorder=0)); ic = 2 + (P * np.log2(P + 1e-9)).sum(1)
    for x, (p, h) in enumerate(zip(P, ic)):
        y0 = 0.0
        for k in np.argsort(p):
            ht = p[k] * h
            if ht < 0.02: continue
            ch = "ACGT"[k]; tp = TextPath((0, 0), ch, size=1, prop=FP); b = tp.get_extents(); tr = Affine2D().translate(-b.x0, -b.y0).scale(0.9 / b.width, ht / b.height).translate(x - c - 0.45, y0); ax.add_patch(PathPatch(tp, transform=tr + ax.transData, fc=COL[ch], lw=0, zorder=2)); y0 += ht
    ax.set_xlim(-lo - .6, hi - .4); ax.set_ylim(0, 2); ax.set_yticks([0, 2]); ax.tick_params(length=2, width=.4, pad=1); ax.set_xticks([]); ax.spines[["top", "right", "bottom"]].set_visible(False); ax.set_ylabel("bits", labelpad=-6); ax.text(-lo - .6, 2.12, t, fontsize=13, fontweight="bold", va="bottom")
done(fig, "motif_logos_aligned")
# --- slide 7, composite vs single-motif elements, same elements for the negative control
T = pd.read_csv(f"{C_}/composite_motif/pairs/pair_tests.tsv", sep="\t"); T = T[(T.design == "real_half_sites") & (T.contrast == "primary_vs_single") & (T.outcome == "cobind_product")]; a = T[T.pair == "MLX-SREBF2"].set_index("fimo_p"); b = T[T.pair.str.startswith("RBBP7")].set_index("fimo_p"); fig = plt.figure(figsize=(4.2, 3.6)); ax = fig.add_axes([0.40, 0.17, 0.56, 0.68]); R = [("MLX–SREBF2", a.loc[1e-5], BL, 3.4), ("RBBP7–SRCAP", b.loc[1e-5], G2, 2.7), ("MLX–SREBF2", a.loc[1e-4], BL, 1.2), ("RBBP7–SRCAP", b.loc[1e-4], G2, .5)]
for l, r, c, y in R:
    ax.plot([r.lo, r.hi], [y, y], color=c, lw=1.6); ax.plot(r.ratio, y, "o", color=c, ms=6); ax.text(-.03, y, l, transform=ax.get_yaxis_transform(), ha="right", va="center", color=c, fontweight="bold"); ax.text(r.hi + .04, y, f"{r.ratio:.2f}", va="center", fontsize=12)
    print(l, r.name, int(r.n_composite), r.ratio, r.lo, r.hi)
ax.axvline(1, color=G1, lw=.7, ls="--"); ax.set_xlim(.2, 2.35); ax.set_ylim(0, 4.6); ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False); ax.set_xlabel("Co-binding, composite / single motif"); ax.tick_params(length=2, width=.4)
ax.text(-.70, 4.05, f"p ≤ 1e-5, primary test: {int(a.loc[1e-5].n_composite)} elements", transform=ax.get_yaxis_transform(), fontweight="bold", fontsize=12); ax.text(-.70, 1.85, f"p ≤ 1e-4: {int(a.loc[1e-4].n_composite)} elements", transform=ax.get_yaxis_transform(), fontweight="bold", fontsize=12); fig.text(0.02, 0.93, "Composite-motif test", fontsize=13, fontweight="bold"); done(fig, "motif_composite_forest")
# --- slide 7, trend over motif-score deciles
Tr = pd.read_csv(f"{C_}/composite_motif/trend_exploratory/trend_results.tsv", sep="\t"); Tr = Tr[Tr.outcome == "cobind_product"]; fig = plt.figure(figsize=(4.2, 3.6)); ax = fig.add_axes([0.17, 0.17, 0.79, 0.68]); x = np.arange(1, 11); shuf = Tr[Tr.test == "MLX-SREBF2 shuffle"]
for k, (_, r) in enumerate(shuf.iterrows()): ax.plot(x, [float(v) for v in r.decile_means.split(",")], color=G3, lw=1, label="Shuffled motifs (3)" if k == 0 else None)
r = Tr[Tr.test == "MLX-SREBF2"].iloc[0]; ax.plot(x, [float(v) for v in r.decile_means.split(",")], color=BL, lw=2, marker="o", ms=4, label="MLX + SRE"); ax.axhline(1, color=G1, lw=.7, ls="--"); ax.set_ylim(.94, 1.06); ax.set_xticks([1, 5, 10]); ax.set_xlabel("Motif score decile"); ax.set_ylabel("Co-binding (obs/exp)"); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=.4); ax.legend(frameon=False, loc="upper left", fontsize=12, handlelength=1.4, borderaxespad=0); fig.text(0.02, 0.93, "Exploratory: no trend with motif score", fontsize=13, fontweight="bold"); print("trend", r.top_vs_bottom, r.lo, r.hi, r.jt_p); done(fig, "motif_trend")
# --- slide 4, EZH2-MTF2 groups as defined in jie_cpg.py
U = f"{C_}/composite_motif/union"; Cv = pd.read_csv(f"{U}/covariates.tsv", sep="\t"); Z = np.load(f"{C_}/jie_motif/cpg_stats.npz"); oe, gc, cgi = Z["oe"], Z["gc"], Z["cgi"]; tfs = [l.strip() for l in open(f"{M}/regulatome/m1_binding/v2/tfs.txt") if l.strip()]; Y = {t: np.load(f"{U}/counts/{t}.npz") for t in tfs}; tot = sum(Y[t]["y"] for t in tfs); NN = sum(int(Y[t]["total"]) for t in tfs); atac = Cv.atac.values / Cv.atac.sum() * 1e6
def bound(t):
    y = Y[t]["y"].astype(float); Rr = int(Y[t]["total"]); pc = (tot - y) / (NN - Rr) * 1e6; Xd = np.column_stack([np.ones(len(y)), np.log1p(atac), np.log1p(pc), np.log(Cv.width.values), Cv.gc.values, Cv.gc.values ** 2, Cv["map"].values]); off = np.full(len(y), np.log(Rr / 1e6)); be = np.zeros(Xd.shape[1]); be[0] = np.log(y.sum() / np.exp(off).sum())
    for _ in range(30):
        eta = off + Xd @ be; mu = np.exp(eta); z = eta - off + (y - mu) / mu; XtW = Xd.T * mu; bn = np.linalg.solve(XtW @ Xd + 1e-6 * np.eye(len(be)), XtW @ z)
        if np.max(np.abs(bn - be)) < 1e-7: be = bn; break
        be = bn
    return np.log2((y + .5) / (np.exp(off + Xd @ be) + .5)) > 0
Ae, Am = bound("EZH2"), bound("MTF2"); co, eo, mo, no = Ae & Am, Ae & ~Am, Am & ~Ae, ~Ae & ~Am; rng = np.random.default_rng(1); st = np.minimum((pd.Series(gc).rank(pct=True).values * 20).astype(int), 19) * 10 + np.minimum((pd.Series(Cv.atac.values).rank(pct=True).values * 10).astype(int), 9); ix = []
for k, m in zip(*np.unique(st[co], return_counts=True)):
    p = np.flatnonzero(no & (st == k))
    if len(p): ix.append(rng.choice(p, m, replace=len(p) < m))
ix = np.concatenate(ix); print("n", co.sum(), eo.sum(), mo.sum(), no.sum(), "cpg", round(oe[co].mean(), 4), round(oe[eo | mo].mean(), 4), round(oe[ix].mean(), 4), "cgi", round(cgi[co].mean(), 4), round(cgi[ix].mean(), 4), "published 7574 26884 29366 .4585 .3857 .464 .3686 .3746")
Gs = [("Both", oe[co], cgi[co], RD), ("EZH2\nonly", oe[eo], cgi[eo], BL), ("MTF2\nonly", oe[mo], cgi[mo], "#D6604D"), ("Matched", oe[ix], cgi[ix], G2)]; fig = plt.figure(figsize=(8.8, 3.6)); ax = fig.add_axes([0.065, 0.17, 0.40, 0.70]); vp = ax.violinplot([g[1] for g in Gs], showextrema=False, widths=.85, bw_method=.15)
for b_, g in zip(vp["bodies"], Gs): b_.set_facecolor(g[3]); b_.set_alpha(.8); b_.set_edgecolor("none")
for i, g in enumerate(Gs, 1): q = np.percentile(g[1], [25, 50, 75]); ax.plot([i, i], q[[0, 2]], color="#1A1A1A", lw=1.2); ax.plot(i, g[1].mean(), "o", color="white", mec="#1A1A1A", ms=4.5, mew=.7); ax.text(i, 1.33, f"{g[1].mean():.2f}", ha="center", fontsize=12)
ax.set_xticks(range(1, 5)); ax.set_xticklabels([g[0] for g in Gs]); ax.set_ylim(0, 1.45); ax.set_ylabel("CpG obs/exp"); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=.4); fig.text(0.065, 0.92, "CpG o/e per element (dot = mean, bar = IQR)", fontsize=13, fontweight="bold")
ax = fig.add_axes([0.545, 0.17, 0.215, 0.70]); r2 = np.random.default_rng(3)
for i, g in enumerate(Gs, 1):
    b_ = [r2.choice(g[2], len(g[2])).mean() for _ in range(1000)]; lo_, hi_ = np.percentile(b_, [2.5, 97.5]); ax.bar(i, g[2].mean(), color=g[3], width=.72); ax.plot([i, i], [lo_, hi_], color="#1A1A1A", lw=.9); ax.text(i, hi_ + .012, f"{g[2].mean():.2f}", ha="center", fontsize=12)
ax.set_xticks(range(1, 5)); ax.set_xticklabels(["Both", "E", "M", "Mat."]); ax.set_ylim(0, .52); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=.4); fig.text(0.51, 0.92, "CpG-island-like fraction", fontsize=13, fontweight="bold")
Ct = pd.read_csv(f"{C_}/jie_motif/cpg_control_pairs.tsv", sep="\t"); Ct = Ct[Ct.control_for.astype(str).str.contains("EZH2")]; Rs = pd.read_csv(f"{C_}/jie_motif/cpg_pair_results.tsv", sep="\t").iloc[0]; print("controls", len(Ct), Ct.cpg_oe_ratio.min(), Ct.cpg_oe_ratio.max()); ax = fig.add_axes([0.825, 0.17, 0.16, 0.70]); xj = np.random.default_rng(5).uniform(-.16, .16, len(Ct)); ax.scatter(xj, Ct.cpg_oe_ratio, s=11, color=G3, lw=0); ax.plot([-.28, .28], [Rs.cpg_oe_ratio] * 2, color=RD, lw=2); ax.plot([0, 0], [Rs.cpg_oe_lo, Rs.cpg_oe_hi], color=RD, lw=1); ax.axhline(1, color=G1, lw=.6, ls="--"); ax.set_xlim(-.5, .5); ax.set_xticks([]); ax.set_ylim(.95, 1.05); ax.spines[["top", "right", "bottom"]].set_visible(False); ax.tick_params(length=2, width=.4); ax.set_ylabel("Both / matched", labelpad=1); fig.text(0.79, 0.92, "vs 40 control pairs", fontsize=13, fontweight="bold"); ax.text(0, .955, f"EZH2–MTF2 {Rs.cpg_oe_ratio:.2f}", color=RD, fontsize=12, fontweight="bold", ha="center"); done(fig, "motif_cpg_panels")
# --- slide 5, bar map and PAE
DOM = json.load(open("jie_domains.json")); GROUP = {"SREBF2": {"basic": None, "HLH": "bHLH-ZIP", "ZIP": None}}; Ch = DOM["MLX_SREBF2"]["chains"]; St = json.load(open("img/MLX_SREBF2.sites.json")); fig = plt.figure(figsize=(7.1, 1.95)); ax = fig.add_axes([0.10, 0.02, 0.88, 0.96]); N = 2; mx = max(c["len"] for c in Ch.values()); ax.set_xlim(0, mx); ax.set_ylim(-.47, N - .35)
for r, (ch, c) in enumerate(Ch.items()):
    y = N - 1 - r; ax.add_patch(Rectangle((0, y - .1), c["len"], .2, fc=c["base"], ec=G2, lw=.3)); ax.text(-mx * .012, y, c["name"], ha="right", va="center", fontsize=12, fontweight="bold"); tx = []
    for nm, a_, b_, col, kind in c["dom"]:
        ax.add_patch(Rectangle((a_, y - .1), b_ - a_ + 1, .2, fc=col, ec="none")); l = GROUP.get(c["name"], {}).get(nm, nm)
        if l: tx.append(((a_ + b_) / 2 if l != "bHLH-ZIP" else 365, l))
    xs = spread([t[0] for t in sorted(tx)], mx * .095)
    for x_, (x0, l) in zip(xs, sorted(tx)): ax.annotate(l, xy=(x0, y - .1), xytext=(x_, y - .22), ha="center", va="top", fontsize=12, arrowprops=dict(arrowstyle="-", lw=.4, color=G2, shrinkA=0, shrinkB=0))
    for v in St[ch]: ax.plot([v, v], [y + .12, y + .27], color="#1A1A1A", lw=.8)
ax.axis("off"); done(fig, "sitemap_MLX_v2")
Zp = np.load("img/MLX_SREBF2.pae.npz"); Mp = np.array(Zp["pae"], float); chn = Zp["chains"]; fig = plt.figure(figsize=(2.3, 1.95)); ax = fig.add_axes([0.02, 0.03, 0.70, 0.80]); im = ax.imshow(Mp, cmap="Greys_r", vmin=0, vmax=30, interpolation="nearest"); b_ = np.flatnonzero(chn[1:] != chn[:-1]) + .5
for x_ in b_: ax.axhline(x_, color=RD, lw=.8); ax.axvline(x_, color=RD, lw=.8)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Predicted error (PAE)", fontsize=12, loc="left"); cax = fig.add_axes([0.76, 0.2, 0.05, 0.5]); cb = fig.colorbar(im, cax=cax); cb.set_ticks([0, 15, 30]); cb.ax.tick_params(labelsize=12, width=.3, length=2); cb.outline.set_linewidth(.3); cb.set_label("Å", fontsize=12); done(fig, "pae_MLX_v2")
