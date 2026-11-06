"""Slide 22, chr16-hub layout at the exact SEdb SE: raw balanced Hi-C triangle (recipe as plot_hub_browser.py, no O/E, cut at 300 kb at true 45 degrees), Peakachu 0.7 loops, Cicero edges among all hub elements (colour = what the edge connects, solid = on a Hi-C loop, dashed = Cicero only), raw TF coverage, ChromHMM, RefSeq Select gene models; neutral partner stripes. Right: edge counts and the direct Hi-C test. Each panel at its exact frame size."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R0 = Path(__file__).resolve().parents[1]; Hb = R0.parent / "se_hub"; O = R0 / "figure_extreme"; V7 = Hb / "figure_v7_refseq"; IN = sys.argv[1]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4, "axes.spines.top": False, "axes.spines.right": False})
COL = dict(ZNF507="#C48311", SOX4="#32A190", NFYA="#D9B637", GLYR1="#C681F1", NME2="#739B64", DR1="#8AC975", MTF2="#65278F", E4F1="#712F55", MBD3="#6E6E6E", MLX="#6E6E6E"); ORDER = ["SOX4", "NFYA", "MBD3", "E4F1", "ZNF507", "MLX", "NME2", "DR1", "GLYR1", "MTF2"]
RED, BLUE, INK = "#C0392B", "#4A6FA5", "#222222"; BAND = dict(se="#8C8C8C", promoter="#A9DDD4", enhancer="#F4E08A", other="#E2E2E2"); TC = dict(se="#262626", promoter="#0F7C72", enhancer="#8F7400", other="#666666"); L0, L1 = 430000, 2250000; SE0, SE1 = 1248039, 1254347; XL = (L0 / 1e6, L1 / 1e6)
E = pd.read_csv(R0 / "figure/edges_in_structure.tsv", sep="\t"); HE = pd.read_csv(R0 / "hub_edges_all.tsv", sep="\t"); TY = dict(promoter="#0072B2", enhancer="#D55E00", other="#BBBBBB"); Tc = pd.read_csv(Hb / "figure_v5_tftracks/tf_cov_2kb.tsv", sep="\t", header=None, names=["tf", "b", "n"]); nb = (L1 - L0) // 2000; ed = (L0 + np.arange(nb + 1) * 2000) / 1e6; mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2; E["apos"] = [mid(x) for x in E.region1]
W, H = 9.75, 5.75; fig = plt.figure(figsize=(W, H)); AW0 = 8.72; Lx, AW = 0.93 / W, AW0 / W; fig.text(0.008, 1 - 0.16 / H, "TF binding and Hi-C at the super-enhancer (SEdb, HEK293T), chr19:1,248,039–1,254,347", fontsize=14, fontweight="bold", va="center")
def stripes(a):
    for p in E.pos: a.axvspan((p - 3000) / 1e6, (p + 3000) / 1e6, color="#E6E6E6", lw=0, zorder=0)
    a.axvspan((SE0 - 2500) / 1e6, (SE1 + 2500) / 1e6, color=BAND["se"], lw=0, zorder=0.5)
ymax = 240.0; hh = AW0 * ymax / (L1 - L0) * 1000; b_h = H - 0.33 - hh; ax = fig.add_axes([Lx, b_h / H, AW, hh / H]); first = ax
M = np.load(R0 / "model/inputs/hic_10kb.npz")["ice"]; n = M.shape[0]; Z = np.log10(np.nan_to_num(M) + 1e-5); xs = L0 + np.arange(n + 1) * 10000; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z); ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 50), vmax=np.nanpercentile(Z, 99.5), shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False); ax.set_xlim(*XL); ax.set_ylim(0, ymax); ax.set_axis_off(); fig.text(0.008, (b_h + hh * 0.5) / H, "Hi-C\n10 kb", va="center", ha="left")
loops = set()
for r_ in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == "chr19" and x[3] == "chr19":
            a, b = sorted(((int(x[1]) + int(x[2])) // 2, (int(x[4]) + int(x[5])) // 2))
            if a >= L0 and b <= L1: loops.add((a // 10000, b // 10000))
y2 = b_h - 0.24; ax1 = fig.add_axes([Lx, y2 / H, AW, 0.22 / H], sharex=first); stripes(ax1); ax1.set_ylim(-1.05, 0); ax1.set_axis_off()
for a, b in loops:
    q = np.linspace(a, b, 40) * 0.01; ax1.plot(q, -min(1, (b - a) / 90) * np.sin(np.pi * (q - q[0]) / (q[-1] - q[0])), color="#7F7F7F", lw=0.35, alpha=0.6)
fig.text(0.008, (y2 + 0.11) / H, "Loops", va="center", ha="left")
y3 = y2 - 0.72; ax2 = fig.add_axes([Lx, y3 / H, AW, 0.70 / H], sharex=first); stripes(ax2); ax2.set_ylim(0, 1.08); ax2.set_axis_off()
for r in HE.assign(o_=HE.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["on_loop", "o_"]).itertuples():
    a_, b_ = sorted((mid(r.reg_a) / 1e6, mid(r.reg_b) / 1e6)); q = np.linspace(a_, b_, 80); h_ = min(1.0, 0.12 + 0.88 * (b_ - a_) / 1.1); ax2.plot(q, h_ * np.sin(np.pi * (q - a_) / (b_ - a_)), color=TY[r.type], lw=(2.0 if r.type == "promoter" else 1.6 if r.from_se else 1.1) if r.on_loop else 0.55, ls="-" if r.on_loop else (0, (2.5, 2)), alpha=1 if r.on_loop else 0.45)
fig.text(0.008, (y3 + 0.35) / H, "Cicero\nedges", va="center", ha="left", linespacing=0.95)
rh = 0.172; hub = sorted(set(np.r_[(E.pos.values - L0) // 2000, (np.arange(SE0, SE1, 2000) - L0) // 2000].tolist())); y_t = y3 - 0.03
for k, t in enumerate(ORDER):
    y_ = y_t - (k + 1) * rh; a_ = fig.add_axes([Lx, y_ / H, AW, (rh - 0.012) / H], sharex=first); d = Tc[Tc.tf == t]; tot = int(d[d.b == "total"].n.iloc[0]); h = np.zeros(nb)
    for r in d[d.b != "total"].itertuples(): h[int(r.b)] = r.n
    h = h / tot * 1e6; lim = max(h[hub].max(), np.percentile(h, 99.5)); stripes(a_); a_.fill_between(ed[:-1], 0, np.minimum(h, lim), step="post", color=COL[t], lw=0, zorder=2); a_.set_ylim(0, lim); a_.set_yticks([]); a_.spines["left"].set_visible(False); a_.spines["bottom"].set_linewidth(0.4); a_.tick_params(bottom=False, labelbottom=False); a_.text(-0.006, 0.4, t, transform=a_.transAxes, ha="right", va="center", fontstyle="italic", color=COL[t] if t not in ("MBD3", "MLX") else "#444444")
GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer", "TssA": "TSS/flank", "TssFlnk": "TSS/flank", "TssFlnkU": "TSS/flank", "TssFlnkD": "TSS/flank", "TssBiv": "TSS/flank", "Tx": "transcribed", "TxWk": "transcribed", "ReprPC": "repressed", "ReprPCWk": "repressed", "Het": "repressed", "ZNF/Rpts": "repressed"}; GCOL = {"enhancer": "#E67E22", "TSS/flank": "#C0392B", "transcribed": "#5D8AA8", "repressed": "#7D8B99"}
y_c = y_t - len(ORDER) * rh - 0.17; axc = fig.add_axes([Lx, y_c / H, AW, 0.15 / H], sharex=first); stripes(axc); axc.set_ylim(0, 1); axc.set_axis_off(); Cm = pd.read_csv(V7 / "chromhmm_window.tsv", sep="\t", header=None, names=["c", "s", "e", "st"])
for r in Cm.itertuples():
    g_ = GROUP.get(r.st)
    if g_: axc.add_patch(plt.Rectangle((r.s / 1e6, 0.08), (r.e - r.s) / 1e6, 0.84, color=GCOL[g_], lw=0, zorder=2))
axc.text(-0.006, 0.5, "ChromHMM", transform=axc.transAxes, ha="right", va="center", fontsize=10)
y_g = 0.61; h_g = y_c - y_g - 0.01; ax4 = fig.add_axes([Lx, y_g / H, AW, h_g / H], sharex=first); stripes(ax4); NR = 3; ax4.set_ylim(NR, 0); ax4.set_xlim(*XL); ax4.set_yticks([]); ax4.spines["left"].set_visible(False); ax4.set_xticks(np.arange(0.6, 2.3, 0.2)); ax4.tick_params(labelsize=11); ax4.set_xlabel("chr19 position (Mb, hg38)", fontsize=11, labelpad=0); ax4.text(-0.006, 0.5, "RefSeq\ngenes", transform=ax4.transAxes, ha="right", va="center", linespacing=0.95, fontsize=10)
J = pd.DataFrame(json.load(open(V7 / "refseq_select_chr19_0.43-2.25Mb.json"))["ncbiRefSeqSelect"]); J = J[(J.txEnd > L0) & (J.txStart < L1)].copy(); rnd = fig.canvas.get_renderer(); scale = (L1 - L0) / 1e6 / AW0; prom = {x for l in E.label for x in l.split(" ")[0].split("/")}; near = set()
def tw(t, **kw):
    x = fig.text(0, 0, t, **kw); w_ = x.get_window_extent(rnd).width / fig.dpi * scale; x.remove(); return w_
J["prio"] = [0 if g in prom else 1 if g in near else 2 for g in J.name2]; J["ln"] = J.txEnd - J.txStart; J = J.sort_values(["prio", "ln"], ascending=[True, False]); rows = {k: [] for k in range(NR)}; used = {k: [] for k in range(NR)}; sc = (SE0 + SE1) / 2e6; sew = tw("super-enhancer", fontweight="bold", fontsize=9); rows[0].append((sc - sew / 2 - 0.01, sc + sew / 2 + 0.01)); ax4.text(sc, 0.62, "super-enhancer", ha="center", va="center", color=TC["se"], fontweight="bold", fontsize=9, zorder=5, bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.7))
def free(k, a_, b_): return all(b_ + 0.006 < u or a_ - 0.006 > v for u, v in rows[k])
def model(r, k, col):
    y_ = k + 0.17; a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6; ax4.plot([a_, b_], [y_, y_], color=col, lw=0.7, zorder=3, solid_capstyle="butt")
    for es, ee in zip(map(int, r.exonStarts.strip(",").split(",")), map(int, r.exonEnds.strip(",").split(","))):
        if ee > L0 and es < L1: ax4.add_patch(plt.Rectangle((es / 1e6, y_ - 0.13), max((ee - es) / 1e6, 0.001), 0.26, color=col, lw=0, zorder=4))
    tss = (r.txStart if r.strand == "+" else r.txEnd) / 1e6; d_ = 0.009 if r.strand == "+" else -0.009
    if XL[0] <= tss <= XL[1]: ax4.plot([tss, tss], [y_, y_ - 0.2], color=col, lw=0.7, zorder=4); ax4.annotate("", xy=(tss + d_, y_ - 0.2), xytext=(tss, y_ - 0.2), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
named = 0
for r in J.itertuples():
    col = INK if r.prio == 0 else "#666666"; a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6; kw = dict(fontstyle="italic", fontweight="bold", fontsize=9) if r.prio == 0 else dict(fontstyle="italic", fontsize=8); w_ = tw(r.name2, **kw); c_ = (a_ + b_) / 2; ta, tb = c_ - w_ / 2, c_ + w_ / 2; ok = next((k for k in range(NR) if free(k, min(a_, ta), max(b_, tb))), None)
    if ok is not None and tb < XL[1] and ta > XL[0]: rows[ok].append((min(a_, ta), max(b_, tb))); model(r, ok, col); ax4.text(c_, ok + 0.62, r.name2, ha="center", va="center", color=col, zorder=5, **kw); named += 1
    else: k = min(range(NR), key=lambda q: sum(1 for u, v in used[q] if a_ < v and u < b_)); used[k].append((a_, b_)); model(r, k, col if r.prio < 2 else "#8A8A8A")
print("named", named, "of", len(J), "y_c", round(y_c, 2), "h_g", round(h_g, 2)); kx = 0.008
for c_, l_, wd_, ty in [(GCOL["enhancer"], "enhancer", 0.075, "r"), (GCOL["TSS/flank"], "TSS/flank", 0.08, "r"), (GCOL["transcribed"], "transcribed", 0.088, "r"), (GCOL["repressed"], "repressed", 0.098, "r"), (BAND["se"], "super-enhancer", 0.125, "b"), ("#E6E6E6", "hub partner", 0.1, "b"), (TY["promoter"], "promoter\u2013promoter", 0.135, "l"), (TY["enhancer"], "to an enhancer", 0.115, "l"), (TY["other"], "other", 0.06, "l")]:
    if ty == "l": fig.add_artist(plt.Line2D([kx, kx + 0.016], [0.1 / H, 0.1 / H], transform=fig.transFigure, color=c_, lw=1.6))
    else: fig.add_artist(plt.Rectangle((kx, (0.1 - 0.045) / H), 0.014, 0.09 / H, transform=fig.transFigure, color=c_, lw=0))
    fig.text(kx + 0.019, 0.1 / H, l_, va="center", fontsize=9); kx += wd_
for e in ("svg", "pdf"): fig.savefig(O / f"browser_extreme.{e}", dpi=600, transparent=True)
fig.savefig(O / "browser_extreme.png", dpi=600, facecolor="white"); fig.savefig(O / "browser_extreme_check.png", dpi=144, facecolor="white"); plt.close(fig)
# right panel
Rj = json.load(open(R0 / "hic_test/results_10kb.json")); l_ = Rj["edges_loop_edges_18"]; nn_ = Rj["hub_nonedge_tf"]; W, H = 3.1, 5.75; fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_axis_off(); ax.text(0.05, H - 0.16, "Edges in the hub", fontsize=13, fontweight="bold", va="center"); y = H - 0.56; X_ = [1.72, 2.25, 2.78]
for x_, t_ in zip(X_, ("loop", "Cicero", "all")): ax.text(x_, y, t_, ha="center", va="center")
ax.plot([0.05, 0.05], [y, y], lw=0)
for lab, t_ in (("promoter\u2013promoter", "promoter"), ("to an enhancer", "enhancer"), ("other", "other")):
    y -= 0.3; m_ = HE.type == t_; ax.text(0.05, y, lab, va="center", color=TY[t_] if t_ != "other" else "#777777", fontweight="bold", fontsize=11); a1, b1_ = int((m_ & (HE.on_loop == 1)).sum()), int((m_ & (HE.on_loop == 0)).sum())
    for x_, v_ in zip(X_, (a1, b1_, a1 + b1_)): ax.text(x_, y, str(v_), ha="center", va="center")
y -= 0.3; ax.text(0.05, y, "all", va="center"); a1, b1_ = int((HE.on_loop == 1).sum()), int((HE.on_loop == 0).sum())
for x_, v_ in zip(X_, (a1, b1_, a1 + b1_)): ax.text(x_, y, str(v_), ha="center", va="center")
y -= 0.3; ax.text(0.05, y, "loop = solid, on a Hi-C loop\nCicero = dashed, Cicero only", va="top", fontsize=10.5, color="#444444", linespacing=1.1); y -= 0.72; ax.text(0.05, y, "Direct Hi-C test", fontsize=13, fontweight="bold", va="center"); y -= 0.42; ax.text(0.05, y, f"SE loop-edges (18) vs matched:\n{l_['ratio_tf']:.2f}\u00d7 (p = {l_['p_tf']:.3f})", va="center", linespacing=1.1); y -= 0.56; ax.text(0.05, y, f"Other pairs in the hub, as a set:\n{nn_['ratio']:.2f}\u00d7 (p = {nn_['p']:.2f}, n.s.)", va="center", linespacing=1.1); y -= 0.42; ax.text(0.05, y, "Matched for distance and TF\nclass, 1,000 sets. Edges were\nselected on Hi-C loops, so\nenrichment is partly expected.", va="top", fontsize=10.5, color="#444444", linespacing=1.1)
for e in ("svg", "pdf"): fig.savefig(O / f"browser_side_extreme.{e}", dpi=600, transparent=True)
fig.savefig(O / "browser_side_extreme.png", dpi=600, facecolor="white"); fig.savefig(O / "browser_side_extreme_check.png", dpi=144, facecolor="white")
