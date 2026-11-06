"""Slide 16 panel a in the layout of the chr16 hub browser figure: raw balanced Hi-C triangle (same log10 + 1e-5 and window p50-p99.5 colour recipe as plot_hub_browser.py, no O/E, no smoothing), Peakachu 0.7 loops, Cicero edges, raw TF coverage (CPM per 2 kb, per-track scale), genes, coloured stripes. Partner names sit under the genes as horizontal text with thin leaders. Stripes and labels: orange = SE candidate, teal = promoter, grey = other element. No circles on the triangle. Drawn at its exact frame size. The triangle stops at 340 kb, just above the farthest partner pixel, at true 45-degree scale."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
Bd = Path("/Users/dyy/Desktop/HEK293T-ATGTAG/results_3d_model/se_hub"); O = Bd / "figure_v7_refseq"; O6 = Bd / "figure_v6_hubstyle"; V = Bd / "figure_v2_cand"; T5 = Bd / "figure_v5_tftracks"; IN = sys.argv[1]
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 12, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4, "axes.spines.top": False, "axes.spines.right": False})
COL = dict(ZNF507="#C48311", SOX4="#32A190", NFYA="#D9B637", GLYR1="#C681F1", NME2="#739B64", DR1="#8AC975", MTF2="#65278F", E4F1="#712F55", MBD3="#6E6E6E", MLX="#6E6E6E"); ORDER = ["SOX4", "NFYA", "MBD3", "E4F1", "ZNF507", "MLX", "NME2", "DR1", "GLYR1", "MTF2"]
RED, DRED, LRED, INK = "#C0392B", "#8E2A20", "#E0928A", "#222222"; C_SE, C_P, C_E = "#F9C98D", "#A9DDD4", "#E2E2E2"; T_SE, T_P, T_E = "#C4650A", "#0F7C72", "#666666"; L0, L1 = 430000, 2250000; SE0, SE1 = 1203658, 1267059; cpos = 1254319; XL = (L0 / 1e6, L1 / 1e6)
S = pd.read_csv(V / "spokes.tsv", sep="\t").sort_values("pos").reset_index(drop=True); D = pd.read_csv(O6 / "partner_dots.tsv", sep="\t").set_index("label"); S["real"] = [bool(D.real_dot[l]) for l in S.label]; E = pd.read_csv(Bd / "model/inputs/hub_edges.tsv", sep="\t"); E["pos"] = [sum(map(int, r.split(":")[1].split("-"))) // 2 for r in E.region2]; S = S.merge(E[["pos", "n_tfs"]], on="pos", how="left"); RS = pd.read_csv(Bd / "figure_v7_refseq/partners_refseq.tsv", sep="\t").set_index("pos"); S["old_label"] = S.label; S["label"] = [RS.label[p].replace("RNU6-2/TMEM259", "TMEM259") for p in S.pos]; S["cls"] = [RS.cls[p] for p in S.pos]; G = pd.read_csv(Bd / "figure_v7_refseq/refseq_genes.tsv", sep="\t").rename(columns={"name2": "gene", "txStart": "start", "txEnd": "end"}); Tc = pd.read_csv(T5 / "tf_cov_2kb.tsv", sep="\t", header=None, names=["tf", "b", "n"]); nb = (L1 - L0) // 2000; ed = (L0 + np.arange(nb + 1) * 2000) / 1e6
W, H = 7.35, 5.75; fig = plt.figure(figsize=(W, H)); Lx, AW = 0.92 / W, 6.33 / W; fig.text(0.015, 1 - 0.16 / H, "TF binding and Hi-C at the super-enhancer candidate", fontsize=14, fontweight="bold", va="center")
def stripes(a):
    a.axvspan(SE0 / 1e6, SE1 / 1e6, color=C_SE, lw=0, zorder=0)
    for p, k_ in zip(S.pos, S.cls): a.axvspan((p - 4000) / 1e6, (p + 4000) / 1e6, color=C_P if k_ == "promoter" else C_E, lw=0, zorder=0)
ymax = 340.0; hh = 6.33 * ymax / (L1 - L0) * 1000; b_h = H - 0.33 - hh; ax = fig.add_axes([Lx, b_h / H, AW, hh / H]); first = ax
M = np.load(Bd / "figure_v2/hic_10kb.npz")["ice"]; n = M.shape[0]; Z = np.log10(np.nan_to_num(M) + 1e-5); xs = L0 + np.arange(n + 1) * 10000; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z)
ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 50), vmax=np.nanpercentile(Z, 99.5), shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False)
ax.set_xlim(*XL); ax.set_ylim(0, ymax); ax.set_axis_off()
fig.text(0.008, (b_h + hh * 0.5) / H, "Hi-C\n10 kb", va="center", ha="left")
# loops
loops = set()
for r_ in (5000, 10000):
    for ln in open(f"{IN}/loops/peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == "chr19" and x[3] == "chr19":
            a, b = sorted(((int(x[1]) + int(x[2])) // 2, (int(x[4]) + int(x[5])) // 2))
            if a >= L0 and b <= L1: loops.add((a // 10000, b // 10000))
y2 = b_h - 0.25; ax1 = fig.add_axes([Lx, y2 / H, AW, 0.24 / H], sharex=first); stripes(ax1); ax1.set_ylim(-1.05, 0); ax1.set_yticks([]); ax1.tick_params(bottom=False, labelbottom=False); [s_.set_visible(False) for s_ in ax1.spines.values()]
for a, b in loops:
    q = np.linspace(a, b, 40) * 0.01; ax1.plot(q, -min(1, (b - a) / 90) * np.sin(np.pi * (q - q[0]) / (q[-1] - q[0])), color="#7F7F7F", lw=0.35, alpha=0.6)
fig.text(0.008, (y2 + 0.14) / H, "Loops", va="center", ha="left")
# Cicero
y3 = y2 - 0.38; ax2 = fig.add_axes([Lx, y3 / H, AW, 0.36 / H], sharex=first); stripes(ax2); ax2.set_ylim(0, 1.08); ax2.set_yticks([]); ax2.tick_params(bottom=False, labelbottom=False); [s_.set_visible(False) for s_ in ax2.spines.values()]
for r in S.sort_values("n_tfs").itertuples():
    a_, b_ = sorted((cpos / 1e6, r.pos / 1e6)); q = np.linspace(a_, b_, 80); h_ = min(1.0, 0.22 + 0.78 * (b_ - a_) / 0.65); ax2.plot(q, h_ * np.sin(np.pi * (q - a_) / (b_ - a_)), color=DRED if r.n_tfs >= 2 else LRED, lw=1.8 if r.n_tfs >= 2 else 1.1)
ax2.text(XL[0], 0.52, "Cicero edges\non a Hi-C loop", color=RED, va="center", linespacing=1); ax2.plot([1.95, 2.0], [0.82, 0.82], color=DRED, lw=1.8); ax2.text(2.015, 0.82, "≥ 2 TFs", va="center"); ax2.plot([1.95, 2.0], [0.28, 0.28], color=LRED, lw=1.1); ax2.text(2.015, 0.28, "1 TF", va="center")
# tracks
rh = 0.172; hub = sorted(set(np.r_[(S.pos.values - L0) // 2000, (cpos - L0) // 2000].tolist())); y_t = y3 - 0.03
for k, t in enumerate(ORDER):
    y_ = y_t - (k + 1) * rh; a_ = fig.add_axes([Lx, y_ / H, AW, (rh - 0.012) / H], sharex=first); d = Tc[Tc.tf == t]; tot = int(d[d.b == "total"].n.iloc[0]); h = np.zeros(nb)
    for r in d[d.b != "total"].itertuples(): h[int(r.b)] = r.n
    h = h / tot * 1e6; lim = max(h[hub].max(), np.percentile(h, 99.5)); stripes(a_); a_.fill_between(ed[:-1], 0, np.minimum(h, lim), step="post", color=COL[t], lw=0, zorder=2); a_.set_ylim(0, lim); a_.set_yticks([]); a_.spines["left"].set_visible(False); a_.spines["bottom"].set_linewidth(0.4); a_.tick_params(bottom=False, labelbottom=False); a_.text(-0.008, 0.4, t, transform=a_.transAxes, ha="right", va="center", fontstyle="italic", color=COL[t] if t not in ("MBD3", "MLX") else "#444444")
# ChromHMM
GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer", "TssA": "TSS/flank", "TssFlnk": "TSS/flank", "TssFlnkU": "TSS/flank", "TssFlnkD": "TSS/flank", "TssBiv": "TSS/flank", "Tx": "transcribed", "TxWk": "transcribed", "ReprPC": "repressed", "ReprPCWk": "repressed", "Het": "repressed", "ZNF/Rpts": "repressed"}
GCOL = {"enhancer": "#E67E22", "TSS/flank": "#C0392B", "transcribed": "#5D8AA8", "repressed": "#7D8B99"}
y_c = y_t - len(ORDER) * rh - 0.19; axc = fig.add_axes([Lx, y_c / H, AW, 0.16 / H], sharex=first); stripes(axc); axc.set_ylim(0, 1); axc.set_axis_off(); Cm = pd.read_csv(O / "chromhmm_window.tsv", sep="\t", header=None, names=["c", "s", "e", "st"])
for r in Cm.itertuples():
    g_ = GROUP.get(r.st)
    if g_: axc.add_patch(plt.Rectangle((r.s / 1e6, 0.08), (r.e - r.s) / 1e6, 0.84, color=GCOL[g_], lw=0, zorder=2))
axc.text(-0.008, 0.5, "ChromHMM", transform=axc.transAxes, ha="right", va="center", fontsize=10)
# RefSeq gene models: exon boxes on an intron line, TSS flag, names under the models in 3 rows
y_g = 0.66; h_g = y_c - y_g - 0.01; ax4 = fig.add_axes([Lx, y_g / H, AW, h_g / H], sharex=first); stripes(ax4); ax4.set_ylim(4.0, 0); ax4.set_xlim(*XL); ax4.set_yticks([]); ax4.spines["left"].set_visible(False); ax4.set_xticks([0.6, 1.0, 1.4, 1.8, 2.2]); ax4.set_xlabel("chr19 position (Mb, hg38)", fontsize=12, labelpad=1); ax4.text(-0.008, 0.5, "RefSeq\ngenes", transform=ax4.transAxes, ha="right", va="center", linespacing=0.95, fontsize=10)
J = pd.DataFrame(__import__("json").load(open(O / "refseq_select_chr19_0.43-2.25Mb.json"))["ncbiRefSeqSelect"]); J = J[(J.txEnd > L0) & (J.txStart < L1)].copy(); rnd = fig.canvas.get_renderer(); scale = (L1 - L0) / 1e6 / 6.33
prom = {x for l, k_ in zip(S.label, S.cls) if k_ == "promoter" for x in l.split("/")}; near = {l.split(" ")[0] for l, k_ in zip(S.label, S.cls) if k_ != "promoter"}
def tw(t, **kw):
    x = fig.text(0, 0, t, **kw); w_ = x.get_window_extent(rnd).width / fig.dpi * scale; x.remove(); return w_
J["prio"] = [0 if g in prom else 1 if g in near else 2 for g in J.name2]; J["ln"] = J.txEnd - J.txStart; J = J.sort_values(["prio", "ln"], ascending=[True, False]); rows = {k: [] for k in range(4)}; used = {k: [] for k in range(4)}; sew = tw("SE candidate", fontweight="bold", fontsize=9); sc = (SE0 + SE1) / 2e6; rows[0].append((sc - sew / 2 - 0.01, sc + sew / 2 + 0.01)); ax4.text(sc, 0.62, "SE candidate", ha="center", va="center", color=T_SE, fontweight="bold", fontsize=9, zorder=5)
def free(k, a_, b_): return all(b_ + 0.006 < u or a_ - 0.006 > v for u, v in rows[k])
def model(r, k, col, lw_):
    y_ = k + 0.17; a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6; ax4.plot([a_, b_], [y_, y_], color=col, lw=0.7, zorder=3, solid_capstyle="butt")
    for es, ee in zip(map(int, r.exonStarts.strip(",").split(",")), map(int, r.exonEnds.strip(",").split(","))):
        if ee > L0 and es < L1: ax4.add_patch(plt.Rectangle((es / 1e6, y_ - 0.13), max((ee - es) / 1e6, 0.0012), 0.26, color=col, lw=0, zorder=4))
    tss = (r.txStart if r.strand == "+" else r.txEnd) / 1e6; d_ = 0.012 if r.strand == "+" else -0.012
    if XL[0] <= tss <= XL[1]: ax4.plot([tss, tss], [y_, y_ - 0.22], color=col, lw=0.7, zorder=4); ax4.annotate("", xy=(tss + d_, y_ - 0.22), xytext=(tss, y_ - 0.22), arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
named = []
for r in J.itertuples():
    col = T_P if r.prio == 0 else INK if r.prio == 1 else "#555555"; a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6; kw = dict(fontstyle="italic", fontweight="bold", fontsize=9) if r.prio == 0 else dict(fontstyle="italic", fontsize=8); w_ = tw(r.name2, **kw); c_ = (a_ + b_) / 2; ta, tb = c_ - w_ / 2, c_ + w_ / 2; ok = None
    for k in range(4):
        if free(k, min(a_, ta), max(b_, tb)): ok = k; break
    if ok is not None and tb < XL[1] and ta > XL[0]: rows[ok].append((min(a_, ta), max(b_, tb))); model(r, ok, col, 1); ax4.text(c_, ok + 0.62, r.name2, ha="center", va="center", color=col, zorder=5, **kw); named.append(r.name2)
    else:
        k = min(range(4), key=lambda q: sum(1 for u, v in used[q] if a_ < v and u < b_)); used[k].append((a_, b_)); model(r, k, col if r.prio < 2 else "#8A8A8A", 1)
print("named", named, "of", len(J))
kx = 0.012
for c_, l_, wd_ in [(GCOL["enhancer"], "enhancer", 0.098), (GCOL["TSS/flank"], "TSS/flank", 0.104), (GCOL["transcribed"], "transcribed", 0.116), (GCOL["repressed"], "repressed", 0.122), (C_SE, "SE candidate", 0.135), (C_P, "promoter partner", 0.172), (C_E, "element partner", 0.16)]:
    fig.add_artist(plt.Rectangle((kx, (0.1 - 0.045) / H), 0.018, 0.09 / H, transform=fig.transFigure, color=c_, lw=0)); fig.text(kx + 0.023, 0.1 / H, l_, va="center", fontsize=9); kx += wd_
for e in ("svg", "pdf"): fig.savefig(O / f"se_panel_a_smallfont.{e}", dpi=600, transparent=True)
fig.savefig(O / "se_panel_a_smallfont.png", dpi=600, facecolor="white"); fig.savefig(O / "se_panel_a_smallfont_check.png", dpi=144, facecolor="white")
