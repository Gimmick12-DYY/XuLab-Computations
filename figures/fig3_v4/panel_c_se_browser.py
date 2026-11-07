"""Fig 3c (179 x 52 mm): super-enhancer example, SEdb SE_02_195100173 (HEK293T, chr19:1,248,039-1,254,347) and its
hub, re-laid out at final size from results_3d_model/se_sedb/scripts/28_browser_gold.py (the slide version with the gold
SE band): raw balanced Hi-C (10 kb, log10, true 45 degrees), Peakachu 0.7 loops (5 kb U 10 kb), all 135 Cicero edges
among the SE and its 31 partners (colour = what the edge connects, as in d2; solid = on a Hi-C loop, dashed = Cicero
only; thick = from the SE), raw per-TF coverage (read 5' ends, CPM per 2 kb, per-track scale = max(hub bins, p99.5)),
ChromHMM (palette C groups), RefSeq Select gene models (names = genes whose promoter is a hub partner). Partner stripes neutral.
Inputs (inputs/, copied from the Mac se_sedb/se_hub figure folders): hub_edges_all.tsv, edges_in_structure.tsv,
tf_cov_2kb.tsv, chromhmm_window.tsv, refseq_select_chr19_0.43-2.25Mb.json; Hi-C from se_sedb/model/inputs/hic_10kb.npz."""
import json
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from fig3common import P, INP, XULAB, fig_mm, mm_axes, save, plt

R0 = XULAB / "results_3d_model/se_sedb"; LOOPD = XULAB / "hic/work_wt293t/loops"
N = P.neutral
COL = dict(ZNF507="#C48311", SOX4="#32A190", NFYA="#D9B637", GLYR1="#C681F1", NME2="#739B64", DR1="#8AC975", MTF2="#65278F", E4F1="#712F55", MBD3="#6E6E6E", MLX="#6E6E6E")
ORDER = ["SOX4", "NFYA", "MBD3", "E4F1", "ZNF507", "MLX", "NME2", "DR1", "GLYR1", "MTF2"]
EC = dict(se_promoter="#0072B2", se_enhancer="#D55E00", se_other="#999999", pp_promoter="#56B4E9", pp_enhancer="#F0A35E", pp_other="#D0D0D0")
SEBAND, SEEDGE, SETXT, STRIPE = "#FFE9A8", "#B38F00", "#8A6A00", "#E6E6E6"
L0, L1 = 430000, 2250000; SE0, SE1 = 1248039, 1254347; XL = (L0 / 1e6, L1 / 1e6)
GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer",
         "TssA": "TSS/flank", "TssFlnk": "TSS/flank", "TssFlnkU": "TSS/flank", "TssFlnkD": "TSS/flank", "TssBiv": "TSS/flank",
         "Tx": "transcribed", "TxWk": "transcribed", "ReprPC": "repressed", "ReprPCWk": "repressed", "Het": "repressed", "ZNF/Rpts": "repressed"}
GCOL = {"enhancer": P.chromhmm.active_enh, "TSS/flank": P.chromhmm.tss_flank, "transcribed": P.chromhmm.transcribed, "repressed": P.chromhmm.repressed}
mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2
E = pd.read_csv(INP / "edges_in_structure.tsv", sep="\t")
HE = pd.read_csv(INP / "hub_edges_all.tsv", sep="\t")
HE["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(HE.cls_a, HE.cls_b, HE.from_se, HE.type)]
Tc = pd.read_csv(INP / "tf_cov_2kb.tsv", sep="\t", header=None, names=["tf", "b", "n"])
nb = (L1 - L0) // 2000; ed = (L0 + np.arange(nb + 1) * 2000) / 1e6

W, H = 179.0, 52.0; X0, AW = 14.0, 136.0
fig = fig_mm(W, H)
def stripes(a):
    for p in E.pos: a.axvspan((p - 3000) / 1e6, (p + 3000) / 1e6, color=STRIPE, lw=0, zorder=0)
    a.axvspan((SE0 - 2500) / 1e6, (SE1 + 2500) / 1e6, color=SEBAND, lw=0, zorder=0.5)
    for v in (SE0 - 2500, SE1 + 2500): a.axvline(v / 1e6, color=SEEDGE, lw=0.3, zorder=0.6)
def lab(a, t, **kw): a.text(-0.008, 0.5, t, transform=a.transAxes, ha="right", va="center", fontsize=6, **kw)
y = H - 0.5
# Hi-C triangle at true 45 degrees
hh = 7.5; y -= hh; ax = mm_axes(fig, W, H, X0, y, AW, hh); ymax = hh / AW * (L1 - L0) / 1e3
M = np.load(R0 / "model/inputs/hic_10kb.npz")["ice"]; n = M.shape[0]; Z = np.log10(np.nan_to_num(M) + 1e-5)
xs = L0 + np.arange(n + 1) * 10000; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z)
ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap=P.seq_cmap(), vmin=np.nanpercentile(Z, 50), vmax=np.nanpercentile(Z, 99.5), shading="flat", rasterized=True, lw=0, antialiased=False)
ax.set_xlim(*XL); ax.set_ylim(0, ymax); ax.set_axis_off(); lab(ax, "Hi-C (10 kb)")
# loops
loops = set()
for r_ in (5000, 10000):
    for ln in open(LOOPD / f"peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == "chr19" and x[3] == "chr19":
            a_, b_ = sorted(((int(x[1]) + int(x[2])) // 2, (int(x[4]) + int(x[5])) // 2))
            if a_ >= L0 and b_ <= L1: loops.add((a_ // 10000, b_ // 10000))
y -= 2.6; ax1 = mm_axes(fig, W, H, X0, y, AW, 2.4); stripes(ax1); ax1.set_xlim(*XL); ax1.set_ylim(-1.05, 0); ax1.set_axis_off()
for a_, b_ in loops:
    q = np.linspace(a_, b_, 40) * 0.01; ax1.plot(q, -min(1, (b_ - a_) / 90) * np.sin(np.pi * (q - q[0]) / (q[-1] - q[0])), color=N.mid, lw=0.3)
lab(ax1, "Loops")
# Cicero edges
y -= 6.4; ax2 = mm_axes(fig, W, H, X0, y, AW, 6.2); stripes(ax2); ax2.set_xlim(*XL); ax2.set_ylim(0, 1.08); ax2.set_axis_off()
for r in HE.assign(o_=HE.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["from_se", "on_loop", "o_"]).itertuples():
    a_, b_ = sorted((mid(r.reg_a) / 1e6, mid(r.reg_b) / 1e6)); q = np.linspace(a_, b_, 80); h_ = min(1.0, 0.12 + 0.88 * (b_ - a_) / 1.1)
    ax2.plot(q, h_ * np.sin(np.pi * (q - a_) / (b_ - a_)), color=EC[r.ek], lw=(1.1 if r.from_se else 0.5) if r.on_loop else (0.5 if r.from_se else 0.3),
             ls="-" if r.on_loop else (0, (2.5, 2)), alpha=1 if r.on_loop else (0.85 if r.from_se else 0.55))
lab(ax2, "Cicero edges")
# TF coverage
hub = sorted(set(np.r_[(E.pos.values - L0) // 2000, (np.arange(SE0, SE1, 2000) - L0) // 2000].tolist())); rh = 2.0; y -= 0.3
for t in ORDER:
    y -= rh; a_ = mm_axes(fig, W, H, X0, y + 0.15, AW, rh - 0.15); d = Tc[Tc.tf == t]; tot = int(d[d.b == "total"].n.iloc[0]); h = np.zeros(nb)
    for r in d[d.b != "total"].itertuples(): h[int(r.b)] = r.n
    h = h / tot * 1e6; lim = max(h[hub].max(), np.percentile(h, 99.5)); stripes(a_)
    a_.fill_between(ed[:-1], 0, np.minimum(h, lim), step="post", color=COL[t], lw=0, zorder=2); a_.set_xlim(*XL); a_.set_ylim(0, lim)
    a_.set_yticks([]); a_.set_xticks([]); [a_.spines[s].set_visible(False) for s in ("left", "top", "right")]; a_.spines["bottom"].set_linewidth(0.3); a_.spines["bottom"].set_color(N.light)
    lab(a_, t, fontstyle="italic", color=COL[t] if t not in ("MBD3", "MLX") else N.dark)
# ChromHMM
y -= 1.7; axc = mm_axes(fig, W, H, X0, y, AW, 1.4); stripes(axc); axc.set_xlim(*XL); axc.set_ylim(0, 1); axc.set_axis_off()
Cm = pd.read_csv(INP / "chromhmm_window.tsv", sep="\t", header=None, names=["c", "s", "e", "st"])
for r in Cm.itertuples():
    g_ = GROUP.get(r.st)
    if g_: axc.add_patch(Rectangle((r.s / 1e6, 0.05), (r.e - r.s) / 1e6, 0.9, color=GCOL[g_], lw=0, zorder=2))
lab(axc, "ChromHMM")
# genes: all RefSeq Select models in two lanes, names only for hub-partner promoter genes
y -= 7.1; ax4 = mm_axes(fig, W, H, X0, y, AW, 6.9); stripes(ax4); ax4.set_xlim(*XL); ax4.set_ylim(2.95, -0.3)
J = pd.DataFrame(json.load(open(INP / "refseq_select_chr19_0.43-2.25Mb.json"))["ncbiRefSeqSelect"]); J = J[(J.txEnd > L0) & (J.txStart < L1)].sort_values("txStart")
prom = {x for l, c in zip(HE.label_a.tolist() + HE.label_b.tolist(), HE.cls_a.tolist() + HE.cls_b.tolist()) if c == "promoter" for x in l.split(" ")[0].replace("RNU6-2/", "").split("/")}
bp_mm = (L1 - L0) / 1e6 / AW; ends = [-1.0, -1.0]; named = []
def model(r, yy, col):
    a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6; ax4.plot([a_, b_], [yy, yy], color=col, lw=0.4, solid_capstyle="butt", zorder=3)
    for es, ee in zip(map(int, r.exonStarts.strip(",").split(",")), map(int, r.exonEnds.strip(",").split(","))):
        if ee > L0 and es < L1: ax4.add_patch(Rectangle((es / 1e6, yy - 0.16), max((ee - es) / 1e6, 0.0008), 0.32, color=col, lw=0, zorder=4))
for r in J.itertuples():
    is_p = r.name2 in prom; a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6
    lane = 0 if a_ > ends[0] + 0.004 else 1 if a_ > ends[1] + 0.004 else int(np.argmin(ends)); ends[lane] = max(ends[lane], b_)
    model(r, lane * 0.6, N.ink if is_p else N.mid)
    if is_p: named.append(((r.txStart if r.strand == "+" else r.txEnd) / 1e6, r.name2))
# names under the axis line of the gene lanes, spread apart (1-D repel), italic, hub-partner promoters only
named = sorted({n_: x_ for x_, n_ in named}.items(), key=lambda kv: kv[1]) + [("SE", (SE0 + SE1) / 2e6)]
named = sorted(named, key=lambda kv: kv[1])
for row, ly in ((0, 1.5), (1, 2.5)):   # names alternate between two rows; 1-D repel within a row
    sub = named[row::2]; xs_ = [x_ for _, x_ in sub]; wd = [len(n_) * 6 * 0.58 * 0.3528 * bp_mm + 1.5 * bp_mm for n_, _ in sub]
    for _ in range(300):
        for i in range(1, len(xs_)):
            need = (wd[i] + wd[i - 1]) / 2; d = xs_[i] - xs_[i - 1]
            if d < need: xs_[i - 1] -= (need - d) / 2; xs_[i] += (need - d) / 2
        xs_ = list(np.clip(xs_, XL[0] + wd[0] / 2, XL[1] - wd[-1] / 2))
    for (n_, x_), xl in zip(sub, xs_):
        se = n_ == "SE"
        ax4.plot([x_, x_, xl], [0.95, ly - 0.42, ly - 0.32], color=SEEDGE if se else N.mid, lw=0.3, zorder=5)
        ax4.text(xl, ly, n_, ha="center", va="center", fontsize=6, fontstyle="normal" if se else "italic", fontweight="bold" if se else "normal", color=SETXT if se else N.ink, zorder=5)
ax4.set_yticks([]); [ax4.spines[s].set_visible(False) for s in ("left", "top", "right")]
ax4.set_xticks(np.arange(0.6, 2.3, 0.2), [f"{v:.1f}" for v in np.arange(0.6, 2.3, 0.2)]); ax4.tick_params(axis="x", length=1.5, pad=1)
ax4.set_xlabel("chr19 (Mb, hg38)", labelpad=1, fontsize=6); lab(ax4, "Genes")
# legend (right column)
lg = fig.add_axes([0, 0, 1, 1]); lg.set_xlim(0, W); lg.set_ylim(0, H); lg.set_axis_off(); lg.patch.set_alpha(0)
lx = X0 + AW + 3.0; yy = H - 2.5
lg.text(lx, yy, "Edge from the SE to a", fontsize=6, va="center"); yy -= 2.6
for k, nm in (("se_promoter", "promoter"), ("se_enhancer", "enhancer"), ("se_other", "other site")):
    lg.add_line(Line2D([lx, lx + 3.5], [yy, yy], color=EC[k], lw=1.2)); lg.text(lx + 4.5, yy, nm, fontsize=6, va="center"); yy -= 2.4
lg.text(lx, yy, "Between partners", fontsize=6, va="center"); yy -= 2.6
for k, nm in (("pp_promoter", "promoter–promoter"), ("pp_enhancer", "with an enhancer"), ("pp_other", "other")):
    lg.add_line(Line2D([lx, lx + 3.5], [yy, yy], color=EC[k], lw=0.8)); lg.text(lx + 4.5, yy, nm, fontsize=6, va="center"); yy -= 2.4
lg.add_line(Line2D([lx, lx + 3.5], [yy, yy], color=N.dark, lw=0.8)); lg.text(lx + 4.5, yy, "on a Hi-C loop", fontsize=6, va="center"); yy -= 2.4
lg.add_line(Line2D([lx, lx + 3.5], [yy, yy], color=N.dark, lw=0.8, ls=(0, (2.5, 2)))); lg.text(lx + 4.5, yy, "Cicero only", fontsize=6, va="center"); yy -= 3.2
for c_, nm in ((SEBAND, "Super-enhancer"), (STRIPE, "Hub partner")):
    lg.add_patch(Rectangle((lx, yy - 0.9), 3.5, 1.8, color=c_, lw=0)); lg.text(lx + 4.5, yy, nm, fontsize=6, va="center"); yy -= 2.4
yy -= 0.8
for g_, nm in (("enhancer", "Enhancer"), ("TSS/flank", "TSS/flank"), ("transcribed", "Transcribed"), ("repressed", "Repressed")):
    lg.add_patch(Rectangle((lx, yy - 0.9), 3.5, 1.8, color=GCOL[g_], lw=0)); lg.text(lx + 4.5, yy, nm, fontsize=6, va="center"); yy -= 2.4
lg.text(lx, yy - 0.4, "Coverage: CPM per 2 kb", fontsize=6, va="center", color=N.dark)
print("loops", len(loops), "edges", len(HE), "on loop", int(HE.on_loop.sum()), "named genes", [n_ for n_, _ in named], "legend bottom (mm)", round(yy - 0.4, 1))
save(fig, "fig3c_se_browser")
