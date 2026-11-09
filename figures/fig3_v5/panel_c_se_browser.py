"""Fig 3c v5 (179 x 41 mm): super-enhancer example, SEdb SE_02_195100173 (HEK293T, chr19:1,248,039-1,254,347).
Tracks: raw balanced Hi-C (10 kb, log10, 45 degrees, short), Peakachu 0.7 loops (5 kb U 10 kb, grey), the 31 Cicero
edges from the SE to its partners (red solid = on a Hi-C loop, blue dashed = Cicero only; partner-partner edges left out),
partner ticks (black promoter, orange enhancer, light grey other), the 5 TFs with the highest CPM over the SE among the
10 TFs cached in tf_cov_2kb.tsv (raw read 5' ends, CPM per 2 kb, per-track scale = max(hub bins, p99.5)), RefSeq Select
models of the partner-promoter genes only. Gold band = the SE. Inputs as fig3_v4 (inputs/ + se_sedb/model/inputs/hic_10kb.npz)."""
import json
import numpy as np, pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from fig3common import P, INP, XULAB, RED, BLUE, INK, GREY, LIGHT, PROM, ENH, OTHER, SE_EDGE, FS, fig_mm, mm_axes, save, plt

R0 = XULAB / "results_3d_model/se_sedb"; LOOPD = XULAB / "hic/work_wt293t/loops"
L0, L1 = 430000, 2250000; SE0, SE1 = 1248039, 1254347; XL = (L0 / 1e6, L1 / 1e6); SEBAND = "#FFE9A8"; TRACK = "#4D4D4D"
TCOL = dict(promoter=PROM, enhancer=ENH, other=OTHER)
mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2
E = pd.read_csv(INP / "edges_in_structure.tsv", sep="\t"); assert len(E) == 31
HE = pd.read_csv(INP / "hub_edges_all.tsv", sep="\t"); SEd = HE[HE.from_se == 1].copy(); assert len(SEd) == 31
SEd["pos"] = [mid(rb) if str(a_) == "SE" else mid(ra) for ra, rb, a_ in zip(SEd.reg_a, SEd.reg_b, SEd.a)]
Tc = pd.read_csv(INP / "tf_cov_2kb.tsv", sep="\t", header=None, names=["tf", "b", "n"])
nb = (L1 - L0) // 2000; ed = (L0 + np.arange(nb + 1) * 2000) / 1e6
seb = list(range((SE0 - L0) // 2000, (SE1 - L0) // 2000 + 1))
cov = {}
for t, d in Tc.groupby("tf"):
    tot = int(d[d.b == "total"].n.iloc[0]); h = np.zeros(nb)
    for r in d[d.b != "total"].itertuples(): h[int(r.b)] = r.n
    cov[t] = h / tot * 1e6
top = sorted(cov, key=lambda t: -cov[t][seb].sum())[:5]
print("CPM over the SE:", {t: round(cov[t][seb].sum(), 1) for t in sorted(cov, key=lambda t: -cov[t][seb].sum())}, "-> top 5", top)

W, H = 179.0, 41.0; X0, AW = 13.0, 146.0
fig = fig_mm(W, H)
def band(a): a.axvspan((SE0 - 2500) / 1e6, (SE1 + 2500) / 1e6, color=SEBAND, lw=0, zorder=0)
def lab(a, t, **kw): a.text(-0.006, 0.5, t, transform=a.transAxes, ha="right", va="center", fontsize=FS, **kw)
y = H - 0.6
# Hi-C
hh = 5.0; y -= hh; ax = mm_axes(fig, W, H, X0, y, AW, hh); ymax = hh / AW * (L1 - L0) / 1e3
M = np.load(R0 / "model/inputs/hic_10kb.npz")["ice"]; n = M.shape[0]; Z = np.log10(np.nan_to_num(M) + 1e-5)
xs = L0 + np.arange(n + 1) * 10000; XX, YY = np.meshgrid(xs, xs, indexing="ij"); Zm = np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z)
ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap=P.seq_cmap(), vmin=np.nanpercentile(Z, 50), vmax=np.nanpercentile(Z, 99.5), shading="flat", rasterized=True, lw=0, antialiased=False)
ax.set_xlim(*XL); ax.set_ylim(0, ymax); ax.set_axis_off(); lab(ax, "Hi-C")
# loops
loops = set()
for r_ in (5000, 10000):
    for ln in open(LOOPD / f"peakachu_{r_}.loops.0.7.bedpe"):
        x = ln.split()
        if x[0] == "chr19" and x[3] == "chr19":
            a_, b_ = sorted(((int(x[1]) + int(x[2])) // 2, (int(x[4]) + int(x[5])) // 2))
            if a_ >= L0 and b_ <= L1: loops.add((a_ // 10000, b_ // 10000))
y -= 2.6; ax1 = mm_axes(fig, W, H, X0, y, AW, 2.3); band(ax1); ax1.set_xlim(*XL); ax1.set_ylim(-1.05, 0); ax1.set_axis_off()
for a_, b_ in loops:
    q = np.linspace(a_, b_, 40) * 0.01; ax1.plot(q, -min(1, (b_ - a_) / 90) * np.sin(np.pi * (q - q[0]) / (q[-1] - q[0])), color=GREY, lw=0.3)
lab(ax1, "Loops")
# SE edges
y -= 6.8; ax2 = mm_axes(fig, W, H, X0, y, AW, 6.6); band(ax2); ax2.set_xlim(*XL); ax2.set_ylim(0, 1.05); ax2.set_axis_off()
sx = (SE0 + SE1) / 2e6
for r in SEd.sort_values("on_loop").itertuples():
    a_, b_ = sorted((sx, r.pos / 1e6)); q = np.linspace(a_, b_, 80); h_ = min(1.0, 0.15 + 0.85 * (b_ - a_) / 0.75)
    lp = r.on_loop == 1
    ax2.plot(q, h_ * np.sin(np.pi * (q - a_) / (b_ - a_)), color=RED if lp else BLUE, lw=0.8 if lp else 0.6, ls="-" if lp else (0, (2.2, 1.6)), zorder=3 if lp else 2)
lab(ax2, "SE edges")
# partner ticks
y -= 1.9; axp = mm_axes(fig, W, H, X0, y, AW, 1.7); band(axp); axp.set_xlim(*XL); axp.set_ylim(0, 1); axp.set_axis_off()
for r in E.itertuples(): axp.add_line(Line2D([r.pos / 1e6] * 2, [0.05, 0.95], color=TCOL[r.cls], lw=0.9, solid_capstyle="butt"))
lab(axp, "Partners")
# TF coverage
hub = sorted(set(np.r_[(E.pos.values - L0) // 2000, seb].tolist())); rh = 2.2; y -= 0.4
for t in top:
    y -= rh; a_ = mm_axes(fig, W, H, X0, y + 0.2, AW, rh - 0.2); h = cov[t]; lim = max(h[hub].max(), np.percentile(h, 99.5)); band(a_)
    a_.fill_between(ed[:-1], 0, np.minimum(h, lim), step="post", color=TRACK, lw=0, zorder=2); a_.set_xlim(*XL); a_.set_ylim(0, lim)
    a_.set_yticks([]); a_.set_xticks([]); [a_.spines[s].set_visible(False) for s in ("left", "top", "right")]
    a_.spines["bottom"].set_linewidth(0.3); a_.spines["bottom"].set_color(LIGHT); lab(a_, t)
# genes: partner-promoter genes only, one lane; one label per partner promoter (ADAT3/SCAMP4 share one), one row, 1-D repel
plab = sorted({l.split(" ")[0].replace("RNU6-2/", "") for l, c in zip(HE.label_a.tolist() + HE.label_b.tolist(), HE.cls_a.tolist() + HE.cls_b.tolist()) if c == "promoter"})
prom = {g for l in plab for g in l.split("/")}
J = pd.DataFrame(json.load(open(INP / "refseq_select_chr19_0.43-2.25Mb.json"))["ncbiRefSeqSelect"]); J = J[(J.txEnd > L0) & (J.txStart < L1) & J.name2.isin(prom)].sort_values("txStart")
y -= 6.4; ax4 = mm_axes(fig, W, H, X0, y, AW, 6.0); band(ax4); ax4.set_xlim(*XL); ax4.set_ylim(2.3, -0.4)
bp_mm = (L1 - L0) / 1e6 / AW; tss = {}
for r in J.itertuples():
    a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, L1) / 1e6; ax4.plot([a_, b_], [0, 0], color=INK, lw=0.4, solid_capstyle="butt", zorder=3)
    for es, ee in zip(map(int, r.exonStarts.strip(",").split(",")), map(int, r.exonEnds.strip(",").split(","))):
        if ee > L0 and es < L1: ax4.add_patch(Rectangle((es / 1e6, -0.22), max((ee - es) / 1e6, 0.0008), 0.44, color=INK, lw=0, zorder=4))
    tss[r.name2] = (r.txStart if r.strand == "+" else r.txEnd) / 1e6
named = sorted([(l, float(np.mean([tss[g] for g in l.split("/")]))) for l in plab] + [("SE", sx)], key=lambda kv: kv[1])
ly = 1.55; xs_ = [x_ for _, x_ in named]; wd = [len(n_) * FS * 0.6 * 0.3528 * bp_mm + 1.8 * bp_mm for n_, _ in named]
for _ in range(500):
    for i in range(1, len(xs_)):
        need = (wd[i] + wd[i - 1]) / 2; d = xs_[i] - xs_[i - 1]
        if d < need: xs_[i - 1] -= (need - d) / 2; xs_[i] += (need - d) / 2
    xs_ = list(np.clip(xs_, XL[0] + wd[0] / 2, XL[1] - wd[-1] / 2))
for (n_, x_), xl in zip(named, xs_):
    se = n_ == "SE"
    ax4.plot([x_, x_, xl], [0.3, 0.6, ly - 0.55], color=SE_EDGE if se else GREY, lw=0.3, zorder=5)
    ax4.text(xl, ly, n_, ha="center", va="center", fontsize=FS, fontstyle="normal" if se else "italic", fontweight="bold" if se else "normal", color=SE_EDGE if se else INK)
ax4.set_yticks([]); [ax4.spines[s].set_visible(False) for s in ("left", "top", "right")]; ax4.spines["bottom"].set_linewidth(0.4)
ax4.set_xticks(np.arange(0.6, 2.3, 0.2), [f"{v:.1f}" for v in np.arange(0.6, 2.3, 0.2)]); ax4.tick_params(axis="x", length=1.5, pad=1)
ax4.set_xlabel("chr19 (Mb)", labelpad=1, fontsize=FS); lab(ax4, "Genes")
# compact key, right column
lg = fig.add_axes([0, 0, 1, 1]); lg.set_xlim(0, W); lg.set_ylim(0, H); lg.set_axis_off(); lg.patch.set_alpha(0)
lx = X0 + AW + 2.5; yy = H - 7.5
lg.add_line(Line2D([lx, lx + 3.5], [yy, yy], color=RED, lw=0.9)); lg.text(lx + 4.5, yy, "Hi-C loop", fontsize=FS, va="center"); yy -= 2.6
lg.add_line(Line2D([lx, lx + 3.5], [yy, yy], color=BLUE, lw=0.7, ls=(0, (2.2, 1.6)))); lg.text(lx + 4.5, yy, "Cicero only", fontsize=FS, va="center"); yy -= 3.2
for c_, nm in ((PROM, "Promoter"), (ENH, "Enhancer")):
    lg.add_line(Line2D([lx + 1.7] * 2, [yy - 0.8, yy + 0.8], color=c_, lw=0.9, solid_capstyle="butt")); lg.text(lx + 4.5, yy, nm, fontsize=FS, va="center"); yy -= 2.6
print("loops", len(loops), "SE edges", len(SEd), "on loop", int(SEd.on_loop.sum()), "genes", [n_ for n_, _ in named], "bottom y", round(y, 1))
save(fig, "fig3c_se_browser")
