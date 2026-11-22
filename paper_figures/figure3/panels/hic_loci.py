"""Figure 3 Hi-C locus panels (hic_ets2_locus, hic_rbbp4_locus): one 3-region higher-order coordinated binding module on wild-type
HEK293T Hi-C. Top: balanced 10 kb contacts (log10, pixels as true 45-degree diamonds, full triangle), circles on the three anchor-pair
contacts (A-B, B-C, A-C). Middle: the module (anchors A, B, C and its three edges). Below: number of TFs with a Cicero peak per 5 kb
(dark bars and labels = the anchors: element = promoter of a GENCODE v46 protein-coding/lncRNA gene with a TSS within the locus
promoter window, else distal; n TFs = TFs with a peak midpoint within +-1 kb). Peak set and promoter window per locus (hic_loci.json):
ETS2 = MACS2 q <= 0.05 peaks, +-2.5 kb; RBBP4 = Cicero peaks, +-1 kb (as in the two approved panels). Bottom: genes (anchor-promoter genes first, then protein-coding genes by
length, up to 9, two rows without label overlap; or the genes listed in the locus config).

Inputs (source_data/): hic_loci.json, hic_locus_<name>_10kb.npz, hic_locus_<name>_peaks.tsv, hic_locus_<name>_genes.tsv
Output: hic_<name>_locus.{pdf,svg,png}, 8.30 x 6.60 in (large original; scale it down when placing). Hi-C pixels are raster.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

BLUE, INK, PROM = "#4A6FA5", "#222222", "#0072B2"
W, H = 8.30, 7.50


def draw(src, out, name, L):
    import matplotlib.pyplot as plt
    z = np.load(src / f"hic_locus_{name}_10kb.npz")
    X0, X1, r0, c = int(z["start"]), int(z["end"]), int(z["res"]), str(z["chrom"])
    M = z["balanced"]
    P = pd.read_csv(src / f"hic_locus_{name}_peaks.tsv", sep="\t")
    P["sm"] = (P.start + P.end) // 2
    GG = pd.read_csv(src / f"hic_locus_{name}_genes.tsv", sep="\t")
    GG["tss"] = np.where(GG.strand == "+", GG.start, GG.end)
    mid = lambda r: sum(map(int, r.split(":")[1].split("-"))) // 2
    regs = L["regions"]
    mA, mB, mC = (mid(r) for r in regs)
    cA, cB, cC = L["circles"]

    def elem(reg):
        s, e = map(int, reg.split(":")[1].split("-"))
        pw = L["promoter_window"]
        g = GG[(GG.tss >= s - pw) & (GG.tss <= e + pw)]
        k = P[(P.sm >= s - 1000) & (P.sm <= e + 1000)].tf.nunique()
        if len(g):
            gg = g.sort_values("type", kind="stable").gene.iloc[0]
            return f"{gg} promoter", k, gg
        return "distal", k, ""

    span = (X1 - X0) / 1e3
    n = M.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.log10(M + np.nanmedian(M[M > 0]) * 0.1)
    fig = plt.figure(figsize=(W, H))
    Lx, AW = 1.05 / W, 6.90 / W
    XL = (X0 / 1e6, X1 / 1e6)
    ymax = span / 2
    hh = 6.90 * ymax / span                               # equal x and y scale: pixels are true diamonds
    top = H - 0.28
    ax = fig.add_axes([Lx, (top - hh) / H, AW, hh / H])
    xs = X0 + np.arange(n + 1) * r0
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z))
    pc = ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 20), vmax=np.nanpercentile(Z, 98.5),
                       shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False)
    for nm, i, j in (("A–B", cA, cB), ("B–C", cB, cC), ("A–C", cA, cC)):
        ax.plot((i + j) / 2e6, (j - i) / 2e3, "o", mfc="none", mec=INK, ms=25, mew=2, zorder=5)
        ax.text((i + j) / 2e6, (j - i) / 2e3 + 0.075 * ymax, nm, ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK, zorder=6)
    ax.set_xlim(*XL)
    ax.set_ylim(0, ymax)
    ax.tick_params(bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Distance / 2 (kb)", fontsize=14)
    if L.get("yticks"):
        ax.set_yticks(L["yticks"])
    else:
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    ax.set_title(f"WT HEK293T Hi-C, 10 kb   {c}:{X0 + 1:,}–{X1:,} (hg38)", loc="left", fontsize=14, fontweight="bold", pad=6)
    cb = fig.add_axes([Lx + AW - 1.45 / W, (top - 0.42) / H, 1.4 / W, 0.14 / H])
    plt.colorbar(pc, cax=cb, orientation="horizontal")
    cb.set_title("log10 contact", fontsize=12, pad=3)
    cb.tick_params(labelsize=12)
    cb.xaxis.set_major_locator(plt.MaxNLocator(3))

    y2 = top - hh - 0.86
    ax2 = fig.add_axes([Lx, y2 / H, AW, 0.82 / H], sharex=ax)
    ax2.set_axis_off()
    ax2.set_ylim(0, 1.2)
    AN = [("A", mA), ("B", mB), ("C", mC)]
    for a, b in ((mA, mB), (mB, mC), (mA, mC)):
        q = np.linspace(a, b, 100) / 1e6
        ax2.plot(q, 0.5 + (0.65 * (b - a) / (mC - mA)) * np.sin(np.pi * (q - a / 1e6) / ((b - a) / 1e6)), color=BLUE, lw=2)
    for nm, m in AN:
        ax2.plot(m / 1e6, 0.5, "v", ms=10, color=INK, zorder=4)
        ax2.text(m / 1e6, 0.33, nm, ha="center", va="top", fontsize=14, fontweight="bold")
    ax2.text(XL[1], 1.17, f"Coordinated binding ({L['tf']})", color=BLUE, fontsize=12, va="top", ha="right")

    y3 = y2 - 0.84
    ax3 = fig.add_axes([Lx, y3 / H, AW, 0.68 / H], sharex=ax)
    ed = np.arange(X0 // 5000 * 5000, X1 + 5000, 5000)
    cnt = np.array([P[(P.sm >= e) & (P.sm < e + 5000)].tf.nunique() for e in ed])
    ymx = max(26, int(cnt.max() * 1.3))
    bw = 0.0048 * span / 840
    ax3.bar((ed + 2500) / 1e6, cnt, width=bw, color="#BFBFBF", lw=0)
    ax3.set_ylabel("TFs", fontsize=14)
    ax3.set_ylim(0, ymx)
    ax3.set_yticks([0, 20])
    ax3.tick_params(bottom=False, labelbottom=False)
    anchor_genes = []
    for (nm, m), reg, ha in zip(AN, regs, ("left", "center", "right")):
        lab, k, gg = elem(reg)
        anchor_genes.append(gg)
        ax3.bar(m / 1e6, max(k, 0.8), width=bw, color=INK, lw=0)
        ax3.text(m / 1e6 if nm != "C" else XL[1], ymx * 1.06, f"{lab}, {k} TF" + ("s" if k != 1 else ""), fontsize=12, ha=ha, va="bottom", clip_on=False)

    # genes, genome-browser style: canonical-transcript exons (boxes), introns (line with strand chevrons), TSS arrow = promoter
    y4 = y3 - 1.36
    ax4 = fig.add_axes([Lx, y4 / H, AW, 1.30 / H], sharex=ax)
    NR, RS = 3, 1.25                                         # rows, row spacing
    ax4.set_ylim((NR - 1) * RS + 0.85, -0.55)
    ax4.set_yticks([])
    ax4.spines["left"].set_visible(False)
    # named genes only; read-through genes (two gene names joined by "-") are left out, antisense/divergent (-AS, -DT) kept
    named = ~GG.gene.str.startswith("ENSG") & (~GG.gene.str.contains("-") | GG.gene.str.contains(r"-(?:AS|DT|IT|OT)\d*$"))
    show = GG[named].copy()
    if L.get("genes"):
        show = show[show.gene.isin(L["genes"])]
    key = {g for g in anchor_genes if g}
    show["key"] = show.gene.isin(key)
    show["L"] = show.end - show.start
    show = show.sort_values(["key", "type", "L"], ascending=[False, False, False], kind="stable")
    rnd = fig.canvas.get_renderer()
    mb_per_in = span / 1e3 / 6.90

    def text_w(t, **kw):
        o = fig.text(0, 0, t, **kw)
        w_ = o.get_window_extent(rnd).width / fig.dpi * mb_per_in
        o.remove()
        return w_

    occ = {k: [] for k in range(NR)}
    chev = span / 1e3 / 45                                  # chevron spacing (Mb)
    for g in show.itertuples():
        a, b = max(g.start, X0) / 1e6, min(g.end, X1) / 1e6
        col = PROM if g.key else INK
        kw = dict(fontsize=11, fontstyle="italic", fontweight="bold" if g.key else "normal")
        w_ = text_w(g.gene, **kw)
        c_ = min(max((a + b) / 2, XL[0] + w_ / 2), XL[1] - w_ / 2)
        lo_, hi_ = min(a, c_ - w_ / 2) - 0.004 * span / 840, max(b, c_ + w_ / 2) + 0.004 * span / 840
        rw = next((k for k in range(NR) if all(hi_ < u or lo_ > v for u, v in occ[k])), None)
        if rw is None:
            continue
        occ[rw].append((lo_, hi_))
        y_ = rw * RS + 0.30
        ax4.plot([a, b], [y_, y_], color=col, lw=0.9, solid_capstyle="butt", zorder=2)
        sgn = 1 if g.strand == "+" else -1
        for x in np.arange(a + chev / 2, b - chev / 4, chev):         # strand chevrons along the gene body
            ax4.plot([x - sgn * 0.0018 * span / 840 * 2, x, x - sgn * 0.0018 * span / 840 * 2], [y_ - 0.09, y_, y_ + 0.09], color=col, lw=0.7, zorder=2)
        es = [int(v) for v in str(g.exon_starts).split(",") if v not in ("", "nan")]
        ee = [int(v) for v in str(g.exon_ends).split(",") if v not in ("", "nan")]
        for s0, e0 in zip(es, ee):
            if e0 > X0 and s0 < X1:
                ax4.add_patch(plt.Rectangle((max(s0, X0) / 1e6, y_ - 0.14), max((min(e0, X1) - max(s0, X0)) / 1e6, 0.0006 * span / 840), 0.28,
                                            color=col, lw=0, zorder=3))
        tss = (g.start if g.strand == "+" else g.end) / 1e6
        if XL[0] <= tss <= XL[1]:                                   # promoter: bent TSS arrow in the direction of transcription
            d_ = sgn * 0.012 * span / 840
            ax4.plot([tss, tss, tss + d_ * 0.6], [y_ - 0.14, y_ - 0.42, y_ - 0.42], color=col, lw=1.0, zorder=4)
            ax4.annotate("", xy=(tss + d_, y_ - 0.42), xytext=(tss + d_ * 0.55, y_ - 0.42),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=8, shrinkA=0, shrinkB=0), zorder=4)
        ax4.text(c_, y_ + 0.20, g.gene, ha="center", va="top", color=col, zorder=5, **kw)
    ax4.set_xlim(*XL)
    if L.get("xticks"):
        ax4.set_xticks(L["xticks"])
    else:
        ax4.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax4.set_xlabel(f"{c} (Mb)", fontsize=14, labelpad=2)
    ax4.tick_params(labelsize=12)
    style.save(fig, out, f"hic_{name}_locus")


def main(src, out):
    cfg = json.load(open(src / "hic_loci.json"))
    for name, L in cfg.items():
        style.use(base_pt=12, small_pt=12)
        import matplotlib.pyplot as plt
        plt.rcParams.update({"axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4})
        draw(src, out, name, L)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
