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

BLUE, INK = "#4A6FA5", "#222222"
W, H = 8.30, 6.60


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
    top = 6.32
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
    ax.set_title("WT HEK293T Hi-C, 10 kb", loc="left", fontsize=14, fontweight="bold", pad=6)
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

    y4 = y3 - 0.62
    ax4 = fig.add_axes([Lx, y4 / H, AW, 0.56 / H], sharex=ax)
    ax4.set_ylim(1.75, -0.65)
    ax4.set_yticks([])
    ax4.spines["left"].set_visible(False)
    if L.get("genes"):                                   # genes and rows given in the locus config
        rows = L["genes"]
        for nm, rw in rows.items():
            g = GG[GG.gene == nm].iloc[0]
            a, b = max(g.start, X0) / 1e6, min(g.end, X1) / 1e6
            ax4.plot([a, b], [rw + 0.32] * 2, color=INK, lw=2.2, solid_capstyle="butt")
            ax4.text(max((a + b) / 2, XL[0] + 0.03), rw + 0.2, nm, fontsize=12, fontstyle="italic", ha="center", va="bottom")
    else:                                                # anchor-promoter genes first, then protein-coding genes by length
        key = [g for g in anchor_genes if g]
        rest = GG[(GG.type == "pc") & ~GG.gene.isin(key) & ~GG.gene.str.startswith("ENSG") & ~GG.gene.str.contains("-")]
        rest = rest.assign(L=rest.end - rest.start).sort_values("L", ascending=False, kind="stable").gene.tolist()
        names = list(dict.fromkeys(key + rest))[:9]
        occ = {0: [], 1: []}
        cw = 0.092 * span / 1e3 / 6.9                    # Mb per character of 12 pt italic Arial
        for nm in sorted(names, key=lambda g: GG[GG.gene == g].start.iloc[0]):
            g = GG[GG.gene == nm].iloc[0]
            a, b = max(g.start, X0) / 1e6, min(g.end, X1) / 1e6
            xc = min(max((a + b) / 2, XL[0] + 0.5 * cw * len(nm)), XL[1] - 0.5 * cw * len(nm))
            half = 0.5 * cw * len(nm) + cw
            rw = next((r_ for r_ in (0, 1) if all(xc - half > hi or xc + half < lo for lo, hi in occ[r_])), None)
            if rw is None:
                continue
            occ[rw].append((xc - half, xc + half))
            ax4.plot([a, b], [rw + 0.32] * 2, color=INK, lw=2.2, solid_capstyle="butt")
            ax4.text(xc, rw + 0.2, nm, fontsize=12, fontstyle="italic", ha="center", va="bottom")
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
