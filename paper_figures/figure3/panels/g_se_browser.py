"""Figure 3g: browser view of the chr19 super-enhancer hub (SEdb SE_02_1951, HEK293T), chr19:0.43-1.95 Mb (hg38).

Tracks, top to bottom: wild-type HEK293T Hi-C (10 kb, ICE, log10, shown to 150 kb); Peakachu score >= 0.7 loops (5 kb + 10 kb);
Cicero edges of the hub (colour = partner class, thick = from the SE, solid = on a Hi-C loop, dashed = Cicero only); single-cell
TF fragment coverage (CPM, 2 kb bins, scaled per track to the hub maximum); ChromHMM classes; RefSeq Select genes (promoter
partners in bold). Vertical bands: SE (gold) and partner elements by class.

Inputs (source_data/): se_hic_10kb.npz, peakachu_{5000,10000}.loops.0.7.bedpe.gz, se_hub_edges.tsv, se_partner_bands.tsv,
  se_tf_coverage_2kb.tsv, se_chromhmm_window.tsv, se_refseq_select.json
Output: g_se_browser.{pdf,svg,png}, about 155 mm wide (print size, same scale as the full 0.43-2.25 Mb design); labels 7 pt, ticks/legend 6 pt, minor gene names 5.5 pt.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from common import style  # noqa: E402

L0, L1 = 430000, 2250000          # data window (chr19, hg38): Hi-C matrix and TF coverage bins
XR = 1950000                      # right end of the shown window (PI 2026-11-21: right part removed; last partner at 1.926 Mb)
W0, H = 9.75, 4.87                # design canvas of the full window, inches; positions below are design inches
S = 0.739                         # print scale (183 mm page width incl. margins)
LX = 0.93                         # left margin (design inches)
AW0 = 8.72 * (XR - L0) / (L1 - L0)   # track width: same Mb per inch as the full-window design
W = W0 - (8.72 - AW0)
SE0, SE1 = 1248039, 1254347       # SEdb SE_02_1951
XL = (L0 / 1e6, XR / 1e6)
ORDER = ["SOX4", "NFYA", "MBD3", "E4F1", "ZNF507", "MLX", "NME2", "DR1", "GLYR1", "MTF2"]
TFC = dict(ZNF507="#C48311", SOX4="#32A190", NFYA="#D9B637", GLYR1="#C681F1", NME2="#739B64", DR1="#8AC975", MTF2="#65278F",
           E4F1="#712F55", MBD3="#6E6E6E", MLX="#6E6E6E")
INK = "#222222"
BAND_SE, TXT_SE = "#FFE9A8", "#8A6A00"
PBAND = dict(promoter="#CFE2F3", enhancer="#FCDCC2", other="#E6E6E6")
GROUP = {"EnhA1": "enhancer", "EnhA2": "enhancer", "EnhG1": "enhancer", "EnhG2": "enhancer", "EnhWk": "enhancer", "EnhBiv": "enhancer",
         "TssA": "TSS/flank", "TssFlnk": "TSS/flank", "TssFlnkU": "TSS/flank", "TssFlnkD": "TSS/flank", "TssBiv": "TSS/flank",
         "Tx": "transcribed", "TxWk": "transcribed", "ReprPC": "repressed", "ReprPCWk": "repressed", "Het": "repressed", "ZNF/Rpts": "repressed"}
GCOL = {"enhancer": "#E67E22", "TSS/flank": "#C0392B", "transcribed": "#5D8AA8", "repressed": "#7D8B99"}
F_LAB, F_SMALL, F_MINOR = 7.0, 6.0, 5.5


def mid(r):
    return sum(map(int, r.split(":")[1].split("-"))) // 2


def main(src, out):
    style.use(base_pt=F_LAB, small_pt=F_LAB)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.linewidth": S, "xtick.major.width": S, "ytick.major.width": S, "xtick.major.size": 4 * S, "ytick.major.size": 4 * S})
    EC = style.SE_EDGE
    E = pd.read_csv(src / "se_partner_bands.tsv", sep="\t")
    HE = pd.read_csv(src / "se_hub_edges.tsv", sep="\t")
    HE["ek"] = [("se_" + (cb if ca == "se" else ca)) if fs else "pp_" + ty for ca, cb, fs, ty in zip(HE.cls_a, HE.cls_b, HE.from_se, HE.type)]
    Tc = pd.read_csv(src / "se_tf_coverage_2kb.tsv", sep="\t", header=None, names=["tf", "b", "n"])
    nb = (L1 - L0) // 2000
    ed = (L0 + np.arange(nb + 1) * 2000) / 1e6

    fig = plt.figure(figsize=(W * S, H * S))
    Lx, AW = LX / W, AW0 / W

    def stripes(a):
        for p, c in zip(E.pos, E.cls):
            a.axvspan((p - 3000) / 1e6, (p + 3000) / 1e6, color=PBAND.get(c, PBAND["other"]), lw=0, zorder=0)
        a.axvspan((SE0 - 2500) / 1e6, (SE1 + 2500) / 1e6, color=BAND_SE, lw=0, zorder=0.5)
        for x in (SE0 - 2500, SE1 + 2500):
            a.axvline(x / 1e6, color="#B38F00", lw=0.5 * S, zorder=0.6)

    def label(y, t, **kw):
        fig.text(0.008, y / H, t, va="center", ha="left", **kw)

    # Hi-C triangle
    ymax = 150.0
    hh = AW0 * ymax / (XR - L0) * 1000
    b_h = H - 0.33 - hh
    ax = fig.add_axes([Lx, b_h / H, AW, hh / H])
    first = ax
    Mx = np.load(src / "se_hic_10kb.npz")["ice"]
    Z = np.log10(np.nan_to_num(Mx) + 1e-5)
    xs = L0 + np.arange(Mx.shape[0] + 1) * 10000
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    Zm = np.ma.masked_where(np.tril(np.ones_like(Z), -1) > 0, Z)
    ax.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="Reds", vmin=np.nanpercentile(Z, 50), vmax=np.nanpercentile(Z, 99.5),
                  shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False)
    ax.set_xlim(*XL)
    ax.set_ylim(0, ymax)
    ax.set_axis_off()
    label(b_h + hh * 0.5, "Hi-C\n10 kb")

    # Peakachu loops
    loops = set()
    for r_ in (5000, 10000):
        with gzip.open(src / f"peakachu_{r_}.loops.0.7.bedpe.gz", "rt") as fh:
            for ln in fh:
                x = ln.split()
                if x[0] == "chr19" and x[3] == "chr19":
                    a, b = sorted(((int(x[1]) + int(x[2])) // 2, (int(x[4]) + int(x[5])) // 2))
                    if a >= L0 and b <= XR:
                        loops.add((a // 10000, b // 10000))
    y2 = b_h - 0.24
    ax1 = fig.add_axes([Lx, y2 / H, AW, 0.22 / H], sharex=first)
    stripes(ax1)
    ax1.set_ylim(-1.05, 0)
    ax1.set_axis_off()
    for a, b in sorted(loops):
        q = np.linspace(a, b, 40) * 0.01
        ax1.plot(q, -min(1, (b - a) / 90) * np.sin(np.pi * (q - q[0]) / (q[-1] - q[0])), color="#7F7F7F", lw=0.35 * S, alpha=0.6)
    label(y2 + 0.11, "Loops")

    # Cicero edges
    y3 = y2 - 0.54
    ax2 = fig.add_axes([Lx, y3 / H, AW, 0.52 / H], sharex=first)
    stripes(ax2)
    ax2.set_ylim(0, 1.08)
    ax2.set_axis_off()
    for r in HE.assign(o_=HE.type.map(dict(other=0, enhancer=1, promoter=2))).sort_values(["from_se", "on_loop", "o_"]).itertuples():
        a_, b_ = sorted((mid(r.reg_a) / 1e6, mid(r.reg_b) / 1e6))
        q = np.linspace(a_, b_, 80)
        h_ = min(1.0, 0.12 + 0.88 * (b_ - a_) / 1.1)
        lw = ((1.9 if r.from_se else 0.9) if r.on_loop else (0.9 if r.from_se else 0.5)) * S
        ax2.plot(q, h_ * np.sin(np.pi * (q - a_) / (b_ - a_)), color=EC[r.ek], lw=lw, ls="-" if r.on_loop else (0, (2.5, 2)),
                 alpha=1 if r.on_loop else (0.8 if r.from_se else 0.5))
    label(y3 + 0.26, "Cicero\nedges", linespacing=0.95)

    # TF coverage tracks
    rh, y_t = 0.145, y3 - 0.03
    hub = sorted(set(np.r_[(E.pos.values - L0) // 2000, (np.arange(SE0, SE1, 2000) - L0) // 2000].tolist()))
    for k, t in enumerate(ORDER):
        y_ = y_t - (k + 1) * rh
        a_ = fig.add_axes([Lx, y_ / H, AW, (rh - 0.012) / H], sharex=first)
        d = Tc[Tc.tf == t]
        tot = int(d[d.b == "total"].n.iloc[0])
        h = np.zeros(nb)
        for r in d[d.b != "total"].itertuples():
            h[int(r.b)] = r.n
        h = h / tot * 1e6
        lim = max(h[hub].max(), np.percentile(h, 99.5))
        stripes(a_)
        a_.fill_between(ed[:-1], 0, np.minimum(h, lim), step="post", color=TFC[t], lw=0, zorder=2)
        a_.set_ylim(0, lim)
        a_.set_yticks([])
        a_.spines["left"].set_visible(False)
        a_.spines["bottom"].set_linewidth(0.4 * S)
        a_.tick_params(bottom=False, labelbottom=False)
        a_.text(-0.006, 0.4, t, transform=a_.transAxes, ha="right", va="center", fontstyle="italic", color=TFC[t] if t not in ("MBD3", "MLX") else "#444444")

    # ChromHMM
    y_c = y_t - len(ORDER) * rh - 0.17
    axc = fig.add_axes([Lx, y_c / H, AW, 0.15 / H], sharex=first)
    stripes(axc)
    axc.set_ylim(0, 1)
    axc.set_axis_off()
    for r in pd.read_csv(src / "se_chromhmm_window.tsv", sep="\t", header=None, names=["c", "s", "e", "st"]).itertuples():
        if GROUP.get(r.st):
            axc.add_patch(plt.Rectangle((r.s / 1e6, 0.08), (r.e - r.s) / 1e6, 0.84, color=GCOL[GROUP[r.st]], lw=0, zorder=2))
    axc.text(-0.006, 0.5, "ChromHMM", transform=axc.transAxes, ha="right", va="center", fontsize=F_SMALL)

    # RefSeq genes: promoter partners first, then longest; greedy rows; names only where they fit
    y_g, NR = 0.61, 3
    ax4 = fig.add_axes([Lx, y_g / H, AW, (y_c - y_g - 0.01) / H], sharex=first)
    stripes(ax4)
    ax4.set_ylim(NR, 0)
    ax4.set_xlim(*XL)
    ax4.set_yticks([])
    ax4.spines["left"].set_visible(False)
    ax4.set_xticks(np.arange(0.6, 2.3, 0.2))
    ax4.tick_params(labelsize=F_LAB)
    ax4.set_xlabel("chr19 position (Mb, hg38)", fontsize=F_LAB, labelpad=0)
    ax4.text(-0.006, 0.5, "RefSeq\ngenes", transform=ax4.transAxes, ha="right", va="center", linespacing=0.95, fontsize=F_SMALL)
    J = pd.DataFrame(json.load(open(src / "se_refseq_select.json"))["ncbiRefSeqSelect"])
    J = J[(J.txEnd > L0) & (J.txStart < XR)].copy()
    rnd = fig.canvas.get_renderer()
    mb_per_in = (XR - L0) / 1e6 / (AW0 * S)
    prom = {x for l in E.label for x in l.split(" ")[0].split("/")}

    def tw(t, **kw):
        x = fig.text(0, 0, t, **kw)
        w_ = x.get_window_extent(rnd).width / fig.dpi * mb_per_in
        x.remove()
        return w_

    J["prio"] = [0 if g in prom else 2 for g in J.name2]
    J["ln"] = J.txEnd - J.txStart
    J = J.sort_values(["prio", "ln"], ascending=[True, False], kind="stable")
    rows, used = {k: [] for k in range(NR)}, {k: [] for k in range(NR)}
    sc = (SE0 + SE1) / 2e6
    sew = tw("super-enhancer", fontweight="bold", fontsize=F_SMALL)
    rows[0].append((sc - sew / 2 - 0.01, sc + sew / 2 + 0.01))
    ax4.text(sc, 0.62, "super-enhancer", ha="center", va="center", color=TXT_SE, fontweight="bold", fontsize=F_SMALL, zorder=5,
             bbox=dict(fc="white", ec="none", pad=0.2, alpha=0.7))
    free = lambda k, a_, b_: all(b_ + 0.006 < u or a_ - 0.006 > v for u, v in rows[k])

    def model(r, k, col):
        y_ = k + 0.17
        a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, XR) / 1e6
        ax4.plot([a_, b_], [y_, y_], color=col, lw=0.7 * S, zorder=3, solid_capstyle="butt")
        for es, ee in zip(map(int, r.exonStarts.strip(",").split(",")), map(int, r.exonEnds.strip(",").split(","))):
            if ee > L0 and es < XR:
                ax4.add_patch(plt.Rectangle((es / 1e6, y_ - 0.13), max((ee - es) / 1e6, 0.001), 0.26, color=col, lw=0, zorder=4))
        tss = (r.txStart if r.strand == "+" else r.txEnd) / 1e6
        d_ = 0.009 if r.strand == "+" else -0.009
        if XL[0] <= tss <= XL[1]:
            ax4.plot([tss, tss], [y_, y_ - 0.2], color=col, lw=0.7 * S, zorder=4)
            ax4.annotate("", xy=(tss + d_, y_ - 0.2), xytext=(tss, y_ - 0.2),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7 * S, mutation_scale=5 * S, shrinkA=0, shrinkB=0), zorder=4)

    for r in J.itertuples():
        col = INK if r.prio == 0 else "#666666"
        a_, b_ = max(r.txStart, L0) / 1e6, min(r.txEnd, XR) / 1e6
        kw = dict(fontstyle="italic", fontweight="bold", fontsize=F_SMALL) if r.prio == 0 else dict(fontstyle="italic", fontsize=F_MINOR)
        w_ = tw(r.name2, **kw)
        c_ = (a_ + b_) / 2
        ta, tb = c_ - w_ / 2, c_ + w_ / 2
        ok = next((k for k in range(NR) if free(k, min(a_, ta), max(b_, tb))), None)
        if ok is not None and tb < XL[1] and ta > XL[0]:
            rows[ok].append((min(a_, ta), max(b_, tb)))
            model(r, ok, col)
            ax4.text(c_, ok + 0.62, r.name2, ha="center", va="center", color=col, zorder=5, **kw)
        else:
            k = min(range(NR), key=lambda q: sum(1 for u, v in used[q] if a_ < v and u < b_))
            used[k].append((a_, b_))
            model(r, k, col if r.prio < 2 else "#8A8A8A")

    # bottom key: ChromHMM colours and bands
    kx = 0.008
    for c_, l_, wd_ in [(GCOL["enhancer"], "Enhancer", 0.075), (GCOL["TSS/flank"], "TSS/flank", 0.08), (GCOL["transcribed"], "Transcribed", 0.088),
                        (GCOL["repressed"], "Repressed", 0.098), (BAND_SE, "Super-enhancer", 0.125), (PBAND["promoter"], "Promoter partner", 0.14),
                        (PBAND["enhancer"], "Enhancer partner", 0.14), (PBAND["other"], "Other partner", 0.12)]:
        fig.add_artist(plt.Rectangle((kx, (0.1 - 0.045) / H), 0.014 * W0 / W, 0.09 / H, transform=fig.transFigure, color=c_, lw=0))
        fig.text(kx + 0.019 * W0 / W, 0.1 / H, l_, va="center", fontsize=F_SMALL)
        kx += wd_ * W0 / W
    for a_ in fig.axes:                                  # tracks share x: fix the window on all of them (no autoscale to the data range)
        a_.set_autoscalex_on(False)
    first.set_xlim(*XL)
    style.save(fig, out, "g_se_browser", bbox_inches="tight", pad_inches=0.01)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
