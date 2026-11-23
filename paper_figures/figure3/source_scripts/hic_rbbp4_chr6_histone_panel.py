"""Candidate RBBP4 Hi-C locus panel (2026-11-23): v3 screen candidate #4 (hic_rbbp4_locus_screen_v3.py), RBBP4 p <= 0.05 Multi-region
Coordinated Cluster on the chr6 histone cluster: A = chr6:26188401-26188607 (H4C4 promoter), B = chr6:26304770-26305040 (TssA),
C = chr6:26537317-26537759 (HMGN4 promoter). Same layout as hic_rbbp4_locus (panels/hic_loci.py draw(), inputs built with
prepare_source_data.derive_hic_loci into a separate folder, not source_data/), plus a log2 O/E triangle on the right
(O/E = balanced contacts / mean of the same diagonal over chr6:21-31.5 Mb at 10 kb). Circles = per anchor, the mean of its best
centres within +-10 kb in the v3 scoring. Does not touch hic_rbbp4_locus.
  python hic_rbbp4_chr6_histone_panel.py <outdir>   (Longleaf, hic env)"""
import sys
from pathlib import Path
import numpy as np
X = Path("/work/users/d/y/dyy12/XuLab"); F3 = X / "paper_figures/figure3"
sys.path.insert(0, str(F3)); sys.path.insert(0, str(F3 / "panels")); sys.path.insert(0, str(X / "paper_figures"))
OUT = Path(sys.argv[1]); SRC = OUT / "src_chr6_histone"; SRC.mkdir(parents=True, exist_ok=True)
import prepare_source_data as psd   # noqa: E402
import hic_loci   # noqa: E402
from common import style   # noqa: E402
NAME = "rbbp4_chr6_histone"
psd.LOCI = {NAME: dict(tf="RBBP4", module="RBBP4 p <= 0.05 MCC (v3 screen #4)", regions=["chr6:26188401-26188607", "chr6:26304770-26305040", "chr6:26537317-26537759"],
                       circles=[26185000, 26310000, 26535000], genes=None, yticks=None, xticks=None, peaks="cicero", promoter_window=1000)}
psd.derive_hic_loci(SRC / "hic_loci.json")
import json   # noqa: E402
L = json.load(open(SRC / "hic_loci.json"))[NAME]

# O/E: distance expectation from a 10.5 Mb window around the locus at 10 kb
import h5py   # noqa: E402
f = h5py.File(X / psd.MCOOL, "r"); g = f["resolutions/10000"]; names = [n.decode() for n in g["chroms/name"][:]]; co = g["indexes/chrom_offset"][:]
c = L["chrom"]; ci = names.index(c); W0, W1 = 21_000_000, 31_500_000
i0, i1 = int(co[ci]) + W0 // 10000, int(co[ci]) + W1 // 10000; lo, hi = int(g["indexes/bin1_offset"][i0]), int(g["indexes/bin1_offset"][i1])
a, b, v = g["pixels/bin1_id"][lo:hi], g["pixels/bin2_id"][lo:hi], g["pixels/count"][lo:hi]; k = b < i1; n = i1 - i0
R = np.zeros((n, n)); R[a[k] - i0, b[k] - i0] = v[k]; R[b[k] - i0, a[k] - i0] = v[k]; w = g["bins/weight"][i0:i1]; M = R * np.outer(w, w)
bad = ~np.isfinite(w); M[bad, :] = np.nan; M[:, bad] = np.nan
exp = np.array([np.nanmean(np.diagonal(M, d)) for d in range(n)])
X0, X1 = L["window"]; s0 = (X0 - W0) // 10000; s1 = (X1 - W0) // 10000
ij = np.abs(np.subtract.outer(np.arange(s0, s1), np.arange(s0, s1)))
with np.errstate(all="ignore"): OE = np.log2(M[s0:s1, s0:s1] / exp[ij])

real_save = hic_loci.style.save
def save_with_oe(fig, out, name, **kw):
    import matplotlib.pyplot as plt
    ax = fig.axes[0]; p = ax.get_position(); Wf = fig.get_size_inches()[0]
    # A and B are 116 kb apart: start the B element label just right of the A label
    fig.canvas.draw(); rr = fig.canvas.get_renderer()
    lab = [t for a_ in fig.axes for t in a_.texts if t.get_text().endswith("TFs") and "," in t.get_text()]
    lab = sorted(lab, key=lambda t: t.get_position()[0])
    if len(lab) == 3:
        ta, tb = lab[0], lab[1]; ex = ta.get_window_extent(rr); axb = tb.axes
        x_new = axb.transData.inverted().transform((ex.x1 + 0.12 * fig.dpi, 0))[0]
        if tb.get_window_extent(rr).x0 < ex.x1 + 0.05 * fig.dpi: tb.set_x(x_new); tb.set_ha("left")
    ax_oe = fig.add_axes([p.x0 + 7.75 / Wf, p.y0, p.width, p.height])
    n_ = OE.shape[0]; xs = X0 + np.arange(n_ + 1) * 10000; XX, YY = np.meshgrid(xs, xs, indexing="ij")
    Zm = np.ma.masked_invalid(np.ma.masked_where(np.tril(np.ones_like(OE), -1) > 0, OE))
    pc = ax_oe.pcolormesh((XX + YY) / 2e6, (YY - XX) / 2e3, Zm, cmap="RdBu_r", vmin=-1.5, vmax=1.5, shading="flat", rasterized=True, lw=0, edgecolors="none", antialiased=False)
    cA, cB, cC = L["circles"]
    for nm, i, j in (("A–B", cA, cB), ("B–C", cB, cC), ("A–C", cA, cC)):
        ax_oe.plot((i + j) / 2e6, (j - i) / 2e3, "o", mfc="none", mec="#222222", ms=25, mew=2, zorder=5)
        ax_oe.text((i + j) / 2e6, (j - i) / 2e3 + 0.075 * ax.get_ylim()[1], nm, ha="center", va="bottom", fontsize=14, fontweight="bold", color="#222222", zorder=6)
    ax_oe.set_xlim(ax.get_xlim()); ax_oe.set_ylim(ax.get_ylim()); ax_oe.set_yticks(ax.get_yticks()); ax_oe.set_ylim(ax.get_ylim())
    ax_oe.tick_params(bottom=True, labelbottom=True, labelsize=12); ax_oe.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_oe.set_xlabel(f"{c} (Mb)", fontsize=14, labelpad=2); ax_oe.set_ylabel("Distance / 2 (kb)", fontsize=14)
    ax_oe.set_title("Observed / expected (log2)", loc="left", fontsize=14, fontweight="bold", pad=6)
    cb = fig.add_axes([p.x0 + 7.75 / Wf + p.width - 1.45 / Wf, p.y1 - 0.5 / fig.get_size_inches()[1], 1.4 / Wf, 0.14 / fig.get_size_inches()[1]])
    plt.colorbar(pc, cax=cb, orientation="horizontal"); cb.tick_params(labelsize=12); cb.xaxis.set_major_locator(plt.MaxNLocator(3))
    return real_save(fig, OUT, "hic_rbbp4_locus_chr6_histone", **kw)
hic_loci.style.save = save_with_oe
hic_loci.W = 8.30 + 7.75   # same physical layout on the left, O/E triangle to the right
style.use(base_pt=12, small_pt=12)
import matplotlib.pyplot as plt   # noqa: E402
plt.rcParams.update({"axes.linewidth": 1, "xtick.major.width": 1, "ytick.major.width": 1, "xtick.major.size": 4, "ytick.major.size": 4})
hic_loci.draw(SRC, OUT, NAME, L)
print("written:", sorted(p.name for p in OUT.glob("hic_rbbp4_locus_chr6_histone.*")))
